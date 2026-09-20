#!/usr/bin/env python3
"""Resume the authorized GPT-4o suite without retrying malformed judge outputs.

Only completed outputs accepted by the explicit exclusion policy are terminal
failures. Transport ambiguity, unfinished requests, and corrupt state stop all
further dispatch. Existing game records and historical attempts are never
rewritten. The budget is the frozen cumulative billing dispatch guard; billing
lag and one bounded batch can overshoot it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arena_exclusion_policy import POLICY_ID, classify_record, validate_policy

TAGS = ("base", "sft-init", "grpo", "lam2", "lam4", "lam8")
STATES = ("missing", "valid", "judge_failed", "blocked")


def load_module(path):
    """Import a frozen file by path without modifying its source or bytecode."""
    path = Path(path).resolve()
    name = "_arena_exclusions_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def load_suite(suite):
    continuation = load_module(suite / "continue_evaluation.py")
    # The frozen continuation's loader uses importlib; prevent new __pycache__
    # writes in the frozen source tree while retaining its exact JudgeRun API.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        run = continuation.load_judge_run(suite)
        judge = continuation.judge_module(suite)
    finally:
        sys.dont_write_bytecode = previous
    if (set(run.answers) != set(TAGS) or len(run.questions) != 500
            or run.judge_model != "gpt-4o" or run.baseline_model != "gpt-4o-mini-2024-07-18"):
        raise ValueError("Exclusions runner requires the complete authorized GPT-4o suite")
    return run, judge


def default_relay_factory(suite):
    transport = load_module(suite / "resilient_judge.py")
    key = os.environ.get("LINKAPI_KEY") or Path("/root/.linkapi_key").read_text().strip()
    return transport.make_relay(key, suite=suite)


def validate_saved_billing(run):
    """Reject corrupt saved accounting before the frozen guard updates it."""
    path = run.directory / "state/billing.json"
    if not path.exists():
        return
    try:
        billing = json.loads(path.read_text())
        baseline, latest, spent = (billing[name] for name in
                                   ("usage0_cny", "latest_usage_cny", "spent_cny"))
        valid = (isinstance(billing.get("start_date"), str) and bool(billing["start_date"])
                 and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                         for value in (baseline, latest, spent))
                 and latest >= baseline
                 and math.isclose(spent, latest - baseline, rel_tol=0, abs_tol=1e-6))
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("Saved billing metadata is corrupt; dispatch blocked")


def run_suite(suite, *, policy, budget_cny, workers=32, max_requests=0, relay_factory=None):
    """Run each missing game once, preserving the frozen request/state protocol.

    ``relay_factory`` replaces only external HTTP/billing for offline tests.
    A bounded invocation returns ``status='partial'`` and never writes the
    completion marker until all 6,000 games are valid or judge-output failures.
    """
    policy_value = validate_policy(json.loads(Path(policy).read_text()))
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("max-requests must be a nonnegative integer")
    if (type(budget_cny) not in (int, float) or not math.isfinite(budget_cny)
            or budget_cny <= 0):
        raise ValueError("budget-cny must be finite and positive")
    suite = Path(suite).resolve()
    run, judge = load_suite(suite)
    factory = relay_factory if relay_factory is not None else lambda: default_relay_factory(suite)
    with judge.exclusive_lock(run.directory):
        run.prepare()
        states, pending = {}, []
        for uid in run.questions:
            for tag in TAGS:
                for order in (0, 1):
                    item = (tag, uid, order)
                    try:
                        state = classify_record(run.load_record(*item), judge)
                    except (ValueError, TypeError, KeyError, OSError):
                        state = "blocked"
                    states[item] = state
                    if state == "missing":
                        pending.append(item)
        dispatched, relay, complete_pairs = 0, None, {}

        def progress(status):
            by_model = {tag: {state: 0 for state in STATES} for tag in TAGS}
            for (tag, _, _), state in states.items():
                by_model[tag][state] += 1
            total = {state: sum(counts[state] for counts in by_model.values()) for state in STATES}
            complete = (status in ("complete", "complete_with_judge_exclusions")
                        and len(states) == 6000 and total["missing"] == total["blocked"] == 0
                        and total["valid"] + total["judge_failed"] == 6000)
            return {"policy_id": POLICY_ID, "policy_sha256": judge.digest(policy_value),
                    "judge": run.judge_model, "baseline": run.baseline_model,
                    "status": status, "complete": complete, "expected_games": 6000,
                    "dispatched_this_run": dispatched, "workers": workers,
                    "budget_cny": budget_cny, "max_requests": max_requests,
                    "counts": {"total": total, "by_model": by_model},
                    "complete_pairs": complete_pairs, "updated_at": judge.now()}

        def save_progress(status):
            document = progress(status)
            judge.atomic_json(suite / "exclusions_progress.json", document)
            return document

        try:
            if "blocked" in states.values():
                raise RuntimeError("Judge run blocked by unresolved or corrupt saved requests")
            validate_saved_billing(run)
            complete_pairs = run.export()
            save_progress("running")
            if pending:
                relay = factory()
            while pending and (not max_requests or dispatched < max_requests):
                run.billing_check(relay, budget_cny)
                batch_size = min(workers, len(pending),
                                 max_requests - dispatched if max_requests else workers)
                batch, pending = pending[:batch_size], pending[batch_size:]
                records = {item: run.save_inflight(*item) for item in batch}
                for item in batch:
                    states[item] = "blocked"  # Durable inflight records are never retryable.
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(relay.judge_call, record["request"]): item
                               for item, record in records.items()}
                    dispatched += len(batch)
                    for future in as_completed(futures):
                        item = futures[future]
                        record = records[item]
                        try:
                            response = future.result()
                            record.update({key: response.get(key) for key in
                                           ("answer", "usage", "finish_reason", "response_id",
                                            "response_model", "provider_request_id")})
                            score = judge.parse_score(record["answer"], run.protocol["regex_patterns"])
                            valid = (score is not None and record["finish_reason"] == "stop"
                                     and judge.valid_usage(record["usage"]))
                            record.update(status="valid" if valid else "invalid", score=score)
                        except Exception as error:
                            # Exception messages can contain credentials or transport headers.
                            record.update(status="ambiguous", error_type=type(error).__name__)
                        record["finished_at"] = judge.now()
                        judge.atomic_json(run.game_path(*item), record)
                        run._records[item] = record
                        run._dirty_tags.add(item[0])
                        states[item] = classify_record(record, judge)
                complete_pairs = run.export()
                save_progress("running")
                if "blocked" in states.values():
                    raise RuntimeError("Judge run blocked by an ambiguous or corrupt response; progress saved")
            complete_pairs = run.export()
            if relay is not None:
                run.billing_check(relay, budget_cny)
            status = ("partial" if pending else "complete_with_judge_exclusions"
                      if "judge_failed" in states.values() else "complete")
            result = save_progress(status)
            if result["complete"]:
                judge.atomic_json(suite / "exclusions_judging_complete.json", result)
            return result
        except Exception as error:
            document = progress("blocked")
            document["error_type"] = type(error).__name__
            judge.atomic_json(suite / "exclusions_progress.json", document)
            raise
        finally:
            if relay is not None:
                relay.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--budget-cny", type=float, required=True)
    parser.add_argument("--max-requests", type=int, default=0,
                        help="Maximum new paid requests in this invocation; 0 means all missing games")
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_suite(args.suite, policy=args.policy, workers=args.workers,
                       budget_cny=args.budget_cny, max_requests=args.max_requests)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Exclusions judge stopped: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
