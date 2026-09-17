#!/usr/bin/env python3
"""Durable, bounded host continuation for this already-authorized Arena suite.

Run with /root/.venvs/alpacaeval/bin/python after reviewing this file. A single
host process waits for GPU generation, reuses a fixed paid pilot, records its
cost decision, resumes full judging, and requires the independent verifier.
Only the predeclared same-request structural-invalid policy may retry a paid
game, at most five total attempts. Ambiguous/inflight requests never retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SUITE = Path(__file__).resolve().parent
CPU_PYTHON = "/root/miniconda3/envs/sml/bin/python"
JUDGE_PYTHON = "/root/.venvs/alpacaeval/bin/python"
TAGS = ("base", "sft-init", "grpo", "lam2", "lam4", "lam8")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def judge_module(suite):
    path = Path(suite) / "source/scripts/judge_arena_hard.py"
    spec = importlib.util.spec_from_file_location("frozen_arena_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retry_module(suite):
    path = Path(suite) / "retry_invalid_judgments.py"
    spec = importlib.util.spec_from_file_location("declared_arena_retry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(suite, path, value):
    judge_module(suite).atomic_json(Path(path), value)


def load_judge_run(suite):
    suite = Path(suite)
    judge = judge_module(suite)
    return judge.JudgeRun(
        suite / "model_judgment/gpt-4.1", judge.load_jsonl(suite / "question.jsonl"),
        judge.load_jsonl(suite / "model_answer/o3-mini-2025-01-31.jsonl"),
        {tag: judge.load_jsonl(suite / "model_answer" / f"{tag}.jsonl") for tag in TAGS},
        judge.load_protocol(suite / "source/third_party/arena_hard"))


def pilot_uids(suite):
    uids = read_json(Path(suite) / "pilot_uids.json")
    if not isinstance(uids, list) or len(uids) != 5 or not all(isinstance(uid, str) for uid in uids) or len(set(uids)) != 5:
        raise ValueError("Pilot requires five fixed unique UIDs")
    return uids


def wait_for_generation(suite, poll_seconds=15, *, sleep=time.sleep):
    suite = Path(suite)
    while True:
        status = read_json(suite / "status.json") if (suite / "status.json").exists() else {}
        if status.get("state") == "generation_failed":
            raise RuntimeError("GPU status is generation_failed; no judging will start")
        if (suite / "generation_complete.json").exists():
            completion = read_json(suite / "generation_complete.json")
            if completion.get("state") != "generation_complete":
                raise RuntimeError("Invalid generation completion marker")
            return
        sleep(poll_seconds)


def wait_for_verifier(suite, *, sleep=time.sleep, monotonic=time.monotonic, timeout=120):
    path = Path(suite) / "verify_final_scoring.py"
    deadline = monotonic() + timeout
    while not path.is_file():
        if monotonic() >= deadline:
            raise RuntimeError("Independent final verifier is missing; continuation stopped before paid dispatch")
        sleep(5)


def has_nonpilot_requests(suite):
    expected = set(pilot_uids(suite))
    directory = Path(suite) / "model_judgment/gpt-4.1/state/games"
    for path in directory.glob("*/*.json"):
        record = read_json(path)
        if record.get("tag") not in TAGS or record.get("uid") not in expected:
            return True
    return False


def pilot_status(suite, *, strict_scope=True):
    suite = Path(suite)
    run = load_judge_run(suite)
    judge = judge_module(suite)
    uids = pilot_uids(suite)
    if not set(uids) <= set(run.questions):
        raise ValueError("Pilot UID not present in canonical questions")
    state = run.directory / "state"
    if (state / "protocol.json").exists() and read_json(state / "protocol.json") != run.identity:
        raise RuntimeError("Pilot protocol/input identity mismatch")
    for tag, answers in run.answers.items():
        model_path = state / "models" / f"{tag}.json"
        if model_path.exists() and read_json(model_path) != {
            "answers_sha256": judge.digest(list(answers.values())), "tag": tag, "model": tag}:
            raise RuntimeError(f"Pilot model identity mismatch: {tag}")
    valid, missing, pairs, records = 0, [], {}, []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for tag in TAGS:
        output = run.directory / f"{tag}.jsonl"
        rows = judge.load_jsonl(output) if output.exists() else []
        indexed = {row["uid"]: row for row in rows}
        if len(indexed) != len(rows) or (strict_scope and not set(indexed) <= set(uids)):
            raise RuntimeError("Unexpected or duplicate pilot output rows")
        pairs[tag] = 0
        for uid in uids:
            games = []
            for order in (0, 1):
                record = run.load_record(tag, uid, order)
                if record is None:
                    missing.append([tag, uid, order])
                    continue
                if record["status"] != "valid":
                    raise RuntimeError(f"Pilot has unresolved {record['status']} game {tag}:{uid}:{order}; no automatic retry")
                valid += 1
                records.append({"tag": tag, "uid": uid, "order": order,
                                "sha256": file_hash(run.game_path(tag, uid, order))})
                for name in usage:
                    usage[name] += record["usage"][name]
                games.append({"score": record["score"], "judgment": {"answer": record["answer"]},
                              "prompt": record["request"]["messages"]})
            if uid in indexed:
                expected = {"uid": uid, "category": "hard_prompt", "judge": "gpt-4.1", "model": tag,
                            "baseline": "o3-mini-2025-01-31", "games": games}
                if len(games) != 2 or indexed[uid] != expected:
                    raise RuntimeError(f"Pilot official output does not match saved games: {tag}:{uid}")
                pairs[tag] += 1
    complete = valid == 60 and all(count == 5 for count in pairs.values())
    if complete:
        if not (state / "protocol.json").exists() or any(not (state / "models" / f"{tag}.json").exists() for tag in TAGS):
            raise RuntimeError("Completed pilot lacks persisted identity")
    return {"complete": complete, "valid_games": valid, "complete_pairs": pairs, "missing_games": missing,
            "uids": uids, "usage": usage, "records_sha256": object_hash(records), "records": records}


def calculate_cost_decision(pilot_cost):
    if not isinstance(pilot_cost, (int, float)) or not math.isfinite(pilot_cost) or pilot_cost <= 0:
        raise ValueError("Pilot cost must be positive and finite")
    # Preserve observed cost exactly; never round an excess below a review gate.
    cost = Decimal(str(pilot_cost))
    estimate = cost * 100
    budget = int(max(Decimal(30), estimate * 3).to_integral_value(rounding=ROUND_CEILING))
    return {"pilot_cost_cny": float(cost), "pilot_games": 60, "full_games": 6000,
            "estimated_full_cny": float(estimate), "dispatch_budget_cny": budget,
            "automatic_continuation": estimate <= 300 and budget <= 900,
            "formula": "estimate=pilot_cost*100; budget=ceil(max(30,3*estimate))",
            "budget_scope": "Cumulative same-directory relay usage, including pilot; dispatch guard, not a hard cap"}


def read_only_usage(suite, start_date):
    judge = judge_module(suite)
    key = os.environ.get("LINKAPI_KEY") or Path("/root/.linkapi_key").read_text().strip()
    relay = judge.Relay(key, timeout=15)
    try:
        return relay.usage(start_date)
    finally:
        relay.close()


def wait_for_positive_cost(suite, *, usage_reader=None, timeout=120, sleep=time.sleep, monotonic=time.monotonic):
    suite = Path(suite)
    path = suite / "model_judgment/gpt-4.1/state/billing.json"
    billing = read_json(path)
    baseline, current = billing["usage0_cny"], billing["latest_usage_cny"]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (baseline, current)):
        raise RuntimeError("Invalid persisted pilot billing")
    if current < baseline or not math.isclose(billing["spent_cny"], current - baseline, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("Inconsistent persisted pilot billing")
    if billing["spent_cny"] > 0:
        return billing["spent_cny"]
    deadline = monotonic() + timeout
    while True:
        usage = (usage_reader or (lambda start: read_only_usage(suite, start)))(billing["start_date"])
        if not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < current - 1e-6:
            raise RuntimeError("Invalid/decreasing read-only pilot usage")
        billing.update(latest_usage_cny=usage, spent_cny=max(0.0, usage - baseline),
                       checked_at=now(), continuation_usage_recheck=True)
        save(suite, path, billing)
        if billing["spent_cny"] > 0:
            return billing["spent_cny"]
        if monotonic() >= deadline:
            raise RuntimeError("Pilot cost remains zero after billing settlement wait; paid calls will not repeat")
        sleep(10)


def commands(suite):
    suite = Path(suite)
    return {
        "validate_generation": [CPU_PYTHON, str(suite / "run_evaluation.py"), "--validate-only"],
        "pilot": [JUDGE_PYTHON, str(suite / "source/scripts/judge_arena_hard.py"),
                  "--questions", str(suite / "question.jsonl"),
                  "--baseline", str(suite / "model_answer/o3-mini-2025-01-31.jsonl"),
                  "--answers-dir", str(suite / "model_answer"),
                  "--output-dir", str(suite / "model_judgment/gpt-4.1"),
                  "--tags", *TAGS, "--workers", "12", "--budget-cny", "20",
                  "--uids-file", str(suite / "pilot_uids.json")],
        "verify_final": [CPU_PYTHON, str(suite / "verify_final_scoring.py")],
    }


def run_command(suite, label, command):
    suite = Path(suite)
    directory = suite / "job/continuation"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (directory / f"{label}-{attempt}.command.json").exists():
        attempt += 1
    prefix = directory / f"{label}-{attempt}"
    record = {"command": command, "started_at": now(), "status": "starting"}
    save(suite, prefix.with_suffix(".command.json"), record)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       TIKTOKEN_CACHE_DIR=str(suite / ".tiktoken-cache"))
    with prefix.with_suffix(".log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=suite)
        record.update(pid=process.pid, status="running")
        save(suite, prefix.with_suffix(".command.json"), record)
        returncode = process.wait()
    judge_module(suite).atomic_text(prefix.with_suffix(".exit_code"), str(returncode) + "\n")
    record.update(status="complete" if returncode == 0 else "failed", returncode=returncode, finished_at=now())
    save(suite, prefix.with_suffix(".command.json"), record)
    return {"returncode": returncode, "stdout": prefix.with_suffix(".log").read_text(), "log": str(prefix.with_suffix(".log"))}


def bind_execution(suite, planned):
    runners = ["run_evaluation.py", "run_full_judging.sh", "verify_final_scoring.py",
               "source/scripts/judge_arena_hard.py", "source/scripts/score_arena_hard.py",
               "source/third_party/arena_hard/config/arena-hard-v2.0.yaml",
               "source/third_party/arena_hard/utils/judge_utils.py", "source/third_party/arena_hard/gen_judgment.py",
               "retry_invalid_judgments.py"]
    manifest = read_json(suite / "experiment.json")
    inputs = ["question.jsonl", "pilot_uids.json", "model_answer/o3-mini-2025-01-31.jsonl", "retry_policy.json"]
    inputs += [f"model_answer/{tag}.jsonl" for tag in TAGS]
    identity = {"runners_sha256": {name: file_hash(suite / name) for name in runners},
                "continuation_sha256": file_hash(__file__), "experiment_sha256": file_hash(suite / "experiment.json"),
                "inputs_sha256": {name: file_hash(suite / name) for name in inputs},
                "models": manifest["models"]}
    path = suite / "host_execution.json"
    document = read_json(path) if path.exists() else {}
    for key, value in retry_module(suite).binding(suite).items():
        if key in document and document[key] != value:
            raise RuntimeError("Host retry policy/helper binding changed")
        document[key] = value
    if "continuation" in document and document["continuation"]["identity"] != identity:
        raise RuntimeError("Host execution identity changed; review required before resuming")
    document.setdefault("continuation", {"identity": identity, "created_at": now(), "commands": planned})
    save(suite, path, document)
    return object_hash(identity)


def checked(runner, suite, label, command):
    output = runner(suite, label, command)
    if output["returncode"] != 0:
        raise RuntimeError(f"{label} failed with exit code {output['returncode']}; inspect saved log")
    return output


def run_with_structural_retries(suite, runner, label, command, budget, uids=None):
    helper = retry_module(suite)
    # Every pass either runs the fixed driver or increments at least one counter.
    # The per-game counter and this global bound prevent unbounded retry loops.
    for _ in range(6000 * helper.MAX_ATTEMPTS + 1):
        targets = helper.invalid_targets(suite, uids)
        if targets:
            run = helper.load_run(suite)
            previous = {item: read_json(run.game_path(*item)).get("attempt", 0) for item in targets}
            retry_command = [JUDGE_PYTHON, str(suite / "retry_invalid_judgments.py"), "--budget-cny", str(budget)]
            for tag, uid, order in targets:
                retry_command += ["--game", f"{tag}:{uid}:{order}"]
            host = read_json(suite / "host_execution.json")
            host["continuation"]["commands"][label + "_retry"] = retry_command
            save(suite, suite / "host_execution.json", host)
            output = runner(suite, label + "_retry", retry_command)
            if output["returncode"] not in (0, 2):
                raise RuntimeError(f"{label}_retry blocked with exit code {output['returncode']}; no automatic transport retry")
            if any(read_json(run.game_path(*item)).get("attempt") != attempt + 1 for item, attempt in previous.items()):
                raise RuntimeError("Retry helper did not persist exactly one new attempt per explicit target")
            continue
        output = runner(suite, label, command)
        if output["returncode"] == 0:
            return output
        if not helper.invalid_targets(suite, uids):
            raise RuntimeError(f"{label} failed with exit code {output['returncode']}; no eligible structural invalid games")
    raise RuntimeError("Uniform retry bound exhausted; evaluation remains incomplete")


def continue_suite(suite=SUITE, *, poll_seconds=15, runner=run_command, usage_reader=None,
                   sleep=time.sleep, monotonic=time.monotonic):
    suite = Path(suite).resolve()
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll-seconds must be finite and positive")
    judge = judge_module(suite)
    # Separate host lock: the frozen judge owns its own lock while requests run.
    with judge.exclusive_lock(suite / "job/continuation-lock"):
        state = {"state": "running", "phase": "waiting_generation", "pid": os.getpid(), "started_at": now()}
        def phase(name):
            state.update(phase=name, updated_at=now())
            save(suite, suite / "continuation_state.json", state)
        try:
            phase("waiting_generation")
            wait_for_generation(suite, poll_seconds, sleep=sleep)
            wait_for_verifier(suite, sleep=sleep, monotonic=monotonic)
            planned = commands(suite)
            phase("validate_generation")
            checked(runner, suite, "validate_generation", planned["validate_generation"])
            identity = bind_execution(suite, planned)
            decision_path = suite / "cost_decision.json"
            if decision_path.exists():
                bound_hash = read_json(suite / "host_execution.json")["continuation"].get("cost_decision_sha256")
                if bound_hash is not None and bound_hash != file_hash(decision_path):
                    raise RuntimeError("Saved cost decision hash changed; review required")
                decision = read_json(decision_path)
                phase("reuse_cost_decision")
                pilot = pilot_status(suite, strict_scope=False)
                if not pilot["complete"] or decision["host_identity_sha256"] != identity or decision["pilot_records_sha256"] != pilot["records_sha256"]:
                    raise RuntimeError("Saved pilot/cost decision identity mismatch")
                expected = calculate_cost_decision(decision["pilot_cost_cny"])
                if any(decision[key] != value for key, value in expected.items()):
                    raise RuntimeError("Saved cost decision does not match approved formula")
                if any(decision.get(key) != value for key, value in retry_module(suite).binding(suite).items()):
                    raise RuntimeError("Saved cost decision retry policy binding mismatch")
            else:
                if has_nonpilot_requests(suite):
                    raise RuntimeError("Full judging already started without a persisted pilot cost decision; review required")
                phase("pilot")
                if retry_module(suite).invalid_targets(suite, pilot_uids(suite)):
                    run_with_structural_retries(suite, runner, "pilot", planned["pilot"], 20, pilot_uids(suite))
                pilot = pilot_status(suite)
                if not pilot["complete"]:
                    run_with_structural_retries(suite, runner, "pilot", planned["pilot"], 20, pilot_uids(suite))
                    pilot = pilot_status(suite)
                if not pilot["complete"]:
                    raise RuntimeError("Pilot did not produce exactly 60 valid games and five complete pairs per model")
                save(suite, suite / "pilot_summary.json", dict(pilot, verified_at=now(), host_identity_sha256=identity))
                phase("pilot_cost")
                cost = wait_for_positive_cost(suite, usage_reader=usage_reader, sleep=sleep, monotonic=monotonic)
                decision = dict(calculate_cost_decision(cost), host_identity_sha256=identity,
                                pilot_records_sha256=pilot["records_sha256"], decided_at=now())
                decision.update(retry_module(suite).binding(suite))
                save(suite, decision_path, decision)
            if not decision["automatic_continuation"]:
                raise RuntimeError("Unexpected pilot cost: estimate exceeds 300 CNY or dispatch budget exceeds 900 CNY; review required")
            full_command = ["bash", str(suite / "run_full_judging.sh"), str(decision["dispatch_budget_cny"])]
            host = read_json(suite / "host_execution.json")
            host["continuation"]["commands"]["full_judging"] = full_command
            host["continuation"]["cost_decision_sha256"] = file_hash(decision_path)
            save(suite, suite / "host_execution.json", host)
            if not (suite / "scores").exists():
                phase("full_judging")
                run_with_structural_retries(suite, runner, "full_judging", full_command, decision["dispatch_budget_cny"])
            phase("verify_final")
            output = checked(runner, suite, "verify_final", planned["verify_final"])
            verification = json.loads(output["stdout"])
            expected = {"status": "passed", "current_games": 6000, "exact_request_and_parse_checks": 6000,
                        "judgments": 3000, "questions": 500, "numeric_comparisons": 42}
            if any(verification.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Independent final verifier did not confirm strict complete coverage")
            difference = verification.get("max_absolute_difference")
            if not isinstance(difference, (int, float)) or not math.isfinite(difference) or not 0 <= difference <= 1e-10:
                raise RuntimeError("Independent final numerical verification failed")
            state.update(state="evaluation_complete", phase="complete", completed_at=now(),
                         verification=verification, cost_decision_sha256=file_hash(decision_path),
                         host_execution_sha256=file_hash(suite / "host_execution.json"))
            save(suite, suite / "evaluation_complete.json", state)
            save(suite, suite / "continuation_state.json", state)
            return state
        except BaseException as error:
            state.update(state="failed", error_type=type(error).__name__, error=str(error), failed_at=now())
            save(suite, suite / "continuation_state.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    result = continue_suite(poll_seconds=args.poll_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
