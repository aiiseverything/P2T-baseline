#!/usr/bin/env python3
"""One explicit same-request retry per listed structurally invalid Arena game.

Uses the predeclared retry_policy.json: at most five total attempts, first valid
result stops, no winner-based selection, no ambiguous/inflight retries. Frozen
judge/protocol and generation caches remain unchanged. Exit 2 means completed
retry responses are still invalid; exit 1 means a blocking error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

SUITE = Path(__file__).resolve().parent
TAGS = ("base", "sft-init", "grpo", "lam2", "lam4", "lam8")
MAX_ATTEMPTS = 5
FROZEN_JUDGE_SHA256 = "aa0bbddb99d408e3d81156f63b8299381ed592a5f1abe2e252380ccb298d501f"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def judge_module(suite):
    path = Path(suite) / "source/scripts/judge_arena_hard.py"
    if file_hash(path) != FROZEN_JUDGE_SHA256:
        raise RuntimeError("Frozen judge source changed")
    spec = importlib.util.spec_from_file_location("unchanged_arena_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(suite):
    policy = read_json(Path(suite) / "retry_policy.json")
    if (policy.get("policy") != "arena_identical_request_invalid_output_retry_v1"
            or policy.get("max_total_attempts_per_game") != MAX_ATTEMPTS
            or policy.get("frozen_judge_sha256") != FROZEN_JUDGE_SHA256):
        raise RuntimeError("Unexpected predeclared retry policy")
    declared = datetime.fromisoformat(policy["declared_at"].replace("Z", "+00:00"))
    if declared > datetime.now(timezone.utc):
        raise RuntimeError("Retry policy declaration is in the future")
    return {"retry_policy": policy, "retry_policy_sha256": file_hash(Path(suite) / "retry_policy.json"),
            "retry_helper_sha256": file_hash(__file__)}


def load_run(suite):
    suite = Path(suite)
    judge = judge_module(suite)
    return judge.JudgeRun(suite / "model_judgment/gpt-4.1", judge.load_jsonl(suite / "question.jsonl"),
                         judge.load_jsonl(suite / "model_answer/gpt-4o-mini-2024-07-18.jsonl"),
                         {tag: judge.load_jsonl(suite / "model_answer" / f"{tag}.jsonl") for tag in TAGS},
                         judge.load_protocol(suite / "source/third_party/arena_hard", baseline_model="gpt-4o-mini-2024-07-18"))


def request_bytes(request):
    return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def eligible(record, judge):
    return (record.get("status") == "invalid" and isinstance(record.get("finished_at"), str)
            and isinstance(record.get("answer"), str) and record.get("finish_reason") in ("stop", "length")
            and record.get("score") == judge.parse_score(record["answer"])
            and (record["finish_reason"] == "length" or judge.parse_score(record["answer"]) is None))


def validate_chain(run, item, head, judge):
    expected = request_bytes(run.request(*item))
    current = head
    seen = set()
    while True:
        if (current.get("baseline_model") != run.baseline_model
                or current.get("protocol_sha256") != judge.digest(run.protocol)):
            raise RuntimeError("Custom baseline identity mismatch in retry chain")
        if not eligible(current, judge):
            raise RuntimeError("Retry chain contains an attempt that is not completed structural invalid")
        attempt = current.get("attempt", 0)
        if type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
            raise RuntimeError("Invalid attempt counter in retry chain")
        if (request_bytes(current.get("request")) != expected or current.get("request_sha256") != run.protocol_digest_request
                or (current.get("tag"), current.get("uid"), current.get("order")) != item):
            raise RuntimeError("Request identity mismatch in retry chain")
        local_id = current.get("local_request_id")
        if not isinstance(local_id, str) or local_id in seen or Path(local_id).name != local_id:
            raise RuntimeError("Invalid request ID in retry chain")
        seen.add(local_id)
        if attempt == 0:
            if current.get("supersedes_local_request_id") is not None:
                raise RuntimeError("Unexpected predecessor on first attempt")
            return
        previous_id = current.get("supersedes_local_request_id")
        if not isinstance(previous_id, str) or Path(previous_id).name != previous_id:
            raise RuntimeError("Missing predecessor in retry chain")
        archive = run.directory / "state/attempts" / item[0] / f"{item[1]}-{item[2]}" / f"{previous_id}.json"
        if not archive.exists():
            raise RuntimeError("Missing archive in retry chain")
        previous = read_json(archive)
        if previous.get("attempt", 0) != attempt - 1 or previous.get("local_request_id") != previous_id:
            raise RuntimeError("Discontinuous retry chain")
        current = previous


def invalid_targets(suite, uids=None):
    run, judge = load_run(suite), judge_module(suite)
    selected = set(run.questions) if uids is None else set(uids)
    if not selected <= set(run.questions):
        raise ValueError("Unknown retry UID")
    targets = []
    for tag in TAGS:
        for uid in selected:
            for order in (0, 1):
                record = run.load_record(tag, uid, order)
                if record is None or record["status"] == "valid":
                    continue
                if record["status"] in ("inflight", "ambiguous"):
                    raise RuntimeError(f"Unresolved {record['status']} game; automatic retry forbidden: {tag}:{uid}:{order}")
                if not eligible(record, judge):
                    raise RuntimeError(f"Invalid game is not eligible for structural retry: {tag}:{uid}:{order}")
                if type(record.get("attempt", 0)) is not int or record.get("attempt", 0) >= MAX_ATTEMPTS - 1:
                    raise RuntimeError(f"Five-attempt policy exhausted; evaluation remains incomplete: {tag}:{uid}:{order}")
                targets.append((tag, uid, order))
    return sorted(targets)


def retry_games(suite, games, *, budget_cny, relay_factory=None):
    suite = Path(suite).resolve()
    judge, run = judge_module(suite), load_run(suite)
    policy = binding(suite)
    if not math.isfinite(budget_cny) or budget_cny <= 0:
        raise ValueError("budget-cny must be finite and positive")
    if not games or len(games) != len(set(games)):
        raise ValueError("Explicit game list must be nonempty and unique")
    with judge.exclusive_lock(run.directory):
        run.prepare()
        originals = {}
        for item in games:
            if len(item) != 3 or item[0] not in TAGS or item[1] not in run.questions or item[2] not in (0, 1):
                raise ValueError("Invalid explicit game")
            record = run.load_record(*item)
            if record is None or not eligible(record, judge):
                raise RuntimeError("Only completed structural invalid judgments may retry; valid/ambiguous/inflight are forbidden")
            if record.get("attempt", 0) >= MAX_ATTEMPTS - 1:
                raise RuntimeError("Five-attempt policy exhausted; no question is dropped")
            run.protocol_digest_request = judge.digest(run.request(*item))
            validate_chain(run, item, record, judge)
            originals[item] = record
        host_path = suite / "host_execution.json"
        host = read_json(host_path) if host_path.exists() else {}
        for key, value in policy.items():
            if key in host and host[key] != value:
                raise RuntimeError("Host retry policy/helper binding changed")
            host[key] = value
        judge.atomic_json(host_path, host)
        invocation_id = str(uuid.uuid4())
        ledger_path = suite / "job/retry_invocations" / f"{invocation_id}.json"
        ledger = dict(policy, games=[list(item) for item in games], budget_cny=budget_cny,
                      started_at=now(), status="running", attempts=[])
        judge.atomic_json(ledger_path, ledger)
        relay = None
        try:
            if relay_factory is None:
                key = os.environ.get("LINKAPI_KEY") or Path("/root/.linkapi_key").read_text().strip()
                relay = judge.Relay(key)
            else:
                relay = relay_factory()
            for item, previous in originals.items():
                run.billing_check(relay, budget_cny)
                path = run.game_path(*item)
                previous_bytes = path.read_bytes()
                if read_json(path) != previous:
                    raise RuntimeError("Current game changed before retry")
                archive = run.directory / "state/attempts" / item[0] / f"{item[1]}-{item[2]}" / f"{previous['local_request_id']}.json"
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    if archive.read_bytes() != previous_bytes:
                        raise RuntimeError("Existing raw attempt archive differs")
                else:
                    with archive.open("xb") as stream:
                        stream.write(previous_bytes);stream.flush();os.fsync(stream.fileno())
                    descriptor = os.open(archive.parent, os.O_RDONLY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
                request = previous["request"]
                if request_bytes(request) != request_bytes(run.request(*item)):
                    raise RuntimeError("Retry request bytes changed")
                record = dict(policy, tag=item[0], uid=item[1], order=item[2], status="inflight",
                              request=request, request_sha256=previous["request_sha256"],
                              local_request_id=str(uuid.uuid4()), attempt=previous.get("attempt", 0) + 1,
                              supersedes_local_request_id=previous["local_request_id"], started_at=now())
                record.update(baseline_model=run.baseline_model, protocol_sha256=judge.digest(run.protocol))
                judge.atomic_json(path, record)
                try:
                    try:
                        response = relay.judge_call(request)
                        record.update({key: response.get(key) for key in
                                       ("answer", "usage", "finish_reason", "response_id", "response_model", "provider_request_id")})
                        score = judge.parse_score(record["answer"], run.protocol["regex_patterns"])
                        okay = score is not None and record["finish_reason"] == "stop" and judge.valid_usage(record["usage"])
                        record.update(score=score, status="valid" if okay else "invalid")
                    except Exception as error:
                        record.update(status="ambiguous", error_type=type(error).__name__)
                    record["finished_at"] = now()
                    judge.atomic_json(path, record)
                    run._records[item] = record
                    run._dirty_tags.add(item[0])
                    run.export()
                    ledger["attempts"].append({"game": list(item), "attempt": record["attempt"],
                                               "status": record["status"], "record_sha256": file_hash(path)})
                    judge.atomic_json(ledger_path, ledger)
                finally:
                    # The frozen driver does not settle failed batches; refresh even on invalid responses.
                    ledger["billing_after"] = run.billing_check(relay, budget_cny)
                    judge.atomic_json(ledger_path, ledger)
                if record["status"] == "ambiguous":
                    raise RuntimeError("Retry transport ambiguous; no further automatic attempts")
            complete = all(row["status"] == "valid" for row in ledger["attempts"])
            ledger.update(status="complete" if complete else "invalid_remaining", complete=complete, finished_at=now())
            judge.atomic_json(ledger_path, ledger)
            return ledger
        except BaseException as error:
            ledger.update(status="blocked", error_type=type(error).__name__, finished_at=now())
            judge.atomic_json(ledger_path, ledger)
            raise
        finally:
            if relay is not None and hasattr(relay, "close"):
                relay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", required=True, metavar="TAG:UID:ORDER")
    parser.add_argument("--budget-cny", type=float, required=True)
    args = parser.parse_args()
    games = []
    for value in args.game:
        fields = value.split(":")
        if len(fields) != 3 or fields[2] not in ("0", "1"):
            parser.error("game must be TAG:UID:0 or TAG:UID:1")
        games.append((fields[0], fields[1], int(fields[2])))
    result = retry_games(SUITE, games, budget_cny=args.budget_cny)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Structural retry stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None
