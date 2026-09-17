#!/usr/bin/env python3
"""Official Arena-Hard v2 hard-prompt judging, with durable per-game accounting.

Each request uses GPT-4.1, the pinned official system/template, temperature 0,
max_tokens 16000, and one of both answer orders against o3-mini-2025-01-31.
No Alpaca m/M or logprob protocol is used. Only complete, valid two-game pairs
are exported in official JSONL format; all requests retain separate state.

Transport failures, invalid judgments, and interrupted inflight requests are
never automatically retried. --budget-cny is a cumulative dispatch guard for
this output directory, not a hard spending cap: billing delays and one bounded
batch of in-flight requests can overshoot it. --max-requests limits new calls.
Pilot --uids selections share the same records with later full-coverage runs.
After inspecting an unresolved game, --retry-game TAG:UID:ORDER explicitly
authorizes one replacement attempt, retaining the previous attempt unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.linkapi.ai/v1"
JUDGE_MODEL = "gpt-4.1"
BASELINE_MODEL = "o3-mini-2025-01-31"
SCORES = frozenset(("A>>B", "A>B", "A=B", "B>A", "B>>A"))
PATTERNS = [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_jsonl(path):
    # splitlines() corrupts legal JSON strings containing U+2028/U+2029/NEL.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".judge.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another judge process holds this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_protocol(official_root):
    import yaml
    official_root = Path(official_root)
    config_path = official_root / "config/arena-hard-v2.0.yaml"
    settings_path = official_root / "utils/judge_utils.py"
    config = yaml.safe_load(config_path.read_text())
    settings = runpy.run_path(str(settings_path))["JUDGE_SETTINGS"]["hard_prompt"]
    require(config["judge_model"] == JUDGE_MODEL and config["temperature"] == 0.0
            and config["max_tokens"] == 16000 and config["reference"] is None
            and config["regex_patterns"] == PATTERNS and settings["baseline"] == BASELINE_MODEL,
            "Unexpected pinned official judge protocol")
    return {
        "protocol": "arena_hard_v2_gpt41_two_order_v1",
        "judge": JUDGE_MODEL, "baseline": BASELINE_MODEL,
        "temperature": 0.0, "max_tokens": 16000,
        "system_prompt": settings["system_prompt"], "prompt_template": config["prompt_template"],
        "regex_patterns": config["regex_patterns"], "base_url": BASE_URL,
        "source_sha256": {name: hashlib.sha256((official_root / name).read_bytes()).hexdigest()
                          for name in ("config/arena-hard-v2.0.yaml", "utils/judge_utils.py", "gen_judgment.py")},
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def parse_score(text, patterns=PATTERNS):
    """Official pattern priority and last match, then restrict to five outcomes."""
    if not isinstance(text, str):
        return None
    for pattern in patterns:
        matches = [match for match in re.findall(pattern, text.upper()) if match != ""]
        if matches:
            result = matches[-1].strip("\n")
            return result if result in SCORES else None
    return None


def make_request(question, baseline, answer, order, protocol):
    require(order in (0, 1), "Invalid answer order")
    first, second = (baseline, answer) if order == 0 else (answer, baseline)
    return {
        "model": JUDGE_MODEL, "temperature": protocol["temperature"], "max_tokens": protocol["max_tokens"],
        "messages": [
            {"role": "system", "content": protocol["system_prompt"]},
            {"role": "user", "content": protocol["prompt_template"].format(
                QUESTION=question["prompt"], ANSWER_A=first["messages"][-1]["content"]["answer"],
                ANSWER_B=second["messages"][-1]["content"]["answer"])},
        ],
    }


def indexed(rows, label):
    require(len(rows) == 500, f"{label}: expected exactly 500 rows")
    result = {}
    for row in rows:
        uid = row.get("uid")
        require(isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", uid), f"{label}: invalid uid")
        require(uid not in result, f"{label}: duplicate uid {uid}")
        result[uid] = row
    return result


def validate_answers(rows, questions, label, model=None):
    answers = indexed(rows, label)
    require(set(answers) == set(questions), f"{label}: uid coverage mismatch")
    models = set()
    for uid, answer in answers.items():
        model_name = answer.get("model")
        require(isinstance(model_name, str) and bool(model_name), f"{label}: missing model")
        models.add(model_name)
        messages = answer.get("messages")
        require(isinstance(messages, list) and len(messages) == 2, f"{label}: expected user/assistant messages")
        require(messages[0].get("role") == "user" and messages[0].get("content") == questions[uid]["prompt"],
                f"{label}: prompt mismatch for {uid}")
        require(messages[-1].get("role") == "assistant" and isinstance(messages[-1].get("content"), dict)
                and isinstance(messages[-1]["content"].get("answer"), str), f"{label}: missing answer for {uid}")
    require(len(models) == 1 and (model is None or models == {model}), f"{label}: wrong/mixed model identity")
    return answers


def valid_usage(usage):
    return (isinstance(usage, dict)
            and all(type(usage.get(key)) is int and usage[key] >= 0
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])


class Relay:
    """One HTTP attempt per paid call; never include response errors/keys in logs."""
    def __init__(self, key, timeout=600):
        import httpx
        require(isinstance(key, str) and bool(key.strip()), "Missing relay credential")
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                                   limits=httpx.Limits(max_connections=32))

    def usage(self, start_date):
        try:
            response = self.client.get(f"{BASE_URL}/dashboard/billing/usage", params={
                "start_date": start_date, "end_date": datetime.now(timezone.utc).date().isoformat()})
            response.raise_for_status()
            payload = response.json()
            amount = float(payload["total_usage"])
            if "error" in payload or not math.isfinite(amount) or amount < 0:
                raise ValueError("invalid usage")
            return amount / 100.0
        except Exception:
            raise RuntimeError("Cannot verify relay usage; dispatch stopped") from None

    def judge_call(self, request):
        response = self.client.post(f"{BASE_URL}/chat/completions", json=request)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError("Relay returned an error")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Relay returned invalid choices")
        choice = choices[0]
        return {"answer": choice.get("message", {}).get("content"), "usage": payload.get("usage"),
                "finish_reason": choice.get("finish_reason"), "response_id": payload.get("id"),
                "response_model": payload.get("model"), "provider_request_id": response.headers.get("x-request-id")}

    def close(self):
        self.client.close()


class JudgeRun:
    def __init__(self, output_dir, questions, baseline, answers, protocol):
        self.directory = Path(output_dir)
        self.protocol = protocol
        self.questions = indexed(questions, "questions")
        require(all(q.get("category") == "hard_prompt" and isinstance(q.get("prompt"), str)
                    and q["prompt"].strip() for q in questions), "Only 500 nonempty hard_prompt questions are accepted")
        require(len({q["prompt"] for q in questions}) == 500, "Duplicate question prompts")
        self.baseline = validate_answers(baseline, self.questions, "baseline", BASELINE_MODEL)
        require(isinstance(answers, dict) and bool(answers), "No candidate answers")
        self.answers = {}
        for tag, rows in answers.items():
            require(isinstance(tag, str) and re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", tag), "Invalid tag")
            self.answers[tag] = validate_answers(rows, self.questions, tag, model=tag)
        self.identity = {"protocol": protocol, "questions_sha256": digest(questions),
                         "baseline_sha256": digest(baseline)}
        self._records = {}
        self._export_counts = {}
        self._dirty_tags = set(self.answers)

    def bind(self, path, identity):
        if path.exists():
            require(json.loads(path.read_text()) == identity, f"Saved identity mismatch: {path.name}")
        else:
            atomic_json(path, identity)

    def prepare(self):
        # One locked invocation owns this cache; reread disk on any new run.
        self._records.clear()
        self._export_counts.clear()
        self._dirty_tags = set(self.answers)
        self.bind(self.directory / "state/protocol.json", self.identity)
        for tag, rows in self.answers.items():
            model_path = self.directory / "state/models" / f"{tag}.json"
            if not model_path.exists() and (self.directory / f"{tag}.jsonl").exists():
                raise ValueError(f"Existing {tag} output has no per-game identity records; refusing overwrite")
            self.bind(model_path, {"answers_sha256": digest(list(rows.values())), "tag": tag,
                                   "model": next(iter(rows.values()))["model"]})

    def game_path(self, tag, uid, order):
        return self.directory / "state/games" / tag / f"{uid}-{order}.json"

    def request(self, tag, uid, order):
        return make_request(self.questions[uid], self.baseline[uid], self.answers[tag][uid], order, self.protocol)

    def load_record(self, tag, uid, order):
        item = (tag, uid, order)
        if item in self._records:
            return self._records[item]
        path = self.game_path(tag, uid, order)
        if not path.exists():
            self._records[item] = None
            return None
        record = json.loads(path.read_text())
        request = self.request(tag, uid, order)
        require(record.get("tag") == tag and record.get("uid") == uid and record.get("order") == order
                and record.get("request") == request and record.get("request_sha256") == digest(request),
                f"Saved game identity mismatch: {tag}/{uid}/{order}")
        require(record.get("status") in ("inflight", "ambiguous", "invalid", "valid"), "Unknown game state")
        if record["status"] == "valid":
            require(record.get("score") in SCORES
                    and parse_score(record.get("answer"), self.protocol["regex_patterns"]) == record["score"]
                    and record.get("finish_reason") == "stop" and valid_usage(record.get("usage")),
                    f"Corrupt valid game: {tag}/{uid}/{order}")
        self._records[item] = record
        return record

    def save_inflight(self, tag, uid, order, *, retry=False):
        path = self.game_path(tag, uid, order)
        previous = self.load_record(tag, uid, order)
        if retry:
            require(previous is not None and previous["status"] != "valid" and previous.get("attempt", 0) == 0,
                    "Explicit retry requires an unresolved first attempt")
            archive = (self.directory / "state/attempts" / tag / f"{uid}-{order}"
                       / f"{previous['local_request_id']}.json")
            archive.parent.mkdir(parents=True, exist_ok=True)
            previous_bytes = path.read_bytes()
            if archive.exists():
                require(archive.read_bytes() == previous_bytes, "Prior retry archive changed")
            else:
                # Exclusive creation makes historical attempt files immutable here.
                with archive.open("xb") as stream:
                    stream.write(previous_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                descriptor = os.open(archive.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        else:
            require(previous is None and not path.exists(), "Refusing to overwrite an existing paid request")
        request = self.request(tag, uid, order)
        record = {"tag": tag, "uid": uid, "order": order, "status": "inflight", "started_at": now(),
                  "local_request_id": str(uuid.uuid4()), "request": request, "request_sha256": digest(request),
                  "attempt": 1 if retry else 0}
        if retry:
            record["supersedes_local_request_id"] = previous["local_request_id"]
        atomic_json(path, record)
        self._records[(tag, uid, order)] = record
        self._dirty_tags.add(tag)
        return record

    def export(self):
        counts = {}
        for tag, answers in self.answers.items():
            if tag not in self._dirty_tags:
                counts[tag] = self._export_counts[tag]
                continue
            rows = []
            for uid, question in self.questions.items():
                pair = [self.load_record(tag, uid, order) for order in (0, 1)]
                if not all(record and record["status"] == "valid" for record in pair):
                    continue
                rows.append({"uid": uid, "category": question["category"], "judge": JUDGE_MODEL,
                             "model": answers[uid]["model"], "baseline": BASELINE_MODEL,
                             "games": [{"score": r["score"], "judgment": {"answer": r["answer"]},
                                        "prompt": r["request"]["messages"]} for r in pair]})
            atomic_text(self.directory / f"{tag}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            counts[tag] = len(rows)
            self._export_counts[tag] = len(rows)
            self._dirty_tags.discard(tag)
        return counts

    def billing_check(self, relay, budget):
        path = self.directory / "state/billing.json"
        billing = json.loads(path.read_text()) if path.exists() else {
            "start_date": datetime.now(timezone.utc).date().replace(day=1).isoformat()}
        usage = relay.usage(billing["start_date"])
        if not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0:
            raise RuntimeError("Invalid relay usage; dispatch stopped")
        baseline = billing.setdefault("usage0_cny", usage)
        if usage < baseline - 1e-6 or usage < billing.get("latest_usage_cny", baseline) - 1e-6:
            raise RuntimeError("Relay usage decreased; dispatch stopped")
        billing.update(latest_usage_cny=usage, spent_cny=max(0.0, usage - baseline),
                       checked_at=now(), dispatch_guard_cny=budget)
        atomic_json(path, billing)
        if billing["spent_cny"] >= budget:
            raise RuntimeError("Cumulative budget dispatch guard reached; paid progress is saved")
        return billing

    def run(self, relay_factory, uids=None, *, workers=4, budget_cny, max_requests=0, retry_games=None):
        require(type(workers) is int and 1 <= workers <= 32, "workers must be between 1 and 32")
        require(math.isfinite(budget_cny) and budget_cny > 0, "budget-cny must be finite and positive")
        require(type(max_requests) is int and max_requests >= 0, "max-requests must be nonnegative")
        selected = list(self.questions) if uids is None else list(uids)
        require(selected and len(selected) == len(set(selected)) and set(selected) <= set(self.questions),
                "uids must be unique known question IDs")
        retries = list(retry_games or [])
        require(len(retries) == len(set(retries)) and all(
            len(item) == 3 and item[0] in self.answers and item[1] in selected and item[2] in (0, 1)
            for item in retries), "Invalid or duplicate explicit retry selection")
        retries = set(retries)
        with exclusive_lock(self.directory):
            self.prepare()
            pending, blocked = [], []
            for uid in selected:
                for tag in self.answers:
                    for order in (0, 1):
                        record = self.load_record(tag, uid, order)
                        item = (tag, uid, order)
                        if item in retries:
                            require(record is not None and record["status"] != "valid" and record.get("attempt", 0) == 0,
                                    "Explicit retry requires an unresolved first attempt; successful/second attempts cannot retry")
                            pending.append(item)
                        elif record is None:
                            pending.append((tag, uid, order))
                        elif record["status"] != "valid":
                            blocked.append((tag, uid, order, record["status"]))
            if blocked:
                raise RuntimeError(f"Judge run blocked by {len(blocked)} unresolved paid requests; no automatic retry")
            relay, dispatched = None, 0
            try:
                if pending:
                    relay = relay_factory()
                while pending and (not max_requests or dispatched < max_requests):
                    self.billing_check(relay, budget_cny)
                    batch_size = min(workers, len(pending), max_requests - dispatched if max_requests else workers)
                    batch, pending = pending[:batch_size], pending[batch_size:]
                    records = {item: self.save_inflight(*item, retry=item in retries) for item in batch}
                    failures = []
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(relay.judge_call, record["request"]): item
                                   for item, record in records.items()}
                        dispatched += len(batch)
                        for future in as_completed(futures):
                            item = futures[future]
                            record = records[item]
                            try:
                                response = future.result()
                                # Persist usable judge text/usage even if score or finish is invalid.
                                record.update({key: response.get(key) for key in
                                               ("answer", "usage", "finish_reason", "response_id", "response_model", "provider_request_id")})
                                score = parse_score(record["answer"], self.protocol["regex_patterns"])
                                okay = score is not None and record["finish_reason"] == "stop" and valid_usage(record["usage"])
                                record.update(status="valid" if okay else "invalid", score=score)
                            except Exception as error:
                                # No exception text: providers/transport errors may echo credentials.
                                record.update(status="ambiguous", error_type=type(error).__name__)
                            record["finished_at"] = now()
                            atomic_json(self.game_path(*item), record)
                            self._records[item] = record
                            self._dirty_tags.add(item[0])
                            if record["status"] != "valid":
                                failures.append(item)
                    counts = self.export()
                    progress = {"dispatched_this_run": dispatched, "selected_uids": selected,
                                "complete_pairs": counts, "blocked_games": len(failures), "updated_at": now()}
                    atomic_json(self.directory / "progress.json", progress)
                    if failures:
                        raise RuntimeError(f"Judge run blocked by {len(failures)} invalid/ambiguous requests; progress saved")
                counts = self.export()
                if relay is not None:
                    self.billing_check(relay, budget_cny)
                summary = {"complete": not pending, "dispatched_this_run": dispatched,
                           "selected_uid_count": len(selected), "selected_game_count": len(selected) * len(self.answers) * 2,
                           "complete_pairs": counts, "remaining_selected_games": len(pending), "updated_at": now()}
                atomic_json(self.directory / "summary.json", summary)
                return summary
            finally:
                if relay is not None and hasattr(relay, "close"):
                    relay.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--answers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--official-root", type=Path, default=ROOT / "third_party/arena_hard")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uids", nargs="+")
    group.add_argument("--uids-file", type=Path, help="JSON array of fixed pilot UIDs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-cny", type=float, required=True, help="Cumulative output-directory dispatch guard")
    parser.add_argument("--max-requests", type=int, default=0, help="Maximum NEW paid requests this invocation; 0 means all")
    parser.add_argument("--key-file", type=Path, default=Path("/root/.linkapi_key"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retry-game", action="append", default=[], metavar="TAG:UID:ORDER",
                        help="Explicitly retry one inspected invalid/ambiguous/inflight game once; retains old paid attempt")
    parser.add_argument("--dry-run", action="store_true", help="Validate protocol/full inputs and selection; no writes/network")
    args = parser.parse_args()
    require(len(args.tags) == len(set(args.tags)), "Duplicate model tags")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
    require(1 <= args.workers <= 32 and args.max_requests >= 0, "Invalid worker/request limit")
    require(math.isfinite(args.budget_cny) and args.budget_cny > 0, "budget-cny must be finite and positive")
    retries = []
    for spec in args.retry_game:
        pieces = spec.split(":")
        require(len(pieces) == 3 and pieces[2] in ("0", "1"), "retry-game must be TAG:UID:0 or TAG:UID:1")
        retries.append((pieces[0], pieces[1], int(pieces[2])))
    protocol = load_protocol(args.official_root)
    run = JudgeRun(args.output_dir, load_jsonl(args.questions), load_jsonl(args.baseline),
                   {tag: load_jsonl(args.answers_dir / f"{tag}.jsonl") for tag in args.tags}, protocol)
    uids = json.loads(args.uids_file.read_text()) if args.uids_file else args.uids
    if uids is not None:
        require(isinstance(uids, list) and uids and all(isinstance(x, str) for x in uids)
                and len(uids) == len(set(uids)) and set(uids) <= set(run.questions), "Invalid UID selection")
    if args.dry_run:
        print(json.dumps({"status": "inputs_validated", "models": args.tags, "questions": 500,
                          "selected_uids": len(uids) if uids else 500, "protocol_sha256": digest(protocol)}))
        return
    def relay_factory():
        key = os.environ.get("LINKAPI_KEY") or args.key_file.read_text().strip()
        return Relay(key, timeout=args.timeout)
    result = run.run(relay_factory, uids, workers=args.workers, budget_cny=args.budget_cny,
                     max_requests=args.max_requests, retry_games=retries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Arena judge stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None
