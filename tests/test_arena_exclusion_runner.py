"""Offline resume/exclusion lifecycle using the frozen judge and real disk state."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("base", "sft-init", "grpo", "lam2", "lam4", "lam8")
BASELINE = "gpt-4o-mini-2024-07-18"
SOURCE_SUITE = Path(os.environ.get("ARENA_EXCLUSIONS_SOURCE_SUITE", str(
    ROOT / "runs/arena-hard-v2-gpt4o-judge-20260917")))
if not SOURCE_SUITE.is_dir():
    SOURCE_SUITE = Path("/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/arena-hard-v2-gpt4o-judge-20260917")


def load_runner():
    path = ROOT / "scripts/run_arena_with_exclusions.py"
    assert path.is_file(), "The additive exclusions runner has not been implemented"
    spec = importlib.util.spec_from_file_location("exclusions_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(path):
    spec = importlib.util.spec_from_file_location("fixture_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def suite(tmp_path):
    assert SOURCE_SUITE.is_dir(), "Set ARENA_EXCLUSIONS_SOURCE_SUITE to the frozen GPT-4o suite"
    for relative in ("continue_evaluation.py", "source/scripts/judge_arena_hard.py",
                     "source/third_party/arena_hard/config/arena-hard-v2.0.yaml",
                     "source/third_party/arena_hard/utils/judge_utils.py",
                     "source/third_party/arena_hard/gen_judgment.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_SUITE / relative, destination)
    questions = [{"uid": f"u{i}", "category": "hard_prompt", "prompt": f"Question {i}"}
                 for i in range(500)]
    write_jsonl(tmp_path / "question.jsonl", questions)
    for tag in (*TAGS, BASELINE):
        rows = [{"uid": q["uid"], "model": tag, "messages": [
            {"role": "user", "content": q["prompt"]},
            {"role": "assistant", "content": {"answer": f"{tag} answer {q['uid']}"}}]}
            for q in questions]
        write_jsonl(tmp_path / "model_answer" / f"{tag}.jsonl", rows)
    (tmp_path / "exclusions_policy.json").write_text(json.dumps({
        "policy_id": "arena_judge_output_exclusions_v2", "judge": "gpt-4o", "baseline": BASELINE}))
    return tmp_path


def frozen_run(suite):
    return load_module(suite / "continue_evaluation.py").load_judge_run(suite)


def response(text="Reasoning. [[B>A]]", finish="stop"):
    return {"answer": text, "finish_reason": finish, "response_id": "offline-response",
            "response_model": "gpt-4o", "provider_request_id": "offline-provider-id",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


class FakeRelay:
    """Only the external billing/judging boundary is replaced."""
    def __init__(self, outcomes=(), usage_values=None):
        self.outcomes = iter(outcomes)
        self.usage_values = iter(usage_values or [100.0] * 1000)
        self.calls, self.usage_calls = [], []
        self.closed = False
        self.lock = threading.Lock()

    def usage(self, start):
        self.usage_calls.append(start)
        return next(self.usage_values)

    def judge_call(self, request):
        with self.lock:
            self.calls.append(request)
            outcome = next(self.outcomes, response())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def seed(suite, tag="base", uid="u0", order=0, *, status="valid", attempt=0, **changes):
    run = frozen_run(suite)
    run.prepare()
    record = run.save_inflight(tag, uid, order)
    if status != "inflight":
        record.update(response(), status=status, score="B>A", finished_at="2026-09-17T00:00:00+00:00")
    record.update(attempt=attempt, **changes)
    path = run.game_path(tag, uid, order)
    path.write_text(json.dumps(record))
    return path


def invoke(suite, relay=None, **kwargs):
    return load_runner().run_suite(suite, policy=suite / "exclusions_policy.json",
                                   relay_factory=(lambda: relay) if relay else lambda: pytest.fail("No network expected"),
                                   budget_cny=kwargs.pop("budget_cny", 10), **kwargs)


def test_old_and_new_malformed_outputs_are_terminal_and_other_batches_continue(suite):
    old = seed(suite, status="invalid", attempt=4, answer="No verdict", score=None)
    archived = old.parents[2] / "attempts/base/u0-0/old-attempt.json"
    archived.parent.mkdir(parents=True)
    archived.write_text('{"historical":"unchanged bytes"}\n')
    before, history = old.read_bytes(), archived.read_bytes()
    relay = FakeRelay([response("No verdict"), response("unfinished [[A>B]]", "length")])
    result = invoke(suite, relay, workers=2, max_requests=5)
    assert result["status"] == "partial" and result["complete"] is False
    assert result["dispatched_this_run"] == 5
    assert result["counts"]["total"] == {"missing": 5994, "valid": 3, "judge_failed": 3, "blocked": 0}
    assert old.read_bytes() == before and archived.read_bytes() == history
    assert len(relay.calls) == 5 and len(relay.usage_calls) == 4 and relay.closed
    assert not (suite / "exclusions_judging_complete.json").exists()
    assert json.loads((suite / "exclusions_progress.json").read_text())["status"] == "partial"
    saved = {path: path.read_bytes() for path in old.parents[1].glob("*/*.json")}
    second = FakeRelay()
    again = invoke(suite, second, workers=2, max_requests=2)
    assert again["counts"]["total"]["judge_failed"] == 3
    assert again["counts"]["total"]["valid"] == 5
    assert all(path.read_bytes() == data for path, data in saved.items())
    assert not {json.dumps(item, sort_keys=True) for item in relay.calls} & {
        json.dumps(item, sort_keys=True) for item in second.calls}


def test_frozen_exact_requests_cover_both_orders_without_repeating_on_resume(suite):
    relay = FakeRelay()
    invoke(suite, relay, workers=2, max_requests=2)
    run = frozen_run(suite)
    expected = [run.request("base", "u0", order) for order in (0, 1)]
    assert relay.calls == expected
    assert relay.calls[0]["model"] == "gpt-4o"
    assert relay.calls[0]["max_tokens"] == 16000 and relay.calls[0]["temperature"] == 0
    rows = [json.loads(line) for line in (run.directory / "base.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and len(rows[0]["games"]) == 2 and rows[0]["baseline"] == BASELINE
    second = FakeRelay()
    invoke(suite, second, max_requests=2)
    assert second.calls == [run.request("sft-init", "u0", order) for order in (0, 1)]


@pytest.mark.parametrize("change", [{"policy_id": "wrong"}, {"judge": "gpt-4.1"},
                                     {"baseline": "o3-mini-2025-01-31"}])
def test_unapproved_policy_rejected_before_any_mutation_or_network(suite, change):
    policy = suite / "exclusions_policy.json"
    value = json.loads(policy.read_text()); value.update(change); policy.write_text(json.dumps(value))
    before = {str(path.relative_to(suite)) for path in suite.rglob("*")}
    with pytest.raises(ValueError, match="policy|judge|baseline"):
        invoke(suite, max_requests=1)
    assert {str(path.relative_to(suite)) for path in suite.rglob("*")} == before


@pytest.mark.parametrize("status,changes", [
    ("inflight", {}), ("ambiguous", {"error_type": "ReadTimeout"}),
    ("invalid", {"answer": "No verdict", "score": None, "usage": None}),
    ("invalid", {"answer": "No verdict", "score": "B>A"}),
    ("invalid", {"answer": "No verdict", "score": None, "finish_reason": "content_filter"}),
])
def test_existing_ambiguous_or_corrupt_record_blocks_all_dispatch(suite, status, changes):
    path = seed(suite, status=status, **changes)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="blocked"):
        invoke(suite, max_requests=3)
    assert path.read_bytes() == before
    progress = json.loads((suite / "exclusions_progress.json").read_text())
    assert progress["status"] == "blocked" and progress["counts"]["total"]["blocked"] == 1
    assert not (suite / "exclusions_judging_complete.json").exists()


def test_network_failure_persists_ambiguity_and_finishes_current_batch_only(suite, capsys):
    secret = "credential-that-must-never-appear"
    relay = FakeRelay([RuntimeError(secret), response()])
    with pytest.raises(RuntimeError, match="blocked") as error:
        invoke(suite, relay, workers=2, max_requests=6)
    assert len(relay.calls) == 2 and relay.closed
    records = [json.loads(path.read_text()) for path in (suite / "model_judgment/gpt-4o/state/games").glob("*/*.json")]
    assert sorted(row["status"] for row in records) == ["ambiguous", "valid"]
    assert secret not in str(error.value) + capsys.readouterr().out + capsys.readouterr().err
    assert all(secret not in path.read_text() for path in suite.rglob("*.json"))
    with pytest.raises(RuntimeError, match="blocked"):
        invoke(suite, max_requests=1)


def test_cumulative_billing_stops_before_following_batch_and_keeps_saved_games(suite):
    relay = FakeRelay(usage_values=[100, 110])
    with pytest.raises(RuntimeError, match="budget"):
        invoke(suite, relay, workers=2, max_requests=4, budget_cny=5)
    assert len(relay.calls) == 2 and relay.closed
    billing = json.loads((suite / "model_judgment/gpt-4o/state/billing.json").read_text())
    assert billing["usage0_cny"] == 100 and billing["spent_cny"] == 10
    assert not (suite / "exclusions_judging_complete.json").exists()


def test_exclusive_frozen_judge_lock_prevents_parallel_dispatch(suite):
    run = frozen_run(suite)
    judge = load_module(suite / "source/scripts/judge_arena_hard.py")
    with judge.exclusive_lock(run.directory):
        with pytest.raises(RuntimeError, match="Another judge"):
            invoke(suite, max_requests=1)
    assert not (run.directory / "state").exists()


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"workers": 33}, {"workers": True},
                                    {"max_requests": -1}, {"max_requests": True},
                                    {"budget_cny": float("nan")}, {"budget_cny": 0}])
def test_invalid_limits_rejected_before_state_changes(suite, kwargs):
    with pytest.raises(ValueError):
        invoke(suite, **kwargs)
    assert not (suite / "model_judgment").exists()


def test_completion_requires_all_6000_terminal_games_and_resume_makes_zero_calls(suite):
    run = frozen_run(suite)
    run.prepare()
    judge = load_module(suite / "source/scripts/judge_arena_hard.py")
    missing = {("lam8", "u499", 0), ("lam8", "u499", 1)}
    for tag in TAGS:
        for uid in run.questions:
            for order in (0, 1):
                if (tag, uid, order) in missing:
                    continue
                request = run.request(tag, uid, order)
                row = dict(response(), tag=tag, uid=uid, order=order, status="valid", score="B>A",
                           attempt=0, local_request_id=f"{tag}-{uid}-{order}", request=request,
                           request_sha256=judge.digest(request), baseline_model=BASELINE, judge_model="gpt-4o",
                           protocol_sha256=judge.digest(run.protocol), started_at="2026-09-17T00:00:00+00:00",
                           finished_at="2026-09-17T00:00:01+00:00")
                path = run.game_path(tag, uid, order)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))
    partial = invoke(suite, FakeRelay(), workers=1, max_requests=1)
    assert partial["complete"] is False and partial["counts"]["total"]["missing"] == 1
    assert not (suite / "exclusions_judging_complete.json").exists()
    complete = invoke(suite, FakeRelay([response("No verdict")]), workers=1)
    assert complete["status"] == "complete_with_judge_exclusions" and complete["complete"] is True
    assert complete["expected_games"] == 6000
    assert complete["counts"]["total"] == {"missing": 0, "valid": 5999, "judge_failed": 1, "blocked": 0}
    assert complete["counts"]["by_model"]["lam8"]["judge_failed"] == 1
    assert complete["complete_pairs"]["lam8"] == 499
    assert all(complete["complete_pairs"][tag] == 500 for tag in TAGS[:-1])
    assert json.loads((suite / "exclusions_judging_complete.json").read_text())["complete"] is True
    again = invoke(suite)
    assert again["status"] == "complete_with_judge_exclusions" and again["dispatched_this_run"] == 0


def test_cli_sanitizes_unexpected_exceptions_and_does_not_echo_environment(suite):
    path = ROOT / "scripts/run_arena_with_exclusions.py"
    assert path.exists(), "The exclusions CLI has not been implemented"
    completed = subprocess.run([sys.executable, str(path), "--suite", str(suite), "--policy",
                                str(suite / "exclusions_policy.json"), "--budget-cny", "10",
                                "--max-requests", "1"], capture_output=True, text=True,
                               env={**os.environ, "LINKAPI_KEY": "secret-test-credential"})
    assert completed.returncode != 0  # Fixture deliberately has no resilient relay module.
    assert "secret-test-credential" not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stderr
    assert "stopped" in completed.stderr


@pytest.mark.parametrize("damage", [{"usage0_cny": float("nan")}, {"latest_usage_cny": -1},
                                     {"spent_cny": 0}, {"usage0_cny": "100"}])
def test_corrupt_persisted_billing_blocks_before_relay_or_paid_calls(suite, damage):
    run = frozen_run(suite)
    run.prepare()
    billing = {"start_date": "2026-09-01", "usage0_cny": 100,
               "latest_usage_cny": 101, "spent_cny": 1}
    billing.update(damage)
    path = run.directory / "state/billing.json"
    path.write_text(json.dumps(billing))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="billing"):
        invoke(suite, max_requests=1)
    assert path.read_bytes() == before
    assert not (run.directory / "state/games").exists()


def test_existing_and_new_content_filter_are_terminal_without_paid_retry(suite):
    old = seed(suite, status='invalid', answer=None, score=None, finish_reason='content_filter')
    before = old.read_bytes()
    relay = FakeRelay([response(None, 'content_filter'), response()])
    result = invoke(suite, relay, workers=1, max_requests=2)
    assert old.read_bytes() == before
    assert result['counts']['total'] == {'missing': 5997, 'valid': 1, 'judge_failed': 2, 'blocked': 0}
    assert relay.calls == [frozen_run(suite).request('base', 'u0', 1),
                           frozen_run(suite).request('sft-init', 'u0', 0)]
    second = FakeRelay()
    again = invoke(suite, second, max_requests=1)
    assert again['counts']['total']['judge_failed'] == 2
    assert second.calls == [frozen_run(suite).request('sft-init', 'u0', 1)]
