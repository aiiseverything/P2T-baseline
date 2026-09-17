"""Offline continuation gates; fake transports never contact a judge endpoint."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUITE = ROOT / "runs/arena-hard-v2-canonical-20260917"
VERIFIED = {"status": "passed", "current_games": 6000, "exact_request_and_parse_checks": 6000,
            "judgments": 3000, "questions": 500, "numeric_comparisons": 42, "max_absolute_difference": 0.0}
spec = importlib.util.spec_from_file_location("arena_continuation", SOURCE_SUITE / "continue_evaluation.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "source").symlink_to(SOURCE_SUITE / "source", target_is_directory=True)
    (tmp_path / "job").mkdir()
    questions = [{"uid": f"u{i}", "category": "hard_prompt", "prompt": f"Question {i}"} for i in range(500)]
    write_jsonl(tmp_path / "question.jsonl", questions)
    for tag in (*continuation.TAGS, "o3-mini-2025-01-31"):
        rows = [{"uid": q["uid"], "model": tag, "messages": [
            {"role": "user", "content": q["prompt"]},
            {"role": "assistant", "content": {"answer": tag + " answer"}}]} for q in questions]
        write_jsonl(tmp_path / "model_answer" / f"{tag}.jsonl", rows)
    for name in ["run_evaluation.py", "verify_final_scoring.py", "run_full_judging.sh"]:
        (tmp_path / name).write_text("# test fixture\n")
    for name in ["retry_policy.json", "retry_invalid_judgments.py"]:
        (tmp_path / name).write_bytes((SOURCE_SUITE / name).read_bytes())
    (tmp_path / "pilot_uids.json").write_text(json.dumps([f"u{i}" for i in range(5)]))
    (tmp_path / "experiment.json").write_text(json.dumps({"models": {tag: {"adapter": tag} for tag in continuation.TAGS}}))
    (tmp_path / "status.json").write_text('{"state":"generation_complete"}')
    (tmp_path / "generation_complete.json").write_text('{"state":"generation_complete"}')
    return tmp_path


class FakeRelay:
    def __init__(self, text="[[A=B]]"):
        self.calls = []
        self.text = text
    def usage(self, _):
        return 100 + .01 * len(self.calls)
    def judge_call(self, request):
        self.calls.append(request)
        return {"answer": self.text, "finish_reason": "stop", "response_id": "test",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


def make_pilot(suite, relay=None, max_requests=0):
    run = continuation.load_judge_run(suite)
    relay = relay or FakeRelay()
    run.run(lambda: relay, [f"u{i}" for i in range(5)], workers=12, budget_cny=20,
            max_requests=max_requests)
    return relay


@pytest.mark.parametrize("cost,estimate,budget,approved", [
    (.1, 10, 30, True), (.6, 60, 180, True), (3, 300, 900, True),
    (3.01, 301, 903, False),
])
def test_cost_extrapolation_and_review_gate(cost, estimate, budget, approved):
    result = continuation.calculate_cost_decision(cost)
    assert result["estimated_full_cny"] == pytest.approx(estimate)
    assert result["dispatch_budget_cny"] == budget
    assert result["automatic_continuation"] is approved


@pytest.mark.parametrize("cost", [0, -1, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_cost_rejected(cost):
    with pytest.raises(ValueError):
        continuation.calculate_cost_decision(cost)


def test_generation_failure_stops_without_polling(suite):
    (suite / "status.json").write_text('{"state":"generation_failed"}')
    with pytest.raises(RuntimeError, match="generation_failed"):
        continuation.wait_for_generation(suite, sleep=lambda _: pytest.fail("must not wait"))


def test_generation_poll_waits_for_completion_marker(suite):
    (suite / "generation_complete.json").unlink()
    (suite / "status.json").write_text('{"state":"generating"}')
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        (suite / "generation_complete.json").write_text('{"state":"generation_complete"}')
    continuation.wait_for_generation(suite, sleep=sleep)
    assert waits == [15]


def test_pilot_requires_exact_60_valid_games_and_five_pairs_per_model(suite):
    make_pilot(suite, max_requests=12)
    partial = continuation.pilot_status(suite)
    assert partial["valid_games"] == 12 and not partial["complete"]
    # Preserve billing baseline when continuing the fake pilot.
    relay = FakeRelay()
    relay.calls = [None] * 12
    make_pilot(suite, relay=relay)
    full = continuation.pilot_status(suite)
    assert full["valid_games"] == 60 and full["complete"]
    assert full["complete_pairs"] == {tag: 5 for tag in continuation.TAGS}


def test_unresolved_paid_game_is_blocked_without_automatic_retry(suite):
    with pytest.raises(RuntimeError):
        make_pilot(suite, FakeRelay("invalid verdict"))
    with pytest.raises(RuntimeError, match="unresolved"):
        continuation.pilot_status(suite)


def test_zero_cost_waits_read_only_and_preserves_baseline(suite):
    make_pilot(suite)
    path = suite / "model_judgment/gpt-4.1/state/billing.json"
    original = json.loads(path.read_text())
    original.update(latest_usage_cny=100, spent_cny=0)
    path.write_text(json.dumps(original))
    observed = iter([100, 100.6])
    waits, calls = [], []
    def reader(start):
        calls.append(start)
        return next(observed)
    cost = continuation.wait_for_positive_cost(suite, usage_reader=reader, sleep=waits.append)
    assert cost == pytest.approx(.6)
    assert len(calls) == 2 and waits
    assert json.loads(path.read_text())["usage0_cny"] == 100


def test_zero_cost_timeout_never_resets_baseline(suite):
    make_pilot(suite)
    path = suite / "model_judgment/gpt-4.1/state/billing.json"
    billing = json.loads(path.read_text());billing.update(latest_usage_cny=100, spent_cny=0)
    path.write_text(json.dumps(billing))
    ticks = iter([0, 0, 121])
    with pytest.raises(RuntimeError, match="cost"):
        continuation.wait_for_positive_cost(suite, usage_reader=lambda _:100, sleep=lambda _:None,
                                            monotonic=lambda:next(ticks))
    assert json.loads(path.read_text())["usage0_cny"] == 100


def test_restart_reuses_paid_pilot_and_persisted_cost_not_full_billing(suite):
    labels = []
    def runner(s, label, command):
        labels.append(label)
        if label == "pilot": make_pilot(s)
        if label == "full_judging":
            (s / "scores").mkdir(exist_ok=True)
            (s / "scores/results.json").write_text('{}')
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    continuation.continue_suite(suite, runner=runner)
    decision = json.loads((suite / "cost_decision.json").read_text())
    assert decision["pilot_cost_cny"] == pytest.approx(.6)
    assert decision["dispatch_budget_cny"] == 180
    path = suite / "model_judgment/gpt-4.1/state/billing.json"
    billing = json.loads(path.read_text());billing.update(latest_usage_cny=150, spent_cny=50)
    path.write_text(json.dumps(billing))
    labels.clear()
    continuation.continue_suite(suite, runner=runner,
                                usage_reader=lambda _:pytest.fail("must reuse pilot cost decision"))
    assert "pilot" not in labels and "full_judging" not in labels
    assert json.loads((suite / "cost_decision.json").read_text()) == decision
    assert json.loads((suite / "continuation_state.json").read_text())["state"] == "evaluation_complete"


def test_final_verifier_failure_never_marks_complete(suite):
    def runner(s, label, command):
        if label == "pilot": make_pilot(s)
        if label == "full_judging":
            (s / "scores").mkdir()
            (s / "scores/results.json").write_text('{}')
        return {"returncode": 1 if label == "verify_final" else 0, "stdout": '{}'}
    with pytest.raises(RuntimeError, match="verify_final"):
        continuation.continue_suite(suite, runner=runner)
    state = json.loads((suite / "continuation_state.json").read_text())
    assert state["state"] == "failed" and state["phase"] == "verify_final"


def test_prior_full_progress_without_cost_decision_requires_review(suite):
    make_pilot(suite)
    run = continuation.load_judge_run(suite)
    relay = FakeRelay();relay.calls = [None] * 60
    run.run(lambda:relay, ["u5"], workers=12, budget_cny=20)
    with pytest.raises(RuntimeError, match="cost decision"):
        continuation.continue_suite(suite, runner=lambda *_:{"returncode":0,"stdout":"{}"})
    assert not (suite / "cost_decision.json").exists()


def test_cost_boundary_never_rounds_an_excess_back_into_automatic_approval():
    assert continuation.calculate_cost_decision(3 + 1e-12)["automatic_continuation"] is False
    assert continuation.calculate_cost_decision(1e-12)["pilot_cost_cny"] > 0


def test_unexpected_pilot_cost_persists_decision_but_never_runs_full(suite):
    make_pilot(suite)
    path = suite / "model_judgment/gpt-4.1/state/billing.json"
    billing = json.loads(path.read_text());billing.update(latest_usage_cny=103.01, spent_cny=3.01)
    path.write_text(json.dumps(billing))
    labels = []
    def runner(s, label, command):
        labels.append(label)
        return {"returncode": 0, "stdout": '{}'}
    with pytest.raises(RuntimeError, match="Unexpected pilot cost"):
        continuation.continue_suite(suite, runner=runner)
    assert labels == ["validate_generation"]
    assert json.loads((suite / "cost_decision.json").read_text())["automatic_continuation"] is False
    assert json.loads((suite / "continuation_state.json").read_text())["state"] == "failed"


def test_exit_zero_without_strict_verification_fields_cannot_mark_complete(suite):
    make_pilot(suite)
    def runner(s, label, command):
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": '{"status":"passed"}'}
    with pytest.raises(RuntimeError, match="strict complete coverage"):
        continuation.continue_suite(suite, runner=runner)
    assert not (suite / "evaluation_complete.json").exists()


def test_singleton_lock_rejects_second_controller_without_running_commands(suite):
    judge = continuation.judge_module(suite)
    with judge.exclusive_lock(suite / "job/continuation-lock"):
        with pytest.raises(RuntimeError, match="holds"):
            continuation.continue_suite(suite, runner=lambda *_:pytest.fail("no child command"))


def test_resume_rejects_changed_previously_bound_cost_decision(suite):
    make_pilot(suite)
    def runner(s, label, command):
        if label == "full_judging": (s / "scores").mkdir(exist_ok=True)
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    continuation.continue_suite(suite, runner=runner)
    path = suite / "cost_decision.json"
    changed = json.loads(path.read_text())
    changed.update(continuation.calculate_cost_decision(1.2))
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="cost decision hash"):
        continuation.continue_suite(suite, runner=runner)


def test_controller_repairs_only_structural_invalid_then_reuses_completed_games(suite):
    relay = FakeRelay("No verdict")
    with pytest.raises(RuntimeError):
        continuation.load_judge_run(suite).run(lambda:relay, ["u0"], workers=1, budget_cny=20)
    labels = []
    def runner(s, label, command):
        labels.append(label)
        if label == "pilot_retry":
            relay.text = "[[A=B]]"
            helper = continuation.retry_module(s)
            result = helper.retry_games(s, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda:relay)
            return {"returncode": 0 if result["complete"] else 2, "stdout": '{}'}
        if label == "pilot": make_pilot(s, relay=relay)
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    continuation.continue_suite(suite, runner=runner)
    assert labels.count("pilot_retry") == 1
    assert len(relay.calls) == 61
    host = json.loads((suite / "host_execution.json").read_text())
    decision = json.loads((suite / "cost_decision.json").read_text())
    for name in ["retry_policy", "retry_policy_sha256", "retry_helper_sha256"]:
        assert host[name] == decision[name]


def test_controller_stops_after_five_invalid_attempts_without_dropping_game(suite):
    relay = FakeRelay("No verdict")
    with pytest.raises(RuntimeError):
        continuation.load_judge_run(suite).run(lambda:relay, ["u0"], workers=1, budget_cny=20)
    def runner(s, label, command):
        if label == "pilot_retry":
            report = continuation.retry_module(s).retry_games(s, [("base", "u0", 0)],
                budget_cny=20, relay_factory=lambda:relay)
            return {"returncode": 0 if report["complete"] else 2, "stdout": '{}'}
        assert label == "validate_generation"
        return {"returncode": 0, "stdout": '{}'}
    with pytest.raises(RuntimeError, match="exhausted"):
        continuation.continue_suite(suite, runner=runner)
    assert len(relay.calls) == 5
    assert not (suite / "cost_decision.json").exists()


def test_full_driver_resumes_after_structural_retry_without_repeating_pilot(suite):
    relay = make_pilot(suite)
    labels = []
    def runner(s, label, command):
        labels.append(label)
        if label == "full_judging" and labels.count(label) == 1:
            relay.text = "No verdict"
            with pytest.raises(RuntimeError):
                continuation.load_judge_run(s).run(lambda: relay, ["u5"], workers=1, budget_cny=180)
            return {"returncode": 1, "stdout": '{}'}
        if label == "full_judging_retry":
            relay.text = "[[A=B]]"
            result = continuation.retry_module(s).retry_games(s, [("base", "u5", 0)],
                budget_cny=180, relay_factory=lambda: relay)
            return {"returncode": 0 if result["complete"] else 2, "stdout": '{}'}
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    continuation.continue_suite(suite, runner=runner)
    assert labels.count("full_judging") == 2
    assert labels.count("full_judging_retry") == 1
    assert "pilot" not in labels
    assert len(relay.calls) == 62
    assert json.loads((suite / "cost_decision.json").read_text())["pilot_cost_cny"] == pytest.approx(.6)
