"""Independent custom-reference lifecycle checks. All paid transport is fake."""
import importlib.util
import json
from pathlib import Path
import shlex

import pytest


SOURCE_SUITE = Path(__file__).resolve().parent
BASELINE = "gpt-4o-mini-2024-07-18"
VERIFIED = {"status": "passed", "current_games": 6000, "exact_request_and_parse_checks": 6000,
            "judgments": 3000, "questions": 500, "numeric_comparisons": 42,
            "max_absolute_difference": 0.0}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


continuation = module("custom_reference_continuation", SOURCE_SUITE / "continue_evaluation.py")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "source").symlink_to(SOURCE_SUITE / "source", target_is_directory=True)
    (tmp_path / "job").mkdir()
    questions = [{"uid": f"u{i}", "category": "hard_prompt", "prompt": f"Question {i}"}
                 for i in range(500)]
    write_jsonl(tmp_path / "question.jsonl", questions)
    for tag in (*continuation.TAGS, BASELINE):
        write_jsonl(tmp_path / "model_answer" / f"{tag}.jsonl", [
            {"uid": q["uid"], "model": tag, "messages": [
                {"role": "user", "content": q["prompt"]},
                {"role": "assistant", "content": {"answer": tag + " unique answer " + q["uid"]}}]}
            for q in questions])
    for name in ("retry_policy.json", "retry_invalid_judgments.py", "run_full_judging.sh"):
        (tmp_path / name).write_bytes((SOURCE_SUITE / name).read_bytes())
    for name in ("run_evaluation.py", "verify_final_scoring.py"):
        (tmp_path / name).write_text("# offline runner fixture; never executed\n")
    (tmp_path / "pilot_uids.json").write_text(json.dumps([f"u{i}" for i in range(5)]))
    (tmp_path / "experiment.json").write_text(json.dumps({"models": {tag: {"adapter": tag}
                                                                                for tag in continuation.TAGS}}))
    (tmp_path / "status.json").write_text('{"state":"generation_complete"}')
    (tmp_path / "generation_complete.json").write_text('{"state":"generation_complete"}')
    return tmp_path


class FakeRelay:
    def __init__(self, text="[[A=B]]", error=None):
        self.text, self.error = text, error
        self.calls = []

    def usage(self, _):
        return 100 + .01 * len(self.calls)

    def judge_call(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return {"answer": self.text, "finish_reason": "stop", "response_id": "offline",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


def pilot(suite, relay=None, max_requests=0):
    relay = relay or FakeRelay()
    continuation.load_judge_run(suite).run(lambda: relay, [f"u{i}" for i in range(5)],
        workers=12, budget_cny=20, max_requests=max_requests)
    return relay


@pytest.fixture
def failed(suite):
    run = continuation.load_judge_run(suite)
    relay = FakeRelay("No structural verdict")
    for explicit in (None, [("base", "u0", 0)]):
        with pytest.raises(RuntimeError, match="blocked"):
            run.run(lambda: relay, ["u0"], workers=1, budget_cny=20, retry_games=explicit)
    assert len(relay.calls) == 2
    return suite, run, relay, continuation.retry_module(suite)


def test_pilot_and_full_commands_select_the_custom_reference(suite):
    commands = continuation.commands(suite)
    pilot_command = commands["pilot"]
    assert pilot_command[pilot_command.index("--baseline-model") + 1] == BASELINE
    assert pilot_command[pilot_command.index("--baseline") + 1] == str(suite / "model_answer" / f"{BASELINE}.jsonl")
    assert "--validate-only" in commands["validate_generation"]
    shell = (suite / "run_full_judging.sh").read_text().replace("\\\n", " ")
    lines = [shlex.split(line) for line in shell.splitlines()
             if "source/scripts/judge_arena_hard.py" in line or "source/scripts/score_arena_hard.py" in line]
    assert len(lines) == 2
    for command in lines:
        assert command[command.index("--baseline-model") + 1] == BASELINE
    assert all("o3-mini-2025-01-31.jsonl" not in " ".join(command) for command in lines)


def test_exact_custom_pilot_resume_and_saved_game_identities(suite):
    relay = pilot(suite, max_requests=12)
    assert not continuation.pilot_status(suite)["complete"]
    pilot(suite, relay=relay)
    status = continuation.pilot_status(suite)
    assert status["complete"] and status["valid_games"] == 60
    assert status["complete_pairs"] == {tag: 5 for tag in continuation.TAGS}
    assert len(relay.calls) == 60
    run = continuation.load_judge_run(suite)
    judge = continuation.judge_module(suite)
    for tag in continuation.TAGS:
        for uid in status["uids"]:
            for order in (0, 1):
                record = run.load_record(tag, uid, order)
                assert record["baseline_model"] == BASELINE
                assert record["protocol_sha256"] == judge.digest(run.protocol)
                assert BASELINE + " unique answer " + uid in record["request"]["messages"][1]["content"]
        assert all(row["baseline"] == BASELINE for row in judge.load_jsonl(run.directory / f"{tag}.jsonl"))
    pilot(suite, relay=relay)
    assert len(relay.calls) == 60


def test_custom_controller_cost_then_full_and_completed_resume(suite):
    labels = []
    def runner(s, label, command):
        labels.append(label)
        if label == "pilot":
            pilot(s)
        if label == "full_judging":
            (s / "scores").mkdir(exist_ok=True)
            (s / "scores/results.json").write_text('{}')
            assert command[:2] == ["bash", str(s / "run_full_judging.sh")]
            assert int(command[2]) == 180
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    continuation.continue_suite(suite, runner=runner)
    assert labels == ["validate_generation", "pilot", "full_judging", "verify_final"]
    decision = json.loads((suite / "cost_decision.json").read_text())
    assert decision["pilot_cost_cny"] == pytest.approx(.6)
    assert decision["estimated_full_cny"] == pytest.approx(60)
    assert decision["dispatch_budget_cny"] == 180
    assert decision["automatic_continuation"] is True
    labels.clear()
    continuation.continue_suite(suite, runner=runner,
        usage_reader=lambda _: pytest.fail("persisted cost must be reused"))
    assert labels == ["validate_generation", "verify_final"]
    assert json.loads((suite / "cost_decision.json").read_text()) == decision


def test_structural_retry_retains_custom_identity_and_stops_at_first_valid(failed):
    suite, run, relay, retry = failed
    path = run.game_path("base", "u0", 0)
    before = path.read_bytes()
    relay.text = "[[B>A]]"
    result = retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda: relay)
    assert result["complete"] and len(relay.calls) == 3
    record = continuation.load_judge_run(suite).load_record("base", "u0", 0)
    assert record["attempt"] == 2 and record["status"] == "valid"
    assert record["baseline_model"] == BASELINE
    assert record["protocol_sha256"] == json.loads(before)["protocol_sha256"]
    assert record["request"] == json.loads(before)["request"] == relay.calls[-1]
    archive = run.directory / "state/attempts/base/u0-0" / (json.loads(before)["local_request_id"] + ".json")
    assert archive.read_bytes() == before
    with pytest.raises(RuntimeError, match="structural"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda: pytest.fail("first valid verdict is final"))


def test_five_total_attempt_limit_counts_initial_and_explicit_attempts(failed):
    suite, run, relay, retry = failed
    for attempt in (2, 3, 4):
        result = retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda: relay)
        assert not result["complete"] and len(relay.calls) == attempt + 1
    with pytest.raises(RuntimeError, match="exhausted"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda: pytest.fail("no sixth attempt"))


@pytest.mark.parametrize("status", ["inflight", "ambiguous"])
def test_transport_uncertainty_stops_helper_and_controller_before_dispatch(failed, status):
    suite, run, _, retry = failed
    path = run.game_path("base", "u0", 0)
    record = json.loads(path.read_text()); record["status"] = status
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="forbidden"):
        retry.invalid_targets(suite)
    with pytest.raises(RuntimeError):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda: pytest.fail("no automatic transport retry"))
    with pytest.raises(RuntimeError, match="forbidden"):
        continuation.run_with_structural_retries(suite,
            lambda *_: pytest.fail("controller must not dispatch"), "pilot", [], 20, ["u0"])


@pytest.mark.parametrize("field,value", [("baseline_model", "o3-mini-2025-01-31"),
                                         ("protocol_sha256", "wrong-protocol")])
def test_every_retry_archive_binds_the_custom_baseline(failed, field, value):
    suite, run, _, retry = failed
    head = json.loads(run.game_path("base", "u0", 0).read_text())
    archive = run.directory / "state/attempts/base/u0-0" / (head["supersedes_local_request_id"] + ".json")
    old = json.loads(archive.read_text()); old[field] = value
    archive.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="identity"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda: pytest.fail("inspect archive identities before paying"))


def test_retry_transport_error_is_ambiguous_and_cannot_be_retried(failed):
    suite, run, relay, retry = failed
    relay.error = TimeoutError("offline synthetic timeout")
    with pytest.raises(RuntimeError, match="ambiguous"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda: relay)
    assert len(relay.calls) == 3
    assert continuation.load_judge_run(suite).load_record("base", "u0", 0)["status"] == "ambiguous"
    with pytest.raises(RuntimeError, match="forbidden"):
        retry.invalid_targets(suite)


def test_controller_structural_retry_selects_missing_work_without_repaying_valid_games(failed):
    suite, run, relay, retry = failed
    (suite / "host_execution.json").write_text(json.dumps({"continuation": {"commands": {}}}))
    labels = []
    relay.text = "[[A=B]]"
    def runner(s, label, command):
        labels.append(label)
        if label == "pilot_retry":
            assert command[command.index("--game") + 1] == "base:u0:0"
            report = retry.retry_games(s, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda: relay)
            return {"returncode": 0 if report["complete"] else 2, "stdout": "{}"}
        fresh = continuation.load_judge_run(s)
        fresh.run(lambda: relay, ["u0"], workers=12, budget_cny=20)
        return {"returncode": 0, "stdout": "{}"}
    continuation.run_with_structural_retries(suite, runner, "pilot", [], 20, ["u0"])
    assert labels == ["pilot_retry", "pilot"]
    # Two failed initial attempts + one successful replacement + 11 untouched games.
    assert len(relay.calls) == 14
    run = continuation.load_judge_run(suite)
    before = run.game_path("base", "u0", 0).read_bytes()
    continuation.run_with_structural_retries(suite, runner, "pilot", [], 20, ["u0"])
    assert len(relay.calls) == 14 and run.game_path("base", "u0", 0).read_bytes() == before
