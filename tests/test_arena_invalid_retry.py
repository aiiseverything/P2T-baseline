"""The declared structural-invalid retry policy, tested without network calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_arena_continuation import suite, continuation, FakeRelay

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "runs/arena-hard-v2-canonical-20260917"
spec = importlib.util.spec_from_file_location("arena_invalid_retry", SUITE / "retry_invalid_judgments.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


@pytest.fixture
def failed(suite):
    (suite / "retry_policy.json").write_bytes((SUITE / "retry_policy.json").read_bytes())
    (suite / "retry_invalid_judgments.py").write_bytes((SUITE / "retry_invalid_judgments.py").read_bytes())
    run = continuation.load_judge_run(suite)
    relay = FakeRelay("No official verdict")
    for explicit in [None, [("base", "u0", 0)]]:
        with pytest.raises(RuntimeError):
            run.run(lambda: relay, ["u0"], workers=1, budget_cny=20, retry_games=explicit)
    assert len(relay.calls) == 2
    return suite, run, relay


def test_third_attempt_is_identical_and_archives_raw_predecessor(failed):
    suite, run, relay = failed
    path = run.game_path("base", "u0", 0)
    previous = path.read_bytes()
    relay.text = "[[B>A]]"
    report = retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda:relay)
    current = json.loads(path.read_text())
    assert report["complete"] and current["attempt"] == 2 and current["status"] == "valid"
    assert current["request"] == json.loads(previous)["request"] == relay.calls[-1]
    assert current["supersedes_local_request_id"] == json.loads(previous)["local_request_id"]
    archive = suite / "model_judgment/gpt-4.1/state/attempts/base/u0-0" / (json.loads(previous)["local_request_id"] + ".json")
    assert archive.read_bytes() == previous
    assert current["retry_policy_sha256"] == retry.file_hash(suite / "retry_policy.json")
    assert current["retry_helper_sha256"] == retry.file_hash(SUITE / "retry_invalid_judgments.py")
    assert json.loads((suite / "host_execution.json").read_text())["retry_policy"] == current["retry_policy"]


def test_total_five_attempts_stop_explicitly_and_each_invocation_only_calls_once(failed):
    suite, run, relay = failed
    for expected in [2, 3, 4]:
        old_calls = len(relay.calls)
        report = retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda:relay)
        assert not report["complete"] and len(relay.calls) == old_calls + 1
        assert json.loads(run.game_path("base", "u0", 0).read_text())["attempt"] == expected
    with pytest.raises(RuntimeError, match="exhausted"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda:pytest.fail("must not make a sixth attempt"))
    assert len(relay.calls) == 5


@pytest.mark.parametrize("status", ["valid", "ambiguous", "inflight"])
def test_never_retries_valid_or_transport_uncertainty(failed, status):
    suite, run, relay = failed
    path = run.game_path("base", "u0", 0)
    record = json.loads(path.read_text());record["status"] = status
    if status == "valid": record.update(answer="[[A=B]]", score="A=B", finish_reason="stop")
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda:pytest.fail("no paid request"))


def test_legal_verdict_stop_with_bad_usage_is_not_retry_eligible(failed):
    suite, run, relay = failed
    path = run.game_path("base", "u0", 0)
    record = json.loads(path.read_text());record.update(answer="[[A=B]]", score="A=B", finish_reason="stop", usage={})
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="structural"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda:pytest.fail("no paid request"))


def test_usage_is_refreshed_after_invalid_attempt_too(failed):
    suite, run, relay = failed
    retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20, relay_factory=lambda:relay)
    billing = json.loads((run.directory / "state/billing.json").read_text())
    assert billing["usage0_cny"] == 100
    assert billing["spent_cny"] == pytest.approx(.03)


def test_identity_or_broken_archive_chain_refuses_before_transport(failed):
    suite, run, relay = failed
    path = run.game_path("base", "u0", 0)
    record = json.loads(path.read_text())
    record["attempt"] = 3
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError), match="chain"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda:pytest.fail("no paid request"))


def test_every_ancestor_must_be_completed_structural_invalid(failed):
    suite, run, relay = failed
    head = json.loads(run.game_path("base", "u0", 0).read_text())
    archive = run.directory / "state/attempts/base/u0-0" / (head["supersedes_local_request_id"] + ".json")
    old = json.loads(archive.read_text());old.update(status="valid", answer="[[A=B]]", score="A=B")
    archive.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="chain"):
        retry.retry_games(suite, [("base", "u0", 0)], budget_cny=20,
                          relay_factory=lambda:pytest.fail("must inspect ancestors before paying"))
