"""Offline checks of the additive manual transport-recovery controller."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_arena_continuation import SOURCE_SUITE, VERIFIED, continuation, make_pilot, suite


def load_wrapper():
    path = SOURCE_SUITE / "resume_manual_recovery.py"
    assert path.is_file(), "Independent recovery wrapper has not been implemented"
    spec = importlib.util.spec_from_file_location("manual_arena_continuation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(suite):
    wrapper = load_wrapper()
    for name in ["continue_evaluation.py", "verify_final_scoring.py", "resume_manual_recovery.py"]:
        (suite / name).write_bytes((SOURCE_SUITE / name).read_bytes())
    (suite / "verify_manual_recovery.py").write_text("# offline verifier fixture\n")
    approval = suite / "manual_transport_recovery.json"
    approval.write_text(json.dumps({"schema": "offline explicit transport approval", "game": ["base", "u5", 1]}))
    make_pilot(suite)
    def stop_after_decision(s, label, command):
        return {"returncode": 1 if label == "full_judging" else 0, "stdout": '{}'}
    with pytest.raises(RuntimeError, match="no eligible"):
        continuation.continue_suite(suite, runner=stop_after_decision)
    incident = suite / "job/transport_incident"
    incident.mkdir()
    for name in ["host_execution.json", "cost_decision.json"]:
        (incident / name).write_bytes((suite / name).read_bytes())
    approval.write_text(json.dumps({"schema": "offline explicit transport approval",
        "game": ["base", "u5", 1],
        "host_execution_before_sha256": wrapper.file_hash(suite / "host_execution.json"),
        "cost_decision_sha256": wrapper.file_hash(suite / "cost_decision.json")}))
    return suite, approval, wrapper


def test_only_final_verifier_command_changes_and_paid_pilot_is_reused(prepared):
    suite, approval, wrapper = prepared
    host_before = json.loads((suite / "host_execution.json").read_text())
    cost_before = (suite / "cost_decision.json").read_bytes()
    calls = []
    def runner(s, label, command):
        calls.append((label, command))
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": json.dumps(dict(VERIFIED, manual_transport_recovery=True))}
    result = wrapper.resume(suite, approval=approval, runner=runner)
    assert result["state"] == "evaluation_complete"
    assert [label for label, _ in calls] == ["validate_generation", "full_judging", "verify_final"]
    planned = continuation.commands(suite)
    assert calls[0][1] == planned["validate_generation"]
    assert calls[-1][1] == [continuation.CPU_PYTHON, str(suite / "verify_manual_recovery.py"), "--approval", str(approval)]
    host = json.loads((suite / "host_execution.json").read_text())
    assert host["continuation"] == host_before["continuation"]
    assert (suite / "cost_decision.json").read_bytes() == cost_before
    override = host["recovery_execution"]["identity"]["command_override"]
    assert override["original_command"] == planned["verify_final"]
    assert override["actual_command"] == calls[-1][1]
    assert json.loads((suite / "manual_recovery_state.json").read_text())["state"] == "evaluation_complete"


@pytest.mark.parametrize("name", ["continue_evaluation.py", "verify_final_scoring.py"])
def test_original_pinned_sources_must_match_before_any_command(prepared, name):
    suite, approval, wrapper = prepared
    (suite / name).write_text("changed original source")
    with pytest.raises(RuntimeError, match="source|SHA|hash"):
        wrapper.resume(suite, approval=approval, runner=lambda *_: pytest.fail("must not dispatch"))


@pytest.mark.parametrize("changed", ["manual_transport_recovery.json", "verify_manual_recovery.py"])
def test_restart_rejects_changed_recovery_bindings(prepared, changed):
    suite, approval, wrapper = prepared
    def runner(s, label, command):
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": json.dumps(VERIFIED)}
    wrapper.resume(suite, approval=approval, runner=runner)
    (suite / changed).write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="binding|identity"):
        wrapper.resume(suite, approval=approval, runner=lambda *_: pytest.fail("must not dispatch"))


def test_recovery_verifier_failure_remains_failed(prepared):
    suite, approval, wrapper = prepared
    def runner(s, label, command):
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 1 if label == "verify_final" else 0, "stdout": '{}'}
    with pytest.raises(RuntimeError, match="verify_final"):
        wrapper.resume(suite, approval=approval, runner=runner)
    assert not (suite / "evaluation_complete.json").exists()
    assert json.loads((suite / "manual_recovery_state.json").read_text())["state"] == "failed"


def test_original_strict_numeric_gate_is_still_required(prepared):
    suite, approval, wrapper = prepared
    def runner(s, label, command):
        if label == "full_judging": (s / "scores").mkdir()
        return {"returncode": 0, "stdout": json.dumps(dict(VERIFIED, current_games=5999))}
    with pytest.raises(RuntimeError, match="strict complete coverage"):
        wrapper.resume(suite, approval=approval, runner=runner)
    assert not (suite / "evaluation_complete.json").exists()


@pytest.mark.parametrize("path", ["job/transport_incident/host_execution.json",
    "job/transport_incident/cost_decision.json", "cost_decision.json"])
def test_incident_archive_and_cost_must_match_approval_before_dispatch(prepared, path):
    suite, approval, wrapper = prepared
    (suite / path).write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="hash mismatch"):
        wrapper.resume(suite, approval=approval, runner=lambda *_: pytest.fail("must not dispatch"))
