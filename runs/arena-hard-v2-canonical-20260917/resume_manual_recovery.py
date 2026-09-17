#!/usr/bin/env python3
"""Resume the fixed Arena controller with one explicitly bound verifier override.

The original controller, judge, retry helper, verifier, and cost decision remain
unchanged. This wrapper never performs the manual transport retry itself. Every
child command is still recorded by the original controller's run_command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SUITE = Path(__file__).resolve().parent
CONTROLLER_SHA256 = "fe65bbb1d5f0ef5d6d34909e56272e9e42b56c97394191e977b5dc2dc2a80167"
ORIGINAL_VERIFIER_SHA256 = "dfc8fead1bf760a98c11058f74360de9874b46d82b4fa4d659b8590d990bb5bc"
FROZEN_JUDGE_SHA256 = "0f59fd78c0dd285fc7026c23290485adc45bd093b5a3788fcf5743abd06a4c0a"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_controller(suite):
    expected = {"continue_evaluation.py": CONTROLLER_SHA256,
                "verify_final_scoring.py": ORIGINAL_VERIFIER_SHA256,
                "source/scripts/judge_arena_hard.py": FROZEN_JUDGE_SHA256}
    for name, digest in expected.items():
        if file_hash(suite / name) != digest:
            raise RuntimeError(f"Original source SHA mismatch: {name}")
    spec = importlib.util.spec_from_file_location("unchanged_arena_controller", suite / "continue_evaluation.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    return controller


def bind_recovery(suite, approval, controller):
    if approval != suite / "manual_transport_recovery.json":
        raise ValueError("Recovery requires this suite's fixed approval artifact")
    authorization = read_json(approval)
    if not isinstance(authorization, dict) or not authorization:
        raise ValueError("Missing explicit manual transport recovery approval")
    original_command = controller.commands(suite)["verify_final"]
    actual_command = [controller.CPU_PYTHON, str(suite / "verify_manual_recovery.py"),
                      "--approval", str(approval)]
    identity = {"controller_sha256": CONTROLLER_SHA256,
                "original_verifier_sha256": ORIGINAL_VERIFIER_SHA256,
                "frozen_judge_sha256": FROZEN_JUDGE_SHA256,
                "approval_sha256": file_hash(approval), "wrapper_sha256": file_hash(__file__),
                "recovery_verifier_sha256": file_hash(suite / "verify_manual_recovery.py"),
                "command_override": {"label": "verify_final", "original_command": original_command,
                                     "actual_command": actual_command}}
    host_path = suite / "host_execution.json"
    host_bytes = host_path.read_bytes()
    host = json.loads(host_bytes)
    old = host.get("recovery_execution")
    if old is not None and old.get("identity") != identity:
        raise RuntimeError("Manual recovery execution identity/binding changed")
    backup = suite / "job/transport_incident/host_execution.json"
    expected_before = authorization.get("host_execution_before_sha256")
    if not isinstance(expected_before, str) or len(expected_before) != 64:
        raise RuntimeError("Approval is missing the original host execution hash")
    if not backup.is_file() or file_hash(backup) != expected_before:
        raise RuntimeError("Original host execution backup hash mismatch")
    if old is None and hashlib.sha256(host_bytes).hexdigest() != expected_before:
        raise RuntimeError("Original host execution hash mismatch before recovery binding")
    expected_cost = authorization.get("cost_decision_sha256")
    if any(file_hash(path) != expected_cost for path in
           (suite / "cost_decision.json", suite / "job/transport_incident/cost_decision.json")):
        raise RuntimeError("Original cost decision hash mismatch")
    if old is None:
        host["recovery_execution"] = {"identity": identity, "created_at": controller.now(),
            "verification_note": "Final verification uses verify_manual_recovery.py; the original strict verifier is not claimed to pass."}
        controller.save(suite, host_path, host)
    return identity


def resume(suite=SUITE, *, approval=None, runner=None):
    suite = Path(suite).resolve()
    approval = Path(approval or suite / "manual_transport_recovery.json").resolve()
    controller = load_controller(suite)
    judge = controller.judge_module(suite)
    # The original controller retains its original continuation singleton lock.
    with judge.exclusive_lock(suite / "job/manual-recovery-lock"):
        identity = bind_recovery(suite, approval, controller)
        state_path = suite / "manual_recovery_state.json"
        state = {"state": "running", "started_at": controller.now(), "pid": os.getpid(),
                 "recovery_identity_sha256": controller.object_hash(identity),
                 "actual_final_verifier": str(suite / "verify_manual_recovery.py")}
        controller.save(suite, state_path, state)
        dispatch = runner or controller.run_command
        def reviewed_runner(current_suite, label, command):
            # Check the additive artifacts again before every possible dispatch.
            if bind_recovery(suite, approval, controller) != identity:
                raise RuntimeError("Recovery binding changed before command dispatch")
            if label == "verify_final":
                override = identity["command_override"]
                if command != override["original_command"]:
                    raise RuntimeError("Unexpected original verifier command")
                command = override["actual_command"]
            return dispatch(current_suite, label, command)
        try:
            result = controller.continue_suite(suite, runner=reviewed_runner)
            state.update(state="evaluation_complete", finished_at=controller.now(),
                         evaluation_complete_sha256=file_hash(suite / "evaluation_complete.json"))
            controller.save(suite, state_path, state)
            return result
        except BaseException as error:
            state.update(state="failed", finished_at=controller.now(),
                         error_type=type(error).__name__, error=str(error))
            controller.save(suite, state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, default=SUITE / "manual_transport_recovery.json")
    args = parser.parse_args()
    print(json.dumps(resume(approval=args.approval), indent=2))


if __name__ == "__main__":
    main()
