"""Resume the unchanged controller after the one reviewed pilot replacement.

The only command override is the additive final verifier. All other commands,
cost decisions, request limits, source hashes and retry policy stay unchanged.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parent


def helper(suite):
    path = Path(suite) / 'manual_transport_recovery.py'
    spec = importlib.util.spec_from_file_location('_reviewed_manual_helper', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def bind_recovery(suite, api, ctrl):
    approval = api.validate_review(suite)
    run = ctrl.load_judge_run(suite)
    record = run.load_record(*api.GAME)
    api.require(record is not None and record.get('attempt') == 1 and record.get('status') == 'valid',
                'Resume requires the single valid manual replacement')
    verifier = api.load('_reviewed_manual_verifier', suite / 'verify_manual_recovery.py')
    base = verifier.load_original(suite, approval)
    judge = ctrl.judge_module(suite)
    verifier.verify_manual_chain(base, record, run.directory / 'state', *api.GAME,
        run.request(*api.GAME), lambda text, patterns: judge.parse_score(text, patterns), approval, run.protocol)
    identity = {'approval_sha256': api.file_hash(suite / 'manual_transport_recovery.json'),
                'source_sha256': approval['source_sha256'],
                'host_execution_before_sha256': approval['host_execution_before_sha256'],
                'billing_before_sha256': approval['billing_before_sha256'],
                'preserved_valid_games': approval['valid_records_sha256'],
                'command_override': {'label': 'verify_final',
                    'original_command': ctrl.commands(suite)['verify_final'],
                    'actual_command': [ctrl.CPU_PYTHON, str(suite / 'verify_manual_recovery.py'),
                                       '--approval', str(suite / 'manual_transport_recovery.json')]}}
    path = suite / 'host_execution.json'
    host = api.read_json(path)
    existing = host.get('recovery_execution')
    if existing is None:
        api.require(api.file_hash(path) == approval['host_execution_before_sha256']
                    and not (suite / 'cost_decision.json').exists(), 'Unexpected changes before pilot recovery binding')
        host['recovery_execution'] = {'identity': identity, 'created_at': ctrl.now(),
            'verification_note': 'Only the reviewed ConnectError chain uses the additive verifier; original strict verifier remains unchanged.'}
        ctrl.save(suite, path, host)
    else:
        api.require(existing.get('identity') == identity, 'Manual recovery execution binding changed')
    return identity


def resume(suite=SUITE, *, runner=None):
    suite = Path(suite).resolve()
    api = helper(suite); approval = api.validate_review(suite)
    ctrl = api.controller(suite); judge = ctrl.judge_module(suite)
    with judge.exclusive_lock(suite / 'job/manual-recovery-lock'):
        identity = bind_recovery(suite, api, ctrl)
        state = {'state': 'running', 'started_at': ctrl.now(),
                 'recovery_identity_sha256': ctrl.object_hash(identity)}
        ctrl.save(suite, suite / 'manual_recovery_state.json', state)
        dispatch = runner or ctrl.run_command
        def reviewed_runner(current_suite, label, command):
            api.require(bind_recovery(suite, api, ctrl) == identity, 'Recovery changed before dispatch')
            if label == 'verify_final':
                override = identity['command_override']
                api.require(command == override['original_command'], 'Unexpected original verifier command')
                command = override['actual_command']
            return dispatch(current_suite, label, command)
        try:
            result = ctrl.continue_suite(suite, runner=reviewed_runner)
            state.update(state='evaluation_complete', finished_at=ctrl.now(),
                         evaluation_complete_sha256=api.file_hash(suite / 'evaluation_complete.json'))
            ctrl.save(suite, suite / 'manual_recovery_state.json', state)
            return result
        except BaseException as error:
            state.update(state='failed', error_type=type(error).__name__, finished_at=ctrl.now())
            ctrl.save(suite, suite / 'manual_recovery_state.json', state)
            raise


if __name__ == '__main__':
    print(json.dumps(resume(), indent=2))
