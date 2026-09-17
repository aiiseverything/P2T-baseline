"""Read-only final QA with one explicitly approved transport-recovery exception.

The original verifier stays byte-identical. Its request/parse/coverage and
official score computations are reused. Only the named two-node retry chain
is checked here; every other chain uses the original strict function.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parent
CPU_PYTHON = '/root/miniconda3/envs/sml/bin/python'
GAME = ('base', '34fd667185674f47', 1)
ORIGINAL_VERIFIER_SHA = 'dfc8fead1bf760a98c11058f74360de9874b46d82b4fa4d659b8590d990bb5bc'
ORIGINAL_CONTROLLER_SHA = 'fe65bbb1d5f0ef5d6d34909e56272e9e42b56c97394191e977b5dc2dc2a80167'
FROZEN_JUDGE_SHA = '0f59fd78c0dd285fc7026c23290485adc45bd093b5a3788fcf5743abd06a4c0a'
RETRY_HELPER_SHA = 'e9eab7189d1d93b1143a4367cdc0050f3d9124e3b3696ade620423ee99a05f4d'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_original(suite):
    import hashlib
    path = Path(suite) / 'verify_final_scoring.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_VERIFIER_SHA,
            'Original verifier source changed')
    spec = importlib.util.spec_from_file_location('_arena_verifier_before_manual_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_approval_scope(approval):
    require(approval.get('policy') == 'arena_manual_single_transport_recovery_v1'
            and (approval.get('tag'), approval.get('uid'), approval.get('order')) == GAME
            and type(approval.get('order')) is int
            and type(approval.get('original_attempt')) is int and approval['original_attempt'] == 0
            and type(approval.get('max_additional_attempts')) is int and approval['max_additional_attempts'] == 1
            and approval.get('error_type') == 'RemoteProtocolError', 'Manual approval scope mismatch')


def request_bytes(request):
    return json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def verify_manual_chain(base, record, state_dir, tag, uid, order, request, parse, approval):
    check_approval_scope(approval)
    require((tag, uid, order) == GAME and type(order) is int, 'Unapproved manual transport game')
    require(type(record.get('attempt')) is int and record['attempt'] == 1
            and record.get('status') == 'valid', 'Manual recovery must be the first and valid retry')
    require(not any(key in record for key in ('retry_helper_sha256', 'retry_policy', 'retry_policy_sha256')),
            'Manual recovery must use the unchanged frozen CLI')
    old_id = approval['original_local_request_id']
    require(isinstance(old_id, str) and old_id and Path(old_id).name == old_id
            and record.get('supersedes_local_request_id') == old_id, 'Manual predecessor ID mismatch')
    new_id = record.get('local_request_id')
    require(isinstance(new_id, str) and new_id and Path(new_id).name == new_id and new_id != old_id,
            'Manual retry request ID mismatch')
    archive = Path(state_dir) / 'attempts' / tag / f'{uid}-{order}' / f'{old_id}.json'
    require(base.file_hash(archive) == approval['original_record_sha256'], 'Original ambiguous bytes changed')
    old = json.loads(archive.read_text())
    require((old.get('tag'), old.get('uid'), old.get('order')) == GAME and type(old.get('order')) is int
            and type(old.get('attempt')) is int and old['attempt'] == 0
            and old.get('local_request_id') == old_id
            and old.get('status') == 'ambiguous' and old.get('error_type') == 'RemoteProtocolError'
            and 'supersedes_local_request_id' not in old, 'Original ambiguous identity mismatch')
    require(not any(key in old for key in ('answer', 'usage', 'score', 'finish_reason',
                                         'response_id', 'response_model', 'provider_request_id')),
            'Original transport failure contains response evidence; manual exception does not apply')
    expected_sha = base.digest(request)
    require(approval['request_sha256'] == expected_sha
            and old.get('request_sha256') == record.get('request_sha256') == expected_sha
            and request_bytes(old.get('request')) == request_bytes(record.get('request')) == request_bytes(request),
            'Manual recovery request changed')
    require(old.get('started_at') == approval['original_started_at']
            and old.get('finished_at') == approval['original_finished_at'], 'Original incident timestamps changed')
    original_start, original_end = base.timestamp(old['started_at']), base.timestamp(old['finished_at'])
    declaration = base.timestamp(approval['declared_at'])
    retry_start, retry_end = base.timestamp(record.get('started_at')), base.timestamp(record.get('finished_at'))
    require(original_start <= original_end <= declaration <= retry_start <= retry_end,
            'Manual approval/retry timeline mismatch')
    game = {'score': record.get('score'), 'judgment': {'answer': record.get('answer')},
            'prompt': request['messages']}
    base.verify_game(record, game, request, tag, uid, order, parse)
    return {'archives': {str(archive): approval['original_record_sha256']}, 'request_ids': {old_id, new_id}}


def chain_dispatcher(base, original, approval):
    counts = {'manual': 0, 'ordinary': 0}
    def dispatch(record, state_dir, tag, uid, order, request, parse, binding=None):
        if (tag, uid, order) == GAME:
            require(counts['manual'] == 0, 'Manual exception applied more than once')
            result = verify_manual_chain(base, record, state_dir, tag, uid, order, request, parse, approval)
            counts['manual'] += 1
            return result
        result = original(record, state_dir, tag, uid, order, request, parse, binding)
        counts['ordinary'] += 1
        return result
    return dispatch, counts


def evidence_binding(base, suite, approval_path, approval):
    check_approval_scope(approval)
    source_hashes = {
        'verify_final_scoring.py': (ORIGINAL_VERIFIER_SHA, 'original_verifier_sha256'),
        'continue_evaluation.py': (ORIGINAL_CONTROLLER_SHA, 'original_controller_sha256'),
        'source/scripts/judge_arena_hard.py': (FROZEN_JUDGE_SHA, 'frozen_judge_sha256'),
        'retry_invalid_judgments.py': (RETRY_HELPER_SHA, 'retry_helper_sha256')}
    evidence = {str(approval_path): base.file_hash(approval_path)}
    for name, (expected, field) in source_hashes.items():
        path = suite / name
        actual = base.file_hash(path)
        require(actual == expected == approval.get(field), f'Manual approval source mismatch: {name}')
        evidence[str(path)] = actual
    before_path = suite / 'job/transport_incident/host_execution.json'
    require(base.file_hash(before_path) == approval['host_execution_before_sha256'], 'Original host snapshot changed')
    for path in (suite / 'cost_decision.json', suite / 'job/transport_incident/cost_decision.json'):
        require(base.file_hash(path) == approval['cost_decision_sha256'], 'Manual recovery cost decision changed')
        evidence[str(path)] = base.file_hash(path)
    evidence[str(before_path)] = base.file_hash(before_path)
    host_path = suite / 'host_execution.json'
    host = json.loads(host_path.read_text())
    before = json.loads(before_path.read_text())
    require(host['continuation']['identity'] == before['continuation']['identity'],
            'Original continuation identity changed during manual recovery')
    original_command = [CPU_PYTHON, str(suite / 'verify_final_scoring.py')]
    actual_command = [CPU_PYTHON, str(suite / 'verify_manual_recovery.py'), '--approval', str(approval_path)]
    wrapper_path = suite / 'resume_manual_recovery.py'
    expected_identity = {
        'controller_sha256': ORIGINAL_CONTROLLER_SHA, 'original_verifier_sha256': ORIGINAL_VERIFIER_SHA,
        'frozen_judge_sha256': FROZEN_JUDGE_SHA, 'approval_sha256': base.file_hash(approval_path),
        'wrapper_sha256': base.file_hash(wrapper_path), 'recovery_verifier_sha256': base.file_hash(__file__),
        'command_override': {'label': 'verify_final', 'original_command': original_command, 'actual_command': actual_command}}
    require(host.get('recovery_execution', {}).get('identity') == expected_identity,
            'Manual recovery execution binding mismatch')
    for path in (host_path, wrapper_path, Path(__file__)):
        evidence[str(path)] = base.file_hash(path)
    return evidence


def verify(suite=SUITE, approval_path=None):
    suite = Path(suite).resolve()
    approval_path = Path(approval_path).resolve() if approval_path else suite / 'manual_transport_recovery.json'
    require(approval_path == suite / 'manual_transport_recovery.json', 'Unexpected manual approval path')
    base = load_original(suite)
    approval = json.loads(approval_path.read_text())
    evidence = evidence_binding(base, suite, approval_path, approval)
    original = base.verify_retry_chain
    dispatch, counts = chain_dispatcher(base, original, approval)
    base.verify_retry_chain = dispatch
    try:
        result = base.verify(suite)
    finally:
        base.verify_retry_chain = original
    require(counts == {'manual': 1, 'ordinary': 5999}, 'Manual exception coverage mismatch')
    require(result['initial_invalid_games'] >= 1
            and result['retry_counts']['base']['initial_invalid_by_order']['1'] >= 1,
            'Missing manual retry in original attempt accounting')
    result['initial_invalid_games'] -= 1
    result['retry_counts']['base']['initial_invalid_by_order']['1'] -= 1
    result.update(manual_transport_retries=1, initial_ambiguous_games=1,
                  verification_protocol='arena_manual_single_transport_recovery_v1',
                  original_verifier_sha256=ORIGINAL_VERIFIER_SHA,
                  manual_verifier_sha256=base.file_hash(__file__),
                  manual_recovery_evidence_sha256=evidence,
                  manual_recovery={'game': list(GAME), 'original_record_sha256': approval['original_record_sha256'],
                                   'original_local_request_id': approval['original_local_request_id'],
                                   'approval_sha256': base.file_hash(approval_path),
                                   'declared_at': approval['declared_at'], 'ordinary_chain_checks': 5999})
    result['limitations'].append('One RemoteProtocolError attempt was explicitly recovered once with the identical request; original ambiguous bytes remain archived. Possible duplicate remote billing is included in the cumulative account usage.')
    result['limitations'].append('The original verifier is unchanged; this supplemental verifier overrides only the approved two-attempt transport chain. Its original structural-invalid-only chain rule alone would reject that exception.')
    for path, expected in evidence.items():
        require(base.file_hash(path) == expected, f'Manual recovery evidence changed during verification: {path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, default=SUITE / 'manual_transport_recovery.json')
    args = parser.parse_args()
    print(json.dumps(verify(SUITE, args.approval), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
