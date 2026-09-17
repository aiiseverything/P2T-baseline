"""Independent additive exception for one reviewed two-node ConnectError chain.

The frozen verifier checks every request, custom binding, export, all 6000
games and the official numerical aggregation. Only the named retry chain is
substituted here; the other 5999 retain the strict structural-invalid policy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parent
GAME = ('lam8', '160e7f5bbfe84ce0', 1)
BASELINE = 'gpt-4o-mini-2024-07-18'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original(suite, approval):
    path = Path(suite) / 'verify_final_scoring.py'
    require(file_hash(path) == approval['source_sha256']['verify_final_scoring.py'], 'Original verifier changed')
    spec = importlib.util.spec_from_file_location('_strict_before_manual_recovery', path)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    return base


def check_scope(approval):
    require(approval.get('policy') == 'arena_manual_single_transport_recovery_v1'
            and tuple(approval.get('game', [])) == GAME and approval.get('error_type') == 'ConnectError'
            and approval.get('baseline_model') == BASELINE and approval.get('cost_decision_existed') is False
            and type(approval.get('max_additional_attempts')) is int
            and approval['max_additional_attempts'] == 1 and approval.get('dispatch_guard_cny') == 20,
            'Manual recovery scope mismatch')


def verify_manual_chain(base, record, state_dir, tag, uid, order, request, parse, approval, protocol):
    check_scope(approval)
    require((tag, uid, order) == GAME and type(order) is int, 'Unapproved manual transport game')
    base.verify_custom_binding(record, protocol)
    require(type(record.get('attempt')) is int and record['attempt'] == 1 and record.get('status') == 'valid',
            'Manual replacement must be the first and valid retry')
    require(not any(key in record for key in ('retry_helper_sha256', 'retry_policy', 'retry_policy_sha256')),
            'Manual replacement must use the unchanged frozen driver')
    old_id = approval['original_local_request_id']
    new_id = record.get('local_request_id')
    require(isinstance(old_id, str) and old_id and Path(old_id).name == old_id
            and record.get('supersedes_local_request_id') == old_id
            and isinstance(new_id, str) and new_id and new_id != old_id and Path(new_id).name == new_id,
            'Manual request IDs or predecessor changed')
    archive = Path(state_dir) / 'attempts' / tag / f'{uid}-{order}' / f'{old_id}.json'
    require(file_hash(archive) == approval['original_record_sha256'], 'Original ambiguous bytes changed')
    old = json.loads(archive.read_text())
    base.verify_custom_binding(old, protocol)
    require((old.get('tag'), old.get('uid'), old.get('order')) == GAME
            and type(old.get('order')) is int and type(old.get('attempt')) is int and old['attempt'] == 0
            and old.get('local_request_id') == old_id and old.get('status') == 'ambiguous'
            and old.get('error_type') == 'ConnectError' and 'supersedes_local_request_id' not in old,
            'Original ConnectError identity mismatch')
    require(not any(key in old for key in ('answer', 'usage', 'score', 'finish_reason', 'response_id',
                                         'response_model', 'provider_request_id')),
            'Original transport failure contains response evidence')
    require(old.get('request') == record.get('request') == request
            and old.get('request_sha256') == record.get('request_sha256') == approval['request_sha256'] == base.digest(request)
            and old.get('protocol_sha256') == record.get('protocol_sha256') == approval['protocol_sha256'],
            'Manual recovery request/protocol changed')
    require(old['started_at'] == approval['original_started_at']
            and old['finished_at'] == approval['original_finished_at'], 'Original incident timestamps changed')
    require(base.timestamp(old['started_at']) <= base.timestamp(old['finished_at'])
            <= base.timestamp(approval['declared_at']) <= base.timestamp(record.get('started_at'))
            <= base.timestamp(record.get('finished_at')), 'Manual review/retry timeline mismatch')
    game = {'score': record.get('score'), 'judgment': {'answer': record.get('answer')},
            'prompt': request['messages']}
    base.verify_game(record, game, request, tag, uid, order, parse, protocol)
    return {'archives': {str(archive): approval['original_record_sha256']}, 'request_ids': {old_id, new_id}}


def chain_dispatcher(base, original, approval):
    counts = {'manual': 0, 'ordinary': 0}
    def dispatch(record, state_dir, tag, uid, order, request, parse, binding=None, protocol=None):
        if (tag, uid, order) == GAME:
            require(counts['manual'] == 0, 'Manual exception applied more than once')
            result = verify_manual_chain(base, record, state_dir, tag, uid, order, request, parse, approval, protocol)
            counts['manual'] += 1
            return result
        result = original(record, state_dir, tag, uid, order, request, parse, binding, protocol)
        counts['ordinary'] += 1
        return result
    return dispatch, counts


def evidence_binding(base, suite, approval_path, approval):
    check_scope(approval)
    require(approval_path == suite / 'manual_transport_recovery.json', 'Unexpected review artifact path')
    paths = [approval_path, suite / 'experiment.json']
    require(file_hash(suite / 'experiment.json') == approval['experiment_sha256'], 'Original experiment changed')
    for name, expected in approval['source_sha256'].items():
        require(file_hash(suite / name) == expected, f'Recovery source mismatch: {name}')
        paths.append(suite / name)
    incident = suite / 'job/transport_incident'
    for name, field in (('ambiguous-record.json', 'original_record_sha256'),
                        ('host_execution.json', 'host_execution_before_sha256'),
                        ('billing.json', 'billing_before_sha256')):
        require(file_hash(incident / name) == approval[field], 'Original review snapshot changed')
        paths.append(incident / name)
    require(len(approval['valid_records_sha256']) == 11, 'Missing preserved pilot identities')
    for name, expected in approval['valid_records_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Invalid preserved-game path')
        require(file_hash(suite / name) == expected, 'Previously valid pilot game changed')
        paths.append(suite / name)
    host_path = suite / 'host_execution.json'
    host = json.loads(host_path.read_text())
    before = json.loads((incident / 'host_execution.json').read_text())
    require(host['continuation']['identity'] == before['continuation']['identity'], 'Original host identity changed')
    expected = {'approval_sha256': file_hash(approval_path), 'source_sha256': approval['source_sha256'],
        'host_execution_before_sha256': approval['host_execution_before_sha256'],
        'billing_before_sha256': approval['billing_before_sha256'],
        'preserved_valid_games': approval['valid_records_sha256'],
        'command_override': {'label': 'verify_final',
            'original_command': ['/root/miniconda3/envs/sml/bin/python', str(suite / 'verify_final_scoring.py')],
            'actual_command': ['/root/miniconda3/envs/sml/bin/python', str(suite / 'verify_manual_recovery.py'),
                               '--approval', str(approval_path)]}}
    require(host.get('recovery_execution', {}).get('identity') == expected, 'Recovery wrapper binding mismatch')
    billing_path = suite / 'model_judgment/gpt-4.1/state/billing.json'
    billing = json.loads(billing_path.read_text()); prior = json.loads((incident / 'billing.json').read_text())
    require(billing['usage0_cny'] == prior['usage0_cny'] and billing['start_date'] == prior['start_date'],
            'Original cumulative cost baseline changed')
    decision_path = suite / 'cost_decision.json'
    require(host['continuation']['cost_decision_sha256'] == file_hash(decision_path), 'Pilot cost decision binding changed')
    decision = json.loads(decision_path.read_text())
    require(base.timestamp(decision['decided_at']) >= base.timestamp(approval['declared_at']),
            'Pilot cost decision predates recovery review')
    paths += [host_path, billing_path, decision_path]
    return {str(path): file_hash(path) for path in paths}


def verify(suite=SUITE, approval_path=None):
    suite = Path(suite).resolve()
    approval_path = Path(approval_path).resolve() if approval_path else suite / 'manual_transport_recovery.json'
    approval = json.loads(approval_path.read_text())
    base = load_original(suite, approval)
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
            and result['retry_counts'][GAME[0]]['initial_invalid_by_order'][str(GAME[2])] >= 1,
            'Manual retry missing from attempt accounting')
    result['initial_invalid_games'] -= 1
    result['retry_counts'][GAME[0]]['initial_invalid_by_order'][str(GAME[2])] -= 1
    result.update(manual_transport_retries=1, initial_ambiguous_games=1,
        verification_protocol='arena_manual_single_transport_recovery_v1',
        original_verifier_sha256=file_hash(suite / 'verify_final_scoring.py'),
        manual_verifier_sha256=file_hash(__file__), manual_recovery_evidence_sha256=evidence,
        manual_recovery={'game': list(GAME), 'approval_sha256': file_hash(approval_path),
                         'original_record_sha256': approval['original_record_sha256'],
                         'original_local_request_id': approval['original_local_request_id'],
                         'declared_at': approval['declared_at'], 'ordinary_chain_checks': 5999})
    result['limitations'].append('One reviewed ConnectError was replaced once with the identical custom-reference request; original ambiguous bytes remain archived and possible remote duplicate billing is included.')
    result['limitations'].append('The unchanged original verifier alone rejects this transport exception; this additive verifier replaces only that two-node chain check.')
    for path, expected in evidence.items():
        require(file_hash(path) == expected, 'Recovery evidence changed during final verification')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, default=SUITE / 'manual_transport_recovery.json')
    args = parser.parse_args()
    print(json.dumps(verify(approval_path=args.approval), indent=2, allow_nan=False))
