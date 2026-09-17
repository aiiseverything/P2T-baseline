"""Additive, reviewed recovery of exactly one ConnectError during the pilot.

No frozen input is changed. ``prepare`` snapshots evidence without network.
``retry`` permits one byte-identical replacement through the frozen driver;
it never retries a failed replacement or any of the eleven valid games.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

SUITE = Path(__file__).resolve().parent
GAME = ('lam8', '160e7f5bbfe84ce0', 1)
BASELINE = 'gpt-4o-mini-2024-07-18'
POLICY = 'arena_manual_single_transport_recovery_v1'
BUDGET = 20
RESPONSE_FIELDS = ('answer', 'usage', 'score', 'finish_reason', 'response_id',
                   'response_model', 'provider_request_id')
SOURCE_NAMES = ('continue_evaluation.py', 'verify_final_scoring.py', 'retry_invalid_judgments.py',
                'source/scripts/judge_arena_hard.py', 'source/scripts/score_arena_hard.py',
                'manual_transport_recovery.py', 'resume_manual_recovery.py', 'verify_manual_recovery.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def controller(suite):
    return load('_manual_original_controller', Path(suite) / 'continue_evaluation.py')


def _snapshot(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f'Recovery snapshot changed: {path.name}')
        return
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _scope(record):
    require((record.get('tag'), record.get('uid'), record.get('order')) == GAME
            and type(record.get('order')) is int and type(record.get('attempt')) is int
            and record['attempt'] == 0 and record.get('status') == 'ambiguous'
            and record.get('error_type') == 'ConnectError'
            and record.get('baseline_model') == BASELINE
            and 'supersedes_local_request_id' not in record
            and not any(key in record for key in RESPONSE_FIELDS),
            'Manual scope requires this initial ConnectError without response evidence')


def validate_review(suite):
    """Read-only evidence check, also valid after the original controller resumes."""
    suite = Path(suite).resolve()
    approval = read_json(suite / 'manual_transport_recovery.json')
    require(approval.get('policy') == POLICY and tuple(approval.get('game', [])) == GAME
            and approval.get('error_type') == 'ConnectError' and approval.get('baseline_model') == BASELINE
            and type(approval.get('max_additional_attempts')) is int
            and approval['max_additional_attempts'] == 1
            and approval.get('cost_decision_existed') is False and approval.get('dispatch_guard_cny') == BUDGET,
            'Manual recovery approval scope changed')
    require(file_hash(suite / 'experiment.json') == approval['experiment_sha256'], 'Frozen experiment changed')
    experiment = read_json(suite / 'experiment.json')
    for name, expected in experiment['files_sha256'].items():
        require(file_hash(suite / name) == expected, f'Frozen input changed: {name}')
    require(set(approval['source_sha256']) == set(SOURCE_NAMES), 'Recovery source binding incomplete')
    for name, expected in approval['source_sha256'].items():
        require(file_hash(suite / name) == expected, f'Recovery source changed: {name}')
    incident = suite / 'job/transport_incident'
    for name, field in (('ambiguous-record.json', 'original_record_sha256'),
                        ('host_execution.json', 'host_execution_before_sha256'),
                        ('billing.json', 'billing_before_sha256')):
        require(file_hash(incident / name) == approval[field], f'Original recovery evidence changed: {name}')
    old = read_json(incident / 'ambiguous-record.json')
    _scope(old)
    require(old['request_sha256'] == approval['request_sha256']
            and old['protocol_sha256'] == approval['protocol_sha256']
            and old['local_request_id'] == approval['original_local_request_id']
            and old['started_at'] == approval['original_started_at']
            and old['finished_at'] == approval['original_finished_at'], 'Original incident binding changed')
    valid = approval['valid_records_sha256']
    require(len(valid) == 11, 'Expected eleven preserved valid pilot games')
    for name, expected in valid.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and name.startswith('model_judgment/gpt-4.1/state/games/'), 'Invalid preserved-game path')
        require(file_hash(suite / name) == expected and read_json(suite / name)['status'] == 'valid',
                'A previously valid pilot game changed')
    prior_host = read_json(incident / 'host_execution.json')
    host = read_json(suite / 'host_execution.json')
    require(host['continuation']['identity'] == prior_host['continuation']['identity'],
            'Original continuation identity changed')
    before = read_json(incident / 'billing.json')
    billing = read_json(suite / 'model_judgment/gpt-4.1/state/billing.json')
    require(billing['start_date'] == before['start_date'] and billing['usage0_cny'] == before['usage0_cny'],
            'Original cumulative billing baseline changed')
    values = [billing.get(k) for k in ('usage0_cny', 'latest_usage_cny', 'spent_cny')]
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values)
            and billing['latest_usage_cny'] >= before['latest_usage_cny']
            and math.isclose(billing['spent_cny'], billing['latest_usage_cny'] - billing['usage0_cny'],
                             rel_tol=0, abs_tol=1e-6), 'Recovery billing is inconsistent')
    return approval


def prepare(suite=SUITE, *, reviewed_by):
    suite = Path(suite).resolve()
    require(isinstance(reviewed_by, str) and reviewed_by.strip(), 'A named operator review is required')
    ctrl = controller(suite); judge = ctrl.judge_module(suite)
    with ExitStack() as locks:
        for directory in (suite / 'job/manual-recovery-lock', suite / 'job/continuation-lock',
                          suite / 'model_judgment/gpt-4.1'):
            locks.enter_context(judge.exclusive_lock(directory))
        approval_path = suite / 'manual_transport_recovery.json'
        if approval_path.exists():
            return validate_review(suite)
        require(not (suite / 'cost_decision.json').exists(), 'This review is restricted to the interrupted pilot')
        validator = load('_manual_reuse_validator', suite / 'run_evaluation.py')
        validator.verify_reused_generation(suite)
        run = ctrl.load_judge_run(suite)
        record = run.load_record(*GAME); _scope(record)
        paths = list((run.directory / 'state/games').glob('*/*.json'))
        expected = {(tag, GAME[1], order) for tag in ctrl.TAGS for order in (0, 1)}
        require(len(paths) == 12 and {(read_json(p)['tag'], read_json(p)['uid'], read_json(p)['order'])
                                     for p in paths} == expected, 'Expected only the first twelve pilot attempts')
        require(not list((run.directory / 'state/attempts').rglob('*.json')), 'Existing retry archives are outside this review')
        valid = {}
        for path in paths:
            row = read_json(path)
            if (row['tag'], row['uid'], row['order']) == GAME:
                continue
            checked = run.load_record(row['tag'], row['uid'], row['order'])
            require(checked['status'] == 'valid', 'Another unresolved game is outside this review')
            valid[str(path.relative_to(suite))] = file_hash(path)
        incident = suite / 'job/transport_incident'
        for original, target in ((run.game_path(*GAME), 'ambiguous-record.json'),
                                  (suite / 'host_execution.json', 'host_execution.json'),
                                  (run.directory / 'state/billing.json', 'billing.json')):
            _snapshot(incident / target, original.read_bytes())
        approval = {'policy': POLICY, 'game': list(GAME), 'error_type': 'ConnectError',
            'baseline_model': BASELINE, 'protocol_sha256': judge.digest(run.protocol),
            'original_record_sha256': file_hash(run.game_path(*GAME)),
            'original_local_request_id': record['local_request_id'], 'request_sha256': record['request_sha256'],
            'original_started_at': record['started_at'], 'original_finished_at': record['finished_at'],
            'declared_at': datetime.now(timezone.utc).isoformat(), 'reviewed_by': reviewed_by,
            'max_additional_attempts': 1, 'dispatch_guard_cny': BUDGET, 'cost_decision_existed': False,
            'host_execution_before_sha256': file_hash(incident / 'host_execution.json'),
            'billing_before_sha256': file_hash(incident / 'billing.json'), 'valid_records_sha256': valid,
            'experiment_sha256': file_hash(suite / 'experiment.json'),
            'source_sha256': {name: file_hash(suite / name) for name in SOURCE_NAMES},
            'authorization_basis': 'Existing user request to complete this evaluation; specific operator review of one transport failure',
            'billing_note': 'No local response exists; remote billing is unconfirmed. Any duplicate charge remains within the unchanged cumulative pilot dispatch guard.',
            'request_invariance': 'Same frozen request, baseline and protocol; eleven valid games remain byte-identical; no automatic transport retry policy change'}
        judge.atomic_json(approval_path, approval)
        validate_review(suite)
        return approval


def retry_once(suite=SUITE, *, relay_factory=None):
    suite = Path(suite).resolve()
    ctrl = controller(suite); judge = ctrl.judge_module(suite)
    with ExitStack() as locks:
        for directory in (suite / 'job/manual-recovery-lock', suite / 'job/continuation-lock'):
            locks.enter_context(judge.exclusive_lock(directory))
        approval = validate_review(suite)
        run = ctrl.load_judge_run(suite)
        current = run.load_record(*GAME)
        if current is not None and current.get('attempt') == 1 and current.get('status') == 'valid':
            # Only the approved completed replacement is idempotently accepted.
            verifier = load('_manual_completion_chain', suite / 'verify_manual_recovery.py')
            base = verifier.load_original(suite, approval)
            verifier.verify_manual_chain(base, current, run.directory / 'state', *GAME,
                run.request(*GAME), lambda text, patterns: judge.parse_score(text, patterns), approval, run.protocol)
            return {'status': 'already_recovered', 'game': list(GAME), 'new_requests': 0}
        if current is None or current.get('attempt') != 0:
            raise RuntimeError('The single additional manual attempt has already been used; recovery remains blocked')
        require(file_hash(run.game_path(*GAME)) == approval['original_record_sha256'], 'Original ambiguous record changed')
        _scope(current)
        require(file_hash(suite / 'host_execution.json') == approval['host_execution_before_sha256'],
                'Host changed before the reviewed retry')
        require(not (suite / 'cost_decision.json').exists(), 'Pilot cost decision appeared before recovery')
        if relay_factory is None:
            def relay_factory():
                key = os.environ.get('LINKAPI_KEY') or Path('/root/.linkapi_key').read_text().strip()
                return judge.Relay(key)
        # With eleven valid same-UID peers, only GAME is pending. max_requests=1
        # independently caps paid dispatch even if future driver code differs.
        summary = run.run(relay_factory, [GAME[1]], workers=1, budget_cny=BUDGET,
                          max_requests=1, retry_games=[GAME])
        validate_review(suite)
        current = ctrl.load_judge_run(suite).load_record(*GAME)
        require(current['attempt'] == 1 and current['status'] == 'valid', 'Single replacement is not valid')
        verifier = load('_manual_result_chain', suite / 'verify_manual_recovery.py')
        base = verifier.load_original(suite, approval)
        verifier.verify_manual_chain(base, current, run.directory / 'state', *GAME,
            run.request(*GAME), lambda text, patterns: judge.parse_score(text, patterns), approval, run.protocol)
        result = {'status': 'recovered', 'game': list(GAME), 'new_requests': summary['dispatched_this_run'],
                  'approval_sha256': file_hash(suite / 'manual_transport_recovery.json'),
                  'record_sha256': file_hash(run.game_path(*GAME))}
        judge.atomic_json(suite / 'manual_recovery_attempt.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'retry'))
    parser.add_argument('--reviewed-by')
    args = parser.parse_args()
    result = prepare(reviewed_by=args.reviewed_by) if args.action == 'prepare' else retry_once()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
