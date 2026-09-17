"""Additive, read-only verification of explicitly bounded Arena recovery.

All frozen 6000-request and 42 numerical checks still execute. This module
extends only the retry-chain dispatcher and audits the separately declared
recovery policy; it never sends API requests or modifies the frozen suite.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SUITE = Path(__file__).resolve().parent
POLICY = 'arena_bounded_connection_recovery_v1'
GAME = ('lam8', '160e7f5bbfe84ce0', 1)
RESPONSE_FIELDS = ('answer', 'usage', 'score', 'finish_reason', 'response_id',
                   'response_model', 'provider_request_id')
REQUIRED_SOURCES = ('continue_resilient.py', 'retry_resilient_judgments.py',
                    'resilient_judge.py', 'verify_resilient_scoring.py',
                    'source/scripts/judge_arena_hard.py', 'source/scripts/score_arena_hard.py',
                    'verify_final_scoring.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_path(suite, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts, 'Invalid relative evidence path')
    return Path(suite) / name


def failure_kind(base, record, parse):
    if record.get('status') == 'invalid':
        require(isinstance(record.get('answer'), str)
                and record.get('finish_reason') in ('stop', 'length'),
                'Only completed structural-invalid responses may be retried')
        parsed = parse(record['answer'], base.PATTERNS)
        accepted = parsed if parsed in base.SCORES else None
        require(record.get('score') == accepted, 'Archived structural verdict mismatch')
        require(record['finish_reason'] == 'length' or accepted is None,
                'A valid verdict was retried; first valid must stop')
        return 'structural'
    require(record.get('status') == 'ambiguous'
            and record.get('error_type') in ('ConnectError', 'ConnectTimeout')
            and not any(key in record for key in RESPONSE_FIELDS),
            'Uncertain transport or response-bearing attempt cannot be retried')
    return 'connection'


def verify_resilient_chain(base, record, state_dir, tag, uid, order, request, parse, binding, protocol):
    """Verify every predecessor and the exact authorization for each transition."""
    policy = binding['policy']
    require(policy.get('policy') == POLICY and type(policy.get('max_total_attempts_per_game')) is int
            and policy['max_total_attempts_per_game'] == 5, 'Unexpected bounded retry policy')
    require(record.get('status') == 'valid', 'Final recovered game must be valid')
    attempt = record.get('attempt')
    require(type(attempt) is int and 0 < attempt < 5, 'Invalid attempt counter or MAX5 exceeded')
    declared = base.timestamp(policy['declared_at'])
    node, archives, ids = record, {}, set()
    counts = {'connect_failures': 0, 'structural_invalid_attempts': 0, 'initial_failure_kind': None}
    while True:
        base.verify_custom_binding(node, protocol)
        require(type(node.get('attempt')) is int and node['attempt'] == attempt
                and node.get('tag') == tag and node.get('uid') == uid
                and type(node.get('order')) is int and node['order'] == order
                and node.get('request') == request and node.get('request_sha256') == base.digest(request),
                'Resilient archive/request binding mismatch')
        local_id = node.get('local_request_id')
        require(isinstance(local_id, str) and local_id and Path(local_id).name == local_id
                and local_id not in ids, 'Duplicate or invalid resilient request ID')
        ids.add(local_id)
        started, finished = base.timestamp(node.get('started_at')), base.timestamp(node.get('finished_at'))
        require(started <= finished, 'Invalid resilient attempt timeline')
        tagged = any(k in node for k in ('resilient_policy_sha256', 'resilient_helper_sha256'))
        if attempt > 0 or tagged:
            require(node.get('resilient_policy_sha256') == binding['policy_sha256']
                    and node.get('resilient_helper_sha256') == binding['helper_sha256'],
                    'Resilient retry helper/policy binding mismatch')
            require(started >= declared, 'Resilient retry preceded policy declaration')
        if attempt == 0:
            require('supersedes_local_request_id' not in node, 'Initial attempt has predecessor')
            break
        prior_id = node.get('supersedes_local_request_id')
        require(isinstance(prior_id, str) and prior_id and Path(prior_id).name == prior_id
                and prior_id not in ids, 'Invalid predecessor or retry cycle')
        archive = Path(state_dir) / 'attempts' / tag / f'{uid}-{order}' / f'{prior_id}.json'
        require(archive.is_file(), 'Missing resilient predecessor archive')
        previous = read(archive)
        require(previous.get('local_request_id') == prior_id, 'Archived filename/request ID mismatch')
        previous_started = base.timestamp(previous.get('started_at'))
        previous_finished = base.timestamp(previous.get('finished_at'))
        require(previous_started <= previous_finished <= started, 'Retry started before predecessor finished')
        kind = failure_kind(base, previous, parse)
        if kind == 'connection':
            if previous_started < declared:
                item = policy.get('reviewed_existing_failures', {}).get(f'{tag}:{uid}:{order}', {})
                expected_path = f'model_judgment/gpt-4.1/state/games/{tag}/{uid}-{order}.json'
                require(previous.get('attempt') == 0 and item.get('path') == expected_path
                        and item.get('sha256') == file_hash(archive),
                        'Pre-policy connection failure lacks exact reviewed byte whitelist')
            counts['connect_failures'] += 1
        else:
            counts['structural_invalid_attempts'] += 1
        if previous.get('attempt') == 0:
            counts['initial_failure_kind'] = kind
        archives[str(archive)] = file_hash(archive)
        node, attempt = previous, attempt - 1
    return {'archives': archives, 'request_ids': ids, **counts}


def load_binding(suite, policy_path=None, *, require_execution=True):
    suite = Path(suite).resolve()
    path = Path(policy_path).resolve() if policy_path else suite / 'resilient_policy.json'
    require(path == suite / 'resilient_policy.json', 'Unexpected resilient policy path')
    policy = read(path)
    require(policy.get('policy') == POLICY and type(policy.get('max_total_attempts_per_game')) is int
            and policy['max_total_attempts_per_game'] == 5
            and type(policy.get('transport_max_connect_attempts')) is int
            and policy['transport_max_connect_attempts'] == 4
            and policy.get('allowed_transport_errors') == ['ConnectError', 'ConnectTimeout'],
            'Unexpected resilient policy scope or attempt limits')
    sources = policy.get('source_sha256', {})
    require(set(REQUIRED_SOURCES).issubset(sources), 'Missing resilient source bindings')
    evidence = {str(path): file_hash(path)}
    for name, expected in sources.items():
        p = local_path(suite, name)
        require(file_hash(p) == expected, f'Resilient source changed: {name}')
        evidence[str(p)] = expected
    base = load('_strict_before_resilient_recovery', suite / 'verify_final_scoring.py')
    base.timestamp(policy.get('declared_at'))
    approval_path = suite / 'manual_transport_recovery.json'
    require(file_hash(approval_path) == policy.get('legacy_manual_approval_sha256'), 'Legacy manual approval changed')
    approval = read(approval_path)
    manual = load('_legacy_manual_chain_only', suite / 'verify_manual_recovery.py')
    manual.check_scope(approval)
    require(file_hash(suite / 'experiment.json') == policy.get('original_experiment_sha256')
            == approval.get('experiment_sha256'), 'Original experiment changed')
    require(file_hash(suite / 'host_execution.json') == policy.get('original_host_execution_sha256'),
            'Original host execution changed')
    for name, expected in approval['source_sha256'].items():
        p = local_path(suite, name)
        require(file_hash(p) == expected, f'Legacy recovery source changed: {name}')
        evidence[str(p)] = expected
    preserved = policy.get('preserved_valid_records_sha256', {})
    require(len(preserved) == 22, 'Expected all22 preserved valid games')
    for name, expected in preserved.items():
        p = local_path(suite, name)
        require(p.parent.parent == suite / 'model_judgment/gpt-4.1/state/games'
                and read(p).get('status') == 'valid' and file_hash(p) == expected,
                'Previously valid game changed')
        evidence[str(p)] = expected
    require(all(preserved.get(name) == expected for name, expected in approval['valid_records_sha256'].items()),
            'Legacy eleven valid games missing from preservation list')
    for name, field in (('ambiguous-record.json', 'original_record_sha256'),
                        ('host_execution.json', 'host_execution_before_sha256'),
                        ('billing.json', 'billing_before_sha256')):
        p = suite / 'job/transport_incident' / name
        require(file_hash(p) == approval[field], 'Legacy manual incident evidence changed')
        evidence[str(p)] = approval[field]
    billing_path = suite / 'model_judgment/gpt-4.1/state/billing.json'
    billing, origin = read(billing_path), policy.get('billing_origin', {})
    require(set(origin) == {'start_date', 'usage0_cny'}
            and all(billing.get(k) == v for k, v in origin.items()), 'Cumulative billing origin changed')
    prior_billing = read(suite / 'job/transport_incident/billing.json')
    require(all(prior_billing.get(k) == v for k, v in origin.items()), 'Legacy billing origin changed')
    require(type(origin['usage0_cny']) in (int, float) and math.isfinite(origin['usage0_cny'])
            and origin['usage0_cny'] >= 0, 'Invalid cumulative billing origin')
    for p in (suite / 'experiment.json', suite / 'host_execution.json', approval_path, billing_path):
        evidence[str(p)] = file_hash(p)
    identity = {key: policy[key] for key in ('source_sha256', 'original_experiment_sha256',
                'legacy_manual_approval_sha256', 'billing_origin', 'original_host_execution_sha256')}
    identity['resilient_policy_sha256'] = file_hash(path)
    execution = None
    if require_execution:
        execution_path = suite / 'resilient_execution.json'
        execution = read(execution_path)
        require(execution.get('identity') == identity, 'Resilient execution identity changed')
        evidence[str(execution_path)] = file_hash(execution_path)
    return {'suite': suite, 'policy': policy, 'policy_sha256': file_hash(path),
            'helper_sha256': sources['retry_resilient_judgments.py'],
            'base': base, 'manual': manual, 'approval': approval, 'evidence': evidence,
            'identity': identity, 'execution': execution}


def verify_cost_decision(base, binding):
    suite = binding['suite']; path = suite / 'resilient_cost_decision.json'
    execution, policy = binding['execution'], binding['policy']
    require(execution.get('cost_decision_sha256') == file_hash(path), 'Resilient cost decision binding changed')
    decision = read(path)
    require(decision.get('resilient_policy_sha256') == binding['policy_sha256'], 'Cost decision policy mismatch')
    cost = decision.get('pilot_cost_cny')
    require(type(cost) in (int, float) and math.isfinite(cost) and cost > 0, 'Invalid pilot cost')
    estimate = Decimal(str(cost)) * 100
    budget = int(max(Decimal(30), estimate * 3).to_integral_value(rounding=ROUND_CEILING))
    expected = {'pilot_cost_cny': float(cost), 'pilot_games': 60, 'full_games': 6000,
        'estimated_full_cny': float(estimate), 'dispatch_budget_cny': budget,
        'automatic_continuation': estimate <= 300 and budget <= 900,
        'formula': 'estimate=pilot_cost*100; budget=ceil(max(30,3*estimate))',
        'budget_scope': 'Cumulative same-directory relay usage, including pilot; dispatch guard, not a hard cap'}
    require(expected['automatic_continuation'] and all(decision.get(k) == v for k, v in expected.items()),
            'Resilient cost decision formula/guard mismatch')
    require(base.timestamp(decision.get('decided_at')) >= base.timestamp(policy['declared_at']),
            'Resilient cost decision predates policy')
    uids = read(suite / 'pilot_uids.json')
    require(isinstance(uids, list) and len(uids) == len(set(uids)) == 5, 'Wrong frozen pilot scope')
    records = []
    for tag in base.TAGS:
        for uid in uids:
            for order in (0, 1):
                p = suite / 'model_judgment/gpt-4.1/state/games' / tag / f'{uid}-{order}.json'
                require(read(p).get('status') == 'valid', 'Pilot contains nonvalid game')
                records.append({'tag': tag, 'uid': uid, 'order': order, 'sha256': file_hash(p)})
    require(base.digest(records) == decision.get('pilot_records_sha256'), 'Pilot decision/game bytes changed')
    return {str(path): file_hash(path)}


def chain_dispatcher(binding):
    base, manual, approval = binding['base'], binding['manual'], binding['approval']
    original = base.verify_retry_chain
    counts = {'manual': 0, 'ordinary': 0, 'resilient': 0,
              'connect_failures': 0, 'structural_invalid_attempts': 0}
    initial_connections = []
    def dispatch(record, state_dir, tag, uid, order, request, parse, old_binding=None, protocol=None):
        if (tag, uid, order) == GAME:
            require(counts['manual'] == 0, 'Legacy manual exception applied twice')
            require(not any(k in record for k in ('resilient_policy_sha256', 'resilient_helper_sha256')),
                    'Legacy valid manual game was retried')
            result = manual.verify_manual_chain(base, record, state_dir, tag, uid, order,
                                               request, parse, approval, protocol)
            counts['manual'] += 1
            initial_connections.append((tag, uid, order))
        elif any(k in record for k in ('resilient_policy_sha256', 'resilient_helper_sha256')):
            result = verify_resilient_chain(base, record, state_dir, tag, uid, order,
                                           request, parse, binding, protocol)
            counts['resilient'] += 1
            for key in ('connect_failures', 'structural_invalid_attempts'):
                counts[key] += result[key]
            if result['initial_failure_kind'] == 'connection':initial_connections.append((tag, uid, order))
        else:
            result = original(record, state_dir, tag, uid, order, request, parse, old_binding, protocol)
            counts['ordinary'] += 1
        return result
    return dispatch, counts, initial_connections


def verify_transport_audit(base, binding):
    """Bind every post-policy logical record to a complete audited HTTP call."""
    suite, policy = binding['suite'], binding['policy']
    declared = base.timestamp(policy['declared_at'])
    phases = {'connection.connect_tcp.failed', 'connection.start_tls.failed', 'proxy.start_tls.failed'}
    evidence, runs = {}, {}
    for path in (suite / 'job/transport_bindings').glob('*.json'):
        row = read(path)
        require(row.get('schema') == 'arena_hard_connection_transport_v1'
                and row.get('run_id') == path.stem and path.stem not in runs,
                'Invalid or duplicate transport binding')
        require(row.get('resilient_policy_sha256') == binding['policy_sha256']
                and row.get('transport_source_sha256') == policy['source_sha256']['resilient_judge.py']
                and row.get('frozen_judge_sha256') == policy['source_sha256']['source/scripts/judge_arena_hard.py'],
                'Transport source/policy binding mismatch')
        require(type(row.get('transport_max_connect_attempts')) is int
                and row['transport_max_connect_attempts'] == 4
                and row.get('backoff_seconds') == [1., 2., 4.]
                and row.get('retry_exception_types') == ['httpx.ConnectError', 'httpx.ConnectTimeout']
                and row.get('eligible_failure_phases') == sorted(phases)
                and row.get('require_no_application_headers_started') is True,
                'Transport retry limits or eligibility changed')
        require(row.get('max_connections') == row.get('max_keepalive_connections') == 32
                and row.get('keepalive_expiry_seconds') == 300
                and row.get('endpoint') == 'https://api.linkapi.ai/v1'
                and base.timestamp(row.get('created_at')) >= declared,
                'Transport pool/endpoint/declaration mismatch')
        runs[path.stem] = row
        evidence[str(path)] = file_hash(path)
    path = suite / 'job/connect_attempts.jsonl'
    require(path.is_file() and runs, 'Missing resilient transport audit/bindings')
    audits = defaultdict(list)
    for row in base.read_jsonl(path):
        require(row.get('schema') == 'arena_hard_connection_transport_v1'
                and row.get('method') == 'POST' and row.get('run_id') in runs
                and isinstance(row.get('audit_id'), str) and row['audit_id'], 'Unknown transport audit identity')
        audits[row['audit_id']].append(row)
    evidence[str(path)] = file_hash(path)
    state = suite / 'model_judgment/gpt-4.1/state'
    records = defaultdict(list)
    expected_records = set()
    for p in [*(state / 'games').glob('*/*.json'), *(state / 'attempts').glob('*/*/*.json')]:
        record = read(p)
        if base.timestamp(record.get('started_at')) >= declared:
            records[record['request_sha256']].append((p, record))
            expected_records.add(p)
    used, logical, physical, internal_retries = set(), 0, 0, 0
    for audit_id, rows in sorted(audits.items(), key=lambda kv: kv[1][0].get('timestamp', '')):
        require(len(rows) in (2, 4, 6, 8), 'Incomplete or excessive connection attempt trace')
        first, final = rows[0], rows[-1]
        last_time = base.timestamp(runs[first['run_id']]['created_at'])
        for index in range(len(rows) // 2):
            start, finish = rows[index * 2:index * 2 + 2]
            attempt = index + 1
            require(start.get('event') == 'attempt_started' and finish.get('event') == 'attempt_finished'
                    and type(start.get('transport_attempt')) is int
                    and type(finish.get('transport_attempt')) is int
                    and start['transport_attempt'] == finish['transport_attempt'] == attempt,
                    'Missing, reordered or discontinuous transport attempts')
            require(all(row.get(k) == first.get(k) for row in (start, finish)
                        for k in ('schema', 'run_id', 'audit_id', 'request_sha256', 'method')),
                    'Transport attempt request identity changed')
            begin, end = base.timestamp(start.get('timestamp')), base.timestamp(finish.get('timestamp'))
            require(last_time <= begin <= end, 'Transport attempt timeline mismatch')
            last_time = end
            events = finish.get('events')
            require(isinstance(events, list) and all(isinstance(e, dict) and isinstance(e.get('event'), str)
                                                    for e in events), 'Invalid connection trace events')
            app_started = any(e['event'].endswith('.send_request_headers.started')
                              and e.get('method') != 'CONNECT' for e in events)
            failure_phases = [e['event'] for e in events if e['event'] in phases]
            failure_phase = failure_phases[-1] if failure_phases else None
            require(finish.get('application_headers_started') is app_started
                    and finish.get('connection_failure_phase') == failure_phase,
                    'Connection trace safety flags mismatch')
            if finish.get('outcome') == 'response':
                require(type(finish.get('status_code')) is int and 100 <= finish['status_code'] <= 599,
                        'Invalid transport HTTP status')
                eligible = False
            else:
                require(finish.get('outcome') == 'exception', 'Unknown connection attempt outcome')
                classes = finish.get('exception_classes')
                require(isinstance(classes, list) and classes and all(isinstance(x, str) for x in classes),
                        'Missing transport exception classes')
                eligible = classes[0] in ('httpx.ConnectError', 'httpx.ConnectTimeout') \
                    and failure_phase in phases and not app_started
                expected_error = ('UncertainTransportError' if classes[0] in ('httpx.ConnectError', 'httpx.ConnectTimeout')
                                  and not eligible else classes[0].rsplit('.', 1)[-1])
                require(finish.get('raised_error_type') == expected_error,
                        'Unproven connection error was not marked uncertain')
            will_retry = eligible and attempt < 4
            require(finish.get('retry_eligible') is eligible and finish.get('will_retry') is will_retry
                    and finish.get('backoff_seconds') == ((1., 2., 4.)[attempt - 1] if will_retry else 0),
                    'Transport retry decision or backoff mismatch')
            require(will_retry == (index < len(rows) // 2 - 1), 'Transport continued after terminal response/error')
            physical += 1
            internal_retries += int(will_retry)
        request_sha = first['request_sha256']
        candidates = []
        for p, record in records.get(request_sha, []):
            if p in used:continue
            response = record.get('status') in ('valid', 'invalid')
            outcome_matches = (final['outcome'] == 'response' and 200 <= final['status_code'] < 300) if response else (
                final['outcome'] == 'exception' and final.get('raised_error_type') == record.get('error_type'))
            if (outcome_matches and base.timestamp(record['started_at']) <= base.timestamp(first['timestamp'])
                    <= base.timestamp(final['timestamp']) <= base.timestamp(record['finished_at'])):
                candidates.append((p, record))
        require(candidates, 'Transport audit has no matching durable logical request')
        matched = min(candidates, key=lambda item: base.timestamp(item[1]['finished_at']))[0]
        used.add(matched); logical += 1
    require(used == expected_records and logical == len(expected_records),
            'Some post-policy logical requests lack complete transport audit')
    return {'logical_requests_after_policy': logical, 'transport_attempts_after_policy': physical,
            'connection_establishment_retries': internal_retries, 'transport_bindings': len(runs),
            'evidence': evidence}


def verify(suite=SUITE, policy_path=None):
    binding = load_binding(suite, policy_path)
    base = binding['base']
    evidence = {**binding['evidence'], **verify_cost_decision(base, binding)}
    transport = verify_transport_audit(base, binding)
    evidence.update(transport.pop('evidence'))
    dispatch, counts, initial_connections = chain_dispatcher(binding)
    original = base.verify_retry_chain
    base.verify_retry_chain = dispatch
    try:
        result = base.verify(binding['suite'])
    finally:
        base.verify_retry_chain = original
    require(counts['manual'] == 1 and sum(counts[k] for k in ('manual', 'ordinary', 'resilient')) == 6000,
            'Resilient retry dispatcher coverage mismatch')
    for tag, uid, order in initial_connections:
        result['initial_invalid_games'] -= 1
        result['retry_counts'][tag]['initial_invalid_by_order'][str(order)] -= 1
    require(result['initial_invalid_games'] >= 0, 'Retry classification accounting underflow')
    result.update(verification_protocol=POLICY, resilient_policy_sha256=binding['policy_sha256'],
        resilient_verifier_sha256=file_hash(__file__), original_verifier_sha256=file_hash(binding['suite']/'verify_final_scoring.py'),
        manual_transport_retries=1, initial_ambiguous_games=len(initial_connections),
        resilient_retry_checks=counts, resilient_evidence_sha256=evidence,
        transport_audit=transport,
        total_recorded_transport_attempts=result['total_recorded_api_attempts'] + transport['connection_establishment_retries'])
    result['limitations'].append('Bounded connection failures with no response evidence may be replaced under the frozen additive policy; all logical attempts remain recorded and share the original cumulative billing origin.')
    for path, expected in evidence.items():
        require(file_hash(path) == expected, f'Resilient evidence changed during verification: {path}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.suite, args.policy), indent=2, allow_nan=False))
