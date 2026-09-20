#!/usr/bin/env python3
"""Bounded identical GPT-4o retries, then one model-only GPT-4.1 fallback.

Preparation and validation are read-only with respect to the completed source
suite. All new request intents, responses, transport audits and billing updates
belong to a separate campaign. Uncertain delivery and corrupt state fail closed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import arena_exclusion_policy as policy
from scripts import run_arena_with_exclusions as original

SCHEMA = 'arena_identical_retry5_gpt41_fallback_v1'
BASELINE = 'gpt-4o-mini-2024-07-18'
BASE_USAGE = 103.817238
LAST_USAGE = 150.024436
BUDGET = 144.0
POLICY = dict(gpt4o_total_attempts=5, gpt41_fallback_attempts=1,
              stop_at_first_valid=True, max_concurrent_targets=32,
              billing_max_attempts=3, dispatch_guard_cny=BUDGET,
              usage0_cny=BASE_USAGE, usage_start_date='2026-09-01')


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def immutable_json(path, value):
    immutable_bytes(path, encoded(value))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(uuid.uuid4()) + '.tmp')
    immutable_bytes(temp, encoded(value))
    os.replace(temp, path)
    sync_dir(path.parent)


@contextmanager
def controller_lock(campaign):
    with (campaign / '.controller.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another controller holds the campaign lock') from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_suite(suite):
    # Existing loaders execute only definitions and reconstruct requests. Avoid
    # importlib bytecode writes anywhere in the immutable source suite.
    before = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return original.load_suite(Path(suite))
    finally:
        sys.dont_write_bytecode = before


def valid_family(returned, requested):
    return isinstance(returned, str) and re.fullmatch(
        re.escape(requested) + r'(?:-\d{4}-\d{2}-\d{2})?', returned) is not None


def item_of(row):
    return row['tag'], row['uid'], row['order']


def target_dir(campaign, target):
    return campaign / 'attempts' / target['tag'] / f"{target['uid']}-{target['order']}"


def billing_valid(value):
    require(isinstance(value, dict), 'Corrupt saved billing')
    baseline, latest, spent = (value.get(k) for k in ('usage0_cny', 'latest_usage_cny', 'spent_cny'))
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                for x in (baseline, latest, spent))
            and baseline == BASE_USAGE and latest >= LAST_USAGE
            and math.isclose(spent, latest-baseline, rel_tol=0, abs_tol=1e-6)
            and value.get('start_date') == POLICY['usage_start_date']
            and value.get('dispatch_guard_cny') == BUDGET, 'Corrupt saved billing identity')
    return value


def verify_files(files):
    for name, expected in files.items():
        require(Path(name).is_file() and not Path(name).is_symlink()
                and file_hash(name) == expected, f'Bound input changed: {name}')


def prepare_campaign(source_suite, campaign, *, exclusion_manifest=None, expected_targets=590):
    """Freeze exactly the failed original-attempt targets; make no network calls.

    ``expected_targets`` is explicit for fixture suites; the CLI always requires
    the authorized 590. Existing directories may contain frozen source code,
    but never an existing manifest, billing or attempt state.
    """
    suite, campaign = Path(source_suite).resolve(), Path(campaign).resolve()
    require(suite != campaign and suite not in campaign.parents,
            'Campaign must be outside the source suite')
    require(type(expected_targets) is int and expected_targets > 0, 'Invalid expected target count')
    if campaign.exists():
        require(not any((campaign / p).exists() for p in
            ('manifest.json', 'manifest.sha256', 'state', 'attempts', 'originals', 'dispatches', 'complete.json')),
            'Campaign state already exists; preparation is immutable')
    exclusion = Path(exclusion_manifest).resolve() if exclusion_manifest else (
        suite / 'continuation-with-exclusions-v2/scores/exclusions.jsonl')
    results_path = exclusion.parent / 'results.json'
    prior = read_json(results_path)
    source_files = dict(prior['inputs_sha256'])
    verify_files(source_files)
    run, judge = load_suite(suite)
    expected_games = {run.game_path(tag, uid, order) for tag in original.TAGS
                      for uid in run.questions for order in (0, 1)}
    require(set((run.directory / 'state/games').rglob('*.json')) == expected_games,
            'Source suite game coverage changed')
    require(read_json(run.directory / 'state/protocol.json') == run.identity,
            'Source protocol identity mismatch')
    for tag in original.TAGS:
        require(read_json(run.directory / 'state/models' / f'{tag}.json') == dict(
            answers_sha256=judge.digest(list(run.answers[tag].values())), tag=tag, model=tag),
            'Source model identity mismatch')
    rows = [json.loads(line) for line in exclusion.read_text().splitlines() if line.strip()]
    failed = {}
    for row in rows:
        if row['record_classification'] == 'judge_failed':
            item = row['model'], row['uid'], row['order']
            require(item not in failed, 'Duplicate source exclusion')
            failed[item] = row
    targets, games = [], {}
    for path in sorted(expected_games):
        raw = read_json(path)
        item = item_of(raw)
        require(type(item[2]) is int, 'Invalid original answer order')
        record = run.load_record(*item)
        state = policy.classify_record(record, judge)
        require(state in ('valid', 'judge_failed'), 'Source has blocked or unfinished games')
        sha = file_hash(path)
        games[str(path.resolve())] = sha
        require(source_files.get(str(path.resolve())) == sha, 'Original game missing from prior scoring hashes')
        if state == 'judge_failed':
            require(item in failed and failed[item]['game_sha256'] == sha,
                    'Original failure does not match exclusion manifest')
            count = record.get('attempt', 0)
            require(type(count) is int and count in (0, 4), 'Unexpected preexisting attempt count')
            if count == 0:
                require(record.get('supersedes_local_request_id') is None,
                        'Original first attempt unexpectedly has predecessor')
                require(not list((run.directory / 'state/attempts' / item[0] /
                                  f'{item[1]}-{item[2]}').glob('*.json')),
                        'Original first attempt has historical attempts')
                targets.append(dict(tag=item[0], uid=item[1], order=item[2],
                    original_path=str(path.resolve()), original_sha256=sha,
                    request_sha256=judge.digest(run.request(*item))))
    require(set(failed) == {item_of(read_json(name)) for name in games
        if policy.classify_record(read_json(name), judge) == 'judge_failed'},
        'Source failure manifest coverage mismatch')
    require(len(targets) == expected_targets, 'Exact retry target count mismatch')
    targets.sort(key=item_of)
    billing_path = run.directory / 'state/billing.json'
    billing = billing_valid(read_json(billing_path))
    essential = [exclusion, results_path, billing_path, suite / 'continue_evaluation.py',
        suite / 'resilient_judge.py', suite / 'resilient_policy.json',
        suite / 'source/scripts/judge_arena_hard.py', suite / 'question.jsonl',
        run.directory / 'state/protocol.json']
    essential += list((run.directory / 'state/models').glob('*.json'))
    essential += list((run.directory / 'state/attempts').rglob('*.json'))
    essential += [suite / 'model_answer' / f'{tag}.jsonl' for tag in (*original.TAGS, BASELINE)]
    essential += [suite / 'source/third_party/arena_hard' / name for name in run.protocol['source_sha256']]
    source_files.update({str(p.resolve()): file_hash(p) for p in essential})
    execution_files = {str(Path(p).resolve()): file_hash(p)
                       for p in (__file__, original.__file__, policy.__file__)}
    manifest = dict(schema=SCHEMA, created_at=now(), source_suite=str(suite),
        policy=POLICY, target_count=len(targets), targets=targets, source_files_sha256=source_files,
        source_game_files_sha256=games, execution_files_sha256=execution_files,
        source_exclusion_manifest=str(exclusion), source_exclusion_sha256=file_hash(exclusion),
        source_results_path=str(results_path), source_results_sha256=file_hash(results_path),
        protocol=run.protocol, protocol_sha256=judge.digest(run.protocol), baseline=BASELINE,
        original_billing=billing, frozen_judge_sha256=file_hash(suite / 'source/scripts/judge_arena_hard.py'))
    # Recheck bound inputs before creating any campaign state.
    verify_files(source_files)
    campaign.mkdir(parents=True, exist_ok=True)
    with controller_lock(campaign):
        require(not (campaign / 'manifest.json').exists(), 'Campaign was concurrently prepared')
        for target in targets:
            destination = campaign / 'originals' / target['tag'] / f"{target['uid']}-{target['order']}" / 'original.json'
            immutable_bytes(destination, Path(target['original_path']).read_bytes())
        immutable_json(campaign / 'manifest.json', manifest)
        immutable_bytes(campaign / 'manifest.sha256', (file_hash(campaign / 'manifest.json')+'\n').encode())
        immutable_json(campaign / 'state/billing.json', billing)
    return manifest


def _load_context(campaign):
    campaign = Path(campaign).resolve()
    sha = file_hash(campaign / 'manifest.json')
    require((campaign / 'manifest.sha256').read_text().strip() == sha, 'Campaign manifest hash mismatch')
    manifest = read_json(campaign / 'manifest.json')
    require(manifest.get('schema') == SCHEMA and manifest.get('policy') == POLICY,
            'Campaign retry policy changed')
    require(type(manifest.get('target_count')) is int and manifest['target_count'] > 0
            and len(manifest['targets']) == manifest['target_count'], 'Invalid target count')
    verify_files(manifest['source_files_sha256'])
    verify_files(manifest['execution_files_sha256'])
    suite = Path(manifest['source_suite'])
    require(suite.is_absolute() and suite != campaign and suite not in campaign.parents,
            'Invalid source suite binding')
    run, judge = load_suite(suite)
    require(manifest['baseline'] == BASELINE and manifest['protocol'] == run.protocol
            and manifest['protocol_sha256'] == judge.digest(run.protocol), 'Protocol binding changed')
    expected_games = {str(run.game_path(tag, uid, order).resolve()) for tag in original.TAGS
                      for uid in run.questions for order in (0, 1)}
    require(set(manifest['source_game_files_sha256']) == expected_games
            and {str(p.resolve()) for p in (run.directory / 'state/games').rglob('*.json')} == expected_games,
            'Source game coverage changed')
    require(all(manifest['source_files_sha256'].get(path) == sha
                for path, sha in manifest['source_game_files_sha256'].items()), 'Source game hashes are unbound')
    require(manifest['source_files_sha256'].get(manifest['source_exclusion_manifest'])
            == manifest['source_exclusion_sha256']
            and manifest['source_files_sha256'].get(manifest['source_results_path'])
            == manifest['source_results_sha256'], 'Source results are not bound')
    # Reconstruct the authorized target set independently of its saved list.
    # A missing target must not silently reduce the completed denominator.
    source_failed = [json.loads(line) for line in
        Path(manifest['source_exclusion_manifest']).read_text().splitlines() if line.strip()]
    exact_targets = set()
    for exclusion in source_failed:
        if exclusion['record_classification'] != 'judge_failed':
            continue
        item = exclusion['model'], exclusion['uid'], exclusion['order']
        source_record = run.load_record(*item)
        require(source_record is not None and policy.classify_record(source_record, judge) == 'judge_failed'
                and exclusion['game_sha256'] == manifest['source_game_files_sha256'][
                    str(run.game_path(*item).resolve())], 'Source failed target binding changed')
        if source_record.get('attempt', 0) == 0:
            require(item not in exact_targets, 'Duplicate target in source exclusions')
            exact_targets.add(item)
    require({item_of(target) for target in manifest['targets']} == exact_targets,
            'Exact authorized target set changed')
    seen, original_paths = set(), set()
    for target in manifest['targets']:
        item = item_of(target)
        require(item[0] in original.TAGS and item[1] in run.questions and type(item[2]) is int
                and item[2] in (0, 1) and item not in seen, 'Invalid or duplicate target')
        seen.add(item)
        require(target['original_path'] == str(run.game_path(*item).resolve())
                and target['original_sha256'] == manifest['source_game_files_sha256'][target['original_path']]
                and target['request_sha256'] == judge.digest(run.request(*item)), 'Target identity changed')
        path = campaign / 'originals' / item[0] / f'{item[1]}-{item[2]}' / 'original.json'
        original_paths.add(path)
        require(file_hash(path) == target['original_sha256'], 'Original snapshot changed')
        record = run.load_record(*item)
        require(type(record.get('attempt', 0)) is int and record.get('attempt', 0) == 0
                and policy.classify_record(record, judge) == 'judge_failed', 'Target is not an original failed attempt')
    require(set((campaign / 'originals').rglob('*.*')) == original_paths,
            'Extra or missing original snapshots')
    billing_valid(read_json(campaign / 'state/billing.json'))
    manifest['manifest_sha256'] = sha
    return campaign, manifest, run, judge


def _resolutions(context, *, inspected_ambiguity=None):
    campaign, manifest, run, judge = context
    resolutions, expected_attempt_files, expected_dispatches, files = [], set(), set(), {}
    recovery = None
    if (campaign / 'transport_recovery').exists():
        from scripts import arena_retry_transport_recovery as recovery_helper
        recovery = recovery_helper.load_binding(context)
        files.update(recovery['_files'])
    for target in manifest['targets']:
        item = item_of(target)
        path = campaign / 'originals' / item[0] / f'{item[1]}-{item[2]}' / 'original.json'
        record = read_json(path)
        previous_sha = file_hash(path)
        seen_ids = {record['local_request_id']}
        evidence = [dict(total_attempt=1, judge_model='gpt-4o', record_path=str(path),
            record_sha256=previous_sha, status=record['status'], local_request_id=record['local_request_id'],
            request_sha256=record['request_sha256'], response_model=record.get('response_model'))]
        files[str(path)] = previous_sha
        resolution, selected, recovered = 'pending', None, None
        folder = target_dir(campaign, target)
        dirs = sorted(folder.iterdir()) if folder.exists() else []
        require(all(p.is_dir() and p.name in ('02', '03', '04', '05', '06') for p in dirs),
                'Unknown attempt directory')
        require([p.name for p in dirs] == [f'{n:02d}' for n in range(2, len(dirs)+2)],
                'Discontinuous attempt history')
        for number, directory in enumerate(dirs, 2):
            require(selected is None and resolution not in ('failed', 'blocked'), 'Attempt after terminal judgment')
            inflight_path, response_path, hash_path = (directory / name for name in
                                                       ('inflight.json', 'record.json', 'record.sha256'))
            expected_attempt_files.update((inflight_path, response_path, hash_path))
            require(all(p.is_file() and not p.is_symlink() for p in (inflight_path, response_path, hash_path)),
                    'Unresolved inflight attempt; automatic resend blocked')
            intent, response = read_json(inflight_path), read_json(response_path)
            current_sha = file_hash(response_path)
            require(hash_path.read_text().strip() == current_sha, 'Completed attempt hash mismatch')
            model = 'gpt-4.1' if number == 6 else 'gpt-4o'
            request = {**run.request(*item), 'model': model}
            expected = dict(tag=item[0], uid=item[1], order=item[2], total_attempt=number,
                attempt=number-1, judge_model=model, baseline_model=BASELINE,
                protocol_sha256=manifest['protocol_sha256'], request=request,
                request_sha256=judge.digest(request), campaign_manifest_sha256=manifest['manifest_sha256'],
                predecessor_record_sha256=previous_sha, supersedes_local_request_id=record['local_request_id'])
            require(all(intent.get(key) == value and response.get(key) == value
                        for key, value in expected.items()), 'Attempt request or predecessor identity changed')
            require(type(response.get('order')) is int and type(response.get('total_attempt')) is int
                    and type(response.get('attempt')) is int, 'Invalid attempt counter')
            require(intent.get('status') == 'inflight' and isinstance(intent.get('started_at'), str)
                    and response.get('started_at') == intent['started_at'], 'Invalid inflight identity')
            request_id = intent.get('local_request_id')
            require(isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', request_id)
                    and request_id not in seen_ids and response.get('local_request_id') == request_id,
                    'Duplicate or missing local request ID')
            seen_ids.add(request_id)
            dispatch_path = campaign / 'dispatches' / (request_id + '.json')
            expected_dispatches.add(dispatch_path)
            require(read_json(dispatch_path) == dict(tag=item[0], uid=item[1], order=item[2],
                total_attempt=number, inflight_path=str(inflight_path.relative_to(campaign)),
                inflight_sha256=file_hash(inflight_path), local_request_id=request_id,
                campaign_manifest_sha256=manifest['manifest_sha256']), 'Dispatch intent changed')
            # Bind the original attempt before selecting any inspected replay.
            for bound in (inflight_path, response_path, hash_path, dispatch_path):
                files[str(bound)] = file_hash(bound)
            state = policy.classify_record(response, judge)
            if state == 'blocked' and recovery is not None and item == item_of(recovery['target']):
                recovery_helper.validate_ambiguity(response, recovery['target'], response_path)
                if (recovery['_folder'] / 'inflight.json').exists() or inspected_ambiguity is None:
                    response, response_path, current_sha = recovery_helper.load_overlay(
                        context, recovery, intent, response, expected)
                    request_id = response['local_request_id']
                    require(request_id not in seen_ids, 'Duplicate recovery request ID')
                    seen_ids.add(request_id)
                    state = policy.classify_record(response, judge)
                    recovered = recovery_helper.evidence(recovery, response, response_path)
            inspecting = (state == 'blocked' and inspected_ambiguity is not None
                          and item == item_of(inspected_ambiguity) and recovered is None)
            if inspecting:
                from scripts import arena_retry_transport_recovery as recovery_helper
                recovery_helper.validate_ambiguity(response, inspected_ambiguity, response_path)
                resolution = 'blocked'
            else:
                require(state in ('valid', 'judge_failed'), 'Ambiguous or corrupt attempt blocked')
                require(valid_family(response.get('response_model'), model), 'Unknown returned judge model; blocked')
                require(isinstance(response.get('response_id'), str) and bool(response['response_id']),
                        'Missing response identity; blocked')
            evidence.append(dict(total_attempt=number, judge_model=model, record_path=str(response_path),
                record_sha256=current_sha, status=response['status'], local_request_id=request_id,
                request_sha256=response['request_sha256'], response_model=response.get('response_model')))
            record, previous_sha = response, current_sha
            if state == 'valid':
                resolution, selected = ('gpt41' if number == 6 else 'gpt4o'), response
            elif number == 6:
                resolution = 'failed'
        resolutions.append(dict(tag=item[0], uid=item[1], order=item[2], resolution=resolution,
                                selected_record=selected, attempts=evidence, transport_recovery=recovered))
    actual = {p for p in (campaign / 'attempts').rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == expected_attempt_files, 'Extra or missing attempt files')
    require(set((campaign / 'dispatches').glob('*')) == expected_dispatches,
            'Extra or missing dispatch history; resend blocked')
    for name in ('manifest.json', 'manifest.sha256'):
        files[str(campaign / name)] = file_hash(campaign / name)
    return resolutions, files


def _summary(rows, dispatched, workers, status=None):
    counts = {state: sum(r['resolution'] == state for r in rows)
              for state in ('gpt4o', 'gpt41', 'failed', 'pending')}
    complete = counts['pending'] == 0 and all(row['resolution'] != 'blocked' for row in rows)
    return dict(schema=SCHEMA, status=status or ('complete' if complete else 'partial'),
        complete=complete and status in (None, 'complete'), counts=counts,
        target_count=len(rows), dispatched_this_run=dispatched, workers=workers,
        new_attempts=sum(len(r['attempts'])-1 for r in rows),
        extra_physical_requests=sum(bool(r.get('transport_recovery')) for r in rows),
        unknown_usage_requests=sum(bool(r.get('transport_recovery')) for r in rows), updated_at=now())


def load_resolution(campaign, require_complete=True):
    """Pure read-only chain verification; never create relay/access credentials."""
    context = _load_context(campaign)
    campaign, manifest, _, _ = context
    rows, files = _resolutions(context)
    completion = campaign / 'complete.json'
    if completion.exists():
        saved = read_json(completion)
        require(saved.get('complete') is True and saved.get('status') == 'complete'
                and saved.get('manifest_sha256') == manifest['manifest_sha256']
                and saved.get('campaign_files_sha256') == files
                and saved.get('counts') == _summary(rows, 0, 0)['counts'], 'Completion binding mismatch')
        files[str(completion)] = file_hash(completion)
    if require_complete:
        require(completion.exists() and all(row['resolution'] != 'pending' for row in rows),
                'Campaign incomplete; completion required')
    manifest['campaign_files_sha256'] = files
    recoveries = [row['transport_recovery'] for row in rows if row.get('transport_recovery')]
    require(len(recoveries) <= 1, 'Multiple transport recoveries are forbidden')
    manifest['transport_recovery'] = recoveries[0] if recoveries else None
    return manifest, rows


def billing_check(campaign, relay):
    path = campaign / 'state/billing.json'
    value = billing_valid(read_json(path))
    usage = None
    for attempt in range(3):
        try:
            usage = relay.usage(value['start_date'])
            if type(usage) not in (float, int) or not math.isfinite(usage) or usage < 0:
                raise ValueError('Invalid billing usage')
            break
        except Exception:
            if attempt == 2:
                raise RuntimeError('Free billing usage failed after three attempts; dispatch blocked') from None
            time.sleep(attempt + 1)
    require(usage >= value['latest_usage_cny'] - 1e-6, 'Billing usage decreased; dispatch blocked')
    value.update(latest_usage_cny=usage, spent_cny=usage-BASE_USAGE, checked_at=now())
    atomic_json(path, value)
    if value['spent_cny'] >= BUDGET:
        raise RuntimeError('Cumulative billing budget guard reached; dispatch blocked')
    return value


def default_relay_factory(campaign, manifest, judge):
    """Reuse audited pre-POST connection retries, writing only into campaign."""
    import httpx
    import httpcore
    import httpcore._sync.connection
    suite = Path(manifest['source_suite'])
    transport = original.load_module(suite / 'resilient_judge.py')
    require(file_hash(suite / 'source/scripts/judge_arena_hard.py') == transport.FROZEN_JUDGE_SHA256
            and read_json(suite / 'resilient_policy.json')['transport_max_connect_attempts'] == 4,
            'Frozen transport binding mismatch')
    key = os.environ.get('LINKAPI_KEY') or Path('/root/.linkapi_key').read_text().strip()
    require(isinstance(key, str) and bool(key.strip()), 'Missing relay credential')
    run_id = str(uuid.uuid4())
    binding = dict(schema=transport.SCHEMA, run_id=run_id, created_at=now(),
        campaign_manifest_sha256=manifest['manifest_sha256'],
        resilient_policy_sha256=file_hash(suite / 'resilient_policy.json'),
        transport_source_sha256=file_hash(suite / 'resilient_judge.py'),
        frozen_judge_sha256=manifest['frozen_judge_sha256'], transport_max_connect_attempts=4,
        backoff_seconds=list(transport.BACKOFF_SECONDS),
        eligible_failure_phases=sorted(transport.CONNECTION_FAILURES),
        require_no_application_headers_started=True, trust_env=True,
        max_connections=32, max_keepalive_connections=32, keepalive_expiry_seconds=300,
        timeout_seconds=600, endpoint=transport.BASE_URL, proxy_environment=transport.proxy_summary(),
        python=sys.version.split()[0], httpx=httpx.__version__, httpcore=httpcore.__version__,
        httpx_transport_source_sha256=file_hash(inspect.getfile(httpx.HTTPTransport.__init__)),
        httpcore_connection_source_sha256=file_hash(inspect.getfile(httpcore._sync.connection.HTTPConnection)))
    immutable_json(campaign / 'job/transport_bindings' / (run_id+'.json'), binding)
    relay = object.__new__(judge.Relay)
    relay.client = transport.RetryingClient(httpx.Client(timeout=600,
        headers={'Authorization': f'Bearer {key}'}, trust_env=True,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=32, keepalive_expiry=300)),
        transport.AuditLog(campaign / 'job/connect_attempts.jsonl'), run_id=run_id)
    return relay


def _dispatch_intent(context, row):
    campaign, manifest, run, _ = context
    number = len(row['attempts']) + 1
    require(row['resolution'] == 'pending' and 2 <= number <= 6, 'Attempt policy exhausted')
    previous = row['attempts'][-1]
    request = {**run.request(*item_of(row)), 'model': 'gpt-4.1' if number == 6 else 'gpt-4o'}
    record = dict(tag=row['tag'], uid=row['uid'], order=row['order'], status='inflight',
        total_attempt=number, attempt=number-1, judge_model=request['model'], baseline_model=BASELINE,
        protocol_sha256=manifest['protocol_sha256'], campaign_manifest_sha256=manifest['manifest_sha256'],
        request=request, request_sha256=digest(request), local_request_id=str(uuid.uuid4()), started_at=now(),
        predecessor_record_sha256=previous['record_sha256'],
        supersedes_local_request_id=previous['local_request_id'])
    folder = target_dir(campaign, row) / f'{number:02d}'
    inflight = folder / 'inflight.json'
    immutable_json(inflight, record)
    immutable_json(campaign / 'dispatches' / (record['local_request_id']+'.json'), dict(
        tag=row['tag'], uid=row['uid'], order=row['order'], total_attempt=number,
        inflight_path=str(inflight.relative_to(campaign)), inflight_sha256=file_hash(inflight),
        local_request_id=record['local_request_id'], campaign_manifest_sha256=manifest['manifest_sha256']))
    return folder, record


def _paid_call(folder, intent, relay, judge):
    record = dict(intent)
    try:
        response = relay.judge_call(intent['request'])
        record.update({key: response.get(key) for key in
            ('answer', 'usage', 'finish_reason', 'response_id', 'response_model', 'provider_request_id')})
        score = judge.parse_score(record.get('answer'))
        valid = score is not None and record['finish_reason'] == 'stop' and judge.valid_usage(record['usage'])
        record.update(score=score, status='valid' if valid else 'invalid')
    except Exception as error:
        record.update(status='ambiguous', error_type=type(error).__name__)
    record['finished_at'] = now()
    immutable_json(folder / 'record.json', record)
    immutable_bytes(folder / 'record.sha256', (file_hash(folder / 'record.json')+'\n').encode())
    return record


def run_campaign(campaign, *, workers=32, max_requests=0, relay_factory=None):
    """Dispatch bounded batches, one serial next attempt per target per batch."""
    require(type(workers) is int and 1 <= workers <= 32, 'workers must be between 1 and 32')
    require(type(max_requests) is int and max_requests >= 0, 'max_requests must be nonnegative')
    campaign = Path(campaign).resolve()
    require(campaign.is_dir(), 'Prepare the campaign first')
    with controller_lock(campaign):
        context = _load_context(campaign)
        _, manifest, _, judge = context
        # All chains and any completion marker must validate before even billing.
        _, rows = load_resolution(campaign, require_complete=False)
        relay, dispatched = None, 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while any(row['resolution'] == 'pending' for row in rows):
                    if max_requests and dispatched >= max_requests:
                        break
                    if relay is None:
                        relay = relay_factory() if relay_factory else default_relay_factory(campaign, manifest, judge)
                    billing_check(campaign, relay)
                    pending = [row for row in rows if row['resolution'] == 'pending']
                    count = min(workers, max_requests-dispatched) if max_requests else workers
                    batch = pending[:count]
                    intents = [_dispatch_intent(context, row) for row in batch]
                    futures = [pool.submit(_paid_call, folder, record, relay, judge) for folder, record in intents]
                    dispatched += len(futures)
                    for future in as_completed(futures):
                        future.result()
                    rows, _ = _resolutions(context)
                    atomic_json(campaign / 'progress.json', _summary(rows, dispatched, workers, 'running'))
                # A crash or failed final billing check may leave all paid
                # judgments terminal without a completion marker. Resume that
                # state using only a fresh bounded free billing check.
                if (relay is None and not (campaign / 'complete.json').exists()
                        and all(row['resolution'] != 'pending' for row in rows)):
                    relay = relay_factory() if relay_factory else default_relay_factory(campaign, manifest, judge)
                if relay is not None:
                    billing_check(campaign, relay)
            rows, files = _resolutions(context)
            result = _summary(rows, dispatched, workers)
            result['manifest_sha256'] = manifest['manifest_sha256']
            atomic_json(campaign / 'progress.json', result)
            if result['complete'] and not (campaign / 'complete.json').exists():
                verify_files(manifest['source_files_sha256'])
                immutable_json(campaign / 'complete.json', {**result, 'campaign_files_sha256': files})
            return result
        except BaseException as error:
            progress = _summary(rows, dispatched, workers, 'blocked')
            progress['error_type'] = type(error).__name__
            atomic_json(campaign / 'progress.json', progress)
            raise
        finally:
            if relay is not None:
                relay.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-suite', type=Path, required=True)
    prepare.add_argument('--campaign', type=Path, required=True)
    prepare.add_argument('--exclusion-manifest', type=Path)
    execute = sub.add_parser('run')
    execute.add_argument('--campaign', type=Path, required=True)
    execute.add_argument('--workers', type=int, default=32)
    execute.add_argument('--max-requests', type=int, default=0)
    validate = sub.add_parser('validate')
    validate.add_argument('--campaign', type=Path, required=True)
    validate.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        manifest = prepare_campaign(args.source_suite, args.campaign, exclusion_manifest=args.exclusion_manifest)
        result = dict(status='prepared', target_count=manifest['target_count'])
    elif args.command == 'run':
        result = run_campaign(args.campaign, workers=args.workers, max_requests=args.max_requests)
    else:
        _, rows = load_resolution(args.campaign, require_complete=not args.allow_incomplete)
        result = _summary(rows, 0, 0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('Retry/fallback stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
