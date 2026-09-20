#!/usr/bin/env python3
"""One authorized replay of the inspected logical-attempt-2 transport ambiguity.

The original POST may already have been processed and billed. Its immutable
record remains evidence of an unknown outcome, never a completed invalid vote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import arena_retry_fallback as runner

SCHEMA = 'arena_inspected_logical2_replay_v1'
RECOVERY_ID = 'lam2-d63881609ade4307-1-once'
AUTHORIZED_TARGET = dict(tag='lam2', uid='d63881609ade4307', order=1, total_attempt=2,
    record_sha256='105b42d52ae4b6357e580deab2df37112854abc756871e80e4da69d78ec68f9c',
    local_request_id='7eeb2401-087f-4c8e-9982-671942d4e4e3')
AUTHORIZATION = 'User explicitly authorized one inspected replay after disclosure that the POST was sent and may already have been billed.'
COST_UNCERTAINTY = 'Original POST may have been processed/billed; one inspected replay can add one charge.'
ALLOWED_FILES = {'manifest.json', 'manifest.sha256', 'inflight.json', 'record.json', 'record.sha256'}


def validate_ambiguity(record, target, path):
    runner.require(runner.file_hash(path) == target['record_sha256']
        and runner.item_of(record) == runner.item_of(target)
        and record.get('local_request_id') == target['local_request_id']
        and record.get('total_attempt') == target['total_attempt'] == 2
        and record.get('attempt') == 1 and record.get('judge_model') == 'gpt-4o'
        and record.get('status') == 'ambiguous' and record.get('error_type') == 'RemoteProtocolError'
        and isinstance(record.get('finished_at'), str) and bool(record['finished_at'])
        and all(record.get(key) is None for key in
            ('answer', 'score', 'usage', 'finish_reason', 'response_id', 'response_model', 'provider_request_id')),
        'Recovery target is not the exact inspected ambiguity')


def _target_binding(context, target):
    campaign, manifest, _, _ = context
    target = dict(target)
    if manifest['target_count'] == 590:
        runner.require(target == AUTHORIZED_TARGET,
                       'Production recovery must match the exact authorized target scope')
    runner.require(set(target) == set(AUTHORIZED_TARGET) and type(target.get('order')) is int
        and type(target.get('total_attempt')) is int and target['total_attempt'] == 2,
        'Invalid inspected target scope')
    runner.require(any(runner.item_of(row) == runner.item_of(target) for row in manifest['targets']),
                   'Inspected target is outside the campaign')
    path = runner.target_dir(campaign, target) / '02/record.json'
    record = runner.read_json(path)
    validate_ambiguity(record, target, path)
    return {**target, 'record_path': str(path), 'request_sha256': record['request_sha256']}


def _trace_evidence(campaign, target):
    path = campaign / 'job/connect_attempts.jsonl'
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    matching = [row for row in rows if row.get('request_sha256') == target['request_sha256']]
    runner.require(len(matching) == 2 and matching[0].get('event') == 'attempt_started'
        and matching[1].get('event') == 'attempt_finished'
        and matching[0].get('audit_id') == matching[1].get('audit_id')
        and all(row.get('method') == 'POST' and row.get('transport_attempt') == 1 for row in matching),
        'Inspected transport trace is not a unique original POST')
    ended = matching[1]
    phases = [event['event'] for event in ended.get('events', [])]
    runner.require(ended.get('outcome') == 'exception' and ended.get('raised_error_type') == 'RemoteProtocolError'
        and ended.get('application_headers_started') is True and ended.get('connection_failure_phase') is None
        and ended.get('retry_eligible') is False and ended.get('will_retry') is False
        and 'http11.send_request_headers.complete' in phases
        and 'http11.send_request_body.complete' in phases
        and 'http11.receive_response_headers.failed' in phases,
        'Transport trace differs from the inspected uncertain delivery')
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), events=matching)


def prepare_recovery(campaign, *, recovery_id=RECOVERY_ID, target=None):
    """Offline immutable binding; CLI uses the one explicitly authorized target.

    A supplied target is an explicit library-level inspected scope, useful for
    offline fixture campaigns. It does not change the one-replay policy.
    """
    campaign = Path(campaign).resolve()
    runner.require(isinstance(recovery_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', recovery_id),
                   'Invalid recovery ID')
    with runner.controller_lock(campaign):
        runner.require(not (campaign / 'transport_recovery').exists(), 'A recovery already exists')
        runner.require(not (campaign / 'complete.json').exists(), 'Completed campaign cannot be recovered')
        context = runner._load_context(campaign)
        binding = _target_binding(context, AUTHORIZED_TARGET if target is None else target)
        rows, files = runner._resolutions(context, inspected_ambiguity=binding)
        runner.require([runner.item_of(row) for row in rows if row['resolution'] == 'blocked']
                       == [runner.item_of(binding)], 'Exactly the inspected ambiguity is required')
        trace = _trace_evidence(campaign, binding)
        execution = {str(Path(path).resolve()): runner.file_hash(path) for path in
                     (__file__, runner.__file__, runner.original.__file__, runner.policy.__file__)}
        manifest = dict(schema=SCHEMA, recovery_id=recovery_id, created_at=runner.now(),
            parent_manifest_sha256=context[1]['manifest_sha256'], target=binding,
            authorization=AUTHORIZATION, max_physical_replays=1, logical_attempt=2,
            completed_attempt_policy=context[1]['policy'], cost_uncertainty=COST_UNCERTAINTY,
            paused_campaign_files_sha256=files, transport_audit_prefix=trace,
            execution_files_sha256=execution)
        runner.verify_files(files)
        folder = campaign / 'transport_recovery' / recovery_id
        runner.immutable_json(folder / 'manifest.json', manifest)
        runner.immutable_bytes(folder / 'manifest.sha256', (runner.file_hash(folder / 'manifest.json')+'\n').encode())
        return manifest


def load_binding(context):
    """Read and verify one additive recovery binding, including immutable past."""
    campaign, parent, _, _ = context
    root = campaign / 'transport_recovery'
    if not root.exists():
        return None
    folders = list(root.iterdir())
    runner.require(len(folders) == 1 and folders[0].is_dir() and not folders[0].is_symlink(),
                   'Exactly one inspected recovery is allowed')
    folder = folders[0]
    runner.require(all(path.is_file() and not path.is_symlink() for path in folder.iterdir())
                   and {path.name for path in folder.iterdir()} <= ALLOWED_FILES,
                   'Unexpected recovery files')
    manifest_path = folder / 'manifest.json'
    manifest = runner.read_json(manifest_path)
    sha = runner.file_hash(manifest_path)
    runner.require((folder / 'manifest.sha256').read_text().strip() == sha,
                   'Recovery authorization hash mismatch')
    runner.require(manifest.get('schema') == SCHEMA and manifest.get('recovery_id') == folder.name
        and manifest.get('parent_manifest_sha256') == parent['manifest_sha256']
        and manifest.get('max_physical_replays') == 1 and manifest.get('logical_attempt') == 2
        and manifest.get('authorization') == AUTHORIZATION
        and manifest.get('cost_uncertainty') == COST_UNCERTAINTY
        and manifest.get('completed_attempt_policy') == parent['policy'], 'Recovery authorization changed')
    runner.verify_files(manifest['paused_campaign_files_sha256'])
    runner.verify_files(manifest['execution_files_sha256'])
    expected_execution = {str(Path(path).resolve()): runner.file_hash(path) for path in
                          (__file__, runner.__file__, runner.original.__file__, runner.policy.__file__)}
    runner.require(manifest['execution_files_sha256'] == expected_execution,
                   'Recovery execution source is not the frozen authorized implementation')
    target = manifest['target']
    actual = _target_binding(context, {key: target[key] for key in AUTHORIZED_TARGET})
    runner.require(actual == target, 'Recovery target binding changed')
    trace = manifest['transport_audit_prefix']
    runner.require(trace['path'] == str(campaign / 'job/connect_attempts.jsonl')
        and type(trace['bytes']) is int and trace['bytes'] > 0, 'Invalid transport audit prefix')
    with Path(trace['path']).open('rb') as stream:
        prefix = stream.read(trace['bytes'])
    runner.require(len(prefix) == trace['bytes'] and hashlib.sha256(prefix).hexdigest() == trace['sha256'],
                   'Original transport audit prefix changed')
    matching = [json.loads(line) for line in prefix.splitlines() if line.strip()
                and json.loads(line).get('request_sha256') == target['request_sha256']]
    runner.require(matching == trace['events'], 'Inspected transport evidence changed')
    files = {str(path): runner.file_hash(path) for path in folder.iterdir()}
    files.update(manifest['execution_files_sha256'])
    return dict(manifest, _folder=folder, _sha256=sha, _files=files)


def load_overlay(context, binding, old_intent, old_response, expected):
    """Verify the replay as logical attempt 2 without rewriting the ambiguity."""
    folder = binding['_folder']
    paths = [folder / name for name in ('inflight.json', 'record.json', 'record.sha256')]
    runner.require(all(path.is_file() for path in paths), 'Recovery inflight or not yet dispatched; blocked')
    intent, record = runner.read_json(paths[0]), runner.read_json(paths[1])
    sha = runner.file_hash(paths[1])
    runner.require(paths[2].read_text().strip() == sha, 'Recovery response hash mismatch')
    replay_fields = dict(replays_record_sha256=binding['target']['record_sha256'],
        replays_local_request_id=old_response['local_request_id'],
        recovery_manifest_sha256=binding['_sha256'], recovery_id=binding['recovery_id'])
    runner.require(all(intent.get(key) == value and record.get(key) == value
                       for key, value in {**expected, **replay_fields}.items()),
                   'Recovery request or predecessor identity changed')
    request_id = intent.get('local_request_id')
    runner.require(isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', request_id)
        and request_id != old_response['local_request_id'] and record.get('local_request_id') == request_id
        and intent.get('status') == 'inflight' and isinstance(intent.get('started_at'), str)
        and bool(intent['started_at']) and record.get('started_at') == intent['started_at']
        and all(type(record.get(key)) is int and type(intent.get(key)) is int
                for key in ('order', 'attempt', 'total_attempt')), 'Recovery dispatch identity changed')
    return record, paths[1], sha


def evidence(binding, replacement, path):
    target = binding['target']
    old = dict(record_path=target['record_path'], record_sha256=target['record_sha256'],
        local_request_id=target['local_request_id'], request_sha256=target['request_sha256'],
        status='ambiguous', error_type='RemoteProtocolError')
    new = dict(record_path=str(path), record_sha256=runner.file_hash(path),
        local_request_id=replacement['local_request_id'], request_sha256=replacement['request_sha256'])
    return dict(recovery_id=binding['recovery_id'], logical_attempt=2, physical_replay_attempts=1,
        unknown_usage_requests=1, original_ambiguous=old, replacement=new,
        authorization_path=str(binding['_folder'] / 'manifest.json'), authorization_sha256=binding['_sha256'],
        cost_uncertainty=COST_UNCERTAINTY)


def retry_once(campaign, *, relay_factory=None):
    """At most one paid replay; repeat invocation can only settle free billing."""
    campaign = Path(campaign).resolve()
    with runner.controller_lock(campaign):
        context = runner._load_context(campaign)
        binding = load_binding(context)
        runner.require(binding is not None, 'Explicit recovery authorization required')
        folder = binding['_folder']
        target = binding['target']
        # Every interrupted intent or corrupt response blocks before a relay.
        if (folder / 'inflight.json').exists():
            rows, _ = runner._resolutions(context)
            replay_complete = True
        else:
            runner.require(not (folder / 'record.json').exists() and not (folder / 'record.sha256').exists(),
                           'Recovery response has no durable intent')
            rows, _ = runner._resolutions(context, inspected_ambiguity=target)
            replay_complete = False
        relay, paid = None, 0
        try:
            relay = relay_factory() if relay_factory else runner.default_relay_factory(campaign, context[1], context[3])
            runner.billing_check(campaign, relay)
            if not replay_complete:
                old = runner.read_json(Path(target['record_path']).with_name('inflight.json'))
                intent = dict(old, local_request_id=str(uuid.uuid4()), started_at=runner.now(),
                    replays_record_sha256=target['record_sha256'], replays_local_request_id=target['local_request_id'],
                    recovery_manifest_sha256=binding['_sha256'], recovery_id=binding['recovery_id'])
                runner.immutable_json(folder / 'inflight.json', intent)
                paid = 1
                runner._paid_call(folder, intent, relay, context[3])
                runner.billing_check(campaign, relay)
            # Verify the response and all unrelated chains before allowing resume.
            rows, _ = runner._resolutions(context)
            row = next(row for row in rows if runner.item_of(row) == runner.item_of(target))
            return dict(status='complete', resume_eligible=True, paid_calls_this_invocation=paid,
                physical_replay_attempts=1, unknown_usage_requests=1,
                transport_recovery=row['transport_recovery'])
        finally:
            if relay is not None:
                relay.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'retry'))
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_recovery(args.campaign) if args.command == 'prepare' else retry_once(args.campaign)
    print(json.dumps(result if args.command == 'retry' else
                     dict(status='prepared', recovery_id=result['recovery_id'], target=result['target']), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Inspected retry stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
