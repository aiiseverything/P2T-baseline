#!/usr/bin/env python3
"""One operator-inspected retry, preserving the frozen judge's retry archive API."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arena_exclusion_policy import classify_record
from scripts.run_arena_with_billing_retries import BillingRetryRelay
from scripts import run_arena_with_exclusions as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retry_once(suite, recovery, *, relay_factory=None, sleep=time.sleep):
    suite, recovery = Path(suite).resolve(), Path(recovery).resolve()
    approval = json.loads((recovery / 'approval.json').read_text())
    require(approval.get('policy_id') == 'arena_operator_inspected_transport_retry_v1'
            and approval.get('budget_cny') == 144 and approval.get('max_paid_calls') == 1,
            'Unrecognized inspected recovery limits')
    target = approval['target']
    item = target['model'], target['uid'], target['order']
    require(type(item[2]) is int and item[2] in (0, 1), 'Invalid inspected answer order')
    result_path = recovery / 'retry_result.json'
    require(not result_path.exists(), 'Recovery already attempted; explicit review required')
    run, judge = runner.load_suite(suite)
    relay, result = None, {'status': 'checking', 'paid_calls': 0, 'resume_eligible': False,
        'target': dict(target), 'cost_uncertainty': approval['cost_uncertainty'],
        'budget_cny': 144, 'started_at': judge.now()}
    with judge.exclusive_lock(run.directory):
        run.prepare()
        path = run.game_path(*item)
        require(sha(path) == target['game_sha256'], 'Inspected game bytes changed')
        original = run.load_record(*item)
        require(original['status'] == 'ambiguous' and original.get('error_type') == 'RemoteProtocolError'
                and type(original.get('attempt')) is int and original['attempt'] == 0
                and all(original.get(key) is None for key in ('answer', 'score', 'finish_reason', 'usage'))
                and isinstance(original.get('finished_at'), str) and original['finished_at']
                and original.get('local_request_id') == target['local_request_id']
                and original.get('request_sha256') == target['request_sha256']
                and judge.digest(original['request']) == target['request_sha256'],
                'Target is not the inspected unresolved first transport attempt')
        counts, blocked = Counter({state: 0 for state in runner.STATES}), []
        for uid in run.questions:
            for tag in runner.TAGS:
                for order in (0, 1):
                    key = (tag, uid, order)
                    state = classify_record(run.load_record(*key), judge)
                    counts[state] += 1
                    if state == 'blocked':
                        blocked.append(key)
        require(blocked == [item] and dict(counts) == approval['expected_counts'],
                'Suite coverage changed or additional blocked records need inspection')
        runner.validate_saved_billing(run)
        original_files = {p: sha(p) for p in (run.directory / 'state/games').rglob('*.json')}
        archive = run.directory / 'state/attempts' / item[0] / f'{item[1]}-{item[2]}' / (original['local_request_id'] + '.json')
        try:
            factory = relay_factory if relay_factory is not None else lambda: runner.default_relay_factory(suite)
            relay = BillingRetryRelay(factory(), sleep=sleep)
            run.billing_check(relay, 144)
            result.update(status='dispatching')
            judge.atomic_json(result_path, result)
            record = run.save_inflight(*item, retry=True)
            require(record['request'] == original['request']
                    and record['request_sha256'] == target['request_sha256']
                    and sha(archive) == target['game_sha256'], 'Original retry archive/request mismatch')
            result['paid_calls'] = 1
            try:
                response = relay.judge_call(record['request'])
                record.update({key: response.get(key) for key in
                               ('answer', 'usage', 'finish_reason', 'response_id',
                                'response_model', 'provider_request_id')})
                score = judge.parse_score(record['answer'], run.protocol['regex_patterns'])
                valid = (score is not None and record['finish_reason'] == 'stop'
                         and judge.valid_usage(record['usage']))
                record.update(status='valid' if valid else 'invalid', score=score)
            except Exception as error:
                record.update(status='ambiguous', error_type=type(error).__name__)
            record['finished_at'] = judge.now()
            judge.atomic_json(path, record)
            run._records[item] = record
            run._dirty_tags.add(item[0])
            state = classify_record(record, judge)
            for old_path, expected in original_files.items():
                require(sha(archive if old_path == path else old_path) == expected,
                        'An original record changed without its exact retry archive')
            run.export()
            result.update(classification=state, record_sha256=sha(path),
                          archive=str(archive), archive_sha256=sha(archive),
                          attempt=record['attempt'], finished_at=judge.now())
            # The original cumulative guard remains authoritative after this call.
            run.billing_check(relay, 144)
            result.update(status='complete' if state in ('valid', 'judge_failed') else 'blocked',
                          resume_eligible=state in ('valid', 'judge_failed'))
            judge.atomic_json(result_path, result)
            return result
        except Exception as error:
            result.update(status='failed', resume_eligible=False, error_type=type(error).__name__,
                          finished_at=judge.now())
            judge.atomic_json(result_path, result)
            raise
        finally:
            if relay is not None:
                relay.close()



def archive_stopped_execution(continuation, history, files_sha256):
    """Copy and fsync stopped mutable metadata before removing active filenames."""
    from scripts import watch_arena_exclusion_scores as watcher
    continuation, history = Path(continuation).resolve(), Path(history).resolve()
    allowed = {'judge_launch.json', 'watcher_launch.json', 'judging_exit_code',
               'judging.log', 'watcher.log', 'controller_state.json',
               'prelaunch_audit.json', 'launch_audit.json'}
    require(isinstance(files_sha256, dict) and bool(files_sha256)
            and set(files_sha256) <= allowed
            and {'judge_launch.json', 'watcher_launch.json', 'judging_exit_code'} <= set(files_sha256),
            'Archive plan must contain only stopped mutable execution files')
    require(history.parent == continuation / 'history', 'Archive must be a direct history child')
    # Inspect every input before any mutation. On interrupted copy/removal,
    # the exact archived bytes are an acceptable replacement for missing inputs.
    locations = {}
    for name, expected in files_sha256.items():
        source, archived = continuation / name, history / name
        locations[name] = source if source.exists() else archived
        require(locations[name].is_file() and sha(locations[name]) == expected,
                'Stopped execution metadata changed')
        if archived.exists():
            require(sha(archived) == expected, 'Prior execution archive changed')
    for name in ('judge_launch.json', 'watcher_launch.json'):
        require(not watcher.alive(json.loads(locations[name].read_text())['pid']),
                'Cannot archive a live judge or watcher')
    require(int(locations['judging_exit_code'].read_text().strip()) != 0,
            'Only the inspected failed execution may be resumed')
    history.mkdir(parents=True, exist_ok=True)
    for name in files_sha256:
        archived = history / name
        if not archived.exists():
            with archived.open('xb') as stream:
                stream.write(locations[name].read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
    descriptor = os.open(history, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    report = {'status': 'archived', 'files_sha256': files_sha256,
              'history': str(history)}
    watcher.atomic_json(history / 'archive_manifest.json', report)
    for name, expected in files_sha256.items():
        require(sha(history / name) == expected, 'Archive verification failed')
        active = continuation / name
        if active.exists():
            require(sha(active) == expected, 'Active execution metadata changed before removal')
            active.unlink()
    descriptor = os.open(continuation, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--execute-inspected-retry', action='store_true')
    args = parser.parse_args(argv)
    require(args.execute_inspected_retry, 'Explicit inspected-retry execution is required')
    print(json.dumps(retry_once(args.suite, args.recovery), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Inspected recovery stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
