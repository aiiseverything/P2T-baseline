#!/usr/bin/env python3
"""Wait for completed judging, verify frozen inputs, and publish CPU scores once.

This controller has no judge or API client. Interrupted score directories are
preserved for inspection and are never automatically overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

MODELS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
OUTPUTS = ('results.json', 'results.csv', 'common_subset_results.csv', 'exclusions.jsonl')
BINDINGS = ('judging_source_manifest.json', 'scoring_source_manifest.json',
            'preexisting_records.json', 'policy.json', '../exclusions_judging_complete.json')
CPU_PYTHON = '/root/miniconda3/envs/sml/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def alive(pid):
    require(type(pid) is int and pid > 0, 'Invalid judge PID')
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(')', 1)[1].split()[0] not in ('Z', 'X', 'x')


def verify_hashes(base, entries):
    require(isinstance(entries, dict) and bool(entries), 'Missing file hash bindings')
    for name, expected in entries.items():
        require(isinstance(name, str) and isinstance(expected, str)
                and file_hash(base / name) == expected, 'Bound file hash mismatch')


def verify_frozen_inputs(suite, e):
    for name in ('judging', 'scoring'):
        verify_hashes(e, read_json(e / f'{name}_source_manifest.json')['files_sha256'])
    verify_hashes(suite, read_json(e / 'preexisting_records.json')['records_sha256'])
    launch = read_json(e / 'judge_launch.json')
    if 'source_manifest_sha256' in launch:
        require(file_hash(e / 'judging_source_manifest.json') == launch['source_manifest_sha256'],
                'Judging launch source manifest hash mismatch')


def validate_judging(suite):
    completed = read_json(suite / 'exclusions_judging_complete.json')
    counts = completed['counts']['total']
    require(completed.get('complete') is True and completed.get('expected_games') == 6000
            and all(type(counts.get(k)) is int and counts[k] >= 0
                    for k in ('valid', 'judge_failed', 'missing', 'blocked'))
            and counts['missing'] == counts['blocked'] == 0
            and counts['valid'] + counts['judge_failed'] == 6000,
            'Judging completion coverage is incomplete')
    return counts


def validate_scores(scores, judging_counts):
    for name in OUTPUTS:
        require((scores / name).is_file(), 'Missing required score output')
    report = read_json(scores / 'results.json')
    coverage = report['coverage']
    per_model, models = coverage['per_model'], report['models']
    common = report['common_valid_subset']
    require(set(per_model) == set(models) == set(common['models']) == set(MODELS),
            'Score reports must contain the six expected models')
    require(coverage['attempted_games'] == 6000 and coverage['questions'] == 500
            and coverage['valid_games'] == judging_counts['valid']
            and coverage['judge_failed_games'] == judging_counts['judge_failed'],
            'Score coverage differs from completed judging')
    require(type(common['questions']) is int and 0 <= common['questions'] <= 500,
            'Invalid common subset coverage')
    count_keys = ('attempted_games', 'valid_games', 'judge_failed_games', 'excluded_prompts',
                  'valid_but_discarded_partner_games', 'retained_prompts', 'retained_games')
    for model, count in per_model.items():
        require(all(type(count.get(k)) is int and count[k] >= 0 for k in count_keys),
                'Invalid per-model score counts')
        require(count['attempted_games'] == 1000
                and count['valid_games'] + count['judge_failed_games'] == 1000
                and count['retained_games'] == 2 * count['retained_prompts']
                and count['retained_prompts'] + count['excluded_prompts'] == 500
                and count['valid_games'] == count['retained_games'] + count['valid_but_discarded_partner_games']
                and count['judge_failed_games'] + count['valid_but_discarded_partner_games'] == 2 * count['excluded_prompts'],
                'Inconsistent per-model retained/excluded coverage')
        require(models[model]['prompts'] == count['retained_prompts']
                and models[model]['games'] == count['retained_games']
                and common['models'][model]['prompts'] == common['questions']
                and common['models'][model]['games'] == 2 * common['questions']
                and common['questions'] <= count['retained_prompts'],
                'Score result denominators differ from coverage')
    for key in ('attempted_games', 'valid_games', 'judge_failed_games',
                'valid_but_discarded_partner_games'):
        require(coverage[key] == sum(count[key] for count in per_model.values()),
                'Total score coverage differs from per-model counts')
    require(coverage['excluded_model_prompts'] == sum(count['excluded_prompts'] for count in per_model.values()),
            'Excluded prompt totals differ')
    manifest = report.get('exclusion_manifest', {})
    with (scores / 'exclusions.jsonl').open() as stream:
        exclusions = [json.loads(line) for line in stream if line.strip()]
    require(len(exclusions) == 2 * coverage['excluded_model_prompts']
            and ('rows' not in manifest or manifest['rows'] == len(exclusions)),
            'Exclusion manifest row count differs from dropped prompts')
    if 'sha256' in manifest:
        require(manifest.get('path') == 'exclusions.jsonl'
                and file_hash(scores / 'exclusions.jsonl') == manifest['sha256'],
                'Exclusion manifest hash mismatch')
    for key in ('output_files_sha256', 'outputs_sha256', 'files_sha256', 'inputs_sha256'):
        if key in report:
            verify_hashes(scores, report[key])
    return coverage


def watch_suite(suite, *, sleep=time.sleep, runner=subprocess.run, poll_seconds=15,
                continuation_dir=None):
    require(type(poll_seconds) in (int, float) and math.isfinite(poll_seconds)
            and 0 < poll_seconds <= 15, 'Polling interval must be at most 15 seconds')
    suite = Path(suite).resolve()
    e = (suite / Path(continuation_dir or 'continuation-with-exclusions')).resolve()
    require(e.parent == suite and e.is_dir(),
            'continuation directory must be an existing direct child of the suite')
    state = {'state': 'running', 'phase': 'waiting_judging', 'started_at': now(), 'pid': os.getpid()}
    with (e / '.scoring-controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another scoring watcher holds the suite lock') from None
        try:
            completion_path = e / 'evaluation_complete.json'
            if completion_path.exists():
                completed = read_json(completion_path)
                require(completed.get('phase') == 'complete' and completed.get('complete') is True,
                        'Invalid controller completion marker')
                require(set(completed['output_files_sha256']) == {'scores/' + name for name in OUTPUTS}
                        and set(completed['bindings_sha256']) == set(BINDINGS),
                        'Incomplete controller output hash bindings')
                verify_hashes(e, completed['output_files_sha256'])
                verify_hashes(e, completed['bindings_sha256'])
                verify_frozen_inputs(suite, e)
                atomic_json(e / 'controller_state.json', completed)
                return completed
            pid = read_json(e / 'judge_launch.json')['pid']
            exit_path = e / 'judging_exit_code'
            while not exit_path.exists():
                if not alive(pid) and not exit_path.exists():
                    raise RuntimeError('Judge process exited without a durable exit marker')
                state.update(judge_pid=pid, updated_at=now())
                atomic_json(e / 'controller_state.json', state)
                sleep(poll_seconds)
            code = int(exit_path.read_text().strip())
            state['judging_returncode'] = code
            if code != 0:
                raise RuntimeError('Judging failed; scoring will not start')
            state['phase'] = 'verifying_judging'
            counts = validate_judging(suite)
            verify_frozen_inputs(suite, e)
            scores = e / 'scores'
            if scores.exists():
                raise FileExistsError('Partial score output exists; a fresh output is required')
            state.update(phase='scoring', updated_at=now())
            atomic_json(e / 'controller_state.json', state)
            command = [CPU_PYTHON, str(e / 'source/scripts/score_arena_with_exclusions.py'),
                       '--suite', str(suite), '--policy', str(e / 'policy.json'), '--output', str(scores)]
            environment = dict(os.environ, PYTHONPATH=f'{e / "source"}:{suite / "source"}',
                               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
            with (e / 'scoring.log').open('w') as stream:
                process = runner(command, cwd=suite, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
            state['scoring_returncode'] = process.returncode
            if process.returncode != 0:
                raise RuntimeError('CPU scoring failed; inspect scoring.log')
            coverage = validate_scores(scores, counts)
            verify_frozen_inputs(suite, e)
            state.update(state='complete', phase='complete', complete=True, completed_at=now(),
                         coverage=coverage,
                         output_files_sha256={'scores/' + name: file_hash(scores / name) for name in OUTPUTS},
                         bindings_sha256={name: file_hash(e / name) for name in BINDINGS})
            atomic_json(completion_path, state)
            atomic_json(e / 'controller_state.json', state)
            return state
        except Exception as error:
            state.update(state='failed', failed_at=now(), error_type=type(error).__name__)
            atomic_json(e / 'controller_state.json', state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--continuation-dir', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(watch_suite(args.suite, continuation_dir=args.continuation_dir), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Scoring watcher stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
