"""Durable watcher tests: real files/process state, fake scorer subprocess only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')


def watcher():
    path = ROOT / 'scripts/watch_arena_exclusion_scores.py'
    assert path.is_file(), 'Durable scoring watcher has not been implemented'
    spec = importlib.util.spec_from_file_location('score_watcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def suite(tmp_path):
    e = tmp_path / 'continuation-with-exclusions'
    for filename in ('source/scripts/run_arena_with_exclusions.py',
                     'source/scripts/score_arena_with_exclusions.py'):
        path = e / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen offline fixture\n')
    for name, source in (('judging', 'run_arena_with_exclusions.py'),
                         ('scoring', 'score_arena_with_exclusions.py')):
        relative = 'source/scripts/' + source
        put(e / f'{name}_source_manifest.json', {'files_sha256': {relative: sha(e / relative)}})
    record = tmp_path / 'model_judgment/gpt-4o/state/games/base/old.json'
    put(record, {'original': 'preserve bytes'})
    put(e / 'preexisting_records.json', {'records_sha256': {str(record.relative_to(tmp_path)): sha(record)}})
    put(e / 'judge_launch.json', {'pid': os.getpid(),
        'source_manifest_sha256': sha(e / 'judging_source_manifest.json')})
    put(e / 'policy.json', {'policy_id': 'arena_judge_output_exclusions_v1'})
    put(tmp_path / 'exclusions_judging_complete.json', {'complete': True, 'expected_games': 6000,
        'counts': {'total': {'valid': 5999, 'judge_failed': 1, 'missing': 0, 'blocked': 0}}})
    (e / 'judging_exit_code').write_text('0\n')
    return tmp_path


def output_report(suite, continuation='continuation-with-exclusions'):
    e = suite / continuation
    scores = e / 'scores'
    scores.mkdir()
    for name in ('results.csv', 'common_subset_results.csv'):
        (scores / name).write_text('model,prompts\n' + ''.join(f'{m},499\n' for m in MODELS))
    exclusions = [
        {'model': 'base', 'uid': 'excluded-u0', 'order': 0,
         'reason': 'missing_verdict', 'record_classification': 'judge_failed',
         'finish_reason': 'stop', 'score': None, 'game_sha256': 'offline-game-0'},
        {'model': 'base', 'uid': 'excluded-u0', 'order': 1,
         'reason': 'partner_judge_failed', 'record_classification': 'valid',
         'finish_reason': 'stop', 'score': 'B>A', 'game_sha256': 'offline-game-1'},
    ]
    (scores / 'exclusions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exclusions))
    per_model = {m: {'attempted_games': 1000, 'valid_games': 1000, 'judge_failed_games': 0,
                    'excluded_prompts': 0, 'valid_but_discarded_partner_games': 0,
                    'retained_prompts': 500, 'retained_games': 1000, 'expanded_rows': 1000}
                 for m in MODELS}
    per_model['base'].update(valid_games=999, judge_failed_games=1, excluded_prompts=1,
                             valid_but_discarded_partner_games=1, retained_prompts=499,
                             retained_games=998, expanded_rows=998)
    result = {'coverage': {'attempted_games': 6000, 'questions': 500, 'valid_games': 5999,
                           'judge_failed_games': 1, 'per_model': per_model,
                           'excluded_model_prompts': 1, 'valid_but_discarded_partner_games': 1},
              'models': {m: {'prompts': per_model[m]['retained_prompts'],
                             'games': per_model[m]['retained_games']} for m in MODELS},
              'common_valid_subset': {'questions': 499,
                  'models': {m: {'prompts': 499, 'games': 998} for m in MODELS}},
              'exclusion_manifest': {'path': 'exclusions.jsonl', 'rows': 2,
                                     'sha256': sha(scores / 'exclusions.jsonl')}}
    put(scores / 'results.json', result)
    return result


def run_ok(suite):
    def run(command, **kwargs):
        assert command == ['/root/miniconda3/envs/sml/bin/python',
            str(suite / 'continuation-with-exclusions/source/scripts/score_arena_with_exclusions.py'),
            '--suite', str(suite), '--policy', str(suite / 'continuation-with-exclusions/policy.json'),
            '--output', str(suite / 'continuation-with-exclusions/scores')]
        assert kwargs['cwd'] == suite
        assert kwargs['env']['PYTHONPATH'] == f'{suite}/continuation-with-exclusions/source:{suite}/source'
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == kwargs['env']['OMP_NUM_THREADS'] == '1'
        kwargs['stdout'].write('offline scoring output\n')
        output_report(suite)
        return SimpleNamespace(returncode=0)
    return run


def no_scoring(*args, **kwargs):
    pytest.fail('Scoring must not be launched')


def test_live_judge_is_polled_then_scored_once_and_completed_resume_reuses_hashes(suite):
    e = suite / 'continuation-with-exclusions'
    (e / 'judging_exit_code').unlink()
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        (e / 'judging_exit_code').write_text('0\n')
    result = watcher().watch_suite(suite, sleep=sleep, runner=run_ok(suite))
    assert waits == [15]
    assert result['phase'] == 'complete' and result['coverage']['valid_games'] == 5999
    assert set(result['output_files_sha256']) == {'scores/results.json', 'scores/results.csv',
        'scores/common_subset_results.csv', 'scores/exclusions.jsonl'}
    assert not (suite / 'evaluation_complete.json').exists()
    assert json.loads((e / 'controller_state.json').read_text())['phase'] == 'complete'
    assert (e / 'scoring.log').read_text() == 'offline scoring output\n'
    assert watcher().watch_suite(suite, runner=no_scoring) == result
    (e / 'scores/results.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        watcher().watch_suite(suite, runner=no_scoring)


@pytest.mark.parametrize('code', ['1', '2', '-9', 'invalid'])
def test_nonzero_or_invalid_judge_exit_never_scores(suite, code):
    e = suite / 'continuation-with-exclusions'
    (e / 'judging_exit_code').write_text(code)
    with pytest.raises((RuntimeError, ValueError)):
        watcher().watch_suite(suite, runner=no_scoring)
    assert json.loads((e / 'controller_state.json').read_text())['state'] == 'failed'
    assert not (e / 'scores').exists()


def test_zombie_judge_without_exit_marker_fails_without_sleeping(suite):
    e = suite / 'continuation-with-exclusions'
    (e / 'judging_exit_code').unlink()
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        launch = json.loads((e / 'judge_launch.json').read_text()); launch['pid'] = pid
        put(e / 'judge_launch.json', launch)
        with pytest.raises(RuntimeError, match='exited|dead'):
            watcher().watch_suite(suite, runner=no_scoring,
                                  sleep=lambda _: pytest.fail('Zombie must not keep watcher waiting'))
    finally:
        os.waitpid(pid, 0)
    assert json.loads((e / 'controller_state.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('damage', ['complete', 'total', 'missing'])
def test_exit_zero_requires_complete_terminal_coverage(suite, damage):
    path = suite / 'exclusions_judging_complete.json'
    record = json.loads(path.read_text())
    if damage == 'complete': record['complete'] = False
    elif damage == 'total': record['counts']['total']['valid'] -= 1
    else: record['counts']['total']['missing'] = 1
    put(path, record)
    with pytest.raises(ValueError, match='completion|coverage'):
        watcher().watch_suite(suite, runner=no_scoring)
    assert not (suite / 'continuation-with-exclusions/scores').exists()


@pytest.mark.parametrize('relative', ['continuation-with-exclusions/source/scripts/run_arena_with_exclusions.py',
    'continuation-with-exclusions/source/scripts/score_arena_with_exclusions.py',
    'model_judgment/gpt-4o/state/games/base/old.json'])
def test_changed_frozen_sources_or_old_records_prevent_scoring(suite, relative):
    (suite / relative).write_text('changed bytes')
    with pytest.raises(ValueError, match='hash'):
        watcher().watch_suite(suite, runner=no_scoring)
    assert not (suite / 'continuation-with-exclusions/scores').exists()


def test_interrupted_partial_score_directory_is_not_overwritten(suite):
    path = suite / 'continuation-with-exclusions/scores'
    path.mkdir(); (path / 'partial').write_text('keep')
    with pytest.raises(FileExistsError, match='fresh|partial|exist'):
        watcher().watch_suite(suite, runner=no_scoring)
    assert (path / 'partial').read_text() == 'keep'


def test_scorer_failure_persists_failure_without_completion(suite):
    with pytest.raises(RuntimeError, match='scor'):
        watcher().watch_suite(suite, runner=lambda *a, **k: SimpleNamespace(returncode=3))
    e = suite / 'continuation-with-exclusions'
    assert not (e / 'evaluation_complete.json').exists()
    assert json.loads((e / 'controller_state.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('damage', ['attempted', 'permodel', 'retained', 'models', 'common',
                                    'hash', 'exclusions', 'exclusion_rows'])
def test_zero_exit_cannot_publish_inconsistent_score_report(suite, damage):
    def run(*args, **kwargs):
        result = output_report(suite)
        if damage == 'attempted': result['coverage']['attempted_games'] = 5999
        elif damage == 'permodel': result['coverage']['per_model']['base']['valid_games'] = 998
        elif damage == 'retained': result['coverage']['per_model']['base']['retained_games'] = 999
        elif damage == 'models': del result['models']['lam8']
        elif damage == 'common': result['common_valid_subset']['models']['base']['games'] = 1
        elif damage == 'hash': result['exclusion_manifest']['sha256'] = 'bad'
        elif damage == 'exclusion_rows': result['exclusion_manifest']['rows'] = 3
        else:
            path = suite / 'continuation-with-exclusions/scores/exclusions.jsonl'
            path.write_text('')
            result['exclusion_manifest']['sha256'] = sha(path)
        put(suite / 'continuation-with-exclusions/scores/results.json', result)
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError):
        watcher().watch_suite(suite, runner=run)
    assert not (suite / 'continuation-with-exclusions/evaluation_complete.json').exists()


def test_explicit_v2_directory_scores_there_and_preserves_v1(suite):
    original = suite / 'continuation-with-exclusions'
    target = suite / 'continuation-with-exclusions-v2'
    import shutil
    shutil.copytree(original, target)
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    def run(command, **kwargs):
        assert command[1] == str(target / 'source/scripts/score_arena_with_exclusions.py')
        assert command[command.index('--policy') + 1] == str(target / 'policy.json')
        assert command[command.index('--output') + 1] == str(target / 'scores')
        assert kwargs['env']['PYTHONPATH'] == f'{target}/source:{suite}/source'
        output_report(suite, target.name)
        return SimpleNamespace(returncode=0)
    result = watcher().watch_suite(suite, continuation_dir=target, runner=run)
    assert result['phase'] == 'complete'
    assert (target / 'evaluation_complete.json').is_file()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert watcher().watch_suite(suite, continuation_dir=target, runner=no_scoring) == result


@pytest.mark.parametrize('directory', ['..', '/', 'missing-child'])
def test_watcher_rejects_invalid_continuation_directory(suite, directory):
    with pytest.raises(ValueError, match='continuation'):
        watcher().watch_suite(suite, continuation_dir=directory, runner=no_scoring)
