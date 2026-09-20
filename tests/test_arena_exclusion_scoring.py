"""Offline selection and scoring of terminal two-order Arena judgments."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUITE = ROOT / 'runs/arena-hard-v2-gpt4o-judge-20260917'
BASELINE = 'gpt-4o-mini-2024-07-18'
MODELS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')


def scorer():
    path = ROOT / 'scripts/score_arena_with_exclusions.py'
    assert path.is_file(), 'The exclusion-aware scorer is missing'
    return importlib.import_module('scripts.score_arena_with_exclusions')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(length=20, headers=0, lists=0, bold=0):
    return {'token_len': length,
            'header_count': {'h1': headers, **{f'h{i}': 0 for i in range(2, 7)}},
            'list_count': {'ordered': lists, 'unordered': 0},
            'bold_count': {'**': bold, '__': 0}}


@pytest.fixture
def small_run(tmp_path, request):
    # Exercise the real frozen record validator, including exact request hashes.
    judge = module_at('exclusion_test_frozen_judge', SOURCE_SUITE / 'source/scripts/judge_arena_hard.py')
    questions = [{'uid': f'q{i}', 'category': 'hard_prompt', 'prompt': f'Question {i}'} for i in range(500)]
    def answers(model):
        return [{'uid': q['uid'], 'model': model, 'metadata': metadata(), 'messages': [
            {'role': 'user', 'content': q['prompt']},
            {'role': 'assistant', 'content': {'answer': f'{model} answer {q["uid"]}'}}]} for q in questions]
    protocol = judge.load_protocol(SOURCE_SUITE / 'source/third_party/arena_hard',
                                   baseline_model=BASELINE, judge_model='gpt-4o')
    run = judge.JudgeRun(tmp_path / 'judgments', questions, answers(BASELINE),
                         {model: answers(model) for model in MODELS}, protocol)
    # Keep the pair selector fixture small. The suite-loading boundary separately
    # requires the production 500 questions and all 6000 terminal games.
    selected = {f'q{i}' for i in range(getattr(request, 'param', 3))}
    run.questions = {uid: row for uid, row in run.questions.items() if uid in selected}
    run.baseline = {uid: row for uid, row in run.baseline.items() if uid in run.questions}
    run.answers = {model: {uid: row for uid, row in rows.items() if uid in run.questions}
                   for model, rows in run.answers.items()}
    run.identity = {'protocol': protocol, 'questions_sha256': judge.digest(list(run.questions.values())),
                    'baseline_sha256': judge.digest(list(run.baseline.values()))}
    run.prepare()
    for model in MODELS:
        for uid in run.questions:
            for order in (0, 1):
                request = run.request(model, uid, order)
                path = run.game_path(model, uid, order)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'tag': model, 'uid': uid, 'order': order,
                    'status': 'valid', 'request': request, 'request_sha256': judge.digest(request),
                    'baseline_model': BASELINE, 'judge_model': 'gpt-4o', 'protocol_sha256': judge.digest(protocol),
                    'score': 'A>B', 'answer': 'Offline synthetic [[A>B]]', 'finish_reason': 'stop',
                    'finished_at': '2026-09-17T00:00:00+00:00',
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}))
    return run, judge


def change(run, model, uid, order, **updates):
    path = run.game_path(model, uid, order)
    row = json.loads(path.read_text())
    row.update(updates)
    path.write_text(json.dumps(row))
    run._records.clear()


def fail_game(run, model, uid, order):
    change(run, model, uid, order, status='invalid', score=None, answer='No verdict', finish_reason='stop')


def test_one_failed_order_discards_both_and_records_valid_partner(small_run):
    run, judge = small_run
    fail_game(run, 'base', 'q0', 0)
    bundle = scorer().collect_subsets(run, judge, importlib.import_module('scripts.score_arena_hard'))
    assert bundle.valid_uids['base'] == ['q1', 'q2']
    assert len(bundle.battles[bundle.battles.model == 'base']) == 4
    assert bundle.counts['base'] == {'attempted_games': 6, 'valid_games': 5, 'judge_failed_games': 1,
        'excluded_prompts': 1, 'valid_but_discarded_partner_games': 1,
        'retained_prompts': 2, 'retained_games': 4, 'expanded_rows': 4}
    assert [(r['uid'], r['model'], r['order'], r['reason']) for r in bundle.exclusions] == [
        ('q0', 'base', 0, 'missing_verdict'), ('q0', 'base', 1, 'partner_judge_failed')]


def test_model_denominators_differ_and_common_subset_is_full_intersection(small_run):
    run, judge = small_run
    fail_game(run, 'base', 'q0', 0)
    fail_game(run, 'grpo', 'q1', 1)
    fail_game(run, 'lam8', 'q0', 0)
    fail_game(run, 'lam8', 'q1', 1)
    bundle = scorer().collect_subsets(run, judge, importlib.import_module('scripts.score_arena_hard'))
    assert {m: r['retained_prompts'] for m, r in bundle.counts.items()} == {
        'base': 2, 'sft-init': 3, 'grpo': 2, 'lam2': 3, 'lam4': 3, 'lam8': 1}
    assert bundle.common_uids == ['q2']
    assert len(bundle.common_battles) == 12
    assert set(bundle.common_battles.uid) == {'q2'}


def test_manual_decisive_and_tie_weights_preserve_order_orientation(small_run):
    run, judge = small_run
    change(run, 'base', 'q0', 0, score='A>>B', answer='[[A>>B]]')
    change(run, 'base', 'q1', 0, score='A=B', answer='[[A=B]]')
    change(run, 'base', 'q1', 1, score='A=B', answer='[[A=B]]')
    api = scorer()
    strict = importlib.import_module('scripts.score_arena_hard')
    bundle = api.collect_subsets(run, judge, strict)
    rows = bundle.battles.query("model == 'base'")
    assert rows.scores.tolist() == [1, 0, 0, 0, .5, .5, 1, 0]
    result = api.score_subset(bundle.battles, bundle.answers, strict, upstream=strict.UPSTREAM, rounds=5)
    assert result['models']['base']['raw']['weighted_direct_mean'] == pytest.approx(3 / 8)
    assert result['models']['base']['expanded_rows'] == 8
    assert result['models']['base']['prompts'] == 3
    assert result['models']['base']['games'] == 6


@pytest.mark.parametrize('failure', ['missing', 'inflight', 'ambiguous', 'score_tamper',
                                     'request_tamper', 'identity_tamper', 'usage_tamper',
                                     'metadata_tamper', 'unknown_invalid_reason', 'extra_game',
                                     'boolean_order', 'coerced_request'])
def test_incomplete_or_corrupt_data_never_becomes_an_exclusion(small_run, failure):
    run, judge = small_run
    path = run.game_path('base', 'q0', 0)
    if failure == 'missing':
        path.unlink()
    elif failure in ('inflight', 'ambiguous'):
        change(run, 'base', 'q0', 0, status=failure)
    elif failure == 'score_tamper':
        change(run, 'base', 'q0', 0, score='A=B')
    elif failure == 'request_tamper':
        change(run, 'base', 'q0', 0, request={})
    elif failure == 'boolean_order':
        record = json.loads(path.read_text())
        record['order'] = False
        path.write_text(json.dumps(record))
    elif failure == 'coerced_request':
        record = json.loads(path.read_text())
        record['request']['temperature'] = False
        path.write_text(json.dumps(record))
    elif failure == 'identity_tamper':
        (run.directory / 'state/protocol.json').write_text('{}')
    elif failure == 'usage_tamper':
        change(run, 'base', 'q0', 0, status='invalid', score=None, answer='No verdict', usage=None)
    elif failure == 'metadata_tamper':
        run.answers['base']['q0']['metadata']['token_len'] = -1
        # Keep the answer identity consistent so metadata validation is exercised.
        path = run.directory / 'state/models/base.json'
        row = json.loads(path.read_text())
        row['answers_sha256'] = judge.digest(list(run.answers['base'].values()))
        path.write_text(json.dumps(row))
    elif failure == 'unknown_invalid_reason':
        change(run, 'base', 'q0', 0, status='invalid', finish_reason='content_filter')
    else:
        (path.parent / 'unexpected-0.json').write_text(path.read_text())
    with pytest.raises((ValueError, RuntimeError)):
        scorer().collect_subsets(run, judge, importlib.import_module('scripts.score_arena_hard'))


def test_empty_subset_and_constant_controls_have_explicit_unavailable_scores(small_run):
    run, judge = small_run
    strict = importlib.import_module('scripts.score_arena_hard')
    api = scorer()
    bundle = api.collect_subsets(run, judge, strict)
    result = api.score_subset(bundle.battles, bundle.answers, strict, upstream=strict.UPSTREAM, rounds=3)
    assert result['style_fit']['status'] == 'unavailable'
    assert 'constant' in result['style_fit']['reason']
    assert result['models']['base']['raw']['weighted_direct_mean'] == .5
    assert result['models']['base']['length_markdown_controlled']['status'] == 'unavailable'
    empty = api.score_subset(bundle.battles.iloc[:0], bundle.answers, strict,
                             upstream=strict.UPSTREAM, rounds=3)
    assert empty['models']['base']['raw']['status'] == 'unavailable'
    assert empty['models']['base']['prompts'] == 0


def test_programming_errors_are_not_reported_as_style_unavailability(small_run, monkeypatch):
    run, judge = small_run
    strict = importlib.import_module('scripts.score_arena_hard')
    api = scorer()
    bundle = api.collect_subsets(run, judge, strict)
    def bug(*args, **kwargs):
        raise RuntimeError('programming error')
    monkeypatch.setattr(strict, 'style_scores', bug)
    with pytest.raises(RuntimeError, match='programming error'):
        api.score_subset(bundle.battles, bundle.answers, strict, upstream=strict.UPSTREAM, rounds=3)


def test_cli_requires_exact_policy_and_fresh_destination_before_loading_suite(tmp_path):
    api = scorer()
    output = tmp_path / 'scores'
    output.mkdir()
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'policy_id': 'arena_judge_output_exclusions_v2',
                                  'judge': 'gpt-4o', 'baseline': BASELINE}))
    with pytest.raises(FileExistsError):
        api.main(['--suite', str(tmp_path), '--policy', str(policy), '--output', str(output)])
    output.rmdir()
    policy.write_text(json.dumps({'policy_id': 'wrong', 'judge': 'gpt-4o', 'baseline': BASELINE}))
    with pytest.raises(ValueError, match='policy'):
        api.main(['--suite', str(tmp_path), '--policy', str(policy), '--output', str(output)])
    assert not output.exists()


def test_small_numerical_fit_uses_frozen_math_and_retained_candidate_set(small_run):
    import numpy as np
    import torch
    run, judge = small_run
    api = scorer()
    strict = api.load_module(SOURCE_SUITE / 'source/scripts/score_arena_hard.py', 'test_numerical')
    bundle = api.collect_subsets(run, judge, strict)
    rng = np.random.RandomState(17)
    for rows in bundle.answers.values():
        for row in rows.values():
            row['metadata'] = metadata(int(rng.randint(5, 50)), int(rng.randint(0, 5)),
                                       int(rng.randint(0, 7)), int(rng.randint(0, 6)))
    torch.set_num_threads(1)
    result = api.score_subset(bundle.battles, bundle.answers, strict, upstream=strict.UPSTREAM, rounds=3)
    assert result['style_fit']['status'] == 'available'
    assert result == api.score_subset(bundle.battles, bundle.answers, strict,
                                      upstream=strict.UPSTREAM, rounds=3)
    for row in result['models'].values():
        assert row['raw']['weighted_direct_mean'] == .5
        assert 0 <= row['length_markdown_controlled']['official_bootstrap_median'] <= 1
        low, high = row['length_markdown_controlled']['ci90']
        assert 0 <= low <= high <= 1


@pytest.mark.parametrize('small_run', [500], indirect=True)
def test_cli_writes_exact_denominators_exclusions_and_secondary_table(small_run, tmp_path, monkeypatch):
    import csv
    import hashlib
    run, judge = small_run
    fail_game(run, 'base', 'q0', 0)
    fail_game(run, 'grpo', 'q1', 1)
    api = scorer()
    strict = api.load_module(SOURCE_SUITE / 'source/scripts/score_arena_hard.py', 'test_output')
    bundle = api.collect_subsets(run, judge, strict)
    # Suite reuse validation is tested at its frozen source. This boundary
    # substitution supplies fully validated temporary game files, without
    # copying the original generation's extensive provenance into pytest.
    monkeypatch.setattr(api, 'load_suite', lambda _: (bundle, strict, strict.UPSTREAM, {'status': 'passed'}))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'policy_id': 'arena_judge_output_exclusions_v2',
                                  'judge': 'gpt-4o', 'baseline': BASELINE}))
    output = tmp_path / 'scores'
    result = api.main(['--suite', str(tmp_path), '--policy', str(policy), '--output', str(output)])
    saved = json.loads((output / 'results.json').read_text())
    assert saved == result
    assert saved['status'] == 'complete_with_judge_exclusions'
    assert saved['bootstrap_rounds'] == 100 and saved['seed'] == 42
    assert saved['coverage']['attempted_games'] == 6000
    assert saved['coverage']['valid_games'] == 5998
    assert saved['coverage']['judge_failed_games'] == 2
    assert saved['coverage']['valid_but_discarded_partner_games'] == 2
    assert saved['models']['base']['prompts'] == 499
    assert saved['models']['lam8']['prompts'] == 500
    assert saved['common_valid_subset']['questions'] == 498
    manifest = output / 'exclusions.jsonl'
    assert saved['exclusion_manifest']['sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(manifest.read_text().splitlines()) == 4
    assert saved['inputs_sha256'][str(policy)] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert saved['full_500_official_leaderboard'] is False
    with (output / 'results.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert [row['model'] for row in table] == list(MODELS)
    assert [row['prompts'] for row in table] == ['499', '500', '499', '500', '500', '500']
    assert all(row['style_status'] == 'unavailable' and row['raw_status'] == 'available' for row in table)
    with (output / 'common_subset_results.csv').open() as stream:
        assert {row['prompts'] for row in csv.DictReader(stream)} == {'498'}


def test_changed_validated_input_aborts_before_any_success_output(small_run, tmp_path, monkeypatch):
    run, judge = small_run
    api = scorer()
    strict = importlib.import_module('scripts.score_arena_hard')
    bundle = api.collect_subsets(run, judge, strict)
    monkeypatch.setattr(api, 'load_suite', lambda _: (bundle, strict, strict.UPSTREAM, {'status': 'passed'}))
    # Simulate a concurrent modification after the immutable snapshot was read.
    change(run, 'base', 'q0', 0, answer='Changed after validation [[A>B]]')
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'policy_id': 'arena_judge_output_exclusions_v2',
                                  'judge': 'gpt-4o', 'baseline': BASELINE}))
    output = tmp_path / 'scores'
    with pytest.raises(ValueError, match='Input changed during scoring'):
        api.main(['--suite', str(tmp_path), '--policy', str(policy), '--output', str(output)])
    assert not output.exists()


def test_content_filter_removes_whole_pair_and_reports_reason(small_run):
    run, judge = small_run
    change(run, 'lam4', 'q0', 1, status='invalid', answer=None, score=None,
           finish_reason='content_filter')
    before = {p: p.read_bytes() for p in (run.directory / 'state/games').rglob('*.json')}
    bundle = scorer().collect_subsets(run, judge, importlib.import_module('scripts.score_arena_hard'))
    assert bundle.valid_uids['lam4'] == ['q1', 'q2']
    assert bundle.common_uids == ['q1', 'q2']
    assert [(r['uid'], r['model'], r['order'], r['reason']) for r in bundle.exclusions] == [
        ('q0', 'lam4', 0, 'partner_judge_failed'), ('q0', 'lam4', 1, 'judge_content_filter')]
    assert all(p.read_bytes() == data for p, data in before.items())
