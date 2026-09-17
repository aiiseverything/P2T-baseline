import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def scorer():
    module = Path(__file__).resolve().parents[1] / 'scripts/score_arena_hard.py'
    assert module.is_file(), 'Arena scorer implementation is missing'
    return importlib.import_module('scripts.score_arena_hard')


def metadata(length=10, headers=0, lists=0, bold=0):
    return {'token_len': length,
            'header_count': {'h1': headers, **{f'h{i}': 0 for i in range(2, 7)}},
            'list_count': {'ordered': lists, 'unordered': 0},
            'bold_count': {'**': bold, '__': 0}}


@pytest.mark.parametrize('labels,expected', [
    (('A>>B', 'A>B'), [1, 0, 0, 0]),
    (('A<B', 'A<<B'), [0, 0, 0, 1]),
    (('B>>A', 'B<A'), [1, 1, 1, 1]),
    (('A=B', 'B=A'), [.5, .5]),
])
def test_game_zero_baseline_orientation_and_decisive_triplication(labels, expected):
    assert scorer().expand_outcomes([{'score': x} for x in labels]) == expected


@pytest.mark.parametrize('games', [[], [{'score': 'A>B'}], [None, {'score': 'A>B'}],
                                     [{'score': None}, {'score': 'A>B'}],
                                     [{'score': 'unknown'}, {'score': 'A>B'}]])
def test_invalid_judgment_is_rejected_instead_of_dropped(games):
    with pytest.raises(ValueError):
        scorer().expand_outcomes(games)


def test_raw_bootstrap_is_expanded_row_resampling_not_question_resampling():
    # One decisive loss plus one ordinary win = [1, 0, 0, 0].
    # Seed 42's five replacement samples have means .25,.25,0,.25,.75.
    battles = pd.DataFrame({'uid': ['one'] * 4, 'model': ['toy'] * 4,
                            'scores': [0., 0., 0., 1.]})
    result = scorer().raw_scores(battles, seed=42, rounds=5)['toy']
    assert result['weighted_direct_mean'] == .25
    assert result['official_bootstrap_mean'] == pytest.approx(.30)
    assert result['ci90'] == pytest.approx([.05, .65])
    assert result['expanded_rows'] == 4


def test_metadata_order_does_not_swap_length_and_markdown_controls():
    candidate = metadata(3, 1, 2, 0)
    candidate = dict(reversed(list(candidate.items())))
    reference = metadata(1, 1, 0, 1)
    actual = scorer().style_contrast(candidate, reference)
    assert actual == pytest.approx([.5, -1 / 7, 1 / 3, -1 / 3])


@pytest.mark.parametrize('mutate', [
    lambda m: m.update(token_len=True),
    lambda m: m.update(token_len=-1),
    lambda m: m.update(token_len=float('nan')),
    lambda m: m['list_count'].update(ordered=-1),
    lambda m: m.pop('bold_count'),
])
def test_malformed_style_metadata_fails_closed(mutate):
    m = metadata(); mutate(m)
    with pytest.raises(ValueError):
        scorer().style_contrast(m, metadata())


def test_zero_length_denominator_fails_instead_of_nan():
    with pytest.raises(ValueError, match='length'):
        scorer().style_contrast(metadata(0), metadata(0))


def test_constant_control_is_not_silently_removed():
    battles = pd.DataFrame({'uid': ['x', 'y'], 'model': ['toy', 'toy'], 'scores': [0., 1.]})
    answers = {'toy': {'x': {'metadata': metadata()}, 'y': {'metadata': metadata()}},
               'o3-mini-2025-01-31': {'x': {'metadata': metadata()}, 'y': {'metadata': metadata()}}}
    with pytest.raises(ValueError, match='constant'):
        scorer().style_design(battles, answers)


@pytest.fixture
def official_layout(tmp_path):
    tags = ['base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8']
    baseline = 'o3-mini-2025-01-31'
    questions = [{'uid': f'q{i:03}', 'category': 'hard_prompt',
                  'prompt': f'Question {i}\u2028kept\u0085inside JSON'} for i in range(500)]
    answers = tmp_path / 'model_answer'; answers.mkdir()
    judgments = tmp_path / 'model_judgment'; judgments.mkdir()

    def write(path, rows):
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

    qp = tmp_path / 'questions.jsonl'; write(qp, questions)
    for model in [baseline, *tags]:
        write(answers / f'{model}.jsonl', [
            {'uid': q['uid'], 'model': model, 'metadata': metadata(),
             'messages': [{'role': 'user', 'content': q['prompt']},
                          {'role': 'assistant', 'content': {'answer': 'answer'}}]}
            for q in questions])
    for model in tags:
        write(judgments / f'{model}.jsonl', [
            {'uid': q['uid'], 'category': 'hard_prompt', 'model': model, 'judge': 'gpt-4.1',
             'baseline': baseline, 'games': [{'score': 'A>B'}, {'score': 'A>B'}]}
            for q in questions])
    return qp, answers, judgments, write


def test_exact_six_hard500_coverage_and_literal_unicode_jsonl(official_layout):
    q, a, j, _ = official_layout
    bundle = scorer().load_inputs(q, a, j)
    assert len(bundle.questions) == 500
    assert '\u2028' in bundle.questions[0]['prompt']
    assert '\u0085' in bundle.questions[0]['prompt']
    assert len(bundle.battles) == 6000
    assert set(bundle.answers) == {'base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8', 'o3-mini-2025-01-31'}


CUSTOM_BASELINE = 'gpt-4o-mini-2024-07-18'


def test_custom_baseline_cannot_score_old_or_only_relabelled_judgments(official_layout):
    q, a, j, write = official_layout
    with pytest.raises(ValueError):
        scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)
    old = a / f'{scorer().BASELINE}.jsonl'
    rows = scorer()._jsonl(old)
    for row in rows:
        row['model'] = CUSTOM_BASELINE
    write(a / f'{CUSTOM_BASELINE}.jsonl', rows)
    old.unlink()
    # Even altering exported baseline labels must not turn an o3 run into a
    # valid custom-reference run without its matching requests/provenance.
    for path in j.glob('*.jsonl'):
        rows = scorer()._jsonl(path)
        for row in rows:
            row['baseline'] = CUSTOM_BASELINE
        write(path, rows)
    with pytest.raises(ValueError, match='provenance'):
        scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)


def test_custom_reference_is_used_by_style_design_and_probability_conversion():
    api = scorer()
    battles = pd.DataFrame({'uid': [f'q{i}' for i in range(18)],
                            'model': ['a', 'b', 'c'] * 6,
                            'scores': [0., 1., .5, 1., 0., .5] * 3})
    rng = np.random.RandomState(51)
    answers = {m: {} for m in ['a', 'b', 'c', CUSTOM_BASELINE]}
    for row in battles.itertuples():
        for model in (row.model, CUSTOM_BASELINE):
            answers[model][row.uid] = {'metadata': metadata(int(rng.randint(5, 50)),
                int(rng.randint(0, 5)), int(rng.randint(0, 7)), int(rng.randint(0, 6)))}
    design = api.style_design(battles, answers, baseline_model=CUSTOM_BASELINE)
    assert design['models'] == ['a', 'b', 'c', CUSTOM_BASELINE]
    result = api.style_scores(battles, answers, baseline_model=CUSTOM_BASELINE, rounds=3)
    assert result['models'][CUSTOM_BASELINE]['official_bootstrap_median'] == .5
    # A mere baseline label change preserves the exact design and math.
    answers[api.BASELINE] = answers.pop(CUSTOM_BASELINE)
    original = api.style_scores(battles, answers, rounds=3)
    for model in ('a', 'b', 'c'):
        assert result['models'][model] == original['models'][model]


@pytest.fixture
def custom_layout(official_layout):
    from scripts import judge_arena_hard as judge
    q, a, j, write = official_layout
    old = a / f'{scorer().BASELINE}.jsonl'
    rows = scorer()._jsonl(old)
    for row in rows:
        row['model'] = CUSTOM_BASELINE
        row['messages'][-1]['content']['answer'] = 'new reference response ' + row['uid']
    write(a / f'{CUSTOM_BASELINE}.jsonl', rows)
    old.unlink()
    # Generate authentic record structures entirely offline. No Relay is
    # constructed and no synthetic record is written outside pytest tmp_path.
    rng = np.random.RandomState(71)
    for path in sorted(a.glob('*.jsonl')):
        varied = scorer()._jsonl(path)
        for row in varied:
            row['metadata'] = metadata(int(rng.randint(5, 60)), int(rng.randint(0, 5)),
                                       int(rng.randint(0, 6)), int(rng.randint(0, 7)))
        write(path, varied)
    rows = scorer()._jsonl(a / f'{CUSTOM_BASELINE}.jsonl')
    for path in j.glob('*.jsonl'):
        path.unlink()
    protocol = judge.load_protocol(judge.ROOT / 'third_party/arena_hard', baseline_model=CUSTOM_BASELINE)
    run = judge.JudgeRun(j, judge.load_jsonl(q), rows,
        {m: judge.load_jsonl(a / f'{m}.jsonl') for m in scorer().MODELS}, protocol)
    run.prepare()
    for tag in scorer().MODELS:
        for uid in run.questions:
            for order in (0, 1):
                request = run.request(tag, uid, order)
                path = run.game_path(tag, uid, order)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'tag': tag, 'uid': uid, 'order': order,
                    'status': 'valid', 'request': request, 'request_sha256': judge.digest(request),
                    'baseline_model': CUSTOM_BASELINE, 'protocol_sha256': judge.digest(protocol),
                    'score': 'A>B', 'answer': 'Offline synthetic verdict [[A>B]]', 'finish_reason': 'stop',
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}))
    run.export()
    return q, a, j, write, run


def test_custom_complete_inputs_require_actual_requests_and_settings(custom_layout):
    q, a, j, write, run = custom_layout
    bundle = scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)
    assert len(bundle.battles) == 6000
    assert len(bundle.files_sha256) == 6021  # 14 inputs + 7 identity + 6000 games.
    assert CUSTOM_BASELINE in bundle.answers and scorer().BASELINE not in bundle.answers
    with pytest.raises(ValueError, match='judge'):
        scorer().load_inputs(q, a, j, judge='another-judge', baseline_model=CUSTOM_BASELINE)
    # Changing verdict labels without changing the original recorded response
    # fails even though the exported labels are otherwise syntactically valid.
    path = j / 'base.jsonl'
    rows = scorer()._jsonl(path)
    rows[0]['games'][0]['score'] = 'A=B'
    write(path, rows)
    with pytest.raises(ValueError, match='export differs'):
        scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)
    # Restore then alter the bound run settings instead of the visible labels.
    rows[0]['games'][0]['score'] = 'A>B'; write(path, rows)
    state = j / 'state/protocol.json'
    identity = json.loads(state.read_text())
    identity['protocol']['max_tokens'] = 100
    state.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='provenance'):
        scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)


def test_custom_rejects_old_game_records_even_with_identical_answer_text(custom_layout):
    q, a, j, _, run = custom_layout
    path = run.game_path('base', 'q000', 0)
    record = json.loads(path.read_text())
    record.pop('baseline_model'); record.pop('protocol_sha256')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='game identity'):
        scorer().load_inputs(q, a, j, baseline_model=CUSTOM_BASELINE)


def test_custom_cli_scores_real_math_and_marks_nonofficial_reference(custom_layout):
    q, a, j, _, _ = custom_layout
    output = q.parent / 'custom-scores'
    result = scorer().main(['--questions', str(q), '--answers-dir', str(a), '--judgments-dir', str(j),
                            '--output', str(output), '--baseline-model', CUSTOM_BASELINE])
    assert result['baseline'] == CUSTOM_BASELINE
    assert result['official_baseline_model'] == scorer().BASELINE
    assert result['uses_official_baseline'] is False
    assert result['protocol'] == 'arena_hard_v2_custom_baseline_combined_controls_v1'
    assert result['coverage']['total_games'] == 6000
    assert len(result['inputs_sha256']) == 6021
    for row in result['models'].values():
        assert row['raw']['weighted_direct_mean'] == .5
        assert .4 < row['length_markdown_controlled']['official_bootstrap_median'] < .6


@pytest.mark.parametrize('failure', ['missing_model', 'missing_uid', 'duplicate_uid', 'wrong_baseline',
                                     'wrong_model', 'wrong_judge', 'invalid_game', 'extra_model',
                                     'wrong_category', 'answer_prompt_mismatch'])
def test_incomplete_or_misidentified_suites_fail_before_scoring(official_layout, failure):
    q, a, j, write = official_layout
    p = j / 'lam8.jsonl'
    with p.open() as f:
        rows = [json.loads(line) for line in f]
    if failure == 'missing_model': p.unlink()
    elif failure == 'missing_uid': write(p, rows[:-1])
    elif failure == 'duplicate_uid': write(p, rows[:-1] + [rows[0]])
    elif failure == 'wrong_baseline': rows[0]['baseline'] = 'other'; write(p, rows)
    elif failure == 'wrong_model': rows[0]['model'] = 'grpo'; write(p, rows)
    elif failure == 'wrong_judge': rows[0]['judge'] = 'other'; write(p, rows)
    elif failure == 'invalid_game': rows[0]['games'][0]['score'] = None; write(p, rows)
    elif failure == 'extra_model': write(j / 'unrequested.jsonl', rows)
    elif failure == 'wrong_category': rows[0]['category'] = 'creative_writing'; write(p, rows)
    elif failure == 'answer_prompt_mismatch':
        p = a / 'base.jsonl'
        with p.open() as f: rows = [json.loads(line) for line in f]
        rows[0]['messages'][0]['content'] = 'different question'; write(p, rows)
    with pytest.raises(ValueError):
        scorer().load_inputs(q, a, j)


def test_style_fit_matches_pinned_official_combined_math_and_is_reproducible():
    torch.set_num_threads(1)
    api = scorer()
    battles = pd.DataFrame({'uid': [f'q{i}' for i in range(18)],
                            'model': ['a', 'b', 'c'] * 6,
                            'scores': [0., 1., .5, 1., 0., .5] * 3})
    rng = np.random.RandomState(5)
    answers = {m: {} for m in ['a', 'b', 'c', api.BASELINE]}
    for row in battles.itertuples():
        for model in (row.model, api.BASELINE):
            answers[model][row.uid] = {'metadata': metadata(int(rng.randint(5, 50)),
                  int(rng.randint(0, 5)), int(rng.randint(0, 7)), int(rng.randint(0, 6)))}
    design = api.style_design(battles, answers)
    assert design['models'] == ['a', 'b', 'c', api.BASELINE]
    torch.testing.assert_close(design['features'][:3, :4],
                               torch.tensor([[1., 0., 0., -1.], [0., 1., 0., -1.], [0., 0., 1., -1.]]))
    torch.testing.assert_close(design['features'][:, -4:].mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(design['features'][:, -4:].std(0), torch.ones(4), atol=1e-6, rtol=0)
    upstream = api.load_official_math()
    np.random.seed(42)
    coefs, _ = upstream.bootstrap_pairwise_model(design['features'], torch.tensor(battles.scores.tolist()),
                                                 loss_type='bt', num_round=3)
    expected = upstream.to_winrate_probabilities(coefs[:, :-4], design['models'], api.BASELINE)
    result = api.style_scores(battles, answers, seed=42, rounds=3)
    for index, model in enumerate(design['models']):
        # Official pandas quantiles interpolate; check exact chosen columns.
        values = pd.Series(expected[:, index].tolist())
        assert result['models'][model]['official_bootstrap_median'] == pytest.approx(float(values.quantile(.5)), abs=1e-8)
        assert result['models'][model]['ci90'] == pytest.approx(values.quantile([.05, .95]).tolist())
    assert result == api.style_scores(battles, answers, seed=42, rounds=3)


def test_changed_upstream_source_is_rejected_even_after_module_cache(tmp_path):
    api = scorer()
    api.load_official_math()
    for name in api.PINNED_HASHES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((api.UPSTREAM / name).read_bytes())
    api.verify_upstream(tmp_path)
    with (tmp_path / 'utils/math_utils.py').open('a') as stream:
        stream.write('\n# unreviewed change\n')
    with pytest.raises(ValueError, match='Pinned official'):
        api.load_official_math(tmp_path)


def test_cli_complete_real_bootstrap_outputs_and_refuses_overwrite(official_layout):
    import csv

    api = scorer()
    q, answers, judgments, write = official_layout
    # Independent style variation makes all four controls estimable. Each
    # identical-feature pair has one win and one loss, so no model dominates.
    rng = np.random.RandomState(71)
    for path in sorted(answers.glob('*.jsonl')):
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            row['metadata'] = metadata(int(rng.randint(5, 60)), int(rng.randint(0, 5)),
                                       int(rng.randint(0, 6)), int(rng.randint(0, 7)))
        write(path, rows)
    output = q.parent / 'scores'
    argv = ['--questions', str(q), '--answers-dir', str(answers),
            '--judgments-dir', str(judgments), '--output', str(output)]
    actual = api.main(argv)
    assert actual == json.loads((output / 'results.json').read_text())
    assert actual['coverage']['total_games'] == 6000
    assert actual['coverage']['invalid_dropped'] == 0
    assert len(actual['inputs_sha256']) == 14
    assert actual['math']['upstream_files_sha256'] == api.PINNED_HASHES
    assert actual['bootstrap_rounds'] == 100
    with (output / 'results.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert [row['model'] for row in table] == list(api.MODELS)
    for row in table:
        model = actual['models'][row['model']]
        assert float(row['raw_weighted_direct_pct']) == 50
        assert float(row['raw_official_bootstrap_pct']) == model['raw']['official_bootstrap_mean'] * 100
        assert float(row['length_markdown_controlled_pct']) == model['length_markdown_controlled']['official_bootstrap_median'] * 100
        assert 40 < float(row['controlled_ci90_low_pct']) < float(row['controlled_ci90_high_pct']) < 60
    with pytest.raises(FileExistsError):
        api.main(argv)


def test_cli_invalid_suite_leaves_no_success_output(official_layout):
    q, answers, judgments, _ = official_layout
    (judgments / 'lam8.jsonl').unlink()
    output = q.parent / 'scores'
    with pytest.raises(ValueError):
        scorer().main(['--questions', str(q), '--answers-dir', str(answers),
                       '--judgments-dir', str(judgments), '--output', str(output)])
    assert not output.exists()
