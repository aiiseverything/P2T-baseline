"""Strict six-model Arena-Hard v2 aggregation using the pinned official math.

No model inference or judge requests are performed. Official confidence
intervals resample expanded game rows, not independent prompt clusters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, file_hash

MODELS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
BASELINE = 'o3-mini-2025-01-31'
CATEGORY = 'hard_prompt'
UPSTREAM = ROOT / 'third_party/arena_hard'
REVISION = '196f6b826783b3da7310e361a805fa36f0be83f3'
PINNED_HASHES = {
    'show_result.py': '5da0fa30a807a150da37e97811c2bc248959015eedb884d9d471e894c8bf90c6',
    'utils/math_utils.py': 'c2e927dc0f45771d9b387c29837c76c2b5008b1f6f81819259c288c01f49ade0',
    'utils/add_markdown_info.py': 'a34f3cc457bb35b6ef66854b55359c8669b1c7307c466230cce1ebfa28ab54be',
    'gen_answer.py': '929a2362ad0c55c83a897d0d099c5ef5c77be7295e7eb682bb0df33c1e016d20',
    'utils/judge_utils.py': 'cbb093f4fc27408d8447e946392f26f98538b1e3950f574aca91a0f7595167b4',
}
STYLE_FIELDS = ('token_len', 'header_count', 'list_count', 'bold_count')
STYLE_KEYS = {'header_count': {f'h{i}' for i in range(1, 7)},
              'list_count': {'ordered', 'unordered'}, 'bold_count': {'**', '__'}}
LABEL_SCORES = {'A>B': [1], 'A>>B': [1] * 3, 'A=B': [.5], 'A<<B': [0] * 3,
                'A<B': [0], 'B>A': [0], 'B>>A': [0] * 3, 'B=A': [.5],
                'B<<A': [1] * 3, 'B<A': [1]}


@dataclass
class Inputs:
    questions: list
    answers: dict
    battles: pd.DataFrame
    files_sha256: dict


def expand_outcomes(games):
    """Game 0: reference=A; game 1: candidate=A. Retain official row order."""
    if not isinstance(games, list) or len(games) != 2:
        raise ValueError('Each judgment must contain exactly two ordered games')
    for game in games:
        if not isinstance(game, dict) or not isinstance(game.get('score'), str) or game['score'] not in LABEL_SCORES:
            raise ValueError('Missing or invalid judgment score; no invalid rows may be dropped')
    return LABEL_SCORES[games[1]['score']] + [1 - x for x in LABEL_SCORES[games[0]['score']]]


def _metadata_values(metadata):
    if not isinstance(metadata, dict) or set(metadata) != set(STYLE_FIELDS):
        raise ValueError('Style metadata requires exactly token_len/header_count/list_count/bold_count')
    length = metadata['token_len']
    if type(length) is not int or length < 0:
        raise ValueError('Style token length must be a nonnegative integer')
    values = [length]
    for key in STYLE_FIELDS[1:]:
        counts = metadata[key]
        if not isinstance(counts, dict) or set(counts) != STYLE_KEYS[key]:
            raise ValueError(f'Invalid {key} metadata')
        if any(type(x) is not int or x < 0 for x in counts.values()):
            raise ValueError(f'Invalid nonnegative counts in {key}')
        values.append(sum(counts.values()))
    return values


def style_contrast(candidate, baseline):
    """Official combined length and three Markdown density contrasts."""
    left, right = _metadata_values(candidate), _metadata_values(baseline)
    if left[0] + right[0] == 0:
        raise ValueError('Both style token lengths are zero')
    values = [(left[0] - right[0]) / (left[0] + right[0])]
    for x, y in zip(left[1:], right[1:]):
        a, b = x / (left[0] + 1), y / (right[0] + 1)
        values.append((a - b) / (a + b + 1))
    return values


def _jsonl(path):
    rows = []
    # File iteration splits on physical newlines, preserving literal U+2028/U+0085.
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected an object at {path}:{line_number}')
            rows.append(row)
    return rows


def _by_uid(rows, expected, where):
    uids = [row.get('uid') for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in uids):
        raise ValueError(f'Invalid uid in {where}')
    if len(uids) != len(set(uids)):
        raise ValueError(f'Duplicate uid in {where}')
    if set(uids) != expected:
        raise ValueError(f'Incomplete or extra uid coverage in {where}')
    return dict(zip(uids, rows))


def load_inputs(questions_path, answers_dir, judgments_dir, judge='gpt-4.1', *,
                baseline_model=BASELINE, upstream=UPSTREAM):
    from scripts.judge_arena_hard import validate_baseline_model
    validate_baseline_model(baseline_model)
    if baseline_model in MODELS:
        raise ValueError('Candidate and baseline model identities must differ')
    questions_path, answers_dir, judgments_dir = map(Path, (questions_path, answers_dir, judgments_dir))
    questions = _jsonl(questions_path)
    if len(questions) != 500 or any(q.get('category') != CATEGORY or not isinstance(q.get('prompt'), str)
                                    or not q['prompt'].strip() for q in questions):
        raise ValueError('Exactly 500 hard_prompt questions are required')
    question_map = _by_uid(questions, {q.get('uid') for q in questions}, questions_path)
    uids = set(question_map)
    if len(uids) != 500:
        raise ValueError('Exactly 500 unique question uids are required')
    for directory, names in [(answers_dir, (*MODELS, baseline_model)), (judgments_dir, MODELS)]:
        if {p.name for p in directory.glob('*.jsonl')} != {f'{name}.jsonl' for name in names}:
            raise ValueError(f'Expected exactly the requested model files in {directory}')
    answers, files = {}, {str(questions_path.resolve()): file_hash(questions_path)}
    for model in (*MODELS, baseline_model):
        path = answers_dir / f'{model}.jsonl'
        answers[model] = _by_uid(_jsonl(path), uids, path)
        files[str(path.resolve())] = file_hash(path)
        for uid, row in answers[model].items():
            if row.get('model') != model:
                raise ValueError(f'Answer model mismatch in {path}')
            messages = row.get('messages')
            if not isinstance(messages, list) or len(messages) < 2 or any(not isinstance(m, dict) for m in messages):
                raise ValueError(f'Invalid answer messages in {path}')
            user = [m for m in messages if m.get('role') == 'user']
            last = messages[-1]
            if (not user or user[-1].get('content') != question_map[uid]['prompt']
                    or last.get('role') != 'assistant' or not isinstance(last.get('content'), dict)
                    or not isinstance(last['content'].get('answer'), str)):
                raise ValueError(f'Answer prompt or response mismatch in {path}, uid={uid}')
            _metadata_values(row.get('metadata'))
    custom_run = None
    if baseline_model != BASELINE:
        # A new reference requires actual new requests, not relabelled exports.
        # Validate the frozen run settings, input identities and each request
        # without calling prepare(), which would create missing provenance.
        from scripts.judge_arena_hard import JudgeRun, load_protocol
        protocol = load_protocol(upstream, baseline_model=baseline_model)
        if judge != protocol['judge']:
            raise ValueError('Custom baseline judge does not match the pinned run settings')
        custom_run = JudgeRun(judgments_dir, questions, list(answers[baseline_model].values()),
                              {m: list(answers[m].values()) for m in MODELS}, protocol)
        expected = {judgments_dir / 'state/protocol.json': custom_run.identity}
        from scripts.judge_arena_hard import digest
        expected.update({judgments_dir / 'state/models' / f'{m}.json': {
            'answers_sha256': digest(list(answers[m].values())), 'tag': m, 'model': m} for m in MODELS})
        for path, identity in expected.items():
            if not path.is_file() or json.loads(path.read_text()) != identity:
                raise ValueError(f'Custom baseline run provenance mismatch or missing: {path}')
            files[str(path.resolve())] = file_hash(path)
    battles = []
    for model in MODELS:
        path = judgments_dir / f'{model}.jsonl'
        judgments = _by_uid(_jsonl(path), uids, path)
        files[str(path.resolve())] = file_hash(path)
        for uid, row in judgments.items():
            if (row.get('model') != model or row.get('baseline') != baseline_model
                    or row.get('category') != CATEGORY or row.get('judge') != judge):
                raise ValueError(f'Judgment identity mismatch in {path}, uid={uid}')
            if custom_run is not None:
                expected_games = []
                for order in (0, 1):
                    record = custom_run.load_record(model, uid, order)
                    if record is None or record['status'] != 'valid':
                        raise ValueError(f'Custom baseline valid game provenance missing: {model}/{uid}/{order}')
                    expected_games.append({'score': record['score'],
                        'judgment': {'answer': record['answer']}, 'prompt': record['request']['messages']})
                    game_path = custom_run.game_path(model, uid, order)
                    files[str(game_path.resolve())] = file_hash(game_path)
                if row.get('games') != expected_games:
                    raise ValueError(f'Custom baseline export differs from paid game provenance: {model}/{uid}')
            battles.extend({'uid': uid, 'model': model, 'scores': value}
                           for value in expand_outcomes(row.get('games')))
    return Inputs(questions, answers, pd.DataFrame(battles), files)


def verify_upstream(directory=UPSTREAM):
    directory = Path(directory)
    for name, expected in PINNED_HASHES.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Pinned official Arena math/source changed: {name}')
    return directory


def load_official_math(directory=UPSTREAM):
    directory = verify_upstream(directory)
    name = '_arena_hard_pinned_math_' + PINNED_HASHES['utils/math_utils.py'][:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / 'utils/math_utils.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


@contextmanager
def _seeded(seed, rounds):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(rounds) is not int or rounds < 1:
        raise ValueError('Invalid bootstrap seed or number of rounds')
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def raw_scores(battles, *, seed=42, rounds=100):
    # This follows show_result.print_leaderboard, including its unusual center:
    # the mean of the bootstrap means rather than the empirical weighted mean.
    frame = battles[['model', 'scores']].copy()
    with _seeded(seed, rounds):
        bootstrap = pd.concat([frame.groupby('model').sample(frac=1.0, replace=True)
                              .groupby('model').mean() for _ in range(rounds)])
    bootstrap['scores'] = bootstrap['scores'].astype(float)
    result = {}
    for model, rows in frame.groupby('model'):
        scores = bootstrap.loc[[model], 'scores']
        result[model] = {'weighted_direct_mean': float(rows.scores.mean()),
                         'official_bootstrap_mean': float(scores.mean()),
                         'ci90': scores.quantile([.05, .95]).tolist(),
                         'expanded_rows': len(rows)}
    return result


def style_design(battles, answers, *, upstream=UPSTREAM, baseline_model=BASELINE):
    official = load_official_math(upstream)
    # Match the upstream FP32 tensor arithmetic and its default sample std.
    left = torch.tensor([_metadata_values(answers[row.model][row.uid]['metadata'])
                         for row in battles.itertuples()], dtype=torch.float32)
    right = torch.tensor([_metadata_values(answers[baseline_model][row.uid]['metadata'])
                          for row in battles.itertuples()], dtype=torch.float32)
    if ((left[:, 0] + right[:, 0]) <= 0).any():
        raise ValueError('Both style token lengths are zero')
    contrast = torch.zeros_like(left)
    contrast[:, 0] = (left[:, 0] - right[:, 0]) / (left[:, 0] + right[:, 0])
    a, b = left[:, 1:] / (left[:, :1] + 1), right[:, 1:] / (right[:, :1] + 1)
    contrast[:, 1:] = (a - b) / (a + b + 1)
    mean, std = contrast.mean(0), contrast.std(0)
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('A style control is constant or invalid; refusing to drop a control')
    normalized = (contrast - mean) / std
    one_hot, models = official.one_hot_encode(battles.model.tolist(), baseline=baseline_model)
    features = torch.cat([one_hot, normalized], dim=1)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite combined style design')
    return {'features': features, 'models': models, 'contrast_mean': mean.tolist(),
            'contrast_std': std.tolist()}


def style_scores(battles, answers, *, seed=42, rounds=100, upstream=UPSTREAM, baseline_model=BASELINE):
    official = load_official_math(upstream)
    design = style_design(battles, answers, upstream=upstream, baseline_model=baseline_model)
    outcomes = torch.tensor(battles.scores.tolist())
    with _seeded(seed, rounds):
        coefs, _ = official.bootstrap_pairwise_model(design['features'], outcomes,
                                                    loss_type='bt', num_round=rounds)
    probabilities = official.to_winrate_probabilities(coefs[:, :-4], design['models'], baseline_model)
    if not torch.isfinite(coefs).all() or not torch.isfinite(probabilities).all():
        raise ValueError('Nonfinite official style bootstrap result')
    table = pd.DataFrame(probabilities.tolist(), columns=design['models'])
    return {'models': {model: {'official_bootstrap_median': float(table[model].quantile(.5)),
                               'ci90': table[model].quantile([.05, .95]).tolist()}
                       for model in design['models']},
            'controls': list(STYLE_FIELDS), 'contrast_mean': design['contrast_mean'],
            'contrast_std': design['contrast_std'],
            'control_coefficient_medians': torch.quantile(coefs[:, -4:], .5, axis=0).tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--answers-dir', type=Path, required=True)
    parser.add_argument('--judgments-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh scoring output directory')
    parser.add_argument('--judge', default='gpt-4.1')
    parser.add_argument('--baseline-model', default=BASELINE,
                        help='Exact reference identity; custom baselines require matching per-game run provenance')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Use a fresh scoring output: {args.output}')
    if args.threads < 1:
        parser.error('--threads must be positive')
    verify_upstream(args.upstream)
    bundle = load_inputs(args.questions, args.answers_dir, args.judgments_dir, args.judge,
                         baseline_model=args.baseline_model, upstream=args.upstream)
    torch.set_num_threads(args.threads)
    raw = raw_scores(bundle.battles, seed=args.seed)
    style = style_scores(bundle.battles, bundle.answers, seed=args.seed, upstream=args.upstream,
                          baseline_model=args.baseline_model)
    result = {
        'status': 'complete', 'protocol': 'arena_hard_v2_official_combined_controls_v1',
        'judge': args.judge, 'baseline': args.baseline_model, 'category': CATEGORY,
        'coverage': {'candidates': list(MODELS), 'prompts_per_candidate': 500,
                     'games_per_candidate': 1000, 'total_judgments': 3000,
                     'total_games': 6000, 'invalid_dropped': 0},
        'seed': args.seed, 'bootstrap_rounds': 100, 'confidence_level': .90,
        'models': {model: {'raw': raw[model], 'length_markdown_controlled': style['models'][model]}
                   for model in MODELS},
        'style_fit': {key: value for key, value in style.items() if key != 'models'},
        'math': {'upstream_revision': REVISION, 'upstream_files_sha256': PINNED_HASHES,
                 'scorer_sha256': file_hash(__file__),
                 'raw_point': 'Direct decisive-weighted mean is prominent; official central score is mean of 100 bootstrap means',
                 'raw_bootstrap_unit': 'Expanded game rows within each candidate separately',
                 'style_bootstrap_unit': 'Expanded game rows pooled across all six candidates',
                 'decisive_weight': 3, 'tie_score': .5, 'style_point': 'Median of 100 bootstrap fitted win probabilities',
                 'ci': '5th and 95th percentiles with pandas linear interpolation',
                 'style_normalization': 'All expanded rows, FP32, sample standard deviation (correction=1)',
                 'style_reference': 'Normalized controls set to zero, i.e. pooled observed mean contrast, not necessarily zero raw contrast',
                 'optimizer': 'Official unregularized BT logistic loss; LBFGS lr=.1, max_iter=50, tolerance=1e-9',
                 'style_length': 'Official answer metadata token_len: gpt-4o tiktoken/o200k_base; never actor generation token counts',
                 'markdown': 'Official header/list/bold counts after upstream code-fence-content removal'},
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'torch')},
        'inputs_sha256': bundle.files_sha256,
        'caveats': ['Official row bootstrap treats both answer orders and decisive replications as resampling units; it is not prompt-cluster uncertainty.',
                    'Style regression is fitted jointly to these six candidates, so it is candidate-set dependent and is not a causal estimate.',
                    'Official answer metadata are structurally validated and hash-bound; this scorer does not recompute tokenizer or Markdown metadata from text.',
                    'No API calls, new judgments, or GPU model inference are performed by this scorer.'],
    }
    if args.baseline_model != BASELINE:
        result.update(protocol='arena_hard_v2_custom_baseline_combined_controls_v1',
                      official_baseline_model=BASELINE, uses_official_baseline=False)
        result['caveats'].append('The selected reference differs from the official o3-mini baseline; '
                                 'these win rates are a custom-baseline evaluation, not official leaderboard scores.')
    stream = io.StringIO()
    columns = ['model', 'raw_weighted_direct_pct', 'raw_official_bootstrap_pct', 'raw_ci90_low_pct',
               'raw_ci90_high_pct', 'length_markdown_controlled_pct', 'controlled_ci90_low_pct',
               'controlled_ci90_high_pct', 'prompts', 'games', 'expanded_rows']
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for model in MODELS:
        r, s = raw[model], style['models'][model]
        writer.writerow(dict(zip(columns, [model, r['weighted_direct_mean'] * 100,
            r['official_bootstrap_mean'] * 100, *[x * 100 for x in r['ci90']],
            s['official_bootstrap_median'] * 100, *[x * 100 for x in s['ci90']], 500, 1000,
            r['expanded_rows']])))
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_text(args.output / 'results.json', json.dumps(result, indent=2, allow_nan=False))
    atomic_text(args.output / 'results.csv', stream.getvalue())
    print(stream.getvalue(), end='')
    return result


if __name__ == '__main__':
    main()
