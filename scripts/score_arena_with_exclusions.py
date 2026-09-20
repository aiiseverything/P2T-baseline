#!/usr/bin/env python3
"""Score a fully attempted custom GPT-4o Arena run under an explicit exclusion policy.

Only complete, valid two-order pairs contribute. The frozen strict scorer and
pinned upstream math are reused unchanged; these are custom-judge exclusion
results, not full-500 official leaderboard scores. No judge/API calls occur.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import arena_exclusion_policy as exclusion_policy

MODELS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
BASELINE = 'gpt-4o-mini-2024-07-18'
JUDGE = 'gpt-4o'
PROTOCOL = 'arena_hard_v2_custom_judge_exclusions_v2'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_module(path, label):
    path = Path(path).resolve()
    name = '_arena_exclusions_' + label + '_' + file_hash(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass
class Subsets:
    answers: dict
    battles: pd.DataFrame
    common_battles: pd.DataFrame
    valid_uids: dict
    common_uids: list
    exclusions: list
    counts: dict
    files_sha256: dict


def collect_subsets(run, judge, strict):
    """Validate every current record before selecting whole two-order pairs."""
    if (tuple(run.answers) != MODELS or run.protocol.get('judge') != JUDGE
            or run.baseline_model != BASELINE):
        raise ValueError('Unexpected candidate/judge/baseline identity')
    identities = {run.directory / 'state/protocol.json': run.identity}
    identities.update({run.directory / 'state/models' / f'{model}.json': {
        'answers_sha256': judge.digest(list(run.answers[model].values())),
        'tag': model, 'model': model} for model in MODELS})
    hashes = {}
    for path, expected in identities.items():
        if not path.is_file() or judge.digest(read_json(path)) != judge.digest(expected):
            raise ValueError(f'Missing or corrupt run identity: {path}')
        hashes[str(path.resolve())] = file_hash(path)
    answers = {**run.answers, BASELINE: run.baseline}
    for rows in answers.values():
        for row in rows.values():
            strict._metadata_values(row.get('metadata'))
    expected_paths = {run.game_path(model, uid, order) for model in MODELS
                      for uid in run.questions for order in (0, 1)}
    if set((run.directory / 'state/games').rglob('*.json')) != expected_paths:
        raise ValueError('Missing or extra game files: all expected games must be terminal')
    run._records.clear()
    battles, exclusions, valid_uids, counts = [], [], {}, {}
    for model in MODELS:
        valid_uids[model] = []
        count = {'attempted_games': 2 * len(run.questions), 'valid_games': 0,
                 'judge_failed_games': 0, 'excluded_prompts': 0,
                 'valid_but_discarded_partner_games': 0, 'retained_prompts': 0,
                 'retained_games': 0, 'expanded_rows': 0}
        for uid in run.questions:
            records, states = [], []
            for order in (0, 1):
                path = run.game_path(model, uid, order)
                before = file_hash(path)
                # The frozen validator binds request text/hash, all identities,
                # protocol, score parsing and valid-game response metadata.
                record = run.load_record(model, uid, order)
                if (record is not None and (type(record.get('order')) is not int
                        or judge.digest(record.get('request')) != record.get('request_sha256'))):
                    raise ValueError(f'Corrupt game order or request digest: {model}/{uid}/{order}')
                if file_hash(path) != before:
                    raise ValueError(f'Game changed during validation: {path}')
                hashes[str(path.resolve())] = before
                state = exclusion_policy.classify_record(record, judge)
                if state not in ('valid', 'judge_failed'):
                    raise ValueError(f'Nonterminal or corrupt game ({state}): {model}/{uid}/{order}')
                records.append(record)
                states.append(state)
                count['valid_games' if state == 'valid' else 'judge_failed_games'] += 1
            if 'judge_failed' in states:
                count['excluded_prompts'] += 1
                count['valid_but_discarded_partner_games'] += states.count('valid')
                for order, (record, state) in enumerate(zip(records, states)):
                    reason = (exclusion_policy.judge_failure_reason(record, judge)
                              if state == 'judge_failed' else 'partner_judge_failed')
                    exclusions.append({'uid': uid, 'model': model, 'order': order,
                        'reason': reason, 'record_classification': state,
                        'finish_reason': record['finish_reason'], 'score': record['score'],
                        'game_sha256': hashes[str(run.game_path(model, uid, order).resolve())]})
                continue
            valid_uids[model].append(uid)
            outcomes = strict.expand_outcomes([{'score': record['score']} for record in records])
            battles.extend({'uid': uid, 'model': model, 'scores': score} for score in outcomes)
            count['expanded_rows'] += len(outcomes)
        count['retained_prompts'] = len(valid_uids[model])
        count['retained_games'] = 2 * count['retained_prompts']
        counts[model] = count
    frame = pd.DataFrame(battles, columns=['uid', 'model', 'scores'])
    common = set.intersection(*(set(valid_uids[model]) for model in MODELS))
    common_uids = [uid for uid in run.questions if uid in common]
    return Subsets(answers, frame, frame[frame.uid.isin(common)].copy(), valid_uids,
                   common_uids, exclusions, counts, hashes)


def score_subset(battles, answers, strict, *, upstream, rounds=100):
    """Run unchanged official math, preserving explicit missing-score states."""
    raw = strict.raw_scores(battles, seed=42, rounds=rounds) if len(battles) else {}
    style = {'status': 'unavailable', 'reason': 'No complete valid pairs'}
    if len(battles):
        try:
            style = dict(strict.style_scores(battles, answers, seed=42, rounds=rounds,
                                             upstream=upstream, baseline_model=BASELINE), status='available')
        except ValueError as exc:
            # These are declared data-dependent failures in the frozen scorer.
            # Any other exception, including programming errors, must propagate.
            if str(exc) not in ('A style control is constant or invalid; refusing to drop a control',
                                'Both style token lengths are zero', 'Nonfinite combined style design',
                                'Nonfinite official style bootstrap result'):
                raise
            style = {'status': 'unavailable', 'reason': str(exc)}
    models = {}
    for model in MODELS:
        rows = battles[battles.model == model]
        prompts = int(rows.uid.nunique())
        controlled = style.get('models', {}).get(model)
        if controlled is None:
            controlled = {'status': 'unavailable',
                          'reason': style.get('reason', 'No complete valid pairs for this model')}
        models[model] = {'prompts': prompts, 'games': 2 * prompts, 'expanded_rows': len(rows),
            'raw': raw.get(model, {'status': 'unavailable', 'reason': 'No complete valid pairs'}),
            'length_markdown_controlled': controlled}
    return {'models': models, 'style_fit': {k: v for k, v in style.items() if k != 'models'}}


def load_suite(suite):
    """Read and verify the original frozen suite; never prepare/export/mutate it."""
    suite = Path(suite).resolve()
    verifier = load_module(suite / 'run_evaluation.py', 'reuse')
    reuse = verifier.verify_reused_generation(suite)
    if reuse.get('status') != 'passed':
        raise ValueError('Frozen generation reuse verification did not pass')
    continuation = load_module(suite / 'continue_evaluation.py', 'continuation')
    judge = continuation.judge_module(suite)
    run = continuation.load_judge_run(suite)
    strict = load_module(suite / 'source/scripts/score_arena_hard.py', 'strict_score')
    upstream = suite / 'source/third_party/arena_hard'
    strict.verify_upstream(upstream)
    if len(run.questions) != 500 or tuple(run.answers) != MODELS:
        raise ValueError('Exactly 500 questions and the six fixed candidates are required')
    experiment = read_json(suite / 'experiment.json')
    names = {'experiment.json', 'run_evaluation.py', 'continue_evaluation.py', 'question.jsonl',
             'source/scripts/score_arena_hard.py', 'source/scripts/judge_arena_hard.py',
             'reference/provenance.json', 'reference/audit.json'}
    names.update(experiment['files_sha256'])
    for name in experiment['generation_reuse']['original_files_sha256']:
        names.add('original_generation/' + name if name in
                  ('experiment.json', 'evaluation_complete.json', 'generation_complete.json') else name)
    names.update(f'model_answer/{model}.jsonl' for model in (*MODELS, BASELINE))
    names.update(f'manifests/{model}.json' for model in MODELS)
    hashes = {str((suite / name).resolve()): file_hash(suite / name) for name in sorted(names)}
    bundle = collect_subsets(run, judge, strict)
    hashes.update(bundle.files_sha256)
    bundle.files_sha256 = hashes
    return bundle, strict, upstream, reuse


def table_csv(result):
    stream = io.StringIO()
    columns = ['model', 'prompts', 'games', 'expanded_rows', 'raw_weighted_direct_pct',
               'raw_official_bootstrap_pct', 'raw_ci90_low_pct', 'raw_ci90_high_pct',
               'length_markdown_controlled_pct', 'controlled_ci90_low_pct',
               'controlled_ci90_high_pct', 'raw_status', 'style_status', 'style_reason']
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for model in MODELS:
        row = result['models'][model]
        raw, style = row['raw'], row['length_markdown_controlled']
        values = {'model': model, **{key: row[key] for key in ('prompts', 'games', 'expanded_rows')},
                  'raw_status': raw.get('status', 'available'),
                  'style_status': style.get('status', 'available'), 'style_reason': style.get('reason', '')}
        for data, mapping in [(raw, {'weighted_direct_mean': 'raw_weighted_direct_pct',
                                    'official_bootstrap_mean': 'raw_official_bootstrap_pct'}),
                              (style, {'official_bootstrap_median': 'length_markdown_controlled_pct'})]:
            values.update({column: data[key] * 100 for key, column in mapping.items() if key in data})
        for data, prefix in ((raw, 'raw'), (style, 'controlled')):
            if 'ci90' in data:
                values[prefix + '_ci90_low_pct'], values[prefix + '_ci90_high_pct'] = [x * 100 for x in data['ci90']]
        writer.writerow(values)
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh scoring directory')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Use a fresh scoring output: {args.output}')
    policy = exclusion_policy.validate_policy(read_json(args.policy))
    policy_hash = file_hash(args.policy)
    bundle, strict, upstream, reuse = load_suite(args.suite)
    torch.set_num_threads(1)
    primary = score_subset(bundle.battles, bundle.answers, strict, upstream=upstream)
    common = score_subset(bundle.common_battles, bundle.answers, strict, upstream=upstream)
    common.update(question_uids=bundle.common_uids, questions=len(bundle.common_uids))
    bundle.files_sha256.update({str(args.policy.resolve()): policy_hash,
        str(Path(__file__).resolve()): file_hash(__file__),
        str(Path(exclusion_policy.__file__).resolve()): file_hash(exclusion_policy.__file__)})
    result = {'status': 'complete_with_judge_exclusions', 'protocol': PROTOCOL,
        'exclusion_policy': policy, 'policy_sha256': policy_hash,
        'judge': JUDGE, 'baseline': BASELINE, 'uses_official_judge': False,
        'uses_official_baseline': False, 'full_500_official_leaderboard': False,
        'suite': str(args.suite.resolve()), 'generation_reuse_verification': reuse,
        'coverage': {'questions': 500, 'attempted_games': 6000,
                     'per_model': bundle.counts,
                     'valid_games': sum(c['valid_games'] for c in bundle.counts.values()),
                     'judge_failed_games': sum(c['judge_failed_games'] for c in bundle.counts.values()),
                     'excluded_model_prompts': sum(c['excluded_prompts'] for c in bundle.counts.values()),
                     'valid_but_discarded_partner_games': sum(c['valid_but_discarded_partner_games'] for c in bundle.counts.values())},
        'seed': 42, 'bootstrap_rounds': 100, 'confidence_level': .90,
        **primary, 'retained_question_uids': bundle.valid_uids, 'common_valid_subset': common,
        'exclusion_manifest': {'path': 'exclusions.jsonl', 'rows': len(bundle.exclusions)},
        'math': {'upstream_revision': strict.REVISION, 'upstream_files_sha256': strict.PINNED_HASHES,
                 'frozen_scorer_sha256': file_hash(strict.__file__), 'decisive_weight': 3, 'tie_score': .5,
                 'raw_point': 'Direct weighted mean; official central score is mean of bootstrap means',
                 'bootstrap_unit': 'Expanded game rows; raw within model, style pooled across retained models',
                 'style_point': 'Median of 100 official combined length/Markdown bootstrap win probabilities',
                 'style_reference': 'Zero normalized controls (pooled mean observed contrast)',
                 'ci': '5th and 95th percentiles with pandas linear interpolation'},
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'torch')},
        'inputs_sha256': bundle.files_sha256,
        'caveats': ['Custom GPT-4o judge, custom GPT-4o-mini reference and explicit exclusions; not an official full-500 leaderboard.',
                    'Each model uses its own valid two-order pairs; denominators can differ. The secondary common-valid subset permits like-question comparison.',
                    'Judge-output exclusions can bias the remaining questions; no ranking preference is imposed.',
                    'Both order rows and decisive replications are resampling units; uncertainty is not prompt-cluster uncertainty.',
                    'Style fits pool retained candidates and are candidate-set dependent, not causal estimates.',
                    'Answer metadata are validated and hash-bound, not recomputed from response text.']}
    # Bind results to the bytes actually validated, including all current games.
    for name, expected in bundle.files_sha256.items():
        if file_hash(name) != expected:
            raise ValueError(f'Input changed during scoring: {name}')
    manifest = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in bundle.exclusions)
    result['exclusion_manifest']['sha256'] = hashlib.sha256(manifest.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    strict.atomic_text(args.output / 'exclusions.jsonl', manifest)
    strict.atomic_text(args.output / 'results.csv', table_csv(primary))
    strict.atomic_text(args.output / 'common_subset_results.csv', table_csv(common))
    strict.atomic_text(args.output / 'results.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(table_csv(primary), end='')
    return result


if __name__ == '__main__':
    main()
