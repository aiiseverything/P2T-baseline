#!/usr/bin/env python3
"""Score a completed retry campaign with pure GPT-4o and mixed-judge overlays.

Original games are immutable. Only frozen targets can supply replacements; all
attempt evidence is checked before applying the unchanged official score math.
This command is read-only until it writes a fresh scoring output directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import arena_exclusion_policy as exclusion_policy
from scripts import score_arena_with_exclusions as original_score

MODELS = original_score.MODELS
VARIANTS = ('pure_gpt4o', 'mixed_gpt4o_gpt41')
PROTOCOL = 'arena_hard_v2_retry5_gpt41_fallback_exclusions_v1'


@dataclass
class Variant:
    bundle: original_score.Subsets
    provenance: list


def load_resolution(campaign, require_complete=True):
    # The runner's loader performs no relay construction or network access.
    from scripts.arena_retry_fallback import load_resolution as read
    return read(campaign, require_complete=require_complete)


def load_frozen_judge(suite):
    with _without_bytecode():
        return original_score.load_module(Path(suite) / 'source/scripts/judge_arena_hard.py',
                                          'retry_judge')


@contextmanager
def _without_bytecode():
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        yield
    finally:
        sys.dont_write_bytecode = previous


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(row):
    _require(isinstance(row, dict) and row.get('tag') in MODELS
             and isinstance(row.get('uid'), str) and type(row.get('order')) is int
             and row['order'] in (0, 1), 'Invalid game identity')
    return row['tag'], row['uid'], row['order']


def _merge_hashes(*mappings):
    merged = {}
    for mapping in mappings:
        _require(isinstance(mapping, dict), 'Missing input hash map')
        for path, digest in mapping.items():
            name = str(Path(path).resolve())
            _require(name not in merged or merged[name] == digest,
                     f'Conflicting input hash: {name}')
            merged[name] = digest
    return merged


def _verify_hashes(hashes):
    for name, expected in hashes.items():
        _require(Path(name).is_file() and original_score.file_hash(name) == expected,
                 f'Input changed during scoring: {name}')


def _read_bound(path, expected):
    path = Path(path).resolve()
    _require(path.is_file(), f'Missing attempt evidence: {path}')
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected, f'Input changed: {path}')
    return json.loads(data)


def _original_records(source, frozen_hashes, judge):
    originals = {}
    for path, digest in source.files_sha256.items():
        location = Path(path)
        if tuple(location.parts[-4:-2]) != ('state', 'games'):
            continue
        _require(frozen_hashes.get(str(location.resolve())) == digest,
                 f'Original game is not bound to campaign: {path}')
        record = _read_bound(path, digest)
        key = _key(record)
        _require(key not in originals, 'Duplicate original game identity')
        _require(record.get('judge_model') == 'gpt-4o'
                 and record.get('baseline_model') == original_score.BASELINE
                 and judge.digest(record.get('request')) == record.get('request_sha256')
                 and exclusion_policy.classify_record(record, judge) in ('valid', 'judge_failed'),
                 f'Invalid original game: {key}')
        originals[key] = (record, str(location.resolve()), digest)
    expected = {(tag, uid, order) for tag in MODELS for uid in source.answers[tag]
                for order in (0, 1)}
    _require(set(originals) == expected, 'Missing or extra original games')
    return originals


def _validate_resolution(target, resolution, original, hashes, judge):
    """Recheck first-valid selection against the saved bytes, never the winner."""
    record, original_path, original_hash = original
    key = _key(target)
    _require(_key(resolution) == key and target.get('original_path') == original_path
             and target.get('original_sha256') == original_hash
             and target.get('request_sha256') == record.get('request_sha256')
             and type(record.get('attempt', 0)) is int and record.get('attempt', 0) == 0
             and exclusion_policy.classify_record(record, judge) == 'judge_failed',
             f'Target identity or original failure mismatch: {key}')
    attempts = resolution.get('attempts')
    _require(isinstance(attempts, list) and 2 <= len(attempts) <= 6,
             f'Incomplete retry attempt chain: {key}')
    final = None
    for number, evidence in enumerate(attempts, 1):
        model = 'gpt-4o' if number <= 5 else 'gpt-4.1'
        _require(isinstance(evidence, dict) and type(evidence.get('total_attempt')) is int
                 and evidence['total_attempt'] == number and evidence.get('judge_model') == model,
                 f'Attempt order or judge mismatch: {key}')
        path = str(Path(evidence['record_path']).resolve())
        _require(hashes.get(path) == evidence.get('record_sha256'),
                 f'Unbound attempt evidence: {key}')
        current = _read_bound(path, evidence['record_sha256'])
        expected_request = dict(record['request'], model=model)
        _require(_key(current) == key and current.get('judge_model') == model
                 and current.get('baseline_model') == record.get('baseline_model')
                 and current.get('protocol_sha256') == record.get('protocol_sha256')
                 and judge.digest(current.get('request')) == judge.digest(expected_request)
                 and current.get('request_sha256') == judge.digest(expected_request),
                 f'Attempt identity or request mismatch: {key}')
        _require(all(evidence.get(field) == current.get(field) for field in
                     ('status', 'request_sha256', 'local_request_id', 'response_model')),
                 f'Attempt provenance mismatch: {key}')
        if number == 1:
            _require(judge.digest(current) == judge.digest(record),
                     f'Original attempt snapshot mismatch: {key}')
        else:
            _require(current.get('total_attempt') == number
                     and isinstance(current.get('local_request_id'), str)
                     and bool(current['local_request_id']), f'Missing attempt identity: {key}')
        response_model = current.get('response_model')
        if response_model is not None:
            _require(isinstance(response_model, str)
                     and re.fullmatch(re.escape(model) + r'(?:-\d{4}-\d{2}-\d{2})?', response_model),
                     f'Returned judge model mismatch: {key}')
        state = exclusion_policy.classify_record(current, judge)
        _require(state in ('valid', 'judge_failed'), f'Nonterminal or corrupt attempt: {key}')
        _require(state != 'valid' or number == len(attempts),
                 f'Attempt after first valid result: {key}')
        final = current
    selected = resolution.get('selected_record')
    final_valid = exclusion_policy.classify_record(final, judge) == 'valid'
    expected_resolution = ('gpt4o' if len(attempts) <= 5 else 'gpt41') if final_valid else 'failed'
    _require(resolution.get('resolution') == expected_resolution
             and (final_valid or len(attempts) == 6)
             and (judge.digest(selected) == judge.digest(final) if final_valid else selected is None),
             f'Incomplete or inconsistent first-valid resolution: {key}')


def _validate_transport_recovery(manifest, resolutions, hashes, judge):
    """Keep an inspected ambiguous request outside the completed-verdict chain."""
    affected = [row for row in resolutions if row.get('transport_recovery') is not None]
    recovery = manifest.get('transport_recovery')
    if recovery is None:
        _require(not affected, 'Transport recovery is missing from the campaign manifest')
        return
    _require(isinstance(recovery, dict) and len(affected) == 1
             and judge.digest(affected[0]['transport_recovery']) == judge.digest(recovery),
             'Transport recovery differs between campaign and game provenance')
    _require(all(type(recovery.get(field)) is int and recovery[field] == expected
                 for field, expected in (('logical_attempt', 2), ('physical_replay_attempts', 1),
                                          ('unknown_usage_requests', 1)))
             and isinstance(recovery.get('recovery_id'), str) and bool(recovery['recovery_id'])
             and isinstance(recovery.get('cost_uncertainty'), str) and bool(recovery['cost_uncertainty']),
             'Transport recovery must retain the extra physical request and unknown usage')
    original, replacement = recovery.get('original_ambiguous'), recovery.get('replacement')
    _require(isinstance(original, dict) and isinstance(replacement, dict),
             'Missing transport recovery attempt evidence')
    evidence = affected[0]['attempts'][1]
    _require(all(replacement.get(field) == evidence.get(field) for field in
                 ('record_path', 'record_sha256', 'local_request_id', 'request_sha256')),
             'Transport recovery replacement is not completed logical attempt 2')
    for path, digest in ((original.get('record_path'), original.get('record_sha256')),
                         (recovery.get('authorization_path'), recovery.get('authorization_sha256'))):
        _require(isinstance(path, str) and hashes.get(str(Path(path).resolve())) == digest,
                 'Unbound transport recovery evidence')
        _read_bound(path, digest)
    ambiguous = _read_bound(original['record_path'], original['record_sha256'])
    _require(_key(ambiguous) == _key(affected[0]) and ambiguous.get('total_attempt') == 2
             and ambiguous.get('status') == original.get('status') == 'ambiguous'
             and ambiguous.get('error_type') == original.get('error_type') == 'RemoteProtocolError'
             and ambiguous.get('usage') is None
             and ambiguous.get('local_request_id') == original.get('local_request_id')
             and original.get('local_request_id') != replacement.get('local_request_id')
             and ambiguous.get('request_sha256') == original.get('request_sha256')
                 == replacement.get('request_sha256')
             and judge.digest(ambiguous.get('request')) == replacement.get('request_sha256'),
             'Original transport request identity or unknown-usage provenance changed')


def _request_accounting(resolutions):
    completed = sum(len(row['attempts']) - 1 for row in resolutions)
    recoveries = [row['transport_recovery'] for row in resolutions if row.get('transport_recovery')]
    extra = sum(row['physical_replay_attempts'] for row in recoveries)
    unknown = sum(row['unknown_usage_requests'] for row in recoveries)
    return {'completed_new_attempts': completed, 'extra_physical_requests': extra,
            'physical_new_requests': completed + extra, 'unknown_usage_requests': unknown,
            'usage_accounting_complete': unknown == 0}


def collect_variants(source, manifest, resolutions, judge, strict):
    """Overlay validated selected games, preserving all other original records."""
    hashes = _merge_hashes(source.files_sha256, manifest.get('source_files_sha256'),
                           manifest.get('campaign_files_sha256'))
    _verify_hashes(hashes)
    originals = _original_records(source, manifest['source_files_sha256'], judge)
    targets = manifest.get('targets')
    _require(isinstance(targets, list) and isinstance(resolutions, list), 'Missing target resolutions')
    target_keys, resolution_keys = [_key(row) for row in targets], [_key(row) for row in resolutions]
    _require(len(set(target_keys)) == len(target_keys)
             and len(set(resolution_keys)) == len(resolution_keys)
             and set(target_keys) == set(resolution_keys)
             and set(target_keys) <= set(originals), 'Target and resolution sets differ')
    resolved = dict(zip(resolution_keys, resolutions))
    for target in targets:
        key = _key(target)
        _validate_resolution(target, resolved[key], originals[key], hashes, judge)
    _validate_transport_recovery(manifest, resolutions, hashes, judge)

    variants = {}
    # The validated game hashes retain the original question iteration order.
    # Keeping it also preserves deterministic bootstrap row order.
    questions = [uid for tag, uid, order in originals if tag == MODELS[0] and order == 0]
    for variant in VARIANTS:
        battles, provenance, exclusions, valid_uids, counts = [], [], [], {}, {}
        for tag in MODELS:
            valid_uids[tag] = []
            count = {'attempted_games': 2 * len(questions), 'valid_games': 0,
                'judge_failed_games': 0, 'excluded_prompts': 0,
                'valid_but_discarded_partner_games': 0, 'retained_prompts': 0,
                'retained_games': 0, 'expanded_rows': 0, 'recovered_gpt4o_games': 0,
                'fallback_gpt41_games': 0}
            for uid in questions:
                records, states, pair_provenance = [], [], []
                for order in (0, 1):
                    key = tag, uid, order
                    record, path, digest = originals[key]
                    resolution = resolved.get(key)
                    use_retry = resolution is not None and (resolution['resolution'] == 'gpt4o'
                        or variant == 'mixed_gpt4o_gpt41' and resolution['resolution'] == 'gpt41')
                    total_attempt = record.get('attempt', 0) + 1
                    if use_retry:
                        record = resolution['selected_record']
                        evidence = resolution['attempts'][-1]
                        path, digest = evidence['record_path'], evidence['record_sha256']
                        total_attempt = evidence['total_attempt']
                        count['recovered_gpt4o_games' if resolution['resolution'] == 'gpt4o'
                              else 'fallback_gpt41_games'] += 1
                    state = exclusion_policy.classify_record(record, judge)
                    records.append(record)
                    states.append(state)
                    count['valid_games' if state == 'valid' else 'judge_failed_games'] += 1
                    pair_provenance.append({'tag': tag, 'uid': uid, 'order': order,
                        'source': 'retry_campaign' if use_retry else 'original',
                        'judge_model': record['judge_model'], 'response_model': record.get('response_model'),
                        'total_attempt': total_attempt, 'record_path': path, 'record_sha256': digest,
                        'request_sha256': record['request_sha256'], 'score': record['score'],
                        'record_classification': state, 'is_retry_target': resolution is not None,
                        'transport_recovery': resolution.get('transport_recovery') if resolution else None,
                        'campaign_resolution': resolution['resolution'] if resolution else None})
                retained = states == ['valid', 'valid']
                provenance.extend(dict(row, retained_pair=retained) for row in pair_provenance)
                if not retained:
                    count['excluded_prompts'] += 1
                    count['valid_but_discarded_partner_games'] += states.count('valid')
                    for record, state, evidence in zip(records, states, pair_provenance):
                        exclusions.append({**evidence, 'model': tag,
                            'reason': exclusion_policy.judge_failure_reason(record, judge)
                                if state == 'judge_failed' else 'partner_judge_failed',
                            'finish_reason': record['finish_reason'],
                            'game_sha256': evidence['record_sha256']})
                    continue
                valid_uids[tag].append(uid)
                outcomes = strict.expand_outcomes([{'score': row['score']} for row in records])
                battles.extend({'uid': uid, 'model': tag, 'scores': score} for score in outcomes)
                count['expanded_rows'] += len(outcomes)
            count['retained_prompts'] = len(valid_uids[tag])
            count['retained_games'] = 2 * count['retained_prompts']
            counts[tag] = count
        frame = pd.DataFrame(battles, columns=['uid', 'model', 'scores'])
        common = set.intersection(*(set(valid_uids[tag]) for tag in MODELS))
        bundle = original_score.Subsets(source.answers, frame, frame[frame.uid.isin(common)].copy(),
            valid_uids, [uid for uid in questions if uid in common], exclusions, counts, dict(hashes))
        variants[variant] = Variant(bundle, provenance)
    return variants


def _jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)


def _artifact(name, rows):
    content = _jsonl(rows)
    return content, {'path': name, 'rows': len(rows),
                     'sha256': hashlib.sha256(content.encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh scoring directory')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Use a fresh scoring output: {args.output}')
    manifest, resolutions = load_resolution(args.campaign, require_complete=True)
    suite = Path(manifest['source_suite']).resolve()
    with _without_bytecode():
        source, strict, upstream, reuse = original_score.load_suite(suite)
        judge = load_frozen_judge(suite)
    variants = collect_variants(source, manifest, resolutions, judge, strict)
    request_accounting = _request_accounting(resolutions)
    implementation_files = [Path(__file__).resolve(), Path(original_score.__file__).resolve(),
                            Path(exclusion_policy.__file__).resolve()]
    runner_path = ROOT / 'scripts/arena_retry_fallback.py'
    if runner_path.is_file():
        implementation_files.append(runner_path)
    code_hashes = {str(path): original_score.file_hash(path) for path in implementation_files}
    torch.set_num_threads(1)
    results, payloads = {}, {}
    for name, variant in variants.items():
        bundle = variant.bundle
        bundle.files_sha256 = _merge_hashes(bundle.files_sha256, code_hashes)
        primary = original_score.score_subset(bundle.battles, bundle.answers, strict, upstream=upstream)
        common = original_score.score_subset(bundle.common_battles, bundle.answers, strict, upstream=upstream)
        common.update(question_uids=bundle.common_uids, questions=len(bundle.common_uids))
        exclusions, exclusions_meta = _artifact('exclusions.jsonl', bundle.exclusions)
        provenance, provenance_meta = _artifact('game_provenance.jsonl', variant.provenance)
        observed_judges = {row['judge_model'] for row in variant.provenance}
        judges = [model for model in ('gpt-4o', 'gpt-4.1') if model in observed_judges]
        coverage = {field: sum(count[field] for count in bundle.counts.values()) for field in
                    ('attempted_games', 'valid_games', 'judge_failed_games', 'excluded_prompts',
                     'valid_but_discarded_partner_games', 'retained_prompts', 'retained_games',
                     'recovered_gpt4o_games', 'fallback_gpt41_games')}
        coverage['excluded_model_prompts'] = coverage.pop('excluded_prompts')
        coverage['retained_model_prompts'] = coverage.pop('retained_prompts')
        coverage.update(questions=len(bundle.answers[MODELS[0]]), per_model=bundle.counts)
        result = {'status': 'complete_with_judge_exclusions', 'protocol': PROTOCOL,
            'variant': name, 'judge_models': judges, 'baseline': original_score.BASELINE,
            'uses_official_judge': False, 'uses_official_baseline': False,
            'full_500_official_leaderboard': False, 'suite': str(suite),
            'campaign': str(args.campaign.resolve()), 'campaign_manifest_sha256': manifest['manifest_sha256'],
            'target_games': len(resolutions), 'generation_reuse_verification': reuse,
            'transport_recovery': manifest.get('transport_recovery'),
            'request_accounting': request_accounting,
            'coverage': coverage, 'seed': 42, 'bootstrap_rounds': 100, 'confidence_level': .90,
            **primary, 'retained_question_uids': bundle.valid_uids, 'common_valid_subset': common,
            'exclusion_manifest': exclusions_meta, 'game_provenance': provenance_meta,
            'math': {'upstream_revision': strict.REVISION, 'upstream_files_sha256': strict.PINNED_HASHES,
                'frozen_scorer_sha256': original_score.file_hash(strict.__file__),
                'decisive_weight': 3, 'tie_score': .5,
                'raw_point': 'Direct weighted mean; official central score is mean of bootstrap means',
                'bootstrap_unit': 'Expanded game rows; raw within model, style pooled across retained models',
                'style_point': 'Median of 100 official combined length/Markdown bootstrap win probabilities',
                'style_reference': 'Zero normalized controls (pooled mean observed contrast)',
                'ci': '5th and 95th percentiles with pandas linear interpolation'},
            'versions': {package: importlib.metadata.version(package) for package in ('numpy', 'pandas', 'torch')},
            'inputs_sha256': bundle.files_sha256,
            'caveats': ['Custom reference and explicit exclusions; not an official full-500 leaderboard.',
                'Each model uses its own valid two-order pairs; the common-valid subset uses their intersection.',
                'Pure GPT-4o retains only accepted GPT-4o recoveries; the mixed variant also accepts GPT-4.1 fallback.',
                'Judge provenance is attached to every selected game; invalid outputs remain excluded as whole pairs.',
                'Judge-output exclusions and fallback selection can bias the remaining sample.',
                'Expanded game rows are bootstrap units; uncertainty is not prompt-cluster uncertainty.',
                'Style fits pool retained candidates and are candidate-set dependent, not causal estimates.',
                'Answer metadata are validated and hash-bound, not recomputed from response text.']}
        results[name] = result
        payloads[name] = {'results.json': json.dumps(result, indent=2, allow_nan=False) + '\n',
            'results.csv': original_score.table_csv(primary),
            'common_subset_results.csv': original_score.table_csv(common),
            'exclusions.jsonl': exclusions, 'game_provenance.jsonl': provenance}
    for variant in variants.values():
        _verify_hashes(variant.bundle.files_sha256)
    summary = {'status': 'complete', 'campaign': str(args.campaign.resolve()),
        'campaign_manifest_sha256': manifest['manifest_sha256'], 'target_games': len(resolutions),
        'transport_recovery': manifest.get('transport_recovery'),
        'request_accounting': request_accounting,
        'resolutions': {state: sum(row['resolution'] == state for row in resolutions)
                        for state in ('gpt4o', 'gpt41', 'failed')},
        'variants': {name: {'path': name + '/results.json',
            'sha256': hashlib.sha256(payloads[name]['results.json'].encode()).hexdigest(),
            'coverage': result['coverage'], 'common_valid_questions': result['common_valid_subset']['questions']}
            for name, result in results.items()}}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, files in payloads.items():
        (args.output / name).mkdir()
        for filename, content in files.items():
            strict.atomic_text(args.output / name / filename, content)
    strict.atomic_text(args.output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': summary['status'], 'resolutions': summary['resolutions'],
                      'output': str(args.output.resolve())}, allow_nan=False))
    return summary


if __name__ == '__main__':
    main()
