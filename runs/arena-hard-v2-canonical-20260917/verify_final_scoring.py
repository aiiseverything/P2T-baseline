"""Independent, read-only final Arena QA; stdout JSON, no API/GPU execution.

The pinned upstream function bodies are compiled unchanged from their AST to
avoid importing API clients. Only input lookup and leaderboard capture are
substituted; the official raw/style math, fitting, and formatting are executed.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys

SUITE = Path(__file__).resolve().parent
TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
BASELINE = 'o3-mini-2025-01-31'
PATTERNS = [r'\[\[([AB<>=]+)\]\]', r'\[([AB<>=]+)\]']
SCORES = {'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A'}
PINS = {
    'show_result.py': '5da0fa30a807a150da37e97811c2bc248959015eedb884d9d471e894c8bf90c6',
    'utils/math_utils.py': 'c2e927dc0f45771d9b387c29837c76c2b5008b1f6f81819259c288c01f49ade0',
    'utils/judge_utils.py': 'cbb093f4fc27408d8447e946392f26f98538b1e3950f574aca91a0f7595167b4',
    'gen_judgment.py': 'eddf78a88ce0e3348dd54debd3cb668e1b1d8773d4b239e47b71e5f2564bb402',
    'config/arena-hard-v2.0.yaml': 'ee68e5c6ac98a79720bf11014fc3f7fc3c3e5899153b3b08efe3db0e174215a0',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f'Invalid JSONL objects: {path}')
    return rows


def index_rows(rows, label):
    require(len(rows) == 500, f'Expected exactly 500 rows: {label}')
    require(all(isinstance(r.get('uid'), str) and r['uid'] for r in rows), f'Bad uid: {label}')
    result = {r['uid']: r for r in rows}
    require(len(result) == 500, f'Duplicate uid: {label}')
    return result


def official_functions(upstream):
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    upstream = Path(upstream)
    for name, expected in PINS.items():
        require(file_hash(upstream / name) == expected, f'Official source changed: {name}')
    name = '_arena_independent_official_math'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, upstream / 'utils/math_utils.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    math_module = sys.modules[name]
    namespace = {'pd': pd, 'np': np, 'torch': torch, 'tqdm': tqdm, 'os': os,
                 'JUDGE_SETTINGS': runpy.run_path(str(upstream / 'utils/judge_utils.py'))['JUDGE_SETTINGS']}
    namespace.update({name: getattr(math_module, name) for name in (
        'one_hot_encode', 'to_winrate_probabilities', 'bootstrap_pairwise_model')})
    for filename, names in [('gen_judgment.py', {'get_score'}),
                            ('show_result.py', {'load_judgments', 'get_model_style_metadata',
                             'format_confidence_interval', 'print_leaderboard',
                             'print_leaderboard_with_style_features'})]:
        tree = ast.parse((upstream / filename).read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({node.name for node in nodes} == names, f'Missing official functions: {filename}')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(upstream / filename), 'exec'), namespace)
    return namespace


def verify_game(record, game, request, tag, uid, order, parse):
    require(record.get('tag') == tag and record.get('uid') == uid
            and type(record.get('order')) is int and record['order'] == order,
            f'Game candidate/uid/order mismatch: {tag}/{uid}/{order}')
    require(record.get('request') == request and record.get('request_sha256') == digest(request),
            f'Official request mismatch: {tag}/{uid}/{order}')
    require(record.get('status') == 'valid' and record.get('finish_reason') == 'stop',
            f'Unresolved/truncated game: {tag}/{uid}/{order}')
    text = record.get('answer')
    require(isinstance(text, str) and record.get('score') in SCORES
            and parse(text, PATTERNS) == record['score'], f'Official parse mismatch: {tag}/{uid}/{order}')
    usage = record.get('usage')
    require(isinstance(usage, dict)
            and all(type(usage.get(key)) is int and usage[key] >= 0
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
            f'Invalid usage: {tag}/{uid}/{order}')
    require(game == {'score': record['score'], 'judgment': {'answer': text},
                      'prompt': request['messages']}, f'Export/state mismatch: {tag}/{uid}/{order}')


def verify_generation_cache(cache, tag, policy, answer_sha):
    require(cache.get('outputs') == {f'{tag}.jsonl': answer_sha}, f'Generation output hash mismatch: {tag}')
    config = cache.get('config', {})
    require(config.get('model_tag') == tag and config.get('policy') == policy,
            f'Frozen generation policy mismatch: {tag}')
    require(config.get('recipe') == {'temp': 1., 'n': 1, 'top_p': 1., 'top_k': -1}
            and config.get('seed') == 42 and config.get('max_tokens') == 4096
            and config.get('engine', {}).get('max_model_len') == 16384
            and config.get('engine', {}).get('hf_overrides') == {'head_dtype': 'float32'},
            f'Generation configuration mismatch: {tag}')


def timestamp(value):
    require(isinstance(value, str) and value, 'Missing retry timestamp')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Retry timestamp must include timezone')
    return result


def load_retry_binding(suite, frozen_judge_sha256):
    suite = Path(suite)
    policy_path = suite / 'retry_policy.json'
    if not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text())
    require(policy.get('policy') == 'arena_identical_request_invalid_output_retry_v1'
            and type(policy.get('max_total_attempts_per_game')) is int
            and policy['max_total_attempts_per_game'] == 5
            and policy.get('frozen_judge_sha256') == frozen_judge_sha256, 'Uniform retry policy/frozen judge mismatch')
    timestamp(policy.get('declared_at'))
    helper_path, host_path = suite / 'retry_invalid_judgments.py', suite / 'host_execution.json'
    policy_sha, helper_sha = file_hash(policy_path), file_hash(helper_path)
    expected = {'retry_policy': policy, 'retry_policy_sha256': policy_sha, 'retry_helper_sha256': helper_sha}
    host = json.loads(host_path.read_text())
    require(all(host.get(key) == value for key, value in expected.items()), 'Host retry policy/helper mismatch')
    if 'continuation' in host:
        identity = host['continuation'].get('identity', {})
        require(identity.get('runners_sha256', {}).get('retry_invalid_judgments.py') == helper_sha
                and identity.get('inputs_sha256', {}).get('retry_policy.json') == policy_sha,
                'Continuation retry identity mismatch')
    evidence = {str(path): file_hash(path) for path in (policy_path, helper_path, host_path)}
    decision_path = suite / 'cost_decision.json'
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        require(all(decision.get(key) == value for key, value in expected.items()), 'Cost decision retry binding mismatch')
        evidence[str(decision_path)] = file_hash(decision_path)
    return {'policy': policy, 'policy_sha256': policy_sha, 'helper_sha256': helper_sha,
            'declared_at': policy['declared_at'], 'evidence': evidence}


def verify_retry_chain(record, state_dir, tag, uid, order, request, parse, binding=None):
    """Follow every predecessor; attempt 0/1 may predate the new retry helper."""
    state_dir = Path(state_dir)
    attempt = record.get('attempt')
    require(type(attempt) is int and 0 <= attempt < 5, 'Invalid attempt counter or MAX5 exceeded')
    archives, request_ids = {}, set()
    node = record
    while True:
        local_id = node.get('local_request_id')
        require(isinstance(local_id, str) and local_id and Path(local_id).name == local_id
                and local_id not in request_ids, 'Duplicate or invalid retry request ID')
        request_ids.add(local_id)
        require(type(node.get('attempt')) is int and node['attempt'] == attempt
                and node.get('tag') == tag and node.get('uid') == uid
                and type(node.get('order')) is int and node['order'] == order
                and node.get('request') == request and node.get('request_sha256') == digest(request),
                f'Retry archive binding mismatch: {tag}/{uid}/{order}')
        helper_fields = ('retry_helper_sha256', 'retry_policy', 'retry_policy_sha256')
        uses_helper = any(key in node for key in helper_fields)
        require(attempt < 2 or uses_helper, 'Later retry has no helper provenance')
        if uses_helper:
            require(attempt > 0 and binding is not None
                    and node.get('retry_helper_sha256') == binding['helper_sha256']
                    and node.get('retry_policy_sha256') == binding['policy_sha256']
                    and node.get('retry_policy') == binding['policy'], 'Retry helper/policy binding mismatch')
            require(timestamp(node.get('started_at')) >= timestamp(binding['declared_at']),
                    'Retry preceded uniform policy declaration')
        if attempt == 0:
            require('supersedes_local_request_id' not in node, 'Initial attempt has an unexpected predecessor')
            break
        prior_id = node.get('supersedes_local_request_id')
        require(isinstance(prior_id, str) and prior_id and Path(prior_id).name == prior_id
                and prior_id not in request_ids, 'Invalid superseded ID or retry cycle')
        archive = state_dir / 'attempts' / tag / f'{uid}-{order}' / f'{prior_id}.json'
        require(archive.is_file(), f'Missing retry predecessor archive: {archive}')
        old = json.loads(archive.read_text())
        require(old.get('local_request_id') == prior_id, 'Retry predecessor filename/ID mismatch')
        require(old.get('status') == 'invalid' and isinstance(old.get('answer'), str)
                and old.get('finish_reason') in ('stop', 'length'), 'Only completed-invalid responses may be retried')
        finished = timestamp(old.get('finished_at'))
        require(timestamp(node.get('started_at')) >= finished, 'Retry started before predecessor completed')
        parsed = parse(old['answer'], PATTERNS)
        accepted = parsed if parsed in SCORES else None
        require(old.get('score') == accepted, 'Archived parse mismatch')
        require(old['finish_reason'] == 'length' or accepted is None,
                'A valid verdict was retried; first valid must stop')
        archives[str(archive)] = file_hash(archive)
        node, attempt = old, attempt - 1
    return {'archives': archives, 'request_ids': request_ids}


def direct_score(rows):
    numerator, denominator = 0., 0
    for row in rows:
        require(isinstance(row.get('games'), list) and len(row['games']) == 2, 'Invalid games')
        for order, game in enumerate(row['games']):
            score = game.get('score')
            require(score in SCORES, 'Invalid verdict')
            weight = 3 if '>>' in score else 1
            value = .5 if score == 'A=B' else float(score.startswith('A') == (order == 1))
            numerator += value * weight
            denominator += weight
    return {'weighted_direct_mean': numerator / denominator, 'expanded_rows': denominator,
            'games': len(rows) * 2}


def official_scores(battles, metadata, upstream):
    import numpy as np
    import torch
    official = official_functions(upstream)
    # The real upstream feature function receives exactly the validated metadata.
    official['get_model_style_metadata'] = lambda benchmark: metadata
    original = official['format_confidence_interval']
    captured = {}
    def capture(mean, lower, upper, baseline=None):
        table = mean.merge(lower, on='model').merge(upper, on='model').set_index('model')
        captured.update({model: {'point': float(row.scores), 'ci90': [float(row.lower), float(row.upper)]}
                         for model, row in table.iterrows() if model in TAGS})
        return original(mean, lower, upper, baseline)
    official['format_confidence_interval'] = capture
    previous_rng, previous_threads = np.random.get_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            np.random.seed(42)
            official['print_leaderboard'](battles.copy(), 'hard_prompt')
            raw = dict(captured); captured.clear()
            np.random.seed(42)
            official['print_leaderboard_with_style_features'](
                battles.copy(), 'verified-inputs', 'hard_prompt', ['length', 'markdown'])
            controlled = dict(captured)
        require(set(raw) == set(controlled) == set(TAGS), 'Official leaderboard candidate mismatch')
        return {'raw': raw, 'controlled': controlled}
    finally:
        np.random.set_state(previous_rng)
        torch.set_num_threads(previous_threads)


def verify(suite=SUITE, results_path=None):
    import yaml
    suite = Path(suite).resolve()
    results_path = Path(results_path) if results_path else suite / 'scores/results.json'
    upstream = suite / 'source/third_party/arena_hard'
    official = official_functions(upstream)
    manifest_path = suite / 'experiment.json'
    manifest = json.loads(manifest_path.read_text())
    require(tuple(manifest['models']) == TAGS, 'Frozen candidate identities changed')
    used_sources = ['source/scripts/judge_arena_hard.py', 'source/scripts/score_arena_hard.py']
    used_sources += ['source/third_party/arena_hard/' + name for name in PINS]
    used_sources += ['source/third_party/arena_hard/gen_answer.py',
                     'source/third_party/arena_hard/utils/add_markdown_info.py']
    for name in [*used_sources, 'question.jsonl', f'model_answer/{BASELINE}.jsonl']:
        require(file_hash(suite / name) == manifest['files_sha256'][name], f'Frozen input/source changed: {name}')
    question_path = suite / 'question.jsonl'
    question_rows = read_jsonl(question_path)
    questions = index_rows(question_rows, 'questions')
    require(all(q.get('category') == 'hard_prompt' and isinstance(q.get('prompt'), str)
                and q['prompt'].strip() for q in question_rows), 'Invalid question category/prompt')
    require(len({q['prompt'] for q in question_rows}) == 500, 'Duplicate question prompts')
    answer_dir, judgment_dir = suite / 'model_answer', suite / 'model_judgment/gpt-4.1'
    require({p.stem for p in answer_dir.glob('*.jsonl')} == {*TAGS, BASELINE}, 'Answer file coverage mismatch')
    require({p.stem for p in judgment_dir.glob('*.jsonl')} == set(TAGS), 'Judgment file coverage mismatch')
    files = {str(question_path): file_hash(question_path)}
    evidence = {}
    retry_binding = load_retry_binding(suite, file_hash(suite / 'source/scripts/judge_arena_hard.py'))
    if retry_binding:
        evidence.update(retry_binding['evidence'])
    answers, answer_rows, metadata = {}, {}, {}
    for tag in (*TAGS, BASELINE):
        path = answer_dir / f'{tag}.jsonl'
        rows = read_jsonl(path)
        indexed = index_rows(rows, tag)
        require(set(indexed) == set(questions), f'Answer uid coverage: {tag}')
        for uid, row in indexed.items():
            require(row.get('model') == tag and isinstance(row.get('messages'), list)
                    and len(row['messages']) == 2
                    and row['messages'][0] == {'role': 'user', 'content': questions[uid]['prompt']},
                    f'Answer identity/prompt mismatch: {tag}/{uid}')
            last = row['messages'][1]
            require(last.get('role') == 'assistant' and isinstance(last.get('content'), dict)
                    and isinstance(last['content'].get('answer'), str), f'Invalid response: {tag}/{uid}')
        answers[tag], answer_rows[tag] = indexed, rows
        metadata[tag] = {uid: row['metadata'] for uid, row in indexed.items()}
        files[str(path)] = file_hash(path)
        if tag != BASELINE:
            cache_path = suite / 'manifests' / f'{tag}.json'
            verify_generation_cache(json.loads(cache_path.read_text()), tag,
                                    manifest['models'][tag]['policy'], files[str(path)])
            evidence[str(cache_path)] = file_hash(cache_path)
    config = yaml.safe_load((upstream / 'config/arena-hard-v2.0.yaml').read_text())
    require(config['judge_model'] == 'gpt-4.1' and config['temperature'] == 0.0
            and config['max_tokens'] == 16000 and config['regex_patterns'] == PATTERNS
            and config['reference'] is None, 'Official protocol mismatch')
    protocol = {'protocol': 'arena_hard_v2_gpt41_two_order_v1', 'judge': 'gpt-4.1', 'baseline': BASELINE,
        'temperature': 0.0, 'max_tokens': 16000,
        'system_prompt': official['JUDGE_SETTINGS']['hard_prompt']['system_prompt'],
        'prompt_template': config['prompt_template'], 'regex_patterns': PATTERNS,
        'base_url': 'https://api.linkapi.ai/v1',
        'source_sha256': {name: file_hash(upstream / name) for name in
                          ('config/arena-hard-v2.0.yaml', 'utils/judge_utils.py', 'gen_judgment.py')},
        'driver_sha256': file_hash(suite / 'source/scripts/judge_arena_hard.py')}
    state_dir = judgment_dir / 'state'
    for path in [state_dir / 'protocol.json', *[state_dir / 'models' / f'{tag}.json' for tag in TAGS]]:
        evidence[str(path)] = file_hash(path)
    expected_identity = {'protocol': protocol, 'questions_sha256': digest(question_rows),
                          'baseline_sha256': digest(answer_rows[BASELINE])}
    require(json.loads((state_dir / 'protocol.json').read_text()) == expected_identity, 'Judge protocol/input identity mismatch')
    require({p.stem for p in (state_dir / 'models').glob('*.json')} == set(TAGS), 'State model coverage mismatch')
    expected_records, request_ids, counts, all_records, archive_paths = set(), set(), {}, {}, set()
    retry_counts = {tag: {'games_retried': 0, 'extra_attempts': 0,
                          'initial_invalid_by_order': {'0': 0, '1': 0}} for tag in TAGS}
    for tag in TAGS:
        require(json.loads((state_dir / 'models' / f'{tag}.json').read_text()) == {
            'answers_sha256': digest(answer_rows[tag]), 'tag': tag, 'model': tag}, f'State answer digest mismatch: {tag}')
        path = judgment_dir / f'{tag}.jsonl'
        rows = read_jsonl(path)
        indexed = index_rows(rows, f'judgments/{tag}')
        require(set(indexed) == set(questions), f'Judgment uid coverage: {tag}')
        counts[tag] = direct_score(rows)
        files[str(path)] = file_hash(path)
        for uid, row in indexed.items():
            require((row.get('model'), row.get('baseline'), row.get('judge'), row.get('category'))
                    == (tag, BASELINE, 'gpt-4.1', 'hard_prompt'), f'Judgment identity mismatch: {tag}/{uid}')
            for order, game in enumerate(row['games']):
                first, second = (BASELINE, tag) if order == 0 else (tag, BASELINE)
                request = {'model': 'gpt-4.1', 'temperature': 0.0, 'max_tokens': 16000,
                    'messages': [{'role': 'system', 'content': protocol['system_prompt']},
                                 {'role': 'user', 'content': config['prompt_template'].format(
                        QUESTION=questions[uid]['prompt'],
                        ANSWER_A=answers[first][uid]['messages'][1]['content']['answer'],
                        ANSWER_B=answers[second][uid]['messages'][1]['content']['answer'])}]}
                state_path = state_dir / 'games' / tag / f'{uid}-{order}.json'
                expected_records.add(state_path)
                record = json.loads(state_path.read_text())
                verify_game(record, game, request, tag, uid, order, official['get_score'])
                chain = verify_retry_chain(record, state_dir, tag, uid, order, request,
                                           official['get_score'], retry_binding)
                require(not request_ids.intersection(chain['request_ids']), 'Reused request ID across games')
                request_ids.update(chain['request_ids'])
                all_records.update(chain['archives'])
                archive_paths.update(Path(path) for path in chain['archives'])
                if chain['archives']:
                    retry_counts[tag]['games_retried'] += 1
                    retry_counts[tag]['extra_attempts'] += len(chain['archives'])
                    retry_counts[tag]['initial_invalid_by_order'][str(order)] += 1
                all_records[str(state_path)] = file_hash(state_path)
    require(set((state_dir / 'games').rglob('*.json')) == expected_records and len(expected_records) == 6000,
            'Expected exactly 6000 current game states')
    require(set((state_dir / 'attempts').rglob('*.json')) == archive_paths, 'Unbound retry archive')
    # Run the actual upstream loader as well: its normally silent invalid drop
    # is disallowed by the strict state/export checks above and this row count.
    official['glob'] = lambda pattern: [str(judgment_dir / f'{tag}.jsonl') for tag in TAGS]
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        battles = official['load_judgments'](['gpt-4.1'], 'verified-inputs')
    require(len(battles) == sum(row['expanded_rows'] for row in counts.values()), 'Official loader dropped rows')
    expected = official_scores(battles, metadata, upstream)
    results = json.loads(results_path.read_text())
    require(results['status'] == 'complete' and results['seed'] == 42 and results['bootstrap_rounds'] == 100
            and results['confidence_level'] == .9 and results['baseline'] == BASELINE
            and results['judge'] == 'gpt-4.1' and set(results['models']) == set(TAGS), 'Scorer protocol mismatch')
    require(results['inputs_sha256'] == files, 'Scorer input digest mismatch')
    require(results['math']['scorer_sha256'] == file_hash(suite / 'source/scripts/score_arena_hard.py'), 'Scorer source mismatch')
    require(results['coverage'] == {'candidates': list(TAGS), 'prompts_per_candidate': 500,
        'games_per_candidate': 1000, 'total_judgments': 3000, 'total_games': 6000, 'invalid_dropped': 0},
        'Reported coverage mismatch')
    differences = []
    def compare(actual, wanted, where):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and math.isfinite(wanted), f'Nonfinite {where}')
        difference = abs(actual - wanted)
        differences.append(difference)
        require(difference <= 1e-10, f'Score mismatch {where}: {actual} != {wanted}')
    for tag in TAGS:
        raw, style = results['models'][tag]['raw'], results['models'][tag]['length_markdown_controlled']
        require(raw['expanded_rows'] == counts[tag]['expanded_rows'], f'Wrong decisive weight count: {tag}')
        compare(raw['weighted_direct_mean'], counts[tag]['weighted_direct_mean'], f'{tag}/weighted_direct_mean')
        compare(raw['official_bootstrap_mean'], expected['raw'][tag]['point'], f'{tag}/raw_center')
        compare(style['official_bootstrap_median'], expected['controlled'][tag]['point'], f'{tag}/controlled_center')
        for kind, actual in [('raw', raw), ('controlled', style)]:
            require(len(actual['ci90']) == 2, f'Wrong CI size: {tag}/{kind}')
            for index in (0, 1):
                compare(actual['ci90'][index], expected[kind][tag]['ci90'][index], f'{tag}/{kind}/ci{index}')
    # Reject any input/state changes during this read-only verification.
    for path, expected_hash in {**files, **all_records, **evidence}.items():
        require(file_hash(path) == expected_hash, f'Input changed during audit: {path}')
    return {'status': 'passed', 'candidates': list(TAGS), 'questions': 500, 'answers': 3500,
        'judgments': 3000, 'current_games': 6000, 'archived_attempts': len(archive_paths),
        'total_recorded_api_attempts': 6000 + len(archive_paths), 'retry_counts': retry_counts,
        'initial_invalid_games': sum(row['games_retried'] for row in retry_counts.values()),
        'retry_policy_binding': None if retry_binding is None else {
            key: retry_binding[key] for key in ('policy', 'policy_sha256', 'helper_sha256', 'declared_at')},
        'exact_request_and_parse_checks': 6000, 'official_bootstrap_rounds': 100, 'seed': 42,
        'numeric_comparisons': len(differences), 'max_absolute_difference': max(differences),
        'direct_scores': counts, 'official_scores': expected,
        'results_sha256': file_hash(results_path), 'manifest_sha256': file_hash(manifest_path),
        'helper_sha256': file_hash(__file__), 'upstream_sha256': PINS,
        'input_files_sha256': files, 'state_file_count': len(all_records),
        'state_files_digest': digest(all_records), 'identity_files_sha256': evidence,
        'limitations': ['Reproduces official expanded-row bootstrap, not prompt-cluster intervals.',
                       'Verifies exported judgments and recorded requests; does not assess judge factual correctness.',
                       'AST extraction preserves official function bodies; only input lookup and dataframe capture are substituted.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.suite, args.results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
