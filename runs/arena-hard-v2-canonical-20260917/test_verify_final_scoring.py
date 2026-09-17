"""Offline independent Arena verification; not part of the frozen eval stack."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent


def verifier():
    path = SUITE / 'verify_final_scoring.py'
    assert path.exists(), 'Independent verification helper is not implemented'
    spec = importlib.util.spec_from_file_location('arena_independent_qa', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_game():
    request = {'model': 'gpt-4.1', 'temperature': 0.0, 'max_tokens': 16000,
               'messages': [{'role': 'system', 'content': 'system'},
                            {'role': 'user', 'content': 'literal\u2028line\u0085prompt'}]}
    record = {'tag': 'lam2', 'uid': 'u1', 'order': 0, 'status': 'valid',
              'request': request, 'request_sha256': verifier().digest(request),
              'score': 'B>>A', 'answer': 'First [[A>B]], final [[B>>A]]',
              'finish_reason': 'stop', 'usage': {'prompt_tokens': 3, 'completion_tokens': 4,
                                                'total_tokens': 7},
              'local_request_id': 'unique-local-id', 'attempt': 0}
    game = {'score': record['score'], 'judgment': {'answer': record['answer']},
            'prompt': request['messages']}
    return request, record, game


def official_parser():
    return verifier().official_functions(SUITE / 'source/third_party/arena_hard')['get_score']


def test_real_official_parser_and_game_state_binding():
    request, record, game = fixture_game()
    verifier().verify_game(record, game, request, 'lam2', 'u1', 0, official_parser())


@pytest.mark.parametrize('damage', ['request', 'hash', 'tag', 'order', 'score', 'text',
                                  'export_prompt', 'export_score', 'finish', 'usage', 'status'])
def test_game_corruption_is_rejected(damage):
    request, record, game = fixture_game()
    record, game = copy.deepcopy(record), copy.deepcopy(game)
    if damage == 'request': record['request']['messages'][1]['content'] += ' changed'
    elif damage == 'hash': record['request_sha256'] = '0' * 64
    elif damage == 'tag': record['tag'] = 'lam4'
    elif damage == 'order': record['order'] = 1
    elif damage == 'score': record['score'] = 'A>B'
    elif damage == 'text': record['answer'] = '[[A>B]]'
    elif damage == 'export_prompt': game['prompt'][1]['content'] += ' changed'
    elif damage == 'export_score': game['score'] = 'A>B'
    elif damage == 'finish': record['finish_reason'] = 'length'
    elif damage == 'usage': record['usage']['total_tokens'] = 8
    else: record['status'] = 'ambiguous'
    with pytest.raises(ValueError):
        verifier().verify_game(record, game, request, 'lam2', 'u1', 0, official_parser())


def test_weighted_direct_means_respect_answer_order_and_decisive_weight():
    api = verifier()
    rows = [{'games': [{'score': 'A>>B'}, {'score': 'A>B'}]},
            {'games': [{'score': 'A=B'}, {'score': 'B>>A'}]}]
    assert api.direct_score(rows) == {'weighted_direct_mean': 1.5 / 8,
                                      'expanded_rows': 8, 'games': 4}


def test_literal_unicode_separators_remain_inside_json(tmp_path):
    path = tmp_path / 'rows.jsonl'
    rows = [{'prompt': 'x\u2028y\u0085z'}, {'prompt': 'next'}]
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    assert verifier().read_jsonl(path) == rows


def test_generation_cache_must_bind_actual_answer_and_frozen_policy():
    api = verifier()
    policy = {'policy_head_dtype': 'float32', 'metadata': [{'sha256': 'expected'}]}
    cache = {'outputs': {'lam2.jsonl': 'answer-sha'},
             'config': {'model_tag': 'lam2', 'policy': policy,
                        'recipe': {'temp': 1., 'n': 1, 'top_p': 1., 'top_k': -1},
                        'seed': 42, 'max_tokens': 4096,
                        'engine': {'max_model_len': 16384, 'hf_overrides': {'head_dtype': 'float32'}}}}
    api.verify_generation_cache(cache, 'lam2', policy, 'answer-sha')
    for bad in ('hash', 'policy', 'tag', 'seed', 'head'):
        changed = copy.deepcopy(cache)
        if bad == 'hash': changed['outputs']['lam2.jsonl'] = 'other-answer'
        elif bad == 'policy': changed['config']['policy']['metadata'] = []
        elif bad == 'tag': changed['config']['model_tag'] = 'lam4'
        elif bad == 'seed': changed['config']['seed'] = 43
        else: changed['config']['engine']['hf_overrides']['head_dtype'] = 'native'
        with pytest.raises(ValueError):
            api.verify_generation_cache(changed, 'lam2', policy, 'answer-sha')


def retry_chain(tmp_path, attempts=5):
    api = verifier()
    request, template, _ = fixture_game()
    binding = {'helper_sha256': 'a' * 64, 'policy': {'max_total_attempts_per_game': 5},
               'policy_sha256': 'c' * 64, 'declared_at': '2026-09-17T00:00:00Z'}
    records = []
    for index in range(attempts):
        row = copy.deepcopy(template)
        row.update(attempt=index, local_request_id=f'attempt-{index}', finished_at='2026-09-17T00:00:00Z',
                   started_at='2026-09-17T00:00:00Z')
        if index:
            row['supersedes_local_request_id'] = f'attempt-{index - 1}'
        if index >= 2:
            row.update(retry_helper_sha256=binding['helper_sha256'], retry_policy=binding['policy'],
                       retry_policy_sha256=binding['policy_sha256'])
        if index != attempts - 1:
            row.update(status='invalid', score=None, answer='repeated explanation without a verdict', finish_reason='length')
        records.append(row)
    directory = tmp_path / 'attempts/lam2/u1-0'
    directory.mkdir(parents=True)
    for row in records[:-1]:
        (directory / f"{row['local_request_id']}.json").write_text(json.dumps(row))
    return request, records, binding, directory


def test_contiguous_five_attempt_chain_retains_frozen_cli_first_retry(tmp_path):
    api = verifier()
    request, records, binding, _ = retry_chain(tmp_path)
    assert callable(getattr(api, 'verify_retry_chain', None)), 'Retry chain verification is missing'
    verified = api.verify_retry_chain(records[-1], tmp_path, 'lam2', 'u1', 0, request, official_parser(), binding)
    assert len(verified['archives']) == 4
    assert verified['request_ids'] == {f'attempt-{index}' for index in range(5)}


@pytest.mark.parametrize('damage', ['skip', 'request', 'hash', 'valid', 'ambiguous', 'inflight',
                                  'unfinished', 'parse', 'helper', 'policy', 'missing_binding',
                                  'missing_helper', 'policy_hash', 'before_declaration',
                                  'fake_invalid', 'traversal', 'six_attempts'])
def test_invalid_retry_chain_rejected(tmp_path, damage):
    api = verifier()
    request, records, binding, directory = retry_chain(tmp_path, 6 if damage == 'six_attempts' else 5)
    if damage == 'missing_binding': binding = None
    elif damage == 'missing_helper': records[-1].pop('retry_helper_sha256')
    elif damage == 'traversal': records[-1]['supersedes_local_request_id'] = '../attempt-3'
    else:
        old = records[2]
        if damage == 'skip': old['attempt'] = 1
        elif damage == 'request': old['request']['messages'][1]['content'] = 'altered'
        elif damage == 'hash': old['request_sha256'] = '0' * 64
        elif damage in ('valid', 'ambiguous', 'inflight'): old['status'] = damage
        elif damage == 'unfinished': old.pop('finished_at')
        elif damage == 'parse': old['score'] = 'A>B'
        elif damage == 'helper': old['retry_helper_sha256'] = 'b' * 64
        elif damage == 'policy': old['retry_policy'] = {'max_total_attempts': 99}
        elif damage == 'policy_hash': old['retry_policy_sha256'] = '0' * 64
        elif damage == 'before_declaration': old['started_at'] = '2026-09-16T00:00:00Z'
        elif damage == 'fake_invalid': old.update(score='A>B', answer='[[A>B]]', finish_reason='stop')
        (directory / f"{old['local_request_id']}.json").write_text(json.dumps(old))
    with pytest.raises(ValueError):
        api.verify_retry_chain(records[-1], tmp_path, 'lam2', 'u1', 0, request, official_parser(), binding)


@pytest.mark.parametrize('damage', [None, 'policy_hash', 'helper_hash', 'max_attempts',
                                  'frozen_judge', 'continuation', 'decision'])
def test_retry_host_binding_rejects_changed_policy_or_code(tmp_path, damage):
    api = verifier()
    assert callable(getattr(api, 'load_retry_binding', None)), 'Host retry provenance binding is missing'
    policy = json.loads((SUITE / 'retry_policy.json').read_text())
    policy['frozen_judge_sha256'] = 'frozen-judge-sha'
    if damage == 'max_attempts': policy['max_total_attempts_per_game'] = 6
    if damage == 'frozen_judge': policy['frozen_judge_sha256'] = 'other'
    (tmp_path / 'retry_policy.json').write_text(json.dumps(policy))
    (tmp_path / 'retry_invalid_judgments.py').write_text('# fixed retry helper\n')
    policy_sha = api.file_hash(tmp_path / 'retry_policy.json')
    helper_sha = api.file_hash(tmp_path / 'retry_invalid_judgments.py')
    host = {'retry_policy': policy, 'retry_policy_sha256': policy_sha, 'retry_helper_sha256': helper_sha,
            'continuation': {'identity': {'runners_sha256': {'retry_invalid_judgments.py': helper_sha},
                                          'inputs_sha256': {'retry_policy.json': policy_sha}}}}
    decision = {key: host[key] for key in ('retry_policy', 'retry_policy_sha256', 'retry_helper_sha256')}
    if damage == 'policy_hash': host['retry_policy_sha256'] = 'wrong'
    if damage == 'helper_hash': host['retry_helper_sha256'] = 'wrong'
    if damage == 'continuation': host['continuation']['identity']['inputs_sha256']['retry_policy.json'] = 'wrong'
    if damage == 'decision': decision['retry_helper_sha256'] = 'wrong'
    (tmp_path / 'host_execution.json').write_text(json.dumps(host))
    (tmp_path / 'cost_decision.json').write_text(json.dumps(decision))
    if damage:
        with pytest.raises(ValueError): api.load_retry_binding(tmp_path, 'frozen-judge-sha')
    else:
        result = api.load_retry_binding(tmp_path, 'frozen-judge-sha')
        assert result['helper_sha256'] == helper_sha and result['policy_sha256'] == policy_sha


def test_official_print_functions_match_known_neutral_synthetic_scores():
    import pandas as pd
    import numpy as np
    api = verifier()
    rng = np.random.RandomState(123)
    metadata, rows = {m: {} for m in (*api.TAGS, api.BASELINE)}, []
    for tag in api.TAGS:
        for index in range(12):
            uid = str(index)
            for model in (tag, api.BASELINE):
                metadata[model][uid] = {'token_len': int(rng.randint(5, 100)),
                    'header_count': {'h1': int(rng.randint(0, 5))},
                    'list_count': {'ordered': int(rng.randint(0, 5))},
                    'bold_count': {'**': int(rng.randint(0, 5))}}
            for outcome in (0., 1.):
                rows.append({'uid': uid, 'model': tag, 'category': 'hard_prompt', 'scores': outcome})
    result = api.official_scores(pd.DataFrame(rows), metadata,
                                  SUITE / 'source/third_party/arena_hard')
    assert set(result) == {'raw', 'controlled'}
    for tag in api.TAGS:
        assert .35 < result['raw'][tag]['point'] < .65
        assert result['controlled'][tag]['ci90'][0] < .5 < result['controlled'][tag]['ci90'][1]
    assert result == api.official_scores(pd.DataFrame(rows), metadata,
                                        SUITE / 'source/third_party/arena_hard')


def test_full_six_thousand_game_independent_verification(tmp_path):
    """Real frozen judge serialization + scorer, then independent upstream QA."""
    import sys
    import numpy as np
    api = verifier()
    suite = tmp_path / 'suite'; suite.mkdir()
    (suite / 'source').symlink_to(SUITE / 'source', target_is_directory=True)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    judge = load('_qa_fixture_frozen_judge', SUITE / 'source/scripts/judge_arena_hard.py')
    scorer = load('_qa_fixture_frozen_scorer', SUITE / 'source/scripts/score_arena_hard.py')
    questions = [{'uid': f'u{i}', 'category': 'hard_prompt',
                  'prompt': f'Question {i} literal\u2028line\u0085 {{ANSWER_A}}'} for i in range(500)]
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write_rows(suite / 'question.jsonl', questions)
    rng = np.random.RandomState(718)
    answers = {}
    for tag in (*api.TAGS, api.BASELINE):
        answers[tag] = [{
            'uid': q['uid'], 'model': tag,
            'messages': [{'role': 'user', 'content': q['prompt']},
                         {'role': 'assistant', 'content': {'answer': tag + ' answer ' + q['uid']}}],
            'metadata': {'token_len': int(rng.randint(5, 100)),
                         'header_count': {f'h{i}': int(rng.randint(0, 4)) for i in range(1, 7)},
                         'list_count': {'ordered': int(rng.randint(0, 5)), 'unordered': 0},
                         'bold_count': {'**': int(rng.randint(0, 5)), '__': 0}}} for q in questions]
        write_rows(suite / 'model_answer' / f'{tag}.jsonl', answers[tag])
    upstream = suite / 'source/third_party/arena_hard'
    run = judge.JudgeRun(suite / 'model_judgment/gpt-4.1', questions, answers[api.BASELINE],
                        {tag: answers[tag] for tag in api.TAGS}, judge.load_protocol(upstream))
    run.prepare()
    for tag in api.TAGS:
        for index, q in enumerate(questions):
            for order in (0, 1):
                request = run.request(tag, q['uid'], order)
                score = 'A>>B' if index % 2 else 'A>B'
                record = {'tag': tag, 'uid': q['uid'], 'order': order, 'status': 'valid',
                    'request': request, 'request_sha256': judge.digest(request),
                    'score': score, 'answer': f'Reasoning\u2028 then [[{score}]]', 'finish_reason': 'stop',
                    'local_request_id': f'{tag}-{index}-{order}', 'attempt': 0,
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}}
                path = run.game_path(tag, q['uid'], order)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False))
    run.export()
    names = ['source/scripts/judge_arena_hard.py', 'source/scripts/score_arena_hard.py',
             'source/third_party/arena_hard/gen_answer.py',
             'source/third_party/arena_hard/utils/add_markdown_info.py',
             'question.jsonl', f'model_answer/{api.BASELINE}.jsonl']
    names += ['source/third_party/arena_hard/' + name for name in api.PINS]
    policy = {'policy_head_dtype': 'float32', 'metadata': []}
    manifest = {'models': {tag: {'policy': policy} for tag in api.TAGS},
                'files_sha256': {name: api.file_hash(suite / name) for name in names}}
    (suite / 'experiment.json').write_text(json.dumps(manifest))
    (suite / 'manifests').mkdir()
    for tag in api.TAGS:
        cache = {'outputs': {f'{tag}.jsonl': api.file_hash(suite / 'model_answer' / f'{tag}.jsonl')},
            'config': {'model_tag': tag, 'policy': policy,
                'recipe': {'temp': 1., 'n': 1, 'top_p': 1., 'top_k': -1}, 'seed': 42, 'max_tokens': 4096,
                'engine': {'max_model_len': 16384, 'hf_overrides': {'head_dtype': 'float32'}}}}
        (suite / 'manifests' / f'{tag}.json').write_text(json.dumps(cache))
    # One real exported game succeeds only on its fifth identical attempt.
    # Earlier attempts remain archived, including the original CLI's attempt 1.
    retry_policy = json.loads((SUITE / 'retry_policy.json').read_text())
    retry_policy['frozen_judge_sha256'] = api.file_hash(suite / 'source/scripts/judge_arena_hard.py')
    (suite / 'retry_policy.json').write_text(json.dumps(retry_policy))
    (suite / 'retry_invalid_judgments.py').write_text('# synthetic helper identity\n')
    policy_sha, helper_sha = api.file_hash(suite / 'retry_policy.json'), api.file_hash(suite / 'retry_invalid_judgments.py')
    host = {'retry_policy': retry_policy, 'retry_policy_sha256': policy_sha, 'retry_helper_sha256': helper_sha}
    (suite / 'host_execution.json').write_text(json.dumps(host))
    state_path = run.game_path('lam2', 'u0', 0)
    current = json.loads(state_path.read_text())
    archive_dir = run.directory / 'state/attempts/lam2/u0-0'; archive_dir.mkdir(parents=True)
    for index in range(5):
        row = copy.deepcopy(current)
        row.update(attempt=index, local_request_id=f'lam2-u0-0-retry-{index}',
                   started_at=retry_policy['declared_at'], finished_at=retry_policy['declared_at'])
        if index: row['supersedes_local_request_id'] = f'lam2-u0-0-retry-{index-1}'
        if index >= 2:
            row.update(retry_policy=retry_policy, retry_policy_sha256=policy_sha, retry_helper_sha256=helper_sha)
        if index < 4:
            row.update(status='invalid', answer='loop without verdict', score=None, finish_reason='length')
            (archive_dir / f"{row['local_request_id']}.json").write_text(json.dumps(row))
        else:
            state_path.write_text(json.dumps(row))
    scorer.main(['--questions', str(suite / 'question.jsonl'),
                 '--answers-dir', str(suite / 'model_answer'),
                 '--judgments-dir', str(suite / 'model_judgment/gpt-4.1'),
                 '--output', str(suite / 'scores')])
    result = api.verify(suite)
    assert result['status'] == 'passed' and result['current_games'] == 6000
    assert result['numeric_comparisons'] == 42
    assert result['max_absolute_difference'] <= 1e-10
    assert result['total_recorded_api_attempts'] == 6004 and result['archived_attempts'] == 4
    assert result['initial_invalid_games'] == 1
    assert result['retry_counts']['lam2'] == {'games_retried': 1, 'extra_attempts': 4,
                                            'initial_invalid_by_order': {'0': 1, '1': 0}}
    assert all(row['weighted_direct_mean'] == .5 for row in result['direct_scores'].values())
    state_path = run.game_path('lam2', 'u0', 0)
    previous = state_path.read_text()
    record = json.loads(previous)
    record['request']['messages'][1]['content'] += ' altered'
    record['request_sha256'] = judge.digest(record['request'])
    state_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Official request mismatch'):
        api.verify(suite)
    state_path.write_text(previous)
    path = suite / 'scores/results.json'
    result = json.loads(path.read_text())
    result['models']['grpo']['raw']['weighted_direct_mean'] += .01
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Score mismatch grpo/weighted_direct_mean'):
        api.verify(suite)
