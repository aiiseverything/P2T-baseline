"""One explicit transport-recovery exception; offline fixtures only."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent


def verifier():
    path = SUITE / 'verify_manual_recovery.py'
    assert path.exists(), 'Manual transport verification is not implemented'
    spec = importlib.util.spec_from_file_location('arena_manual_qa_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair(tmp_path):
    api = verifier()
    base = api.load_original(SUITE)
    request = {'model': 'gpt-4.1', 'temperature': 0.0, 'max_tokens': 16000,
               'messages': [{'role': 'system', 'content': 'official system'},
                            {'role': 'user', 'content': 'literal\u2028prompt'}]}
    old = {'tag': 'base', 'uid': '34fd667185674f47', 'order': 1, 'attempt': 0,
           'status': 'ambiguous', 'error_type': 'RemoteProtocolError',
           'local_request_id': 'original-id', 'request': request, 'request_sha256': base.digest(request),
           'started_at': '2026-09-17T00:00:00+00:00', 'finished_at': '2026-09-17T00:01:00+00:00'}
    current = {key: value for key, value in old.items() if key != 'error_type'}
    current.update(attempt=1, status='valid', local_request_id='recovered-id',
                   supersedes_local_request_id='original-id', started_at='2026-09-17T00:02:01+00:00',
                   finished_at='2026-09-17T00:02:10+00:00', answer='Final [[B>>A]]', score='B>>A',
                   finish_reason='stop', usage={'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7})
    path = tmp_path / 'attempts/base/34fd667185674f47-1/original-id.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2))
    approval = {'policy': 'arena_manual_single_transport_recovery_v1', 'declared_at': '2026-09-17T00:02:00+00:00',
                'tag': old['tag'], 'uid': old['uid'], 'order': old['order'], 'original_attempt': 0,
                'original_record_sha256': base.file_hash(path), 'original_local_request_id': old['local_request_id'],
                'request_sha256': old['request_sha256'], 'error_type': old['error_type'],
                'original_started_at': old['started_at'], 'original_finished_at': old['finished_at'],
                'max_additional_attempts': 1}
    return api, base, request, old, current, approval, path


def verify_pair(pair):
    api, base, request, old, current, approval, path = pair
    parser = base.official_functions(SUITE / 'source/third_party/arena_hard')['get_score']
    state_dir = path.parents[3]
    return api.verify_manual_chain(base, current, state_dir, *api.GAME, request, parser, approval)


def test_exact_authorized_ambiguous_archive_passes_without_relabeling(pair):
    result = verify_pair(pair)
    assert result['request_ids'] == {'original-id', 'recovered-id'}
    assert len(result['archives']) == 1
    assert json.loads(pair[-1].read_text())['status'] == 'ambiguous'


@pytest.mark.parametrize('damage', ['other_game', 'late_approval', 'second_retry', 'archive_bytes',
                                  'request', 'request_hash', 'original_id', 'prior_answer',
                                  'prior_usage', 'prior_score', 'error_type', 'wrong_status',
                                  'verdict', 'truncation', 'usage', 'timeline', 'max_attempts'])
def test_no_other_transport_or_incomplete_recovery_is_accepted(pair, damage):
    api, base, request, old, current, approval, path = pair
    if damage == 'other_game': approval['tag'] = 'lam2'
    elif damage == 'late_approval': approval['declared_at'] = '2026-09-17T00:03:00+00:00'
    elif damage == 'second_retry': current['attempt'] = 2
    elif damage == 'archive_bytes': path.write_text(path.read_text() + '\n')
    elif damage == 'request': current['request'] = copy.deepcopy(request);current['request']['model'] = 'other'
    elif damage == 'request_hash': current['request_sha256'] = '0' * 64
    elif damage == 'original_id': approval['original_local_request_id'] = 'other'
    elif damage == 'verdict': current['score'] = 'A>B'
    elif damage == 'truncation': current['finish_reason'] = 'length'
    elif damage == 'usage': current['usage']['total_tokens'] = 8
    elif damage == 'timeline': current['finished_at'] = '2026-09-17T00:01:59+00:00'
    elif damage == 'max_attempts': approval['max_additional_attempts'] = 2
    else:
        if damage == 'prior_answer': old['answer'] = 'possible answer'
        elif damage == 'prior_usage': old['usage'] = {'total_tokens': 1}
        elif damage == 'prior_score': old['score'] = 'A>B'
        elif damage == 'error_type': old['error_type'] = 'TimeoutError'
        elif damage == 'wrong_status': old['status'] = 'invalid'
        path.write_text(json.dumps(old))
        approval['original_record_sha256'] = base.file_hash(path)
    with pytest.raises(ValueError): verify_pair(pair)


def test_original_verifier_hash_must_match_pinned_source(tmp_path):
    api = verifier()
    (tmp_path / 'verify_final_scoring.py').write_text('# replaced verifier\n')
    with pytest.raises(ValueError, match='verifier'):
        api.load_original(tmp_path)


def test_exception_dispatch_is_exact_and_other_games_use_original_chain(pair):
    api, base, request, old, current, approval, path = pair
    seen = []
    def original(*args): seen.append(args[2:5]); return {'original': True}
    parser = base.official_functions(SUITE / 'source/third_party/arena_hard')['get_score']
    dispatch, counts = api.chain_dispatcher(base, original, approval)
    result = dispatch(current, path.parents[3], *api.GAME, request, parser, None)
    assert len(result['archives']) == 1 and counts == {'manual': 1, 'ordinary': 0}
    assert dispatch({}, None, 'lam2', 'other', 0, {}, None) == {'original': True}
    assert seen == [('lam2', 'other', 0)] and counts == {'manual': 1, 'ordinary': 1}
    with pytest.raises(ValueError, match='more than once'):
        dispatch(current, path.parents[3], *api.GAME, request, parser, None)
