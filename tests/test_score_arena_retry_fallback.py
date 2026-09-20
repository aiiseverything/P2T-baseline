"""Offline retry overlays retain whole pairs and bind actual judge evidence."""
import copy
import importlib
import json
from pathlib import Path

import pytest

from test_arena_exclusion_scoring import small_run, change, fail_game
from test_arena_retry_fallback import source, prepared, Relay, invoke
from test_arena_retry_transport_recovery import paused, recovery_module


def scorer():
    path = Path(__file__).resolve().parents[1] / 'scripts/score_arena_retry_fallback.py'
    assert path.is_file(), 'The retry/fallback scorer is missing'
    return importlib.import_module('scripts.score_arena_retry_fallback')


@pytest.fixture
def campaign(small_run, tmp_path):
    run, judge = small_run
    old = importlib.import_module('scripts.score_arena_with_exclusions')
    strict = importlib.import_module('scripts.score_arena_hard')
    fail_game(run, 'base', 'q0', 0)
    fail_game(run, 'grpo', 'q1', 1)
    fail_game(run, 'lam8', 'q2', 0)
    change(run, 'lam8', 'q2', 0, attempt=4)
    source = old.collect_subsets(run, judge, strict)
    directory = tmp_path / 'campaign'
    directory.mkdir()
    targets, resolutions = [], []
    hashes = {}
    for tag, uid, order, final, final_model, score in [
            ('base', 'q0', 0, 3, 'gpt-4o', 'B>>A'),
            ('grpo', 'q1', 1, 6, 'gpt-4.1', 'A=B')]:
        original_path = run.game_path(tag, uid, order).resolve()
        original = json.loads(original_path.read_text())
        target = {'tag': tag, 'uid': uid, 'order': order,
                  'original_path': str(original_path),
                  'original_sha256': old.file_hash(original_path),
                  'request_sha256': original['request_sha256']}
        targets.append(target)
        attempts = []
        for number in range(1, final + 1):
            row = copy.deepcopy(original)
            model = 'gpt-4o' if number <= 5 else 'gpt-4.1'
            row['judge_model'] = model
            row['request']['model'] = model
            row['request_sha256'] = judge.digest(row['request'])
            if number > 1:
                row.update(total_attempt=number, attempt=number - 1,
                           local_request_id=f'{tag}-{number}', response_model=model)
            if number == final:
                row.update(status='valid', score=score, answer=f'[[{score}]]')
            path = directory / f'{tag}-{number}.json'
            path.write_text(json.dumps(row))
            hashes[str(path)] = old.file_hash(path)
            attempts.append({'total_attempt': number, 'judge_model': model,
                'record_path': str(path), 'record_sha256': old.file_hash(path),
                'status': row['status'], 'request_sha256': row['request_sha256'],
                'local_request_id': row.get('local_request_id'),
                'response_model': row.get('response_model')})
        resolutions.append({'tag': tag, 'uid': uid, 'order': order,
            'resolution': 'gpt4o' if final_model == 'gpt-4o' else 'gpt41',
            'selected_record': row, 'attempts': attempts})
    manifest = {'source_suite': str(tmp_path), 'targets': targets,
                'source_files_sha256': dict(source.files_sha256),
                'campaign_files_sha256': hashes, 'manifest_sha256': 'fixture'}
    return run, judge, strict, source, manifest, resolutions, directory


def variants(campaign):
    _, judge, strict, source, manifest, resolutions, _ = campaign
    return scorer().collect_variants(source, manifest, resolutions, judge, strict)


def test_variants_recover_only_valid_pairs_and_keep_actual_judge(campaign):
    run, _, _, _, _, _, _ = campaign
    original = {p: p.read_bytes() for p in (run.directory / 'state/games').rglob('*.json')}
    result = variants(campaign)
    pure, mixed = result['pure_gpt4o'], result['mixed_gpt4o_gpt41']
    assert pure.bundle.valid_uids['base'] == ['q0', 'q1', 'q2']
    assert pure.bundle.valid_uids['grpo'] == ['q0', 'q2']
    assert mixed.bundle.valid_uids['grpo'] == ['q0', 'q1', 'q2']
    assert pure.bundle.common_uids == ['q0']
    assert mixed.bundle.common_uids == ['q0', 'q1']
    assert pure.bundle.counts['grpo']['valid_but_discarded_partner_games'] == 1
    assert mixed.bundle.counts['grpo']['valid_but_discarded_partner_games'] == 0
    assert pure.bundle.counts['lam8']['judge_failed_games'] == 1
    assert mixed.bundle.counts['lam8']['excluded_prompts'] == 1
    selected = next(r for r in mixed.provenance if (r['tag'], r['uid'], r['order']) == ('grpo', 'q1', 1))
    assert selected['judge_model'] == 'gpt-4.1'
    assert selected['source'] == 'retry_campaign' and selected['total_attempt'] == 6
    excluded = next(r for r in pure.provenance if (r['tag'], r['uid'], r['order']) == ('grpo', 'q1', 1))
    assert excluded['judge_model'] == 'gpt-4o' and excluded['source'] == 'original'
    assert excluded['retained_pair'] is False
    assert len(pure.provenance) == len(mixed.provenance) == 36
    assert all(p.read_bytes() == before for p, before in original.items())


def test_unchanged_partner_and_decisive_orientation_reuse_pinned_math(campaign):
    run, _, strict, _, _, _, _ = campaign
    api = scorer()
    result = variants(campaign)['pure_gpt4o']
    rows = result.bundle.battles.query("model == 'base'")
    assert rows.scores.tolist() == [1, 1, 1, 1, 1, 0, 1, 0]
    scores = api.original_score.score_subset(result.bundle.battles, result.bundle.answers,
                                            strict, upstream=strict.UPSTREAM, rounds=3)
    assert scores['models']['base']['raw']['weighted_direct_mean'] == .75
    partner = next(r for r in result.provenance if (r['tag'], r['uid'], r['order']) == ('base', 'q0', 1))
    assert partner['source'] == 'original'
    assert partner['record_path'] == str(run.game_path('base', 'q0', 1).resolve())
    assert partner['record_sha256'] == api.original_score.file_hash(partner['record_path'])


def rewrite_attempt(campaign, resolution_index, attempt_index, **updates):
    _, _, _, _, manifest, resolutions, _ = campaign
    evidence = resolutions[resolution_index]['attempts'][attempt_index]
    path = Path(evidence['record_path'])
    record = json.loads(path.read_text())
    record.update(updates)
    path.write_text(json.dumps(record))
    sha = importlib.import_module('scripts.score_arena_with_exclusions').file_hash(path)
    evidence['record_sha256'] = sha
    manifest['campaign_files_sha256'][str(path)] = sha
    return record


def test_still_failed_fallback_keeps_both_variants_excluded(campaign):
    *_, resolutions, _ = campaign
    record = rewrite_attempt(campaign, 1, -1, status='invalid', score=None, answer='No verdict')
    resolutions[1].update(resolution='failed', selected_record=None)
    resolutions[1]['attempts'][-1]['status'] = 'invalid'
    result = variants(campaign)
    assert result['mixed_gpt4o_gpt41'].bundle.valid_uids['grpo'] == ['q0', 'q2']
    assert result['mixed_gpt4o_gpt41'].bundle.common_uids == ['q0']


@pytest.mark.parametrize('damage', ['target_uid', 'target_hash', 'duplicate_target',
    'missing_resolution', 'extra_resolution', 'request', 'identity', 'judge',
    'request_digest', 'attempt_hash', 'evidence_status', 'attempt_number',
    'first_valid', 'early_fallback', 'selected_record', 'pending', 'original_changed'])
def test_mismatched_or_unproven_attempts_never_enter_scores(campaign, damage):
    run, judge, _, _, manifest, resolutions, _ = campaign
    if damage == 'target_uid':
        manifest['targets'][0]['uid'] = 'q2'
    elif damage == 'target_hash':
        manifest['targets'][0]['original_sha256'] = 'wrong'
    elif damage == 'duplicate_target':
        manifest['targets'].append(copy.deepcopy(manifest['targets'][0]))
    elif damage == 'missing_resolution':
        resolutions.pop()
    elif damage == 'extra_resolution':
        resolutions.append(copy.deepcopy(resolutions[0]))
    elif damage in ('request', 'identity', 'judge', 'request_digest'):
        row = copy.deepcopy(resolutions[0]['selected_record'])
        if damage == 'request':
            row['request']['temperature'] = 1
            row['request_sha256'] = judge.digest(row['request'])
        elif damage == 'identity':
            row['uid'] = 'q2'
        elif damage == 'judge':
            row['judge_model'] = 'gpt-4.1'
        else:
            row['request_sha256'] = 'wrong'
        record = rewrite_attempt(campaign, 0, -1, **row)
        resolutions[0]['selected_record'] = record
    elif damage == 'attempt_hash':
        Path(resolutions[0]['attempts'][-1]['record_path']).write_text('{}')
    elif damage == 'evidence_status':
        resolutions[0]['attempts'][-1]['status'] = 'invalid'
    elif damage == 'attempt_number':
        resolutions[0]['attempts'][-1]['total_attempt'] = 5
    elif damage == 'first_valid':
        rewrite_attempt(campaign, 0, 1, status='valid', score='A>>B', answer='[[A>>B]]')
        resolutions[0]['attempts'][1]['status'] = 'valid'
    elif damage == 'early_fallback':
        resolutions[1]['attempts'].pop(3)
    elif damage == 'selected_record':
        resolutions[0]['selected_record']['score'] = 'A=B'
    elif damage == 'pending':
        resolutions[0].update(resolution='pending', selected_record=None)
    else:
        change(run, 'base', 'q0', 1, answer='Changed [[A>B]]')
    with pytest.raises(ValueError):
        variants(campaign)


@pytest.mark.parametrize('accepted_fallback', [True, False])
def test_cli_saves_two_variants_and_hash_bound_provenance(campaign, monkeypatch, tmp_path, accepted_fallback):
    _, judge, strict, source, manifest, resolutions, directory = campaign
    if not accepted_fallback:
        rewrite_attempt(campaign, 1, -1, status='invalid', score=None, answer='No verdict')
        resolutions[1].update(resolution='failed', selected_record=None)
        resolutions[1]['attempts'][-1]['status'] = 'invalid'
    api = scorer()
    monkeypatch.setattr(api, 'load_resolution', lambda path, require_complete: (manifest, resolutions))
    monkeypatch.setattr(api.original_score, 'load_suite',
                        lambda path: (source, strict, strict.UPSTREAM, {'status': 'passed'}))
    monkeypatch.setattr(api, 'load_frozen_judge', lambda suite: judge)
    output = tmp_path / 'scores'
    result = api.main(['--campaign', str(directory), '--output', str(output)])
    assert json.loads((output / 'summary.json').read_text()) == result
    for variant in ('pure_gpt4o', 'mixed_gpt4o_gpt41'):
        saved = json.loads((output / variant / 'results.json').read_text())
        assert saved['variant'] == variant
        assert saved['coverage']['attempted_games'] == 36
        assert saved['full_500_official_leaderboard'] is False
        assert saved['bootstrap_rounds'] == 100
        assert saved['inputs_sha256']
        provenance = output / variant / 'game_provenance.jsonl'
        assert saved['game_provenance']['sha256'] == api.original_score.file_hash(provenance)
        assert len(provenance.read_text().splitlines()) == 36
        assert (output / variant / 'results.csv').is_file()
        assert (output / variant / 'common_subset_results.csv').is_file()
    pure = json.loads((output / 'pure_gpt4o/results.json').read_text())
    mixed = json.loads((output / 'mixed_gpt4o_gpt41/results.json').read_text())
    assert pure['judge_models'] == ['gpt-4o']
    assert mixed['judge_models'] == (['gpt-4o', 'gpt-4.1'] if accepted_fallback else ['gpt-4o'])
    assert pure['coverage']['judge_failed_games'] == 2
    assert mixed['coverage']['judge_failed_games'] == (1 if accepted_fallback else 2)
    assert pure['common_valid_subset']['questions'] == 1
    assert mixed['common_valid_subset']['questions'] == (2 if accepted_fallback else 1)


def test_cli_rechecks_input_hashes_after_scoring_before_writing(campaign, monkeypatch, tmp_path):
    _, judge, strict, source, manifest, resolutions, directory = campaign
    api = scorer()
    monkeypatch.setattr(api, 'load_resolution', lambda path, require_complete: (manifest, resolutions))
    monkeypatch.setattr(api.original_score, 'load_suite',
                        lambda path: (source, strict, strict.UPSTREAM, {'status': 'passed'}))
    monkeypatch.setattr(api, 'load_frozen_judge', lambda suite: judge)
    real_score = api.original_score.score_subset
    def mutate(*args, **kwargs):
        result = real_score(*args, **kwargs)
        Path(resolutions[0]['attempts'][-1]['record_path']).write_text('{}')
        return result
    monkeypatch.setattr(api.original_score, 'score_subset', mutate)
    output = tmp_path / 'scores'
    with pytest.raises(ValueError, match='Input changed'):
        api.main(['--campaign', str(directory), '--output', str(output)])
    assert not output.exists()


def test_question_order_comes_from_validated_games_not_answer_file_order(campaign):
    expected = variants(campaign)['pure_gpt4o'].bundle.battles.copy()
    source = campaign[3]
    source.answers['base'] = dict(reversed(list(source.answers['base'].items())))
    result = variants(campaign)['pure_gpt4o'].bundle
    assert result.valid_uids['base'] == ['q0', 'q1', 'q2']
    assert result.battles.equals(expected)


def test_frozen_judge_loading_does_not_write_bytecode_into_source(tmp_path):
    path = tmp_path / 'source/scripts/judge_arena_hard.py'
    path.parent.mkdir(parents=True)
    path.write_text('VALUE = 17\n')
    module = scorer().load_frozen_judge(tmp_path)
    assert module.VALUE == 17
    assert not list(tmp_path.rglob('__pycache__'))


def test_one_failed_target_discards_its_recovered_target_partner(campaign):
    run, judge, strict, _, manifest, resolutions, directory = campaign
    old = importlib.import_module('scripts.score_arena_with_exclusions')
    fail_game(run, 'grpo', 'q1', 0)
    path = run.game_path('grpo', 'q1', 0).resolve()
    original = json.loads(path.read_text())
    target = {'tag': 'grpo', 'uid': 'q1', 'order': 0,
        'original_path': str(path), 'original_sha256': old.file_hash(path),
        'request_sha256': original['request_sha256']}
    resolution = copy.deepcopy(resolutions[1])
    resolution.update(order=0, resolution='failed', selected_record=None)
    for number, evidence in enumerate(resolution['attempts'], 1):
        row = copy.deepcopy(original)
        if number > 1:
            model = 'gpt-4o' if number < 6 else 'gpt-4.1'
            row['request']['model'] = model
            row.update(judge_model=model, total_attempt=number, local_request_id=f'extra-{number}',
                       response_model=model, request_sha256=judge.digest(row['request']))
        path = directory / f'extra-{number}.json'
        path.write_text(json.dumps(row))
        manifest['campaign_files_sha256'][str(path)] = old.file_hash(path)
        evidence.update(record_path=str(path), record_sha256=old.file_hash(path), status='invalid',
            request_sha256=row['request_sha256'], local_request_id=row.get('local_request_id'),
            response_model=row.get('response_model'))
    manifest['targets'].append(target)
    resolutions.append(resolution)
    source = old.collect_subsets(run, judge, strict)
    manifest['source_files_sha256'] = dict(source.files_sha256)
    result = scorer().collect_variants(source, manifest, resolutions, judge, strict)
    assert result['mixed_gpt4o_gpt41'].bundle.valid_uids['grpo'] == ['q0', 'q2']
    assert result['mixed_gpt4o_gpt41'].bundle.counts['grpo']['valid_but_discarded_partner_games'] == 1
    failed = [row for row in result['mixed_gpt4o_gpt41'].bundle.exclusions if row['model'] == 'grpo']
    assert [(row['order'], row['reason'], row['judge_model']) for row in failed] == [
        (0, 'missing_verdict', 'gpt-4o'), (1, 'partner_judge_failed', 'gpt-4.1')]


def test_real_runner_completion_and_evidence_feed_the_two_overlay_variants(prepared):
    mod, directory = prepared
    # First target recovers immediately; second needs all four retries and fallback.
    invoke(prepared, Relay(['[[B>A]]', 'No verdict', 'No verdict', 'No verdict',
                            'No verdict', '[[A=B]]']))
    api = scorer()
    manifest, resolutions = api.load_resolution(directory, require_complete=True)
    run, judge = mod.load_suite(manifest['source_suite'])
    old = api.original_score
    # This fixture has real frozen game validation but omits generation metadata;
    # only pair selection is exercised here, not the separate source-suite audit.
    source_bundle = old.Subsets({**run.answers, old.BASELINE: run.baseline}, None, None,
                               {}, [], [], {}, dict(manifest['source_files_sha256']))
    strict = importlib.import_module('scripts.score_arena_hard')
    selected = api.collect_variants(source_bundle, manifest, resolutions, judge, strict)
    assert len(selected['pure_gpt4o'].provenance) == 6000
    assert selected['pure_gpt4o'].bundle.counts['base']['retained_prompts'] == 499
    assert selected['mixed_gpt4o_gpt41'].bundle.counts['base']['retained_prompts'] == 500
    assert selected['mixed_gpt4o_gpt41'].bundle.counts['lam8']['retained_prompts'] == 499
    assert len(selected['pure_gpt4o'].bundle.common_uids) == 498
    assert len(selected['mixed_gpt4o_gpt41'].bundle.common_uids) == 499
    (directory / 'complete.json').unlink()
    with pytest.raises(ValueError, match='completion required'):
        api.load_resolution(directory, require_complete=True)


def add_transport_recovery(campaign, resolution_index=0):
    _, _, _, _, manifest, resolutions, directory = campaign
    old = importlib.import_module('scripts.score_arena_with_exclusions')
    resolution = resolutions[resolution_index]
    replacement = resolution['attempts'][1]
    ambiguous = json.loads(Path(replacement['record_path']).read_text())
    ambiguous.update(status='ambiguous', error_type='RemoteProtocolError',
                     local_request_id='interrupted-original-request')
    for name in ('answer', 'score', 'usage', 'response_model'):
        ambiguous.pop(name, None)
    path = directory / 'original-ambiguous.json'
    path.write_text(json.dumps(ambiguous))
    authorization = directory / 'recovery-authorization.json'
    authorization.write_text(json.dumps({'authorized': True, 'physical_replay_attempts': 1}))
    for bound in (path, authorization):
        manifest['campaign_files_sha256'][str(bound)] = old.file_hash(bound)
    recovery = {'recovery_id': 'inspected-once', 'logical_attempt': 2,
        'physical_replay_attempts': 1, 'unknown_usage_requests': 1,
        'original_ambiguous': {'record_path': str(path), 'record_sha256': old.file_hash(path),
            'local_request_id': ambiguous['local_request_id'],
            'request_sha256': ambiguous['request_sha256'], 'status': 'ambiguous',
            'error_type': 'RemoteProtocolError'},
        'replacement': {key: replacement[key] for key in
                        ('record_path', 'record_sha256', 'local_request_id', 'request_sha256')},
        'authorization_path': str(authorization), 'authorization_sha256': old.file_hash(authorization),
        'cost_uncertainty': 'Original POST may have been processed/billed; one inspected replay can add one charge.'}
    manifest['transport_recovery'] = recovery
    resolution['transport_recovery'] = copy.deepcopy(recovery)
    return recovery


@pytest.mark.parametrize('resolution_index', [0, 1])
def test_transport_recovery_preserves_scores_and_provenance_in_both_variants(campaign, resolution_index):
    before = variants(campaign)
    recovery = add_transport_recovery(campaign, resolution_index)
    after = variants(campaign)
    target = campaign[5][resolution_index]
    key = target['tag'], target['uid'], target['order']
    for name in before:
        assert after[name].bundle.battles.equals(before[name].bundle.battles)
        assert after[name].bundle.counts == before[name].bundle.counts
        row = next(r for r in after[name].provenance if (r['tag'], r['uid'], r['order']) == key)
        assert row['transport_recovery'] == recovery
        assert row['transport_recovery']['unknown_usage_requests'] == 1
        assert all(r['transport_recovery'] is None for r in after[name].provenance
                   if (r['tag'], r['uid'], r['order']) != key)
    assert len(target['attempts']) == (3 if resolution_index == 0 else 6)


@pytest.mark.parametrize('damage', ['lost_manifest_event', 'lost_row_event', 'replacement',
    'unknown_usage_hidden', 'physical_request_hidden', 'ambiguous_bytes', 'authorization_bytes',
    'original_not_ambiguous', 'original_request'])
def test_transport_recovery_requires_matching_bound_evidence(campaign, damage):
    recovery = add_transport_recovery(campaign)
    manifest, resolutions = campaign[4:6]
    if damage == 'lost_manifest_event':
        del manifest['transport_recovery']
    elif damage == 'lost_row_event':
        del resolutions[0]['transport_recovery']
    elif damage in ('replacement', 'unknown_usage_hidden', 'physical_request_hidden'):
        if damage == 'replacement':
            recovery['replacement']['record_path'] = resolutions[0]['attempts'][-1]['record_path']
        else:
            recovery['unknown_usage_requests' if damage == 'unknown_usage_hidden'
                     else 'physical_replay_attempts'] = 0
        resolutions[0]['transport_recovery'] = copy.deepcopy(recovery)
    elif damage in ('ambiguous_bytes', 'authorization_bytes'):
        path = (recovery['original_ambiguous']['record_path'] if damage == 'ambiguous_bytes'
                else recovery['authorization_path'])
        Path(path).write_text('{}')
    else:
        old = recovery['original_ambiguous']
        path = Path(old['record_path'])
        row = json.loads(path.read_text())
        if damage == 'original_not_ambiguous':
            row['status'] = old['status'] = 'invalid'
        else:
            row['request']['model'] = 'gpt-4.1'
        path.write_text(json.dumps(row))
        old['record_sha256'] = scorer().original_score.file_hash(path)
        manifest['campaign_files_sha256'][str(path)] = old['record_sha256']
        resolutions[0]['transport_recovery'] = copy.deepcopy(recovery)
    with pytest.raises(ValueError):
        variants(campaign)


@pytest.mark.parametrize('has_recovery', [True, False])
def test_cli_accounts_for_physical_replay_and_unknown_usage(campaign, monkeypatch, tmp_path, has_recovery):
    _, judge, strict, source, manifest, resolutions, directory = campaign
    recovery = add_transport_recovery(campaign) if has_recovery else None
    api = scorer()
    monkeypatch.setattr(api, 'load_resolution', lambda path, require_complete: (manifest, resolutions))
    monkeypatch.setattr(api.original_score, 'load_suite',
                        lambda path: (source, strict, strict.UPSTREAM, {'status': 'passed'}))
    monkeypatch.setattr(api, 'load_frozen_judge', lambda suite: judge)
    result = api.main(['--campaign', str(directory), '--output', str(tmp_path / 'scores')])
    expected = {'completed_new_attempts': 7, 'extra_physical_requests': int(has_recovery),
        'physical_new_requests': 8 if has_recovery else 7,
        'unknown_usage_requests': int(has_recovery), 'usage_accounting_complete': not has_recovery}
    assert result['request_accounting'] == expected
    assert result['transport_recovery'] == recovery
    for variant in ('pure_gpt4o', 'mixed_gpt4o_gpt41'):
        saved = json.loads((tmp_path / 'scores' / variant / 'results.json').read_text())
        assert saved['request_accounting'] == expected
        assert saved['transport_recovery'] == recovery
        assert saved['coverage']['attempted_games'] == 36


@pytest.mark.parametrize('replay_is_valid', [True, False])
def test_real_transport_overlay_preserves_logical_attempts_and_variant_selection(paused, replay_is_valid):
    from scripts import arena_retry_fallback as runner
    directory, target = paused
    recovery = recovery_module()
    recovery.prepare_recovery(directory, target=target)
    recovery.retry_once(directory, relay_factory=lambda: Relay(['[[B>A]]' if replay_is_valid else 'invalid']))
    outcomes = ['[[A=B]]'] if replay_is_valid else ['invalid'] * 3 + ['[[B>A]]', '[[A=B]]']
    runner.run_campaign(directory, workers=1, relay_factory=lambda: Relay(outcomes))
    api = scorer()
    manifest, resolutions = api.load_resolution(directory, require_complete=True)
    run, judge = runner.load_suite(manifest['source_suite'])
    source_bundle = api.original_score.Subsets({**run.answers, api.original_score.BASELINE: run.baseline},
        None, None, {}, [], [], {}, dict(manifest['source_files_sha256']))
    selected = api.collect_variants(source_bundle, manifest, resolutions, judge,
                                    importlib.import_module('scripts.score_arena_hard'))
    assert len(resolutions[0]['attempts']) == (2 if replay_is_valid else 6)
    assert selected['pure_gpt4o'].bundle.counts['base']['retained_prompts'] == (500 if replay_is_valid else 499)
    assert selected['mixed_gpt4o_gpt41'].bundle.counts['base']['retained_prompts'] == 500
    assert selected['mixed_gpt4o_gpt41'].bundle.counts['lam8']['retained_prompts'] == 499
    evidence = next(row for row in selected['mixed_gpt4o_gpt41'].provenance
                    if (row['tag'], row['uid'], row['order']) == ('base', 'u0', 0))
    assert evidence['transport_recovery'] == manifest['transport_recovery']
    assert evidence['judge_model'] == ('gpt-4o' if replay_is_valid else 'gpt-4.1')
    assert api._request_accounting(resolutions) == {
        'completed_new_attempts': 2 if replay_is_valid else 6, 'extra_physical_requests': 1,
        'physical_new_requests': 3 if replay_is_valid else 7, 'unknown_usage_requests': 1,
        'usage_accounting_complete': False}
