"""Bounded retry/fallback lifecycle against real frozen JudgeRun records, offline."""
import copy
import importlib.util
import json
from pathlib import Path
import threading

import pytest

from test_arena_exclusion_runner import (suite as fixture_suite, frozen_run,
    load_module, response, SOURCE_SUITE, TAGS, BASELINE)

ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / 'scripts/arena_retry_fallback.py'
    assert path.is_file(), 'The bounded retry/fallback runner is not implemented'
    spec = importlib.util.spec_from_file_location('retry_fallback_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    path = fixture_suite.__wrapped__(tmp_path_factory.mktemp('retry-source'))
    for name in ('resilient_judge.py', 'resilient_policy.json'):
        (path / name).write_bytes((SOURCE_SUITE / name).read_bytes())
    run = frozen_run(path)
    run.prepare()
    judge = load_module(path / 'source/scripts/judge_arena_hard.py')
    exclusions = []
    hashes = {}
    for tag in TAGS:
        for uid in run.questions:
            for order in (0, 1):
                request = run.request(tag, uid, order)
                invalid = (tag, uid, order) in {('base', 'u0', 0), ('base', 'u1', 1), ('lam8', 'u499', 0)}
                record = dict(response('No verdict' if invalid else '[[B>A]]'), tag=tag, uid=uid,
                    order=order, status='invalid' if invalid else 'valid', score=None if invalid else 'B>A',
                    attempt=4 if (tag, uid, order) == ('lam8', 'u499', 0) else 0,
                    local_request_id=f'{tag}-{uid}-{order}', request=request,
                    request_sha256=judge.digest(request), baseline_model=BASELINE, judge_model='gpt-4o',
                    protocol_sha256=judge.digest(run.protocol), started_at='2026-09-17T00:00:00+00:00',
                    finished_at='2026-09-17T00:00:01+00:00')
                if (tag, uid, order) == ('base', 'u1', 1):
                    record.update(answer=None, score=None, finish_reason='content_filter')
                game = run.game_path(tag, uid, order)
                game.parent.mkdir(parents=True, exist_ok=True)
                game.write_text(json.dumps(record))
                import hashlib
                sha = hashlib.sha256(game.read_bytes()).hexdigest()
                hashes[str(game)] = sha
                if invalid:
                    exclusions.append(dict(model=tag, uid=uid, order=order,
                        record_classification='judge_failed', game_sha256=sha))
    billing = dict(start_date='2026-09-01', usage0_cny=103.817238, latest_usage_cny=150.024436,
        spent_cny=150.024436-103.817238, dispatch_guard_cny=144.0)
    (run.directory / 'state/billing.json').write_text(json.dumps(billing))
    scores = path / 'continuation-with-exclusions-v2/scores'
    scores.mkdir(parents=True)
    (scores / 'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exclusions))
    (scores / 'results.json').write_text(json.dumps(dict(inputs_sha256=hashes)))
    return path


@pytest.fixture
def prepared(source, tmp_path):
    mod = runner()
    campaign = tmp_path / 'campaign'
    mod.prepare_campaign(source, campaign, expected_targets=2)
    return mod, campaign


class Relay:
    def __init__(self, outcomes=(), usages=()):
        self.outcomes = iter(outcomes)
        self.usages = iter(usages)
        self.calls = []
        self.lock = threading.Lock()
        self.usage_calls = 0

    def usage(self, start):
        self.usage_calls += 1
        value = next(self.usages, 150.024436)
        if isinstance(value, Exception):
            raise value
        return value

    def judge_call(self, request):
        with self.lock:
            self.calls.append(copy.deepcopy(request))
            outcome = next(self.outcomes, '[[B>A]]')
        if isinstance(outcome, Exception):
            raise outcome
        result = response(outcome)
        result['response_model'] = request['model'] + '-2024-01-01'
        if outcome is None:
            result['finish_reason'] = 'content_filter'
        return result

    def close(self):
        pass


def invoke(prepared, relay, **kwargs):
    mod, campaign = prepared
    return mod.run_campaign(campaign, workers=kwargs.pop('workers', 1),
                            relay_factory=lambda: relay, **kwargs)


def test_first_valid_stops_and_originals_are_immutable(prepared, source):
    mod, campaign = prepared
    before = {p: p.read_bytes() for p in campaign.rglob('original.json')}
    relay = Relay(['[[A>B]]', '[[B>A]]'])
    report = invoke(prepared, relay)
    assert report['complete'] and report['dispatched_this_run'] == 2
    manifest, rows = mod.load_resolution(campaign)
    assert [r['resolution'] for r in rows] == ['gpt4o', 'gpt4o']
    assert all(len(r['attempts']) == 2 for r in rows)
    assert rows[0]['selected_record']['score'] == 'A>B'
    assert manifest['source_suite'] == str(source)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert len(manifest['source_game_files_sha256']) == 6000
    assert invoke(prepared, Relay())['dispatched_this_run'] == 0


def test_fifth_gpt4o_success_never_uses_fallback(prepared):
    relay = Relay(['invalid'] * 3 + ['[[A=B]]', '[[B>A]]'])
    invoke(prepared, relay)
    rows = prepared[0].load_resolution(prepared[1])[1]
    assert len(rows[0]['attempts']) == 5
    assert rows[0]['resolution'] == 'gpt4o'
    assert all(r['model'] == 'gpt-4o' for r in relay.calls)


def test_fallback_after_exactly_five_invalid_changes_model_only(prepared):
    relay = Relay(['invalid'] * 4 + ['[[B>>A]]', '[[B>A]]'])
    invoke(prepared, relay)
    rows = prepared[0].load_resolution(prepared[1])[1]
    assert rows[0]['resolution'] == 'gpt41' and len(rows[0]['attempts']) == 6
    assert [r['model'] for r in relay.calls[:5]] == ['gpt-4o'] * 4 + ['gpt-4.1']
    assert relay.calls[4] == {**relay.calls[0], 'model': 'gpt-4.1'}
    assert rows[0]['selected_record']['judge_model'] == 'gpt-4.1'


def test_invalid_fallback_is_terminal_and_content_filter_is_retryable(prepared):
    relay = Relay([None] * 5 + ['[[B>A]]'])
    report = invoke(prepared, relay)
    rows = prepared[0].load_resolution(prepared[1])[1]
    assert report['complete'] and rows[0]['resolution'] == 'failed'
    assert rows[0]['selected_record'] is None and len(rows[0]['attempts']) == 6
    assert invoke(prepared, Relay())['dispatched_this_run'] == 0


def test_partial_resume_never_duplicates_and_missing_completion_blocks_scoring(prepared):
    relay = Relay(['invalid', '[[A=B]]'])
    report = invoke(prepared, relay, max_requests=1)
    assert report['status'] == 'partial' and not report['complete']
    assert not (prepared[1] / 'complete.json').exists()
    with pytest.raises((RuntimeError, ValueError), match='incomplete|complete|pending'):
        prepared[0].load_resolution(prepared[1])
    report = invoke(prepared, relay)
    assert report['complete'] and report['dispatched_this_run'] == 2
    assert len(relay.calls) == 3


@pytest.mark.parametrize('damage', ['ambiguous', 'inflight', 'request', 'hash', 'counter', 'extra'])
def test_ambiguous_or_corrupt_history_blocks_all_resume(prepared, damage):
    mod, campaign = prepared
    invoke(prepared, Relay(['invalid']), max_requests=1)
    record_path = next((campaign / 'attempts').rglob('record.json'))
    record = json.loads(record_path.read_text())
    if damage == 'ambiguous': record['status'] = 'ambiguous'
    elif damage == 'inflight': record_path.unlink()
    elif damage == 'request': record['request']['messages'][0]['content'] += ' altered'
    elif damage == 'hash': record['request_sha256'] = '0' * 64
    elif damage == 'counter': record['total_attempt'] = 6
    elif damage == 'extra':
        other = campaign / 'attempts/lam8/u499-0/02'
        other.mkdir(parents=True)
        (other / 'record.json').write_text(json.dumps(record))
    if damage != 'inflight': record_path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        mod.run_campaign(campaign, relay_factory=lambda: pytest.fail('must block before relay'))


def test_transport_failure_is_saved_once_and_never_resubmitted(prepared):
    relay = Relay([RuntimeError('secret-not-for-logs')])
    with pytest.raises((RuntimeError, ValueError), match='blocked|ambiguous'):
        invoke(prepared, relay)
    assert len(relay.calls) == 1
    assert 'secret-not-for-logs' not in ''.join(p.read_text() for p in prepared[1].rglob('*.json'))
    with pytest.raises((RuntimeError, ValueError)):
        invoke(prepared, Relay())


def test_billing_retries_are_free_bounded_and_guard_prevents_batch(prepared, monkeypatch):
    monkeypatch.setattr(prepared[0].time, 'sleep', lambda _: None)
    relay = Relay(usages=[RuntimeError('secret')] * 3)
    with pytest.raises(RuntimeError, match='billing|usage'):
        invoke(prepared, relay)
    assert relay.usage_calls == 3 and relay.calls == []
    relay = Relay(usages=[250.0])
    with pytest.raises(RuntimeError, match='budget|guard'):
        invoke(prepared, relay)
    assert relay.calls == []


def test_billing_guard_stops_after_bounded_batch(prepared):
    relay = Relay(usages=[150.024436, 250.0])
    with pytest.raises(RuntimeError, match='budget|guard'):
        invoke(prepared, relay, workers=1)
    assert len(relay.calls) == 1
    assert not (prepared[1] / 'complete.json').exists()


def test_lock_blocks_duplicate_controller(prepared):
    mod, campaign = prepared
    import fcntl
    with (campaign / '.controller.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='lock|controller'):
            invoke(prepared, Relay())


def test_target_set_cannot_be_reduced_even_with_updated_manifest_checksum(prepared):
    mod, campaign = prepared
    path = campaign / 'manifest.json'
    manifest = json.loads(path.read_text())
    removed = manifest['targets'].pop()
    manifest['target_count'] = 1
    path.write_text(json.dumps(manifest))
    (campaign / 'manifest.sha256').write_text(mod.file_hash(path) + '\n')
    (campaign / 'originals' / removed['tag'] / f"{removed['uid']}-{removed['order']}" / 'original.json').unlink()
    with pytest.raises((ValueError, RuntimeError), match='target|Target'):
        mod.run_campaign(campaign, relay_factory=lambda: pytest.fail('target changes must block'))



def test_unrecognized_returned_model_blocks_before_next_request(prepared):
    relay = Relay()
    def bad_model(request):
        relay.calls.append(request)
        return dict(response('[[A=B]]'), response_model='gpt-4o-mini-2024-07-18')
    relay.judge_call = bad_model
    with pytest.raises((ValueError, RuntimeError), match='model|blocked'):
        invoke(prepared, relay)
    assert len(relay.calls) == 1
    record = json.loads(next((prepared[1] / 'attempts').rglob('record.json')).read_text())
    assert record['response_model'] == 'gpt-4o-mini-2024-07-18'


def test_completed_attempts_resume_after_marker_crash_with_free_billing_only(prepared):
    invoke(prepared, Relay())
    (prepared[1] / 'complete.json').unlink()
    relay = Relay()
    result = prepared[0].run_campaign(prepared[1], relay_factory=lambda: relay)
    assert result['complete'] and result['dispatched_this_run'] == 0
    assert relay.calls == [] and relay.usage_calls == 1
    prepared[0].load_resolution(prepared[1])


@pytest.mark.parametrize('options', [{'workers': 0}, {'workers': 33}, {'workers': True},
                                     {'max_requests': -1}, {'max_requests': True}])
def test_invalid_dispatch_limits_never_create_campaign(tmp_path, options):
    campaign = tmp_path / 'absent'
    with pytest.raises(ValueError):
        runner().run_campaign(campaign, **options)
    assert not campaign.exists()


def test_same_target_attempts_stay_serial_across_parallel_batches(prepared):
    relay = Relay(['invalid', 'invalid', '[[A=B]]', '[[B>A]]'])
    result = invoke(prepared, relay, workers=2)
    assert result['complete'] and result['dispatched_this_run'] == 4
    rows = prepared[0].load_resolution(prepared[1])[1]
    assert all(len(row['attempts']) == 3 for row in rows)
    assert len({json.dumps(request['messages']) for request in relay.calls[:2]}) == 2
    for row in rows:
        evidence = row['attempts']
        third = json.loads(Path(evidence[2]['record_path']).read_text())
        assert third['predecessor_record_sha256'] == evidence[1]['record_sha256']


def test_running_progress_waits_for_final_billing_before_claiming_complete(prepared):
    relay = Relay()
    observed = []
    def usage(start):
        relay.usage_calls += 1
        if relay.usage_calls == 2:
            observed.append(json.loads((prepared[1] / 'progress.json').read_text()))
        return 150.024436
    relay.usage = usage
    result = invoke(prepared, relay, workers=2)
    assert observed[0]['status'] == 'running' and observed[0]['complete'] is False
    assert result['complete'] and result['status'] == 'complete'
