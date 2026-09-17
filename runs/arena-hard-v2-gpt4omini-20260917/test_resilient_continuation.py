"""Bounded recovery preserves successful games and never retries uncertain POSTs."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_judge_only_lifecycle import suite, continuation, FakeRelay

SOURCE = Path(__file__).resolve().parent


def module():
    spec = importlib.util.spec_from_file_location('resilient_retry_test', SOURCE / 'retry_resilient_judgments.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_connection_only_eligibility():
    api = module()
    judge = continuation.judge_module(SOURCE)
    for error in ('ConnectError', 'ConnectTimeout'):
        row = {'status': 'ambiguous', 'error_type': error, 'finished_at': '2026-09-17T12:00:00Z'}
        assert api.eligible(row, judge)
        for field in api.RESPONSE_FIELDS:
            assert not api.eligible(dict(row, **{field: None}), judge)
    for error in ('ReadTimeout', 'WriteError', 'RemoteProtocolError', 'HTTPStatusError'):
        assert not api.eligible({'status': 'ambiguous', 'error_type': error,
                                'finished_at': '2026-09-17T12:00:00Z'}, judge)
    assert not api.eligible({'status': 'inflight', 'error_type': 'ConnectError'}, judge)


@pytest.fixture
def failed(suite, monkeypatch):
    api = module()
    (suite / 'continue_evaluation.py').write_bytes((SOURCE / 'continue_evaluation.py').read_bytes())
    run = continuation.load_judge_run(suite)
    class ConnectError(Exception): pass
    relay = FakeRelay(error=ConnectError('never log exception text'))
    with pytest.raises(RuntimeError):
        run.run(lambda: relay, ['u0'], workers=1, max_requests=1, budget_cny=20)
    item = ('base', 'u0', 0)
    policy = {'policy': 'arena_bounded_connection_recovery_v1',
              'declared_at': '2026-09-17T00:00:00Z', 'max_total_attempts_per_game': 5,
              'reviewed_existing_failures': {}}
    (suite / 'resilient_policy.json').write_text(json.dumps(policy))
    monkeypatch.setattr(api, 'binding', lambda _: policy)
    return suite, api, relay, item


def test_connection_retry_then_structural_retry_uses_identical_request_and_first_valid(failed):
    suite, api, relay, item = failed
    run = continuation.load_judge_run(suite)
    original = run.game_path(*item).read_bytes()
    relay.error = None; relay.text = 'No accepted verdict'
    result = api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: relay)
    assert not result['complete'] and len(relay.calls) == 2
    relay.text = '[[B>A]]'
    result = api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: relay)
    assert result['complete'] and len(relay.calls) == 3
    current = json.loads(run.game_path(*item).read_text())
    assert current['attempt'] == 2 and current['status'] == 'valid'
    assert all(call == json.loads(original)['request'] for call in relay.calls)
    archived = run.directory / 'state/attempts/base/u0-0' / (json.loads(original)['local_request_id'] + '.json')
    assert archived.read_bytes() == original
    with pytest.raises((RuntimeError, ValueError)):
        api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: pytest.fail('valid cannot retry'))


def test_five_total_attempt_limit(failed):
    suite, api, relay, item = failed
    for attempt in range(1, 5):
        result = api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: relay)
        assert not result['complete'] and len(relay.calls) == attempt + 1
    with pytest.raises((RuntimeError, ValueError), match='exhausted'):
        api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: pytest.fail('sixth attempt'))


def test_unsafe_transport_and_tampered_ancestor_rejected(failed):
    suite, api, relay, item = failed
    api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: relay)
    run = continuation.load_judge_run(suite)
    path = next((run.directory / 'state/attempts/base/u0-0').glob('*.json'))
    row = json.loads(path.read_text()); row['error_type'] = 'ReadTimeout'; path.write_text(json.dumps(row))
    with pytest.raises((RuntimeError, ValueError)):
        api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: pytest.fail('unsafe chain'))


def test_read_error_after_retry_stops_without_another_attempt(failed):
    suite, api, relay, item = failed
    class ReadTimeout(Exception): pass
    relay.error = ReadTimeout()
    with pytest.raises((RuntimeError, ValueError), match='uncertain|eligible'):
        api.retry_games(suite, [item], budget_cny=20, relay_factory=lambda: relay)
    assert len(relay.calls) == 2


def controller():
    spec = importlib.util.spec_from_file_location('resilient_controller_test', SOURCE / 'continue_resilient.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_phase_recovers_connection_then_completes_and_resumes_without_rejudging(failed):
    suite, api, relay, item = failed
    relay.error = None
    ctrl = controller()
    result = ctrl.complete_phase(suite, ['u0'], workers=12, budget=20, helper=api, relay_factory=lambda: relay)
    assert result['complete'] and len(relay.calls) == 13
    before = {p: p.read_bytes() for p in (suite / 'model_judgment/gpt-4.1/state/games').glob('*/*.json')}
    ctrl.complete_phase(suite, ['u0'], workers=12, budget=20, helper=api, relay_factory=lambda: relay)
    assert len(relay.calls) == 13 and all(p.read_bytes() == value for p, value in before.items())


def test_phase_never_retries_uncertain_judgment(failed):
    suite, api, relay, item = failed
    class ReadTimeout(Exception): pass
    relay.error = ReadTimeout()
    with pytest.raises(RuntimeError, match='uncertain|eligible'):
        controller().complete_phase(suite, ['u0'], workers=12, budget=20, helper=api, relay_factory=lambda: relay)
    assert len(relay.calls) == 2


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('unresolved', [False, True])
def test_phase_readonly_billing_failures_are_bounded_without_paid_dispatch(failed, wrapped, unresolved):
    import httpx
    suite, api, relay, item = failed
    usage_calls = []
    if not unresolved:
        continuation.load_judge_run(suite).game_path(*item).unlink()
    def broken_usage(_):
        usage_calls.append(1)
        if wrapped:
            raise RuntimeError('Cannot verify relay usage; dispatch stopped')
        raise httpx.ConnectError('billing only')
    relay.usage = broken_usage
    with pytest.raises(RuntimeError if wrapped else httpx.ConnectError):
        controller().complete_phase(suite, ['u0'], workers=12, budget=20, helper=api,
                                    relay_factory=lambda: relay, sleep=lambda _: None)
    assert len(usage_calls) == 4 and len(relay.calls) == 1


def test_phase_connection_attempt_exhaustion_stops(failed):
    suite, api, relay, item = failed
    with pytest.raises(RuntimeError, match='exhausted'):
        controller().complete_phase(suite, ['u0'], workers=12, budget=20, helper=api, relay_factory=lambda: relay)
    assert len(relay.calls) == 5


def test_phase_nonprogress_error_stops(failed, monkeypatch):
    suite, api, relay, item = failed
    calls = []
    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('disk or identity failure')
    monkeypatch.setattr(api, 'retry_games', broken)
    with pytest.raises(RuntimeError, match='disk or identity'):
        controller().complete_phase(suite, ['u0'], workers=12, budget=20, helper=api, relay_factory=lambda: relay)
    assert calls == [1]


@pytest.fixture
def lifecycle(failed, monkeypatch):
    suite, api, relay, item = failed
    ctrl = controller()
    policy = api.binding(suite)
    policy.update(source_sha256={}, original_experiment_sha256='frozen', legacy_manual_approval_sha256='legacy',
                  billing_origin={'start_date': '2026-09-01', 'usage0_cny': 100}, original_host_execution_sha256='old')
    (suite / 'resilient_policy.json').write_text(json.dumps(policy))
    (suite / 'host_execution.json').write_text('{"must_remain":"unchanged"}')
    real_load = ctrl.load
    monkeypatch.setattr(ctrl, 'load', lambda name, path: api if path.name == 'retry_resilient_judgments.py'
                        else real_load(name, path))
    relay.error = None
    phases, labels = [], []
    def phase_runner(s, uids, *, workers, budget):
        phases.append((uids, workers, budget))
        if uids is not None:
            return ctrl.complete_phase(s, uids, workers=workers, budget=budget, helper=api, relay_factory=lambda: relay)
        return {'complete': True}
    from test_judge_only_lifecycle import VERIFIED
    def runner(s, label, command):
        labels.append(label)
        if label == 'resilient_score':
            (suite / 'scores').mkdir()
            (suite / 'scores/results.json').write_text('{}')
            (suite / 'scores/results.csv').write_text('model,score\nbase,1\n')
        return {'returncode': 0, 'stdout': json.dumps(VERIFIED), 'log': 'offline'}
    return suite, ctrl, relay, phase_runner, runner, phases, labels


def test_controller_pilot_cost_full_verifier_and_resume_preserve_old_host(lifecycle):
    suite, ctrl, relay, phase, runner, phases, labels = lifecycle
    before = (suite / 'host_execution.json').read_bytes()
    result = ctrl.continue_suite(suite, runner=runner, phase_runner=phase)
    assert result['state'] == 'evaluation_complete'
    assert [entry[1] for entry in phases] == [12, 32]
    assert len(relay.calls) == 61
    assert not (suite / 'cost_decision.json').exists()
    assert (suite / 'host_execution.json').read_bytes() == before
    first = (suite / 'resilient_cost_decision.json').read_bytes()
    ctrl.continue_suite(suite, runner=runner, phase_runner=phase)
    assert len(relay.calls) == 61 and (suite / 'resilient_cost_decision.json').read_bytes() == first


def test_controller_rejects_changed_pilot_cost_before_new_paid_work(lifecycle):
    suite, ctrl, relay, phase, runner, phases, labels = lifecycle
    ctrl.continue_suite(suite, runner=runner, phase_runner=phase)
    path = suite / 'resilient_cost_decision.json'
    decision = json.loads(path.read_text()); decision['dispatch_budget_cny'] += 1
    path.write_text(json.dumps(decision))
    count = len(phases)
    with pytest.raises(RuntimeError, match='hash changed'):
        ctrl.continue_suite(suite, runner=runner, phase_runner=phase)
    assert len(phases) == count


def test_controller_cannot_claim_complete_without_strict_final_coverage(lifecycle):
    suite, ctrl, relay, phase, runner, phases, labels = lifecycle
    def bad_runner(s, label, command):
        value = runner(s, label, command)
        if label == 'resilient_verify_final':
            value['stdout'] = '{"status":"passed", "current_games":5999}'
        return value
    with pytest.raises(RuntimeError, match='complete coverage'):
        ctrl.continue_suite(suite, runner=bad_runner, phase_runner=phase)
    assert not (suite / 'evaluation_complete.json').exists()


def test_partial_score_output_is_preserved_and_recomputed(suite):
    (suite / 'scores').mkdir()
    (suite / 'scores/results.json').write_text('{"saved":true}')
    assert not controller().scores_complete(suite)
    assert not (suite / 'scores').exists()
    archives = list((suite / 'job').glob('incomplete-scores-*'))
    assert len(archives) == 1 and (archives[0] / 'results.json').read_text() == '{"saved":true}'
