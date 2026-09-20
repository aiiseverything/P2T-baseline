"""Real frozen state transitions; only external billing/judging is replaced."""
import importlib.util
import json
from pathlib import Path
import hashlib

import pytest
from test_arena_exclusion_runner import FakeRelay, frozen_run, response, seed, suite
from test_arena_billing_retries import BillingRelay

ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / 'scripts/recover_inspected_arena_transport.py'
    assert path.exists(), 'Inspected single-retry recovery is not implemented'
    spec = importlib.util.spec_from_file_location('inspected_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(suite):
    target = seed(suite, tag='grpo', status='ambiguous', error_type='RemoteProtocolError',
                  answer=None, score=None, usage=None, finish_reason=None)
    excluded = seed(suite, tag='base', status='invalid', answer='No verdict', score=None)
    recovery = suite / 'recovery'
    recovery.mkdir()
    row = json.loads(target.read_text())
    approval = {'policy_id': 'arena_operator_inspected_transport_retry_v1',
                'target': {'model': 'grpo', 'uid': 'u0', 'order': 0,
                           'game_sha256': sha(target), 'request_sha256': row['request_sha256'],
                           'local_request_id': row['local_request_id']},
                'budget_cny': 144, 'max_paid_calls': 1,
                'expected_counts': {'missing': 5998, 'valid': 0, 'judge_failed': 1, 'blocked': 1},
                'cost_uncertainty': 'Original RemoteProtocolError may already have been charged.'}
    (recovery / 'approval.json').write_text(json.dumps(approval))
    return suite, recovery, target, excluded


def test_inspected_retry_archives_original_once_keeps_body_and_other_failed_games(prepared):
    suite, recovery, target, excluded = prepared
    before, other = target.read_bytes(), excluded.read_bytes()
    original = json.loads(before)
    relay = BillingRelay(usage_values=[RuntimeError('redacted'), 100, 100])
    waits = []
    result = api().retry_once(suite, recovery, relay_factory=lambda: relay, sleep=waits.append)
    assert len(relay.calls) == 1 and relay.calls[0] == original['request']
    assert waits == [1] and relay.closed
    row = json.loads(target.read_text())
    assert row['attempt'] == 1 and row['status'] == 'valid'
    assert row['supersedes_local_request_id'] == original['local_request_id']
    archive = target.parents[2] / 'attempts/grpo/u0-0' / (original['local_request_id'] + '.json')
    assert archive.read_bytes() == before and excluded.read_bytes() == other
    assert result['status'] == 'complete' and result['resume_eligible'] is True
    assert result['paid_calls'] == 1 and result['classification'] == 'valid'
    with pytest.raises((ValueError, RuntimeError)):
        api().retry_once(suite, recovery, relay_factory=lambda: pytest.fail('No second retry'))
    assert archive.read_bytes() == before


@pytest.mark.parametrize('output,classification', [
    (response('No verdict'), 'judge_failed'),
    (response(None, 'content_filter'), 'judge_failed'),
    (RuntimeError('must-not-appear-credential'), 'blocked'),
    (response(None, 'stop'), 'blocked'),
])
def test_one_retry_preserves_real_outcome_without_inventing_score(prepared, output, classification):
    suite, recovery, target, excluded = prepared
    before = excluded.read_bytes()
    relay = FakeRelay([output])
    result = api().retry_once(suite, recovery, relay_factory=lambda: relay)
    assert len(relay.calls) == 1
    assert result['classification'] == classification
    assert result['resume_eligible'] is (classification != 'blocked')
    assert json.loads(target.read_text())['attempt'] == 1
    assert excluded.read_bytes() == before
    assert 'must-not-appear-credential' not in (target.read_text() + (recovery / 'retry_result.json').read_text())


@pytest.mark.parametrize('change', [
    {'status': 'valid'}, {'error_type': 'ReadTimeout'}, {'attempt': 1},
    {'answer': 'No verdict'}, {'score': 'A>B'}, {'finish_reason': 'stop'},
    {'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}},
])
def test_only_exact_inspected_ambiguous_attempt_zero_can_be_retried(prepared, change):
    suite, recovery, target, excluded = prepared
    row = json.loads(target.read_text()); row.update(change); target.write_text(json.dumps(row))
    # Bind altered bytes to ensure structural checks are exercised, not only hash checks.
    approval = json.loads((recovery / 'approval.json').read_text())
    approval['target']['game_sha256'] = sha(target)
    (recovery / 'approval.json').write_text(json.dumps(approval))
    before = target.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        api().retry_once(suite, recovery, relay_factory=lambda: pytest.fail('No paid call'))
    assert target.read_bytes() == before
    assert not (target.parents[2] / 'attempts').exists()


def test_bad_hash_or_extra_blocker_prevents_archiving_and_network(prepared):
    suite, recovery, target, excluded = prepared
    target.write_text(target.read_text() + '\n')
    with pytest.raises((ValueError, RuntimeError)):
        api().retry_once(suite, recovery, relay_factory=lambda: pytest.fail('No paid call'))
    assert not (target.parents[2] / 'attempts').exists()


def test_budget_and_three_failed_billing_reads_stop_before_archive(prepared):
    suite, recovery, target, excluded = prepared
    before = target.read_bytes()
    relay = BillingRelay(usage_values=[RuntimeError('secret')] * 3)
    waits = []
    with pytest.raises(RuntimeError):
        api().retry_once(suite, recovery, relay_factory=lambda: relay, sleep=waits.append)
    assert waits == [1, 2] and len(relay.calls) == 0
    assert target.read_bytes() == before and not (target.parents[2] / 'attempts').exists()
    assert 'secret' not in (recovery / 'retry_result.json').read_text()


def test_cumulative_budget_limit_does_not_reset_for_explicit_retry(prepared):
    suite, recovery, target, excluded = prepared
    run = frozen_run(suite)
    p = run.directory / 'state/billing.json'
    p.write_text(json.dumps({'start_date': '2026-09-01', 'usage0_cny': 100,
                            'latest_usage_cny': 101, 'spent_cny': 1}))
    relay = FakeRelay(usage_values=[244])
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match='budget'):
        api().retry_once(suite, recovery, relay_factory=lambda: relay)
    assert relay.calls == [] and target.read_bytes() == before
    assert json.loads(p.read_text())['usage0_cny'] == 100


@pytest.fixture
def stopped_execution(tmp_path):
    e = tmp_path / 'continuation'; e.mkdir()
    for name in ['judge_launch.json', 'watcher_launch.json']:
        (e / name).write_text(json.dumps({'pid': 2147483647}))
    (e / 'judging_exit_code').write_text('1\n')
    (e / 'judging.log').write_text('old stopped process log\n')
    (e / 'controller_state.json').write_text('{"state":"failed"}\n')
    (e / 'source.py').write_text('must stay frozen\n')
    names = ['judge_launch.json','watcher_launch.json','judging_exit_code','judging.log','controller_state.json']
    return e, e / 'history' / 'inspected-recovery', {name: sha(e / name) for name in names}


def test_stopped_execution_archive_is_exact_durable_and_idempotent(stopped_execution):
    e, history, hashes = stopped_execution
    original = {name: (e / name).read_bytes() for name in hashes}
    result = api().archive_stopped_execution(e, history, hashes)
    assert result['status'] == 'archived'
    assert all(not (e / name).exists() and (history / name).read_bytes() == data
               for name, data in original.items())
    assert (e / 'source.py').read_text() == 'must stay frozen\n'
    assert api().archive_stopped_execution(e, history, hashes)['status'] == 'archived'


def test_changed_or_live_launch_metadata_never_archived(stopped_execution):
    e, history, hashes = stopped_execution
    import os
    (e / 'judge_launch.json').write_text(json.dumps({'pid': os.getpid()}))
    hashes['judge_launch.json'] = sha(e / 'judge_launch.json')
    with pytest.raises((ValueError, RuntimeError)):
        api().archive_stopped_execution(e, history, hashes)
    assert all((e / name).exists() for name in hashes)
    assert not history.exists()


def test_frozen_source_cannot_be_in_archive_plan(stopped_execution):
    e, history, hashes = stopped_execution
    hashes['source.py'] = sha(e / 'source.py')
    with pytest.raises((ValueError, RuntimeError)):
        api().archive_stopped_execution(e, history, hashes)
    assert all((e / name).exists() for name in hashes)
    assert not history.exists()


def test_post_call_budget_failure_preserves_attempt_and_blocks_resume(prepared):
    suite, recovery, target, excluded = prepared
    relay = FakeRelay(usage_values=[100, 244])
    with pytest.raises(RuntimeError, match='budget'):
        api().retry_once(suite, recovery, relay_factory=lambda: relay)
    row = json.loads(target.read_text())
    report = json.loads((recovery / 'retry_result.json').read_text())
    assert len(relay.calls) == 1 and row['attempt'] == 1 and row['status'] == 'valid'
    assert report['status'] == 'failed' and report['resume_eligible'] is False
    assert report['paid_calls'] == 1


def test_unrelated_transport_ambiguity_blocks_the_inspected_retry(prepared):
    suite, recovery, target, excluded = prepared
    seed(suite, tag='lam4', status='ambiguous', error_type='RemoteProtocolError')
    with pytest.raises(ValueError, match='blocked|coverage'):
        api().retry_once(suite, recovery, relay_factory=lambda: pytest.fail('No paid call'))
    assert not (target.parents[2] / 'attempts').exists()


def test_changed_log_blocks_archive_before_any_removal(stopped_execution):
    e, history, hashes = stopped_execution
    (e / 'judging.log').write_text('unexpected appended data')
    with pytest.raises(ValueError, match='changed'):
        api().archive_stopped_execution(e, history, hashes)
    assert all((e / name).exists() for name in hashes) and not history.exists()
