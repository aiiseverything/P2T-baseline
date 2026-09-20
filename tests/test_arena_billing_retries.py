"""Only external billing/judging is faked; frozen dispatch/state remain real."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from test_arena_exclusion_runner import (FakeRelay, SOURCE_SUITE, frozen_run,
                                         response, seed, suite)

ROOT = Path(__file__).resolve().parents[1]


def wrapper():
    path = ROOT / 'scripts/run_arena_with_billing_retries.py'
    assert path.is_file(), 'Additive billing retry entrypoint has not been implemented'
    spec = importlib.util.spec_from_file_location('billing_retries', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(suite):
    destination = suite / 'continuation-with-exclusions/source/scripts'
    destination.mkdir(parents=True)
    for name in ('run_arena_with_exclusions.py', 'arena_exclusion_policy.py'):
        shutil.copyfile(SOURCE_SUITE / 'continuation-with-exclusions/source/scripts' / name,
                        destination / name)
    return suite


class BillingRelay(FakeRelay):
    def usage(self, start):
        self.usage_calls.append(start)
        value = next(self.usage_values)
        if isinstance(value, Exception):
            raise value
        return value


def resume(suite, relay, sleeps, **kwargs):
    return wrapper().resume_suite(suite, policy=suite / 'exclusions_policy.json',
        budget_cny=kwargs.pop('budget_cny', 10), max_requests=kwargs.pop('max_requests', 1),
        relay_factory=lambda: relay, sleep=sleeps.append, **kwargs)


def test_transient_free_billing_reads_recover_before_one_paid_call(prepared, capsys):
    secret = 'credential-that-must-not-be-logged'
    relay = BillingRelay(usage_values=[RuntimeError(secret), RuntimeError(secret), 100, 100])
    sleeps = []
    result = resume(prepared, relay, sleeps)
    assert result['dispatched_this_run'] == 1 and result['counts']['total']['valid'] == 1
    assert len(relay.calls) == 1 and len(relay.usage_calls) == 4
    assert sleeps == [1, 2] and relay.closed
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert 'attempt 1/3' in captured.err and 'attempt 2/3' in captured.err
    billing = json.loads((prepared / 'model_judgment/gpt-4o/state/billing.json').read_text())
    assert billing['usage0_cny'] == 100 and billing['spent_cny'] == 0


def test_three_failed_free_reads_stop_before_any_paid_dispatch(prepared, capsys):
    secret = 'sensitive-error-message'
    relay = BillingRelay(usage_values=[RuntimeError(secret)] * 3)
    sleeps = []
    with pytest.raises(RuntimeError, match='billing.*3 attempts') as error:
        resume(prepared, relay, sleeps)
    assert len(relay.usage_calls) == 3 and relay.calls == [] and sleeps == [1, 2]
    assert relay.closed
    assert not (prepared / 'model_judgment/gpt-4o/state/games').exists()
    captured = capsys.readouterr()
    assert secret not in str(error.value) + captured.out + captured.err
    progress = json.loads((prepared / 'exclusions_progress.json').read_text())
    assert progress['status'] == 'blocked' and progress['dispatched_this_run'] == 0


def test_paid_judge_exception_is_never_retried(prepared):
    relay = BillingRelay(outcomes=[RuntimeError('paid response ambiguous')], usage_values=[100])
    sleeps = []
    with pytest.raises(RuntimeError, match='blocked'):
        resume(prepared, relay, sleeps, max_requests=3, workers=1)
    assert len(relay.calls) == 1 and len(relay.usage_calls) == 1 and sleeps == []
    record = json.loads((prepared / 'model_judgment/gpt-4o/state/games/base/u0-0.json').read_text())
    assert record['status'] == 'ambiguous' and record['attempt'] == 0 and relay.closed


@pytest.mark.parametrize('usage,budget,message', [(100, 10, 'decreased'), (101, .5, 'budget')])
def test_monotonicity_and_budget_errors_stay_outside_free_read_retries(prepared, usage, budget, message):
    run = frozen_run(prepared)
    run.prepare()
    path = run.directory / 'state/billing.json'
    path.write_text(json.dumps({'start_date': '2026-09-01', 'usage0_cny': 100,
                                'latest_usage_cny': 101, 'spent_cny': 1}))
    relay = BillingRelay(usage_values=[usage])
    sleeps = []
    with pytest.raises(RuntimeError, match=message):
        resume(prepared, relay, sleeps, budget_cny=budget)
    assert len(relay.usage_calls) == 1 and relay.calls == [] and sleeps == [] and relay.closed


def test_existing_valid_and_excluded_records_are_reused_byte_for_byte(prepared):
    valid = seed(prepared, order=0)
    excluded = seed(prepared, order=1, status='invalid', attempt=4, answer='No verdict', score=None)
    before = {path: path.read_bytes() for path in (valid, excluded)}
    relay = BillingRelay(usage_values=[RuntimeError('transient'), 100, 100])
    result = resume(prepared, relay, [])
    assert all(path.read_bytes() == data for path, data in before.items())
    assert result['counts']['total'] == {'valid': 2, 'judge_failed': 1, 'missing': 5997, 'blocked': 0}
    assert len(relay.calls) == 1
    assert relay.calls[0] == frozen_run(prepared).request('sft-init', 'u0', 0)


def test_non_runtime_billing_errors_are_not_retried(prepared):
    relay = BillingRelay(usage_values=[ValueError('unexpected programmer error')])
    sleeps = []
    with pytest.raises(ValueError):
        resume(prepared, relay, sleeps)
    assert len(relay.usage_calls) == 1 and relay.calls == [] and sleeps == [] and relay.closed


def test_explicit_v2_source_resumes_filtered_record_and_keeps_v1_frozen(prepared):
    original = prepared / 'continuation-with-exclusions'
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    destination = prepared / 'continuation-with-exclusions-v2/source/scripts'
    destination.mkdir(parents=True)
    for name in ('run_arena_with_exclusions.py', 'arena_exclusion_policy.py'):
        shutil.copyfile(ROOT / 'scripts' / name, destination / name)
    seed(prepared, status='invalid', answer=None, score=None, finish_reason='content_filter')
    relay = BillingRelay()
    result = resume(prepared, relay, [], continuation_dir=destination.parent.parent)
    assert result['counts']['total'] == {'valid': 1, 'judge_failed': 1, 'missing': 5998, 'blocked': 0}
    assert all(p.read_bytes() == data for p, data in before.items())
    assert len(relay.calls) == 1


@pytest.mark.parametrize('directory', ['..', '/', 'missing-child'])
def test_continuation_must_be_an_existing_direct_child_before_dispatch(prepared, directory):
    relay = BillingRelay()
    with pytest.raises(ValueError, match='continuation'):
        resume(prepared, relay, [], continuation_dir=directory)
    assert relay.calls == []
    assert not (prepared / 'model_judgment').exists()
