#!/usr/bin/env python3
"""Resume this frozen custom-reference suite with bounded connection recovery."""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import uuid

import httpx

SUITE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def complete_phase(suite, uids, *, workers, budget, helper=None, relay_factory=None, sleep=time.sleep):
    suite = Path(suite).resolve()
    helper = helper or load('_resilient_retry_controller', suite / 'retry_resilient_judgments.py')
    if relay_factory is None:
        transport = load('_resilient_transport_controller', suite / 'resilient_judge.py')
        key = os.environ.get('LINKAPI_KEY') or Path('/root/.linkapi_key').read_text().strip()
        relay_factory = lambda: transport.make_relay(key, suite=suite)
    billing_failures = 0
    for _ in range(6000 * helper.MAX_ATTEMPTS + 1):
        # This scan rejects inflight or uncertain outcomes before any new call.
        targets = helper.invalid_targets(suite, uids)
        run = helper.load_run(suite)
        before = {item: helper.file_hash(run.game_path(*item)) for item in targets}
        try:
            if targets:
                helper.retry_games(suite, targets, budget_cny=budget, relay_factory=relay_factory)
            else:
                result = helper.load_run(suite).run(relay_factory, uids, workers=workers, budget_cny=budget)
                if not result['complete']:
                    raise RuntimeError('Unexpected incomplete phase without a request limit')
                return result
            billing_failures = 0
        except httpx.TransportError:
            # Paid-call errors are caught and persisted by both drivers. A raw
            # transport exception here comes from their read-only usage lookup.
            # Recheck saved records so an uncertain paid call cannot be retried.
            helper.invalid_targets(suite, uids)
            billing_failures += 1
            if billing_failures >= 4:
                raise
            sleep(2 ** billing_failures)
        except RuntimeError as error:
            if str(error) == 'Cannot verify relay usage; dispatch stopped':
                helper.invalid_targets(suite, uids)
                billing_failures += 1
                if billing_failures >= 4:
                    raise
                sleep(2 ** billing_failures)
                continue
            # Structural/connection outcomes have saved eligible records. Budget,
            # input, or other failures without such records must not loop.
            if not helper.invalid_targets(suite, uids):
                raise
            if before and all(helper.file_hash(run.game_path(*item)) == digest for item, digest in before.items()):
                raise
            # A budget failure can coexist with eligible outcomes. Check persisted
            # cumulative usage before retrying so the guard remains definitive.
            billing_path = suite / 'model_judgment/gpt-4.1/state/billing.json'
            if billing_path.exists() and helper.read_json(billing_path).get('spent_cny', 0) >= budget:
                raise
    raise RuntimeError('Bounded recovery exhausted')


def execution_identity(ctrl, suite, policy):
    result = {key: policy[key] for key in ('source_sha256', 'original_experiment_sha256',
              'legacy_manual_approval_sha256', 'billing_origin', 'original_host_execution_sha256')}
    result['resilient_policy_sha256'] = ctrl.file_hash(suite / 'resilient_policy.json')
    return result


def scores_complete(suite):
    directory = suite / 'scores'
    if not directory.exists():
        return False
    try:
        valid = (isinstance(json.loads((directory / 'results.json').read_text()), dict)
                 and len((directory / 'results.csv').read_text().splitlines()) >= 2)
    except (OSError, ValueError):
        valid = False
    if valid:
        return True
    # Preserve interrupted CPU output, then recompute from saved judgments.
    directory.rename(suite / 'job' / ('incomplete-scores-' + str(uuid.uuid4())))
    return False


def cost_decision(ctrl, suite, identity, *, usage_reader=None):
    path = suite / 'resilient_cost_decision.json'
    pilot = ctrl.pilot_status(suite, strict_scope=not path.exists())
    if not pilot['complete']:
        raise RuntimeError('Pilot must contain exactly60 valid games before cost decision')
    execution = ctrl.read_json(suite / 'resilient_execution.json')
    if path.exists():
        decision = ctrl.read_json(path)
        if execution.get('cost_decision_sha256') not in (None, ctrl.file_hash(path)):
            raise RuntimeError('Cost decision hash changed')
        if (decision.get('pilot_records_sha256') != pilot['records_sha256']
                or decision.get('resilient_policy_sha256') != identity['resilient_policy_sha256']):
            raise RuntimeError('Cost decision pilot/policy mismatch')
        expected = ctrl.calculate_cost_decision(decision['pilot_cost_cny'])
        if any(decision.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Cost decision formula changed')
    else:
        cost = ctrl.wait_for_positive_cost(suite, usage_reader=usage_reader)
        decision = dict(ctrl.calculate_cost_decision(cost), decided_at=ctrl.now(),
                        pilot_records_sha256=pilot['records_sha256'],
                        resilient_policy_sha256=identity['resilient_policy_sha256'])
        ctrl.save(suite, path, decision)
    if not decision['automatic_continuation']:
        raise RuntimeError('Pilot cost exceeds existing automatic continuation limit')
    execution['cost_decision_sha256'] = ctrl.file_hash(path)
    ctrl.save(suite, suite / 'resilient_execution.json', execution)
    ctrl.save(suite, suite / 'resilient_pilot_summary.json', dict(pilot, verified_at=ctrl.now()))
    return decision


def continue_suite(suite=SUITE, *, runner=None, phase_runner=None, usage_reader=None):
    suite = Path(suite).resolve()
    ctrl = load('_original_controller_functions', suite / 'continue_evaluation.py')
    helper = load('_resilient_retry_controller', suite / 'retry_resilient_judgments.py')
    judge = ctrl.judge_module(suite)
    runner = runner or ctrl.run_command
    phase_runner = phase_runner or complete_phase
    with ExitStack() as stack:
        for name in ('resilient-lock', 'continuation-lock', 'manual-recovery-lock'):
            stack.enter_context(judge.exclusive_lock(suite / 'job' / name))
        state = {'state': 'running', 'pid': os.getpid(), 'started_at': ctrl.now()}
        def phase(name):
            state.update(phase=name, updated_at=ctrl.now())
            ctrl.save(suite, suite / 'resilient_status.json', state)
            print(json.dumps(state), flush=True)
        try:
            phase('validate_generation')
            ctrl.checked(runner, suite, 'resilient_validate_generation', ctrl.commands(suite)['validate_generation'])
            policy = helper.binding(suite)
            identity = execution_identity(ctrl, suite, policy)
            execution_path = suite / 'resilient_execution.json'
            if execution_path.exists():
                if ctrl.read_json(execution_path).get('identity') != identity:
                    raise RuntimeError('Resilient execution identity changed')
            else:
                ctrl.save(suite, execution_path, {'identity': identity, 'created_at': ctrl.now(),
                          'pilot_workers': 12, 'full_workers': 32})
            if not (suite / 'resilient_cost_decision.json').exists():
                if ctrl.has_nonpilot_requests(suite):
                    raise RuntimeError('Nonpilot requests exist without a cost decision')
                phase('pilot')
                phase_runner(suite, ctrl.pilot_uids(suite), workers=12, budget=20)
            phase('pilot_cost')
            decision = cost_decision(ctrl, suite, identity, usage_reader=usage_reader)
            phase('full_judging')
            phase_runner(suite, None, workers=32, budget=decision['dispatch_budget_cny'])
            phase('score')
            if not scores_complete(suite):
                command = [ctrl.CPU_PYTHON, str(suite / 'source/scripts/score_arena_hard.py'),
                    '--baseline-model', 'gpt-4o-mini-2024-07-18', '--questions', str(suite / 'question.jsonl'),
                    '--answers-dir', str(suite / 'model_answer'), '--judgments-dir', str(suite / 'model_judgment/gpt-4.1'),
                    '--output', str(suite / 'scores'), '--seed', '42', '--threads', '1']
                ctrl.checked(runner, suite, 'resilient_score', command)
            phase('verify_final')
            output = ctrl.checked(runner, suite, 'resilient_verify_final',
                [ctrl.CPU_PYTHON, str(suite / 'verify_resilient_scoring.py'), '--suite', str(suite)])
            verification = json.loads(output['stdout'])
            expected = {'status': 'passed', 'current_games': 6000, 'exact_request_and_parse_checks': 6000,
                        'judgments': 3000, 'questions': 500, 'numeric_comparisons': 42}
            if any(verification.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Final verifier did not confirm complete coverage')
            difference = verification.get('max_absolute_difference')
            if type(difference) not in (int, float) or not math.isfinite(difference) or not 0 <= difference <= 1e-10:
                raise RuntimeError('Final numeric verification mismatch')
            state.update(state='evaluation_complete', phase='complete', completed_at=ctrl.now(),
                         verification=verification, resilient_policy_sha256=identity['resilient_policy_sha256'],
                         resilient_execution_sha256=ctrl.file_hash(execution_path))
            ctrl.save(suite, suite / 'evaluation_complete.json', state)
            ctrl.save(suite, suite / 'resilient_status.json', state)
            return state
        except BaseException as error:
            # Error text may contain network details or credentials.
            state.update(state='failed', error_type=type(error).__name__, failed_at=ctrl.now())
            ctrl.save(suite, suite / 'resilient_status.json', state)
            print(json.dumps(state), flush=True)
            raise


if __name__ == '__main__':
    try:
        print(json.dumps(continue_suite(), indent=2))
    except Exception as error:
        print('Resilient continuation stopped: ' + type(error).__name__, flush=True)
        raise SystemExit(1) from None
