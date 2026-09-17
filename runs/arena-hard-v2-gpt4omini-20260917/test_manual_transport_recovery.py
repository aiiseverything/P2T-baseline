"""One reviewed ConnectError recovery; synthetic transport only."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SUITE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def incident(tmp_path):
    required = ('experiment.json', 'question.jsonl', 'pilot_uids.json', 'generation_summary.json',
                'generation_complete.json', 'status.json')
    missing = [name for name in required if not (SUITE / name).is_file()]
    if missing:
        pytest.skip('Requires local archived Arena experiment artifacts: ' + ', '.join(missing[:3]))
    experiment = json.loads((SUITE / 'experiment.json').read_text())
    artifact_names = [name for name in experiment['files_sha256']
                      if not name.startswith('source/') and Path(name).suffix not in ('.py', '.sh')]
    missing = [name for name in artifact_names if not (SUITE / name).is_file()]
    original = Path(experiment['generation_reuse']['source_suite'])
    missing.extend(str(original / name) for name in experiment['generation_reuse']['original_files_sha256']
                   if not (original / name).is_file())
    if missing:
        pytest.skip('Requires local archived answers and generation provenance: ' + ', '.join(missing[:3]))
    helper = load('manual_transport_helper', SUITE / 'manual_transport_recovery.py')
    controller = load('manual_transport_controller', SUITE / 'continue_evaluation.py')
    # Symlinks are read-only fixtures; mutable state exists only below tmp_path.
    for name in ('source', 'model_answer', 'manifests', 'reference', 'original_generation', '.tiktoken-cache'):
        (tmp_path / name).symlink_to(SUITE / name, target_is_directory=True)
    for name in ('experiment.json', 'question.jsonl', 'pilot_uids.json', 'generation_summary.json',
                 'generation_complete.json', 'status.json', 'continue_evaluation.py', 'run_evaluation.py',
                 'run_full_judging.sh', 'verify_final_scoring.py', 'retry_policy.json',
                 'retry_invalid_judgments.py', 'manual_transport_recovery.py',
                 'resume_manual_recovery.py', 'verify_manual_recovery.py'):
        (tmp_path / name).write_bytes((SUITE / name).read_bytes())
    (tmp_path / 'job').mkdir()
    controller.bind_execution(tmp_path, controller.commands(tmp_path))
    run = controller.load_judge_run(tmp_path)
    bad_request = run.request(*helper.GAME)
    class ConnectError(Exception):
        pass
    class Relay:
        calls = []
        fail = True
        answer = '[[A=B]]'
        def usage(self, _):
            return 100 + .01 * len(self.calls)
        def judge_call(self, request):
            self.calls.append(request)
            if self.fail and request == bad_request:
                raise ConnectError('synthetic error must not be logged')
            return {'answer': self.answer, 'finish_reason': 'stop',
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
    relay = Relay()
    with pytest.raises(RuntimeError, match='blocked'):
        run.run(lambda: relay, [helper.GAME[1]], workers=12, budget_cny=20)
    assert len(relay.calls) == 12
    valid = {str(p.relative_to(tmp_path)): p.read_bytes()
             for p in (run.directory / 'state/games').glob('*/*.json')
             if json.loads(p.read_text())['status'] == 'valid'}
    assert len(valid) == 11
    return tmp_path, helper, controller, relay, valid


def test_prepare_and_one_identical_retry_preserve_all_valid_games(incident):
    suite, helper, controller, relay, valid = incident
    approval = helper.prepare(suite, reviewed_by='offline test reviewer')
    assert approval['cost_decision_existed'] is False
    assert len(approval['valid_records_sha256']) == 11
    assert approval['baseline_model'] == 'gpt-4o-mini-2024-07-18'
    assert approval['dispatch_guard_cny'] == 20
    relay.fail = False
    result = helper.retry_once(suite, relay_factory=lambda: relay)
    assert result['status'] == 'recovered' and len(relay.calls) == 13
    run = controller.load_judge_run(suite)
    record = run.load_record(*helper.GAME)
    assert record['attempt'] == 1 and record['status'] == 'valid'
    assert record['request'] == relay.calls[-1]
    assert record['baseline_model'] == approval['baseline_model']
    assert record['protocol_sha256'] == approval['protocol_sha256']
    for name, data in valid.items():
        assert (suite / name).read_bytes() == data
    helper.retry_once(suite, relay_factory=lambda: pytest.fail('no second manual retry'))
    assert len(relay.calls) == 13


def test_invalid_single_replacement_stays_blocked_without_more_calls(incident):
    suite, helper, _, relay, _ = incident
    helper.prepare(suite, reviewed_by='offline test reviewer')
    relay.fail = False; relay.answer = 'invalid judgment'
    with pytest.raises(RuntimeError):
        helper.retry_once(suite, relay_factory=lambda: relay)
    assert len(relay.calls) == 13
    with pytest.raises(RuntimeError, match='already|single|attempt'):
        helper.retry_once(suite, relay_factory=lambda: pytest.fail('only one replacement authorized'))


@pytest.mark.parametrize('damage', ['valid_record', 'billing_baseline', 'approval_scope'])
def test_mutated_review_evidence_blocks_before_transport(incident, damage):
    suite, helper, controller, relay, valid = incident
    helper.prepare(suite, reviewed_by='offline test reviewer')
    if damage == 'valid_record':
        path = suite / next(iter(valid)); path.write_bytes(path.read_bytes() + b'\n')
    elif damage == 'billing_baseline':
        path = suite / 'model_judgment/gpt-4.1/state/billing.json'
        data = json.loads(path.read_text()); data['usage0_cny'] = 0; path.write_text(json.dumps(data))
    else:
        path = suite / 'manual_transport_recovery.json'
        data = json.loads(path.read_text()); data['max_additional_attempts'] = 2; path.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)):
        helper.retry_once(suite, relay_factory=lambda: pytest.fail('no paid call after evidence mutation'))


def test_supplemental_verifier_accepts_only_exact_two_node_custom_chain(incident):
    suite, helper, controller, relay, _ = incident
    approval = helper.prepare(suite, reviewed_by='offline test reviewer')
    relay.fail = False; helper.retry_once(suite, relay_factory=lambda: relay)
    api = load('manual_supplemental_verifier', suite / 'verify_manual_recovery.py')
    base = api.load_original(suite, approval)
    run = controller.load_judge_run(suite)
    record = run.load_record(*helper.GAME)
    request = run.request(*helper.GAME)
    original = base.verify_retry_chain
    with pytest.raises(ValueError, match='invalid'):
        original(record, run.directory / 'state', *helper.GAME, request,
                 lambda text, patterns: controller.judge_module(suite).parse_score(text, patterns),
                 protocol=run.protocol)
    parse = lambda text, patterns: controller.judge_module(suite).parse_score(text, patterns)
    result = api.verify_manual_chain(base, record, run.directory / 'state', *helper.GAME,
                                     request, parse, approval, run.protocol)
    assert len(result['archives']) == 1 and len(result['request_ids']) == 2
    record['baseline_model'] = 'o3-mini-2025-01-31'
    with pytest.raises(ValueError):
        api.verify_manual_chain(base, record, run.directory / 'state', *helper.GAME,
                                request, parse, approval, run.protocol)


def test_wrapper_finishes_remaining_pilot_and_only_overrides_final_verifier(incident):
    suite, helper, controller, relay, valid = incident
    approval = helper.prepare(suite, reviewed_by='offline test reviewer')
    relay.fail = False; helper.retry_once(suite, relay_factory=lambda: relay)
    api = load('reviewed_resume_wrapper', suite / 'resume_manual_recovery.py')
    verifier = load('reviewed_resume_verifier', suite / 'verify_manual_recovery.py')
    labels = []
    def runner(s, label, command):
        labels.append(label)
        if label == 'pilot':
            assert command == controller.commands(s)['pilot']
            controller.load_judge_run(s).run(lambda: relay, controller.pilot_uids(s), workers=12, budget_cny=20)
        elif label == 'full_judging':
            assert command[:2] == ['bash', str(s / 'run_full_judging.sh')]
            (s / 'scores').mkdir()
        elif label == 'verify_final':
            assert command == [controller.CPU_PYTHON, str(s / 'verify_manual_recovery.py'),
                               '--approval', str(s / 'manual_transport_recovery.json')]
            base = verifier.load_original(s, approval)
            evidence = verifier.evidence_binding(base, s, s / 'manual_transport_recovery.json', approval)
            assert str(s / 'cost_decision.json') in evidence
        return {'returncode': 0, 'stdout': json.dumps({'status': 'passed', 'current_games': 6000,
            'exact_request_and_parse_checks': 6000, 'judgments': 3000, 'questions': 500,
            'numeric_comparisons': 42, 'max_absolute_difference': 0.0})}
    result = api.resume(suite, runner=runner)
    assert result['state'] == 'evaluation_complete'
    assert labels == ['validate_generation', 'pilot', 'full_judging', 'verify_final']
    assert len(relay.calls) == 61  # 60 eventual valid games plus the one ambiguous original.
    decision = json.loads((suite / 'cost_decision.json').read_text())
    assert decision['pilot_cost_cny'] == pytest.approx(.61)
    for name, data in valid.items():
        assert (suite / name).read_bytes() == data
    labels.clear()
    api.resume(suite, runner=runner)
    assert labels == ['validate_generation', 'verify_final'] and len(relay.calls) == 61


def test_wrapper_refuses_invalid_replacement_before_any_controller_command(incident):
    suite, helper, _, relay, _ = incident
    helper.prepare(suite, reviewed_by='offline test reviewer')
    relay.fail = False; relay.answer = 'invalid'
    with pytest.raises(RuntimeError):
        helper.retry_once(suite, relay_factory=lambda: relay)
    api = load('blocked_resume_wrapper', suite / 'resume_manual_recovery.py')
    with pytest.raises(ValueError, match='single valid'):
        api.resume(suite, runner=lambda *_: pytest.fail('must not enter automatic retry controller'))


def test_supplemental_dispatcher_preserves_all_ordinary_strict_calls(incident):
    suite, helper, controller, relay, _ = incident
    approval = helper.prepare(suite, reviewed_by='offline test reviewer')
    relay.fail = False; helper.retry_once(suite, relay_factory=lambda: relay)
    api = load('dispatch_verifier', suite / 'verify_manual_recovery.py')
    base = api.load_original(suite, approval)
    run = controller.load_judge_run(suite)
    calls = []
    def ordinary(*args):
        calls.append(args)
        return {'archives': {}, 'request_ids': {'ordinary'}}
    dispatch, counts = api.chain_dispatcher(base, ordinary, approval)
    parse = lambda text, patterns: controller.judge_module(suite).parse_score(text, patterns)
    dispatch(run.load_record(*helper.GAME), run.directory / 'state', *helper.GAME,
             run.request(*helper.GAME), parse, protocol=run.protocol)
    other = ('base', helper.GAME[1], 0)
    dispatch(run.load_record(*other), run.directory / 'state', *other,
             run.request(*other), parse, {'sentinel': 'binding'}, run.protocol)
    assert counts == {'manual': 1, 'ordinary': 1}
    assert calls[0][-2:] == ({'sentinel': 'binding'}, run.protocol)
    with pytest.raises(ValueError, match='more than once'):
        dispatch(run.load_record(*helper.GAME), run.directory / 'state', *helper.GAME,
                 run.request(*helper.GAME), parse, protocol=run.protocol)


def test_complete_six_thousand_game_scorer_and_supplemental_verifier(incident):
    """Real scoring and strict verification; only network generation is fake."""
    import shutil
    suite, helper, controller, relay, valid = incident
    # Production answers are ordinary files. Avoid resolving fixture symlinks
    # to the production path in the scorer's input-files provenance keys.
    (suite / 'model_answer').unlink()
    shutil.copytree(SUITE / 'model_answer', suite / 'model_answer')
    helper.prepare(suite, reviewed_by='offline end-to-end test')
    relay.fail = False; helper.retry_once(suite, relay_factory=lambda: relay)
    wrapper = load('full_reviewed_wrapper', suite / 'resume_manual_recovery.py')
    supplemental = load('full_supplemental_verifier', suite / 'verify_manual_recovery.py')
    scorer = load('full_manual_frozen_scorer', suite / 'source/scripts/score_arena_hard.py')
    judge = controller.judge_module(suite)
    full_verification = {}
    def runner(s, label, command):
        if label == 'validate_generation':
            load('full_reuse_validator', s / 'run_evaluation.py').verify_reused_generation(s)
        elif label == 'pilot':
            controller.load_judge_run(s).run(lambda: relay, controller.pilot_uids(s), workers=12, budget_cny=20)
        elif label == 'full_judging':
            run = controller.load_judge_run(s)
            for tag in controller.TAGS:
                for uid in run.questions:
                    for order in (0, 1):
                        path = run.game_path(tag, uid, order)
                        if path.exists():
                            continue
                        request = run.request(tag, uid, order)
                        row = {'tag': tag, 'uid': uid, 'order': order, 'status': 'valid', 'attempt': 0,
                            'local_request_id': f'offline-full-{tag}-{uid}-{order}',
                            'request': request, 'request_sha256': judge.digest(request),
                            'baseline_model': helper.BASELINE, 'protocol_sha256': judge.digest(run.protocol),
                            'score': 'A=B', 'answer': 'Offline complete-suite verdict [[A=B]]', 'finish_reason': 'stop',
                            'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}}
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(row))
            run.export()
            scorer.main(['--questions', str(s / 'question.jsonl'), '--answers-dir', str(s / 'model_answer'),
                '--judgments-dir', str(s / 'model_judgment/gpt-4.1'), '--output', str(s / 'scores'),
                '--baseline-model', helper.BASELINE])
        elif label == 'verify_final':
            assert command[1] == str(s / 'verify_manual_recovery.py')
            full_verification.update(supplemental.verify(s))
            return {'returncode': 0, 'stdout': json.dumps(full_verification)}
        return {'returncode': 0, 'stdout': '{}'}
    result = wrapper.resume(suite, runner=runner)
    assert result['state'] == 'evaluation_complete'
    assert full_verification['status'] == 'passed'
    assert full_verification['current_games'] == full_verification['exact_request_and_parse_checks'] == 6000
    assert full_verification['numeric_comparisons'] == 42
    assert full_verification['max_absolute_difference'] <= 1e-10
    assert full_verification['manual_transport_retries'] == full_verification['initial_ambiguous_games'] == 1
    assert full_verification['initial_invalid_games'] == 0
    assert full_verification['total_recorded_api_attempts'] == 6001
    assert full_verification['manual_recovery']['ordinary_chain_checks'] == 5999
    assert len(relay.calls) == 61
    for name, data in valid.items():
        assert (suite / name).read_bytes() == data
