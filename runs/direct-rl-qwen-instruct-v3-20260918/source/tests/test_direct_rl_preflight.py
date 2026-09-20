"""CPU contract checks for fresh-LoRA GPU preflight; never allocate a GPU."""
import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def actor():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, attention_dropout=0., pad_token_id=0))
    return get_peft_model(base, LoraConfig(r=64, lora_alpha=128, lora_dropout=0,
        bias='none', task_type='CAUSAL_LM', target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))


def test_profile_arguments_never_load_sft_and_share_explicit_scale(tmp_path):
    from scripts import direct_rl_preflight as gate
    from scripts.profile_vllm_full import parse_args
    common = {'model': 'base', 'rm': 'rm', 'init_adapter': '', 'seed': 42,
              'length_reward_mode': 'soft', 'max_response_tokens': 2048,
              'credit_microbatch_responses': 1, 'learning_rate': 5e-5}
    grpo = parse_args(gate.profile_arguments(common, 'grpo', tmp_path / 'grpo'))
    lam4 = parse_args(gate.profile_arguments(common, 'lam4', tmp_path / 'lam4', 2.5))
    assert grpo.init_adapter == lam4.init_adapter == ''
    assert grpo.max_rollouts == lam4.max_rollouts == 2
    assert grpo.length_reward_sigma0 is None and lam4.length_reward_sigma0 == 2.5
    assert lam4.credit_lambda == 4 and lam4.freeze_stop_tokens and lam4.freeze_structural
    assert grpo.method == 'grpo' and not grpo.freeze_stop_tokens
    with pytest.raises(ValueError, match='initial|SFT|adapter'):
        gate.profile_arguments(dict(common, init_adapter='/sft'), 'grpo', tmp_path)
    with pytest.raises(ValueError, match='arm'):
        gate.profile_arguments(common, 'lam8', tmp_path)


def test_initial_gate_requires_zero_contribution_and_frozen_base(actor):
    from scripts.direct_rl_preflight import verify_initial_adapters
    report = verify_initial_adapters(actor, '')
    assert report['status'] == 'passed'
    assert report['a_tensor_count'] == report['b_tensor_count'] == 7
    assert report['reference_mode'] == 'disabled_adapter_base'
    assert report['default_sha256'] != report['base_sha256']
    with torch.no_grad():
        next(p for n, p in actor.named_parameters() if '.lora_B.' in n).fill_(.1)
    with pytest.raises(ValueError, match='zero'):
        verify_initial_adapters(actor, '')


@pytest.mark.parametrize('mutation', ['zero_a', 'trainable_base', 'extra_adapter', 'nonfinite'])
def test_initial_gate_rejects_invalid_initialization(actor, mutation):
    from peft import LoraConfig
    from scripts.direct_rl_preflight import verify_initial_adapters
    if mutation == 'extra_adapter':
        actor.add_adapter('ref', LoraConfig(r=2, target_modules=['q_proj']))
    elif mutation == 'trainable_base':
        next(p for n, p in actor.named_parameters() if '.default.' not in n).requires_grad_(True)
    else:
        with torch.no_grad():
            next(p for n, p in actor.named_parameters() if '.lora_A.' in n).fill_(
                0 if mutation == 'zero_a' else float('nan'))
    with pytest.raises(ValueError):
        verify_initial_adapters(actor, '')


def test_update_requires_nonzero_b_and_detects_base_mutation(actor):
    from scripts.direct_rl_preflight import verify_initial_adapters, verify_adapter_update
    initial = verify_initial_adapters(actor, '')
    with pytest.raises(ValueError, match='update|changed|zero'):
        verify_adapter_update(actor, initial)
    # Weight decay alone can change A without introducing any policy delta.
    with torch.no_grad():
        next(p for n, p in actor.named_parameters() if '.lora_A.' in n).add_(.1)
    with pytest.raises(ValueError, match='B|zero'):
        verify_adapter_update(actor, initial)
    actor.train()
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=.01)
    ids = torch.tensor([[1, 2, 3, 4]])
    actor(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    updated = verify_adapter_update(actor, initial)
    assert updated['status'] == 'passed'
    assert updated['base_sha256'] == initial['base_sha256']
    with pytest.raises(ValueError, match='changed|update'):
        verify_adapter_update(actor, initial, previous_default_sha256=updated['default_sha256'])
    with torch.no_grad():
        next(p for n, p in actor.named_parameters() if '.default.' not in n).add_(.1)
    with pytest.raises(ValueError, match='base|reference'):
        verify_adapter_update(actor, initial)


def test_initial_reference_compares_actual_logits_with_disabled_adapter(actor):
    from scripts.direct_rl_preflight import verify_initial_reference
    ids = torch.tensor([[1, 2, 3]])
    trainer = SimpleNamespace(actor=actor, reference_adapter='base', reference_actor=None,
        cfg=SimpleNamespace(init_adapter='', kl_reference='init'),
        _encode_prompts=lambda prompts: ({'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, ['p']))
    report = verify_initial_reference(trainer)
    assert report['max_abs_error'] == 0
    assert actor.active_adapters == ['default']
    with torch.no_grad():
        for name, value in actor.named_parameters():
            if '.lora_B.' in name:
                value.fill_(.25)
    with pytest.raises(ValueError, match='reference|base|logit'):
        verify_initial_reference(trainer)


def test_pair_gate_rejects_different_initial_seed_weights_or_prompts():
    from scripts.direct_rl_preflight import validate_arm_pair
    grpo = {'status': 'passed', 'rollouts': 2, 'seed': 42,
            'initial_adapters': {'default_sha256': 'same', 'base_sha256': 'base'},
            'prompt_sha256': [['p1'], ['p2']]}
    assert validate_arm_pair(grpo, copy.deepcopy(grpo))['status'] == 'passed'
    for mutation in ['weights', 'base', 'seed', 'prompts']:
        other = copy.deepcopy(grpo)
        if mutation == 'weights': other['initial_adapters']['default_sha256'] = 'different'
        elif mutation == 'base': other['initial_adapters']['base_sha256'] = 'different'
        elif mutation == 'seed': other['seed'] = 43
        else: other['prompt_sha256'][0] = ['different']
        with pytest.raises(ValueError):
            validate_arm_pair(grpo, other)


def test_base_quality_canaries_are_diagnostic_even_for_zero_score():
    from scripts.direct_rl_preflight import quality_diagnostic
    result = quality_diagnostic(['unrelated completion'] * 8, baseline_score=0)
    assert result['score'] == 0 and result['baseline_score'] == 0
    assert result['gating'] is False
    assert result['used_for_checkpoint_selection'] is False
    assert result['algorithm_evaluation'] is False


def test_probability_gate_keeps_canonical_limits():
    from scripts.direct_rl_preflight import logprob_error_report
    assert logprob_error_report([-1., -2.], [-1.001, -1.999])['status'] == 'passed'
    with pytest.raises(ValueError, match='tolerance'):
        logprob_error_report([-1.], [-1.4])


def test_sparse_logprob_tail_warns_without_rejecting_well_matched_batch():
    from scripts.direct_rl_preflight import logprob_error_report
    expected = torch.full((1000,), -2., dtype=torch.float64)
    actual = expected.clone(); actual[-1] += .35
    report = logprob_error_report(actual, expected)
    assert report['status'] == 'passed'
    assert report['tail_abs_gt_0_3_count'] == 1
    assert report['warnings'] and report['ess_ratio'] >= .99
    assert report['ratio_clip_fraction'] == .001


@pytest.mark.parametrize('delta,count,want', [(.021, 10000, 'mean_abs_error'),
    (.16, 200, 'p99_abs_error'),
    (1., 100, 'ess_ratio'), (.23, 110, 'ratio_clip_fraction')])
def test_probability_failures_report_failed_status(delta, count, want):
    import json
    from scripts.direct_rl_preflight import logprob_error_report
    expected = torch.full((10000,), -2., dtype=torch.float64)
    actual = expected.clone(); actual[-count:] += delta
    with pytest.raises(ValueError) as failure:
        logprob_error_report(actual, expected)
    report = failure.value.report
    assert report['status'] == 'failed' and want in report['violations']
    assert '"status": "passed"' not in str(failure.value)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('tokens,delta', [(57588, -1.382), (10000, 1.01)])
def test_rare_large_error_warns_when_global_probability_distribution_is_healthy(tokens, delta):
    from scripts.direct_rl_preflight import logprob_error_report
    engine = torch.full((tokens,), -3., dtype=torch.float64)
    actual = engine.clone(); actual[-1] += delta
    report = logprob_error_report(actual, engine)
    assert report['status'] == 'passed'
    assert report['tokens'] == tokens and report['max_abs_error'] == pytest.approx(abs(delta))
    assert report['tail_abs_gt_0_3_count'] == 1 and report['warnings']
    assert report['mean_abs_error'] <= .02 and report['p99_abs_error'] <= .15
    assert report['ess_ratio'] >= .99 and report['ratio_clip_fraction'] <= .01


def test_single_large_positive_ratio_still_fails_global_ess_gate():
    from scripts.direct_rl_preflight import logprob_error_report
    engine = torch.full((57588,), -10., dtype=torch.float64)
    actual = engine.clone(); actual[-1] = -1.
    with pytest.raises(ValueError) as failure:
        logprob_error_report(actual, engine)
    report = failure.value.report
    assert report['status'] == 'failed' and 'ess_ratio' in report['violations']
    assert report['mean_abs_error'] <= .02 and report['p99_abs_error'] <= .15
    assert report['ratio_clip_fraction'] <= .01 and report['ess_ratio'] < .99


def test_short_response_positive_tail_fails_loss_weighted_ess_only():
    import math
    from scripts.direct_rl_preflight import logprob_error_report
    lengths = [2] + [914] * 62 + [918]
    engine = torch.full((57588,), -5., dtype=torch.float64)
    actual = engine.clone(); actual[0] += math.log(10.)
    with pytest.raises(ValueError) as failure:
        logprob_error_report(actual, engine, response_lengths=lengths)
    report = failure.value.report
    assert report['violations'] == ['loss_weighted_importance_ess_fraction']
    assert report['ess_ratio'] >= .99
    assert report['mean_abs_error'] <= .02 and report['p99_abs_error'] <= .15
    assert report['ratio_clip_fraction'] <= .01
    rho = (actual.float() - engine.float()).exp().double()
    means = torch.stack([row.mean() for row in rho.split(lengths)])
    second_moments = torch.stack([row.square().mean() for row in rho.split(lengths)])
    expected = (means.mean().square() / second_moments.mean()).item()
    assert report['loss_weighted_importance_ess_fraction'] == pytest.approx(expected)
    assert report['loss_weighted_importance_ess_fraction'] < .99


@pytest.mark.parametrize('delta,engine_logp', [(-1000., -1.), (100., -101.)])
@pytest.mark.parametrize('gating', [True, False])
def test_fp32_importance_weight_underflow_and_overflow_fail_closed(delta, engine_logp, gating):
    import json
    from scripts.direct_rl_preflight import logprob_error_report
    engine = torch.full((57588,), engine_logp, dtype=torch.float64)
    actual = engine.clone(); actual[0] += delta
    with pytest.raises(ValueError, match='positive and finite in float32') as failure:
        logprob_error_report(actual, engine, gating=gating)
    report = failure.value.report
    assert report['status'] == 'failed'
    assert 'fp32_importance_weights' in report['violations']
    if delta < 0:
        assert report['mean_abs_error'] <= .02 and report['p99_abs_error'] <= .15
        assert report['ess_ratio'] >= .99 and report['ratio_clip_fraction'] <= .01
    json.dumps(report, allow_nan=False)


def test_clip_fraction_can_fail_even_when_interpolated_p99_passes():
    from scripts.direct_rl_preflight import logprob_error_report
    engine = torch.full((129,), -2., dtype=torch.float64)
    actual = engine.clone(); actual[-2:] += .183
    with pytest.raises(ValueError) as failure:
        logprob_error_report(actual, engine)
    assert failure.value.report['violations'] == ['ratio_clip_fraction']
    assert failure.value.report['p99_abs_error'] < .15


def test_extreme_finite_log_ratio_reports_failure_without_exponential_overflow():
    import json
    from scripts.direct_rl_preflight import logprob_error_report
    with pytest.raises(ValueError) as failure:
        logprob_error_report([-1., -1.], [-1001., -1.])
    assert failure.value.report['status'] == 'failed'
    assert failure.value.report['ess_ratio'] == .5
    json.dumps(failure.value.report, allow_nan=False)


class PrefixActor(torch.nn.Module):
    """A tiny causal policy whose selected probabilities depend on exact prefixes."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, logits_to_keep=None, **kwargs):
        self.calls.append((input_ids.detach().clone(), torch.is_grad_enabled()))
        centers = input_ids.cumsum(-1).remainder(11).float()
        logits = -(torch.arange(16)[None, None, :] - centers[:, :, None]).square() / 16
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


def probability_fixture(tmp_path, *, groups=8, n=8, width=130):
    from scripts.direct_rl_preflight import WorkerChecks
    actor = PrefixActor()
    prefixes = [[1, 2 + index % 10] for index in range(groups)]
    prompts = [f'prompt-{index}' for index in range(groups)]
    tokenizer = lambda text, **kwargs: {'input_ids': prefixes[prompts.index(text)]}
    checks = WorkerChecks(SimpleNamespace(output_dir=tmp_path, worker_arm='grpo'))
    checks.trainer = SimpleNamespace(actor=actor, actor_device=torch.device('cpu'),
        actor_tokenizer=tokenizer, output_mask=torch.ones(16, dtype=torch.bool), stop_token_ids=(15,))
    rows = [[4 + index % 4] * width for index in range(groups * n)]
    probabilities = []
    for index, row in enumerate(rows):
        prefix = prefixes[index // n]
        centers = torch.tensor(prefix + row[:-1]).cumsum(0).remainder(11).float()[len(prefix) - 1:]
        logits = -(torch.arange(16)[None, :] - centers[:, None]).square() / 16
        probabilities.append(logits.log_softmax(-1).gather(1, torch.tensor(row)[:, None]).flatten().tolist())
    request = {'prompts': prompts, 'prompt_token_ids': prefixes, 'adapter': '/adapter-1',
        'adapter_id': 1, 'group_size': n, 'max_tokens': 2048, 'min_tokens': 0, 'temperature': 1.,
        'top_p': 1., 'top_k': 0, 'presence_penalty': 0.}
    result = {'ok': True, 'adapter_id': 1, 'logprobs_mode': 'processed_logprobs',
        'prompt_token_ids': prefixes, 'rows': rows, 'selected_logprobs': probabilities,
        'tokens': sum(map(len, rows))}
    return checks, request, result


@pytest.mark.parametrize('groups,n', [(8, 8), (9, 8), (3, 5)])
def test_probability_check_covers_every_response_token_and_group_prefix(tmp_path, groups, n):
    import json
    checks, request, result = probability_fixture(tmp_path, groups=groups, n=n)
    checks.check_probabilities(request, result)
    report = checks.report['probability_checks'][0]
    assert report['responses'] == groups * n and report['tokens'] == groups * n * 130
    assert len(report['pairs']) == groups * n
    for index, pair in enumerate(report['pairs']):
        assert pair['row'] == index and pair['group'] == index // n
        assert pair['prompt_token_ids'] == request['prompt_token_ids'][index // n]
        assert len(pair['hf']) == len(pair['vllm']) == len(pair['token_ids']) == 130
    assert len(report['response_statistics']) == groups * n
    assert len(report['group_statistics']) == groups
    assert report['balanced_statistics']['gating'] is False
    assert all(ids.shape[0] == 1 and not gradients for ids, gradients in checks.trainer.actor.calls)
    persisted = json.loads((tmp_path / 'preflight-arm.json').read_text())
    assert persisted['probability_checks'][0]['pairs'] == report['pairs']
    before = len(checks.trainer.actor.calls)
    checks.check_probabilities(request, result)
    assert len(checks.trainer.actor.calls) == before


@pytest.mark.parametrize('location', [(0, 129), (63, 129)])
def test_probability_check_rejects_mismatch_after_old_sampling_window(tmp_path, location):
    import json
    checks, request, result = probability_fixture(tmp_path)
    row, token = location
    # A single enormous positive ratio disrupts global ESS, even though a
    # maximum token error by itself is only diagnostic.
    result['selected_logprobs'][row][token] -= 8.
    with pytest.raises(ValueError, match='tolerance'):
        checks.check_probabilities(request, result)
    persisted = json.loads((tmp_path / 'preflight-arm.json').read_text())
    assert persisted['status'] == persisted['probability_checks'][0]['status'] == 'failed'
    assert len(persisted['probability_checks'][0]['pairs']) == 64
    assert checks.checked_adapter_ids == set()


def test_response_and_group_tail_statistics_do_not_add_hidden_hard_gates(tmp_path):
    checks, request, result = probability_fixture(tmp_path, width=3)
    result['selected_logprobs'][-1][-1] -= .35
    checks.check_probabilities(request, result)
    entry = checks.report['probability_checks'][0]
    assert entry['status'] == 'passed' and entry['p99_abs_error'] < .15
    assert entry['group_statistics'][-1]['p99_abs_error'] > .15
    assert all(row['gating'] is False for row in entry['group_statistics'] + entry['response_statistics'])


def test_probability_check_persists_failure_for_short_response_positive_tail(tmp_path):
    import json
    import math
    checks, request, result = probability_fixture(tmp_path, width=914)
    result['rows'][0] = result['rows'][0][:2]
    result['selected_logprobs'][0] = result['selected_logprobs'][0][:2]
    result['selected_logprobs'][0][0] -= math.log(10.)
    result['tokens'] = sum(map(len, result['rows']))
    with pytest.raises(ValueError, match='tolerance'):
        checks.check_probabilities(request, result)
    persisted = json.loads((tmp_path / 'preflight-arm.json').read_text())
    report = persisted['probability_checks'][0]
    assert persisted['status'] == report['status'] == 'failed'
    assert len(report['pairs']) == 64 and report['tokens'] == 57584
    assert report['violations'] == ['loss_weighted_importance_ess_fraction']
    assert report['ess_ratio'] >= .99
    assert checks.checked_adapter_ids == set()


def test_all_three_adapter_versions_receive_one_complete_check(tmp_path):
    checks, request, result = probability_fixture(tmp_path, width=3)
    for adapter in (1, 2, 3, 3):
        request['adapter_id'] = result['adapter_id'] = adapter
        checks.check_probabilities(request, result)
    assert checks.checked_adapter_ids == {1, 2, 3}
    assert len(checks.report['probability_checks']) == 3
    assert all(len(entry['pairs']) == 64 for entry in checks.report['probability_checks'])


@pytest.mark.parametrize('damage', ['missing_row', 'extra_logprob_row', 'short_logprobs',
    'empty_response', 'bad_token', 'nonfinite', 'wrong_prefix', 'wrong_adapter', 'hf_nonfinite'])
def test_probability_alignment_and_nonfinite_fail_closed(tmp_path, damage, monkeypatch):
    import json
    checks, request, result = probability_fixture(tmp_path, width=3)
    result = copy.deepcopy(result)
    if damage == 'missing_row': result['rows'].pop()
    elif damage == 'extra_logprob_row': result['selected_logprobs'].append([-.1])
    elif damage == 'short_logprobs': result['selected_logprobs'][-1].pop()
    elif damage == 'empty_response': result['rows'][-1] = []
    elif damage == 'bad_token': result['rows'][-1][-1] = -1
    elif damage == 'nonfinite': result['selected_logprobs'][-1][-1] = float('nan')
    elif damage == 'wrong_prefix': result['prompt_token_ids'][-1] = [1, 9, 9]
    elif damage == 'wrong_adapter': result['adapter_id'] = 99
    else:
        original = checks.trainer.actor.forward
        def bad_forward(**kwargs):
            value = original(**kwargs)
            value.logits.fill_(float('nan'))
            return value
        monkeypatch.setattr(checks.trainer.actor, 'forward', bad_forward)
    with pytest.raises(ValueError):
        checks.check_probabilities(request, result)
    report = json.loads((tmp_path / 'preflight-arm.json').read_text())
    assert report['status'] == report['probability_checks'][0]['status'] == 'failed'
    assert checks.checked_adapter_ids == set()


def test_observed_socket_preserves_probabilities_after_low_quality_diagnostic(tmp_path):
    import json
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from scripts.direct_rl_preflight import WorkerChecks
    checks = WorkerChecks(SimpleNamespace(output_dir=tmp_path / 'report', worker_arm='grpo'))
    checks.trainer = SimpleNamespace(actor_tokenizer=SimpleNamespace(decode=lambda row, **kw: 'not an answer'),
        _encode_prompts=lambda prompts: ({'input_ids': torch.ones((len(prompts), 1), dtype=torch.long),
            'attention_mask': torch.ones((len(prompts), 1), dtype=torch.long)}, list(prompts)))
    # Keep actual socket framing/quality execution. Separate tests check logits.
    compared, sent = [], []
    checks.check_probabilities = lambda request, result: compared.append(request['adapter_id'])
    address = str(tmp_path / 'rpc.sock')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        listener.listen(1)
        listener.settimeout(5)
        def serve():
            for _ in range(2):
                connection, _ = listener.accept()
                connection.settimeout(5)
                with connection, connection.makefile('r') as stream:
                    for line in stream:
                        request = json.loads(line)
                        sent.append(request)
                        result = ({'ok': True, 'rows': [[0]] * 8} if request.get('probe') else
                                  {'ok': True, 'rows': [[1]], 'selected_logprobs': [[-.3]]})
                        connection.sendall((json.dumps(result) + '\n').encode())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(serve)
            for adapter_id, probe in [(1, False), (3, True)]:
                with checks.socket_factory(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(5)
                    connection.connect(address)
                    payload = ({'prompts': ['p'] * 8, 'prompt_token_ids': [[1]] * 8,
                        'group_size': 8, 'max_tokens': 2048, 'temperature': .9, 'min_tokens': 0,
                        'top_p': 1., 'top_k': 0, 'presence_penalty': 0.} if not probe else
                        {'prompts': ['final-single-prompt'], 'max_tokens': 8})
                    payload.update(adapter=f'/fresh-{adapter_id}', adapter_id=adapter_id, probe=probe)
                    connection.sendall(json.dumps(payload).encode())
                    with connection.makefile('r') as stream:
                        assert json.loads(stream.readline())['selected_logprobs'] == [[-.3]]
            future.result(timeout=5)
    assert compared == [1, 3]
    assert checks.report['quality']['before']['score'] == checks.report['quality']['after']['score'] == 0
    assert checks.report['quality']['after']['gating'] is False
    actual = [request for request in sent if not request.get('probe')]
    assert len(actual) == 2
    final = actual[-1]
    assert final['adapter_id'] == 3 and final['adapter'] == '/fresh-3'
    for key in ('prompts', 'prompt_token_ids', 'group_size', 'max_tokens', 'temperature',
                'min_tokens', 'top_p', 'top_k', 'presence_penalty'):
        assert final[key] == actual[0][key]


def test_capacity_updates_use_measured_scale_and_separate_artifacts(tmp_path, monkeypatch, actor):
    from scripts import direct_rl_preflight as gate, check_ssh_capacity as capacity
    from vpo_rm.trainer import TrainerConfig
    arm = tmp_path / 'lam4'
    arm.mkdir()
    metrics = arm / 'metrics.jsonl'
    metrics.write_text('real training\n')
    initial = gate.verify_initial_adapters(actor, '')
    trainer = SimpleNamespace(cfg=TrainerConfig(output_dir=str(arm), credit_microbatch_responses=1,
        length_reward_sigma0=2.5, credit_lambda=4, rollout_importance_correction=True),
        log_path=metrics, actor=actor, actor_tokenizer=object(), reward_tokenizer=object())
    checks = SimpleNamespace(args=SimpleNamespace(output_dir=arm), trainer=trainer,
                             report={'initial_adapters': initial, 'rollouts': 2})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda index: None)
    monkeypatch.setattr(capacity, 'make_synthetic_prompts', lambda *a, **kw: ['p'])
    monkeypatch.setattr(capacity, 'build_synthetic_rollout', lambda *a: (None, {'response_shape': [64, 2048]}))
    monkeypatch.setattr(capacity, '_memory_report', lambda *a: [])
    def synthetic_update(trainer, prompts, rollout, report):
        assert checks.capacity_active
        assert trainer.cfg.credit_lambda == 4 and trainer.cfg.length_reward_sigma0 == 2.5
        assert trainer.cfg.rollout_importance_correction is False
        with torch.no_grad():
            next(p for name, p in actor.named_parameters() if '.lora_B.' in name).fill_(.1)
        with trainer.log_path.open('a') as stream:
            stream.write('synthetic capacity\n')
        report['completed_steps'] = 2
    monkeypatch.setattr(capacity, 'run_capacity_steps', synthetic_update)
    report = gate.run_capacity(checks)
    assert report['source'] == 'trained_lam4_actor_after_two_real_rollouts'
    assert report['calibration_sigma0'] == 2.5
    assert report['reference']['base_sha256'] == initial['base_sha256']
    assert metrics.read_text() == 'real training\n'
    assert (tmp_path / 'capacity/metrics.jsonl').read_text() == 'synthetic capacity\n'
    assert checks.report['rollouts'] == 2 and not checks.capacity_active


@pytest.mark.parametrize('mismatch', ['torch', 'peft', 'small_gpu'])
def test_worker_rejects_invalid_runtime_before_loading(tmp_path, monkeypatch, mismatch):
    from scripts import direct_rl_preflight as gate
    from scripts.corrected_rl_launcher import RUNTIME
    versions = dict(RUNTIME)
    if mismatch != 'small_gpu': versions[mismatch] = 'wrong'
    monkeypatch.setattr(gate, 'experiment_launcher', lambda: SimpleNamespace(
        RUNTIME=RUNTIME, common_config=lambda root: {}, validation_identity=lambda root: {}))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(gate, 'runtime_report', lambda image: {'versions': versions,
        'gpus': [{'memory_bytes': (119 if mismatch == 'small_gpu' else 120) * 2**30}] * 3})
    args = SimpleNamespace(output_dir=tmp_path / 'arm', project_root=tmp_path,
                           runtime_image='image', worker_arm='grpo', sigma0=None)
    with pytest.raises(RuntimeError, match='runtime|120'):
        gate.run_worker(args)
    assert not args.output_dir.exists()


def test_rejected_initialization_does_not_leave_profile_dependencies_replaced(tmp_path, monkeypatch):
    from scripts import direct_rl_preflight as gate
    from scripts.corrected_rl_launcher import RUNTIME
    monkeypatch.setattr(gate, 'experiment_launcher', lambda: SimpleNamespace(
        RUNTIME=RUNTIME, common_config=lambda root: {'seed': 42, 'init_adapter': '/sft'},
        validation_identity=lambda root: {}))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(gate, 'runtime_report', lambda image: {'versions': RUNTIME,
        'gpus': [{'memory_bytes': 120 * 2**30}] * 3})
    original_trainer, original_socket = gate.profile.VPOTrainer, gate.profile.socket
    args = SimpleNamespace(output_dir=tmp_path / 'arm', project_root=tmp_path,
                           runtime_image='image', worker_arm='grpo', sigma0=None)
    try:
        with pytest.raises(ValueError, match='adapter|SFT'):
            gate.run_worker(args)
        assert gate.profile.VPOTrainer is original_trainer
        assert gate.profile.socket is original_socket
    finally:
        gate.profile.VPOTrainer, gate.profile.socket = original_trainer, original_socket


def test_main_shares_fresh_calibration_and_checks_initial_weights(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import direct_rl_preflight as gate
    from scripts.corrected_rl_launcher import common_config
    common = dict(common_config(tmp_path), init_adapter='')
    identity = {'reward_input_protocol': 'canonical_chat_v1', 'identity': 'fixture'}
    monkeypatch.setattr(gate, 'experiment_launcher', lambda: SimpleNamespace(
        common_config=lambda root: common, validation_identity=lambda root: identity))
    # Only process creation is external: all report parsing and calibration
    # validation run against files as in the production coordinator.
    source = tmp_path / 'source'
    for name in ['direct_rl_preflight.py', 'preflight_quality.py', 'check_ssh_capacity.py', 'direct_rl_launcher.py']:
        path = source / 'scripts' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen fixture\n')
    monkeypatch.setattr(gate, 'ROOT', source)
    launched = []
    def worker(command, **kwargs):
        arm = command[command.index('--worker-arm') + 1]
        destination = gate.Path(command[command.index('--output-dir') + 1])
        launched.append(arm)
        if arm == 'grpo': assert '--sigma0' not in command
        else: assert float(command[command.index('--sigma0') + 1]) == 2.5
        report = {**identity, 'status': 'passed', 'rollouts': 2, 'seed': 42,
            'initial_adapters': {'default_sha256': 'same', 'base_sha256': 'base'},
            'prompt_sha256': [['p1'], ['p2']]}
        gate.write_json(destination / 'preflight-arm.json', report)
        if arm == 'grpo':
            calibration = {key: common[key] for key in ('init_adapter', 'seed', 'model',
                'short_response_threshold', 'long_response_threshold', 'short_penalty_strength',
                'long_penalty_strength', 'advantage_std_floor_fraction', 'max_response_tokens')}
            calibration.update(reward_format='canonical_chat_v1', mode='soft', source='initial_policy',
                calibration_prompt_count=128, group_size=8, reward_model=common['rm'], sigma0=2.5,
                responses=[{'reward': 1.}] * 1024, prompt_sha256=[str(i) for i in range(128)],
                sampling={'temperature': 1, 'top_p': 1, 'top_k': 0, 'min_tokens': 0,
                    'presence_penalty': 0, 'policy_head_dtype': 'float32',
                    'rollout_correction': 'detached_token_is_pg_and_kl_v1'})
            gate.write_json(destination / 'length_reward_calibration.json', calibration)
    monkeypatch.setattr(gate.subprocess, 'run', worker)
    out = tmp_path / 'gate'
    gate.main(['--output-dir', str(out), '--project-root', str(tmp_path), '--runtime-image', 'image'])
    result = json.loads((out / 'gpu-validation.json').read_text())
    assert launched == ['grpo', 'lam4']
    assert result['status'] == result['arm_consistency']['status'] == 'passed'
    assert result['calibration']['sha256'] == hashlib.sha256(
        (out / 'grpo/length_reward_calibration.json').read_bytes()).hexdigest()
