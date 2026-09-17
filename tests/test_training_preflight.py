"""CPU gates for the real three-GPU preflight orchestration."""
from pathlib import Path

import pytest
import torch


def test_profile_arguments_keep_main_training_protocol_and_share_calibration(tmp_path):
    from scripts import preflight_training as preflight
    from scripts.profile_vllm_full import parse_args
    common = {'model': 'base', 'rm': 'rm', 'init_adapter': 'sft', 'dataset_path': 'data',
              'max_rollouts': 250, 'max_response_tokens': 2048,
              'learning_rate': 5e-5, 'beta': .03, 'credit_microbatch_responses': 0,
              'checkpoint_interval': 250, 'keep_adapters_every': 250,
              'length_reward_mode': 'soft', 'length_calibration_prompts': 128}
    grpo = parse_args(preflight.profile_arguments(common, 'grpo', tmp_path / 'grpo'))
    vpo = parse_args(preflight.profile_arguments(common, 'lam8', tmp_path / 'lam8', 2.5))
    assert grpo.max_rollouts == vpo.max_rollouts == 2
    assert grpo.length_reward_sigma0 is None and vpo.length_reward_sigma0 == 2.5
    assert vpo.credit_lambda == 8 and vpo.freeze_stop_tokens and vpo.freeze_structural
    assert grpo.method == 'grpo' and not grpo.freeze_stop_tokens
    assert vpo.learning_rate == grpo.learning_rate == 5e-5
    assert vpo.credit_microbatch_responses == 0 and vpo.max_response_tokens == 2048
    assert vpo.checkpoint_interval == vpo.keep_adapters_every == 250


def test_probability_comparison_records_distribution_and_fails_fixed_limits():
    from scripts.preflight_training import logprob_error_report
    result = logprob_error_report(torch.tensor([-1., -2., -3.]), [-1.001, -1.998, -3.003])
    assert result['status'] == 'passed' and result['tokens'] == 3
    assert result['max_abs_error'] == pytest.approx(.003, abs=1e-6)
    with pytest.raises(ValueError, match='logprob'):
        logprob_error_report(torch.tensor([-1.]), [-2.])
    with pytest.raises(ValueError, match='finite'):
        logprob_error_report(torch.tensor([-1.]), [float('nan')])


@pytest.fixture
def adapters(tmp_path):
    from scripts import preflight_training
    from peft import LoraConfig, get_peft_model
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    actor = get_peft_model(GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=16, hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2)),
        LoraConfig(r=2, lora_alpha=4, target_modules=['query_key_value'], task_type='CAUSAL_LM'))
    initial = tmp_path / 'initial'
    actor.save_pretrained(initial)
    actor.load_adapter(initial, adapter_name='ref', is_trainable=False)
    actor.set_adapter('default')
    return actor, initial


def test_initial_adapter_gate_compares_disk_default_and_frozen_reference(adapters):
    from scripts.preflight_training import verify_initial_adapters
    actor, initial = adapters
    report = verify_initial_adapters(actor, initial)
    assert report['status'] == 'passed'
    assert report['default_sha256'] == report['reference_sha256']
    with torch.no_grad():
        next(p for name, p in actor.named_parameters() if '.ref.' in name).add_(.1)
    with pytest.raises(ValueError, match='reference|ref'):
        verify_initial_adapters(actor, initial)


def test_update_gate_requires_changed_trainable_adapter_and_unchanged_reference(adapters):
    from scripts.preflight_training import verify_initial_adapters, verify_adapter_update
    actor, initial = adapters
    baseline = verify_initial_adapters(actor, initial)
    with pytest.raises(ValueError, match='changed|update'):
        verify_adapter_update(actor, baseline)
    with torch.no_grad():
        next(p for name, p in actor.named_parameters() if '.default.' in name).add_(.1)
    assert verify_adapter_update(actor, baseline)['status'] == 'passed'
    next(p for name, p in actor.named_parameters() if '.ref.' in name).requires_grad_(True)
    with pytest.raises(ValueError, match='frozen|ref'):
        verify_adapter_update(actor, baseline)


def test_second_optimizer_step_must_change_weights_again(adapters):
    from scripts.preflight_training import verify_initial_adapters, verify_adapter_update
    actor, initial = adapters
    baseline = verify_initial_adapters(actor, initial)
    with torch.no_grad():
        next(p for name, p in actor.named_parameters() if '.default.' in name).add_(.1)
    first = verify_adapter_update(actor, baseline)
    with pytest.raises(ValueError, match='changed|update'):
        verify_adapter_update(actor, baseline, previous_default_sha256=first['default_sha256'])


def test_observed_socket_runs_quality_before_generation_and_checks_updated_adapter(tmp_path):
    import json
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from scripts.preflight_training import WorkerChecks

    controls = ['4', '15', 'Paris', 'H2O', '9', '7', 'A', 'purple']
    checks = WorkerChecks(SimpleNamespace(output_dir=tmp_path / 'report', worker_arm='grpo'))
    checks.trainer = SimpleNamespace(actor_tokenizer=SimpleNamespace(
        decode=lambda row, **kw: controls[row[0]]), _render_chat_prompt=lambda tok, text: text)
    received, compared = [], []
    # Numerical matching has its own tests; this test exercises actual socket
    # ordering, selected-probability transport and quality gating.
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
                        received.append(request)
                        if request.get('probe'):
                            result = {'ok': True, 'rows': [[i] for i in range(8)]}
                        else:
                            result = {'ok': True, 'rows': [[0]], 'selected_logprobs': [[-.3]],
                                      'prompt_token_ids': [[1]], 'logprobs_mode': 'processed_logprobs'}
                        connection.sendall((json.dumps(result) + '\n').encode())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(serve)
            for adapter_id, probe in [(1, False), (3, True)]:
                with checks.socket_factory(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(5)
                    connection.connect(address)
                    connection.sendall(json.dumps({'prompts': ['p'], 'adapter': '/sft',
                        'adapter_id': adapter_id, 'probe': probe, 'max_tokens': 8}).encode())
                    with connection.makefile('r') as stream:
                        result = json.loads(stream.readline())
                    assert result['selected_logprobs'] == [[-.3]]
                    assert result['prompt_token_ids'] == [[1]]
            future.result(timeout=5)
    assert compared == [1, 3]
    assert len(received) == 4
    assert received[0]['probe'] and received[2]['probe']
    assert received[1]['return_logprobs'] and received[3]['return_logprobs']
    assert received[3]['group_size'] == 1 and received[3]['min_tokens'] == 0
    assert checks.report['quality']['before']['score'] == 8
    assert checks.report['quality']['after']['passed']


def test_capacity_artifacts_do_not_append_to_real_training_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import preflight_training as preflight, check_ssh_capacity as capacity
    from vpo_rm.trainer import TrainerConfig
    arm = tmp_path / 'lam8'
    arm.mkdir()
    metrics = arm / 'metrics.jsonl'
    metrics.write_text('real training\n')
    trainer = SimpleNamespace(cfg=TrainerConfig(output_dir=str(arm), credit_microbatch_responses=1,
                                               rollout_importance_correction=True), log_path=metrics, actor=object(),
                              actor_tokenizer=object(), reward_tokenizer=object())
    checks = SimpleNamespace(args=SimpleNamespace(output_dir=arm), trainer=trainer,
                             report={'initial_adapters': {}, 'rollouts': 2})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda index: None)
    monkeypatch.setattr(capacity, 'make_synthetic_prompts', lambda *a, **kw: ['p'])
    monkeypatch.setattr(capacity, 'build_synthetic_rollout', lambda *a: (None, {}))
    monkeypatch.setattr(capacity, '_memory_report', lambda *a: [])
    monkeypatch.setattr(preflight, 'verify_adapter_update', lambda *a: {'status': 'passed'})
    def synthetic_update(trainer, prompts, rollout, report):
        assert checks.capacity_active
        assert trainer.cfg.credit_microbatch_responses == 1
        assert trainer.cfg.rollout_importance_correction is False
        assert report['sampling_correction_testonly'] == 'disabled_synthetic_tokens_have_no_sampler'
        with trainer.log_path.open('a') as stream:
            stream.write('synthetic capacity\n')
        report['completed_steps'] = 2
    monkeypatch.setattr(capacity, 'run_capacity_steps', synthetic_update)
    preflight.run_capacity(checks)
    assert metrics.read_text() == 'real training\n'
    assert (tmp_path / 'capacity/metrics.jsonl').read_text() == 'synthetic capacity\n'
    assert checks.report['rollouts'] == 2 and not checks.capacity_active


def test_runtime_report_records_exact_formal_package_set(monkeypatch):
    from types import SimpleNamespace
    from scripts import preflight_training as preflight
    from scripts.corrected_rl_launcher import RUNTIME
    monkeypatch.setattr(preflight.importlib.metadata, 'version', lambda name: RUNTIME.get(name, 'extra'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'test GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda index: SimpleNamespace(total_memory=120 * 2**30))
    assert preflight.runtime_report('test-image')['versions'] == RUNTIME


@pytest.mark.parametrize('wrong_package,small_gpu', [
    ('torch', None), ('transformers', None), ('vllm', None),
    ('peft', None), ('pyarrow', None), (None, 0), (None, 1), (None, 2),
])
def test_worker_rejects_wrong_runtime_before_model_loading(tmp_path, monkeypatch, wrong_package, small_gpu):
    from types import SimpleNamespace
    from scripts import preflight_training as preflight, corrected_rl_launcher as launcher
    versions = dict(launcher.RUNTIME)
    if wrong_package:
        versions[wrong_package] = 'wrong-version'
    monkeypatch.setattr(preflight.importlib.metadata, 'version', lambda name: versions.get(name, 'extra'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'test GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(
        total_memory=(119 if index == small_gpu else 120) * 2**30))
    monkeypatch.setattr(launcher, 'validation_identity', lambda root: {})
    monkeypatch.setattr(launcher, 'common_config', lambda root: {})
    def forbidden_loading(cls, config):
        pytest.fail('Model loading was reached before runtime validation')
    monkeypatch.setattr(preflight.profile.VPOTrainer, 'from_pretrained', classmethod(forbidden_loading))
    monkeypatch.setattr(preflight.profile, 'main',
                        lambda: preflight.profile.VPOTrainer.from_pretrained(None))
    args = SimpleNamespace(output_dir=tmp_path / 'arm', project_root=tmp_path,
                           runtime_image='test-image', worker_arm='grpo', sigma0=None)
    with pytest.raises(RuntimeError, match='version|runtime|120'):
        preflight.run_worker(args)
    assert not args.output_dir.exists()
