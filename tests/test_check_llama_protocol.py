"""The launch gate must reject malformed protocols and exercise real scoring."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_llama_reward_protocol import llama_tokenizers


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_checker_writes_failure_report(tmp_path):
    output = tmp_path / "failed.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/check_llama_protocol.py"),
        "--actor", str(tmp_path / "missing-actor"),
        "--reward", str(tmp_path / "missing-rm"),
        "--init-adapter", str(tmp_path / "missing-adapter"),
        "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert output.is_file(), result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["mode"] == "cpu"
    assert report["error"]


def test_protocol_checker_rejects_length_overflow_and_wrong_terminal():
    from scripts.check_llama_protocol import validate_reward_rows
    valid = [128000] + [42] * 4094 + [128009]
    assert validate_reward_rows([valid]) == [4096]
    with pytest.raises(ValueError, match="4096"):
        validate_reward_rows([[128000] + [42] * 4095 + [128009]])
    with pytest.raises(ValueError, match="EOT"):
        validate_reward_rows([[128000, 42, 128001]])
    with pytest.raises(ValueError, match="BOS"):
        validate_reward_rows([[128000, 128000, 42, 128009]])


def test_protocol_checker_rejects_non_native_rm_padding(llama_tokenizers):
    from copy import deepcopy
    from scripts.check_llama_protocol import audit_tokenizers
    actor, reward = llama_tokenizers
    bad = deepcopy(reward)
    bad.pad_token = bad.eos_token
    with pytest.raises(ValueError, match="pad"):
        audit_tokenizers(actor, bad)


def test_protocol_checker_confirms_real_saved_tokenizers(llama_tokenizers):
    from scripts.check_llama_protocol import audit_tokenizers
    report = audit_tokenizers(*llama_tokenizers)
    assert report["stop_token_ids"] == [128001, 128008, 128009]
    assert report["pad_token_id"] == 128004
    assert report["cases"]["ascii"]["response_positions"] == [37, 38, 39, -1]
    assert report["cases"]["trim"]["mapped_content_tokens"] == 3
    assert report["cases"]["unicode"]["mapped_content_tokens"] == 6
    assert report["length_boundary"] == {"accepted_tokens": 4096, "rejected_tokens": 4097,
                                          "serialization_preserved": True}


def test_protocol_checker_accepts_transformers5_dictionary_default(llama_tokenizers):
    """TF5 returns BatchEncoding unless token-list output is requested explicitly."""
    from scripts.check_llama_protocol import audit_tokenizers
    actor, reward = llama_tokenizers

    class DictionaryDefaultTokenizer:
        def __call__(self, *args, **kwargs):
            return reward(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(reward, name)

        def apply_chat_template(self, messages, *, tokenize=True, return_dict=True, **kwargs):
            if not tokenize:
                return reward.apply_chat_template(messages, tokenize=False, **kwargs)
            return reward.apply_chat_template(messages, tokenize=True, return_dict=return_dict, **kwargs)

    report = audit_tokenizers(actor, DictionaryDefaultTokenizer())
    assert report["cases"]["ascii"]["response_positions"] == [37, 38, 39, -1]
    assert report["length_boundary"]["accepted_tokens"] == 4096


def test_cpu_report_binds_actual_model_and_adapter_artifacts(llama_tokenizers):
    from scripts.check_llama_protocol import audit_cpu
    root = Path("/data/VPO-RM/models")
    report = audit_cpu(root / "Llama-3.1-8B-Instruct",
                       root / "Skywork-Reward-Llama-3.1-8B-v0.2",
                       root / "sft-llama31-8b-instruct-clean2k5e2-20260917")
    assert set(report["identity"]) == {"actor", "reward", "init_adapter"}
    assert len(report["identity"]["init_adapter"]["adapter_model.safetensors"]) == 64
    assert report["checkpoint_files"]["reward"]["head_and_embedding_shapes"]["score.weight"] == [1, 4096]
    assert report["checkpoint_files"]["actor"]["head_and_embedding_shapes"]["lm_head.weight"] == [128256, 4096]


@pytest.fixture
def tiny_llama_reward():
    import torch
    from transformers import LlamaConfig, LlamaForSequenceClassification
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=128256, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, pad_token_id=128004,
                         num_labels=1, attention_dropout=0.)
    return LlamaForSequenceClassification(config).eval()


def test_reward_check_compares_singletons_padding_and_input_gradients(
        llama_tokenizers, tiny_llama_reward):
    from scripts.check_llama_protocol import audit_reward_model
    report = audit_reward_model(tiny_llama_reward, llama_tokenizers[1], "cpu", atol=1e-5)
    assert report["max_score_difference"] < 1e-5
    assert report["nonzero_input_gradient"]
    assert report["padding_gradient_max_abs"] == 0
    assert report["frozen_parameters"]
    assert report["terminal_eot_pooling"]
    assert set(report["padding_sides"]) == {"left", "right"}


def test_reward_check_rejects_zero_reward_gradient(llama_tokenizers, tiny_llama_reward):
    import torch
    from scripts.check_llama_protocol import audit_reward_model
    with torch.no_grad():
        tiny_llama_reward.score.weight.zero_()
    with pytest.raises(ValueError, match="gradient"):
        audit_reward_model(tiny_llama_reward, llama_tokenizers[1], "cpu", atol=1e-5)


@pytest.mark.parametrize('failure', ['native_vs_wrapper', 'wrapped_vs_singleton', 'nonfinite'])
def test_gpu_failure_report_preserves_scores_and_identifies_comparison(
        tmp_path, monkeypatch, llama_tokenizers, tiny_llama_reward, failure):
    """Exercise actual scoring/report serialization; replace only GPU/checkpoint loading."""
    from types import SimpleNamespace
    import torch
    import transformers
    from scripts import check_llama_protocol as checker
    from vpo_rm import reward as reward_module
    tiny_llama_reward.bfloat16()

    class DivergentReward(reward_module.LastTokenReward):
        def forward(self, **kwargs):
            score = super().forward(**kwargs)
            if failure == 'nonfinite':
                return score * float('nan')
            return score + (0.5 if kwargs['attention_mask'][:, -1].eq(0).any() else 0)

    monkeypatch.setattr(reward_module, 'LastTokenReward', DivergentReward)
    if failure == 'wrapped_vs_singleton':
        original_forward = tiny_llama_reward.forward

        def shifted_native(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if kwargs['attention_mask'][:, -1].eq(0).any():
                result.logits = result.logits + 0.5
            return result

        monkeypatch.setattr(tiny_llama_reward, 'forward', shifted_native)
    original_audit = checker.audit_reward_model
    monkeypatch.setattr(checker, 'audit_reward_model',
        lambda model, tokenizer, device, **kwargs: original_audit(model, tokenizer, 'cpu', **kwargs))
    monkeypatch.setattr(checker, 'audit_cpu', lambda *args: {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'CPU test fixture')
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: llama_tokenizers[1])
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained',
        lambda *args, **kwargs: SimpleNamespace(to=lambda device: tiny_llama_reward))
    output = tmp_path / 'gpu-failure.json'
    assert checker.main(['--actor', 'unused', '--reward', 'unused', '--init-adapter', 'unused',
                         '--output', str(output), '--gpu']) == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['runtime'].get('gpu_name') == 'CPU test fixture'
    assert 'gpu' in report, 'GPU failure discarded all numerical evidence'
    gpu = report['gpu']
    assert gpu['status'] == 'failed' and gpu['audit_protocol'] == 'llama_reward_precision_audit_v2'
    phase = 'padding_fp32' if failure == 'wrapped_vs_singleton' else 'production_bf16'
    evidence = gpu[phase]
    assert evidence['status'] == 'failed'
    assert evidence['score_atol'] == (0.001 if phase == 'padding_fp32' else 0.125)
    checks = evidence['score_checks']
    if failure == 'nonfinite':
        assert len(checks) == 1
        assert checks[0]['padding_side'] == 'none'
        assert checks[0]['wrapped_score'] is None
        assert checks[0]['finite'] == {'native': True, 'wrapped': False, 'singleton': True}
        assert checks[0]['comparisons']['native_vs_wrapper']['abs_delta'] is None
        assert 'Nonfinite singleton' in report['error']
    else:
        assert len(checks) == 15  # Five singletons, then complete left/right batches.
        right = [row for row in checks if row['padding_side'] == 'right']
        assert len(right) == 5
        assert [row['native_pool_position'] for row in right] == [40, 42, 36, 178, 178]
        assert [row['wrapped_pool_position'] for row in right] == [40, 42, 36, 178, 178]
        assert all(row['batch_width'] == 182 and row['terminal_token_id'] == 128009 for row in right)
        for row in right:
            assert all(row['finite'].values())
            assert all(isinstance(row[key], float) for key in
                       ('native_score', 'wrapped_score', 'singleton_score'))
            assert row['comparisons'][failure]['abs_delta'] == pytest.approx(0.5, abs=1e-5)
            assert row['comparisons'][failure]['passed'] is False
        assert failure in report['error'] and 'right' in report['error']
    json.dumps(report, allow_nan=False)


def test_precision_audit_full_fp32_cast_and_singleton_gradient_parity(
        monkeypatch, llama_tokenizers, tiny_llama_reward):
    import torch
    from scripts import check_llama_protocol as checker
    from vpo_rm import reward as reward_module
    audit = getattr(checker, 'audit_gpu_reward_model', None)
    assert callable(audit), 'Precision-aware GPU audit is missing'
    model = tiny_llama_reward.bfloat16()
    # Manufacture native BF16 batch drift without changing any same-input path.
    hook = model.score.register_forward_hook(lambda module, inputs, output:
        output + (0.5 if module.weight.dtype == torch.bfloat16 and inputs[0].shape[0] > 1 else 0))
    real_gradients = reward_module.reward_input_gradients
    shapes = []

    def tracked_gradients(reward, ids, mask, **kwargs):
        shapes.append(tuple(ids.shape))
        return real_gradients(reward, ids, mask, **kwargs)

    monkeypatch.setattr(reward_module, 'reward_input_gradients', tracked_gradients)
    try:
        report = audit(model, llama_tokenizers[1], 'cpu')
    finally:
        hook.remove()
    assert report['status'] == 'passed'
    assert report['audit_protocol'] == 'llama_reward_precision_audit_v2'
    assert report['production_microbatch_responses'] == 1
    bf16, fp32 = report['production_bf16'], report['padding_fp32']
    assert bf16['parameter_dtype'] == 'torch.bfloat16' and bf16['score_atol'] == 0.125
    assert bf16['batch_invariance_required'] is False and bf16['max_batch_score_difference'] > 0.4
    assert bf16['max_score_difference'] <= 0.125
    assert fp32['parameter_dtype'] == 'torch.float32' and fp32['score_atol'] == 0.001
    assert fp32['batch_invariance_required'] is True and fp32['max_batch_score_difference'] <= 0.001
    assert fp32['tf32_disabled'] is True and fp32['float32_matmul_precision'] == 'highest'
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
    assert shapes == [(1, 41), (1, 43), (1, 37), (1, 179), (1, 179), (5, 182), (5, 182)] * 2
    for phase in (bf16, fp32):
        assert phase['same_input_paths'] == ['native', 'wrapper_no_grad', 'wrapper_input_gradients']
        assert phase['attention_implementation'] == 'sdpa'
        assert phase['padding_gradient_max_abs'] == 0 and phase['frozen_parameters']
        assert phase['mapped_gradient_norm_min'] > 0
        assert all('wrapper_input_gradients_score' in row for row in phase['score_checks'])


@pytest.mark.parametrize('dtype_name,phase_name', [('bfloat16', 'production_bf16'), ('float32', 'padding_fp32')])
def test_precision_audit_rejects_same_input_mismatch_in_either_phase(
        monkeypatch, llama_tokenizers, tiny_llama_reward, dtype_name, phase_name):
    import torch
    from scripts import check_llama_protocol as checker
    from vpo_rm import reward as reward_module
    audit = getattr(checker, 'audit_gpu_reward_model', None)
    assert callable(audit), 'Precision-aware GPU audit is missing'

    class WrongWrapper(reward_module.LastTokenReward):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            return result + (0.5 if self.score_head.weight.dtype == getattr(torch, dtype_name) else 0)

    monkeypatch.setattr(reward_module, 'LastTokenReward', WrongWrapper)
    evidence = {}
    with pytest.raises(ValueError, match='[Nn]ative.*wrapper'):
        audit(tiny_llama_reward.bfloat16(), llama_tokenizers[1], 'cpu', diagnostics=evidence)
    assert evidence['status'] == 'failed' and evidence[phase_name]['status'] == 'failed'
    assert evidence[phase_name]['score_checks'][0]['comparisons']['native_vs_wrapper']['passed'] is False


def test_precision_audit_rejects_fp32_padding_drift(llama_tokenizers, tiny_llama_reward):
    import torch
    from scripts import check_llama_protocol as checker
    audit = getattr(checker, 'audit_gpu_reward_model', None)
    assert callable(audit), 'Precision-aware GPU audit is missing'
    model = tiny_llama_reward.bfloat16()
    hook = model.score.register_forward_hook(lambda module, inputs, output:
        output + (0.01 if module.weight.dtype == torch.float32 and inputs[0].shape[0] > 1 else 0))
    evidence = {}
    try:
        with pytest.raises(ValueError, match='singleton'):
            audit(model, llama_tokenizers[1], 'cpu', diagnostics=evidence)
    finally:
        hook.remove()
    assert evidence['production_bf16']['status'] == 'passed'
    assert evidence['padding_fp32']['status'] == 'failed'
    assert evidence['padding_fp32']['score_atol'] == 0.001
    assert evidence['padding_fp32']['score_checks'][-1]['comparisons']['wrapped_vs_singleton']['abs_delta'] > 0.009


def test_precision_audit_rejects_singleton_gradient_only_score_change(
        monkeypatch, llama_tokenizers, tiny_llama_reward):
    from scripts.check_llama_protocol import audit_gpu_reward_model
    from vpo_rm import reward as reward_module

    class WrongGradientPath(reward_module.LastTokenReward):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            return result + (0.5 if kwargs['inputs_embeds'].requires_grad else 0)

    monkeypatch.setattr(reward_module, 'LastTokenReward', WrongGradientPath)
    evidence = {}
    with pytest.raises(ValueError, match='native_vs_wrapper_input_gradients'):
        audit_gpu_reward_model(tiny_llama_reward.bfloat16(), llama_tokenizers[1], 'cpu', diagnostics=evidence)
    check = evidence['production_bf16']['score_checks'][0]
    assert check['padding_side'] == 'none'
    assert check['comparisons']['native_vs_wrapper']['passed'] is True
    assert check['comparisons']['native_vs_wrapper_input_gradients']['abs_delta'] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def llama_base_tokenizers():
    from transformers import PreTrainedTokenizerFast
    root = Path("/data/VPO-RM/models")
    paths = (root / "sft-llama31-8b-base-clean2k5e2-20260919", root / "Skywork-Reward-Llama-3.1-8B-v0.2")
    if not all((path / "tokenizer.json").is_file() for path in paths):
        pytest.skip("Local Llama base SFT tokenizer artifacts are required")
    return tuple(PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True) for path in paths)


def test_protocol_checker_audits_base_native_eos_actor(llama_base_tokenizers):
    from scripts.check_llama_protocol import audit_tokenizers
    actor, reward = llama_base_tokenizers
    assert actor.eos_token_id == 128001
    with pytest.raises(ValueError, match="BOS/EOT"):
        audit_tokenizers(actor, reward)
    report = audit_tokenizers(actor, reward, actor_eos=128001)
    assert report["actor_response_eos_id"] == 128001 and report["response_eot_id"] == 128009
    assert report["stop_token_ids"] == [128001, 128008, 128009]
    assert report["cases"]["actor_stop"]["response_positions"] == [37, 38, 39, -1]
    assert report["cases"]["actor_stop_empty"]["mapped_content_tokens"] == 0
    assert report["cases"]["ascii"]["response_positions"] == [37, 38, 39, -1]
    assert report["actor_chat_template_sha256"] == report["reward_chat_template_sha256"]


def test_instruct_actor_rejects_base_terminator_expectation(llama_tokenizers):
    from scripts.check_llama_protocol import audit_tokenizers
    actor, reward = llama_tokenizers
    with pytest.raises(ValueError, match="BOS/EOT"):
        audit_tokenizers(actor, reward, actor_eos=128001)
    with pytest.raises(ValueError, match="registered"):
        audit_tokenizers(actor, reward, actor_eos=128004)


def test_cpu_report_binds_base_actor_and_native_eos_adapter(llama_base_tokenizers):
    from scripts.check_llama_protocol import audit_cpu
    root = Path("/data/VPO-RM/models")
    if not (root / "Llama-3.1-8B/DOWNLOAD_MANIFEST.json").is_file():
        pytest.skip("Local verified Llama base checkpoint is required")
    report = audit_cpu(root / "Llama-3.1-8B", root / "Skywork-Reward-Llama-3.1-8B-v0.2",
                       root / "sft-llama31-8b-base-clean2k5e2-20260919", actor_eos=128001)
    assert report["protocol"]["actor_response_eos_id"] == 128001
    assert report["checkpoint_files"]["actor"]["head_and_embedding_shapes"]["lm_head.weight"] == [128256, 4096]
    with pytest.raises(ValueError, match="BOS/EOT|SFT token protocol"):
        audit_cpu(root / "Llama-3.1-8B", root / "Skywork-Reward-Llama-3.1-8B-v0.2",
                  root / "sft-llama31-8b-base-clean2k5e2-20260919")
