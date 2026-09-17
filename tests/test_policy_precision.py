import copy

import pytest
import torch
from torch import nn

from vpo_rm.policy_precision import FrozenFP32OutputHead, enable_fp32_output_head


class TinyActor(nn.Module):
    def __init__(self, *, bias=False):
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.lm_head = nn.Linear(4, 7, bias=bias).requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head


@pytest.mark.parametrize('bias', [False, True])
def test_fp32_head_preserves_state_names_values_and_hidden_gradient(bias):
    torch.manual_seed(17)
    actor = TinyActor(bias=bias)
    initial = copy.deepcopy(actor.state_dict())
    before_names = list(actor.state_dict())
    head = enable_fp32_output_head(actor)
    assert isinstance(head, FrozenFP32OutputHead)
    assert actor.get_output_embeddings() is head
    assert list(actor.state_dict()) == before_names
    for name, value in initial.items():
        torch.testing.assert_close(actor.state_dict()[name], value, rtol=0, atol=0)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected = expected_hidden @ initial['lm_head.weight'].t()
    if bias:
        expected = expected + initial['lm_head.bias']
    actual = head(hidden)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    coefficients = torch.randn_like(actual)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    assert all(not p.requires_grad and p.grad is None for p in head.parameters())
    assert enable_fp32_output_head(actor) is head


def test_cpu_autocast_does_not_round_output_to_bfloat16():
    actor = TinyActor()
    head = enable_fp32_output_head(actor)
    hidden = torch.linspace(-.937, 1.117, 12).reshape(3, 4).requires_grad_()
    expected = hidden @ head.weight.t()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = head(hidden)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, actual.bfloat16().float())
    actual.sum().backward()
    torch.testing.assert_close(hidden.grad, head.weight.sum(0).expand_as(hidden))


def test_bfloat16_storage_is_promoted_without_rounding_forward_output():
    actor = TinyActor(bias=True).to(torch.bfloat16)
    weight, bias = actor.lm_head.weight.float(), actor.lm_head.bias.float()
    head = enable_fp32_output_head(actor)
    assert head.weight.dtype == head.bias.dtype == torch.float32
    hidden = torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(head(hidden), hidden @ weight.t() + bias, rtol=0, atol=0)


@pytest.mark.parametrize('unsafe', ['tied', 'trainable_weight', 'trainable_bias', 'head_lora'])
def test_unsafe_output_heads_fail_before_replacement(unsafe):
    actor = TinyActor(bias=True)
    if unsafe == 'tied':
        actor.lm_head.weight = actor.embedding.weight
        actor.lm_head.weight.requires_grad_(False)
    elif unsafe == 'trainable_weight':
        actor.lm_head.weight.requires_grad_(True)
    elif unsafe == 'trainable_bias':
        actor.lm_head.bias.requires_grad_(True)
    else:
        actor.lm_head = nn.Sequential(actor.lm_head)
    before = actor.lm_head
    with pytest.raises(ValueError, match='frozen|tied|Linear|LoRA'):
        enable_fp32_output_head(actor)
    assert actor.lm_head is before


def test_tiny_peft_adapter_save_reload_requires_reenabling_head(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(21)
    cfg = Qwen3Config(vocab_size=19, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, tie_word_embeddings=False)
    base = Qwen3ForCausalLM(cfg)
    state = copy.deepcopy(base.state_dict())
    actor = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'lora_B' in name:
                param.fill_(.03)
    enable_fp32_output_head(actor)
    ids = torch.tensor([[1, 2, 3]])
    expected = actor(ids).logits.detach()
    actor.save_pretrained(tmp_path)
    saved = load_file(tmp_path / 'adapter_model.safetensors')
    assert saved and all('lora_' in name for name in saved)
    reloaded_base = Qwen3ForCausalLM(cfg)
    reloaded_base.load_state_dict(state)
    reloaded = PeftModel.from_pretrained(reloaded_base, tmp_path).eval()
    assert type(reloaded.get_output_embeddings()) is nn.Linear
    enable_fp32_output_head(reloaded)
    torch.testing.assert_close(reloaded(ids).logits, expected, rtol=0, atol=0)
