"""Real BF16 Qwen/FP32 LoRA backward with the frozen FP32 output head."""
import copy

import pytest
import torch
from torch.nn import functional as F

from vpo_rm.policy_precision import FrozenFP32OutputHead, enable_fp32_output_head


@pytest.mark.parametrize('checkpointing', [False, True])
def test_fp32_head_keeps_default_trainable_and_reference_frozen_through_backward(tmp_path, checkpointing):
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(72)
    config = Qwen3Config(vocab_size=19, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, tie_word_embeddings=False, attention_dropout=0.)
    base = Qwen3ForCausalLM(config).bfloat16()
    base_state = copy.deepcopy(base.state_dict())
    initialized = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0., target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in initialized.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(.03)
    initial = tmp_path / 'initial'
    initialized.save_pretrained(initial, save_embedding_layers=False)

    loaded_base = Qwen3ForCausalLM(config).bfloat16()
    loaded_base.load_state_dict(base_state)
    actor = PeftModel.from_pretrained(loaded_base, initial, is_trainable=True)
    actor.load_adapter(initial, adapter_name='ref', is_trainable=False)
    actor.set_adapter('default')
    head = enable_fp32_output_head(actor)
    assert isinstance(head, FrozenFP32OutputHead)
    assert head.weight.dtype == torch.float32 and not head.weight.requires_grad
    assert actor.get_input_embeddings().weight.dtype == torch.bfloat16
    if checkpointing:
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
    attention = torch.ones_like(ids)
    # Exercise the same default -> reference -> default transition used when
    # caching the SFT KL probabilities before the differentiable actor forward.
    actor.eval()
    with torch.no_grad():
        before_logits = actor(ids, attention_mask=attention, use_cache=False).logits
        actor.set_adapter('ref')
        reference_logits = actor(ids, attention_mask=attention, use_cache=False).logits
        actor.set_adapter('default')
    torch.testing.assert_close(before_logits, reference_logits, rtol=0, atol=0)
    default_before = {name: value.detach().clone() for name, value in
                      get_peft_model_state_dict(actor, adapter_name='default', save_embedding_layers=False).items()}
    reference_before = {name: value.detach().clone() for name, value in
                        get_peft_model_state_dict(actor, adapter_name='ref', save_embedding_layers=False).items()}
    frozen_before = {name: value.detach().clone() for name, value in actor.named_parameters()
                     if not value.requires_grad}
    defaults = [value for name, value in actor.named_parameters() if '.default.' in name]
    references = [value for name, value in actor.named_parameters() if '.ref.' in name]
    assert defaults and all(value.requires_grad and value.dtype == torch.float32 for value in defaults)
    assert references and all(not value.requires_grad for value in references)

    optimizer = torch.optim.SGD((p for p in actor.parameters() if p.requires_grad), lr=.1)
    actor.train()
    logits = actor(ids, attention_mask=attention, use_cache=False).logits
    assert logits.dtype == torch.float32
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in defaults)
    assert any(value.grad.abs().sum() > 0 for value in defaults)
    assert all(value.grad is None for name, value in actor.named_parameters() if name in frozen_before)
    optimizer.step()

    default_after = get_peft_model_state_dict(actor, adapter_name='default', save_embedding_layers=False)
    reference_after = get_peft_model_state_dict(actor, adapter_name='ref', save_embedding_layers=False)
    assert any(not torch.equal(default_after[name], before) for name, before in default_before.items())
    assert all(torch.equal(reference_after[name], before) for name, before in reference_before.items())
    assert all(torch.equal(dict(actor.named_parameters())[name], before) for name, before in frozen_before.items())
    assert actor.active_adapters == ['default']
    assert head.weight.dtype == torch.float32 and head.weight.grad is None
