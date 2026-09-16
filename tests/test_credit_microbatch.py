"""Response credit batching must preserve complete-group VPO updates."""
from unittest.mock import patch

import pytest
import torch

import vpo_rm.trainer as trainer_module
from vpo_rm.reward import LastTokenReward
from vpo_rm.trainer import TrainerConfig, VPOTrainer


class Tokenizer:
    eos_token_id = pad_token_id = 0
    bos_token_id = 1
    all_special_ids = [0, 1, 6]

    def get_vocab(self):
        return {"<|endoftext|>": 0, "p": 1, "a": 2, "b": 3, "c": 4,
                "d": 5, "<|im_end|>": 6, "\n": 7, " ": 8, "q": 9, "z": 10}

    def decode(self, ids, **kwargs):
        vocab = {v: k for k, v in self.get_vocab().items()}
        return "".join(vocab[i] for i in ids)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 9] if text == "p" else [9, 1, 9]}


def make_trainer(path, credit_microbatch, *, method="vpo_rm", dtype=torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification

    torch.manual_seed(37)
    model_cfg = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=8,
                           max_position_embeddings=64, attention_dropout=0.,
                           bos_token_id=1, eos_token_id=0, pad_token_id=0,
                           num_labels=1)
    actor = Qwen3ForCausalLM(model_cfg).to(dtype=dtype)
    rm = Qwen3ForSequenceClassification(model_cfg).to(dtype=dtype)
    reward = LastTokenReward(rm.model, rm.score)
    config = TrainerConfig(
        actor_device="cpu", reward_device="cpu", output_dir=str(path),
        allocated_gpu_count=0, lora=False, gradient_checkpointing=False,
        method=method, group_size=3, prompts_per_rollout=2,
        min_response_tokens=1, max_response_tokens=5,
        microbatch_responses=1, credit_microbatch_responses=credit_microbatch,
        temperature=.7, tau=.8, credit_lambda=4., freeze_stop_tokens=True,
        freeze_structural=True, beta=.2, learning_rate=.002,
        policy_epochs_per_rollout=2, optimizer_minibatch_responses=4,
        token_chunk_size=3, vocab_chunk_size=5)
    trainer = VPOTrainer(actor, Tokenizer(), reward, Tokenizer(), config)
    # Drift from the frozen initial reference, exercising nonzero anchored KL.
    with torch.no_grad():
        actor.get_output_embeddings().weight[2].add_(.04)
    responses = torch.tensor([[2, 3, 7, 4, 0], [5, 7, 6, 0, 0],
                              [2, 8, 3, 4, 10], [3, 4, 0, 0, 0],
                              [4, 5, 2, 7, 6], [5, 3, 0, 0, 0]])
    lengths = torch.tensor([5, 3, 5, 3, 5, 3])
    valid = torch.arange(5)[None, :] < lengths[:, None]
    prefixes = torch.tensor([[0, 1, 9]] * 3 + [[9, 1, 9]] * 3)
    ids = torch.cat((prefixes, responses), -1)
    attention = torch.cat((prefixes != 0, valid), -1).long()
    positions = torch.arange(3, 8)[None, :].expand(6, -1)
    reasons = ["stop", "stop", "length", "stop", "stop", "stop"]

    def rollout(prompts):
        actor.eval()
        return ids, attention, positions, responses, valid, ["p"] * 6, reasons

    trainer.rollout = rollout
    return trainer, valid, responses


def run_and_capture(path, credit_microbatch, *, method="vpo_rm", dtype=torch.float32):
    trainer, valid, responses = make_trainer(path, credit_microbatch, method=method, dtype=dtype)
    original_cache = trainer_module.build_credit_cache
    original_entropy = trainer._old_policy_entropy
    caches, entropies, forward_batches, cache_inputs, events = [], [], [], [], []

    def cache_capture(*args, **kwargs):
        cache_inputs.append((args[0].shape[0], args[2].shape[0]))
        result = original_cache(*args, **kwargs)
        caches.append(result)
        return result

    def entropy_capture(*args, **kwargs):
        result = original_entropy(*args, **kwargs)
        entropies.append(result.clone())
        return result

    def actor_forward(module, args, kwargs):
        if not torch.is_grad_enabled():
            forward_batches.append(kwargs["input_ids"].shape[0])
            events.append("actor")

    actor_hook = trainer.actor.register_forward_pre_hook(actor_forward, with_kwargs=True)
    reward_hook = trainer.reward.register_forward_pre_hook(lambda *args: events.append("reward"))
    before = {name: p.detach().clone() for name, p in trainer.actor.named_parameters()}
    try:
        with patch("vpo_rm.trainer.build_credit_cache", side_effect=cache_capture), \
                patch.object(trainer, "_old_policy_entropy", side_effect=entropy_capture):
            metrics = trainer.train_rollout(["p", "q"])
    finally:
        actor_hook.remove()
        reward_hook.remove()
    params = {name: p.detach().clone() for name, p in trainer.actor.named_parameters()}
    assert any(not torch.equal(before[name], value) for name, value in params.items())
    assert all(p.grad is None for p in trainer.reward.parameters())
    return dict(caches=caches, entropies=entropies, forward_batches=forward_batches,
                cache_inputs=cache_inputs, params=params, metrics=metrics, events=events,
                valid=valid, responses=responses)


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_credit_microbatch_matches_dense_cache_entropy_and_real_update(tmp_path, microbatch):
    dense = run_and_capture(tmp_path / "dense", 0)
    micro = run_and_capture(tmp_path / "micro", microbatch)
    expected_batches = [min(microbatch, 6 - start) for start in range(0, 6, microbatch)]
    assert dense["forward_batches"] == [6]
    assert micro["forward_batches"] == expected_batches
    assert micro["cache_inputs"] == [(size, size) for size in expected_batches]
    assert micro["events"][0] == "reward"
    assert dense["events"][0] == "actor"
    for field in ["old_logp", "advantage", "direction", "weight", "tau_used"]:
        def combine(run):
            return torch.cat([getattr(c, field) if field == "old_logp" else getattr(c.credit, field)
                              for c in run["caches"]])
        torch.testing.assert_close(combine(micro), combine(dense), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.cat(micro["entropies"]), torch.cat(dense["entropies"]),
                               atol=2e-6, rtol=2e-6)
    for name, value in dense["params"].items():
        torch.testing.assert_close(micro["params"][name], value, atol=2e-5, rtol=2e-5)
    for name in ["loss", "response_entropy", "credit_ess_ratio", "kl_to_init", "group_sigma_mean"]:
        assert micro["metrics"][name] == pytest.approx(dense["metrics"][name], abs=2e-5, rel=2e-5)
    assert micro["metrics"]["optimizer_steps"] == dense["metrics"]["optimizer_steps"] == 4
    credit = dense["caches"][0].credit
    valid, tokens = dense["valid"], dense["responses"]
    assert bool(((credit.weight - 1).abs()[valid] > 1e-4).any())
    frozen = valid & ((tokens == 0) | (tokens == 6) | (tokens == 7) | (tokens == 8))
    torch.testing.assert_close(credit.weight[frozen], torch.ones_like(credit.weight[frozen]))
    assert not credit.weight[~valid].any()


def test_bf16_credit_microbatch_keeps_float32_temperature_protocol(tmp_path):
    dense = run_and_capture(tmp_path / "dense", 0, dtype=torch.bfloat16)
    micro = run_and_capture(tmp_path / "micro", 2, dtype=torch.bfloat16)
    for field in ["old_logp", "advantage", "direction", "weight", "tau_used"]:
        wanted = getattr(dense["caches"][0], field) if field == "old_logp" else getattr(dense["caches"][0].credit, field)
        got = torch.cat([getattr(c, field) if field == "old_logp" else getattr(c.credit, field)
                         for c in micro["caches"]])
        torch.testing.assert_close(got, wanted, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.cat(micro["entropies"]), torch.cat(dense["entropies"]),
                               atol=2e-6, rtol=2e-6)
    for name, value in dense["params"].items():
        torch.testing.assert_close(micro["params"][name], value, atol=5e-4, rtol=2e-3)


def test_grpo_ignores_credit_microbatch_and_never_builds_credit_cache(tmp_path):
    dense = run_and_capture(tmp_path / "dense", 0, method="grpo")
    micro = run_and_capture(tmp_path / "micro", 2, method="grpo")
    assert dense["caches"] == micro["caches"] == []
    assert dense["entropies"] == micro["entropies"] == []
    for name, value in dense["params"].items():
        torch.testing.assert_close(micro["params"][name], value, atol=0, rtol=0)


@pytest.mark.parametrize("value", [-1, 1.5, True, False, "1", None])
def test_credit_microbatch_rejects_non_integer_or_negative(value):
    with pytest.raises(ValueError, match="credit_microbatch_responses"):
        TrainerConfig(credit_microbatch_responses=value).resolved()


@pytest.mark.parametrize("value", [0, 1, 2])
def test_credit_microbatch_accepts_nonnegative_integer(value):
    assert TrainerConfig(credit_microbatch_responses=value).resolved().credit_microbatch_responses == value
