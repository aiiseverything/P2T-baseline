"""Observable reward/update behavior for the opt-in soft length policy."""
import json
import types

import pytest
import torch
from torch import nn

from vpo_rm.trainer import TrainerConfig, VPOTrainer
from test_trainer_regressions import TinyActor, TinyTokenizer


def make_trainer(tmp_path, rows=(), **overrides):
    options = dict(actor_device="cpu", reward_device="cpu", allocated_gpu_count=0,
                   output_dir=str(tmp_path), group_size=2, prompts_per_rollout=1,
                   max_response_tokens=32, lora=False, gradient_checkpointing=False,
                   method="grpo", beta=0., length_reward_mode="soft",
                   length_reward_sigma0=2., long_response_threshold=16)
    options.update(overrides)
    t = VPOTrainer(TinyActor(rows), TinyTokenizer(), nn.Linear(1, 1), TinyTokenizer(),
                   TrainerConfig(**options))
    t.optimizer = torch.optim.SGD(t.actor.parameters(), lr=.05)
    t._encode_prompts = lambda prompts: (
        {"input_ids": torch.ones(len(prompts), 1, dtype=torch.long),
         "attention_mask": torch.ones(len(prompts), 1, dtype=torch.long)}, list(prompts))
    t._reward_batch = lambda *args: (torch.full((len(args[3]),), 4.), None, None, None)
    return t


def read_rewards(path):
    return json.loads((path / "rollout-1-rewards.json").read_text())


def test_nonempty_short_answer_gets_soft_cost_and_can_end_before_eight(tmp_path):
    t = make_trainer(tmp_path, [[2, 0], [3] * 7 + [0]])
    metrics = t.train_rollout(["p"])
    assert t.actor.generation_calls[0]["min_new_tokens"] == 0
    rows = read_rewards(tmp_path)
    assert [r["short_penalty"] for r in rows] == [.75, 0.]
    assert [r["reward"] for r in rows] == [3.25, 4.]
    assert [r["advantage"] for r in rows] == [-.375, .375]
    assert metrics["degenerate_responses"] == 0
    assert metrics["optimizer_steps"] == 1


def test_cap_cost_is_independent_of_finish_reason(tmp_path):
    t = make_trainer(tmp_path, [[2] * 31 + [0], [3] * 32])
    metrics = t.train_rollout(["p"])
    rows = read_rewards(tmp_path)
    assert [r["long_penalty"] for r in rows] == [4., 4.]
    assert [r["reward"] for r in rows] == [0., 0.]
    assert metrics["truncated_responses"] == 1
    assert metrics["degenerate_responses"] == 0


def test_empty_answer_is_floored_but_nonempty_short_answer_is_not(tmp_path):
    t = make_trainer(tmp_path, [[0], [2, 0]])
    metrics = t.train_rollout(["p"])
    rows = read_rewards(tmp_path)
    assert metrics["degenerate_responses"] == 1
    assert rows[0]["reward"] < rows[1]["reward"]
    assert rows[1]["reward"] == 3.25
    assert rows[0]["empty"] and not rows[1]["empty"]


def test_all_bad_group_retries_once_then_performs_no_optimizer_update(tmp_path):
    t = make_trainer(tmp_path, [[0], [0], [0], [0]])
    before = t.actor.logits.detach().clone()
    metrics = t.train_rollout(["p"])
    assert metrics["resampled_groups"] == 1
    assert metrics["skipped_groups"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["generated_response_tokens"] == 4
    assert t.rollout_index == 1
    assert torch.equal(before, t.actor.logits)
    assert json.loads((tmp_path / "rollout-1-tokens.json").read_text()) == []


def test_calibration_uses_initial_completed_groups_and_stays_fixed(tmp_path):
    t = make_trainer(tmp_path, [[2, 0], [3, 0], [2, 0], [3, 0]],
                     length_reward_sigma0=None, length_calibration_prompts=2)
    t._reward_batch = lambda *args: (
        torch.tensor([0., 2.]) if args[-1][0] == "first" else torch.tensor([0., 6.]),
        None, None, None)
    before = t.actor.logits.detach().clone()
    result = t.prepare_length_reward(["first", "second"])
    assert result["sigma0"] == 2.
    assert t.cfg.length_reward_sigma0 == 2.
    assert result["calibration_prompt_count"] == 2
    assert t.rollout_index == 0 and not t.optimizer.state
    assert torch.equal(before, t.actor.logits)
    assert t.prepare_length_reward([]) == result
    assert json.loads((tmp_path / "length_reward_calibration.json").read_text())["sigma0"] == 2.


def test_uncalibrated_direct_rollout_is_rejected_before_generation(tmp_path):
    t = make_trainer(tmp_path, length_reward_sigma0=None)
    with pytest.raises(RuntimeError, match="calibrat|prepare_length_reward"):
        t.train_rollout(["p"])
    assert not t.actor.generation_calls


@pytest.mark.parametrize("changes", [
    {"length_reward_sigma0": 0.}, {"length_reward_sigma0": float("nan")},
    {"short_response_threshold": 0}, {"long_response_threshold": 32},
    {"min_response_tokens": 8}, {"length_penalty_slope": .00506},
    {"short_penalty_strength": -1.}, {"advantage_std_floor_fraction": 0.},
    {"degenerate_penalty": -1.}, {"degenerate_penalty": 0.},
    {"degenerate_penalty": float("nan")},
])
def test_bad_soft_config_fails_before_loading_model(tmp_path, changes):
    with pytest.raises(ValueError):
        make_trainer(tmp_path, **changes)


def test_legacy_default_generation_minimum_is_preserved():
    assert TrainerConfig().resolved().min_response_tokens == 8
    assert TrainerConfig(length_reward_mode="soft").resolved().min_response_tokens == 0


def test_soft_vpo_updates_actor_and_preserves_frozen_initial_reference(tmp_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from vpo_rm.reward import LastTokenReward
    from test_trainer_regressions import fixed_rollout

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 4)
            with torch.no_grad():
                self.emb.weight.copy_(torch.arange(32).reshape(8, 4) / 32)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds, attention_mask, **kwargs):
            return types.SimpleNamespace(
                last_hidden_state=(inputs_embeds * attention_mask[..., None]).cumsum(1))

    torch.manual_seed(21)
    actor = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=32, attention_dropout=0., hidden_dropout=0.))
    head = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1.)
    reward = LastTokenReward(Backbone(), head)
    cfg = TrainerConfig(actor_device="cpu", reward_device="cpu", allocated_gpu_count=0,
        output_dir=str(tmp_path), lora=False, gradient_checkpointing=False, group_size=2,
        max_response_tokens=16, method="vpo_rm", beta=.1, learning_rate=.01,
        length_reward_mode="soft", length_reward_sigma0=2., long_response_threshold=12)
    t = VPOTrainer(actor, TinyTokenizer(), reward, TinyTokenizer(), cfg)
    fixed_rollout(t)
    before = actor.get_output_embeddings().weight.detach().clone()
    reference = t.reference_actor.get_output_embeddings().weight.detach().clone()
    metrics = t.train_rollout(["p"])
    assert metrics["optimizer_steps"] == 1
    assert metrics["short_penalty_mean"] == .625
    assert not torch.equal(before, actor.get_output_embeddings().weight)
    assert torch.equal(reference, t.reference_actor.get_output_embeddings().weight)
    assert all(p.grad is None for p in reward.parameters())
    assert torch.isfinite(torch.tensor(metrics["loss"]))


@pytest.mark.parametrize("source", ["random_direction", "random_band"])
def test_soft_vpo_random_credit_ablation_trains_without_rm_gradients(tmp_path, source, monkeypatch):
    """The ablation keeps the VPO loss, KL, band and budget but draws token weights at random."""
    from unittest.mock import patch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from vpo_rm.reward import LastTokenReward
    from test_trainer_regressions import fixed_rollout

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 4)
            with torch.no_grad():
                self.emb.weight.copy_(torch.arange(32).reshape(8, 4) / 32)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds, attention_mask, **kwargs):
            return types.SimpleNamespace(
                last_hidden_state=(inputs_embeds * attention_mask[..., None]).cumsum(1))

    torch.manual_seed(21)
    actor = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=32, attention_dropout=0., hidden_dropout=0.))
    head = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1.)
    reward = LastTokenReward(Backbone(), head)
    cfg = TrainerConfig(actor_device="cpu", reward_device="cpu", allocated_gpu_count=0,
        output_dir=str(tmp_path), lora=False, gradient_checkpointing=False, group_size=2,
        max_response_tokens=16, method="vpo_rm", credit_source=source, credit_lambda=4.,
        freeze_stop_tokens=True, beta=.1, learning_rate=.01,
        length_reward_mode="soft", length_reward_sigma0=2., long_response_threshold=12)
    t = VPOTrainer(actor, TinyTokenizer(), reward, TinyTokenizer(), cfg)
    fixed_rollout(t)
    before = actor.get_output_embeddings().weight.detach().clone()
    reference = t.reference_actor.get_output_embeddings().weight.detach().clone()
    with patch("vpo_rm.trainer.response_reward_gradients",
               side_effect=AssertionError("random credit must not backpropagate through the RM")):
        metrics = t.train_rollout(["p"])
    assert metrics["optimizer_steps"] == 1 and torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, actor.get_output_embeddings().weight)
    assert torch.equal(reference, t.reference_actor.get_output_embeddings().weight)
    assert all(p.grad is None for p in reward.parameters())
    assert 0 < metrics["credit_ess_ratio"] <= 1 and metrics["credit_w_max"] <= 4.
    assert metrics["credit_w_mean"] == pytest.approx(1., abs=1e-5)
    assert "phase_reward_model_forward_sec" in metrics and "phase_reward_model_gradient_sec" not in metrics
    dump = torch.load(tmp_path / "rollout-1-credit.pt")
    assert set(dump) == {"w", "d", "tau"} and dump["w"].shape == dump["d"].shape
    manifest = json.loads((tmp_path / "rollout-1-rewards.json").read_text())
    assert len(manifest) == 2


@pytest.mark.parametrize("source", ["shuffle", "norm_product"])
@pytest.mark.parametrize("credit_micro", [0, 1])
def test_gradient_credit_ablation_runs_real_training(tmp_path, source, credit_micro, monkeypatch):
    """The ablation keeps the VPO loss, KL, band and budget but draws token weights at random."""
    from unittest.mock import patch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from vpo_rm.reward import LastTokenReward
    from test_trainer_regressions import fixed_rollout

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 4)
            with torch.no_grad():
                self.emb.weight.copy_(torch.arange(32).reshape(8, 4) / 32)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds, attention_mask, **kwargs):
            return types.SimpleNamespace(
                last_hidden_state=(inputs_embeds * attention_mask[..., None]).cumsum(1))

    torch.manual_seed(21)
    actor = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=8, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        max_position_embeddings=32, attention_dropout=0., hidden_dropout=0.))
    head = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1.)
    reward = LastTokenReward(Backbone(), head)
    cfg = TrainerConfig(actor_device="cpu", reward_device="cpu", allocated_gpu_count=0,
        output_dir=str(tmp_path), lora=False, gradient_checkpointing=False, group_size=2,
        max_response_tokens=16, method="vpo_rm", credit_source=source, credit_lambda=4.,
        freeze_stop_tokens=True, beta=.1, learning_rate=.01, credit_microbatch_responses=credit_micro,
        length_reward_mode="soft", length_reward_sigma0=2., long_response_threshold=12)
    t = VPOTrainer(actor, TinyTokenizer(), reward, TinyTokenizer(), cfg)
    fixed_rollout(t)
    before = actor.get_output_embeddings().weight.detach().clone()
    reference = t.reference_actor.get_output_embeddings().weight.detach().clone()
    from vpo_rm.trainer import response_reward_gradients
    with patch("vpo_rm.trainer.response_reward_gradients", wraps=response_reward_gradients) as gradient:
        metrics = t.train_rollout(["p"])
        assert gradient.call_count > 0
    assert metrics["optimizer_steps"] == 1 and torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, actor.get_output_embeddings().weight)
    assert torch.equal(reference, t.reference_actor.get_output_embeddings().weight)
    assert all(p.grad is None for p in reward.parameters())
    assert 0 < metrics["credit_ess_ratio"] <= 1 and metrics["credit_w_max"] <= 4.
    assert metrics["credit_w_mean"] == pytest.approx(1., abs=1e-5)
    assert "phase_reward_model_gradient_sec" in metrics and "phase_reward_model_forward_sec" not in metrics
    dump = torch.load(tmp_path / "rollout-1-credit.pt")
    assert set(dump) == {"w", "d", "tau"} and dump["w"].shape == dump["d"].shape
    manifest = json.loads((tmp_path / "rollout-1-rewards.json").read_text())
    assert len(manifest) == 2
