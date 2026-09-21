"""Config contract and one full training step, on CPU with no real model.

The pure-function tests elsewhere cannot see wiring mistakes -- the rollout
contract, the device moves, the credit ordering forced by Eq. (8), the diagnostic
broadcasts.  Those only surface after generation has already been paid for, so
they are exercised here against synthetic reward-model rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch
from torch import nn

from red.autopush import AutoPusher
from red.reward import RED_BETA_C_DEFAULT
from red.rm import LastTokenReward
from red.trainer import REDTrainer, TrainerConfig, load_config


# ------------------------------------------------------------------ config
def test_config_has_no_vpo_or_p2t_knobs():
    """The arms must not silently inherit each other's allocator or constants."""
    fields = set(TrainerConfig.__dataclass_fields__)
    for absent in ("tau", "credit_lambda", "freeze_stop_tokens", "freeze_structural",
                   "credit_source", "method"):
        assert absent not in fields, f"{absent} belongs to VPO-RM, not to RED"
    for absent in ("omega", "alpha", "null_token_id"):
        assert absent not in fields, f"{absent} belongs to P2T, not to RED"
    assert TrainerConfig().resolved().beta_c == RED_BETA_C_DEFAULT


@pytest.mark.parametrize("override,match", [
    (dict(microbatch_responses=2), "microbatch must be one"),
    (dict(top_p=0.9), "top_p=1 and top_k=0"),
    (dict(lora_dropout=0.1), "zero dropout"),
    (dict(clip_eps=1.5), "clip_eps"),
    (dict(sigma0=-1.0), "sigma0"),
    (dict(vllm_gpus=["2"]), "tensor_parallel_size"),
    (dict(length_threshold_long=4096), "short <= long"),
    (dict(beta_c=-0.1), "beta_c"),
    (dict(beta_c=1.5), "beta_c"),
    (dict(beta_c=float("nan")), "beta_c"),
    (dict(beta=-0.1), "beta"),
])
def test_config_rejects_invalid_settings(override, match):
    with pytest.raises(ValueError, match=match):
        TrainerConfig(**override).resolved()


def test_the_smoke_config_is_a_two_step_copy_of_the_formal_one():
    """The smoke must differ only in identity, length and autopush.

    A smoke that quietly used a different model, adapter or sigma0 would validate
    a code path the formal run never takes.
    """
    root = Path(__file__).resolve().parents[2]
    smoke = load_config(root / "configs" / "red-smoke2.json")
    formal = load_config(root / "configs" / "red250.json")
    assert smoke.rollout_iterations == 2
    assert smoke.push_every == 0, "a two-step run has nothing worth publishing"
    assert smoke.output_dir != formal.output_dir
    assert smoke.report_dir != formal.report_dir
    for shared in ("model_name", "reward_model_name", "init_adapter", "sigma0",
                   "beta", "beta_c", "group_size", "prompts_per_rollout",
                   "learning_rate", "max_prompt_tokens", "max_response_tokens",
                   "actor_device", "reward_device", "vllm_gpus",
                   "vllm_tensor_parallel_size", "vllm_gpu_memory_utilization"):
        assert getattr(smoke, shared) == getattr(formal, shared), f"{shared} differs"


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model_name": "x", "tau": 1.0}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_config_rejects_a_p2t_key(tmp_path):
    """A P2T config must not load as a RED run."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model_name": "x", "omega": 0.6}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_shipped_config_is_valid_and_matches_the_sibling_arms():
    """Every training parameter must equal the sibling arm's, or the comparison is void.

    The sibling's config carries P2T-only keys that a RED run must reject, so it
    is read as raw JSON rather than through this package's loader.
    """
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "red250.json")
    sibling = json.loads((root / "configs" / "formal250.json").read_text())
    assert config.beta_c == 1.0
    assert config.group_size == 8
    # Autopush was approved with the sibling's remote and branch: commits land on
    # the local p2t-baseline and go to p2t-origin/main.  Pinned so a silent change
    # of destination is a test failure and not a surprise push.
    assert config.push_every == 5
    assert (config.push_remote, config.push_branch) == ("p2t-origin", "main")
    assert config.push_local_branch == "p2t-baseline"
    # The shared calibration and the common starting point are what make the two
    # arms comparable at all; letting this arm self-calibrate would break that.
    assert config.sigma0 == sibling["sigma0"]
    assert config.init_adapter == sibling["init_adapter"]
    for shared in ("group_size", "prompts_per_rollout", "rollout_iterations",
                   "learning_rate", "weight_decay", "max_grad_norm", "beta",
                   "max_prompt_tokens", "max_response_tokens", "temperature",
                   "top_p", "top_k", "min_response_tokens", "clip_eps",
                   "length_threshold_short", "length_threshold_long",
                   "short_penalty_strength", "long_penalty_strength",
                   "advantage_std_floor_fraction", "checkpoint_interval",
                   "keep_checkpoints", "lora_r", "lora_alpha"):
        assert getattr(config, shared) == sibling[shared], f"{shared} drifted from the sibling arm"


# --------------------------------------------------------------- tiny models
class _TinyBackbone(nn.Module):
    def __init__(self, vocab=19, dim=8):
        super().__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab, dim)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, use_cache=False, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        steps = torch.arange(1, inputs_embeds.shape[1] + 1,
                             device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        hidden = torch.cumsum(inputs_embeds, dim=1) / steps.view(1, -1, 1)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyActor(nn.Module):
    """Minimal causal LM: enough surface for response_logits and the optimizer."""

    def __init__(self, vocab=19, dim=8):
        super().__init__()
        torch.manual_seed(5)
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False,
                return_dict=True, logits_to_keep=None):
        logits = self.head(self.embed(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


class _TinyTokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + int(i) % 26) for i in ids)


def _stub_trainer(tmp_path, *, group_size=2, prompts=1, vocab=19, dim=8,
                  response_length=4):
    """A trainer wired for one CPU step, with the reward-model seam cut.

    ``build_rm_batch`` and ``score_prefixes`` are the only pieces that need a real
    chat template and a real reward model; everything downstream -- rollout
    selection, Eq. (6)-(8), RLOO's baseline, the optimizer loop and the metrics --
    runs for real against synthetic rows.
    """
    trainer = object.__new__(REDTrainer)
    trainer.cfg = TrainerConfig(
        output_dir=str(tmp_path / "run"), report_dir=str(tmp_path / "report"),
        actor_device="cpu", reward_device="cpu", group_size=group_size,
        prompts_per_rollout=prompts, rollout_iterations=1, max_response_tokens=6,
        max_prompt_tokens=4, length_threshold_short=2, length_threshold_long=5,
        microbatch_responses=1, optimizer_minibatch_responses=group_size * prompts,
        sigma0=1.0, checkpoint_interval=0, push_every=0, lora=False,
        gradient_checkpointing=False, calibration_prompts=1).resolved()
    trainer.actor = _TinyActor(vocab, dim)
    head = nn.Linear(dim, 1)
    torch.manual_seed(11)
    nn.init.normal_(head.weight)
    nn.init.normal_(head.bias)
    trainer.reward = LastTokenReward(_TinyBackbone(vocab, dim), head)
    trainer.actor_tokenizer = _TinyTokenizer()
    trainer.reward_tokenizer = _TinyTokenizer()
    trainer.actor_device = trainer.reward_device = torch.device("cpu")
    trainer.output_dir = tmp_path / "run"
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.report_dir = tmp_path / "report"
    trainer.report_dir.mkdir(parents=True, exist_ok=True)
    trainer.metrics_path = trainer.report_dir / "metrics.jsonl"
    trainer.adapter_root = trainer.output_dir / "vllm-adapters"
    trainer.adapter_root.mkdir(parents=True, exist_ok=True)
    trainer.stop_token_ids = (2,)
    trainer.output_mask = torch.ones(vocab, dtype=torch.bool)
    trainer.optimizer = torch.optim.AdamW(trainer.actor.parameters(), lr=1e-3)
    trainer.rollout_index = 0
    trainer.total_tokens = 0
    trainer.adapter_id = 0
    trainer.sigma0 = 1.0
    trainer.filtered_prompt_count = 0
    trainer.init_adapter_sha256 = ""
    trainer._adapter_identity = {}
    trainer.started = time.monotonic()
    trainer.pusher = AutoPusher(enabled=False, every=0, remote="origin",
                                branch="p2t-baseline", report_dir=trainer.report_dir,
                                repo_root=tmp_path)
    trainer.generation = None

    def fake_rollout(batch_prompts):
        count = len(batch_prompts) * trainer.cfg.group_size
        width = trainer.cfg.max_response_tokens
        responses = torch.randint(3, vocab, (count, width))
        responses[:, response_length - 1] = 2
        rmask = torch.zeros((count, width), dtype=torch.long)
        rmask[:, :response_length] = 1
        prompt_width = trainer.cfg.max_prompt_tokens
        input_ids = torch.randint(1, vocab, (count, prompt_width + width))
        full_mask = torch.cat([torch.ones((count, prompt_width), dtype=torch.long), rmask], dim=1)
        positions = torch.arange(prompt_width, prompt_width + width).expand(count, -1)
        logprobs = torch.full((count, width), -1.0)
        rendered = [f"p{i // trainer.cfg.group_size}" for i in range(count)]
        rollout = (input_ids, full_mask, positions, responses, rmask, rendered,
                   ["stop"] * count, logprobs)
        return rollout, {"mean_response_tokens": float(response_length)}

    rows = [[1, 2, 3, 4] for _ in range(prompts * group_size)]
    mapped = torch.full((prompts * group_size, trainer.cfg.max_response_tokens), -1,
                        dtype=torch.long)
    mapped[:, :3] = torch.tensor([1, 2, 3])

    def fake_build_rm_batch(*args, **kwargs):
        fixed = torch.zeros((prompts * group_size, trainer.cfg.max_response_tokens),
                            dtype=torch.bool)
        stats = {"rm_max_input_tokens": 4, "rm_mapped_tokens": 3, "rm_unmapped_tokens": 0,
                 "rm_unmapped_content_tokens": 0, "rm_unmapped_content_fraction": 0.0,
                 "red_unmapped_share_mean": 0.0}
        return rows, mapped.clone(), fixed, stats

    def fake_score_prefixes(reward_model, reward_tokenizer, rows_arg, mapped_arg, responses,
                            response_mask, *, device, microbatch=1):
        count, length = responses.shape
        rewards = torch.randn(count)
        token_rewards = torch.randn(count, length)
        return rewards, token_rewards

    trainer.rollout = fake_rollout
    return trainer, fake_build_rm_batch, fake_score_prefixes


def _wire(monkeypatch, trainer, fake_batch, fake_score):
    monkeypatch.setattr("red.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("red.trainer.score_prefixes", fake_score)
    return trainer


def test_train_rollout_runs_one_full_step_and_reports_diagnostics(tmp_path, monkeypatch):
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, fake_batch, fake_score)
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]

    metrics = trainer.train_rollout(["p"])

    assert metrics["rollout"] == 1 and metrics["optimizer_steps"] == 1
    assert metrics["reward_count"] == trainer.cfg.group_size
    assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["grad_norm"])
    assert math.isfinite(metrics["response_entropy"])
    for key in ("red_beta_c", "red_token_reward_mean", "red_token_reward_abs_mean",
                "red_positive_mass_fraction", "red_share_max_mean",
                "red_flat_response_fraction", "red_varying_over_baseline",
                "red_advantage_flip_fraction", "credit_ess_ratio",
                "rloo_baseline_mean", "kl_to_init", "rollout_is_mean"):
        assert key in metrics, f"{key} missing"
        assert math.isfinite(metrics[key]), f"{key} is not finite"
    for bounded in ("red_positive_mass_fraction", "red_flat_response_fraction",
                    "red_advantage_flip_fraction", "credit_ess_ratio"):
        assert 0.0 <= metrics[bounded] <= 1.0, f"{bounded} out of range"
    assert metrics["red_protocol"] == "prefix_difference_eq6"
    assert any(not torch.equal(old, new) for old, new in
               zip(before, trainer.actor.parameters())), "the optimizer must have moved"
    assert (trainer.output_dir / "rollout-1-credit.pt").is_file()
    assert (trainer.report_dir / "metrics.jsonl").is_file()


def test_credit_dump_carries_the_red_fields(tmp_path, monkeypatch):
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, fake_batch, fake_score)
    trainer.train_rollout(["p"])
    payload = torch.load(trainer.output_dir / "rollout-1-credit.pt", weights_only=False)
    for key in ("w", "d", "advantage", "raw_reward", "token_reward", "baseline",
                "protocol", "advantage_rule", "beta_c"):
        assert key in payload, f"{key} missing from the credit dump"
    assert payload["protocol"] == "prefix_difference_eq6"
    assert payload["advantage_rule"] == "loo_scalar_baseline_r3"
    # The sibling arm's attribution field has no RED counterpart and must not be
    # filled with zeros, which would read as real attribution.
    assert "i" not in payload


def test_every_phase_timing_is_reported_and_nonnegative(tmp_path, monkeypatch):
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, fake_batch, fake_score)
    metrics = trainer.train_rollout(["p"])
    for phase in ("generation_sec", "reward_model_prefix_sec", "credit_sec",
                  "actor_old_logp_sec", "ref_logp_sec", "actor_update_sec",
                  "logging_sec"):
        key = f"phase_{phase}"
        assert key in metrics, f"{key} missing from the step metrics"
        assert metrics[key] >= 0, f"{key} is negative: {metrics[key]}"
    total = sum(metrics[f"phase_{p}"] for p in
                ("generation_sec", "reward_model_prefix_sec", "credit_sec",
                 "actor_old_logp_sec", "ref_logp_sec", "actor_update_sec"))
    assert total <= metrics["elapsed_sec"] + 1e-3


def test_the_kl_actually_reaches_the_reward(tmp_path, monkeypatch):
    """Eq. (8) subtracts the KL inside the reward, so the ordering is load-bearing.

    If credit were still assembled before the reference log-probs, Eq. (8) would
    have nothing to subtract and RED would silently degrade into a pure
    redistribution with no constraint.  Spying on the call site proves the KL is
    real and that it is weighted by the configured beta.
    """
    import red.trainer as trainer_module
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, fake_batch, fake_score)

    # The stub has no LoRA, so the reference forward returns the same numbers as
    # the policy and the KL would be identically zero.  Offset the reference pass
    # so the assertion below tests the wiring and not the fixture.
    original_logp = trainer_module.rollout_logp_microbatch

    def offset_reference(actor, input_ids, full_mask, positions, responses, rmask,
                         output_mask, **kwargs):
        out = original_logp(actor, input_ids, full_mask, positions, responses, rmask,
                            output_mask, **kwargs)
        return out + 0.25 if kwargs.get("adapter") else out

    monkeypatch.setattr(trainer_module, "rollout_logp_microbatch", offset_reference)

    captured = {}
    original = trainer_module.red_final_reward

    def spy(token_rewards, sequence_rewards, kl_reward, mask, *, beta_c, beta):
        captured["kl_abs_sum"] = float(kl_reward.abs().sum())
        captured["beta"] = beta
        captured["beta_c"] = beta_c
        return original(token_rewards, sequence_rewards, kl_reward, mask,
                        beta_c=beta_c, beta=beta)

    monkeypatch.setattr(trainer_module, "red_final_reward", spy)
    trainer.train_rollout(["p"])

    assert captured["kl_abs_sum"] > 0, "the KL never reached Eq. (8)"
    assert captured["beta"] == trainer.cfg.beta
    assert captured["beta_c"] == trainer.cfg.beta_c


def test_train_rollout_diagnostics_survive_a_rectangular_batch(tmp_path, monkeypatch):
    """Wide-but-ragged batches must not broadcast into a [B, B] outer product."""
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path, group_size=4, prompts=2)
    _wire(monkeypatch, trainer, fake_batch, fake_score)
    metrics = trainer.train_rollout(["p", "q"])
    assert metrics["reward_count"] == 8
    assert math.isfinite(metrics["red_share_max_mean"])
    assert 0.0 <= metrics["red_flat_response_fraction"] <= 1.0
