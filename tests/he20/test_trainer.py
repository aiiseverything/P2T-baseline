"""Config contract, the one-step guard, and one full masked step on CPU.

The end-to-end test here is the one that matters.  The pure-function tests in
``test_mask.py`` and ``test_loss.py`` cannot see the wiring: whether the mask is
built over the whole optimizer minibatch or over a single response, whether it
reaches the loss at all, or whether the diagnostics report what the mask did.  All
of those only surface after generation has been paid for, so they are checked here
against synthetic rows with the two tokenizer seams cut.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import time

import pytest
import torch
from torch import nn

from he20.autopush import AutoPusher
from he20.reward import HE20_PROTOCOL
from he20.trainer import HE20Trainer, TrainerConfig, load_config

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------- config
def test_config_rejects_the_sibling_arms_keys(tmp_path):
    """Each arm rejects the others' knobs rather than ignoring them silently."""
    for payload in ({"omega": 0.6}, {"alpha": 0.1}, {"null_token_id": 0}, {"red_alpha": 1.0}):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(path)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5, float("nan"), "0.2", True])
def test_the_mask_ratio_must_be_a_fraction_or_none(ratio):
    with pytest.raises(ValueError, match="entropy_top_ratio"):
        TrainerConfig(entropy_top_ratio=ratio).resolved()
    TrainerConfig(entropy_top_ratio=None).resolved()          # None means "no mask"
    TrainerConfig(entropy_top_ratio=1.0).resolved()           # and so does one


def test_the_mask_rule_must_be_one_of_the_declared_rules():
    with pytest.raises(ValueError, match="entropy_top_rule"):
        TrainerConfig(entropy_top_rule="quantile").resolved()


def test_the_shipped_configs_are_valid_and_pin_the_shared_protocol():
    """Every shared training parameter must equal the sibling arm's.

    This arm exists to measure the mask, so a drifted optimiser or length window
    would change the comparison in a second way at the same time.
    """
    formal = load_config(ROOT / "configs" / "he20250.json").resolved()
    sibling = json.loads((ROOT / "configs" / "formal250.json").read_text())
    smoke = load_config(ROOT / "configs" / "he20-smoke2.json").resolved()
    assert formal.entropy_top_ratio == 0.2, "the paper's main setting"
    assert formal.entropy_top_rule in ("threshold", "topk")
    for shared in ("group_size", "prompts_per_rollout", "optimizer_minibatch_responses",
                   "microbatch_responses", "learning_rate", "weight_decay", "max_grad_norm",
                   "clip_eps", "beta", "max_prompt_tokens", "max_response_tokens",
                   "temperature", "top_p", "top_k", "min_response_tokens", "sigma0",
                   "length_threshold_short", "length_threshold_long",
                   "short_penalty_strength", "long_penalty_strength",
                   "advantage_std_floor_fraction", "checkpoint_interval", "keep_checkpoints",
                   "lora_r", "lora_alpha"):
        assert getattr(formal, shared) == sibling[shared], f"{shared} drifted from the sibling arm"
    assert formal.init_adapter == sibling["init_adapter"]
    # The smoke differs only in identity, length and autopush, as the siblings' does.
    for name in ("entropy_top_ratio", "entropy_top_rule", "model_name", "init_adapter",
                 "sigma0", "group_size", "actor_device", "reward_device", "vllm_gpus"):
        assert getattr(smoke, name) == getattr(formal, name), f"{name} differs in the smoke"
    assert smoke.rollout_iterations == 2 and smoke.push_every == 0


def test_the_mask_configures_a_single_optimizer_step():
    """The guard's precondition, asserted rather than assumed: batch == minibatch."""
    formal = load_config(ROOT / "configs" / "he20250.json").resolved()
    batch = formal.group_size * formal.prompts_per_rollout
    assert batch == formal.optimizer_minibatch_responses, (
        "the mask is built once per rollout from the pre-step entropy, so the rollout "
        "must produce exactly one optimizer step")


# ------------------------------------------------------- the synthetic step
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
        from types import SimpleNamespace
        logits = self.head(self.embed(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


class _TinyTokenizer:
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + int(i) % 26) for i in ids)


def _stub_trainer(tmp_path, *, group_size=2, prompts=1, vocab=19, dim=8,
                  response_length=4, entropy_top_ratio=0.2, entropy_top_rule="threshold"):
    """A trainer wired for one CPU training step, with the tokenizer seams cut.

    ``build_rm_batch`` and ``score_responses`` are the only pieces that need a real
    chat template and a real reward model; everything downstream of them -- rollout
    selection, the group advantage, the mask, the loss, the optimizer loop and the
    metrics -- runs for real against synthetic rows.
    """
    trainer = object.__new__(HE20Trainer)
    trainer.cfg = TrainerConfig(
        output_dir=str(tmp_path / "run"), report_dir=str(tmp_path / "report"),
        actor_device="cpu", reward_device="cpu", group_size=group_size,
        prompts_per_rollout=prompts, rollout_iterations=1, max_response_tokens=6,
        max_prompt_tokens=4, length_threshold_short=2, length_threshold_long=5,
        microbatch_responses=1, optimizer_minibatch_responses=group_size * prompts,
        sigma0=1.0, checkpoint_interval=0, push_every=0, lora=False,
        gradient_checkpointing=False, calibration_prompts=1,
        entropy_top_ratio=entropy_top_ratio, entropy_top_rule=entropy_top_rule).resolved()
    trainer.actor = _TinyActor(vocab, dim)
    trainer.reward = nn.Identity()
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
    trainer.pusher = AutoPusher(enabled=False, every=0, remote="origin", branch="he20-baseline",
                                report_dir=trainer.report_dir, repo_root=tmp_path)

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
                 "he20_unmapped_share_mean": 0.0}
        return rows, mapped.clone(), fixed, stats

    def fake_score_responses(reward_model, reward_tokenizer, rows_arg, mapped_arg, responses,
                             response_mask, *, device, microbatch=1):
        # This arm's RM read returns the pooled sequence score and nothing else.
        return torch.randn(responses.shape[0])

    trainer.rollout = fake_rollout
    return trainer, fake_build_rm_batch, fake_score_responses


def _wire(monkeypatch, trainer, fake_batch, fake_score):
    monkeypatch.setattr("he20.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("he20.trainer.score_responses", fake_score)
    return trainer


def test_one_full_masked_step_reports_the_mask_it_applied(tmp_path, monkeypatch):
    trainer, batch, score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, batch, score)
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    metrics = trainer.train_rollout(["p"])

    assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["grad_norm"])
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters())), \
        "the optimizer must have moved"
    assert metrics["he20_protocol"] == HE20_PROTOCOL
    assert metrics["he20_mask_population"] == "optimizer_minibatch"
    assert metrics["entropy_top_ratio"] == 0.2
    assert metrics["entropy_top_rule"] == "threshold"

    # The mask really restricted the loss, and really selected high-entropy tokens.
    tokens = trainer.cfg.group_size * trainer.cfg.max_response_tokens
    kept = metrics["entropy_top_kept_fraction"] * tokens
    assert abs(kept - 0.2 * tokens) <= 1, f"kept {kept:.1f} of {tokens} tokens"
    assert 0 < metrics["entropy_top_kept_fraction"] < 1
    assert metrics["entropy_top_mean_kept_entropy"] > metrics["entropy_top_mean_all_entropy"]
    assert math.isfinite(metrics["entropy_top_threshold"])

    # There is no token-level credit in this arm, and the row says so.
    assert metrics["credit_ess_ratio"] == pytest.approx(1.0)
    assert (trainer.output_dir / "rollout-1-credit.pt").is_file()
    assert (trainer.report_dir / "metrics.jsonl").is_file()


@pytest.mark.parametrize("ratio", [None, 1.0])
def test_both_ways_of_disabling_the_mask_leave_the_loss_untouched(ratio, tmp_path, monkeypatch):
    """`None` and `1.0` are the same thing everywhere: the unmodified algorithm."""
    trainer, batch, score = _stub_trainer(tmp_path, entropy_top_ratio=ratio)
    _wire(monkeypatch, trainer, batch, score)
    metrics = trainer.train_rollout(["p"])
    assert metrics["he20_mask_ratio_is_effective"] is False
    assert metrics["entropy_top_kept_fraction"] == pytest.approx(1.0)
    assert metrics["entropy_top_threshold"] is None
    assert metrics["entropy_top_mean_kept_entropy"] == pytest.approx(
        metrics["entropy_top_mean_all_entropy"])
    assert math.isfinite(metrics["loss"])


def test_the_one_step_guard_refuses_a_multi_step_rollout(tmp_path, monkeypatch):
    """With more than one step the mask would need pi_theta's entropy per step.

    Ranking the pre-step buffer there would be a silent approximation of Eq. (6),
    so the trainer is required to refuse rather than approximate.
    """
    trainer, batch, score = _stub_trainer(tmp_path, group_size=4, prompts=2)
    trainer.cfg = TrainerConfig(**{**trainer.cfg.__dict__,
                                   "optimizer_minibatch_responses": 4}).resolved()
    _wire(monkeypatch, trainer, batch, score)
    # Two prompts of four responses each: batch 8 against a minibatch of 4, so the
    # rollout would produce two optimizer steps.
    with pytest.raises(ValueError, match="optimizer step"):
        trainer.train_rollout(["p", "q"])


def test_the_guard_is_silent_when_the_mask_is_off(tmp_path, monkeypatch):
    """A multi-step rollout is legitimate without a mask; nothing to approximate."""
    trainer, batch, score = _stub_trainer(tmp_path, group_size=4, prompts=2,
                                          entropy_top_ratio=None)
    trainer.cfg = TrainerConfig(**{**trainer.cfg.__dict__,
                                   "optimizer_minibatch_responses": 4}).resolved()
    _wire(monkeypatch, trainer, batch, score)
    metrics = trainer.train_rollout(["p", "q"])
    assert metrics["optimizer_steps"] == 2
    assert math.isfinite(metrics["loss"])


def _checker():
    """The health checker, loaded as the script it is rather than as a module."""
    path = ROOT / "he20" / "scripts" / "check_he20_health.py"
    spec = importlib.util.spec_from_file_location("check_he20_health", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_metrics_row_satisfies_every_key_the_health_checker_reads(tmp_path, monkeypatch):
    """The trainer and the checker must agree on the vocabulary.

    The checker reads a row's keys by name, so a metric renamed or dropped in the
    trainer makes the checker report a missing-metric problem instead of measuring
    anything -- a false alarm on a healthy run, or worse, a missing signal.
    """
    trainer, batch, score = _stub_trainer(tmp_path)
    _wire(monkeypatch, trainer, batch, score)
    metrics = trainer.train_rollout(["p"])
    checker = _checker()
    for key in (*checker.SHARED, *checker.FINITE):
        assert key in metrics, f"the checker reads {key}, which the trainer no longer emits"
    assert metrics["he20_protocol"] == checker.HE20_PROTOCOL


def test_the_shared_spread_key_reports_the_floored_scale_its_siblings_report(tmp_path, monkeypatch):
    """``group_sigma_*`` must mean the same quantity in every arm.

    VPO and P2T standardise by ``max(std, advantage_std_floor_fraction * sigma0)``
    and report *that* value as the group spread (``vpo_rm/trainer.py:892``); the
    floor binds on 233 of p2t250's 250 steps, so reporting the raw spread instead
    would put a different number under a shared key.  This arm therefore reports the
    floored scale and puts the unfloored one beside it under its own name, which is
    what the health check tests for a degenerate group.  Constant rewards make the
    two distinguishable: the raw spread is exactly zero and the reported one is
    exactly the floor.
    """
    trainer, batch, _ = _stub_trainer(tmp_path)

    def constant_rewards(reward_model, reward_tokenizer, rows_arg, mapped_arg, responses,
                         response_mask, *, device, microbatch=1):
        return torch.ones(responses.shape[0])

    _wire(monkeypatch, trainer, batch, constant_rewards)
    metrics = trainer.train_rollout(["p"])
    floor = trainer.cfg.advantage_std_floor_fraction * trainer.sigma0
    torch.testing.assert_close(torch.tensor(metrics["raw_group_sigma_mean"]), torch.tensor(0.0))
    torch.testing.assert_close(torch.tensor(metrics["group_sigma_mean"]), torch.tensor(floor))
    assert metrics["group_sigma_min"] >= floor
