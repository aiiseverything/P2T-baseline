"""Config contract, the full credit chain on a tiny model, and autopush guardrails."""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from p2t.attribution import null_token_attribution
from p2t.autopush import AutoPusher
from p2t.loss import grpo_policy_loss
from p2t.reward import P2T_ALPHA_SHORT_COT, P2T_OMEGA, group_advantages, p2t_credit
from p2t.rm import LastTokenReward, reward_input_gradients
from p2t.trainer import P2TTrainer, TrainerConfig, load_config


# ------------------------------------------------------------------ config
def test_config_has_no_vpo_allocator_knobs():
    """The paper has no temperature, weight band or token freezing.

    Those knobs live in the parent project's config.  Their absence here is the
    guarantee that a P2T run cannot silently inherit the VPO allocator.
    """
    fields = set(TrainerConfig.__dataclass_fields__)
    for absent in ("tau", "credit_lambda", "freeze_stop_tokens", "freeze_structural",
                   "credit_source", "method"):
        assert absent not in fields, f"{absent} belongs to VPO-RM, not to P2T"
    assert TrainerConfig().resolved().omega == P2T_OMEGA
    assert TrainerConfig().resolved().alpha == P2T_ALPHA_SHORT_COT


@pytest.mark.parametrize("override,match", [
    (dict(microbatch_responses=2), "microbatch must be one"),
    (dict(optimizer_minibatch_responses=6, microbatch_responses=4), "microbatch"),
    (dict(top_p=0.9), "top_p=1 and top_k=0"),
    (dict(top_k=10), "top_p=1 and top_k=0"),
    (dict(omega=-1.0), "omega"),
    (dict(alpha=float("nan")), "alpha"),
    (dict(lora_dropout=0.1), "zero dropout"),
    (dict(clip_eps=1.5), "clip_eps"),
    (dict(sigma0=-1.0), "sigma0"),
    (dict(vllm_gpus=["2"]), "tensor_parallel_size"),
    (dict(length_threshold_long=4096), "short <= long"),
])
def test_config_rejects_invalid_settings(override, match):
    with pytest.raises(ValueError, match=match):
        TrainerConfig(**override).resolved()


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model_name": "x", "tau": 1.0}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_shipped_configs_are_valid_and_self_consistent():
    root = Path(__file__).resolve().parents[2]
    for name in ("smoke10", "formal250", "p2t-pilot"):
        payload = json.loads((root / "configs" / f"{name}.json").read_text())
        config = load_config(root / "configs" / f"{name}.json")
        assert config.push_remote == "p2t-origin" and config.push_branch == "main"
        if name == "smoke10":
            assert config.init_adapter == "", "the smoke must run on the raw base model"
            assert config.rollout_iterations == 10
        if name == "p2t-pilot":
            assert config.rollout_iterations == 2 and config.init_adapter


# ------------------------------------------------------- end-to-end credit
class _TinyReward(nn.Module):
    """A differentiable scalar reward over an embedding table."""

    def __init__(self, vocab=19, dim=8):
        super().__init__()
        torch.manual_seed(3)
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, 1)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        hidden = torch.tanh(inputs_embeds * attention_mask[..., None]).cumsum(1)
        pooled = hidden[torch.arange(hidden.shape[0]), attention_mask.sum(-1) - 1]
        return self.head(pooled)[:, 0]


def test_full_credit_chain_produces_finite_loss_and_actor_gradients():
    """Eq. (2) -> Eq. (3) -> Eq. (4) -> Eq. (5) -> GRPO loss, on a real graph."""
    torch.manual_seed(13)
    vocab, dim, batch, width = 19, 8, 4, 5
    reward_model = _TinyReward(vocab, dim)
    ids = torch.randint(0, vocab, (batch, width))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    attention = mask.long()

    rewards, grads = reward_input_gradients(reward_model, ids, attention)
    assert rewards.shape == (batch,) and grads.shape == (batch, width, dim)

    attribution = null_token_attribution(grads, ids, reward_model.get_input_embeddings().weight,
                                         0, mask)
    assert torch.isfinite(attribution[mask]).all()
    assert (attribution[~mask] == 0).all()

    group_ids = torch.tensor([0, 0, 1, 1])
    advantages, _ = group_advantages(rewards.detach(), group_ids)
    credit = p2t_credit(rewards.detach(), attribution, advantages, mask)

    actor_logits = torch.randn(batch, width, vocab, requires_grad=True)
    new_logp = actor_logits.log_softmax(-1).gather(
        -1, ids[..., None]).squeeze(-1).masked_fill(~mask, 0)
    old_logp = new_logp.detach().clone()
    loss = grpo_policy_loss(new_logp, old_logp, credit.advantage, mask, 0.2)
    assert torch.isfinite(loss)
    gradient = torch.autograd.grad(loss, actor_logits)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reward_model.parameters())


def test_credit_varies_across_tokens_of_one_response():
    """A method whose token term were constant would be GRPO with a shifted advantage."""
    torch.manual_seed(19)
    vocab, dim, batch, width = 19, 8, 2, 6
    reward_model = _TinyReward(vocab, dim)
    ids = torch.randint(0, vocab, (batch, width))
    mask = torch.ones(batch, width, dtype=torch.bool)
    rewards, grads = reward_input_gradients(reward_model, ids, mask.long())
    attribution = null_token_attribution(grads, ids, reward_model.get_input_embeddings().weight,
                                         0, mask)
    advantages = torch.tensor([1.0, -1.0])
    credit = p2t_credit(rewards.detach(), attribution, advantages, mask)
    for row in range(batch):
        assert credit.advantage[row][mask[row]].std() > 0
        assert not torch.allclose(credit.direction[row][mask[row]],
                                  credit.direction[row][mask[row]][0])


def test_token_advantage_mean_carries_the_papers_constant_shift():
    """Eq. (5) is additive, so every token also inherits alpha * R * (1 + omega/T).

    Unlike VPO-RM's multiplicative allocator, P2T does not preserve the sequence
    advantage's mean: Eq. (3)'s R baseline is added to every token and the shares
    sum to one, so the per-response mean grows by alpha*R*(1 + omega/T).  With a
    reward model scored in the tens and alpha = 0.1 that term is not small, which
    is why ``p2t_bonus_over_advantage`` is logged on every step.
    """
    mask = torch.ones(2, 4, dtype=torch.bool)
    attribution = torch.randn(2, 4)
    rewards = torch.tensor([5.0, -5.0])
    advantages = torch.tensor([2.0, -2.0])
    alpha, omega = 0.1, 0.6
    credit = p2t_credit(rewards, attribution, advantages, mask, omega=omega, alpha=alpha)
    expected = advantages + alpha * rewards * (1 + omega / mask.sum(-1))
    torch.testing.assert_close(credit.advantage.mean(-1), expected, atol=1e-6, rtol=1e-6)
    assert (credit.advantage[0] > 0).all() and (credit.advantage[1] < 0).all()
    # The token-varying part is what actually performs credit assignment.
    assert credit.advantage[0].std() > 0


def test_zero_omega_removes_every_token_varying_effect():
    """omega = 0 is the degenerate switch: A~ becomes a constant-shifted A^hat.

    The attribution term disappears, so the token advantages of one response are
    all equal -- exactly the failure mode that a flat softmax over I reproduces
    with omega at its paper value.
    """
    mask = torch.ones(1, 4, dtype=torch.bool)
    credit = p2t_credit(torch.tensor([3.0]), torch.randn(1, 4), torch.tensor([1.0]), mask,
                        omega=0.0, alpha=0.1)
    torch.testing.assert_close(credit.advantage, torch.full((1, 4), 1.0 + 0.3))
    torch.testing.assert_close(credit.direction, torch.full((1, 4), 0.3))
    assert credit.advantage.std() == 0


# ------------------------------------------------- end-to-end training step
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
                  response_length=4):
    """A trainer wired for one CPU training step, with the two tokenizer seams cut.

    ``build_rm_batch`` and ``score_responses`` are the only pieces that need a
    real chat template and a real reward model; everything downstream of them --
    rollout selection, Eq. (3)-(5), the optimizer loop and the metrics -- runs
    for real against synthetic rows.
    """
    trainer = object.__new__(P2TTrainer)
    trainer.cfg = TrainerConfig(
        output_dir=str(tmp_path / "run"), report_dir=str(tmp_path / "report"),
        actor_device="cpu", reward_device="cpu", group_size=group_size,
        prompts_per_rollout=prompts, rollout_iterations=1, max_response_tokens=6,
        max_prompt_tokens=4, length_threshold_short=2, length_threshold_long=5,
        microbatch_responses=1, optimizer_minibatch_responses=group_size * prompts,
        sigma0=1.0, checkpoint_interval=0, push_every=0, lora=False,
        gradient_checkpointing=False, calibration_prompts=1).resolved()
    trainer.actor = _TinyActor(vocab, dim)
    trainer.reward = _TinyReward(vocab, dim)
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
    trainer.null_token_id = 0
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
    trainer.pusher = AutoPusher(enabled=False, every=0, remote="origin", branch="p2t-baseline",
                                report_dir=trainer.report_dir, repo_root=tmp_path)
    trainer.generation = None
    trainer._fixed_weight_mask = None

    def fake_rollout(batch_prompts):
        """Match P2TTrainer.rollout's contract: a (rollout, summary) pair.

        Rows stop at ``response_length`` tokens on token 2, which ``stop_token_ids``
        declares, so the termination metadata is self-consistent.
        """
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
    # First three response positions mapped, the rest unmapped: the realistic mix.
    mapped = torch.full((prompts * group_size, trainer.cfg.max_response_tokens), -1,
                        dtype=torch.long)
    mapped[:, :3] = torch.tensor([1, 2, 3])

    def fake_build_rm_batch(*args, **kwargs):
        fixed = torch.zeros((prompts * group_size, trainer.cfg.max_response_tokens), dtype=torch.bool)
        stats = {"rm_max_input_tokens": 4, "rm_mapped_tokens": 3, "rm_unmapped_tokens": 0,
                 "rm_unmapped_content_tokens": 0, "rm_unmapped_content_fraction": 0.0,
                 "p2t_unmapped_share_mean": 0.0, "p2t_null_token_id": 0}
        return rows, mapped.clone(), fixed, stats

    def fake_score_responses(reward_model, reward_tokenizer, rows_arg, mapped_arg, responses,
                             response_mask, *, device, microbatch=1):
        count, length = responses.shape
        rewards = torch.randn(count)
        grads = torch.randn(count, length, dim)
        return rewards, grads

    trainer.rollout = fake_rollout
    return trainer, fake_build_rm_batch, fake_score_responses


def test_train_rollout_runs_one_full_step_and_reports_diagnostics(tmp_path, monkeypatch):
    """The whole loop, not just the credit maths.

    Wiring mistakes in this path -- the rollout contract, the device moves, the
    diagnostic broadcasts -- are invisible to tests that only touch the pure
    functions, and they only surface after generation has already been paid for.
    """
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]

    metrics = trainer.train_rollout(["p"])

    assert metrics["rollout"] == 1 and metrics["optimizer_steps"] == 1
    assert metrics["reward_count"] == trainer.cfg.group_size
    assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["grad_norm"])
    assert math.isfinite(metrics["response_entropy"])
    # every diagnostic must be a real number, not a crash or a NaN
    for key in ("p2t_attribution_mean", "p2t_varying_bonus_over_advantage",
                "p2t_sign_flip_fraction", "p2t_zero_attribution_share_mass",
                "credit_ess_ratio", "p2t_bonus_over_advantage", "rollout_is_mean",
                "initial_hf_logp_max_abs_error"):
        assert key in metrics, f"{key} missing"
        assert math.isfinite(metrics[key]), f"{key} is not finite"
    assert 0.0 <= metrics["p2t_zero_attribution_share_mass"] <= 1.0
    assert 0.0 <= metrics["p2t_sign_flip_fraction"] <= 1.0
    assert any(not torch.equal(old, new) for old, new in
               zip(before, trainer.actor.parameters())), "the optimizer must have moved"
    assert (trainer.output_dir / "rollout-1-credit.pt").is_file()
    assert (trainer.report_dir / "metrics.jsonl").is_file()


def test_every_phase_timing_is_reported_and_nonnegative(tmp_path, monkeypatch):
    """`phase(name, start)` times the work *before* the call, so ordering matters.

    The calls originally ran before the work they named, which shifted every label
    by one place: `generation_sec` read ~0 while generation's real cost was logged
    as `reward_model_gradient_sec`.  The run's measured time budget comes from these
    keys, so a missing or negative one has to fail here.
    """
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    metrics = trainer.train_rollout(["p"])
    for phase in ("generation_sec", "reward_model_gradient_sec", "credit_sec",
                  "actor_old_logp_sec", "ref_logp_sec", "actor_update_sec",
                  "logging_sec"):
        key = f"phase_{phase}"
        assert key in metrics, f"{key} missing from the step metrics"
        assert metrics[key] >= 0, f"{key} is negative: {metrics[key]}"
    # The phases partition the step, so they cannot exceed it by more than the
    # bookkeeping that runs after `elapsed_sec` is read.
    total = sum(metrics[f"phase_{p}"] for p in
                ("generation_sec", "reward_model_gradient_sec", "credit_sec",
                 "actor_old_logp_sec", "ref_logp_sec", "actor_update_sec"))
    assert total <= metrics["elapsed_sec"] + 1e-3


def test_train_rollout_diagnostics_survive_rectangular_batches(tmp_path, monkeypatch):
    """B != T is the case a broadcast bug hides in: [B,1]*[B] silently becomes [B,B]."""
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path, group_size=3, prompts=2)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    metrics = trainer.train_rollout(["p", "q"])
    assert metrics["reward_count"] == 6
    assert math.isfinite(metrics["p2t_varying_bonus_over_advantage"])
    assert math.isfinite(metrics["p2t_zero_attribution_share_mass"])


def test_sampler_disagreement_gate_fires_before_the_optimizer_step(tmp_path, monkeypatch):
    """A protocol mismatch must abort the step, not be reported after it.

    Gating inside the metrics block would let the first optimizer step apply a
    gradient computed under the very protocol the gate just called broken.
    """
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    steps = {"count": 0}
    real_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        steps["count"] += 1
        return real_step(*args, **kwargs)

    # Shift every cached old log-probability far outside the clipping band.
    real_logp = trainer.rollout

    def shifted_rollout(batch_prompts):
        rollout, summary = real_logp(batch_prompts)
        return rollout, summary

    trainer.optimizer.step = counting_step
    import p2t.policy as policy_module
    real_micro = policy_module.rollout_logp_microbatch

    def shifted_micro(*args, **kwargs):
        out = real_micro(*args, **kwargs)
        return out + 1.0  # 1 nat of disagreement, far past log1p(0.2)

    monkeypatch.setattr("p2t.trainer.rollout_logp_microbatch", shifted_micro)
    trainer.rollout = shifted_rollout

    with pytest.raises(ValueError, match="probability protocols differ"):
        trainer.train_rollout(["p"])
    assert steps["count"] == 0, "the optimizer must not step under a broken protocol"


def test_unmapped_content_gate_rejects_a_failing_mapping(tmp_path, monkeypatch):
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)

    def bad_mapping(*args, **kwargs):
        rows, mapped, fixed, stats = fake_batch(*args, **kwargs)
        stats["rm_unmapped_content_fraction"] = 0.9
        return rows, mapped, fixed, stats

    monkeypatch.setattr("p2t.trainer.build_rm_batch", bad_mapping)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    with pytest.raises(ValueError, match="Unmapped content tokens"):
        trainer.train_rollout(["p"])


def test_train_rollout_skips_the_update_when_no_group_survives(tmp_path, monkeypatch):
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    base = trainer.rollout

    def all_truncated(batch_prompts):
        """Every row fills the cap with no stop token: a consistent truncation."""
        rollout, summary = base(batch_prompts)
        input_ids, full_mask, positions, responses, rmask, rendered, _, logprobs = rollout
        responses = responses.clone()
        responses[responses == 2] = 3  # drop the stop token
        rmask = torch.ones_like(rmask)
        full_mask = torch.cat([torch.ones_like(input_ids[:, :input_ids.shape[1] - rmask.shape[1]]),
                               rmask], dim=1)
        return ((input_ids, full_mask, positions, responses, rmask, rendered,
                 ["length"] * len(responses), logprobs), summary)

    trainer.rollout = all_truncated
    before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    metrics = trainer.train_rollout(["p"])
    assert metrics["skipped_rollout"] is True and metrics["optimizer_steps"] == 0
    assert metrics["loss"] == 0.0
    assert all(torch.equal(old, new) for old, new in zip(before, trainer.actor.parameters()))


# ------------------------------------------------------------------ autopush
def _git_repo(tmp_path: Path) -> Path:
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q", "-b", "p2t-baseline"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "test")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_autopush_is_inert_when_disabled(tmp_path):
    pusher = AutoPusher(enabled=False, every=5, remote="origin", branch="p2t-baseline",
                        report_dir=tmp_path / "report", repo_root=tmp_path)
    assert pusher.push(5, {}) is False


def test_autopush_refuses_a_branch_it_does_not_own(tmp_path):
    repo = _git_repo(tmp_path)
    pusher = AutoPusher(enabled=True, every=1, remote="origin", branch="some-other-branch",
                        report_dir=tmp_path / "report", repo_root=repo)
    (repo / "new.txt").write_text("change")
    assert pusher.push(1, {}) is False
    log = (tmp_path / "report" / "git_push.log").read_text()
    assert "refused_wrong_branch" in log


def test_autopush_refuses_oversized_staged_files(tmp_path):
    repo = _git_repo(tmp_path)
    pusher = AutoPusher(enabled=True, every=1, remote="origin", branch="p2t-baseline",
                        report_dir=tmp_path / "report", repo_root=repo)
    (repo / "big.bin").write_bytes(b"0" * (51 * 1024 * 1024))
    assert pusher.push(1, {}) is False
    assert "refused_oversized" in (tmp_path / "report" / "git_push.log").read_text()


def test_checkpoint_pruning_keeps_only_the_newest(tmp_path):
    """A 30-hour run needs rollback points, but each checkpoint is ~2 GiB.

    Without pruning, 250 steps at `checkpoint_interval=20` would leave 13 of them
    on disk.  The newest is always kept, so the final checkpoint is never the one
    removed.
    """
    trainer, _, _ = _stub_trainer(tmp_path)
    trainer.cfg = type(trainer.cfg)(**{**trainer.cfg.__dict__, "keep_checkpoints": 2})
    for step in (20, 40, 60):
        (trainer.output_dir / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    (trainer.output_dir / "checkpoint-notanumber").mkdir(parents=True, exist_ok=True)
    trainer._prune_checkpoints()
    kept = sorted(p.name for p in trainer.output_dir.glob("checkpoint-*"))
    assert kept == ["checkpoint-40", "checkpoint-60", "checkpoint-notanumber"], \
        "keep the newest two step-numbered checkpoints, and never touch what is not one"


def test_checkpoint_pruning_can_be_disabled(tmp_path):
    trainer, _, _ = _stub_trainer(tmp_path)
    trainer.cfg = type(trainer.cfg)(**{**trainer.cfg.__dict__, "keep_checkpoints": 0})
    for step in (20, 40, 60):
        (trainer.output_dir / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    trainer._prune_checkpoints()
    assert len(list(trainer.output_dir.glob("checkpoint-*"))) == 3


def test_formal_config_checkpoints_periodically_for_rollback():
    """The formal run is ~30 hours; a crash must not cost all of it.

    The interval need not divide the rollout count: `train()` saves a final
    checkpoint unconditionally after the loop, so step 250 is always captured
    even though 250 % 20 != 0.  With `keep_checkpoints=2` the run ends holding
    checkpoint-240 and checkpoint-250.
    """
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "formal250.json")
    assert config.checkpoint_interval == 20
    assert config.keep_checkpoints == 2
    periodic = config.rollout_iterations // config.checkpoint_interval
    assert periodic >= 10, "a 30-hour run needs many rollback points, not a few"


def test_autopush_pushes_a_local_branch_to_a_differently_named_remote_branch(tmp_path):
    """The work lives on `p2t-baseline`; the dedicated P2T remote takes it as `main`.

    Conflating the two names made `_push_locked` abort with `refused_wrong_branch`
    on every single step, so the shipped configs could never have pushed anything.
    The local branch gates committing; the remote name is only a push target.
    """
    repo = _git_repo(tmp_path)  # created on p2t-baseline
    pusher = AutoPusher(enabled=True, every=1, remote="origin", branch="main",
                        local_branch="p2t-baseline",
                        report_dir=tmp_path / "report", repo_root=repo)
    calls = []
    real_run = pusher._run

    def recording_run(*args, check=True, env=None):
        calls.append(tuple(args))
        if args and args[0] == "push":
            class _Result:
                returncode, stdout, stderr = 0, "", ""
            return _Result()
        return real_run(*args, check=check, env=env)

    pusher._run = recording_run
    (repo / "new.txt").write_text("change")
    assert pusher.push(1, {"raw_reward_mean": 1.0}) is True
    assert ("push", "origin", "p2t-baseline:main") in calls, \
        "the refspec must name both branches explicitly"


def test_autopush_still_refuses_a_branch_the_run_does_not_own(tmp_path):
    """Splitting the two names must not weaken the guard on the local branch."""
    repo = _git_repo(tmp_path)  # on p2t-baseline
    pusher = AutoPusher(enabled=True, every=1, remote="origin", branch="main",
                        local_branch="some-other-branch",
                        report_dir=tmp_path / "report", repo_root=repo)
    (repo / "new.txt").write_text("change")
    assert pusher.push(1, {}) is False
    assert "refused_wrong_branch" in (tmp_path / "report" / "git_push.log").read_text()


def test_shipped_configs_declare_both_push_branch_names():
    root = Path(__file__).resolve().parents[2]
    for name in ("smoke10", "formal250", "p2t-pilot"):
        config = load_config(root / "configs" / f"{name}.json")
        assert config.push_branch == "main"
        assert config.push_local_branch == "p2t-baseline"


def test_generation_server_disables_custom_all_reduce_as_an_engine_argument(tmp_path, monkeypatch):
    """`VLLM_DISABLE_CUSTOM_ALL_REDUCE` does not exist in the installed vLLM.

    Setting it left `disable_custom_all_reduce=False`, the custom all-reduce ran
    on a PCIe-bridge topology it does not support, a tensor-parallel worker died
    with a CUDA 'invalid argument' and the engine core never came up.  The
    setting has to travel as a command-line engine argument.
    """
    import p2t.vllm as vllm_module
    captured = {}

    class _FakePopen:
        def __init__(self, command, env=None, stdout=None, stderr=None):
            captured["command"] = list(command)
            captured["env"] = dict(env or {})

        def poll(self):
            return None

    monkeypatch.setattr(vllm_module.subprocess, "Popen", _FakePopen)
    vllm_module.GenerationServer(
        model="m", tokenizer_source="m", socket_path=tmp_path / "s.sock",
        gpus=["2", "3"], max_num_seqs=8, seed=0, gpu_memory_utilization=0.85,
        tensor_parallel_size=2)
    assert "--disable-custom-all-reduce" in captured["command"]
    assert "VLLM_DISABLE_CUSTOM_ALL_REDUCE" not in captured["env"], \
        "the environment variable is unknown to this vLLM and only looks like a guard"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_autopush_honours_the_interval(tmp_path):
    repo = _git_repo(tmp_path)
    pusher = AutoPusher(enabled=True, every=5, remote="origin", branch="p2t-baseline",
                        report_dir=tmp_path / "report", repo_root=repo)
    (repo / "new.txt").write_text("change")
    assert pusher.maybe_push(1, {}) is None
    assert not (tmp_path / "report" / "git_push.log").exists()
