"""Config contract, the full credit chain on a tiny model, and autopush guardrails."""
import json
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
from p2t.trainer import TrainerConfig, load_config


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
    for name in ("smoke10", "formal250"):
        payload = json.loads((root / "configs" / f"{name}.json").read_text())
        payload.pop("_comment", None)
        config = TrainerConfig(**payload).resolved()
        assert config.push_branch == "p2t-baseline"
        if name == "smoke10":
            assert config.init_adapter == "", "the smoke must run on the raw base model"
            assert config.rollout_iterations == 10


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


def test_autopush_honours_the_interval(tmp_path):
    repo = _git_repo(tmp_path)
    pusher = AutoPusher(enabled=True, every=5, remote="origin", branch="p2t-baseline",
                        report_dir=tmp_path / "report", repo_root=repo)
    (repo / "new.txt").write_text("change")
    assert pusher.maybe_push(1, {}) is None
    assert not (tmp_path / "report" / "git_push.log").exists()
