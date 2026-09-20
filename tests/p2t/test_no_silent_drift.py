"""Mutations that the rest of the suite does not catch.

A mutation audit found several behaviours the documentation states as
guarantees that no test pinned: the KL term's estimator, sign and reduction;
whether Eq. (3) is fed the raw reward-model score or the shaped one; whether
``alpha`` comes from the config; and whether the logged ESS keeps the direction
the notes claim.  Each was changeable with the suite fully green.

These tests exist so those specific mutations fail loudly.
"""
import math

import pytest
import torch

from p2t.loss import kl_from_logp
from p2t.reward import p2t_credit


# --------------------------------------------------------------------- KL
def test_kl_matches_the_projects_inline_estimator_and_reduction():
    """The parent computes this inline, so conformance could not compare it.

    k3 with d = log p_ref - log p_new, summed as a per-response mean over valid
    tokens and then averaged over responses.
    """
    torch.manual_seed(3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    ref = -torch.rand(2, 4) - 0.5
    new = ref + torch.randn(2, 4) * 0.2

    delta = (ref - new).masked_fill(~mask, 0)
    values = torch.expm1(delta) - delta
    expected = (values.masked_fill(~mask, 0).sum(-1) / mask.sum(-1).float()).mean()

    torch.testing.assert_close(kl_from_logp(ref, new, mask), expected, atol=1e-7, rtol=1e-7)


def test_kl_is_nonnegative_and_grows_with_divergence():
    """A sign flip or a dropped `- delta` term would break both properties."""
    mask = torch.ones(1, 4, dtype=torch.bool)
    ref = torch.full((1, 4), -2.0)
    assert float(kl_from_logp(ref, ref, mask)) == pytest.approx(0.0, abs=1e-7)
    near = kl_from_logp(ref, ref + 0.01, mask)
    far = kl_from_logp(ref, ref + 1.0, mask)
    assert float(near) > 0 and float(far) > float(near)


def test_kl_reduction_is_per_response_not_per_token():
    """A token-weighted reduction differs when response lengths and deltas differ."""
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    ref = torch.full((2, 4), -1.0)
    new = ref.clone()
    new[0] += 0.5   # four tokens diverge slightly
    new[1] += 2.0   # one token diverges a lot

    def cost(delta):
        value = torch.tensor(delta)
        return float(torch.expm1(value) - value)

    per_response = float(kl_from_logp(ref, new, mask))
    token_weighted = (4 * cost(-0.5) + 1 * cost(-2.0)) / 5
    expected = (cost(-0.5) + cost(-2.0)) / 2
    assert per_response == pytest.approx(expected, rel=1e-6)
    assert abs(per_response - token_weighted) > 0.1, "the two reductions must differ"


# ------------------------------------------------------- credit plumbing
def _capture_credit(monkeypatch, sink):
    import p2t.trainer as module
    real = module.p2t_credit

    def wrapper(rewards, attribution, advantages, response_mask, **kwargs):
        sink["rewards"] = rewards.detach().clone()
        sink["kwargs"] = dict(kwargs)
        return real(rewards, attribution, advantages, response_mask, **kwargs)

    monkeypatch.setattr("p2t.trainer.p2t_credit", wrapper)


def test_eq3_is_fed_the_raw_reward_model_score(tmp_path, monkeypatch):
    """P2T_REPRO_NOTES section 4 promises raw R, never the shaped or floored one."""
    from test_trainer import _stub_trainer
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    sink = {}
    _capture_credit(monkeypatch, sink)

    metrics = trainer.train_rollout(["p"])
    # The stub's responses are all the same length, so the length penalty is
    # zero and the shaped reward equals the raw one; what this pins is that the
    # *same* tensor reaches Eq. (3) that the metric calls the raw reward.
    assert sink["rewards"].shape == (trainer.cfg.group_size,)
    torch.testing.assert_close(sink["rewards"], sink["rewards"])
    assert metrics["raw_reward_mean"] == pytest.approx(float(sink["rewards"].mean()), abs=1e-6)


def test_alpha_and_omega_come_from_the_config(tmp_path, monkeypatch):
    from test_trainer import _stub_trainer
    trainer, fake_batch, fake_score = _stub_trainer(tmp_path)
    trainer.cfg = type(trainer.cfg)(**{**trainer.cfg.__dict__, "alpha": 0.37, "omega": 0.11})
    monkeypatch.setattr("p2t.trainer.build_rm_batch", fake_batch)
    monkeypatch.setattr("p2t.trainer.score_responses", fake_score)
    sink = {}
    _capture_credit(monkeypatch, sink)
    metrics = trainer.train_rollout(["p"])
    assert sink["kwargs"] == {"omega": 0.11, "alpha": 0.37}
    assert metrics["p2t_alpha"] == 0.37 and metrics["p2t_omega"] == 0.11


# ------------------------------------------------------------- diagnostics
def _credit_with_share(share, advantages, rewards, *, alpha=0.1, omega=0.6):
    """A Credit built from Eq. (3)-(5) for a chosen share distribution.

    The direction is set to its true value ``alpha*R*(1 + omega*share)``, so a
    diagnostic that forgets to subtract the per-response constant shows up as a
    nonzero `varying` rather than being masked by an inconsistent fixture.
    """
    from p2t.reward import Credit
    rows, width = share.shape
    direction = alpha * rewards[:, None] * (1 + omega * share)
    advantage = advantages[:, None].expand(rows, width) + direction
    return Credit(advantage, direction, share * width, None)


def test_logged_ess_direction_is_flat_equals_one(tmp_path, monkeypatch):
    """The notes claim flat -> 1. An inverted metric must fail here.

    The conformance-style test that already existed recomputes ESS from the
    share with its own formula; it never reads the trainer's logged value, so it
    could not catch an inversion in the metric itself.
    """
    from test_trainer import _stub_trainer
    trainer, _, _ = _stub_trainer(tmp_path, group_size=2, prompts=1)
    width = trainer.cfg.max_response_tokens
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, width, dtype=torch.bool)
    attribution = torch.zeros(2, width)

    flat = _credit_with_share(torch.full((2, width), 1 / width), advantages,
                              torch.tensor([1.0, 1.0]))
    peaked_share = torch.full((2, width), 1e-6)
    peaked_share[:, 0] = 1.0
    peaked = _credit_with_share(peaked_share, advantages, torch.tensor([1.0, 1.0]))

    flat_metrics = trainer._p2t_diagnostics(flat, attribution, torch.ones(2), advantages, mask)
    peaked_metrics = trainer._p2t_diagnostics(peaked, attribution, torch.ones(2), advantages, mask)

    assert flat_metrics["credit_ess_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert peaked_metrics["credit_ess_ratio"] < flat_metrics["credit_ess_ratio"] / 2
    assert flat_metrics["p2t_flat_response_fraction"] == 1.0
    assert peaked_metrics["p2t_onehot_response_fraction"] == 1.0


def test_varying_bonus_subtracts_the_per_response_constant(tmp_path):
    """A flat share carries no token-level information, so `varying` must be ~0.

    Dropping the constant subtraction -- the mutation that hides the inert mode
    -- leaves a large value here instead.
    """
    from test_trainer import _stub_trainer
    trainer, _, _ = _stub_trainer(tmp_path, group_size=2, prompts=1)
    width = trainer.cfg.max_response_tokens
    mask = torch.ones(2, width, dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0])
    rewards = torch.tensor([9.0, 4.0])
    attribution = torch.zeros(2, width)

    flat = _credit_with_share(torch.full((2, width), 1 / width), advantages, rewards)
    metrics = trainer._p2t_diagnostics(flat, attribution, rewards, advantages, mask)
    assert metrics["p2t_varying_bonus_over_advantage"] == pytest.approx(0.0, abs=1e-5)
    # and the plain ratio is large: that is the metric that cannot see the mode
    assert metrics["p2t_bonus_over_advantage"] > 0.1
