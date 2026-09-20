"""Eq. (3)-(5): hand-computed values, the paper's own inconsistency, and sign behaviour."""
import math

import pytest
import torch

from p2t.reward import (P2T_ALPHA_SHORT_COT, P2T_OMEGA, group_advantages, p2t_credit,
                        p2t_token_advantage, p2t_token_reward)


def test_eq3_matches_hand_computed_values_and_the_paper_sum_is_not_R():
    """Eq. (3) as written sums to (N + omega) * R, not to R as section 3.3.1 claims.

    Pinning the actual value makes the discrepancy a documented property of the
    reproduction instead of a silent reinterpretation.
    """
    attribution = torch.zeros(1, 3)
    rewards = torch.tensor([2.0])
    mask = torch.ones(1, 3, dtype=torch.bool)
    token_reward, share = p2t_token_reward(attribution, rewards, mask, omega=0.6)
    torch.testing.assert_close(share, torch.full((1, 3), 1 / 3))
    # Uniform attribution -> uniform share -> R * (1 + omega / N)
    torch.testing.assert_close(token_reward, torch.full((1, 3), 2.0 * (1 + 0.6 / 3)))
    total = token_reward.sum().item()
    torch.testing.assert_close(total, (3 + 0.6) * 2.0)
    assert not math.isclose(total, 2.0), "the paper's prose claim does not hold for Eq. (3)"


def test_eq3_hand_computed_weighted_case():
    attribution = torch.log(torch.tensor([[2.0, 1.0, 1.0]]))
    rewards = torch.tensor([-1.0])
    mask = torch.ones(1, 3, dtype=torch.bool)
    token_reward, share = p2t_token_reward(attribution, rewards, mask, omega=0.6)
    torch.testing.assert_close(share, torch.tensor([[0.5, 0.25, 0.25]]))
    torch.testing.assert_close(token_reward, torch.tensor([[-1.3, -1.15, -1.15]]))
    # Negative R inverts the meaning of the omega term: the highest-attribution
    # token receives the most negative reward. Paper-faithful, and flagged.
    assert token_reward[0, 0] < token_reward[0, 1]


def test_eq5_and_credit_field_invariants():
    torch.manual_seed(4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    attribution = torch.randn(2, 4)
    rewards = torch.tensor([3.0, -2.0])
    advantages = torch.tensor([1.5, -0.5])
    credit = p2t_credit(rewards, attribution, advantages, mask,
                        omega=P2T_OMEGA, alpha=P2T_ALPHA_SHORT_COT)

    token_reward, _ = p2t_token_reward(attribution, rewards, mask, P2T_OMEGA)
    torch.testing.assert_close(credit.advantage,
                               p2t_token_advantage(token_reward, advantages, mask,
                                                   P2T_ALPHA_SHORT_COT))
    # A~ = A^hat + direction on valid positions, so the VPO arms' utility gauge
    # stays comparable.  Padded positions are zero in all three fields.
    torch.testing.assert_close(credit.advantage[mask],
                               advantages[:, None].expand_as(mask)[mask] + credit.direction[mask],
                               atol=1e-6, rtol=1e-6)
    for field in (credit.advantage, credit.direction, credit.weight):
        assert (field[~mask] == 0).all()
    # Per-response valid-token mean of the diagnostic weight is one.
    counts = mask.sum(-1, keepdim=True).float()
    torch.testing.assert_close(credit.weight.sum(-1), counts.squeeze(-1))
    assert torch.allclose(credit.weight[mask].mean(), torch.tensor(1.0))
    assert (credit.weight[~mask] == 0).all()
    assert credit.tau_used is None
    assert not credit.advantage.requires_grad


def test_group_advantages_standardises_within_each_prompt():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 0.0])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    advantage, scale = group_advantages(rewards, groups)
    for group in groups.unique():
        selected = groups == group
        torch.testing.assert_close(advantage[selected].mean(), torch.tensor(0.0),
                                   atol=1e-6, rtol=1e-6)
        population = rewards[selected].double().std(correction=0).float()
        # eps is additive, the same convention as vpo_rm.core.group_advantages.
        torch.testing.assert_close(scale[selected][0], population + 1e-6, atol=1e-6, rtol=1e-5)


def test_group_advantages_std_floor_binds_only_below_the_floor():
    """The floor replaces eps additively; it never raises a wide group's scale."""
    tight = torch.tensor([1.0, 1.0, 1.1, 1.1])
    groups = torch.zeros(4, dtype=torch.long)
    advantage, scale = group_advantages(tight, groups, std_floor=0.5)
    torch.testing.assert_close(scale, torch.full((4,), 0.5))
    torch.testing.assert_close(advantage, (tight - tight.mean()) / 0.5)

    wide = torch.tensor([1.0, 1.0, 4.0, 4.0])
    _, wide_scale = group_advantages(wide, groups, std_floor=0.5)
    population = wide.double().std(correction=0).float()
    assert population > 0.5
    torch.testing.assert_close(wide_scale, population.expand(4), atol=1e-6, rtol=1e-5)


def test_group_advantages_rejects_singleton_groups_and_bad_eps():
    rewards = torch.tensor([1.0])
    with pytest.raises(ValueError, match="at least two responses"):
        group_advantages(rewards, torch.tensor([0]))
    with pytest.raises(ValueError, match="eps"):
        group_advantages(torch.tensor([1.0, 2.0]), torch.tensor([0, 0]), eps=0.0)


def test_share_ess_convention_matches_the_project():
    """ESS/T = 1/(T*sum p^2): 1 is a FLAT share (inert), 1/T is one-hot.

    The parent project reads its credit ESS the same way -- near one means the
    weighting is doing nothing -- so this pins the direction that the diagnostics
    and the reward-curve axis label both depend on.
    """
    mask = torch.ones(1, 8, dtype=torch.bool)
    rewards = torch.tensor([1.0])

    flat, flat_share = p2t_token_reward(torch.zeros(1, 8), rewards, mask)
    flat_tokens = mask.sum(-1).float()[0]
    flat_ess = 1.0 / (flat_share.square().sum(-1) * flat_tokens)
    torch.testing.assert_close(flat_ess, torch.tensor([1.0]), atol=1e-6, rtol=1e-6)

    peaked, peaked_share = p2t_token_reward(torch.tensor([[50.0] + [-50.0] * 7]), rewards, mask)
    peaked_ess = 1.0 / (peaked_share.square().sum(-1) * flat_tokens)
    assert float(peaked_ess) < 1.5 / 8, "a one-hot share must drive ESS/T towards 1/T"
    assert float(peaked_ess) < float(flat_ess)
    assert torch.isfinite(flat).all() and torch.isfinite(peaked).all()


def test_alpha_zero_leaves_only_the_sequence_advantage():
    mask = torch.ones(1, 2, dtype=torch.bool)
    credit = p2t_credit(torch.tensor([2.0]), torch.randn(1, 2), torch.tensor([0.7]), mask,
                        omega=0.6, alpha=0.0)
    torch.testing.assert_close(credit.advantage, torch.full((1, 2), 0.7))
    assert (credit.direction == 0).all()


@pytest.mark.parametrize("omega,alpha,match", [
    (-0.1, 0.1, "omega"), (0.6, -0.1, "alpha"), (float("nan"), 0.1, "omega"),
])
def test_negative_or_nonfinite_constants_are_rejected(omega, alpha, match):
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match=match):
        p2t_credit(torch.tensor([1.0]), torch.zeros(1, 2), torch.tensor([1.0]), mask,
                   omega=omega, alpha=alpha)
