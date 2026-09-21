"""The REINFORCE objective: no ratio, no clipping, and a detached advantage.

The properties that matter for fidelity are negative ones.  There must be no
importance ratio and no clipping -- RLOO's own paper removes both -- so the loss
has to stay linear in ``new_logp`` however far the policy has moved, which is
exactly what a clipped surrogate would break.  And the advantage must be detached,
or the baseline would receive a gradient through the reward model.
"""
from __future__ import annotations

import math

import pytest
import torch

from red.loss import kl_metric, rloo_policy_loss


def test_loss_is_the_negative_mean_advantage_weighted_log_prob():
    new_logp = torch.tensor([[-0.5, -1.0], [-2.0, -0.25]])
    advantage = torch.tensor([[1.0, 3.0], [-2.0, -4.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    got = rloo_policy_loss(new_logp, advantage, mask)
    per_response = (-(advantage * new_logp).sum(-1) / mask.sum(-1).float())
    torch.testing.assert_close(got, per_response.mean())


def test_no_clipping_so_the_loss_stays_linear_far_outside_any_clip_band():
    """A clipped surrogate would flatten here; REINFORCE must not."""
    mask = torch.ones(1, 4, dtype=torch.bool)
    advantage = torch.ones(1, 4)
    near = rloo_policy_loss(torch.full((1, 4), -1.0), advantage, mask)
    far = rloo_policy_loss(torch.full((1, 4), -50.0), advantage, mask)
    # Linear in new_logp means the loss scales by exactly 50x.
    torch.testing.assert_close(far / near, torch.tensor(50.0), rtol=1e-5, atol=1e-5)


def test_clip_eps_is_not_a_parameter():
    import inspect
    assert "clip" not in inspect.signature(rloo_policy_loss).parameters
    assert "old_logp" not in inspect.signature(rloo_policy_loss).parameters


def test_advantage_is_detached_so_the_baseline_gets_no_gradient():
    new_logp = torch.zeros(1, 2, requires_grad=True)
    advantage = torch.tensor([[1.0, 2.0]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = rloo_policy_loss(new_logp, advantage, mask)
    loss.backward()
    assert advantage.grad is None, "the leave-one-out baseline must not be differentiated"
    assert new_logp.grad is not None, "the policy log-probs must receive the gradient"


def test_importance_weights_multiply_the_log_prob_term():
    new_logp = torch.tensor([[-1.0, -1.0]])
    advantage = torch.tensor([[2.0, 2.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    plain = rloo_policy_loss(new_logp, advantage, mask)
    weighted = rloo_policy_loss(new_logp, advantage, mask,
                               importance_weights=torch.full((1, 2), 3.0))
    torch.testing.assert_close(weighted / plain, torch.tensor(3.0))


def test_zero_advantage_tokens_contribute_nothing():
    new_logp = torch.tensor([[-1.0, -100.0]])
    advantage = torch.tensor([[0.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    torch.testing.assert_close(rloo_policy_loss(new_logp, advantage, mask), torch.tensor(0.0))


def test_masked_positions_are_excluded_from_both_sides():
    new_logp = torch.tensor([[-1.0, -999.0]])
    advantage = torch.tensor([[2.0, 999.0]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    got = rloo_policy_loss(new_logp, advantage, mask)
    torch.testing.assert_close(got, torch.tensor(2.0))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_importance_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        rloo_policy_loss(torch.zeros(1, 1), torch.ones(1, 1),
                         torch.ones(1, 1, dtype=torch.bool),
                         importance_weights=torch.full((1, 1), bad))


def test_kl_metric_is_the_projects_second_order_estimator():
    """The reported metric keeps the project's estimator, not Eq. (4)'s."""
    base = torch.tensor([[0.0, 0.0]])
    new = torch.tensor([[0.0, math.log(2.0)]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    expected = ((math.exp(-math.log(2.0)) - (-math.log(2.0)) - 1) + 0.0) / 2
    torch.testing.assert_close(kl_metric(base, new, mask), torch.tensor(expected))
