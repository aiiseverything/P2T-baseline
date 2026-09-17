import math

import pytest
import torch

from vpo_rm import core


def test_sampler_ratio_is_detached_fp32_and_not_clipped_or_normalized():
    old = torch.tensor([[.25, .75]], dtype=torch.float64).log().requires_grad_()
    rollout = torch.tensor([[.2, .8]], dtype=torch.float64).log().requires_grad_()
    weights = core.rollout_importance_weights(old, rollout, torch.ones(1, 2, dtype=torch.bool))
    assert weights.dtype == torch.float32 and not weights.requires_grad
    torch.testing.assert_close(weights, torch.tensor([[1.25, .9375]]))


def test_identity_sampler_and_padding_nan_produce_exact_unit_weights():
    old = torch.tensor([[-1., float('nan'), -100.]])
    rollout = torch.tensor([[-1., float('inf'), -100.]])
    mask = torch.tensor([[True, False, True]])
    weights = core.rollout_importance_weights(old, rollout, mask)
    assert torch.equal(weights, torch.ones_like(old))
    assert torch.isnan(old[0, 1]) and torch.isinf(rollout[0, 1])


@pytest.mark.parametrize('target', ['old', 'rollout'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf'), .01])
def test_sampler_ratio_rejects_invalid_valid_log_probabilities(target, invalid):
    values = {'old': torch.tensor([[-1.]]), 'rollout': torch.tensor([[-2.]])}
    values[target][0, 0] = invalid
    with pytest.raises(ValueError, match='finite|positive|nonpositive'):
        core.rollout_importance_weights(values['old'], values['rollout'], torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize('target', ['old', 'rollout'])
@pytest.mark.parametrize('invalid', [torch.tensor([[-1]]), torch.tensor([[-1.+0j]]), [[-1.]]])
def test_sampler_ratio_requires_floating_tensors(target, invalid):
    values = {'old': torch.tensor([[-1.]]), 'rollout': torch.tensor([[-2.]])}
    values[target] = invalid
    with pytest.raises(ValueError, match='floating|tensor'):
        core.rollout_importance_weights(values['old'], values['rollout'], torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize('bad', [torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 1, device='meta')])
def test_sampler_ratio_rejects_shape_or_device_mismatch(bad):
    with pytest.raises(ValueError, match='shape|device'):
        core.rollout_importance_weights(torch.tensor([[-1.]]), bad, torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize('mask', [torch.tensor([[2]]), torch.tensor([[False]]), torch.ones(1)])
def test_sampler_ratio_rejects_invalid_or_empty_response_masks(mask):
    with pytest.raises(ValueError, match='mask|valid'):
        core.rollout_importance_weights(torch.tensor([[-1.]]), torch.tensor([[-1.]]), mask)


@pytest.mark.parametrize('old,rollout', [(-1., -1000.), (-1000., -1.), (-1e300, -1e300), (1e-300, -1.)])
def test_sampler_ratio_rejects_unrepresentable_weights_and_positive_double_logp(old, rollout):
    with pytest.raises(ValueError, match='finite|positive|float32'):
        core.rollout_importance_weights(torch.tensor([[old]], dtype=torch.float64),
                                       torch.tensor([[rollout]], dtype=torch.float64),
                                       torch.ones(1, 1, dtype=torch.bool))


@pytest.mark.parametrize('current', [.1, .25, .35, .9])
@pytest.mark.parametrize('advantages', [[1., -1.], [-1., 1.], [0., 0.]])
def test_decoupled_loss_matches_enumerated_proximal_expectation(current, advantages):
    p0 = torch.tensor([.25, .75])
    q = torch.tensor([.2, .8])
    advantage = torch.tensor(advantages)
    logits = torch.tensor([current, 1-current]).log().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    old = p0.log()[:, None].requires_grad_()
    rollout = q.log()[:, None].requires_grad_()
    mask = torch.ones(2, 1, dtype=torch.bool)
    weights = core.rollout_importance_weights(old, rollout, mask)
    # Two one-token rows have a uniform empirical mean; 2*q restores exact
    # expectation under the two-action sampling distribution.
    loss = core.grpo_policy_loss(logits.log_softmax(-1)[:, None], old,
                                (2*q*advantage)[:, None], mask,
                                importance_weights=weights)
    ratio = reference_logits.softmax(-1) / p0
    expected = -(p0 * torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).sum()
    loss.backward(); expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert old.grad is None and rollout.grad is None
    if current == .25 and advantages == [1., -1.]:
        torch.testing.assert_close(loss, torch.tensor(.5))
        torch.testing.assert_close(logits.grad, torch.tensor([-.375, .375]))


def test_unit_importance_is_bitwise_legacy_loss_and_gradient():
    new = torch.tensor([[-.8, -2.], [-2.4, -1.]], requires_grad=True)
    old = torch.full((2, 2), -1.)
    advantage = torch.tensor([[2., -1.], [-3., 0.]])
    mask = torch.tensor([[True, True], [True, False]])
    legacy = core.grpo_policy_loss(new, old, advantage, mask)
    corrected = core.grpo_policy_loss(new, old, advantage, mask, importance_weights=torch.ones_like(new))
    assert torch.equal(legacy, corrected)
    left, = torch.autograd.grad(legacy, new, retain_graph=True)
    right, = torch.autograd.grad(corrected, new)
    assert torch.equal(left, right)


def test_importance_preserves_response_reduction_detaches_and_ignores_padding():
    new = torch.tensor([[-1., -1.], [-1., float('nan')]], requires_grad=True)
    old = new.detach().clone().requires_grad_()
    advantages = torch.tensor([[2., -2.], [3., float('nan')]], requires_grad=True)
    weights = torch.tensor([[.5, 2.], [3., float('nan')]], requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    loss = core.grpo_policy_loss(new, old, advantages, mask, importance_weights=weights)
    loss.backward()
    # -(mean([1, -4]) + mean([9])) / 2 = -3.75, without renormalizing rho.
    torch.testing.assert_close(loss, torch.tensor(-3.75))
    torch.testing.assert_close(new.grad, torch.tensor([[-.25, 1.], [-4.5, 0.]]))
    assert old.grad is None and weights.grad is None and advantages.grad is None


@pytest.mark.parametrize('weights', [
    torch.tensor([[0.]]), torch.tensor([[-1.]]), torch.tensor([[float('nan')]]),
    torch.tensor([[float('inf')]]), torch.tensor([[1]]), torch.tensor([[1.+0j]]),
    torch.ones(1, 2), torch.ones(1, 1, device='meta'), [[1.]],
    torch.tensor([[1e300]], dtype=torch.float64),
])
def test_loss_rejects_invalid_importance_weights(weights):
    with pytest.raises(ValueError, match='importance|weight'):
        core.grpo_policy_loss(torch.tensor([[-1.]]), torch.tensor([[-1.]]),
                              torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
                              importance_weights=weights)


def test_logspace_correction_avoids_overflow_before_small_weight_is_applied():
    new = torch.tensor([[-1.]], requires_grad=True)
    loss = core.grpo_policy_loss(new, torch.tensor([[-101.]]), -torch.ones(1, 1),
                                torch.ones(1, 1, dtype=torch.bool),
                                importance_weights=torch.tensor([[1e-30]]))
    loss.backward()
    expected = torch.tensor(math.exp(100.) * 1e-30)
    torch.testing.assert_close(loss, expected, rtol=5e-6, atol=0)
    torch.testing.assert_close(new.grad[0, 0], expected, rtol=5e-6, atol=0)


def test_final_weighted_objective_overflow_is_rejected():
    with pytest.raises(ValueError, match='overflow|finite'):
        core.grpo_policy_loss(torch.tensor([[-1.]], requires_grad=True), torch.tensor([[-51.]]),
                              -torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
                              importance_weights=torch.tensor([[1e30]]))


def test_zero_advantage_with_extreme_ratio_has_zero_finite_gradient():
    new = torch.tensor([[-1.]], requires_grad=True)
    loss = core.grpo_policy_loss(new, torch.tensor([[-1001.]]), torch.zeros(1, 1),
                                torch.ones(1, 1, dtype=torch.bool),
                                importance_weights=torch.tensor([[1e30]]))
    loss.backward()
    assert loss.item() == 0 and new.grad.item() == 0
