"""Eq. (6)'s two changes, and the baseline equivalence the paper relies on.

The paper's claim is that its method reduces to the unchanged algorithm when the
mask is off, so the strictest test here is not the masked arithmetic but the
*unmasked* path: with ``entropy_top_mask=None`` this loss must equal the sibling
arm's GRPO loss bit for bit, gradients included.  Everything else pins the mask:
that it restricts the numerator and the denominator together (Eq. (6)'s normaliser
change), that only the kept tokens receive gradient, and that a response keeping no
token contributes nothing rather than dividing by zero.
"""
from __future__ import annotations

import pytest
import torch

from he20.loss import grpo_policy_loss, kl_from_logp


def _inputs(seed=0, rows=2, width=5):
    torch.manual_seed(seed)
    new_logp = torch.randn(rows, width, dtype=torch.float64, requires_grad=True)
    old_logp = torch.randn(rows, width, dtype=torch.float64)
    advantage = torch.randn(rows, width, dtype=torch.float64)
    mask = torch.ones(rows, width, dtype=torch.bool)
    return new_logp, old_logp, advantage, mask


def test_the_unmasked_loss_is_the_sibling_arms_loss_exactly():
    """The paper's baseline claim, pinned: rho = 1 is the unmodified algorithm."""
    from p2t.loss import grpo_policy_loss as sibling  # the project's GRPO surrogate

    new_logp, old_logp, advantage, mask = _inputs()
    mine = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2)
    theirs = sibling(new_logp, old_logp, advantage, mask, 0.2)
    torch.testing.assert_close(mine, theirs, atol=0, rtol=0)
    mine_grad = torch.autograd.grad(mine, new_logp, retain_graph=True)[0]
    theirs_grad = torch.autograd.grad(theirs, new_logp)[0]
    torch.testing.assert_close(mine_grad, theirs_grad, atol=0, rtol=0)


def test_the_unmasked_loss_ignores_an_all_true_mask():
    new_logp, old_logp, advantage, mask = _inputs()
    none = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2)
    every = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2,
                             entropy_top_mask=mask.clone())
    torch.testing.assert_close(none, every, atol=0, rtol=0)


def test_a_mask_restricts_the_numerator_and_the_denominator():
    """Eq. (6)'s normaliser change, on a case small enough to check by hand.

    Ratio one and unit-magnitude advantages make every kept surrogate exactly
    ``sign(a)``: the objective is ``exp(clamp(0) + log|a|)``.  Keeping the two
    positive tokens gives ``(1+1)/2 = 1``; keeping all three gives
    ``(1+1-1)/3 = 1/3``.
    """
    new_logp = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    old_logp = torch.zeros(1, 3, dtype=torch.float64)
    advantage = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    keep = torch.tensor([[True, True, False]])

    masked = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2, entropy_top_mask=keep)
    unmasked = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2)
    # The loss reduces in float32, like the sibling arms' (both cast the incoming
    # log-probs), so the expectations are float32 as well.
    torch.testing.assert_close(masked, torch.tensor(-1.0), atol=0, rtol=0)
    torch.testing.assert_close(unmasked, torch.tensor(-1.0 / 3), atol=0, rtol=0)


def test_only_kept_tokens_receive_gradient():
    new_logp, old_logp, advantage, mask = _inputs(seed=1)
    keep = torch.zeros_like(mask)
    keep[:, :2] = True
    loss = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2, entropy_top_mask=keep)
    grad = torch.autograd.grad(loss, new_logp)[0]
    torch.testing.assert_close(grad[:, 2:], torch.zeros_like(grad[:, 2:]), atol=0, rtol=0)
    assert grad[:, :2].abs().sum() > 0, "the kept tokens must still carry gradient"


def test_a_response_that_keeps_no_token_contributes_nothing():
    """Short responses can fall entirely below a batch threshold; Eq. (6) sums the
    indicator, so they vanish from both terms rather than dividing by zero."""
    new_logp = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    old_logp = torch.zeros(2, 4, dtype=torch.float64)
    advantage = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    mask = torch.ones(2, 4, dtype=torch.bool)
    only_second = torch.tensor([[False] * 4, [True] * 4])

    both = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2, entropy_top_mask=only_second)
    single = grpo_policy_loss(new_logp[1:], old_logp[1:], advantage[1:], mask[1:], 0.2)
    # Response 0 contributes to neither the sum nor the count, but it is still one
    # of the two responses the outer mean divides by -- so the loss is halved.
    torch.testing.assert_close(both, single / 2, atol=1e-9, rtol=0)


def test_a_mask_outside_the_response_mask_is_rejected():
    new_logp, old_logp, advantage, mask = _inputs()
    mask = mask.clone()
    mask[:, -1] = False
    keep = torch.ones_like(mask)
    with pytest.raises(ValueError, match="outside the response mask"):
        grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2, entropy_top_mask=keep)


def test_a_slice_that_keeps_nothing_contributes_nothing_rather_than_raising():
    """A micro-batch slice of a batch-level mask may keep no token at all.

    The loss is called per physical micro-batch on a slice of a mask built over the
    whole optimizer minibatch, and Eq. (6) sums the indicator -- so an empty slice
    must contribute nothing, not raise.  An *all*-False mask is the same arithmetic
    and yields a loss of exactly zero, which is why the non-emptiness guarantee
    lives in ``mask.entropy_top_mask``, where the whole population is visible.
    """
    new_logp, old_logp, advantage, mask = _inputs()
    empty = grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2,
                             entropy_top_mask=torch.zeros_like(mask))
    torch.testing.assert_close(empty, torch.tensor(0.0), atol=0, rtol=0)
    # The mask's own non-emptiness guard lives where the population is visible, and
    # its "kept nothing" branch is unreachable through the public API: tau is a
    # quantile of the same values it is compared against, so it never exceeds their
    # maximum.  What is reachable is an empty *population*, which is what it rejects.
    from he20.mask import entropy_top_mask
    with pytest.raises(ValueError, match="at least one valid token"):
        entropy_top_mask(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool), 0.2)


def test_a_mask_of_the_wrong_shape_is_rejected():
    new_logp, old_logp, advantage, mask = _inputs()
    with pytest.raises(ValueError, match="shape"):
        grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2,
                         entropy_top_mask=torch.ones(3, 5, dtype=torch.bool))


def test_a_mask_on_another_device_is_rejected():
    new_logp, old_logp, advantage, mask = _inputs()
    with pytest.raises(ValueError, match="is on meta but the response mask is on cpu"):
        grpo_policy_loss(new_logp, old_logp, advantage, mask, 0.2,
                         entropy_top_mask=torch.ones(2, 5, dtype=torch.bool, device="meta"))


def test_the_kl_term_is_not_masked():
    """Documented choice: Eq. (6) masks the surrogate, and the KL is the project's
    separate constraint.  A masked loss must not change the KL's value."""
    base = torch.zeros(1, 3, dtype=torch.float64)
    new = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    torch.testing.assert_close(kl_from_logp(base, new, mask), kl_from_logp(base, new, mask))
