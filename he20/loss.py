"""The project's GRPO surrogate, optionally restricted to the kept tokens.

The paper's contribution is not a new objective: Eq. (6) changes its base in only
two places, both of them the mask.  So the surrogate, the clipping and the KL here
are **the project's own**, copied from ``p2t/loss.py`` (itself a mirror of
``vpo_rm/core.py:grpo_policy_loss``) so that an ``he20`` run and a P2T run differ in
which tokens reach the loss and in nothing else.  ``tests/he20/test_conformance.py``
asserts that agreement at ``atol=0, rtol=0``, gradients included, for the unmasked
path; ``tests/he20/test_loss.py`` asserts that the masked path differs from it in
exactly the way Eq. (6) says it should.

Two reductions worth stating, because the paper's Eq. (6) writes only one of them:

* The paper's normaliser is a single **global** masked mean over the batch,
  ``sum_i sum_t I[...] * surrogate / sum_i sum_t I[...]``.  This module keeps the
  project's reduction instead -- per-response token mean, then mean over responses
  (``p2t/loss.py:59``) -- with the mask restricting each response's token set.  That
  is the "align with the project" choice recorded in the notes: the arm measures the
  paper's mask against the project's GRPO baseline, and swapping the reduction too
  would confound the two.  Where a response keeps no token it contributes nothing to
  either the numerator or the denominator, which is what Eq. (6) does.
* The difference is why ``entropy_top_mask=None`` must stay **bit-identical** to the
  sibling arms: with no mask the two reductions coincide, and that equivalence is
  the paper's own baseline claim (its method reduces to DAPO at ``rho = 1``).

The KL term is deliberately **not** masked.  Eq. (6) masks the surrogate; the KL is
this project's separate constraint, and the paper has no KL term at all (DAPO
removes it), so it says nothing about the KL's scope.  Masking it would silently
narrow the trust region to the high-entropy tokens, which is a different change from
the one the paper makes.  The alternative reading is recorded in the notes.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .mask import _binary_mask


def _kept_mask(response_mask: Tensor, entropy_top_mask: Tensor | None) -> Tensor:
    """The tokens that may contribute to the loss: the response, minus what Eq. (6) drops."""
    mask = _binary_mask(response_mask)
    if entropy_top_mask is None:
        return mask
    if not isinstance(entropy_top_mask, Tensor) or entropy_top_mask.shape != mask.shape:
        raise ValueError("entropy_top_mask must be a tensor of shape [B, T] matching response_mask")
    if entropy_top_mask.device != mask.device:
        raise ValueError(f"entropy_top_mask is on {entropy_top_mask.device} but the response "
                         f"mask is on {mask.device}")
    selected = entropy_top_mask.bool()
    if (selected & ~mask).any():
        raise ValueError("entropy_top_mask keeps a token outside the response mask")
    # No emptiness check here, deliberately.  This loss is called once per physical
    # micro-batch on a *slice* of a mask that was built over the whole optimizer
    # minibatch, and a slice may legitimately keep nothing -- Eq. (6) sums the
    # indicator, so a response whose every token fell below tau contributes nothing
    # to either term.  The guarantee that the selection is non-empty belongs where
    # the whole population is visible, and lives there: ``mask.entropy_top_mask``
    # raises rather than returning an all-False mask.
    return mask & selected


def grpo_policy_loss(new_logp: Tensor, old_logp: Tensor, token_advantage: Tensor,
                     response_mask: Tensor, clip_eps: float = 0.2, *,
                     importance_weights: Tensor | None = None,
                     entropy_top_mask: Tensor | None = None) -> Tensor:
    """Negative clipped objective: average tokens per response, then responses.

    ``entropy_top_mask`` is Eq. (6)'s ``I[H_t^i >= tau_rho^B]``: the loss is computed
    over ``response_mask & entropy_top_mask`` only, so both the summed surrogate and
    the per-response token count are restricted to the kept tokens.  ``None``
    reproduces the sibling arms exactly.

    Clipping happens in log space before exponentiation: for positive advantage
    ``min(ratio, 1+eps)`` is ``exp(clamp(log_ratio, max=log1p(eps)))``, which
    avoids the ``0 * inf = NaN`` backward of computing ``exp`` first.
    """
    mask = _binary_mask(response_mask)
    if any(x.shape != mask.shape for x in (new_logp, old_logp, token_advantage)):
        raise ValueError("log probabilities, advantage and mask must have shape [B,T]")
    # `kept` is `mask` itself when no mask was supplied, so the unmasked path below
    # runs the same operations in the same order as the sibling arms' loss.
    kept = _kept_mask(response_mask, entropy_top_mask)
    weights = None
    if importance_weights is not None:
        if (not isinstance(importance_weights, Tensor) or not importance_weights.is_floating_point()
                or importance_weights.shape != mask.shape
                or importance_weights.device != new_logp.device
                or importance_weights.device != mask.device):
            raise ValueError("importance_weights must be floating [B,T] on the loss device")
        weights = importance_weights.detach().float().masked_fill(~mask, 1)
        if not (torch.isfinite(weights) & (weights > 0)).all():
            raise ValueError("Valid importance weights must be positive and finite in float32")
    if not 0 < clip_eps < 1:
        raise ValueError("clip_eps must lie in (0,1)")
    log_ratio = (new_logp.float().masked_fill(~kept, 0)
                 - old_logp.detach().float().masked_fill(~kept, 0))
    a = token_advantage.detach().float().masked_fill(~kept, 0)
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(a).all():
        raise ValueError("Valid log probabilities and token advantages must be finite")
    selected_log_ratio = torch.where(
        a >= 0, log_ratio.clamp(max=math.log1p(clip_eps)),
        log_ratio.clamp(min=math.log1p(-clip_eps)))
    selected_log_ratio = selected_log_ratio.masked_fill(a == 0, 0)
    log_scale = a.abs().masked_fill(a == 0, 1).log()
    if weights is not None:
        log_scale = log_scale + weights.log().masked_fill(a == 0, 0)
    objective = a.sign() * (selected_log_ratio + log_scale).exp()
    if not torch.isfinite(objective).all():
        raise ValueError("Policy objective overflow; gradients would not be finite")
    # A response that kept no token has an all-zero objective and is divided by a
    # clamped count, contributing exactly nothing -- what Eq. (6) does by summing the
    # indicator.  Without a mask the count is the response length, as in the siblings.
    counts = kept.sum(-1, keepdim=True)
    loss = -(objective / counts.clamp_min(1)).mean(0).sum()
    if not torch.isfinite(loss):
        raise ValueError("Policy loss overflow")
    return loss


def kl_from_logp(base_logp: Tensor, new_logp: Tensor, response_mask: Tensor, *,
                 importance_weights: Tensor | None = None) -> Tensor:
    """Project KL estimator: ``exp(d) - d - 1`` with ``d = log p_ref - log p_new``.

    Reduced as per-response mean over valid tokens, then mean over responses,
    which is the reduction the project's optimizer step applies to the KL term.
    Not masked: see the module docstring.
    """
    mask = _binary_mask(response_mask)
    if base_logp.shape != mask.shape or new_logp.shape != mask.shape:
        raise ValueError("log probabilities must have shape [B,T]")
    delta = (base_logp.float().masked_fill(~mask, 0)
             - new_logp.float().masked_fill(~mask, 0))
    values = torch.expm1(delta) - delta
    if importance_weights is not None:
        values = values * importance_weights.detach().float().masked_fill(~mask, 1)
    per_response = values.masked_fill(~mask, 0).sum(-1) / mask.sum(-1).float()
    return per_response.mean()
