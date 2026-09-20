"""GRPO policy loss and KL, numerically identical to the project's trainer.

The paper changes only how the token advantage is produced (Eq. (3)-(5)); the
surrogate objective, the clipping and the KL term are exactly GRPO's.  These
functions therefore mirror ``vpo_rm/core.py:grpo_policy_loss`` and the trainer's
KL estimator, so a P2T run and a VPO-RM run differ in credit assignment and
nothing else.  ``tests/test_conformance.py`` asserts the loss agrees with the
project's implementation on random inputs, including gradients.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .attribution import _binary_mask


def grpo_policy_loss(new_logp: Tensor, old_logp: Tensor, token_advantage: Tensor,
                     response_mask: Tensor, clip_eps: float = 0.2, *,
                     importance_weights: Tensor | None = None) -> Tensor:
    """Negative clipped objective: average tokens per response, then responses.

    Clipping happens in log space before exponentiation: for positive advantage
    ``min(ratio, 1+eps)`` is ``exp(clamp(log_ratio, max=log1p(eps)))``, which
    avoids the ``0 * inf = NaN`` backward of computing ``exp`` first.
    """
    mask = _binary_mask(response_mask)
    if any(x.shape != mask.shape for x in (new_logp, old_logp, token_advantage)):
        raise ValueError("log probabilities, advantage and mask must have shape [B,T]")
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
    log_ratio = (new_logp.float().masked_fill(~mask, 0)
                 - old_logp.detach().float().masked_fill(~mask, 0))
    a = token_advantage.detach().float().masked_fill(~mask, 0)
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
    loss = -(objective / mask.sum(-1, keepdim=True)).mean(0).sum()
    if not torch.isfinite(loss):
        raise ValueError("Policy loss overflow")
    return loss


def kl_from_logp(base_logp: Tensor, new_logp: Tensor, response_mask: Tensor, *,
                 importance_weights: Tensor | None = None) -> Tensor:
    """Project KL estimator: ``exp(d) - d - 1`` with ``d = log p_ref - log p_new``.

    Reduced as per-response mean over valid tokens, then mean over responses,
    which is the reduction the project's optimizer step applies to the KL term.
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
