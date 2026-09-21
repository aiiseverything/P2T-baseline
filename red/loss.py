"""The REINFORCE-style RLOO objective, and the KL metric RED reports.

RED's Eq. (8) folds the KL penalty into the per-token reward, so this loss has
**no separate KL term** -- the constraint reaches the policy through the
advantage instead.  That is a deliberate departure from the sibling P2T arm,
which keeps the project's KL-as-a-separate-loss-term convention; the two are not
interchangeable and ``RED_REPRO_NOTES.md`` 2.7 records why the paper's placement
wins here.

The surrogate is REINFORCE, not a clipped ratio surrogate.  RLOO's own paper
removes the importance ratio and the clipping entirely (Section 3.2: the loss is
clipped on average <5% of the time per batch, and removing clipping gives a
slight boost), and its Appendix Eq. (11) writes the k=2 objective as a weight
times a difference of plain sequence log-probabilities.  A clipped GRPO
surrogate would therefore be a different algorithm wearing RED's rewards.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .reward import _binary_mask


def rloo_policy_loss(new_logp: Tensor, advantage: Tensor, response_mask: Tensor, *,
                     importance_weights: Tensor | None = None) -> Tensor:
    """``-(1/B) sum_b (1/|y_b|) sum_t A_{b,t} log pi_theta(y_t | s_t)``.

    Averaged over valid tokens within a response, then over responses -- the same
    reduction the project applies to its own surrogate, so the two arms' losses
    are on the same footing even though the objectives differ.

    ``importance_weights`` keeps the project's sampler-to-trainer correction
    ``exp(old_logp - rollout_logprob)``.  It is a numerical fix for the vLLM/HF
    head disagreement, *not* a clipping device: it stays in the gradient rather
    than being clamped, which is why it is applied multiplicatively here instead
    of through the log-space ratio the P2T surrogate needs.

    The advantage arrives already detached and already carrying the leave-one-out
    baseline (``reward.rloo_red_credit``), so the baseline is inside the
    estimator exactly as RLOO prescribes and never receives a gradient.
    """
    mask = _binary_mask(response_mask)
    if new_logp.shape != mask.shape or advantage.shape != mask.shape:
        raise ValueError("new_logp and advantage must have shape [B,T]")
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
    a = advantage.detach().float().masked_fill(~mask, 0)
    logp = new_logp.float().masked_fill(~mask, 0)
    if not torch.isfinite(a).all() or not torch.isfinite(logp).all():
        raise ValueError("Valid advantages and log probabilities must be finite")
    objective = a * logp
    if weights is not None:
        objective = objective * weights
    loss = -(objective / mask.sum(-1, keepdim=True).float()).mean(0).sum()
    if not torch.isfinite(loss):
        raise ValueError("Policy loss overflow")
    return loss


def kl_metric(base_logp: Tensor, new_logp: Tensor, response_mask: Tensor, *,
              importance_weights: Tensor | None = None) -> Tensor:
    """The project's KL estimator, reported as the ``kl_to_init`` metric.

    ``exp(d) - d - 1`` with ``d = log p_base - log p_new``, reduced as a
    *token-weighted* mean over the batch -- the reduction the sibling arm's inline
    computation uses, so the two arms' drift curves are directly comparable.

    This estimator is **not** what Eq. (8) subtracts: the reward uses the signed
    log ratio.  Keeping them separate is the point -- the metric stays comparable
    with the project and the reward stays as the paper writes it.
    """
    mask = _binary_mask(response_mask)
    if base_logp.shape != mask.shape or new_logp.shape != mask.shape:
        raise ValueError("log probabilities must have shape [B,T]")
    delta = (base_logp.float().masked_fill(~mask, 0)
             - new_logp.float().masked_fill(~mask, 0))
    values = torch.expm1(delta) - delta
    if importance_weights is not None:
        values = values * importance_weights.detach().float().masked_fill(~mask, 1)
    return values.masked_fill(~mask, 0).sum() / mask.sum().float()
