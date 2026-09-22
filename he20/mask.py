"""Eq. (6): keep only the highest-entropy tokens of the population.

Paper: *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
Reinforcement Learning for LLM Reasoning*, Wang et al., NeurIPS 2025
(arXiv 2506.01939).  The paper's Eq. (6) changes its base algorithm in exactly two
places: the per-token surrogate is multiplied by ``I[H_t^i >= tau_rho^B]``, and the
token-count normaliser is restricted to the same kept tokens -- a masked mean over
the kept tokens only.  ``tau_rho^B`` is "the entropy threshold within the batch B
such that only tokens with ``H_t^i >= tau_rho^B``, comprising the top-rho fraction
of all tokens in the batch, are used to compute the gradient" (p.7).

Two things the paper leaves open, and what this module does about them.  Both are
recorded in ``HE20_REPRO_NOTES.md`` rather than settled silently here:

* **Threshold or top-k.**  The paper defines the mask by the *threshold* above, so
  ties at ``tau`` are all kept and the retained fraction can exceed ``rho``.  The
  authors' own reference implementation instead sorts --
  ``top_k = max(1, int(n * top_ratio + 0.9999))`` then ``torch.topk`` -- keeping
  exactly ``ceil(rho*n)`` and breaking ties by index.  The two differ only on ties.
  This arm follows the **paper**, because it reproduces the paper; the reference
  implementation's top-k is reachable through ``rule="topk"`` and is recorded as a
  deviation.  With continuous float entropies ties are rare, so the choice is
  numerically near-moot -- but it is stated, not assumed.
* **The population.**  The paper says "within each batch" (§5.2) and also "within
  each (micro-)batch B" (p.7); with its batch 512 against mini-batch 32 those differ
  16x in pool size, and it never reconciles them.  This module pools over **exactly
  the rows it is handed**, so the caller decides the population explicitly and the
  choice is visible at the call site rather than implicit in a default.  The trainer
  passes the whole optimizer minibatch (see ``trainer.py``), which is the set whose
  gradients are averaged together; pooling over a single response instead would
  silently turn the paper's batch-level mask into a per-response top-20%, which is a
  different method.

The entropy itself is the per-token Shannon entropy of the policy's next-token
distribution over the supported vocabulary, from the same logits the loss forward
already builds -- i.e. from the policy being updated, which is what Eq. (1) means
and what the reference implementation does (``_forward_micro_batch(...,
calculate_entropy=True)``).  ``he20/policy.py``'s ``response_entropy`` computes it.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

# The paper's main setting: rho = 20% (Eq. (6), §5.2).
ENTROPY_TOP_RATIO_DEFAULT = 0.2

# "threshold" is the paper's Eq. (6); "topk" is the reference implementation's rule.
# See the module docstring -- the difference is confined to ties.
ENTROPY_TOP_RULES = ("threshold", "topk")


def _binary_mask(mask: Tensor, name: str = "response_mask") -> Tensor:
    """Validate a [B, T] 0/1 mask with at least one valid token per row.

    Same contract as the sibling arms' copy, so a caller can move between arms
    without the validation changing under them.
    """
    if not isinstance(mask, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be [B, T]")
    if mask.dtype == torch.bool:
        binary = mask
    else:
        if not torch.logical_or(mask == 0, mask == 1).all():
            raise ValueError(f"{name} must be binary")
        binary = mask.bool()
    if not binary.any(-1).all():
        raise ValueError(f"{name} must leave at least one valid token per row")
    return binary


def _validate(entropy: Tensor, response_mask: Tensor, top_ratio: float, rule: str):
    mask = _binary_mask(response_mask)
    if not isinstance(entropy, Tensor) or entropy.shape != mask.shape:
        raise ValueError("entropy must be a tensor of shape [B, T] matching response_mask")
    if not entropy.is_floating_point():
        raise ValueError("entropy must be floating point")
    if rule not in ENTROPY_TOP_RULES:
        raise ValueError(f"rule must be one of {ENTROPY_TOP_RULES}, not {rule!r}")
    if not math.isfinite(top_ratio) or not 0.0 < top_ratio <= 1.0:
        raise ValueError("top_ratio must lie in (0, 1]")
    values = entropy.detach().float()[mask]
    if values.numel() == 0:
        raise ValueError("The selection population is empty; there is nothing to rank")
    if not torch.isfinite(values).all():
        raise ValueError("Valid response entropies must be finite")
    return mask, values


def entropy_threshold(entropy: Tensor, response_mask: Tensor,
                      top_ratio: float = ENTROPY_TOP_RATIO_DEFAULT, *,
                      rule: str = "threshold") -> Tensor:
    """Scalar ``tau_rho^B`` for this population, under the chosen rule.

    ``threshold``: the ``1 - top_ratio`` quantile of the pooled response entropies.
    ``topk``: the ``k``-th largest with ``k = ceil(top_ratio * n)``, which is the
    smallest value the reference implementation would keep.
    """
    mask, values = _validate(entropy, response_mask, top_ratio, rule)
    if rule == "threshold":
        return torch.quantile(values, 1.0 - top_ratio)
    count = values.numel()
    k = max(1, math.ceil(count * top_ratio))
    return torch.topk(values, k=k, largest=True).values.min()


def entropy_top_mask(entropy: Tensor, response_mask: Tensor,
                     top_ratio: float = ENTROPY_TOP_RATIO_DEFAULT, *,
                     rule: str = "threshold") -> Tensor:
    """``I[H_t^i >= tau_rho^B]`` as a bool [B, T], pooled over the rows given.

    Outside the response mask the result is False, so the caller can multiply it
    straight into the loss mask.  The population is every valid token of every row
    passed in; pass the rows you mean to rank together.
    """
    mask, values = _validate(entropy, response_mask, top_ratio, rule)
    threshold = entropy_threshold(entropy, response_mask, top_ratio, rule=rule)
    if rule == "threshold":
        keep = entropy.detach().float() >= threshold
    else:
        # Keep exactly the reference implementation's selection: its top-k indices,
        # mapped back to positions in the pooled vector.  Reproducing the index
        # choice matters on ties, where a threshold would keep more.
        count = values.numel()
        k = max(1, math.ceil(count * top_ratio))
        chosen = torch.topk(values, k=k, largest=True).indices
        flat = torch.zeros(count, dtype=torch.bool, device=values.device)
        flat[chosen] = True
        keep = torch.zeros_like(mask)
        keep[mask] = flat
    keep = keep & mask
    if not keep.any():
        raise ValueError("The selection kept no token; the loss would divide by zero")
    return keep


def kept_fraction(mask: Tensor, response_mask: Tensor) -> Tensor:
    """Fraction of valid response tokens the mask kept, for the metrics row."""
    valid = _binary_mask(response_mask)
    if mask.shape != valid.shape:
        raise ValueError("mask must be [B, T] matching response_mask")
    return (mask.bool() & valid).sum().float() / valid.sum().float()
