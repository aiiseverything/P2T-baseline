"""The arm's credit: GRPO's group-relative advantage, broadcast to every token.

There is deliberately very little here, and saying so is the point.  This arm has
**no token-level credit at all**: the reward is the frozen reward model's sequence
score, standardised within the prompt group, and the same scalar multiplies every
valid token of a response.  Everything the paper changes lives in
``he20/mask.py`` (which tokens survive) and ``he20/loss.py`` (that they are the
only ones the loss sees) -- not in how the advantage is built.  That is why the
paper's method is described as a mask on an unchanged algorithm, and it is why
``tests/he20/test_conformance.py`` can assert that this arm's loss equals the
project's GRPO loss exactly when the mask is switched off.

``group_advantages`` and ``group_sigma`` are the project's, mirrored from
``vpo_rm/core.py`` through the sibling arms so that an he20 run and a P2T or VPO-RM
run differ in the mask and nothing else.

The ``Credit`` dataclass exists for interface parity with the siblings, whose
trainers and diagnostics are written against it.  Two of its fields are degenerate
here by construction, and the module says so rather than inventing content:

* ``direction`` is **all zeros** -- it is the token-only part of the advantage, and
  this arm has none.
* ``weight`` is uniform (1 on every valid token), so the shared ``credit_ess_ratio``
  is exactly 1.  For the sibling arms a ratio near 1 means a flat attribution; here
  it means there is nothing to be flat, which is the truthful statement rather than
  a measured one.  The arm's own concentration diagnostic is
  ``entropy_top_kept_fraction``, which measures the mask, not the credit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .mask import _binary_mask

# Stamped into every metric row and checkpoint manifest so an artifact can never be
# read as a different method.  The mask's rule (threshold vs top-k) is a config
# choice and is stamped separately, because the protocol survives a change of rule.
HE20_PROTOCOL = "entropy_top_mask_eq6"


@dataclass(frozen=True)
class Credit:
    """The tensor bundle the trainer's loss and diagnostics consume.

    ``advantage`` is the only field the policy loss reads.  ``direction`` and
    ``weight`` are degenerate here -- see the module docstring -- and ``tau_used``
    exists only for interface parity with the VPO-RM arms and is always ``None``.
    """

    advantage: Tensor          # [B, T], the group advantage on every valid token
    direction: Tensor          # [B, T], always zero: this arm has no token-level term
    weight: Tensor             # [B, T], uniform: there is no credit to concentrate
    tau_used: Tensor | None = None


@torch.no_grad()
def group_advantages(rewards: Tensor, group_ids: Tensor, eps: float = 1e-6, *,
                     std_floor: float = 0.0) -> tuple[Tensor, Tensor]:
    """The project's group-relative advantage, with its optional fixed scale floor.

    Population standard deviation plus eps, or ``max(std, std_floor)`` when a
    positive floor is supplied.  ``vpo_rm.core.group_advantages`` implements the
    same two modes; ``tests/he20/test_conformance.py`` asserts they agree.
    """
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must have shape [B]")
    if not math.isfinite(eps) or eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite and eps finite and positive")
    if not math.isfinite(std_floor) or std_floor < 0:
        raise ValueError("std_floor must be finite and nonnegative")
    rewards = rewards.float()
    advantage = torch.empty_like(rewards)
    scale = torch.empty_like(rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        r = rewards[selected].double()
        if r.numel() < 2:
            raise ValueError("Each complete prompt group must contain at least two responses")
        std = r.std(correction=0)
        std = std.clamp_min(std_floor) if std_floor > 0 else std + eps
        stored = std.float()
        if not torch.isfinite(stored) or stored <= 0:
            raise ValueError("Group reward scale must be positive and representable in float32")
        advantage[selected] = ((r - r.mean()) / std).float()
        scale[selected] = stored
    return advantage, scale


@torch.no_grad()
def group_sigma(rewards: Tensor, group_ids: Tensor) -> Tensor:
    """Population std of the reward within each prompt group.

    Diagnostics only: this arm does not divide by it (``group_advantages`` does),
    but the sibling arms report the spread and the health checker compares arms on
    it, so it is computed the same way here.  Computed in float64 for the same
    reason the siblings do.
    """
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must have shape [B]")
    if rewards.is_complex() or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be real and finite")
    sigma = torch.empty_like(rewards, dtype=torch.float32)
    for group in group_ids.unique():
        selected = group_ids == group
        r = rewards[selected].double()
        if r.numel() < 2:
            raise ValueError("Each complete prompt group must contain at least two responses")
        sigma[selected] = r.std(correction=0).float()
    return sigma


def grpo_token_advantage(advantages: Tensor, response_mask: Tensor) -> Tensor:
    """Broadcast each response's scalar advantage onto its valid tokens.

    This is the whole credit assignment: one scalar per response, the same on every
    token.  The mask decides which of those tokens the loss will see; it does not
    change this tensor.
    """
    mask = _binary_mask(response_mask)
    if advantages.shape != (mask.shape[0],):
        raise ValueError("advantages must have shape [B] matching response_mask")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite")
    return advantages.float()[:, None].expand_as(mask).masked_fill(~mask, 0.0)


def he20_credit(rewards: Tensor, group_ids: Tensor, response_mask: Tensor, *,
                std_floor: float = 0.0) -> tuple[Credit, Tensor]:
    """The project's credit for this arm: group-relative advantage, broadcast.

    Returns ``(credit, scale)``.  ``std_floor`` is the project's
    ``advantage_std_floor_fraction * sigma0``, passed straight through so this arm's
    advantage scale matches its siblings'.

    The scale is returned rather than recomputed because it is the quantity the
    standardisation actually divided by -- and the *floored* one, since that is what
    ``group_advantages`` applies.  The trainer reports it as the shared
    ``group_sigma_*`` so the key means the same thing here as in VPO and P2T, which
    apply the same floor; RED reports the unfloored spread, which is a different
    number under the same name.
    """
    mask = _binary_mask(response_mask)
    if rewards.shape != (mask.shape[0],):
        raise ValueError("rewards must have shape [B] matching response_mask")
    advantages, scale = group_advantages(rewards, group_ids, std_floor=std_floor)
    advantage = grpo_token_advantage(advantages, mask)
    zeros = torch.zeros_like(advantage)
    # Uniform weight, valid-token mean one -- the same normalisation convention the
    # siblings use, applied to a flat share because there is no token-level credit.
    weight = mask.float()
    credit = Credit(advantage=advantage, direction=zeros, weight=weight, tau_used=None)
    return credit, scale
