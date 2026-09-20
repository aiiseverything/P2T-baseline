"""Paper Eq. (3)-(5): token rewards and the GRPO token advantage.

    R^P2T_i = R + omega * R * exp(I_i) / sum_j exp(I_j)            (Eq. 3)
    A^hat_n = (R_n - mean(R)) / std(R)                             (Eq. 4)
    A~_n,i  = A^hat_n + alpha * R^P2T_n,i                          (Eq. 5)

``R`` in Eq. (3) is the reward model's raw scalar.  It is *not* standardised:
only Eq. (4)'s outcome reward is, and the paper calls R "a stable, minimum
reward signal to every token", which a zero-mean quantity could not be.  We
therefore pass the pristine RM score, never the length-shaped tensor and never
the per-group coupled degeneracy floor (see P2T_REPRO_NOTES.md).

``omega`` = 0.6 for every setting in the paper.  ``alpha`` = 0.1 for short-CoT
models such as the Qwen2.5 family and 1.0 for long-CoT models such as the
R1-Distill family; the actor here is a base Qwen3 generating with
``enable_thinking=False``, i.e. the short-CoT regime.

Two things the paper's text asserts that its own Eq. (3) does not deliver, and
which we reproduce rather than repair:

* Section 3.3.1 claims the token rewards sum to the sequence reward.  Expanding
  Eq. (3) gives sum_i R^P2T_i = sum_i (R + omega*R*p_i) = (N + omega) * R, since
  the shares p_i sum to one.  A convex combination R*(1-omega)/N + omega*R*p_i
  would sum to R; the paper does not write that.  ``test_reward.py`` pins the
  actual value.
* The exp() has no temperature, so the scale of a gradient inner product decides
  whether the softmax is nearly flat (omega-term becomes a per-response
  constant) or nearly one-hot.  We log the share distribution instead of tuning
  it, because the user asked for a strict reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .attribution import _binary_mask

P2T_OMEGA = 0.6
P2T_ALPHA_SHORT_COT = 0.1
P2T_ALPHA_LONG_COT = 1.0


@dataclass(frozen=True)
class Credit:
    """Token advantages plus diagnostics; mirrors ``vpo_rm.core.Credit``.

    ``advantage`` is the only field the policy loss reads.  ``direction`` keeps
    the invariant ``advantage == sequence_advantage + direction`` so the same
    "extra utility spread" gauge the VPO arms log stays comparable.
    """

    advantage: Tensor   # [B, T] A~, cached token advantages
    direction: Tensor   # [B, T] A~ - A^hat = alpha * R * (1 + omega * share)
    weight: Tensor      # [B, T] share scaled so the valid-token mean is one
    tau_used: Tensor | None = None  # unused by P2T; present for interface parity


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


@torch.no_grad()
def group_advantages(rewards: Tensor, group_ids: Tensor, eps: float = 1e-6, *,
                     std_floor: float = 0.0) -> tuple[Tensor, Tensor]:
    """Eq. (4), with the project's optional fixed floor on the group scale.

    Population standard deviation plus eps, or ``max(std, std_floor)`` when a
    positive floor is supplied.  ``vpo_rm.core.group_advantages`` implements the
    same two modes; ``tests/test_conformance.py`` asserts they agree.
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
def p2t_token_reward(attribution: Tensor, rewards: Tensor, response_mask: Tensor,
                     omega: float = P2T_OMEGA) -> tuple[Tensor, Tensor]:
    """Eq. (3) plus the normalised share, from one softmax evaluation.

    The max-subtraction is an exact per-response identity,
    ``exp(I-m)/sum exp(I-m) == exp(I)/sum exp(I)``, not a re-scaling: it is
    required because I is an unnormalised inner product of bf16-derived
    gradients and literal ``exp(I)`` overflows float32 past I > 88.7.
    ``test_numerics.py`` checks it against a float64 reference that does not
    subtract the max.
    """
    mask = _binary_mask(response_mask)
    omega = _finite_scalar(omega, "omega")
    if attribution.shape != mask.shape:
        raise ValueError("attribution must have shape [B, T] matching the response mask")
    if rewards.shape != (mask.shape[0],):
        raise ValueError("rewards must have shape [B]")
    if attribution.device != mask.device or rewards.device != mask.device:
        raise ValueError("Eq. (3) tensors must share one device")
    if not torch.isfinite(rewards).all():
        raise ValueError("Sequence rewards must be finite")
    if not torch.isfinite(attribution[mask]).all():
        raise ValueError("Attribution must be finite on valid response positions")

    values = attribution.float()
    row_max = values.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    shifted = (values - row_max).exp().masked_fill(~mask, 0.0)
    share = shifted / shifted.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    share = share.masked_fill(~mask, 0.0)
    token_reward = rewards.float()[:, None] + omega * rewards.float()[:, None] * share
    if not torch.isfinite(token_reward).all():
        raise ValueError("Eq. (3) token rewards are not finite")
    return token_reward, share


@torch.no_grad()
def p2t_token_advantage(token_reward: Tensor, advantages: Tensor, response_mask: Tensor,
                        alpha: float = P2T_ALPHA_SHORT_COT) -> Tensor:
    """Eq. (5).  ``advantages`` is A^hat from Eq. (4), one scalar per response."""
    mask = _binary_mask(response_mask)
    alpha = _finite_scalar(alpha, "alpha")
    if token_reward.shape != mask.shape:
        raise ValueError("token_reward must have shape [B, T] matching the response mask")
    if advantages.shape != (mask.shape[0],):
        raise ValueError("advantages must have shape [B]")
    if token_reward.device != mask.device or advantages.device != mask.device:
        raise ValueError("Eq. (5) tensors must share one device")
    if not torch.isfinite(advantages).all():
        raise ValueError("Sequence advantages must be finite")
    token_advantage = (advantages.float()[:, None] + alpha * token_reward.float())
    token_advantage = token_advantage.masked_fill(~mask, 0.0)
    if not torch.isfinite(token_advantage).all():
        raise ValueError("Eq. (5) token advantages are not finite")
    return token_advantage


@torch.no_grad()
def p2t_credit(rewards: Tensor, attribution: Tensor, advantages: Tensor,
               response_mask: Tensor, *, omega: float = P2T_OMEGA,
               alpha: float = P2T_ALPHA_SHORT_COT) -> Credit:
    """Eq. (3) then Eq. (5) on already-computed attributions."""
    token_reward, share = p2t_token_reward(attribution, rewards, response_mask, omega)
    advantage = p2t_token_advantage(token_reward, advantages, response_mask, alpha)
    mask = _binary_mask(response_mask)
    # A~ = A^hat + direction by construction, which is what the VPO arms' extra
    # utility spread gauge measures; keep that identity exactly.
    direction = (advantage - advantages.float()[:, None]).masked_fill(~mask, 0.0)
    # Same normalisation convention as vpo_rm.core.allocate: the per-response
    # valid-token mean of the diagnostic weight is one.  ESS/T is therefore
    # 1/(T * sum p^2) and reads the same way as the VPO arms' credit_ess_ratio:
    # near one means the weights are flat -- which for P2T means the attribution
    # softmax carried no token-level information at all -- and near 1/T means a
    # single token absorbed the whole share.
    weight = (share * mask.sum(-1, keepdim=True).float()).masked_fill(~mask, 0.0)
    return Credit(advantage, direction, weight, None)
