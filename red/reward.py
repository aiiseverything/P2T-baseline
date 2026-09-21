"""RED's reward redistribution (Eq. 6-8) and the RLOO advantage rule.

Paper: *RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward
Redistribution*, Jiahui Li et al., EMNLP 2025.  Equations are implemented as
written; where the paper contradicts itself or leaves a quantity undefined the
code follows the equations and the discrepancy is recorded in
``RED_REPRO_NOTES.md`` rather than repaired.

The pipeline, in the order the trainer runs it:

    Eq. (6)   r~_t = R_phi(x, y_<=t) - R_phi(x, y_<=t-1)      prefix difference
    Eq. (7)   r^_t = beta_c * r~_t + (1 - beta_c) * r_t       convex combination
    Eq. (8)   r^final_t = r^_t - beta * r^KL_t                KL folded in
    R3        A_{i,t} = r^final_{i,t} - b_i                   leave-one-out scalar

Eq. (8) puts the KL *inside* the reward, so the trainer carries no separate KL
loss term and the credit step must run after the reference log-probs are
available.  Both departures are recorded, not hidden.

The advantage rule is the one place the paper is silent: it gives a full PPO
recipe but no RLOO details at all.  See ``RED_REPRO_NOTES.md`` 2.2 for why the
two obvious readings of the RLOO baseline are either degenerate or unfaithful,
and why R3 is the one that keeps RLOO's own estimator intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# Stamped into every metric row, checkpoint manifest and credit dump so an
# artifact can never be read as a different redistribution rule.
RED_PROTOCOL = "prefix_difference_eq6"
RLOO_ADVANTAGE_RULE = "loo_scalar_baseline_r3"

# The paper's default.  beta_c = 1 removes the sparse sequence term entirely
# (Eq. 7), leaving the pure redistributed reward; Table 7 uses 1 everywhere
# except LLaMA3 on TL;DR, which uses 0.5.
RED_BETA_C_DEFAULT = 1.0

# Guards for the diagnostic share, mirroring the sibling arm's tiny-epsilon
# handling of a degenerate softmax.
_TINY = 1e-12


def _binary_mask(mask: Tensor, name: str = "response_mask") -> Tensor:
    """Validate a [B, T] 0/1 response mask with at least one valid token per row."""
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


def prefix_boundaries(mapped: Tensor, pooled: Tensor) -> tuple[Tensor, Tensor]:
    """The pair of reward-model positions Eq. (6) differences, per actor token.

    Returns ``(left [B, T], right [B, T])`` such that
    ``r~_t = R_phi(x, y_<=right_t) - R_phi(x, y_<=left_t)``.

    Differences are taken between *boundaries*, not between adjacent actor
    tokens, for two reasons.  ``mapped`` is byte-exact and therefore has holes at
    positions the reward model never saw, and -- more importantly -- the reward
    model pools at the last valid position of the canonical chat, which is a
    special token that is *always* unmapped.  Reading adjacent actor tokens would
    drop that trailing segment and the redistributed rewards would no longer sum
    to the sequence score.  Instead the chain is

        left_0 = first mapped position - 1   (= R_phi(x, empty), the DI term)
        left_t = right_{t-1}
        right_t = mapped_t,  except right at the last mapped token = pooled

    so ``sum_t r~_t = R_phi(x, y) - R_phi(x, empty)`` exactly, and every token the
    reward model never saw contributes zero rather than borrowing a neighbour's
    difference.
    """
    if mapped.ndim != 2 or pooled.shape != (mapped.shape[0],):
        raise ValueError("mapped must be [B, T] and pooled [B]")
    if mapped.dtype != torch.long or pooled.dtype != torch.long:
        raise ValueError("mapped and pooled must be int64")
    valid = mapped.ge(0)
    filled = mapped.clamp_min(0)
    inclusive = torch.cummax(filled.masked_fill(~valid, -1), dim=1).values
    # The last mapped position strictly before t.
    exclusive = torch.cat([torch.full_like(inclusive[:, :1], -1), inclusive[:, :-1]], dim=1)
    indexes = torch.arange(mapped.shape[1], device=mapped.device).expand_as(mapped)
    last_index = indexes.masked_fill(~valid, -1).amax(1)
    is_last = valid & indexes.eq(last_index[:, None])
    right = torch.where(is_last, pooled[:, None].expand_as(mapped), inclusive)
    left = torch.where(exclusive.ge(0), exclusive, mapped - 1)
    # Rows with no mapped token at all (or the leading unmapped run) fall back to
    # 0; those positions are masked out of the reward they feed, so the value only
    # has to be in range to keep the gather well defined.
    return left.clamp_min(0), right.clamp_min(0)


def prefix_token_rewards(right_scores: Tensor, left_scores: Tensor,
                         response_mask: Tensor) -> Tensor:
    """Eq. (6): ``r~_t = R_phi(x, y_<=t) - R_phi(x, y_<=t-1)``.

    ``right_scores`` and ``left_scores`` are the two prefix scores selected by
    :func:`prefix_boundaries`; the subtraction is the whole of Eq. (6).
    """
    mask = _binary_mask(response_mask)
    if right_scores.shape != mask.shape or left_scores.shape != mask.shape:
        raise ValueError("prefix scores must be [B, T] matching response_mask")
    diff = right_scores.float() - left_scores.float()
    if not torch.isfinite(diff).all():
        raise ValueError("Prefix difference is not finite")
    return diff.masked_fill(~mask, 0.0)


def sequence_reward_at_eos(rewards: Tensor, response_mask: Tensor) -> Tensor:
    """The sparse sequence reward of Eq. (3), placed at the final response token.

    Figure 4 writes ``reward_sequence[eos_idx] = reward_model_outputs[eos_idx]``;
    in actor-token space the final valid response token is that position.  Every
    other token gets 0, so Eq. (7) at ``beta_c = 1`` drops it entirely.
    """
    mask = _binary_mask(response_mask)
    if rewards.shape != (mask.shape[0],):
        raise ValueError("rewards must be [B] matching response_mask")
    indexes = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indexes.masked_fill(~mask, -1).amax(1)
    out = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    rows = torch.arange(mask.shape[0], device=mask.device)
    keep = last.ge(0)
    out[rows[keep], last[keep]] = rewards.float()[keep]
    return out.masked_fill(~mask, 0.0)


def red_convex_combination(token_rewards: Tensor, sequence_rewards: Tensor,
                           response_mask: Tensor,
                           beta_c: float = RED_BETA_C_DEFAULT) -> Tensor:
    """Eq. (7): ``r^_t = beta_c * r~_t + (1 - beta_c) * r_t``."""
    mask = _binary_mask(response_mask)
    if token_rewards.shape != mask.shape or sequence_rewards.shape != mask.shape:
        raise ValueError("token_rewards and sequence_rewards must be [B, T]")
    if not math.isfinite(beta_c) or not 0.0 <= beta_c <= 1.0:
        raise ValueError("beta_c must be finite and within [0, 1]")
    combined = beta_c * token_rewards.float() + (1.0 - beta_c) * sequence_rewards.float()
    return combined.masked_fill(~mask, 0.0)


def red_kl_reward(old_logp: Tensor, ref_logp: Tensor, response_mask: Tensor) -> Tensor:
    """Eq. (4) as Figure 4 actually computes it: the *signed* per-token log ratio.

    The paper writes ``KL(pi_theta || pi_ref)``, which is non-negative, but the
    pseudo-code is ``kl_divergence = log_probs - ref_log_probs`` and Eq. (5)
    subtracts it.  The code is authoritative and matches standard RLHF, so this
    returns ``log pi_old - log pi_ref`` -- signed, per token.  Recorded in
    ``RED_REPRO_NOTES.md`` 2.5.

    ``pi_old`` (the rollout policy), not the policy being updated, because
    Figure 4 computes ``log_probs`` immediately after generation.
    """
    mask = _binary_mask(response_mask)
    if old_logp.shape != mask.shape or ref_logp.shape != mask.shape:
        raise ValueError("log-probs must be [B, T] matching response_mask")
    kl = old_logp.float() - ref_logp.float()
    if not torch.isfinite(kl).all():
        raise ValueError("KL reward is not finite")
    return kl.masked_fill(~mask, 0.0)


def red_final_reward(token_rewards: Tensor, sequence_rewards: Tensor,
                     kl_reward: Tensor, response_mask: Tensor, *,
                     beta_c: float = RED_BETA_C_DEFAULT, beta: float) -> Tensor:
    """Eq. (8): ``r^final_t = r^_t - beta * r^KL_t``."""
    mask = _binary_mask(response_mask)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    combined = red_convex_combination(token_rewards, sequence_rewards, mask, beta_c)
    final = combined - beta * kl_reward.float()
    return final.masked_fill(~mask, 0.0)


def sequence_returns(shaped_rewards: Tensor, kl_reward: Tensor, response_mask: Tensor,
                     beta: float) -> Tensor:
    """RLOO's Eq. (3) sequence return, on the project's shaped reward.

    ``R(x, y) = r_shaped(x, y) - beta * sum_t r^KL_t``.  RLOO's baseline is a
    leave-one-out mean of *this* quantity, so the KL has to appear in it; the same
    KL already rides inside ``r^final`` (Eq. 8), and the decomposition in
    ``RED_REPRO_NOTES.md`` 2.2 is what keeps those two facts consistent.

    ``shaped_rewards`` is the project's sequence reward -- the raw reward model
    score minus the soft length window and the degeneracy floor -- so the length
    control keeps acting on the advantage through the baseline.
    """
    mask = _binary_mask(response_mask)
    if shaped_rewards.shape != (mask.shape[0],):
        raise ValueError("shaped_rewards must be [B] matching response_mask")
    if kl_reward.shape != mask.shape:
        raise ValueError("kl_reward must be [B, T] matching response_mask")
    return shaped_rewards.float() - beta * kl_reward.float().sum(-1)


def rloo_baseline(returns: Tensor, group_ids: Tensor) -> Tensor:
    """``b_i = 1/(k-1) * sum_{j != i} R_j`` -- RLOO's parameter-free baseline.

    Every group must hold at least two responses, otherwise there is nothing to
    leave out and the estimator is undefined.
    """
    if returns.ndim != 1 or group_ids.shape != returns.shape:
        raise ValueError("returns and group_ids must have shape [B]")
    if returns.device != group_ids.device:
        # The mask/select below would raise a cryptic torch error; say what broke
        # instead.  RED assembles credit on the actor device while the group ids
        # are built where the reward model runs, so this is the seam that fails.
        raise ValueError(f"returns are on {returns.device} but group_ids are on "
                         f"{group_ids.device}; the leave-one-out baseline indexes "
                         f"one with the other")
    if returns.is_complex() or not torch.isfinite(returns).all():
        raise ValueError("returns must be real and finite")
    baseline = torch.empty_like(returns, dtype=torch.float32)
    for group in group_ids.unique():
        selected = group_ids == group
        r = returns[selected].float()
        size = r.numel()
        if size < 2:
            raise ValueError("Each complete prompt group must contain at least two responses")
        # Leave-one-out mean, computed from the group sum: unbiased, and exact.
        baseline[selected] = (r.sum() - r) / (size - 1)
    return baseline


def group_sigma(rewards: Tensor, group_ids: Tensor) -> Tensor:
    """Population std of the shaped reward within each prompt group.

    RED does not standardise: RLOO's baseline is the leave-one-out mean, and
    dividing by the group spread would silently turn the estimator into GRPO with
    a leave-one-out numerator.  This is reported so the run carries the same
    ``group_sigma_*`` spread the sibling arms divide by, which is what keeps the
    arms' reward scales comparable on one plot.  Computed in float64 for the same
    reason the sibling arm does.
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


@dataclass(frozen=True)
class Credit:
    """The tensor bundle the trainer's loss and diagnostics consume.

    ``advantage`` is the only field the policy loss reads.  ``direction`` is the
    token-only part of it -- ``advantage == direction - b_i``, i.e. the R3
    baseline is the per-response constant that separates the two -- and
    ``weight`` is the positive-credit share rescaled so its valid-token mean is
    one, which is what makes the credit-concentration diagnostics comparable
    across response lengths and across arms.  ``tau_used`` exists only for
    interface parity with the VPO-RM arms and is always ``None`` here.
    """

    advantage: Tensor          # [B, T] A_{i,t}, the tensor the loss reads
    direction: Tensor          # [B, T] r^final, i.e. A_{i,t} + b_i
    weight: Tensor             # [B, T] positive-credit share, valid-token mean 1
    tau_used: Tensor | None = None


def credit_share(final_reward: Tensor, response_mask: Tensor) -> Tensor:
    """Fraction of the *positive* redistributed credit each token carries.

    RED has no softmax, so its token rewards are signed and do not form a
    distribution; the paper defines no analogue of the sibling arm's attribution
    share.  This is a project-level monitoring statistic, stated as such: the
    positive part of the reward is normalised to sum to one over the response, so
    a flat response reads as a uniform share and a concentrated one shows up as a
    spike.  Rows with no positive credit at all fall back to a uniform share
    rather than dividing by zero.
    """
    mask = _binary_mask(response_mask)
    if final_reward.shape != mask.shape:
        raise ValueError("final_reward must be [B, T] matching response_mask")
    positive = final_reward.float().clamp_min(0.0).masked_fill(~mask, 0.0)
    total = positive.sum(-1, keepdim=True)
    flat = total <= _TINY
    uniform = mask.float() / mask.sum(-1, keepdim=True).float()
    share = torch.where(flat.expand_as(positive), uniform, positive / total.clamp_min(_TINY))
    return share.masked_fill(~mask, 0.0)


def rloo_red_credit(final_reward: Tensor, baseline: Tensor,
                    response_mask: Tensor) -> Credit:
    """R3: ``A_{i,t} = r^final_{i,t} - b_i`` (``RED_REPRO_NOTES.md`` 2.2).

    The baseline is RLOO's leave-one-out sequence mean.  Subtracting one scalar
    per response splits the per-token advantage into RLOO's sequence-level term
    and RED's redistribution, which is what makes this an RLOO-RED arm rather
    than a new algorithm: replacing the sequence return with RED's return instead
    would cancel the dynamic-initialisation offset inside the leave-one-out mean
    and reproduce plain RLOO exactly.

    Note that a token the reward model never saw still receives a *non-zero*
    advantage: its redistributed reward is zero by construction, but the baseline
    (and the KL term inside ``r^final``) is not, so ``A = -b_i - beta*KL``.  That
    follows from the R3 rule as specified and is not an oversight -- only the
    redistribution itself is zero there.
    """
    mask = _binary_mask(response_mask)
    if final_reward.shape != mask.shape:
        raise ValueError("final_reward must be [B, T] matching response_mask")
    if baseline.shape != (mask.shape[0],):
        raise ValueError("baseline must be [B] matching response_mask")
    if final_reward.device != baseline.device:
        raise ValueError(f"final_reward is on {final_reward.device} but the baseline "
                         f"is on {baseline.device}")
    if not torch.isfinite(baseline).all():
        raise ValueError("baseline must be finite")
    advantage = (final_reward.float() - baseline.float()[:, None]).masked_fill(~mask, 0.0)
    share = credit_share(final_reward, mask)
    weight = (share * mask.sum(-1, keepdim=True).float()).masked_fill(~mask, 0.0)
    return Credit(advantage=advantage, direction=final_reward.float().masked_fill(~mask, 0.0),
                  weight=weight, tau_used=None)
