"""The ordinary-softmax ST proxy and response-normalized GRPO objective."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class Credit:
    advantage: Tensor  # [B, T], cached token advantages
    direction: Tensor  # [B, T], d_t
    weight: Tensor     # [B, T], valid-token mean is one
    tau_used: Tensor | None = None  # [B], per-response adaptive temperature


def _mask(mask: Tensor) -> Tensor:
    if mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("response_mask must be a binary [B, T] tensor")
    mask = mask.bool()
    if not mask.any(-1).all():
        raise ValueError("Each response must have at least one valid token")
    return mask


@torch.no_grad()
def group_advantages(rewards: Tensor, group_ids: Tensor, eps: float = 1e-6):
    """Population std + eps; supply whole prompt groups, including across ranks."""
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must have shape [B]")
    if eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite and eps positive")
    rewards = rewards.float()
    advantage, scale = torch.empty_like(rewards), torch.empty_like(rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        r = rewards[selected]
        if r.numel() < 2:
            raise ValueError("Each complete prompt group must contain at least two responses")
        std = r.std(correction=0) + eps
        advantage[selected] = (r - r.mean()) / std
        scale[selected] = std
    return advantage, scale


@torch.no_grad()
def allocate(direction: Tensor, advantage: Tensor, response_mask: Tensor,
             tau: float, credit_lambda: float = 2.0) -> Credit:
    """Allocate the sequence advantage over tokens with a scale-free softmax in
    a lambda band.

    Plan B (tau失配与修复方案.md): the direction signal d_t is standardized per
    response before the softmax, so the temperature is dimensionless; degenerate
    directions fall back to uniform (plain GRPO) weights, and ``a/sigma``
    reward-unit normalization cancels exactly.

    Lambda band (lambda区间带方案.md): credit concentration acts as an
    effective-lr multiplier on hot tokens (measured 14-19x in p9g, which
    entropy-collapsed).  A per-response adaptive temperature is solved by
    bisection so that no token's weight exceeds ``credit_lambda`` times the
    uniform level (and stays above 1/lambda): the multiplier is capped by
    construction, smoothly (no clamp discontinuities).  ``credit_lambda = 1``
    recovers exact GRPO weights, making VPO a one-parameter interpolation
    family over credit concentration.  The adaptive temperature only tightens:
    when the natural softmax at ``tau`` already satisfies the band it is kept.
    """
    mask = _mask(response_mask)
    if direction.shape != mask.shape or advantage.shape != (mask.shape[0],):
        raise ValueError("Expected direction [B, T] and advantage [B]")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(credit_lambda) or credit_lambda < 1:
        raise ValueError("credit_lambda must be finite and at least one")
    d = direction.float().masked_fill(~mask, 0)
    a = advantage.float()
    counts = mask.sum(-1, keepdim=True).float()
    ones = torch.ones_like(a)
    if credit_lambda <= 1.0:
        weight = torch.ones_like(d) * counts / counts
        return Credit(a[:, None] * weight, d, weight, ones * tau)
    mean = d.sum(-1, keepdim=True) / counts
    var = d.square().sum(-1, keepdim=True) / counts - mean.square()
    std = var.clamp_min(0).sqrt()
    # Relative floor: if the spread is below 0.1% of the typical magnitude the
    # signal is numerical noise; treat the response as having nothing to
    # allocate by and let the softmax return uniform weights.
    floor = 1e-3 * d.abs().sum(-1, keepdim=True) / counts + torch.finfo(torch.float32).tiny
    u = a[:, None] * (d - mean) / (std + floor)
    if not torch.isfinite(u[mask]).all():
        raise ValueError("Credit utility overflow; inspect advantage and tau")

    def weights_at(tau_per_response):
        q = (u / tau_per_response[:, None]).masked_fill(~mask, -torch.inf).softmax(-1)
        return q * counts

    # Bisection in log space for the minimum tau satisfying max(w) <= lambda.
    log_lo = torch.full_like(a, math.log(1e-2))
    log_hi = torch.full_like(a, math.log(1e2))
    for _ in range(40):
        log_mid = (log_lo + log_hi) / 2
        over = weights_at(torch.exp(log_mid)).max(-1).values > credit_lambda
        log_lo = torch.where(over, log_mid, log_lo)
        log_hi = torch.where(over, log_hi, log_mid)
    tau_used = torch.maximum(torch.exp(log_hi), torch.as_tensor(tau, device=a.device))
    weight = weights_at(tau_used)
    # The lower band (w >= 1/lambda) binds only for extreme negative outliers;
    # flattening further restores it at the cost of a looser upper band.
    for _ in range(8):
        low = weight.masked_fill(~mask, torch.inf).min(-1).values < 1.0 / credit_lambda
        if not bool(low.any()):
            break
        tau_used = torch.where(low, tau_used * 2.0, tau_used)
        weight = weights_at(tau_used)
    weight = weight.masked_fill(~mask, 0)
    return Credit(a[:, None] * weight, d, weight, tau_used)


@torch.no_grad()
def guard_degenerate_rewards(rewards: Tensor, lengths: Tensor, group_ids: Tensor,
                             min_length: int = 8, penalty: float = 1.0,
                             also_floor: Tensor | None = None):
    """Reward guard against degenerate (under-length or flagged) responses.

    A response shorter than ``min_length`` tokens (or selected by the boolean
    ``also_floor`` mask, e.g. truncated/overlong responses) cannot win its
    prompt group: its reward is replaced by the group minimum minus
    ``penalty``, so within GRPO's group-relative advantage it always ranks
    last.  Groups whose every response is flagged collapse to equal rewards
    (zero advantage, no signal) instead of reinforcing the flagged mode.
    Returns the patched rewards and the number of patched responses.
    """
    if rewards.ndim != 1 or lengths.shape != rewards.shape or group_ids.shape != rewards.shape:
        raise ValueError("rewards, lengths and group_ids must share shape [B]")
    if min_length < 1 or penalty < 0:
        raise ValueError("min_length must be positive and penalty non-negative")
    if also_floor is not None and (also_floor.shape != rewards.shape or also_floor.dtype != torch.bool):
        raise ValueError("also_floor must be a boolean tensor of shape [B]")
    rewards = rewards.clone()
    flagged = lengths < min_length
    if also_floor is not None:
        flagged = flagged | also_floor
    if not bool(flagged.any()):
        return rewards, 0
    for group in group_ids[flagged].unique():
        selected = group_ids == group
        floor = rewards[selected].min() - penalty
        rewards[selected & flagged] = floor
    return rewards, int(flagged.sum())


@torch.no_grad()
def compute_credit(old_logits: Tensor, token_ids: Tensor, input_grads: Tensor,
                   rm_weight: Tensor, advantage: Tensor, reward_scale: Tensor,
                   response_mask: Tensor, tau: float, token_chunk_size: int = 128,
                   vocab_chunk_size: int = 8192, credit_lambda: float = 2.0) -> Credit:
    """Exact full-vocabulary d_t with token/vocabulary blocking.

    old_logits [B,T,V] comes from the rollout policy at fixed hard prefixes.
    input_grads [B,T,D] already corresponds to the sampled token's RM position.
    rm_weight [V,D] uses precisely the Actor output token-ID ordering.
    Temporary vocabulary tensors are at most token_chunk_size*vocab_chunk_size.
    The caller supplies the complete logits and an unsharded embedding weight.
    """
    mask = _mask(response_mask)
    if old_logits.ndim != 3 or old_logits.shape[:2] != mask.shape:
        raise ValueError("old_logits must have shape [B, T, V]")
    if token_ids.shape != mask.shape or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be int64 [B, T]")
    B, T, V = old_logits.shape
    if rm_weight.ndim != 2 or rm_weight.shape[0] != V:
        raise ValueError("Actor vocabulary and RM embedding rows must match")
    if input_grads.shape != (B, T, rm_weight.shape[1]):
        raise ValueError("input_grads must have shape [B, T, RM embedding dimension]")
    if advantage.shape != (B,) or reward_scale.shape != (B,):
        raise ValueError("advantage and reward_scale must have shape [B]")
    if token_chunk_size < 1 or vocab_chunk_size < 1:
        raise ValueError("Chunk sizes must be positive")
    tensors = (token_ids, input_grads, rm_weight, advantage, reward_scale, mask)
    if any(x.device != old_logits.device for x in tensors):
        raise ValueError("Credit tensors must be on the same device")
    if not (torch.isfinite(reward_scale) & (reward_scale > 0)).all():
        raise ValueError("reward_scale must be finite and positive")
    ids = token_ids[mask]
    if ((ids < 0) | (ids >= V)).any():
        raise ValueError("Valid sampled token IDs must lie inside the shared vocabulary")
    rows, positions = mask.nonzero(as_tuple=True)
    direction = torch.zeros((B, T), device=old_logits.device, dtype=torch.float32)
    # Compute moments of p and v. E_p[C]=0 gives the exact scalar contraction:
    # d*sigma = p_a*(v_a-mu) - sum(p^2*v) + mu*sum(p^2).
    with torch.autocast(device_type=old_logits.device.type, enabled=False):
        for start in range(0, rows.numel(), token_chunk_size):
            r = rows[start:start + token_chunk_size]
            t = positions[start:start + token_chunk_size]
            a = token_ids[r, t]
            if not torch.isfinite(old_logits[r, t, a]).all():
                raise ValueError("Sampled tokens must belong to the policy output support")
            f = input_grads[r, t].float()
            log_z = torch.full((r.numel(),), -torch.inf, device=r.device)
            for lo in range(0, V, vocab_chunk_size):
                z = old_logits[r, t, lo:lo + vocab_chunk_size].float()
                log_z = torch.logaddexp(log_z, z.logsumexp(-1))
            mu, p2, p2v = (torch.zeros_like(log_z) for _ in range(3))
            for lo in range(0, V, vocab_chunk_size):
                z = old_logits[r, t, lo:lo + vocab_chunk_size].float()
                p = (z - log_z[:, None]).exp()
                v = f @ rm_weight[lo:lo + vocab_chunk_size].float().T
                mu += (p * v).sum(-1)
                p2 += p.square().sum(-1)
                p2v += (p.square() * v).sum(-1)
            pa = (old_logits[r, t, a].float() - log_z).exp()
            va = (f * rm_weight[a].float()).sum(-1)
            direction[r, t] = (pa * (va - mu) - p2v + mu * p2) / reward_scale[r]
    return allocate(direction, advantage, mask, tau, credit_lambda=credit_lambda)


def grpo_policy_loss(new_logp: Tensor, old_logp: Tensor, token_advantage: Tensor,
                     response_mask: Tensor, clip_eps: float = 0.2) -> Tensor:
    """Negative clipped objective: average tokens per response, then responses.

    Add the existing trainer's KL term with its original estimator and reduction.
    """
    mask = _mask(response_mask)
    if any(x.shape != mask.shape for x in (new_logp, old_logp, token_advantage)):
        raise ValueError("log probabilities, advantage and mask must have shape [B,T]")
    if not 0 < clip_eps < 1:
        raise ValueError("clip_eps must lie in (0,1)")
    # Mask before exponentiation so padded NaNs cannot affect backward.
    log_ratio = (new_logp.float().masked_fill(~mask, 0)
                 - old_logp.detach().float().masked_fill(~mask, 0))
    ratio = log_ratio.exp()
    a = token_advantage.detach().float().masked_fill(~mask, 0)
    objective = torch.minimum(ratio * a, ratio.clamp(1-clip_eps, 1+clip_eps) * a)
    return -(objective.sum(-1) / mask.sum(-1)).mean()
