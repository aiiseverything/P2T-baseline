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
             tau: float) -> Credit:
    mask = _mask(response_mask)
    if direction.shape != mask.shape or advantage.shape != (mask.shape[0],):
        raise ValueError("Expected direction [B, T] and advantage [B]")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not torch.isfinite(direction[mask]).all() or not torch.isfinite(advantage).all():
        raise ValueError("Credit inputs must be finite on valid tokens")
    d = direction.float().masked_fill(~mask, 0)
    a = advantage.float()
    utility = a[:, None] * d / tau
    if not torch.isfinite(utility[mask]).all():
        raise ValueError("Credit utility overflow; inspect reward scale and tau")
    q = utility.masked_fill(~mask, -torch.inf).softmax(-1)
    weight = q * mask.sum(-1, keepdim=True)
    return Credit(a[:, None] * weight, d, weight)


@torch.no_grad()
def compute_credit(old_logits: Tensor, token_ids: Tensor, input_grads: Tensor,
                   rm_weight: Tensor, advantage: Tensor, reward_scale: Tensor,
                   response_mask: Tensor, tau: float, token_chunk_size: int = 128,
                   vocab_chunk_size: int = 8192) -> Credit:
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
    return allocate(direction, advantage, mask, tau)


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
