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
def group_advantages(rewards: Tensor, group_ids: Tensor, eps: float = 1e-6,
                     *, std_floor: float = 0.):
    """Group advantages using std+eps, or max(std, a positive fixed floor).

    Supply complete prompt groups, including across ranks. A positive floor
    replaces the additive epsilon; zero preserves the original normalization.
    """
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must have shape [B]")
    if not math.isfinite(eps) or eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite and eps finite and positive")
    if not math.isfinite(std_floor) or std_floor < 0:
        raise ValueError("std_floor must be finite and nonnegative")
    rewards = rewards.float()
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be representable in float32")
    advantage, scale = torch.empty_like(rewards), torch.empty_like(rewards)
    for group in group_ids.unique():
        selected = group_ids == group
        # Reduction in double avoids overflowing a finite FP32 group's mean
        # and variance, including constant groups near the FP32 maximum.
        r = rewards[selected].double()
        if r.numel() < 2:
            raise ValueError("Each complete prompt group must contain at least two responses")
        std = r.std(correction=0)
        std = std.clamp_min(std_floor) if std_floor > 0 else std + eps
        stored_std = std.float()
        if not torch.isfinite(stored_std) or stored_std <= 0:
            raise ValueError("Group reward scale must be positive and representable in float32")
        advantage[selected] = ((r - r.mean()) / std).float()
        scale[selected] = stored_std
    return advantage, scale


@torch.no_grad()
def allocate(direction: Tensor, advantage: Tensor, response_mask: Tensor,
             tau: float, credit_lambda: float = 2.0,
             token_ids: Tensor | None = None,
             freeze_stop_tokens: bool = False,
             freeze_structural: bool = False,
             stop_token_ids: tuple[int, ...] | None = None,
             structural_token_ids: tuple[int, ...] | None = None,
             fixed_weight_mask: Tensor | None = None) -> Credit:
    """Allocate advantage within the final [1/lambda, lambda] weight band.

    Frozen positions receive exactly one unit of credit. Standardization and
    softmax use only the remaining positions, whose budget is their count.
    ``fixed_weight_mask`` adds valid positions without an exact RM mapping to
    those frozen by the tokenizer-derived stop/structural policy.
    Structural freezing requires IDs verified against the actual tokenizer.
    The default stop IDs only preserve direct callers using the Qwen vocabulary;
    production callers should pass both sets from ``token_policy``.
    """
    mask = _mask(response_mask)
    if direction.shape != mask.shape or advantage.shape != (mask.shape[0],):
        raise ValueError("Expected direction [B, T] and advantage [B]")
    if any(x.device != direction.device for x in (advantage, mask)):
        raise ValueError("Credit tensors must be on the same device")
    if fixed_weight_mask is not None:
        if (not isinstance(fixed_weight_mask, Tensor) or fixed_weight_mask.shape != mask.shape
                or fixed_weight_mask.dtype != torch.bool or fixed_weight_mask.device != mask.device):
            raise ValueError("fixed_weight_mask must be boolean [B,T] on the credit device")
        if (fixed_weight_mask & ~mask).any():
            raise ValueError("fixed_weight_mask may select valid response tokens only")
    if not math.isfinite(tau) or tau <= 0 or tau > torch.finfo(torch.float32).max:
        raise ValueError("tau must be finite, positive and representable in float32")
    if not math.isfinite(credit_lambda) or credit_lambda < 1:
        raise ValueError("credit_lambda must be finite and at least one")
    if freeze_stop_tokens or freeze_structural:
        if token_ids is None:
            raise ValueError("freeze_stop_tokens/freeze_structural requires token_ids")
        if token_ids.shape != mask.shape or token_ids.dtype != torch.long or token_ids.device != mask.device:
            raise ValueError("token_ids must be int64 [B,T] on the credit device")
    if freeze_structural and structural_token_ids is None:
        raise ValueError("freeze_structural requires verified structural_token_ids")
    d = direction.float().masked_fill(~mask, 0)
    a = advantage.float()
    if not torch.isfinite(d).all() or not torch.isfinite(a).all():
        raise ValueError("Valid credit directions and advantages must be finite")
    tau_used = torch.full_like(a, max(tau, torch.finfo(torch.float32).tiny))
    # A band narrower than float32's resolution only admits the exactly
    # uniform representable solution; do not search towards overflowing tau.
    if credit_lambda - 1.0 < torch.finfo(torch.float32).eps:
        weight = mask.float()
        return Credit(a[:, None] * weight, d, weight, tau_used)

    frozen = torch.zeros_like(mask) if fixed_weight_mask is None else fixed_weight_mask.clone()
    if freeze_stop_tokens or freeze_structural:
        stop_ids = (151643, 151645) if stop_token_ids is None else stop_token_ids
        ids = tuple(stop_ids) + (tuple(structural_token_ids) if freeze_structural else ())
        if ids:
            frozen |= torch.isin(token_ids, torch.as_tensor(ids, device=mask.device)) & mask
    free = mask & ~frozen
    counts = free.sum(-1, keepdim=True).float()
    divisor = counts.clamp_min(1)
    # Scaling before the centered variance avoids both square overflow and
    # catastrophic cancellation for almost-constant direction values.
    free_d = d.masked_fill(~free, 0)
    magnitude = free_d.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = free_d / magnitude
    mean = scaled.sum(-1, keepdim=True) / divisor
    centered = (scaled - mean).masked_fill(~free, 0)
    std = (centered.square().sum(-1, keepdim=True) / divisor).sqrt()
    floor = 1e-3 * scaled.abs().sum(-1, keepdim=True) / divisor + torch.finfo(torch.float32).tiny
    u = a[:, None] * (centered / (std + floor))
    if not torch.isfinite(u).all():
        raise ValueError("Credit utility overflow; inspect advantage")
    uniform = (u == 0).all(-1)
    maximum = u.masked_fill(~free, -torch.inf).amax(-1, keepdim=True)
    maximum = maximum.masked_fill(counts == 0, 0)
    shifted = u - maximum

    def weights_at(temperature):
        # Subtract max before division so tiny temperatures can only produce
        # harmless negative infinity, never positive-infinity softmax NaNs.
        z = (shifted / temperature[:, None]).masked_fill(~free, -torch.inf)
        z = z.masked_fill(counts == 0, 0)
        weight = (z.softmax(-1) * counts).masked_fill(~free, 0)
        weight = weight.masked_fill(frozen, 1.)
        return torch.where(uniform[:, None], mask.float(), weight)

    def feasible(weight):
        return ((weight.max(-1).values <= credit_lambda)
                & (weight.masked_fill(~mask, torch.inf).min(-1).values >= 1. / credit_lambda)
                & torch.isfinite(weight).all(-1))

    # Bracket an actually feasible temperature. A fixed upper endpoint misses
    # legal long-response outliers; both sides of the band must be checked.
    lower = tau_used.clone()
    upper = tau_used.clone()
    needs_raise = ~feasible(weights_at(upper))
    search = needs_raise.clone()
    while bool(needs_raise.any()):
        doubled = upper * 2.
        if not torch.isfinite(doubled[needs_raise]).all():
            raise ValueError("Credit temperature overflow before a feasible band was found")
        lower = torch.where(needs_raise, upper, lower)
        upper = torch.where(needs_raise, doubled, upper)
        needs_raise = ~feasible(weights_at(upper))
    if bool(search.any()):
        for _ in range(40):
            middle = lower + (upper - lower) / 2.
            accepted = feasible(weights_at(middle))
            upper = torch.where(search & accepted, middle, upper)
            lower = torch.where(search & ~accepted, middle, lower)
    tau_used = upper
    weight = weights_at(tau_used)
    if not bool(feasible(weight).all()):
        raise ValueError("Final credit weights violate the lambda band")
    if not torch.allclose(weight.sum(-1), mask.sum(-1).float(), rtol=2e-6, atol=2e-6):
        raise ValueError("Final credit weights do not preserve the response budget")
    token_advantage = a[:, None] * weight
    if not torch.isfinite(token_advantage).all():
        raise ValueError("Token advantage overflow")
    return Credit(token_advantage, d, weight, tau_used)


RANDOM_CREDIT_SOURCES = ("random_direction", "random_band")
GRADIENT_CREDIT_SOURCES = ("rm_gradient", "shuffle", "norm_product")
CREDIT_SOURCES = GRADIENT_CREDIT_SOURCES + RANDOM_CREDIT_SOURCES


@torch.no_grad()
def shuffle_credit(credit: Credit, response_mask: Tensor, *, generator: torch.Generator) -> Credit:
    """Permute allocated weights within each complete response, excluding padding.

    All valid positions participate, including stops/structure/unmapped tokens:
    their unit weights were fixed during allocation, but their locations are
    deliberately randomized by this control. Preserve the weight multiset and
    the signed token-advantage multiset exactly. Directions stay at source tokens.
    """
    mask = _mask(response_mask)
    if credit.weight.shape != mask.shape or credit.advantage.shape != mask.shape:
        raise ValueError("Credit and response mask must share shape [B,T]")
    weight = torch.zeros_like(credit.weight)
    advantage = torch.zeros_like(credit.advantage)
    for row in range(mask.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        order = torch.randperm(positions.numel(), device=mask.device, generator=generator)
        weight[row, positions] = credit.weight[row, positions[order]]
        advantage[row, positions] = credit.advantage[row, positions[order]]
    return Credit(advantage, credit.direction, weight, credit.tau_used)


def _frozen_positions(mask: Tensor, token_ids: Tensor | None, freeze_stop_tokens: bool,
                      freeze_structural: bool, stop_token_ids, structural_token_ids,
                      fixed_weight_mask: Tensor | None) -> Tensor:
    """Positions whose credit stays exactly one; mirrors ``allocate`` without touching it."""
    if fixed_weight_mask is not None:
        if (not isinstance(fixed_weight_mask, Tensor) or fixed_weight_mask.shape != mask.shape
                or fixed_weight_mask.dtype != torch.bool or fixed_weight_mask.device != mask.device):
            raise ValueError("fixed_weight_mask must be boolean [B,T] on the credit device")
        if (fixed_weight_mask & ~mask).any():
            raise ValueError("fixed_weight_mask may select valid response tokens only")
    if freeze_stop_tokens or freeze_structural:
        if token_ids is None:
            raise ValueError("freeze_stop_tokens/freeze_structural requires token_ids")
        if token_ids.shape != mask.shape or token_ids.dtype != torch.long or token_ids.device != mask.device:
            raise ValueError("token_ids must be int64 [B,T] on the credit device")
    if freeze_structural and structural_token_ids is None:
        raise ValueError("freeze_structural requires verified structural_token_ids")
    frozen = torch.zeros_like(mask) if fixed_weight_mask is None else fixed_weight_mask.clone()
    if freeze_stop_tokens or freeze_structural:
        stop_ids = (151643, 151645) if stop_token_ids is None else stop_token_ids
        ids = tuple(stop_ids) + (tuple(structural_token_ids) if freeze_structural else ())
        if ids:
            frozen |= torch.isin(token_ids, torch.as_tensor(ids, device=mask.device)) & mask
    return frozen


@torch.no_grad()
def random_credit(advantage: Tensor, response_mask: Tensor, tau: float,
                  credit_lambda: float = 2.0, *, generator: torch.Generator,
                  source: str = "random_direction",
                  token_ids: Tensor | None = None,
                  freeze_stop_tokens: bool = False,
                  freeze_structural: bool = False,
                  stop_token_ids: tuple[int, ...] | None = None,
                  structural_token_ids: tuple[int, ...] | None = None,
                  fixed_weight_mask: Tensor | None = None) -> Credit:
    """Ablation control: credit inside the same lambda band without reward-model information.

    ``random_direction`` replaces the RM-gradient direction d_t by i.i.d. standard
    normal noise and runs the unchanged allocator: per-response standardization,
    adaptive tau, the [1/lambda, lambda] band, frozen stop/structural/unmapped
    tokens at exactly one, and the preserved response budget. ``random_band``
    draws free-token weights uniformly inside the band and projects them onto
    the response budget by alternating budget shifts with band clipping. Both
    keep the sequence advantage's sign and mean; the returned direction is the
    noise (or zeros) so credit dumps document the control.
    """
    if source not in RANDOM_CREDIT_SOURCES:
        raise ValueError(f"random credit source must be one of {RANDOM_CREDIT_SOURCES}")
    mask = _mask(response_mask)
    if not isinstance(generator, torch.Generator) or generator.device != mask.device:
        raise ValueError("random credit requires a torch.Generator on the credit device")
    if advantage.shape != (mask.shape[0],) or advantage.device != mask.device:
        raise ValueError("Expected advantage [B] on the credit device")
    if source == "random_direction":
        direction = torch.randn(mask.shape, generator=generator, device=mask.device, dtype=torch.float32)
        return allocate(direction, advantage, mask, tau, credit_lambda=credit_lambda,
                        token_ids=token_ids, freeze_stop_tokens=freeze_stop_tokens,
                        freeze_structural=freeze_structural, stop_token_ids=stop_token_ids,
                        structural_token_ids=structural_token_ids, fixed_weight_mask=fixed_weight_mask)
    if not math.isfinite(tau) or tau <= 0 or tau > torch.finfo(torch.float32).max:
        raise ValueError("tau must be finite, positive and representable in float32")
    if not math.isfinite(credit_lambda) or credit_lambda < 1:
        raise ValueError("credit_lambda must be finite and at least one")
    a = advantage.float()
    if not torch.isfinite(a).all():
        raise ValueError("Valid credit advantages must be finite")
    frozen = _frozen_positions(mask, token_ids, freeze_stop_tokens, freeze_structural,
                               stop_token_ids, structural_token_ids, fixed_weight_mask)
    free = mask & ~frozen
    counts = free.sum(-1, keepdim=True).float()
    low, high = 1.0 / credit_lambda, float(credit_lambda)
    weight = torch.empty(mask.shape, device=mask.device, dtype=torch.float32)
    weight.uniform_(low, high, generator=generator)
    weight = weight.masked_fill(~free, 0.)
    converged = counts.squeeze(-1) == 0
    for _ in range(256):
        total = weight.sum(-1, keepdim=True)
        shift = ((counts - total) / counts.clamp_min(1)).masked_fill(counts == 0, 0.)
        weight = (weight + shift).clamp(low, high).masked_fill(~free, 0.)
        converged = torch.isclose(weight.sum(-1), counts.squeeze(-1), rtol=1e-7, atol=1e-6)
        if bool(converged.all()):
            break
    if not bool(converged.all()):
        raise ValueError("Random band credit did not reach the response budget inside the lambda band")
    weight = weight.masked_fill(frozen, 1.)
    # A zero advantage or a single free token admits only the uniform allocation.
    uniform = (a == 0) | (counts.squeeze(-1) < 2)
    weight = torch.where(uniform[:, None], mask.float(), weight)
    valid_min = weight.masked_fill(~mask, torch.inf).amin(-1)
    if not bool(((weight.amax(-1) <= high) & (valid_min >= low) & torch.isfinite(weight).all(-1)).all()):
        raise ValueError("Final credit weights violate the lambda band")
    if not torch.allclose(weight.sum(-1), mask.sum(-1).float(), rtol=2e-6, atol=2e-6):
        raise ValueError("Final credit weights do not preserve the response budget")
    tau_used = torch.full_like(a, max(tau, torch.finfo(torch.float32).tiny))
    return Credit(a[:, None] * weight, torch.zeros_like(weight), weight, tau_used)


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
                   vocab_chunk_size: int = 8192, credit_lambda: float = 2.0,
                   freeze_stop_tokens: bool = False,
                   freeze_structural: bool = False,
                   stop_token_ids: tuple[int, ...] | None = None,
                   structural_token_ids: tuple[int, ...] | None = None,
                   policy_temperature: float = 1.0,
                   fixed_weight_mask: Tensor | None = None,
                   contraction: str = "dot") -> Credit:
    """Exact full-vocabulary d_t with token/vocabulary blocking.

    old_logits [B,T,V] comes from the rollout policy at fixed hard prefixes.
    input_grads [B,T,D] already corresponds to the sampled token's RM position.
    rm_weight [V,D] uses precisely the Actor output token-ID ordering.
    Temporary vocabulary tensors are at most token_chunk_size*vocab_chunk_size.
    The caller supplies the complete logits and an unsharded embedding weight.
    Sampling temperature is applied after each block's float32 conversion.
    ``norm_product`` replaces <g,k> by ||g||_2 ||k||_2, with
    g = p*(W*f - E_p[W*f])/sigma and k = one_hot(sampled)-p.
    It changes only the scalar signal; signed advantage and allocation stay fixed.
    """
    mask = _mask(response_mask)
    if contraction not in ("dot", "norm_product"):
        raise ValueError("Unknown credit contraction")
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
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be finite and positive")
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
                z = old_logits[r, t, lo:lo + vocab_chunk_size].float() / policy_temperature
                log_z = torch.logaddexp(log_z, z.logsumexp(-1))
            mu, p2, p2v = (torch.zeros_like(log_z) for _ in range(3))
            for lo in range(0, V, vocab_chunk_size):
                z = old_logits[r, t, lo:lo + vocab_chunk_size].float() / policy_temperature
                p = (z - log_z[:, None]).exp()
                v = f @ rm_weight[lo:lo + vocab_chunk_size].float().T
                mu += (p * v).sum(-1)
                p2 += p.square().sum(-1)
                p2v += (p.square() * v).sum(-1)
            pa = (old_logits[r, t, a].float() / policy_temperature - log_z).exp()
            va = (f * rm_weight[a].float()).sum(-1)
            if contraction == "dot":
                direction[r, t] = (pa * (va - mu) - p2v + mu * p2) / reward_scale[r]
            else:
                # Accumulate centered squares directly; subtracting raw moments
                # loses precision for concentrated policies or near-constant v.
                g2, k2 = torch.zeros_like(mu), torch.zeros_like(mu)
                for lo in range(0, V, vocab_chunk_size):
                    z = old_logits[r, t, lo:lo + vocab_chunk_size].float() / policy_temperature
                    p = (z - log_z[:, None]).exp()
                    v = f @ rm_weight[lo:lo + vocab_chunk_size].float().T
                    g2 += (p * (v - mu[:, None])).square().sum(-1)
                    k = -p
                    inside = (a >= lo) & (a < lo + p.shape[-1])
                    k[inside, a[inside] - lo] += 1
                    k2 += k.square().sum(-1)
                direction[r, t] = g2.sqrt() * k2.sqrt() / reward_scale[r]
    return allocate(direction, advantage, mask, tau, credit_lambda=credit_lambda,
                    token_ids=token_ids, freeze_stop_tokens=freeze_stop_tokens,
                    freeze_structural=freeze_structural,
                    stop_token_ids=stop_token_ids, structural_token_ids=structural_token_ids,
                    fixed_weight_mask=fixed_weight_mask)


@torch.no_grad()
def rollout_importance_weights(old_logp: Tensor, rollout_logp: Tensor,
                               response_mask: Tensor) -> Tensor:
    """Frozen token correction from the sampler to the HF clipping anchor.

    Valid probabilities must be finite, nonpositive floating log probabilities.
    Return positive finite FP32 ratios without clipping or normalization; padded
    entries are one. This conditional token correction does not correct the
    complete sequence/group sampling distribution.
    """
    if any(not isinstance(x, Tensor) or not x.is_floating_point()
           for x in (old_logp, rollout_logp)):
        raise ValueError("Old and rollout log probabilities must be floating tensors")
    if not isinstance(response_mask, Tensor):
        raise ValueError("response_mask must be a binary [B, T] tensor")
    if any(x.shape != response_mask.shape for x in (old_logp, rollout_logp)):
        raise ValueError("Old/rollout log probabilities and mask must have shape [B,T]")
    if any(x.device != old_logp.device for x in (rollout_logp, response_mask)):
        raise ValueError("Old/rollout log probabilities and mask must share a device")
    mask = _mask(response_mask)
    for values in (old_logp, rollout_logp):
        if not torch.isfinite(values[mask]).all() or (values[mask] > 0).any():
            raise ValueError("Valid log probabilities must be finite and nonpositive")
    old = old_logp.detach().float().masked_fill(~mask, 0)
    rollout = rollout_logp.detach().float().masked_fill(~mask, 0)
    if not torch.isfinite(old).all() or not torch.isfinite(rollout).all():
        raise ValueError("Valid log probabilities must be representable in float32")
    weights = (old - rollout).exp()
    if not (torch.isfinite(weights) & (weights > 0)).all():
        raise ValueError("Rollout importance weights must be positive and finite in float32")
    return weights


def grpo_policy_loss(new_logp: Tensor, old_logp: Tensor, token_advantage: Tensor,
                     response_mask: Tensor, clip_eps: float = 0.2, *,
                     importance_weights: Tensor | None = None) -> Tensor:
    """Negative clipped objective: average tokens per response, then responses.

    Optional frozen importance weights correct sampler/HF differences outside
    PPO clipping. Add the trainer's KL term separately with its declared
    estimator and the same response reduction.
    """
    mask = _mask(response_mask)
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
    # Mask before exponentiation so padded NaNs cannot affect backward.
    log_ratio = (new_logp.float().masked_fill(~mask, 0)
                 - old_logp.detach().float().masked_fill(~mask, 0))
    a = token_advantage.detach().float().masked_fill(~mask, 0)
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(a).all():
        raise ValueError("Valid log probabilities and token advantages must be finite")
    # For positive A, min(ratio, 1+eps) is clipped in log space *before*
    # exponentiation. Computing exp first gives 0*inf=NaN in backward even
    # when the selected clipped objective is finite. Negative A uses max.
    selected_log_ratio = torch.where(
        a >= 0, log_ratio.clamp(max=math.log1p(clip_eps)),
        log_ratio.clamp(min=math.log1p(-clip_eps)))
    selected_log_ratio = selected_log_ratio.masked_fill(a == 0, 0)
    # Combine the advantage scale before exp: even an overflowing raw ratio
    # may have a representable objective and gradient when |A| is tiny.
    log_scale = a.abs().masked_fill(a == 0, 1).log()
    if weights is not None:
        # Combine factors before exp, retaining small-weight cancellation of a
        # large ratio. Zero advantages stay zero even at the FP32 weight limit.
        log_scale = log_scale + weights.log().masked_fill(a == 0, 0)
    objective = a.sign() * (selected_log_ratio + log_scale).exp()
    if not torch.isfinite(objective).all():
        raise ValueError("Policy objective overflow; gradients would not be finite")
    loss = -(objective / mask.sum(-1, keepdim=True)).mean(0).sum()
    if not torch.isfinite(loss):
        raise ValueError("Policy loss overflow")
    return loss
