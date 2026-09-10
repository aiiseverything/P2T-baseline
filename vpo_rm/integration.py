"""Rollout cache construction and differentiable Actor loss for a GRPO trainer."""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .alignment import gather_response, check_response_tokens
from .core import Credit, compute_credit, grpo_policy_loss
from .reward import reward_input_gradients, position_ids_from_mask


@dataclass(frozen=True)
class RolloutCache:
    old_logp: Tensor
    credit: Credit


def actor_response_logits(actor: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                          response_positions: Tensor, response_mask: Tensor,
                          position_ids: Tensor | None = None,
                          output_mask: Tensor | None = None) -> Tensor:
    """response_positions indexes actual tokens; their predictions are at t-1."""
    valid = response_mask.bool()
    if (response_positions[valid] < 1).any():
        raise ValueError("Each response token needs a preceding Actor context position")
    predecessors = response_positions - 1
    if not gather_response(attention_mask, predecessors, valid)[valid].bool().all():
        raise ValueError("The prediction position must belong to the Actor context")
    if position_ids is None:
        position_ids = position_ids_from_mask(attention_mask)
    # The trainer pads every row to a common prompt width and appends the
    # response contiguously.  Qwen3/Transformers supports logits_to_keep, so
    # request only the predecessor positions needed for the response.  This
    # avoids materializing a full-sequence logits tensor and then copying a
    # second [B,T,V] response tensor (which is prohibitive at 2048 tokens).
    start = int(predecessors[valid].min())
    # The keep range must span the whole padded response width.  Deriving the end
    # from valid positions alone compares a width-T row against a length-L range,
    # which silently disables logits_to_keep for every response shorter than the
    # batch maximum (63 of 64 microbatch forwards in the main configuration).
    width = predecessors.shape[1]
    keep = torch.arange(start, start + width, device=input_ids.device)
    contiguous = torch.equal(predecessors[0], keep) and all(
        torch.equal(row, predecessors[0]) for row in predecessors[1:])
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, use_cache=False, return_dict=True)
    if contiguous:
        try:
            outputs = actor(**kwargs, logits_to_keep=keep)
            logits = outputs.logits
        except TypeError:
            outputs = actor(**kwargs)
            logits = gather_response(outputs.logits, predecessors, valid)
    else:
        outputs = actor(**kwargs)
        logits = gather_response(outputs.logits, predecessors, valid)
    if output_mask is not None:
        if output_mask.shape != (logits.shape[-1],) or not output_mask.bool().any():
            raise ValueError("output_mask must define a nonempty vocabulary support")
        logits = logits.masked_fill(~output_mask.bool(), -torch.inf)
    return logits


def selected_logp_from_logits(logits: Tensor, token_ids: Tensor,
                              response_mask: Tensor, token_chunk_size: int = 128) -> Tensor:
    """Compute selected-token log-probs without a full FP32 B×T×V copy.

    The model output remains BF16.  FP32 reduction is performed one token block
    at a time, which keeps the numerical reduction used by the loss while
    avoiding a second full-vocabulary allocation at long response lengths.
    """
    safe_ids = token_ids.masked_fill(~response_mask.bool(), 0)
    parts = []
    for lo in range(0, logits.shape[1], max(1, int(token_chunk_size))):
        hi = min(logits.shape[1], lo + max(1, int(token_chunk_size)))
        z = logits[:, lo:hi].float()
        parts.append(z.gather(-1, safe_ids[:, lo:hi, None]).squeeze(-1)
                     - z.logsumexp(-1))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def build_credit_cache(old_logits: Tensor, token_ids: Tensor, input_grads: Tensor,
                       rm_weight: Tensor, advantage: Tensor, reward_scale: Tensor,
                       response_mask: Tensor, tau: float, **chunk_sizes) -> RolloutCache:
    """Call after complete-group reward aggregation; reuse throughout this rollout."""
    credit = compute_credit(old_logits, token_ids, input_grads, rm_weight,
                            advantage, reward_scale, response_mask, tau, **chunk_sizes)
    # Token blocks avoid allocating an additional full [B,T,V] log-softmax.
    old_logp = torch.zeros_like(credit.direction)
    rows, times = response_mask.bool().nonzero(as_tuple=True)
    chunk = chunk_sizes.get("token_chunk_size", 128)
    for lo in range(0, rows.numel(), chunk):
        r, t = rows[lo:lo+chunk], times[lo:lo+chunk]
        z = old_logits[r, t].float()
        old_logp[r, t] = z.gather(-1, token_ids[r, t, None]).squeeze(-1) - z.logsumexp(-1)
    return RolloutCache(old_logp, credit)


def response_reward_gradients(reward_model: nn.Module, rm_input_ids: Tensor,
                              rm_attention_mask: Tensor, rm_response_positions: Tensor,
                              token_ids: Tensor, response_mask: Tensor,
                              **score_kwargs) -> tuple[Tensor, Tensor]:
    check_response_tokens(rm_input_ids, rm_attention_mask, rm_response_positions,
                          token_ids, response_mask)
    rewards, all_grads = reward_input_gradients(reward_model, rm_input_ids,
                                              rm_attention_mask, **score_kwargs)
    return rewards, gather_response(all_grads, rm_response_positions, response_mask)


def actor_policy_loss(actor: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                      response_positions: Tensor, token_ids: Tensor,
                      response_mask: Tensor, cache: RolloutCache,
                      clip_eps: float = .2, position_ids: Tensor | None = None,
                      output_mask: Tensor | None = None) -> Tensor:
    check_response_tokens(input_ids, attention_mask, response_positions,
                          token_ids, response_mask)
    logits = actor_response_logits(actor, input_ids, attention_mask, response_positions,
                                   response_mask, position_ids, output_mask)
    # FP32 reductions retain stable log probabilities with bf16 model weights,
    # while token blocking avoids materializing a second full [B,T,V] tensor.
    new_logp = selected_logp_from_logits(logits, token_ids, response_mask)
    return grpo_policy_loss(new_logp, cache.old_logp, cache.credit.advantage,
                            response_mask, clip_eps)
