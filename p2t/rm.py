"""Reward model: scalar scoring and the input gradients Eq. (2) needs.

The reward model is frozen.  We differentiate the scalar score with respect to
its *input embeddings* -- one backward pass yields the gradient at every token,
which is what makes P2T training-free.

Ported from ``vpo_rm/reward.py`` so the gradient is taken through the same path
as the VPO-RM arms: embedding lookup, backbone, scalar head, last-valid pooling.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .mapping import build_reward_input
from .tensors import check_response_tokens, gather_response, position_ids_from_mask


class LastTokenReward(nn.Module):
    """Decoder backbone plus scalar score head; pools at the last valid token."""

    def __init__(self, backbone: nn.Module, score_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.score_head = score_head

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, *, inputs_embeds: Tensor, attention_mask: Tensor,
                position_ids: Tensor | None = None,
                score_positions: Tensor | None = None) -> Tensor:
        if not attention_mask.bool().any(-1).all():
            raise ValueError("Each RM input must contain a valid token")
        if position_ids is None:
            position_ids = position_ids_from_mask(attention_mask)
        hidden = self.backbone(inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask, position_ids=position_ids,
                               use_cache=False, return_dict=True).last_hidden_state
        batch, length = attention_mask.shape
        rows = torch.arange(batch, device=hidden.device)
        if score_positions is None:
            indexes = torch.arange(length, device=hidden.device).expand(batch, -1)
            score_positions = indexes.masked_fill(~attention_mask.bool(), -1).amax(-1)
        if score_positions.shape != (batch,) or score_positions.dtype != torch.long:
            raise ValueError("score_positions must be int64 [B]")
        if ((score_positions < 0) | (score_positions >= length)).any():
            raise ValueError("score_positions outside input sequence")
        if not attention_mask[rows, score_positions].bool().all():
            raise ValueError("Reward pooling must select valid positions")
        score = self.score_head(hidden[rows, score_positions])
        if score.shape != (batch, 1):
            raise ValueError("LastTokenReward requires a scalar head")
        return score[:, 0].float()


def reward_input_gradients(reward_model: nn.Module, input_ids: Tensor,
                           attention_mask: Tensor, **score_kwargs
                           ) -> tuple[Tensor, Tensor]:
    """One scalar reward and one input gradient per response, no batch rescaling."""
    reward_model.requires_grad_(False)
    reward_model.eval()
    with torch.inference_mode(False), torch.enable_grad():
        ids, mask = input_ids.clone(), attention_mask.clone()
        kwargs = {k: v.clone() if isinstance(v, Tensor) else v
                  for k, v in score_kwargs.items()}
        embeddings = reward_model.get_input_embeddings()(ids).detach().requires_grad_(True)
        rewards = reward_model(inputs_embeds=embeddings, attention_mask=mask, **kwargs)
        if rewards.shape != (ids.shape[0],):
            raise ValueError("Reward adapter must return a scalar per response [B]")
        if not torch.isfinite(rewards).all():
            raise ValueError("RM produced non-finite rewards")
        grads = torch.autograd.grad(rewards.sum(), embeddings)[0]
    return rewards.detach(), grads.detach()


@torch.no_grad()
def build_rm_batch(actor_tokenizer, reward_tokenizer, prompts, responses, response_mask,
                   max_prompt_tokens: int, max_response_tokens: int):
    """Canonical RM token rows plus the actor->RM position map.

    Returns ``(rows, mapped, fixed_weight_mask, stats)``: ``rows`` are the RM
    input IDs, ``mapped`` [B, T] holds the RM position of each actor response
    token or -1, and ``fixed_weight_mask`` marks valid positions with no exact
    counterpart (the set that gets I = 0 by construction).
    """
    if len(prompts) != len(responses) or responses.shape != response_mask.shape:
        raise ValueError("Reward prompts, responses and masks must align")
    rows = []
    rm_budget = max_prompt_tokens + max_response_tokens
    native_budget = getattr(reward_tokenizer, "model_max_length", None)
    if isinstance(native_budget, (int, float)) and math.isfinite(native_budget) and native_budget > 0:
        rm_budget = min(rm_budget, int(native_budget))
    mapped = torch.full_like(responses, -1, dtype=torch.long)
    for index, (prompt, response, valid) in enumerate(zip(prompts, responses, response_mask)):
        answer = response[valid].detach().cpu().tolist()
        encoded = build_reward_input(actor_tokenizer, reward_tokenizer, str(prompt), answer)
        if len(encoded.input_ids) > rm_budget:
            raise ValueError(f"Canonical RM input has {len(encoded.input_ids)} tokens, exceeds "
                             f"{rm_budget} token budget; no truncation is allowed")
        rows.append(encoded.input_ids)
        mapped[index, valid] = torch.tensor(encoded.response_positions, device=mapped.device)
    fixed_weight_mask = response_mask & mapped.lt(0)
    stats = {
        "rm_max_input_tokens": max(map(len, rows)),
        "rm_input_token_budget": rm_budget,
        "rm_mapped_tokens": int((response_mask & mapped.ge(0)).sum()),
        "rm_unmapped_tokens": int(fixed_weight_mask.sum()),
    }
    return rows, mapped, fixed_weight_mask, stats


def score_responses(reward_model: nn.Module, reward_tokenizer, rows, mapped: Tensor,
                    responses: Tensor, response_mask: Tensor, *, device, microbatch: int = 1):
    """Score every response and gather its embedding gradient onto actor tokens.

    The physical microbatch is one in the main configuration: retaining the
    backward graph for all 64 responses at once does not fit on a 48 GB card.
    """
    micro = max(1, int(microbatch))
    reward_parts, grad_parts = [], []
    pad = reward_tokenizer.pad_token_id
    for start in range(0, len(rows), micro):
        end = min(len(rows), start + micro)
        chunk = rows[start:end]
        width = max(map(len, chunk))
        ids = torch.full((len(chunk), width), pad, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        positions = mapped[start:end].to(device)
        for j, row in enumerate(chunk):
            ids[j, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            attention[j, :len(row)] = 1
        tokens = responses[start:end].to(device)
        valid = response_mask[start:end].to(device) & positions.ge(0)
        # The RM must be fed exactly the actor's tokens at the mapped positions.
        check_response_tokens(ids, attention, positions, tokens, valid)
        reward, grad = reward_input_gradients(reward_model, ids, attention)
        # The backward graph is over the *canonical reward-model* sequence, whose
        # width differs from the actor response width and varies per chunk. Gather
        # onto the actor's response positions before anything else: Eq. (2) needs
        # [B, T, D] aligned with the sampled tokens, and the gather is also what
        # zeroes I at positions the reward model never saw.
        grad = gather_response(grad, positions, valid)
        reward_parts.append(reward.detach().cpu())
        grad_parts.append(grad.detach().cpu())
        del ids, attention, positions, tokens, valid, reward, grad
    rewards = torch.cat(reward_parts)
    grads = torch.cat(grad_parts)
    if not torch.isfinite(rewards).all() or not torch.isfinite(grads).all():
        raise ValueError("RM produced non-finite rewards or gradients")
    return rewards, grads
