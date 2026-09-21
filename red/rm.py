"""Reward model: scalar scoring plus the per-position prefix scores Eq. (6) needs.

The reward model is frozen.  RED is *forward-only*: it reads the scalar head at
every position of the canonical reward-model row, so ``R(x, y_<=t)`` is available
for each ``t`` and the redistributed reward is the difference of consecutive
prefix scores.  No backward pass, no input gradients, no model modification --
which is the paper's "minimal additional computational costs" claim, and one
pass cheaper than the sibling P2T arm.

The pooled index is the same last-valid position ``forward`` pools at, so
``prefix_scores(...)[b, pooled[b]]`` is exactly the score ``forward`` returns.
That equality is what makes the telescoping identity hold:

    sum_t r~_t = R_phi(x, y) - R_phi(x, empty)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .mapping import build_reward_input
from .tensors import check_response_tokens, position_ids_from_mask
from .reward import prefix_boundaries, prefix_token_rewards


class LastTokenReward(nn.Module):
    """Decoder backbone plus scalar score head; pools at the last valid token."""

    def __init__(self, backbone: nn.Module, score_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.score_head = score_head

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def _pooled_positions(self, attention_mask: Tensor) -> Tensor:
        length = attention_mask.shape[1]
        indexes = torch.arange(length, device=attention_mask.device).expand_as(attention_mask)
        return indexes.masked_fill(~attention_mask.bool(), -1).amax(-1)

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
            score_positions = self._pooled_positions(attention_mask)
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

    def prefix_scores(self, *, input_ids: Tensor, attention_mask: Tensor,
                      position_ids: Tensor | None = None
                      ) -> tuple[Tensor, Tensor]:
        """The scalar head at *every* position, plus the pooled position.

        Returns ``(scores [B, L], pooled [B])`` where ``scores[b, j]`` is the head
        applied to the hidden state after consuming position ``j`` -- i.e.
        ``R_phi(x, y_<=j)`` -- and ``pooled[b]`` is the last valid position.

        The head is a bare ``Linear``, so applying it to a ``[B, L, D]`` hidden
        state is the same arithmetic ``forward`` does for the pooled row alone;
        the two agree at ``pooled`` by construction, and a test pins that.

        Forward-only on purpose: RED never needs a gradient through the reward
        model, so ``input_ids`` goes straight to the backbone and the graph is
        never built.
        """
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        if not attention_mask.bool().any(-1).all():
            raise ValueError("Each RM input must contain a valid token")
        if position_ids is None:
            position_ids = position_ids_from_mask(attention_mask)
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, use_cache=False,
                               return_dict=True).last_hidden_state
        scores = self.score_head(hidden)
        if scores.ndim != 3 or scores.shape[:2] != hidden.shape[:2] or scores.shape[2] != 1:
            raise ValueError("LastTokenReward requires a scalar head over [B, L, D]")
        pooled = self._pooled_positions(attention_mask)
        # scores[b, pooled[b]] must reproduce forward() exactly, or the
        # redistributed rewards stop summing to the sequence score.  A test pins
        # that agreement; asserting it here would cost a second full backbone pass.
        return scores[:, :, 0].float(), pooled


@torch.no_grad()
def build_rm_batch(actor_tokenizer, reward_tokenizer, prompts, responses, response_mask,
                   max_prompt_tokens: int, max_response_tokens: int):
    """Canonical RM token rows plus the actor->RM position map.

    Returns ``(rows, mapped, fixed_weight_mask, stats)``: ``rows`` are the RM
    input IDs, ``mapped`` [B, T] holds the RM position of each actor response
    token or -1, and ``fixed_weight_mask`` marks valid positions with no exact
    counterpart (the set whose redistributed reward is zero by construction).
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
    # Content-restricted unmapped fraction.  The plain unmapped count is
    # dominated by the terminal special token, which the reward model pools at and
    # which is *always* unmapped by construction, so it cannot answer "did the
    # mapping fail for real text".  This is the statistic the parent project
    # bounds at 0.25 in its startup gate.
    from .tokens import get_special_token_ids
    special_ids = torch.tensor(get_special_token_ids(actor_tokenizer), device=responses.device)
    content_mask = response_mask & ~torch.isin(responses, special_ids)
    unmapped_content = (content_mask & fixed_weight_mask).sum()
    stats = {
        "rm_max_input_tokens": max(map(len, rows)),
        "rm_input_token_budget": rm_budget,
        "rm_mapped_tokens": int((response_mask & mapped.ge(0)).sum()),
        "rm_unmapped_tokens": int(fixed_weight_mask.sum()),
        "rm_unmapped_content_tokens": int(unmapped_content),
        "rm_unmapped_content_fraction": float(
            unmapped_content / content_mask.sum().clamp_min(1)),
    }
    return rows, mapped, fixed_weight_mask, stats


@torch.no_grad()
def score_prefixes(reward_model: nn.Module, reward_tokenizer, rows, mapped: Tensor,
                   responses: Tensor, response_mask: Tensor, *, device, microbatch: int = 1):
    """Score every reward-model row at every position and redistribute it.

    Returns ``(rewards [B], token_rewards [B, T])``: ``rewards`` is the pooled
    sequence score ``R_phi(x, y)`` -- the same value the sibling arms use -- and
    ``token_rewards`` is Eq. (6), ``r~_t = R_phi(x, y_<=t) - R_phi(x, y_<=t-1)``,
    gathered onto the actor's response tokens.

    The physical microbatch is one in the main configuration: a whole row's
    ``[L, D]`` hidden state is materialised to read the head at every position,
    and 64 rows at once does not fit on a 48 GB card.
    """
    micro = max(1, int(microbatch))
    reward_parts, token_parts = [], []
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
        scores, pooled = reward_model.prefix_scores(input_ids=ids, attention_mask=attention)
        rows_index = torch.arange(len(chunk), device=device)
        rewards = scores[rows_index, pooled]
        # Eq. (6) over the reward model's own row, then gathered onto actor tokens.
        # Differences are taken between *boundaries* rather than between adjacent
        # actor tokens so that the trailing special token the RM pools at is
        # carried by the last mapped response token instead of being dropped.
        left, right = prefix_boundaries(positions, pooled)
        # Masked with the *actor* response mask, not ``& positions.ge(0)``.  The
        # intersection can empty a row -- an immediate-stop response is a single
        # special token, which is never mappable by construction -- and an empty
        # row fails the "at least one valid token" check, aborting the run on a
        # response the trainer deliberately trains on.  The intersection is also
        # redundant: ``prefix_boundaries`` gives unmapped positions right == left,
        # so their difference is already exactly zero and only the padded slots
        # need excluding.
        token = prefix_token_rewards(scores[rows_index[:, None], right],
                                     scores[rows_index[:, None], left],
                                     response_mask[start:end].to(device))
        reward_parts.append(rewards.detach().cpu())
        token_parts.append(token.detach().cpu())
        del ids, attention, positions, tokens, valid, scores, pooled, left, right, token
    rewards = torch.cat(reward_parts)
    token_rewards = torch.cat(token_parts)
    if not torch.isfinite(rewards).all() or not torch.isfinite(token_rewards).all():
        raise ValueError("RM produced non-finite rewards or prefix differences")
    return rewards, token_rewards
