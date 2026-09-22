"""Small tensor helpers, ported from ``vpo_rm/alignment.py``."""
from __future__ import annotations

import torch
from torch import Tensor


def gather_response(values: Tensor, positions: Tensor, response_mask: Tensor) -> Tensor:
    """Gather [B, L, ...] at explicit [B, T] positions.

    Invalid entries may be -1 and come back as zeros.  That zero is load-bearing
    for he20: an unmapped token simply contributes nothing to the sequence reward,
    rather than pulling a reward-model value from the wrong position.  he20 has no
    per-token reward at all, so the zero is never mistaken for one.
    """
    if positions.shape != response_mask.shape or values.shape[0] != positions.shape[0]:
        raise ValueError("Batch dimensions and response position shapes must match")
    if positions.dtype != torch.long:
        raise ValueError("Response positions must be int64")
    valid = response_mask.bool()
    if ((positions[valid] < 0) | (positions[valid] >= values.shape[1])).any():
        raise ValueError("Valid response position outside source sequence")
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    result = values[rows, positions.masked_fill(~valid, 0)]
    expanded = valid.reshape(*valid.shape, *((1,) * (result.ndim - 2)))
    return result.masked_fill(~expanded, 0)


def check_response_tokens(input_ids: Tensor, attention_mask: Tensor, positions: Tensor,
                          token_ids: Tensor, response_mask: Tensor) -> None:
    """The RM must be fed exactly the actor's tokens at the mapped positions."""
    valid = response_mask.bool()
    gathered = gather_response(input_ids, positions, valid)
    attended = gather_response(attention_mask, positions, valid)
    if not attended[valid].bool().all() or not torch.equal(gathered[valid], token_ids[valid]):
        raise ValueError("Response IDs or valid positions differ between Actor and RM")
    for row in range(positions.shape[0]):
        selected = positions[row, valid[row]]
        if selected.numel() > 1 and not (selected[1:] > selected[:-1]).all():
            raise ValueError("Response positions must be strictly increasing")


def position_ids_from_mask(attention_mask: Tensor) -> Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
