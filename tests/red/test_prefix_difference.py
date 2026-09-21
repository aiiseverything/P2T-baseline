"""Eq. (6) and the boundary construction behind it.

The telescoping identity ``sum_t r~_t = R_phi(x, y) - R_phi(x, empty)`` is what
makes RED a reward *redistribution* rather than a new objective, so it is pinned
here directly.  The interesting cases are the holes: the reward model pools at
the last valid position of the canonical chat, which is a special token and is
therefore *always* unmapped, and byte-level BPE rewrites leave unmapped positions
in the middle.  A naive per-actor-token difference drops the trailing segment and
silently breaks the identity.
"""
from __future__ import annotations

import pytest
import torch

from red.reward import prefix_boundaries, prefix_token_rewards


def _case(seed: int = 7, rows: int = 4, width: int = 6, length: int = 24):
    """Synthetic mapped/score tensors covering the three interesting shapes."""
    torch.manual_seed(seed)
    mapped = torch.tensor([
        [5, 6, 7, 8, 9, 10],        # fully mapped
        [5, -1, 8, 9, 10, 11],      # a hole in the middle
        [5, 6, 7, 8, -1, -1],       # trailing unmapped run
        [-1, 6, 7, -1, 9, 10],      # leading hole as well
    ][:rows])
    mask = torch.ones(rows, width, dtype=torch.bool)
    pooled = torch.tensor([12, 13, 12, 12][:rows])
    scores = torch.randn(rows, length).double() * 3.0
    return mapped, mask, pooled, scores


def test_telescoping_identity_holds_with_holes_and_trailing_specials():
    mapped, mask, pooled, scores = _case()
    left, right = prefix_boundaries(mapped, pooled)
    rows = torch.arange(mapped.shape[0])[:, None]
    got = prefix_token_rewards(scores[rows, right], scores[rows, left], mask).double()

    for b in range(mapped.shape[0]):
        first = mapped[b][mapped[b] >= 0].min()
        want = scores[b, pooled[b]] - scores[b, first - 1]
        torch.testing.assert_close(got[b].sum(), want, rtol=1e-5, atol=1e-5)


def test_unmapped_tokens_contribute_exactly_zero():
    mapped, mask, pooled, scores = _case()
    left, right = prefix_boundaries(mapped, pooled)
    rows = torch.arange(mapped.shape[0])[:, None]
    got = prefix_token_rewards(scores[rows, right], scores[rows, left], mask)
    unmapped = mask & mapped.lt(0)
    assert (got[unmapped] == 0).all(), "an unmapped token must not borrow a neighbour's reward"
    # ...but their span is not lost: it is carried by the following mapped token,
    # which is exactly why the identity above still holds.
    assert got[mask].abs().sum() > 0


def test_boundaries_chain_so_each_mapped_token_owns_its_span():
    """``left[t] == right[previous mapped token]``: the differences chain exactly."""
    mapped, mask, pooled, scores = _case()
    left, right = prefix_boundaries(mapped, pooled)
    for b in range(mapped.shape[0]):
        mapped_indexes = torch.nonzero(mapped[b] >= 0).flatten().tolist()
        for position, t in enumerate(mapped_indexes):
            if position == 0:
                continue
            previous = mapped_indexes[position - 1]
            assert left[b, t].item() == right[b, previous].item(), \
                f"row {b} token {t} does not chain onto token {previous}"


def test_pooled_position_carries_the_trailing_special_token():
    """The last mapped token's right boundary must be the pooled index."""
    mapped, mask, pooled, scores = _case()
    left, right = prefix_boundaries(mapped, pooled)
    for b in range(mapped.shape[0]):
        mapped_indexes = torch.nonzero(mapped[b] >= 0).flatten().tolist()
        last = mapped_indexes[-1]
        assert right[b, last].item() == pooled[b].item()
        if len(mapped_indexes) > 1:
            assert left[b, last].item() == right[b, mapped_indexes[-2]].item()


def test_first_boundary_is_the_dynamic_initialisation_point():
    """``left`` of the first mapped token is one before it: R_phi(x, empty)."""
    mapped, mask, pooled, scores = _case()
    left, _ = prefix_boundaries(mapped, pooled)
    for b in range(mapped.shape[0]):
        first = int(torch.nonzero(mapped[b] >= 0)[0].item())
        assert left[b, first].item() == mapped[b, first].item() - 1


def test_all_unmapped_row_is_safe_and_zero():
    mapped = torch.tensor([[-1, -1, -1]])
    pooled = torch.tensor([2])
    left, right = prefix_boundaries(mapped, pooled)
    assert (left >= 0).all() and (right >= 0).all()
    got = prefix_token_rewards(torch.zeros(1, 3), torch.zeros(1, 3),
                               torch.ones(1, 3, dtype=torch.bool))
    assert (got == 0).all()


@pytest.mark.parametrize("bad", [
    (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long)),      # float mapped
    (torch.zeros(2, 3, dtype=torch.long), torch.zeros(3, dtype=torch.long)),  # pooled width
])
def test_shape_validation(bad):
    with pytest.raises(ValueError):
        prefix_boundaries(*bad)


def test_reward_masks_invalid_positions():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    right = torch.full((2, 3), 1, dtype=torch.long)
    left = torch.zeros(2, 3, dtype=torch.long)
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = prefix_token_rewards(scores, torch.zeros_like(scores), mask)
    assert got[0, 2] == 0 and got[1, 1] == 0 and got[1, 2] == 0


def test_non_binary_mask_is_rejected():
    with pytest.raises(ValueError):
        prefix_token_rewards(torch.ones(1, 2), torch.zeros(1, 2),
                             torch.tensor([[1, 2]]))


def test_row_with_no_valid_token_is_rejected():
    with pytest.raises(ValueError):
        prefix_token_rewards(torch.ones(1, 2), torch.zeros(1, 2),
                             torch.zeros(1, 2, dtype=torch.bool))
