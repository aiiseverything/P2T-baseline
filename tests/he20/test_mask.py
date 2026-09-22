"""Eq. (6): which tokens survive, and what the paper and its code disagree about.

The load-bearing tests are the tie case and the population case.  The paper defines
the mask by a threshold ``H_t^i >= tau_rho^B`` so ties at ``tau`` are all kept and
the retained fraction can exceed ``rho``; the authors' reference implementation
sorts instead and keeps exactly ``ceil(rho*n)``.  The two agree on distinct values
and disagree on ties, and ``test_the_paper_rule_and_the_reference_rule_disagree_on_ties``
pins that difference rather than hiding it.  The population tests pin that this
module ranks exactly the rows it is handed, so a caller cannot silently change the
selection by changing how it batches.
"""
from __future__ import annotations

import pytest
import torch

from he20.mask import (ENTROPY_TOP_RATIO_DEFAULT, ENTROPY_TOP_RULES, entropy_threshold,
                       entropy_top_mask, kept_fraction)


def reference_rule(entropy, response_mask, top_ratio):
    """The authors' ``get_global_entropy_top_mask``, transcribed.

    Kept here so the agreement and the disagreement are both asserted against the
    real thing rather than against a paraphrase of it.
    """
    flat_entropy = entropy.flatten()
    flat_mask = response_mask.flatten().bool()
    response_entropy = flat_entropy[flat_mask]
    top_k = max(1, int(len(response_entropy) * top_ratio + 0.9999))
    _, topk_idx = torch.topk(response_entropy, k=top_k)
    positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
    out = torch.zeros_like(flat_entropy, dtype=torch.bool)
    out[positions[topk_idx]] = True
    return out.view_as(entropy)


def test_the_default_ratio_is_the_papers_main_setting():
    assert ENTROPY_TOP_RATIO_DEFAULT == 0.2
    assert ENTROPY_TOP_RULES == ("threshold", "topk")


def test_the_population_is_exactly_the_rows_handed_in():
    """The caller decides the population; changing the rows changes the selection.

    A token that is below the threshold when ranked against 200 others is above it
    when the population is only its own response -- which is precisely why the
    trainer must pool over the minibatch and not over a single response.
    """
    entropy = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    wide = entropy_top_mask(entropy, mask, 0.25)               # keep the single best
    torch.testing.assert_close(wide, torch.tensor([[True, False, False, False]]))
    # The same token, ranked only against the second half of its own row, survives.
    narrow = entropy_top_mask(entropy[:, 2:], mask[:, 2:], 0.5)
    torch.testing.assert_close(narrow, torch.tensor([[True, False]]))


def test_the_kept_count_is_ceil_of_the_ratio_under_the_topk_rule():
    entropy = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    for ratio, expected in ((0.2, 2), (0.25, 3), (0.5, 5), (1.0, 10), (0.01, 1)):
        kept = entropy_top_mask(entropy, mask, ratio, rule="topk")
        assert int(kept.sum()) == expected, f"ratio {ratio} kept {int(kept.sum())}"


def test_the_threshold_rule_and_the_reference_rule_agree_on_distinct_values():
    torch.manual_seed(0)
    entropy = torch.rand(4, 12) * 5
    mask = torch.ones(4, 12, dtype=torch.bool)
    mine = entropy_top_mask(entropy, mask, 0.25)
    theirs = reference_rule(entropy, mask, 0.25)
    assert torch.equal(mine, theirs), "distinct values must give the same selection"


def test_the_paper_rule_and_the_reference_rule_disagree_on_ties():
    """The recorded contradiction, asserted rather than smoothed over.

    An all-equal population is the extreme case: the paper's threshold admits every
    token (they all sit at tau), while the reference implementation keeps
    ceil(rho*n) of them and breaks the tie by index.
    """
    entropy = torch.full((1, 8), 2.5)
    mask = torch.ones(1, 8, dtype=torch.bool)
    paper = entropy_top_mask(entropy, mask, 0.25, rule="threshold")
    code = entropy_top_mask(entropy, mask, 0.25, rule="topk")
    assert int(paper.sum()) == 8, "the threshold rule keeps every tied token"
    assert int(code.sum()) == 2, "the reference rule keeps ceil(rho*n) and breaks ties by index"
    assert not torch.equal(paper, code)


def test_the_threshold_is_the_complement_quantile():
    entropy = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    threshold = entropy_threshold(entropy, mask, 0.2)
    torch.testing.assert_close(threshold, torch.quantile(entropy.flatten(), 0.8))
    # and every token at or above it is kept
    kept = entropy_top_mask(entropy, mask, 0.2)
    assert torch.equal(kept, entropy >= threshold)


def test_tokens_outside_the_response_mask_are_never_kept():
    """Garbage in the padding columns must not be selected."""
    entropy = torch.tensor([[1.0, 1.0, 9.9, 9.9]])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    kept = entropy_top_mask(entropy, mask, 0.5, rule="topk")
    assert not kept[0, 2] and not kept[0, 3], "the padding column holds the largest value"
    assert int(kept.sum()) == 1     # ceil(0.5 * 2) over the two valid tokens


def test_kept_fraction_reports_the_ratio():
    entropy = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    kept = entropy_top_mask(entropy, mask, 0.3, rule="topk")
    torch.testing.assert_close(kept_fraction(kept, mask), torch.tensor(0.3))


def test_an_empty_population_is_rejected():
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one valid token"):
        entropy_top_mask(torch.zeros(1, 3), mask, 0.2)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5, float("nan")])
def test_the_ratio_must_be_a_fraction(ratio):
    mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="top_ratio"):
        entropy_top_mask(torch.rand(1, 3), mask, ratio)


def test_an_unknown_rule_is_rejected():
    mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="rule must be"):
        entropy_top_mask(torch.rand(1, 3), mask, 0.2, rule="quantile")


def test_non_finite_entropy_is_rejected():
    mask = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="finite"):
        entropy_top_mask(torch.tensor([[1.0, float("nan"), 2.0]]), mask, 0.5)


def test_a_single_token_response_keeps_its_token_at_every_ratio():
    """The population can be smaller than 1/rho; the guard keeps at least one token."""
    entropy = torch.tensor([[4.2]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    for ratio in (0.01, 0.2, 1.0):
        kept = entropy_top_mask(entropy, mask, ratio)
        assert int(kept.sum()) == 1, f"ratio {ratio} kept nothing"


def test_the_paper_rule_keeps_at_least_the_maximum():
    """A threshold at the (1-rho) quantile can never empty the population."""
    torch.manual_seed(3)
    for _ in range(20):
        entropy = torch.rand(3, 7) * 10
        mask = torch.ones(3, 7, dtype=torch.bool)
        kept = entropy_top_mask(entropy, mask, 0.05)
        assert kept.any()
        assert bool(kept.flatten()[int(entropy.flatten().argmax())])
