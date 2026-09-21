"""Eq. (7), Eq. (8) and the sparse sequence term they combine with.

Two things here are easy to get subtly wrong.  ``beta_c = 1`` must make the
sequence term vanish *identically* (not merely shrink), because that is the
paper's default and Table 7 uses it everywhere except LLaMA3 on TL;DR.  And the
sparse term belongs on the final response token -- Eq. (7) writes ``r_t``, which
Eq. (3) defines as zero everywhere except ``t = T``.
"""
from __future__ import annotations

import pytest
import torch

from red.reward import (RED_BETA_C_DEFAULT, red_convex_combination,
                        red_final_reward, red_kl_reward, sequence_reward_at_eos)


def test_sparse_sequence_reward_lands_on_the_final_valid_token():
    rewards = torch.tensor([2.0, -3.0])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    got = sequence_reward_at_eos(rewards, mask)
    torch.testing.assert_close(got, torch.tensor([[0.0, 0.0, 2.0, 0.0],
                                                  [0.0, -3.0, 0.0, 0.0]]))


def test_beta_c_of_one_drops_the_sequence_term_entirely():
    token = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sequence = torch.tensor([[0.0, 0.0], [0.0, -7.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    got = red_convex_combination(token, sequence, mask, beta_c=1.0)
    torch.testing.assert_close(got, token)


def test_beta_c_of_zero_keeps_only_the_sequence_term():
    token = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sequence = torch.tensor([[0.0, 0.0], [0.0, -7.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    got = red_convex_combination(token, sequence, mask, beta_c=0.0)
    torch.testing.assert_close(got, sequence)


def test_beta_c_interpolates():
    token = torch.tensor([[2.0]])
    sequence = torch.tensor([[6.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    got = red_convex_combination(token, sequence, mask, beta_c=0.25)
    # 0.25 * 2 + 0.75 * 6
    torch.testing.assert_close(got, torch.tensor([[5.0]]))


def test_final_reward_subtracts_beta_times_the_kl():
    token = torch.tensor([[1.0, 1.0]])
    sequence = torch.zeros(1, 2)
    kl = torch.tensor([[2.0, -4.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    got = red_final_reward(token, sequence, kl, mask, beta_c=1.0, beta=0.5)
    torch.testing.assert_close(got, torch.tensor([[0.0, 3.0]]))


def test_kl_reward_is_the_signed_log_ratio_not_a_divergence():
    """Eq. (4) writes a divergence; Figure 4 computes a signed difference."""
    old = torch.tensor([[0.5, -1.0]])
    ref = torch.tensor([[0.2, -0.4]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    got = red_kl_reward(old, ref, mask)
    torch.testing.assert_close(got, old - ref)
    assert got[0, 0] > 0 and got[0, 1] < 0, "the sign must survive, not be clamped"


def test_kl_reward_uses_pi_old_not_the_updated_policy():
    """Documented contract: the caller passes the rollout log-probs."""
    old = torch.zeros(1, 1)
    ref = torch.tensor([[1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    torch.testing.assert_close(red_kl_reward(old, ref, mask), torch.tensor([[-1.0]]))


def test_masked_tokens_are_zero_in_every_stage():
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    token = torch.tensor([[1.0, 2.0, 3.0]])
    sequence = torch.tensor([[0.0, 0.0, 0.0]])
    kl = torch.tensor([[1.0, 1.0, 1.0]])
    combined = red_convex_combination(token, sequence, mask)
    final = red_final_reward(token, sequence, kl, mask, beta_c=RED_BETA_C_DEFAULT, beta=1.0)
    assert combined[0, 2] == 0 and final[0, 2] == 0


@pytest.mark.parametrize("beta_c", [-0.1, 1.1, float("nan"), float("inf")])
def test_beta_c_outside_the_unit_interval_is_rejected(beta_c):
    with pytest.raises(ValueError):
        red_convex_combination(torch.ones(1, 1), torch.zeros(1, 1),
                               torch.ones(1, 1, dtype=torch.bool), beta_c=beta_c)


def test_negative_beta_is_rejected():
    with pytest.raises(ValueError):
        red_final_reward(torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
                         torch.ones(1, 1, dtype=torch.bool), beta=-0.5)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        red_convex_combination(torch.ones(1, 2), torch.zeros(1, 3),
                               torch.ones(1, 2, dtype=torch.bool))


def test_non_finite_scores_are_rejected():
    with pytest.raises(ValueError):
        red_kl_reward(torch.tensor([[float("inf")]]), torch.zeros(1, 1),
                      torch.ones(1, 1, dtype=torch.bool))
