"""RLOO's leave-one-out baseline and the R4 advantage rule.

The paper gives a full PPO recipe and *no* RLOO details, so the rule this arm
uses is a choice, and ``RED_REPRO_NOTES.md`` 2.2 records it.  The load-bearing
test here is ``test_replacing_the_return_with_reds_return_is_exactly_rloo``: it
turns the argument for that choice into an executable fact.  RED's return is
``R(x, y) - R(x, empty)``, the dynamic-initialisation offset ``R(x, empty)`` is a
per-prompt constant, and a leave-one-out mean differences constants away -- so
feeding RED's return into RLOO's estimator reproduces plain RLOO exactly and the
arm would measure nothing.

R3 (``A_{i,t} = r^final_{i,t} - b_i``) was the first answer to that and had to be
retired: it subtracts a sequence-scale constant from a token-scale reward, so on
a group whose rewards are all negative it made *every* token's advantage positive
and the update stopped being contrastive.  ``red250`` did exactly that from
rollout 26 on.  R4 puts RLOO's sequence advantage back as the level and centres
RED's term within the response.  ``test_an_all_negative_group_does_not_get_uniformly_positive_advantages``
is the regression test for that failure and asserts the retired rule's collapse
in the same breath, so reinstating it fails here rather than 67 rollouts in.
"""
from __future__ import annotations

import pytest
import torch

from red.reward import (RED_ALPHA_DEFAULT, RED_BETA_C_DEFAULT, rloo_baseline,
                        rloo_red_credit, sequence_returns)


def test_leave_one_out_baseline_is_the_mean_of_the_others():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = torch.zeros(4, dtype=torch.long)
    baseline = rloo_baseline(returns, groups)
    torch.testing.assert_close(baseline, torch.tensor([3.0, 8 / 3, 7 / 3, 2.0]))


def test_leave_one_out_baseline_is_computed_per_group():
    returns = torch.tensor([1.0, 3.0, 10.0, 20.0])
    groups = torch.tensor([0, 0, 1, 1])
    baseline = rloo_baseline(returns, groups)
    torch.testing.assert_close(baseline, torch.tensor([3.0, 1.0, 20.0, 10.0]))


def test_group_with_one_response_is_undefined():
    with pytest.raises(ValueError):
        rloo_baseline(torch.tensor([1.0, 2.0]), torch.tensor([0, 1]))


def test_device_mismatch_in_the_baseline_is_reported_clearly():
    """A real crash on a multi-GPU box, and invisible to a CPU-only suite.

    The trainer builds the group ids where the reward model runs and the returns
    where the actor runs.  Mixing them raises a cryptic torch indexing error deep
    in the select, so the functions check their own arguments; this pins that.
    ``meta`` is the only way to produce two genuinely different devices here.
    """
    returns = torch.zeros(4, device="meta")
    groups = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="group_ids are on"):
        rloo_baseline(returns, groups)


def test_device_mismatch_between_reward_and_advantage_is_reported_clearly():
    final = torch.zeros(1, 3, device="meta")
    advantage = torch.zeros(1)
    with pytest.raises(ValueError, match="sequence advantage is on"):
        rloo_red_credit(final, advantage, torch.ones(1, 3, dtype=torch.bool))


def test_r4_advantage_is_the_sequence_advantage_plus_the_centred_token_term():
    final = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5]])
    advantage = torch.tensor([4.0, -1.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    credit = rloo_red_credit(final, advantage, mask)
    expected_direction = final - final.mean(-1, keepdim=True)
    torch.testing.assert_close(credit.direction, expected_direction)
    torch.testing.assert_close(credit.advantage, advantage[:, None] + expected_direction)


def test_r4_token_term_is_centred_within_each_response():
    """The property that keeps a response's length out of its own advantage."""
    torch.manual_seed(4)
    final = torch.randn(3, 7)
    mask = torch.ones(3, 7, dtype=torch.bool)
    credit = rloo_red_credit(final, torch.zeros(3), mask)
    torch.testing.assert_close(credit.direction.sum(-1), torch.zeros(3), atol=1e-6, rtol=0)


def test_r4_mean_advantage_is_exactly_the_sequence_advantage():
    """So RLOO's contrast reaches the loss unattenuated, whatever the length."""
    torch.manual_seed(6)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
                        dtype=torch.bool)
    final = torch.randn(3, 6)
    advantage = torch.tensor([2.0, -3.0, 0.5])
    credit = rloo_red_credit(final, advantage, mask)
    number = mask.sum(-1).float()
    torch.testing.assert_close((credit.advantage * mask).sum(-1) / number, advantage,
                              atol=1e-6, rtol=0)


def test_r4_alpha_scales_only_the_token_term():
    final = torch.tensor([[1.0, -2.0, 3.0]])
    advantage = torch.tensor([4.0])
    mask = torch.ones(1, 3, dtype=torch.bool)
    half = rloo_red_credit(final, advantage, mask, alpha=0.5)
    torch.testing.assert_close(half.advantage,
                              advantage[:, None] + 0.5 * final - 0.5 * final.mean(-1, keepdim=True))
    # alpha = 0 drops the redistribution and leaves plain RLOO, spread over tokens.
    none = rloo_red_credit(final, advantage, mask, alpha=0.0)
    torch.testing.assert_close(none.advantage, advantage[:, None].expand_as(final))


def test_r4_alpha_must_be_finite_and_nonnegative():
    final = torch.zeros(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            rloo_red_credit(final, torch.zeros(1), mask, alpha=bad)


def test_default_alpha_is_one():
    assert RED_ALPHA_DEFAULT == 1.0


def test_an_all_negative_group_does_not_get_uniformly_positive_advantages():
    """The ``red250`` regression, and the retired rule's collapse in one place.

    At rollout 26 every response in the group scored negative, R3's ``-b_i``
    became a large positive constant on every token, and the arm reported a
    ``red_positive_advantage_fraction`` of exactly 1.0 for the rest of the run.
    """
    torch.manual_seed(11)
    k, width = 4, 5
    groups = torch.zeros(k, dtype=torch.long)
    returns = -torch.tensor([9.0, 10.0, 11.0, 12.0])   # the whole group is bad
    baseline = rloo_baseline(returns, groups)
    assert (baseline < 0).all()
    advantage = returns - baseline
    assert (advantage > 0).any() and (advantage < 0).any(), "RLOO's own contrast takes both signs"

    final = -torch.rand(k, width) - 1.0
    mask = torch.ones(k, width, dtype=torch.bool)
    positive = (rloo_red_credit(final, advantage, mask).advantage > 0).float().mean()
    assert 0.0 < positive < 1.0, f"advantages collapsed onto one sign ({positive:.2f} positive)"

    # The retired rule, on the same state, does collapse -- so this test fails if
    # anyone reinstates it.
    retired = final - baseline[:, None]
    assert (retired > 0).all()


def test_lengthening_a_response_does_not_raise_its_mean_advantage():
    """R3's second channel, closed.

    The loss divides the credit by the response length but R3 did not divide the
    baseline, so padding out a negative-total response raised its mean advantage
    (``red250`` rollout 40: ``corr(len, sum_t r~) = -0.29`` against
    ``corr(len, mean_t A) = +0.28``).  A centred token term has nothing for the
    length to scale.
    """
    advantage = torch.tensor([-1.0])
    short = torch.tensor([[-2.0, 0.0, 1.0]])                      # per-token mean -1/3
    long = torch.tensor([[-2.0, 0.0, 1.0, -1.0, 0.0, 0.0]])       # per-token mean -1/3
    mask_short = torch.ones(1, 3, dtype=torch.bool)
    mask_long = torch.ones(1, 6, dtype=torch.bool)
    mean_short = rloo_red_credit(short, advantage, mask_short).advantage[mask_short].mean()
    mean_long = rloo_red_credit(long, advantage, mask_long).advantage[mask_long].mean()
    torch.testing.assert_close(mean_short, mean_long)
    torch.testing.assert_close(mean_long, advantage[0])


def test_replacing_the_return_with_reds_return_is_exactly_rloo():
    """The degenerate reading, pinned so it can never be used by accident."""
    torch.manual_seed(3)
    k, prompts = 4, 3
    batch = k * prompts
    groups = torch.arange(batch) // k
    raw = torch.randn(batch) * 3.0
    # R(x, empty) is a function of the prompt alone, hence shared by the group.
    dynamic_init = torch.tensor([0.7, -0.3, 1.1]).repeat_interleave(k)

    red_return = raw - dynamic_init                    # beta_c = 1, Eq. (6) summed
    standard = rloo_baseline(raw, groups)
    degenerate = rloo_baseline(red_return, groups)

    plain_rloo_advantage = raw - standard
    fed_red_return = red_return - degenerate
    torch.testing.assert_close(fed_red_return, plain_rloo_advantage, atol=1e-6, rtol=1e-6)


def test_r4_does_differ_from_plain_rloo():
    """The chosen rule must not collapse the way the rejected one does."""
    torch.manual_seed(5)
    k, prompts, width = 4, 2, 6
    batch = k * prompts
    groups = torch.arange(batch) // k
    final = torch.randn(batch, width)
    mask = torch.ones(batch, width, dtype=torch.bool)
    advantage = final.sum(-1) - rloo_baseline(final.sum(-1), groups)

    credit = rloo_red_credit(final, advantage, mask)
    # Plain RLOO would spread one scalar over every token; R4 does not.
    scalar = advantage[:, None].expand_as(final)
    assert not torch.allclose(credit.advantage, scalar)
    # And they agree on the response mean, which is what makes it a redistribution
    # rather than a different sequence-level estimator.
    torch.testing.assert_close(credit.advantage.mean(-1), scalar.mean(-1), atol=1e-6, rtol=0)


def test_sequence_return_carries_the_kl():
    shaped = torch.tensor([5.0, -1.0])
    kl = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    returns = sequence_returns(shaped, kl, mask, beta=0.5)
    torch.testing.assert_close(returns, torch.tensor([5.0 - 0.3, -1.0]))


def test_masked_positions_are_zero_and_do_not_leak_into_the_token_mean():
    """The centring must average over valid tokens only.

    A padded position holds a stale value from whatever occupied that column
    before, so counting it in the mean would shift every valid token's credit by a
    number that depends on the batch's padding.
    """
    final = torch.tensor([[1.0, 1.0, 99.0], [2.0, 99.0, 99.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    credit = rloo_red_credit(final, torch.zeros(2), mask)
    assert credit.advantage[0, 2] == 0 and credit.advantage[1, 1] == 0
    assert credit.direction[0, 2] == 0
    # Row 0's valid tokens are [1.0, 1.0], so the centred term is identically zero
    # and the 99.0 in the pad column contributes nothing.
    torch.testing.assert_close(credit.direction[0, :2], torch.zeros(2))
    # Row 1 has a single valid token, so its centred term is zero by definition.
    torch.testing.assert_close(credit.direction[1, 0], torch.tensor(0.0))


def test_credit_weight_has_unit_mean_over_valid_tokens():
    final = torch.tensor([[3.0, -1.0, 2.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    credit = rloo_red_credit(final, torch.zeros(2), mask)
    torch.testing.assert_close(credit.weight[mask].mean(), torch.tensor(1.0))


def test_a_response_with_no_positive_credit_falls_back_to_a_uniform_share():
    final = torch.tensor([[-1.0, -2.0, -3.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    credit = rloo_red_credit(final, torch.zeros(1), mask)
    torch.testing.assert_close(credit.weight, torch.ones(1, 3))


def test_default_beta_c_matches_the_paper():
    assert RED_BETA_C_DEFAULT == 1.0
