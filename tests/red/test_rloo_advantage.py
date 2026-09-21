"""RLOO's leave-one-out baseline and the R3 advantage rule.

The paper gives a full PPO recipe and *no* RLOO details, so the rule this arm
uses is a choice, and ``RED_REPRO_NOTES.md`` 2.2 records it.  The load-bearing
test here is ``test_replacing_the_return_with_reds_return_is_exactly_rloo``: it
turns the argument for that choice into an executable fact.  RED's return is
``R(x, y) - R(x, empty)``, the dynamic-initialisation offset ``R(x, empty)`` is a
per-prompt constant, and a leave-one-out mean differences constants away -- so
feeding RED's return into RLOO's estimator reproduces plain RLOO exactly and the
arm would measure nothing.
"""
from __future__ import annotations

import pytest
import torch

from red.reward import (RED_BETA_C_DEFAULT, rloo_baseline, rloo_red_credit,
                        sequence_returns)


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


def test_device_mismatch_between_reward_and_baseline_is_reported_clearly():
    final = torch.zeros(1, 3, device="meta")
    baseline = torch.zeros(1)
    with pytest.raises(ValueError, match="baseline is on"):
        rloo_red_credit(final, baseline, torch.ones(1, 3, dtype=torch.bool))


def test_r3_advantage_is_the_token_reward_minus_the_baseline():
    final = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5]])
    baseline = torch.tensor([4.0, -1.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    credit = rloo_red_credit(final, baseline, mask)
    torch.testing.assert_close(credit.advantage, final - baseline[:, None])
    torch.testing.assert_close(credit.direction, final)


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


def test_r3_does_differ_from_plain_rloo():
    """The chosen rule must not collapse the way the rejected one does."""
    torch.manual_seed(5)
    k, prompts, width = 4, 2, 6
    batch = k * prompts
    groups = torch.arange(batch) // k
    final = torch.randn(batch, width)
    mask = torch.ones(batch, width, dtype=torch.bool)
    baseline = rloo_baseline(final.sum(-1), groups)

    credit = rloo_red_credit(final, baseline, mask)
    # Plain RLOO would spread one scalar over every token; R3 does not.
    scalar = (final.sum(-1) - baseline)[:, None].expand_as(final)
    assert not torch.allclose(credit.advantage, scalar)


def test_sequence_return_carries_the_kl():
    shaped = torch.tensor([5.0, -1.0])
    kl = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    returns = sequence_returns(shaped, kl, mask, beta=0.5)
    torch.testing.assert_close(returns, torch.tensor([5.0 - 0.3, -1.0]))


def test_masked_positions_are_zero_and_do_not_leak_into_the_baseline():
    final = torch.tensor([[1.0, 1.0, 99.0], [2.0, 99.0, 99.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    baseline = torch.tensor([0.0, 0.0])
    credit = rloo_red_credit(final, baseline, mask)
    assert credit.advantage[0, 2] == 0 and credit.advantage[1, 1] == 0
    assert credit.direction[0, 2] == 0


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
