import math

import pytest
import torch

from vpo_rm.core import group_advantages


def test_default_group_advantages_keeps_population_std_plus_eps():
    rewards = torch.tensor([1., 3., 9., 9.])
    groups = torch.tensor([2, 2, 7, 7])
    advantage, scale = group_advantages(rewards, groups, eps=.25)
    torch.testing.assert_close(scale, torch.tensor([1.25, 1.25, .25, .25]))
    torch.testing.assert_close(advantage, torch.tensor([-.8, .8, 0., 0.]))


def test_positive_floor_uses_exact_max_without_epsilon():
    rewards = torch.tensor([1., 3., 8., 8.])
    groups = torch.tensor([2, 2, 7, 7])
    advantage, scale = group_advantages(rewards, groups, eps=.25, std_floor=.5)
    torch.testing.assert_close(scale, torch.tensor([1., 1., .5, .5]))
    torch.testing.assert_close(advantage, torch.tensor([-1., 1., 0., 0.]))


@pytest.mark.parametrize('floor', [-1., math.nan, math.inf, -math.inf])
def test_invalid_std_floor_is_rejected(floor):
    with pytest.raises(ValueError, match='std_floor'):
        group_advantages(torch.tensor([0., 1.]), torch.tensor([0, 0]), std_floor=floor)


def test_one_token_length_difference_is_not_normalized_to_unit_advantage():
    # At L=1024 and 1025, max length cost 2*sigma0 gives delta=sigma0/512.
    sigma0 = 2.
    reward_gap = 2. * sigma0 / (2048 - 1024)
    rewards = torch.tensor([0., -reward_gap])
    groups = torch.zeros(2, dtype=torch.long)
    advantage, scale = group_advantages(rewards, groups, std_floor=.5 * sigma0)
    torch.testing.assert_close(scale, torch.ones(2))
    torch.testing.assert_close(advantage, torch.tensor([1. / 512., -1. / 512.]))
    assert advantage.abs().max() < .002


def test_fixed_floor_does_not_create_signal_for_identical_rewards():
    advantage, scale = group_advantages(torch.full((8,), -2.), torch.zeros(8, dtype=torch.long),
                                       std_floor=.5)
    assert advantage.eq(0).all()
    assert scale.eq(.5).all()


def test_floor_is_keyword_only_and_results_stay_detached():
    rewards = torch.tensor([1., 2.], requires_grad=True)
    groups = torch.zeros(2, dtype=torch.long)
    with pytest.raises(TypeError):
        group_advantages(rewards, groups, 1e-6, .5)
    advantage, scale = group_advantages(rewards, groups, std_floor=.5)
    assert not advantage.requires_grad and not scale.requires_grad
