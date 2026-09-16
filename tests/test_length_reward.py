import math

import pytest
import torch

from vpo_rm.length_reward import (
    calibrate_reward_scale,
    response_degeneracy,
    soft_length_penalties,
)


def test_length_penalties_at_contract_boundaries_and_midpoint():
    lengths = torch.tensor([0, 7, 8, 1024, 1536, 2048])
    short, long = soft_length_penalties(lengths, 2.)
    torch.testing.assert_close(short, torch.tensor([1., .125, 0., 0., 0., 0.]))
    torch.testing.assert_close(long, torch.tensor([0., 0., 0., 0., 2., 4.]))
    assert short.device == lengths.device == long.device


def test_custom_penalty_thresholds_and_disabled_strengths():
    lengths = torch.tensor([0., 2., 4., 6., 8., 10.])
    short, long = soft_length_penalties(lengths, .5, short_threshold=4,
                                      long_threshold=6, max_length=10,
                                      short_strength=2., long_strength=3.)
    torch.testing.assert_close(short, torch.tensor([1., .5, 0., 0., 0., 0.]))
    torch.testing.assert_close(long, torch.tensor([0., 0., 0., 0., .75, 1.5]))
    short, long = soft_length_penalties(lengths, .5, short_strength=0., long_strength=0.)
    assert short.eq(0).all() and long.eq(0).all()


def test_equal_rm_scores_do_not_reward_padding_from_eight_to_soft_limit():
    lengths = torch.tensor([7, 8, 900, 1024, 1536, 2048])
    short, long = soft_length_penalties(lengths, 1.)
    shaped = torch.full((6,), 5.) - short - long
    assert shaped[1] == shaped[2] == shaped[3]
    assert shaped[1] > shaped[0] > shaped[4] > shaped[5]


@pytest.mark.parametrize('sigma0', [0., -1., math.inf, -math.inf, math.nan])
def test_invalid_reward_scale_is_rejected(sigma0):
    with pytest.raises(ValueError, match='sigma0'):
        soft_length_penalties(torch.tensor([8]), sigma0)


@pytest.mark.parametrize('kwargs', [
    {'short_threshold': 0}, {'short_threshold': -1},
    {'short_threshold': 1025}, {'long_threshold': 2048},
    {'long_threshold': 7}, {'max_length': 1024},
    {'short_threshold': math.nan}, {'long_threshold': math.inf},
    {'max_length': math.inf}, {'short_strength': -.1},
    {'short_strength': math.nan}, {'long_strength': -1.},
    {'long_strength': math.inf},
])
def test_invalid_penalty_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        soft_length_penalties(torch.tensor([8]), 1., **kwargs)


@pytest.mark.parametrize('lengths', [
    torch.tensor(8), torch.tensor([[8]]), torch.tensor([-1.]),
    torch.tensor([2049.]), torch.tensor([math.nan]), torch.tensor([math.inf]),
])
def test_invalid_response_lengths_are_rejected(lengths):
    with pytest.raises(ValueError, match='lengths'):
        soft_length_penalties(lengths, 1.)


def test_empty_length_batch_remains_empty():
    short, long = soft_length_penalties(torch.empty(0), 1.)
    assert short.shape == long.shape == (0,)


def test_degeneracy_separates_empty_text_from_contiguous_newline_runs():
    texts = ['', ' \t\u2003 ', 'answer', '\n' * 32,
             'answer' + '\n' * 31, 'answer' + '\n' * 32,
             '\n' * 16 + 'answer' + '\n' * 16]
    empty, repeated = response_degeneracy(texts)
    assert empty == [True, True, False, True, False, False, False]
    assert repeated == [False, False, False, True, False, True, False]
    assert response_degeneracy(['a\n\n'], newline_run=2) == ([False], [True])
    assert response_degeneracy([]) == ([], [])


@pytest.mark.parametrize('newline_run', [0, -1, 1.5, True])
def test_invalid_newline_threshold_is_rejected(newline_run):
    with pytest.raises(ValueError, match='newline_run'):
        response_degeneracy(['answer'], newline_run=newline_run)


@pytest.mark.parametrize('texts', ['answer', [None], [1]])
def test_degeneracy_requires_a_sequence_of_strings(texts):
    with pytest.raises(ValueError, match='texts'):
        response_degeneracy(texts)


def test_calibration_uses_population_std_and_usual_even_median():
    # Eligible group stds are 1 and 3. A lone valid sample is not a group estimate.
    rewards = torch.tensor([0., 2., 0., 6., 5., math.nan])
    groups = torch.tensor([10, 10, 20, 20, 30, 30])
    valid = torch.tensor([True, True, True, True, True, False])
    assert calibrate_reward_scale(rewards, groups, valid) == pytest.approx(2.)


def test_calibration_keeps_zero_std_groups_and_uses_odd_median():
    rewards = torch.tensor([3., 3., 0., 2., 0., 8.])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    assert calibrate_reward_scale(rewards, groups, torch.ones(6, dtype=torch.bool)) == pytest.approx(1.)


@pytest.mark.parametrize('rewards,groups,valid', [
    ([], [], []), ([1.], [0], [True]), ([1., 3.], [0, 0], [False, False]),
    ([2., 2.], [0, 0], [True, True]),
    ([0., 0., 1., 1., 0., 4.], [0, 0, 1, 1, 2, 2], [True] * 6),
    ([math.nan, 3.], [0, 0], [True, True]),
    ([math.inf, 3.], [0, 0], [True, True]),
])
def test_calibration_rejects_missing_or_nonpositive_finite_scale(rewards, groups, valid):
    with pytest.raises(ValueError):
        calibrate_reward_scale(torch.tensor(rewards), torch.tensor(groups, dtype=torch.long),
                               torch.tensor(valid, dtype=torch.bool))


@pytest.mark.parametrize('rewards,groups,valid', [
    (torch.tensor([[0., 2.]]), torch.tensor([[0, 0]]), torch.tensor([[True, True]])),
    (torch.tensor([0., 2.]), torch.tensor([0]), torch.tensor([True, True])),
    (torch.tensor([0., 2.]), torch.tensor([0, 0]), torch.tensor([True])),
    (torch.tensor([0., 2.]), torch.tensor([0, 0]), torch.tensor([0, 2])),
    (torch.tensor([0., 2.]), torch.tensor([0., math.nan]), torch.tensor([True, True])),
])
def test_calibration_rejects_invalid_shapes_masks_and_group_ids(rewards, groups, valid):
    with pytest.raises(ValueError):
        calibrate_reward_scale(rewards, groups, valid)
