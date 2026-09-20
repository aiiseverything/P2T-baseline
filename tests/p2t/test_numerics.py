"""Numerics and masking of Eq. (3): stability, degenerate modes, unmapped tokens."""
import pytest
import torch

from p2t.reward import p2t_token_reward


def test_max_subtraction_is_an_identity_against_a_float64_reference():
    """The implementation must equal a reference that never subtracts the max.

    Eq. (3) has no temperature, so I is an unbounded gradient inner product and
    a literal exp(I) overflows float32 past I > 88.7.  Subtracting the per-row
    max is algebraically exact; this test is what makes that claim checkable.
    """
    torch.manual_seed(2)
    mask = torch.ones(2, 6, dtype=torch.bool)
    attribution = torch.randn(2, 6) * 4.0
    rewards = torch.tensor([1.5, -2.5])
    got, _ = p2t_token_reward(attribution, rewards, mask, omega=0.6)

    reference = attribution.double()
    share = reference.exp() / reference.exp().sum(-1, keepdim=True)
    expected = rewards.double()[:, None] + 0.6 * rewards.double()[:, None] * share
    torch.testing.assert_close(got.double(), expected, atol=1e-6, rtol=1e-6)


def test_large_attribution_magnitudes_stay_finite():
    """I is an unnormalised inner product; the softmax must survive |I| ~ 1e4."""
    mask = torch.ones(1, 4, dtype=torch.bool)
    attribution = torch.tensor([[1e4, -1e4, 0.0, 5.0]])
    token_reward, share = p2t_token_reward(attribution, torch.tensor([3.0]), mask, omega=0.6)
    assert torch.isfinite(token_reward).all()
    assert torch.isfinite(share).all()
    torch.testing.assert_close(share.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    # The dominant token takes essentially the whole share: the one-hot mode.
    assert share[0, 0] > 0.99


def test_all_zero_attribution_collapses_to_a_uniform_share():
    """Zero gradients -> p = 1/T -> Eq. (3) becomes a per-response constant.

    That is the silent-degradation mode: the token term varies no more, and the
    arm behaves like GRPO with a shifted advantage.
    """
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    rewards = torch.tensor([2.0, -1.0])
    token_reward, share = p2t_token_reward(torch.zeros(2, 5), rewards, mask, omega=0.6)
    torch.testing.assert_close(share[0], torch.full((5,), 1 / 5))
    torch.testing.assert_close(share[1, :3], torch.full((3,), 1 / 3))
    assert (share[~mask] == 0).all()
    for row, count in enumerate([5, 3]):
        expected = rewards[row] * (1 + 0.6 / count)
        torch.testing.assert_close(token_reward[row, :count], expected.expand(count))
    assert torch.isfinite(token_reward).all()


def test_padding_never_enters_the_softmax():
    """Garbage at padded positions must not move any valid token's share."""
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    clean = torch.tensor([[0.5, -0.25, 0.0, 0.0]])
    dirty = torch.tensor([[0.5, -0.25, 1e9, -1e9]])
    rewards = torch.tensor([1.0])
    first, first_share = p2t_token_reward(clean, rewards, mask, omega=0.6)
    second, second_share = p2t_token_reward(dirty, rewards, mask, omega=0.6)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_share, second_share)


def test_unmapped_positions_get_zero_attribution_but_still_share_the_budget():
    """A response position the RM never saw carries I = 0 and stays in sum_j.

    The paper's normaliser runs over the response's tokens, so dropping those
    positions would renormalise Eq. (3) away from its definition.
    """
    mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    attribution = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # positions 1-3 unmapped
    rewards = torch.tensor([1.0])
    _, share = p2t_token_reward(attribution, rewards, mask, omega=0.6)
    assert (share[0, 1:] > 0).all()
    torch.testing.assert_close(share.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_single_token_response_gets_the_whole_share():
    mask = torch.ones(1, 1, dtype=torch.bool)
    token_reward, share = p2t_token_reward(torch.tensor([[7.0]]), torch.tensor([1.0]),
                                           mask, omega=0.6)
    torch.testing.assert_close(share, torch.ones(1, 1))
    torch.testing.assert_close(token_reward, torch.full((1, 1), 1.6))


def test_nonfinite_attribution_on_valid_positions_is_rejected():
    mask = torch.ones(1, 3, dtype=torch.bool)
    attribution = torch.tensor([[0.0, float("inf"), 0.0]])
    with pytest.raises(ValueError, match="finite"):
        p2t_token_reward(attribution, torch.tensor([1.0]), mask, omega=0.6)


def test_omega_zero_reproduces_the_plain_sequence_reward():
    mask = torch.ones(1, 4, dtype=torch.bool)
    token_reward, _ = p2t_token_reward(torch.randn(1, 4), torch.tensor([2.5]), mask, omega=0.0)
    torch.testing.assert_close(token_reward, torch.full((1, 4), 2.5))


def test_shape_and_mask_validation():
    attribution = torch.randn(2, 4)
    rewards = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        p2t_token_reward(attribution, rewards, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        p2t_token_reward(attribution, torch.tensor([1.0]), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="binary"):
        p2t_token_reward(attribution, rewards, torch.full((2, 4), 0.5))
