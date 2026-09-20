"""Eq. (2): the gradient estimator, checked against autograd and against its own definition."""
import pytest
import torch

from p2t.attribution import null_token_attribution


def _case(seed=11, V=13, D=6, B=2, T=5):
    torch.manual_seed(seed)
    weight = torch.randn(V, D, dtype=torch.float64)
    ids = torch.randint(V, (B, T))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    return weight, ids, mask


def test_attribution_matches_input_embedding_autograd_oracle():
    """I_i must be <grad_{e_i} R, e_i - e_null> for a coupled nonlinear reward."""
    weight, ids, mask = _case()
    null = 3
    embeddings = weight[ids].clone().requires_grad_(True)
    # Nonlinear and coupled across positions, so the gradient is not trivially
    # the per-position coefficient.
    reward = (embeddings * mask[..., None]).sum(1).tanh().square().sum(-1)
    grads = torch.autograd.grad(reward.sum(), embeddings)[0]
    expected = (grads * (embeddings.detach() - weight[null])).sum(-1)

    got = null_token_attribution(grads, ids, weight, null, mask)
    torch.testing.assert_close(got[mask], expected[mask].float(), atol=1e-5, rtol=1e-5)
    assert (got[~mask] == 0).all()
    assert not got.requires_grad


def test_taylor_attribution_is_exact_for_an_affine_reward_model():
    """When R is affine in E the first-order Taylor expansion has no error.

    This is the only setting where the paper's approximation (Eq. 2) can be
    compared against its own definition (Eq. 1) without an approximation gap.
    """
    weight, ids, mask = _case(seed=5)
    null = 2
    coefficients = torch.randn(*ids.shape, weight.shape[1], dtype=torch.float64)

    def reward_of(embeddings):
        return (embeddings * coefficients * mask[..., None]).sum()

    embeddings = weight[ids].clone().requires_grad_(True)
    reward = reward_of(embeddings)
    grads = torch.autograd.grad(reward, embeddings)[0]
    attribution = null_token_attribution(grads, ids, weight, null, mask)

    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            if not mask[b, t]:
                assert attribution[b, t] == 0
                continue
            replaced = weight[ids].clone()
            replaced[b, t] = weight[null]
            # Eq. (1) is R(E) - R(E_{t<-null}), so the replaced forward is the
            # subtrahend.  Reversing these is the exact sign error Eq. (2) already
            # carries once in the paper's own prose.
            exact = (reward - reward_of(replaced)).item()
            torch.testing.assert_close(attribution[b, t].item(), exact,
                                       atol=1e-5, rtol=1e-5)


def test_attribution_reads_the_null_row_not_the_pad_position():
    """Changing W[null] must change I; the null token is an embedding, not a slot."""
    weight, ids, mask = _case(seed=7)
    torch.manual_seed(3)
    grads = torch.randn(*ids.shape, weight.shape[1], dtype=torch.float64)
    baseline = null_token_attribution(grads, ids, weight, 4, mask)
    perturbed = weight.clone()
    perturbed[4] += 3.0
    moved = null_token_attribution(grads, ids, perturbed, 4, mask)
    assert not torch.allclose(baseline[mask], moved[mask])
    # A different null row must give a different answer too.
    assert not torch.allclose(baseline[mask],
                              null_token_attribution(grads, ids, weight, 5, mask)[mask])


def test_zero_gradient_positions_give_zero_attribution():
    """Unmapped tokens carry a zero gradient; I = 0 there, never an invented value."""
    weight, ids, mask = _case(seed=9)
    grads = torch.zeros(*ids.shape, weight.shape[1], dtype=torch.float64)
    grads[0, 2] = 1.0
    got = null_token_attribution(grads, ids, weight, 0, mask)
    assert (got[0, :2] == 0).all() and (got[0, 3:] == 0).all()
    assert got[0, 2] != 0


@pytest.mark.parametrize("kwargs,match", [
    (dict(rm_embedding_weight=torch.zeros(4, 3)), "width"),
    (dict(null_token_id=99), "index the RM embedding"),
    (dict(null_token_id=True), "Python int"),
    (dict(token_chunk_size=0), "token_chunk_size"),
])
def test_attribution_rejects_bad_inputs(kwargs, match):
    weight, ids, mask = _case()
    grads = torch.randn(*ids.shape, weight.shape[1], dtype=torch.float64)
    call = dict(input_grads=grads, token_ids=ids, rm_embedding_weight=weight,
                null_token_id=1, response_mask=mask, token_chunk_size=4)
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        null_token_attribution(**call)


def test_attribution_rejects_nonfinite_gradients_and_out_of_range_ids():
    weight, ids, mask = _case()
    grads = torch.randn(*ids.shape, weight.shape[1], dtype=torch.float64)
    grads[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        null_token_attribution(grads, ids, weight, 1, mask)
    with pytest.raises(ValueError, match="vocabulary"):
        null_token_attribution(torch.zeros_like(grads), ids + 100, weight, 1, mask)


def test_attribution_rejects_malformed_masks():
    weight, ids, mask = _case()
    grads = torch.randn(*ids.shape, weight.shape[1], dtype=torch.float64)
    with pytest.raises(ValueError, match="binary"):
        null_token_attribution(grads, ids, weight, 1, mask.long() * 2)
    with pytest.raises(ValueError, match="at least one valid token"):
        null_token_attribution(grads, ids, weight, 1, torch.zeros_like(mask))
