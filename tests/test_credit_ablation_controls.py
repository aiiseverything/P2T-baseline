"""Controls preserve budgets and replace only the intended mathematical operation."""
import pytest
import torch
from vpo_rm.core import allocate, compute_credit, shuffle_credit

@pytest.mark.parametrize("chunks", [(1, 2), (4, 19)])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_norm_product_matches_independent_autograd(chunks, temperature):
    torch.manual_seed(917)
    logits = torch.randn(2, 4, 11, requires_grad=True)
    w = torch.randn(11, 5)
    ids = torch.tensor([[1, 3, 7, 8], [3, 5, 9, 1]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    scaled = logits / temperature
    p = scaled.softmax(-1)
    soft = p @ w
    emb = soft + (w[ids] - soft).detach()
    reward = (emb * mask[..., None]).sum(1).tanh().square().sum()
    f, g = torch.autograd.grad(reward, (emb, scaled))
    k = torch.nn.functional.one_hot(ids, 11) - p.detach()
    sigma = torch.tensor([.8, 1.2])
    expected = g.norm(dim=-1) * k.norm(dim=-1) / sigma[:, None]
    credit = compute_credit(logits, ids, f, w, torch.tensor([1., -1.]), sigma,
        mask, 1., *chunks, credit_lambda=4., policy_temperature=temperature,
        contraction="norm_product")
    torch.testing.assert_close(credit.direction[mask], expected[mask], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(credit.weight.sum(-1), mask.sum(-1).float())
    assert (credit.direction >= 0).all()
    assert (credit.advantage[1, mask[1]] <= 0).all()
    assert (credit.weight[~mask] == 0).all()


def test_shuffle_preserves_response_multisets_and_independent_rng():
    mask = torch.tensor([[1,1,1,1,1,1], [1,1,1,0,0,0]], dtype=torch.bool)
    d = torch.tensor([[1.,2.,3.,4.,5.,6.], [8.,6.,4.,0.,0.,0.]])
    ids = torch.tensor([[0,1,2,3,4,5], [0,1,5,0,0,0]])
    original = allocate(d, torch.tensor([2.,-3.]), mask, .5, credit_lambda=4.,
                        token_ids=ids, freeze_stop_tokens=True, stop_token_ids=(5,))
    generator = torch.Generator().manual_seed(0)
    global_state = torch.random.get_rng_state().clone()
    result = shuffle_credit(original, mask, generator=generator)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert not torch.equal(result.weight, original.weight)
    for i in range(2):
        assert torch.equal(result.weight[i,mask[i]].sort().values, original.weight[i,mask[i]].sort().values)
        assert torch.equal(result.advantage[i,mask[i]].sort().values, original.advantage[i,mask[i]].sort().values)
    assert (result.weight[~mask] == 0).all()
    # Stops participate after allocation: their original unit weights move.
    assert result.weight[0,-1] != 1
    replay = shuffle_credit(original, mask, generator=torch.Generator().manual_seed(0))
    assert torch.equal(replay.weight, result.weight)


def test_norm_product_zero_gradient_and_masked_support():
    logits = torch.tensor([[[float('-inf'), 1., -1.], [float('-inf'), 80., -80.]]])
    ids = torch.tensor([[1, 1]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    result = compute_credit(logits, ids, torch.zeros(1,2,4), torch.randn(3,4),
        torch.ones(1), torch.ones(1), mask, 1., contraction="norm_product")
    assert torch.equal(result.direction, torch.zeros(1,2))
    assert torch.equal(result.weight, torch.ones(1,2))
