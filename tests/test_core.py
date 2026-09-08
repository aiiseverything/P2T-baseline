import pytest
import torch
from vpo_rm import allocate, compute_credit, group_advantages, grpo_policy_loss


@pytest.mark.parametrize('token_chunk,vocab_chunk', [(1, 1), (3, 5), (128, 8192)])
def test_credit_matches_explicit_st_autograd(token_chunk, vocab_chunk):
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 11, requires_grad=True)
    weight = torch.randn(11, 5)
    ids = torch.randint(11, (2, 4))
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    p = logits.softmax(-1)
    soft = p @ weight
    e = soft + (weight[ids] - soft).detach()
    # Coupled nonlinear sequence reward, ensuring gradients include other positions.
    reward = (e * mask[..., None]).sum(1).tanh().square().sum(-1)
    f, g = torch.autograd.grad(reward.sum(), (e, logits))
    sigma = torch.tensor([.8, 1.2])
    score = torch.nn.functional.one_hot(ids, 11) - p.detach()
    expected = (g * score).sum(-1) / sigma[:, None]
    credit = compute_credit(logits, ids, f, weight, torch.tensor([1., -1.]), sigma,
                            mask, .7, token_chunk, vocab_chunk)
    torch.testing.assert_close(credit.direction[mask], expected[mask], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(credit.weight.sum(-1), mask.sum(-1).float())
    assert not credit.advantage.requires_grad
    assert (credit.advantage[~mask] == 0).all()


def test_direction_finite_difference():
    torch.manual_seed(13)
    logits = torch.randn(1, 3, 7, dtype=torch.double)
    W = torch.randn(7, 4, dtype=torch.double)
    ids = torch.tensor([[2, 3, 1]])
    hard = W[ids].requires_grad_()
    reward = hard.sum(1).sin().sum()
    f, = torch.autograd.grad(reward, hard)
    mask = torch.ones_like(ids, dtype=torch.bool)
    c = compute_credit(logits, ids, f, W, torch.ones(1), torch.ones(1), mask, 1.)
    p = logits.softmax(-1)
    k = torch.nn.functional.one_hot(ids, 7) - p
    for t in range(3):
        delta = torch.zeros_like(logits)
        delta[:, t] = 1e-5 * k[:, t]
        def R(delta):
            e = hard.detach() + (torch.softmax(logits + delta, -1) - p) @ W
            return e.sum(1).sin().sum()
        fd = (R(delta) - R(-delta)) / 2e-5
        torch.testing.assert_close(c.direction[0,t].double(), fd, atol=2e-6, rtol=2e-5)


def test_group_stats_and_zero_advantage():
    a, s = group_advantages(torch.tensor([2., 5., 4., 5.]), torch.tensor([8, 4, 8, 4]))
    torch.testing.assert_close(a, torch.tensor([-1., 0., 1., 0.]), atol=2e-6, rtol=1e-5)
    c = allocate(torch.randn(4, 3), a, torch.ones(4, 3, dtype=torch.bool), .1)
    torch.testing.assert_close(c.advantage.mean(-1), a)
    torch.testing.assert_close(c.weight[1], torch.ones(3))
    with pytest.raises(ValueError, match='two responses'):
        group_advantages(torch.tensor([1.]), torch.tensor([0]))


def test_entropy_allocation_optimality_for_both_signs():
    torch.manual_seed(5)
    d = torch.randn(2, 6)
    a = torch.tensor([1., -1.])
    tau = .8
    q = allocate(d, a, torch.ones_like(d, dtype=torch.bool), tau).weight / 6
    r = torch.randn_like(d).softmax(-1)
    def F(x):
        return (x * (a[:, None] * d - tau * (6*x).log())).sum(-1)
    gap = tau * (r * (r / q).log()).sum(-1)
    torch.testing.assert_close(F(q) - F(r), gap)


def test_loss_clipping_stopgrad_and_padding():
    new = torch.tensor([[0., .5, float('nan')], [-.5, .5, 0.]], requires_grad=True)
    old = torch.zeros_like(new, requires_grad=True)
    adv = torch.tensor([[2., 2., float('nan')], [-1., -1., -1.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    loss = grpo_policy_loss(new, old, adv, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all()
    assert old.grad is None and adv.grad is None
    assert new.grad[0, 1] == 0 and new.grad[0, 2] == 0
    assert new.grad[1, 0] == 0 and new.grad[1, 1] > 0


def test_empty_response_rejected():
    with pytest.raises(ValueError, match='at least one'):
        allocate(torch.zeros(1, 2), torch.ones(1), torch.zeros(1, 2), 1.)


def test_masked_vocabulary_matches_supported_softmax():
    torch.manual_seed(81)
    z = torch.randn(1, 2, 6)
    z[..., [0, 5]] = -torch.inf
    W, f = torch.randn(6, 3), torch.randn(1, 2, 3)
    ids, mask = torch.tensor([[1,4]]), torch.ones(1,2, dtype=torch.bool)
    c = compute_credit(z, ids, f, W, torch.ones(1), torch.ones(1), mask, 1.,
                        vocab_chunk_size=1)
    compact = compute_credit(z[...,1:5], ids-1, f, W[1:5], torch.ones(1),
                              torch.ones(1), mask, 1.)
    torch.testing.assert_close(c.direction, compact.direction)
    with pytest.raises(ValueError, match='output support'):
        compute_credit(z, torch.tensor([[0,4]]), f, W, torch.ones(1),
                        torch.ones(1), mask, 1.)
