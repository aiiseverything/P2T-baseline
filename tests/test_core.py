import pytest
import torch
from vpo_rm import allocate, compute_credit, group_advantages, grpo_policy_loss, \
    guard_degenerate_rewards


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


def test_freeze_stop_tokens():
    """EOS-freeze pins stop-token weights at exactly 1 and redistributes the
    freed credit to the highest-d content token (the artifact 100x cliff is
    removed; see the p9l2 position-wise |d| measurement)."""
    torch.manual_seed(11)
    d = torch.randn(3, 32)
    a = torch.ones(3)
    mask = torch.ones(3, 32, dtype=torch.bool)
    # Put stop tokens at various positions (last, middle, multiple).
    ids = torch.randint(100, 5000, (3, 32))
    ids[0, -1] = 151643  # <|endoftext|> at end
    ids[1, 10] = 151645  # <|im_end|> in middle
    ids[2, 5] = 151643; ids[2, 20] = 151645  # two stop tokens
    frozen = allocate(d, a, mask, 1.0, credit_lambda=4.0,
                      token_ids=ids, freeze_stop_tokens=True)
    unfrozen = allocate(d, a, mask, 1.0, credit_lambda=4.0)
    # Stop tokens must be exactly 1 in the frozen version.
    assert frozen.weight[0, -1] == pytest.approx(1.0, abs=1e-5)
    assert frozen.weight[1, 10] == pytest.approx(1.0, abs=1e-5)
    assert frozen.weight[2, 5] == pytest.approx(1.0, abs=1e-5)
    assert frozen.weight[2, 20] == pytest.approx(1.0, abs=1e-5)
    # Unfrozen version does NOT pin them (they can be anything but 1 for generic d).
    assert not all(float(unfrozen.weight[i, j]) == pytest.approx(1.0, abs=1e-6)
                   for i, j in [(0, -1), (1, 10)])
    # Sum invariant: both allocate the same total credit.
    torch.testing.assert_close(frozen.weight.sum(-1), unfrozen.weight.sum(-1))
    # Advantage mean preserved.
    torch.testing.assert_close(frozen.advantage.mean(-1), a)


def test_entropy_allocation_optimality_for_both_signs():
    torch.manual_seed(5)
    d = torch.randn(2, 6)
    a = torch.tensor([1., -1.])
    tau = .8
    credit = allocate(d, a, torch.ones_like(d, dtype=torch.bool), tau, credit_lambda=50.0)
    q = credit.weight / 6
    r = torch.randn_like(d).softmax(-1)
    # Optimality holds for the standardized utility at the SOLVED temperature
    # (the lambda band only raises tau when the natural softmax is too peaked;
    # with lambda=50 here it stays at the requested tau).
    mean = d.mean(-1, keepdim=True)
    std = d.std(-1, correction=0, keepdim=True)
    floor = 1e-3 * d.abs().mean(-1, keepdim=True) + torch.finfo(torch.float32).tiny
    z = (d - mean) / (std + floor)
    t = credit.tau_used
    def F(x):
        return (x * (a[:, None] * z - t[:, None] * (6*x).log())).sum(-1)
    gap = t * (r * (r / q).log()).sum(-1)
    torch.testing.assert_close(F(q) - F(r), gap)


def test_standardized_allocation_engages_and_bands():
    torch.manual_seed(3)
    # The p9c regime: tiny-magnitude directions flattened softmax to uniform.
    # Per-response standardization must engage regardless of absolute scale.
    d = torch.tensor([[1e-3, 1.2e-3, 0.8e-3]])
    a = torch.tensor([1.0])
    c = allocate(d, a, torch.ones(1, 3, dtype=torch.bool), 1.0)
    torch.testing.assert_close(c.weight.sum(-1), torch.tensor([3.0]))
    assert c.weight.max() > 1.1 and c.weight.min() < 0.9
    torch.testing.assert_close(c.advantage.mean(-1), a)
    # Scale-free: the same relative shape at 1000x magnitude gives identical
    # weights (sigma-style unit normalization is also cancelled exactly).
    c2 = allocate(d * 1000, a, torch.ones(1, 3, dtype=torch.bool), 1.0)
    torch.testing.assert_close(c2.weight, c.weight)
    # Degenerate direction (all tokens equal): falls back to uniform GRPO.
    c3 = allocate(torch.full((1, 4), 5e-3), torch.ones(1), torch.ones(1, 4, dtype=torch.bool), 1.0)
    torch.testing.assert_close(c3.weight, torch.ones(1, 4))
    # Lambda band: one dominant token in a long response stays within the band
    # while the per-response mean of the token advantage is preserved.  The
    # band must bind (adaptive tau raised) for this extreme distribution.
    dd = torch.zeros(1, 64)
    dd[0, 0] = 1.0
    c4 = allocate(dd, torch.ones(1), torch.ones(1, 64, dtype=torch.bool), 0.05)
    assert c4.weight.max() <= 2.0 * 1.001
    assert c4.weight.min() >= 0.5 - 1e-3
    assert float(c4.tau_used[0]) > 0.05  # tau was raised off the requested 0.05
    torch.testing.assert_close(c4.weight.sum(-1), torch.tensor([64.0]))
    torch.testing.assert_close(c4.advantage.mean(-1), torch.ones(1))
    # Golden identity: lambda = 1 recovers exact GRPO uniform weights.
    c5 = allocate(torch.randn(4, 32), torch.randn(4), torch.ones(4, 32, dtype=torch.bool),
                  0.3, credit_lambda=1.0)
    torch.testing.assert_close(c5.weight, torch.ones(4, 32))


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


def test_degenerate_reward_guard():
    # Group 0: one empty response among real ones; it must rank strictly last.
    rewards = torch.tensor([7.0, 7.5, 6.8, 7.2])
    lengths = torch.tensor([1, 40, 35, 28])
    groups = torch.tensor([0, 0, 0, 0])
    patched, n = guard_degenerate_rewards(rewards, lengths, groups, min_length=8, penalty=1.0)
    assert n == 1
    assert patched[0] == pytest.approx(6.8 - 1.0)
    assert torch.equal(patched[1:], rewards[1:])
    assert patched[0] < patched[1:].min()  # never wins its group
    # Group 1: every response degenerate -> equal rewards, zero advantage.
    rewards2 = torch.tensor([7.0, 7.5])
    lengths2 = torch.tensor([1, 2])
    groups2 = torch.tensor([1, 1])
    patched2, n2 = guard_degenerate_rewards(rewards2, lengths2, groups2, min_length=8, penalty=1.0)
    assert n2 == 2 and torch.equal(patched2, patched2[:1].expand(2))
    # Mixed batch across groups; untouched groups pass through unchanged.
    rewards3 = torch.tensor([7.0, 7.5, 3.0, 3.5])
    lengths3 = torch.tensor([50, 45, 60, 55])
    groups3 = torch.tensor([2, 2, 3, 3])
    patched3, n3 = guard_degenerate_rewards(rewards3, lengths3, groups3)
    assert n3 == 0 and torch.equal(patched3, rewards3)
    # Overlong (truncated) responses flagged via also_floor rank last too.
    rewards4 = torch.tensor([7.0, 7.5, 6.0])
    lengths4 = torch.tensor([2048, 500, 480])
    groups4 = torch.tensor([4, 4, 4])
    patched4, n4 = guard_degenerate_rewards(rewards4, lengths4, groups4,
                                            min_length=8, penalty=1.0,
                                            also_floor=lengths4 >= 2047)
    assert n4 == 1 and patched4[0] == pytest.approx(6.0 - 1.0)


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
