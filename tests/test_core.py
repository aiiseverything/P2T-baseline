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


@pytest.mark.parametrize('outlier,tau', [(1., 1.), (-1., .3), (-1., .001)])
def test_lambda_band_handles_long_responses_and_small_temperature(outlier, tau):
    a, _ = group_advantages(torch.tensor([1.] + [0.] * 7), torch.zeros(8, dtype=torch.long))
    d = torch.zeros(1, 2048)
    d[0, 0] = outlier
    c = allocate(d, a[:1], torch.ones_like(d, dtype=torch.bool), tau, credit_lambda=2.)
    assert c.weight.max() <= 2. + 1e-6
    assert c.weight.min() >= .5 - 1e-6
    torch.testing.assert_close(c.weight.sum(-1), torch.tensor([2048.]))


def test_lambda_band_still_holds_after_freezing_terminal_token():
    d = torch.tensor([[4., 0., 0., 0., 5.]])
    ids = torch.tensor([[5, 5, 5, 5, 151643]])
    c = allocate(d, torch.ones(1), torch.ones_like(ids, dtype=torch.bool), 1.,
                 credit_lambda=2., token_ids=ids, freeze_stop_tokens=True)
    assert c.weight.max() <= 2. + 1e-6
    assert c.weight.min() >= .5 - 1e-6
    assert c.weight[0, -1] == 1.
    torch.testing.assert_close(c.weight.sum(-1), torch.tensor([5.]))


def test_lambda_one_zeros_padding_credit():
    c = allocate(torch.tensor([[1., float('nan')]]), torch.tensor([2.]),
                 torch.tensor([[True, False]]), 1., credit_lambda=1.)
    torch.testing.assert_close(c.weight, torch.tensor([[1., 0.]]))
    torch.testing.assert_close(c.advantage, torch.tensor([[2., 0.]]))


def test_positive_clipped_ratio_does_not_overflow_in_backward():
    new = torch.tensor([[-1.]], requires_grad=True)
    loss = grpo_policy_loss(new, torch.tensor([[-101.]]), torch.ones(1, 1),
                            torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    assert loss.item() == pytest.approx(-1.2)
    assert new.grad.item() == 0.


def test_unrepresentable_negative_advantage_loss_fails_before_backward():
    with pytest.raises(ValueError, match='finite|overflow'):
        grpo_policy_loss(torch.tensor([[-1.]], requires_grad=True),
                         torch.tensor([[-101.]]), -torch.ones(1, 1),
                         torch.ones(1, 1, dtype=torch.bool))


def test_structural_freezing_requires_verified_ids():
    with pytest.raises(ValueError, match='structural_token_ids'):
        allocate(torch.ones(1, 2), torch.ones(1), torch.ones(1, 2, dtype=torch.bool),
                 1., token_ids=torch.tensor([[5687, 198]]), freeze_structural=True)


def test_tokenizer_derived_freezing_preserves_content_and_all_frozen_rows():
    d = torch.tensor([[2., -1., 10., 5.], [1., 2., 3., 0.]])
    ids = torch.tensor([[5687, 8, 7, 9], [7, 9, 7, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    c = allocate(d, torch.ones(2), mask, 1., token_ids=ids, freeze_structural=True,
                 stop_token_ids=(9,), structural_token_ids=(7,))
    assert c.weight[0, 0] > 1. and c.weight[0, 1] < 1.
    torch.testing.assert_close(c.weight[0, 2:], torch.ones(2))
    torch.testing.assert_close(c.weight[1], torch.tensor([1., 1., 1., 0.]))
    torch.testing.assert_close(c.weight.sum(-1), mask.sum(-1).float())


def test_sub_float32_lambda_band_has_a_finite_uniform_solution():
    d = torch.arange(41).float()[None, :]
    c = allocate(d, torch.ones(1), torch.ones_like(d, dtype=torch.bool), 1.,
                 credit_lambda=1.000000001)
    torch.testing.assert_close(c.weight, torch.ones_like(d), atol=0., rtol=0.)
    assert torch.isfinite(c.tau_used).all()


def test_credit_scales_bfloat16_policy_logits_in_float32_chunks():
    torch.manual_seed(17)
    z = (torch.randn(2, 4, 9) * 3.).bfloat16()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    f, embedding = torch.randn(2, 4, 3), torch.randn(9, 3)
    a, scale = torch.tensor([1., -1.]), torch.tensor([.8, 1.2])
    expected = compute_credit(z.float() / .7, tokens, f, embedding, a, scale, mask, 1.,
                              token_chunk_size=2, vocab_chunk_size=3)
    actual = compute_credit(z, tokens, f, embedding, a, scale, mask, 1.,
                            token_chunk_size=2, vocab_chunk_size=3, policy_temperature=.7)
    torch.testing.assert_close(actual.direction, expected.direction)
    torch.testing.assert_close(actual.weight, expected.weight)


def _random_credit_case():
    torch.manual_seed(0)
    a = torch.tensor([1.5, -0.7, 0.0])
    mask = torch.ones(3, 40, dtype=torch.bool)
    mask[1, 30:] = False
    ids = torch.randint(100, 5000, (3, 40))
    ids[0, -1] = 151643
    ids[1, 29] = 151645
    fixed = torch.zeros_like(mask)
    fixed[0, 3] = True
    return a, mask, ids, fixed


@pytest.mark.parametrize("source", ["random_direction", "random_band"])
def test_random_credit_keeps_band_budget_freezing_and_sign_without_rm_information(source):
    from vpo_rm.core import random_credit
    a, mask, ids, fixed = _random_credit_case()
    kwargs = dict(credit_lambda=4.0, source=source, token_ids=ids, freeze_stop_tokens=True,
                  fixed_weight_mask=fixed)
    credit = random_credit(a, mask, 1.0, generator=torch.Generator().manual_seed(123), **kwargs)
    w = credit.weight
    assert w[~mask].eq(0).all()
    assert float(w[mask].max()) <= 4.0 and float(w[mask].min()) >= 0.25
    torch.testing.assert_close(w.sum(-1), mask.sum(-1).float())
    torch.testing.assert_close(credit.advantage.sum(-1) / mask.sum(-1), a)
    assert torch.sign(credit.advantage[mask]).eq(torch.sign(a)[:, None].expand_as(w)[mask]).all()
    for row, column in ((0, -1), (1, 29), (0, 3)):
        assert w[row, column] == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(w[2], mask[2].float())          # zero advantage: uniform
    assert float((w[0] - 1).abs().max()) > 0.2         # genuinely non-uniform elsewhere
    assert float((w[1][mask[1]] - 1).abs().max()) > 0.2
    again = random_credit(a, mask, 1.0, generator=torch.Generator().manual_seed(123), **kwargs)
    torch.testing.assert_close(again.weight, w)
    other = random_credit(a, mask, 1.0, generator=torch.Generator().manual_seed(124), **kwargs)
    assert not torch.equal(other.weight, w)
    assert credit.tau_used.shape == (3,) and torch.isfinite(credit.tau_used).all()
    if source == "random_direction":
        # The direction is the noise itself; the unchanged allocator reproduces the weights.
        assert credit.direction[mask].abs().sum() > 0
        replay = allocate(credit.direction, a, mask, 1.0, credit_lambda=4.0, token_ids=ids,
                          freeze_stop_tokens=True, fixed_weight_mask=fixed)
        torch.testing.assert_close(replay.weight, w)
        torch.testing.assert_close(replay.tau_used, credit.tau_used)
    else:
        assert torch.equal(credit.direction, torch.zeros_like(w))
        assert torch.equal(credit.tau_used, torch.ones(3))
        free = mask & ~fixed & ~torch.isin(ids, torch.tensor([151643, 151645]))
        inside = (w > 0.25 + 1e-6) & (w < 4.0 - 1e-6) & free
        assert inside[0].any() and inside[1].any()


def test_random_band_credit_degenerates_to_uniform_at_lambda_one_or_single_free_token():
    from vpo_rm.core import random_credit
    a, mask, ids, _ = _random_credit_case()
    generator = torch.Generator().manual_seed(5)
    one = random_credit(a, mask, 1.0, credit_lambda=1.0, source="random_band", generator=generator)
    torch.testing.assert_close(one.weight, mask.float())
    single = torch.zeros(1, 4, dtype=torch.bool)
    single[0, :2] = True
    frozen = torch.zeros_like(single)
    frozen[0, 0] = True
    credit = random_credit(torch.tensor([2.]), single, 1.0, credit_lambda=4.0, source="random_band",
                           generator=generator, fixed_weight_mask=frozen)
    torch.testing.assert_close(credit.weight, single.float())


def test_random_credit_rejects_bad_sources_generators_and_frozen_ids():
    from vpo_rm.core import random_credit
    a, mask, ids, _ = _random_credit_case()
    generator = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="source"):
        random_credit(a, mask, 1.0, source="rm_gradient", generator=generator)
    with pytest.raises(ValueError, match="Generator"):
        random_credit(a, mask, 1.0, generator=None)
    with pytest.raises(ValueError, match="advantage"):
        random_credit(a[:2], mask, 1.0, generator=generator)
    for source in ("random_direction", "random_band"):
        with pytest.raises(ValueError, match="token_ids"):
            random_credit(a, mask, 1.0, source=source, generator=generator, freeze_stop_tokens=True)
        with pytest.raises(ValueError, match="structural"):
            random_credit(a, mask, 1.0, source=source, generator=generator, token_ids=ids,
                          freeze_structural=True)
