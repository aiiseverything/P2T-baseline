"""Fixed credit for Actor tokens without an exact canonical RM correspondence."""
import pytest
import torch

from vpo_rm.core import allocate, compute_credit
from vpo_rm.integration import build_credit_cache


def test_fixed_weights_union_with_stops_and_preserve_each_response_budget():
    direction = torch.tensor([[-1., 0., 100., -100., 0.], [7., 8., 9., 0., 0.]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    fixed = torch.tensor([[0, 0, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    ids = torch.tensor([[2, 3, 4, 0, 0], [2, 3, 0, 0, 0]])
    credit = allocate(direction, torch.tensor([1., -1.]), mask, .1,
                      token_ids=ids, freeze_stop_tokens=True, stop_token_ids=(0,),
                      fixed_weight_mask=fixed)
    assert credit.weight[0, 0] < 1 < credit.weight[0, 1]
    torch.testing.assert_close(credit.weight[0, 2:], torch.tensor([1., 1., 0.]))
    torch.testing.assert_close(credit.weight[1], torch.tensor([1., 1., 1., 0., 0.]))
    torch.testing.assert_close(credit.weight.sum(-1), torch.tensor([4., 3.]))
    assert credit.weight[mask].min() >= .5 and credit.weight[mask].max() <= 2.


@pytest.mark.parametrize('fixed', [
    torch.tensor([[True]]),
    torch.tensor([[0, 1]]),
    torch.tensor([[False, True]]),
])
@pytest.mark.parametrize('credit_lambda', [1., 2.])
def test_fixed_weight_mask_requires_boolean_shape_and_valid_token_subset(fixed, credit_lambda):
    with pytest.raises(ValueError, match='fixed_weight_mask'):
        allocate(torch.ones(1, 2), torch.ones(1), torch.tensor([[True, False]]), 1.,
                 credit_lambda=credit_lambda, fixed_weight_mask=fixed)


@pytest.mark.parametrize('token_chunk,vocab_chunk', [(1, 1), (128, 8192)])
def test_fixed_mask_reaches_tiled_credit_and_public_cache(token_chunk, vocab_chunk):
    torch.manual_seed(41)
    logits = torch.randn(2, 4, 7)
    ids = torch.tensor([[1, 2, 3, 4], [1, 3, 5, 0]])
    gradients, embeddings = torch.randn(2, 4, 3), torch.randn(7, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    fixed = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.bool)
    advantages, scales = torch.tensor([1., -1.]), torch.tensor([.5, 2.])
    kwargs = dict(token_chunk_size=token_chunk, vocab_chunk_size=vocab_chunk,
                  fixed_weight_mask=fixed)
    credit = compute_credit(logits, ids, gradients, embeddings, advantages, scales, mask, 1., **kwargs)
    cache = build_credit_cache(logits, ids, gradients, embeddings, advantages, scales, mask, 1., **kwargs)
    torch.testing.assert_close(credit.weight[fixed], torch.ones(int(fixed.sum())))
    torch.testing.assert_close(credit.weight[~mask], torch.zeros(int((~mask).sum())))
    torch.testing.assert_close(credit.weight.sum(-1), mask.sum(-1).float())
    torch.testing.assert_close(cache.credit.weight, credit.weight)
    expected = logits.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1).masked_fill(~mask, 0)
    torch.testing.assert_close(cache.old_logp, expected)
