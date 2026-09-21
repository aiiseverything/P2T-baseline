"""The reward-model seam: prefix scores, and the identity they must satisfy.

RED's whole claim is that an off-the-shelf sequence reward model can be read at
*every* position by applying its existing scalar head to the per-position hidden
states.  Two properties have to hold for that to be a redistribution rather than
a different reward:

* ``prefix_scores(...)[b, pooled[b]]`` must equal what ``forward`` returns, or the
  sequence score the sibling arms use and the redistributed rewards disagree.
* the gathered token rewards must sum to ``R_phi(x, y) - R_phi(x, empty)``.

Both are pinned here against a tiny causal model, no real checkpoint needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from red.rm import LastTokenReward, score_prefixes


class _TinyBackbone(nn.Module):
    """Causally-mixed encoder: position ``j`` sees tokens ``<= j`` only.

    A cumulative mean is the cheapest causal, prefix-dependent map there is, so a
    prefix score is a genuinely different number at every position and the
    differences are not all equal.
    """

    def __init__(self, vocab: int = 23, dim: int = 8):
        super().__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab, dim)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, use_cache=False, return_dict=True):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        steps = torch.arange(1, inputs_embeds.shape[1] + 1,
                             device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        hidden = torch.cumsum(inputs_embeds, dim=1) / steps.view(1, -1, 1)
        return SimpleNamespace(last_hidden_state=hidden)


def _reward(vocab: int = 23, dim: int = 8) -> LastTokenReward:
    backbone = _TinyBackbone(vocab, dim)
    head = nn.Linear(dim, 1)
    torch.manual_seed(11)
    nn.init.normal_(head.weight)
    nn.init.normal_(head.bias)
    return LastTokenReward(backbone, head)


def _batch(rows, pad=0):
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for j, row in enumerate(rows):
        ids[j, :len(row)] = torch.tensor(row, dtype=torch.long)
        attention[j, :len(row)] = 1
    return ids, attention


def test_prefix_scores_agree_with_forward_at_the_pooled_position():
    reward = _reward()
    ids, attention = _batch([[4, 5, 6, 7, 8, 9], [3, 4, 5, 0, 0, 0]])
    scores, pooled = reward.prefix_scores(input_ids=ids, attention_mask=attention)
    embeddings = reward.get_input_embeddings()(ids)
    forward = reward(inputs_embeds=embeddings, attention_mask=attention)
    rows = torch.arange(ids.shape[0])
    torch.testing.assert_close(scores[rows, pooled], forward, rtol=1e-6, atol=1e-6)


def test_prefix_scores_are_per_position_and_not_all_equal():
    reward = _reward()
    ids, attention = _batch([[4, 5, 6, 7, 8, 9]])
    scores, pooled = reward.prefix_scores(input_ids=ids, attention_mask=attention)
    assert scores.shape == (1, 6)
    assert pooled.item() == 5
    assert scores[0, :6].unique().numel() > 1, "a per-position head must vary by position"


def test_pooled_position_ignores_right_padding():
    reward = _reward()
    ids, attention = _batch([[4, 5, 6, 7], [3, 4]])
    _, pooled = reward.prefix_scores(input_ids=ids, attention_mask=attention)
    assert pooled.tolist() == [3, 1]


def test_score_prefixes_redistributes_exactly_the_sequence_score():
    """The identity: sum_t r~_t = R_phi(x, y) - R_phi(x, empty)."""
    reward = _reward()
    # prompt tokens 1..5, response tokens 6..8 at positions 5..7, plus one
    # trailing special at position 8 that the reward model pools at.
    rows = [[1, 2, 3, 4, 5, 6, 7, 8]]
    mapped = torch.tensor([[5, 6, 7]])
    responses = torch.tensor([[6, 7, 8]])       # rows[0][mapped] -- the actor's tokens
    mask = torch.ones(1, 3, dtype=torch.bool)
    tokenizer = SimpleNamespace(pad_token_id=0)

    rewards, token_rewards = score_prefixes(reward, tokenizer, rows, mapped, responses,
                                            mask, device=torch.device("cpu"))
    ids, attention = _batch(rows)
    scores, pooled = reward.prefix_scores(input_ids=ids, attention_mask=attention)

    want = scores[0, pooled[0]] - scores[0, 4]          # 4 = first mapped position - 1
    torch.testing.assert_close(token_rewards.sum(), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rewards[0], scores[0, pooled[0]], rtol=1e-6, atol=1e-6)


def test_score_prefixes_zeroes_tokens_the_reward_model_never_saw():
    reward = _reward()
    rows = [[1, 2, 3, 4, 5, 6, 7, 8]]
    mapped = torch.tensor([[5, -1, 7]])
    responses = torch.tensor([[6, 0, 8]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    tokenizer = SimpleNamespace(pad_token_id=0)
    _, token_rewards = score_prefixes(reward, tokenizer, rows, mapped, responses,
                                      mask, device=torch.device("cpu"))
    assert token_rewards[0, 1] == 0


def test_an_entirely_unmapped_response_contributes_zeros_instead_of_aborting():
    """An immediate-stop response is one special token, and specials never map.

    Such a row is unmappable by construction, and the trainer trains on it on
    purpose (``_degeneracy`` flags it and floors its shaped reward), so it must
    come back as zeros.  Masking the redistribution with
    ``response_mask & positions.ge(0)`` emptied the row and tripped the
    "at least one valid token" check, aborting the whole run; the actor mask is
    the right one, and unmapped positions are already zero by construction.
    """
    reward = _reward()
    rows = [[1, 2, 3, 4, 5, 2]]
    mapped = torch.tensor([[-1]])
    responses = torch.tensor([[2]])          # the stop token, nothing else
    mask = torch.ones(1, 1, dtype=torch.bool)
    tokenizer = SimpleNamespace(pad_token_id=0)

    rewards, token_rewards = score_prefixes(reward, tokenizer, rows, mapped, responses,
                                            mask, device=torch.device("cpu"))
    assert torch.isfinite(rewards).all()
    assert (token_rewards == 0).all(), "an unmappable row must contribute nothing"
    # The sequence score is still read, so the baseline and the sparse term hold.
    ids, attention = _batch(rows)
    scores, pooled = reward.prefix_scores(input_ids=ids, attention_mask=attention)
    torch.testing.assert_close(rewards[0], scores[0, pooled[0]], rtol=1e-6, atol=1e-6)


def test_an_unmapped_row_does_not_poison_the_rest_of_the_batch():
    """The failing row must not take its neighbours down with it."""
    reward = _reward()
    rows = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 2]]
    mapped = torch.tensor([[4, 5], [-1, -1]])
    responses = torch.tensor([[5, 6], [2, 2]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    tokenizer = SimpleNamespace(pad_token_id=0)
    _, token_rewards = score_prefixes(reward, tokenizer, rows, mapped, responses,
                                      mask, device=torch.device("cpu"))
    assert (token_rewards[1] == 0).all()
    assert token_rewards[0].abs().sum() > 0, "the mapped row must still be redistributed"


def test_score_prefixes_microbatch_is_invariant():
    reward = _reward()
    rows = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]]
    mapped = torch.tensor([[4, 5], [4, 5], [4, 5]])
    responses = torch.tensor([[5, 6], [6, 7], [7, 8]])   # rows[i][mapped]
    mask = torch.ones(3, 2, dtype=torch.bool)
    tokenizer = SimpleNamespace(pad_token_id=0)
    one = score_prefixes(reward, tokenizer, rows, mapped, responses, mask,
                         device=torch.device("cpu"), microbatch=1)
    three = score_prefixes(reward, tokenizer, rows, mapped, responses, mask,
                           device=torch.device("cpu"), microbatch=3)
    torch.testing.assert_close(one[0], three[0])
    torch.testing.assert_close(one[1], three[1])


def test_forward_rejects_a_non_scalar_head():
    reward = _reward()
    reward.score_head = nn.Linear(8, 3)
    ids, attention = _batch([[4, 5, 6]])
    with pytest.raises(ValueError):
        reward(inputs_embeds=reward.get_input_embeddings()(ids), attention_mask=attention)


def test_prefix_scores_rejects_a_non_scalar_head():
    reward = _reward()
    reward.score_head = nn.Linear(8, 3)
    ids, attention = _batch([[4, 5, 6]])
    with pytest.raises(ValueError):
        reward.prefix_scores(input_ids=ids, attention_mask=attention)


def test_prefix_scores_rejects_a_row_with_no_valid_token():
    reward = _reward()
    ids = torch.tensor([[4, 5, 6]])
    attention = torch.zeros_like(ids)
    with pytest.raises(ValueError):
        reward.prefix_scores(input_ids=ids, attention_mask=attention)
