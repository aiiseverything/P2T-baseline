"""Reward-model batching: the gradient must come back aligned to actor tokens.

This is the seam the first review found broken: a chunk is padded to its own
canonical reward-model width, so the raw embedding gradient is ``[B, L_rm, D]``
with a per-chunk ``L_rm``.  It has to be gathered onto the actor's response
positions before anything downstream can use it, and that gather is also what
zeroes the attribution at positions the reward model never saw.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from p2t.rm import score_responses


class _RMModule(nn.Module):
    """Differentiable scalar reward with a real embedding table."""

    def __init__(self, vocab=19, dim=8):
        super().__init__()
        torch.manual_seed(3)
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, 1)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        hidden = torch.tanh(inputs_embeds * attention_mask[..., None]).cumsum(1)
        pooled = hidden[torch.arange(hidden.shape[0]), attention_mask.sum(-1) - 1]
        return self.head(pooled)[:, 0]


def _tokenizer():
    return SimpleNamespace(pad_token_id=0)


def test_gradients_are_gathered_onto_actor_positions():
    """Rows of different canonical widths must still yield a dense [B, T, D]."""
    reward_model = _RMModule()
    rows = [[1, 2, 3, 4, 5, 6], [7, 8, 9]]          # widths 6 and 3
    mapped = torch.tensor([[1, 2, 3], [1, 2, -1]], dtype=torch.long)
    responses = torch.tensor([[2, 3, 4], [8, 9, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    rewards, grads = score_responses(reward_model, _tokenizer(), rows, mapped, responses, mask,
                                     device=torch.device("cpu"), microbatch=1)
    assert rewards.shape == (2,)
    assert grads.shape == (2, 3, reward_model.embedding.embedding_dim)
    assert torch.isfinite(rewards).all() and torch.isfinite(grads).all()
    # A position with no reward-model counterpart carries exactly zero gradient,
    # which is what makes I = 0 there rather than an invented value.
    assert (grads[1, 2] == 0).all()


def test_gradient_matches_a_direct_autograd_oracle():
    """The gathered rows must be the true dR/de at the mapped RM positions."""
    reward_model = _RMModule()
    rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mapped = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)
    responses = torch.tensor([[2, 3], [6, 8]], dtype=torch.long)
    mask = torch.ones(2, 2, dtype=torch.bool)

    _, grads = score_responses(reward_model, _tokenizer(), rows, mapped, responses, mask,
                               device=torch.device("cpu"), microbatch=2)

    weight = reward_model.embedding.weight
    embeddings = weight[torch.tensor(rows)].clone().requires_grad_(True)
    attention = torch.ones(2, 4).long()
    reward = reward_model(inputs_embeds=embeddings, attention_mask=attention)
    full = torch.autograd.grad(reward.sum(), embeddings)[0]
    expected = torch.stack([full[0, mapped[0]], full[1, mapped[1]]])
    torch.testing.assert_close(grads, expected.detach(), atol=1e-6, rtol=1e-5)


def test_microbatch_partitioning_does_not_change_the_result():
    """The physical microbatch is one in the main configuration; it must be inert."""
    rows = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
    mapped = torch.tensor([[1, 2, 3], [1, -1, -1], [1, 2, -1]], dtype=torch.long)
    responses = torch.tensor([[2, 3, 4], [6, 0, 0], [8, 9, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)

    outputs = []
    for micro in (1, 2, 3):
        torch.manual_seed(3)
        model = _RMModule()
        outputs.append(score_responses(model, _tokenizer(), rows, mapped, responses, mask,
                                       device=torch.device("cpu"), microbatch=micro))
    for rewards, grads in outputs[1:]:
        torch.testing.assert_close(rewards, outputs[0][0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grads, outputs[0][1], atol=1e-6, rtol=1e-6)


def test_mismatched_response_tokens_are_rejected():
    """The RM must be fed the actor's own tokens at the mapped positions."""
    reward_model = _RMModule()
    rows = [[1, 2, 3, 4]]
    mapped = torch.tensor([[1, 2]], dtype=torch.long)
    wrong = torch.tensor([[9, 9]], dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="differ between Actor and RM"):
        score_responses(reward_model, _tokenizer(), rows, mapped, wrong, mask,
                        device=torch.device("cpu"), microbatch=1)
