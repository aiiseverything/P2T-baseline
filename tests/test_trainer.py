import types
from unittest.mock import patch
import torch
from torch import nn

from vpo_rm.reward import LastTokenReward
from vpo_rm.trainer import VPOTrainer
from bytelevel_fixtures import ByteLevelTestTokenizer


class _Tok(ByteLevelTestTokenizer):
    def __init__(self):
        super().__init__({"<|endoftext|>": 0, "p": 1, "a": 2, "b": 3,
                          "c": 4, "d": 5, "e": 6, "<|im_end|>": 7})


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
    def get_input_embeddings(self):
        return self.emb
    def forward(self, inputs_embeds, attention_mask, position_ids=None, use_cache=False, return_dict=True):
        return types.SimpleNamespace(last_hidden_state=(inputs_embeds * attention_mask[..., None]).cumsum(1))


class _Score(nn.Module):
    def forward(self, hidden):
        return hidden.sum(-1, keepdim=True)


def _trainer(method):
    t = object.__new__(VPOTrainer)
    t.cfg = types.SimpleNamespace(method=method, microbatch_responses=1)
    t.reward_device = torch.device("cpu")
    t.actor_tokenizer = t.reward_tokenizer = _Tok()
    t.reward = LastTokenReward(_Backbone(), _Score())
    return t


def test_grpo_reward_is_forward_only_and_matches_gradient_path_score():
    responses = torch.tensor([[3, 4, 0], [5, 6, 0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    dummy = torch.zeros((2, 5), dtype=torch.long)
    mask = torch.ones_like(dummy)
    positions = torch.tensor([[2, 3, -1], [2, 3, -1]])

    torch.manual_seed(7)
    grpo = _trainer("grpo")
    with patch("vpo_rm.trainer.response_reward_gradients", side_effect=AssertionError("GRPO must not backpropagate")):
        grpo_scores, grpo_grads, *_ = grpo._reward_batch(dummy, mask, positions, responses, valid, ["a", "b"])
    torch.manual_seed(7)
    vpo = _trainer("vpo_rm")
    vpo_scores, vpo_grads, *_ = vpo._reward_batch(dummy, mask, positions, responses, valid, ["a", "b"])

    torch.testing.assert_close(grpo_scores, vpo_scores)
    assert grpo_grads is None
    assert vpo_grads.shape == (2, 3, 4)
    assert vpo_grads[valid].abs().sum() > 0
    assert all(p.grad is None for p in grpo.reward.parameters())


def test_chunked_old_logp_matches_dense_masked_reference():
    t = object.__new__(VPOTrainer)
    t.cfg = types.SimpleNamespace(microbatch_responses=1)
    t.actor = object()
    t.actor_device = torch.device("cpu")
    t.output_mask = torch.ones(5, dtype=torch.bool)
    logits = torch.tensor([[[1., 2., 3., 4., 5.], [2., 1., 0., -1., -2.]],
                           [[-1., 0., 1., 2., 3.], [3., 2., 1., 0., -1.]]])
    ids = torch.tensor([[3, 4], [0, 2]])
    actor_inputs = torch.tensor([[0], [1]])
    valid = torch.tensor([[True, True], [False, True]])
    def fake_logits(actor, input_ids, attention_mask, positions, response_mask, output_mask=None):
        return logits[input_ids[0, 0]:input_ids[0, 0] + input_ids.shape[0]]
    with patch("vpo_rm.trainer.actor_response_logits", side_effect=fake_logits):
        got = t._old_logp_microbatch(actor_inputs, actor_inputs.bool(), torch.zeros_like(ids), ids, valid)
    dense = logits.gather(-1, ids[..., None]).squeeze(-1) - logits.logsumexp(-1)
    torch.testing.assert_close(got, dense)
