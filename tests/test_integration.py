import pytest
import torch
from torch import nn
from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM,
                          GPTNeoXForSequenceClassification,
                          Qwen3Config, Qwen3ForSequenceClassification)
from vpo_rm import (LastTokenReward, reward_input_gradients, check_tokenizers,
                    actor_response_logits, response_reward_gradients,
                    group_advantages, build_credit_cache, actor_policy_loss,
                    gather_response, MeanStepReward)


def neox_config():
    return GPTNeoXConfig(vocab_size=19, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=1, num_attention_heads=4,
                         max_position_embeddings=64, pad_token_id=0,
                         num_labels=1, attention_dropout=0., hidden_dropout=0.)


@pytest.mark.parametrize('family', ['neox', 'qwen3'])
def test_rm_padding_pooling_and_frozen_parameter_gradients(family):
    torch.manual_seed(31)
    if family == 'neox':
        model = GPTNeoXForSequenceClassification(neox_config()).eval()
    else:
        config = Qwen3Config(vocab_size=19, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             pad_token_id=0, num_labels=1)
        model = Qwen3ForSequenceClassification(config).eval()
    adapter = LastTokenReward(model.base_model, model.score)
    ids = torch.tensor([[0, 0, 2, 3, 4], [2, 3, 4, 0, 0]])
    mask = ids != 0
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask,
                         position_ids=(mask.long().cumsum(-1)-1).clamp_min(0)).logits[:,0]
    with torch.inference_mode():
        rewards, grads = reward_input_gradients(adapter, ids, mask)
    torch.testing.assert_close(rewards, expected)
    torch.testing.assert_close(rewards[0], rewards[1])
    assert grads.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in adapter.parameters())


def test_full_rollout_cache_and_actor_update():
    torch.manual_seed(3)
    actor = GPTNeoXForCausalLM(neox_config()).eval()
    rm = GPTNeoXForSequenceClassification(neox_config()).eval()
    scorer = LastTokenReward(rm.base_model, rm.score)
    # Actor and RM have different prompt lengths and padding placements.
    ids = torch.tensor([[0, 2, 3, 4, 5], [0, 2, 3, 6, 7]])
    mask = ids != 0
    rids = torch.tensor([[8, 2, 3, 4, 5, 0], [8, 2, 3, 6, 7, 0]])
    rmask = rids != 0
    positions = torch.tensor([[3, 4], [3, 4]])
    tokens = torch.tensor([[4, 5], [6, 7]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    rewards, f = response_reward_gradients(scorer, rids, rmask, positions,
                                          tokens, valid)
    advantage, sigma = group_advantages(rewards, torch.tensor([10, 10]))
    output_mask = torch.ones(19, dtype=torch.bool)
    output_mask[[0, 18]] = False
    with torch.no_grad():
        old_logits = actor_response_logits(actor, ids, mask, positions, valid,
                                           output_mask=output_mask)
    cache = build_credit_cache(old_logits, tokens, f, scorer.get_input_embeddings().weight,
                               advantage, sigma, valid, tau=1., vocab_chunk_size=7)
    before = actor.get_output_embeddings().weight.detach().clone()
    opt = torch.optim.SGD(actor.parameters(), lr=.01)
    opt.zero_grad()
    loss = actor_policy_loss(actor, ids, mask, positions, tokens, valid, cache,
                              output_mask=output_mask)
    loss.backward()
    assert actor.get_output_embeddings().weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(before, actor.get_output_embeddings().weight)
    assert all(p.grad is None for p in scorer.parameters())
    with pytest.raises(ValueError, match='IDs or valid positions'):
        response_reward_gradients(scorer, rids, rmask, positions-1, tokens, valid)


def test_prm_marker_reduction():
    from types import SimpleNamespace
    class PRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(9, 4)
            self.head = nn.Linear(4, 2)
        def get_input_embeddings(self):
            return self.embed
        def forward(self, *, inputs_embeds, **kwargs):
            return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))
    scorer = MeanStepReward(PRM())
    ids = torch.tensor([[1,2,3,4]])
    mask = torch.ones_like(ids)
    steps = torch.tensor([[0,1,0,1]])
    r, f = reward_input_gradients(scorer, ids, mask, step_mask=steps)
    with torch.no_grad():
        expected = scorer.token_model(inputs_embeds=scorer.get_input_embeddings()(ids)).logits
        expected = expected.softmax(-1)[0, [1,3], 1].mean()
    torch.testing.assert_close(r[0], expected)
    assert f[0,0].abs().sum() > 0


def test_token_id_identity():
    class Tokenizer:
        pad_token_id = 0
        def __init__(self, vocab): self.vocab = vocab
        def get_vocab(self): return self.vocab
    tok = Tokenizer({'PAD': 0, 'a': 1, 'b': 2})
    check_tokenizers(tok, tok, 4, 4)
    with pytest.raises(ValueError, match='token-to-ID'):
        check_tokenizers(tok, Tokenizer({'PAD': 0, 'a': 2, 'b': 1}), 4, 4)
    with pytest.raises(ValueError, match='rows'):
        check_tokenizers(tok, tok, 4, 5)


def test_gather_padding_sentinel():
    values = torch.arange(12).reshape(1, 4, 3)
    actual = gather_response(values, torch.tensor([[2,-1]]), torch.tensor([[1,0]]))
    torch.testing.assert_close(actual, torch.tensor([[[6,7,8],[0,0,0]]]))
