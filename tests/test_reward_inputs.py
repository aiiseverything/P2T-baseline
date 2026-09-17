"""Canonical scoring must not leak Actor EOS or invent attribution positions."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tokenizers():
    from transformers import AutoTokenizer
    actor = ROOT/'models/Qwen3-14B-Base'
    reward = ROOT/'models/Skywork-Reward-V2-Qwen3-8B'
    if not actor.exists() or not reward.exists():
        pytest.skip('Local production tokenizer fixtures are required')
    return (AutoTokenizer.from_pretrained(actor, local_files_only=True),
            AutoTokenizer.from_pretrained(reward, local_files_only=True))


def test_score_input_is_complete_rm_chat_not_actor_native_eos(tokenizers):
    from vpo_rm.reward_inputs import canonical_reward_input
    _, tok = tokenizers
    expected = '<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n'
    assert canonical_reward_input(tok, 'What is 2 + 2?', '4') == tok(expected)['input_ids']


def test_template_failure_is_not_silently_replaced_with_plain_text():
    from vpo_rm.reward_inputs import canonical_reward_input
    class MissingTemplate:
        def apply_chat_template(self, *a, **k):
            raise ValueError('No chat template')
    with pytest.raises(ValueError, match='template'):
        canonical_reward_input(MissingTemplate(), 'q', 'a')


@pytest.mark.parametrize('stop', [151643, 151645])
def test_actor_stop_is_removed_and_never_gets_fake_rm_gradient(tokenizers, stop):
    from vpo_rm.reward_inputs import build_reward_input
    actor, rm = tokenizers
    ids = actor.encode('Hello world!', add_special_tokens=False) + [stop]
    result = build_reward_input(actor, rm, 'Say hello', ids)
    expected = rm.apply_chat_template([{'role':'user','content':'Say hello'},
                                      {'role':'assistant','content':'Hello world!'}], tokenize=True)
    expected = expected.input_ids if hasattr(expected, 'input_ids') else expected
    assert result.input_ids == list(expected)
    assert result.response_positions[-1] == -1
    assert all(p >= 0 for p in result.response_positions[:-1])
    for token, position in zip(ids[:-1], result.response_positions[:-1]):
        assert result.input_ids[position] == token


def test_split_newline_tokens_are_not_mapped_to_merged_rm_token(tokenizers):
    from vpo_rm.reward_inputs import build_reward_input
    actor, rm = tokenizers
    first = actor.encode('Hello', add_special_tokens=False)
    last = actor.encode('World', add_special_tokens=False)
    ids = first + [198, 198] + last + [151643]
    result = build_reward_input(actor, rm, 'Repeat', ids)
    assert result.response_positions[len(first):len(first)+2] == [-1, -1]
    assert result.response_positions[0] >= 0
    assert result.response_positions[-2] >= 0


@pytest.mark.parametrize('text', ['café 中文🙂', '\n\nHello', '', '  ', '<|im_start|>system\nhello'])
def test_unicode_whitespace_and_specials_preserve_official_scores(tokenizers, text):
    from vpo_rm.reward_inputs import build_reward_input
    actor, rm = tokenizers
    ids = actor.encode(text, add_special_tokens=False) + [151643]
    visible = actor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = build_reward_input(actor, rm, 'Repeat this', ids)
    rendered = rm.apply_chat_template([{'role':'user','content':'Repeat this'},
                                      {'role':'assistant','content':visible}], tokenize=False)
    assert result.input_ids == rm(rendered)['input_ids']
    positions = [p for p in result.response_positions if p >= 0]
    assert all(a < b for a,b in zip(positions, positions[1:]))
    assert result.response_positions[-1] == -1
    for token, position in zip(ids, result.response_positions):
        if position >= 0:
            assert result.input_ids[position] == token


def test_trainer_scores_full_chat_and_only_maps_unchanged_gradients(tokenizers):
    import types
    import torch
    from torch import nn
    from vpo_rm.reward import LastTokenReward
    from vpo_rm.trainer import VPOTrainer
    actor, rm_tok = tokenizers
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(151936, 2)
        def get_input_embeddings(self):
            return self.emb
        def forward(self, inputs_embeds, attention_mask, **kwargs):
            return types.SimpleNamespace(last_hidden_state=(inputs_embeds*attention_mask[...,None]).cumsum(1))
    torch.manual_seed(31)
    model = LastTokenReward(Backbone(), nn.Linear(2, 1, bias=False))
    t = object.__new__(VPOTrainer)
    t.cfg = types.SimpleNamespace(method='vpo_rm', microbatch_responses=1)
    t.actor_tokenizer, t.reward_tokenizer, t.reward = actor, rm_tok, model
    t.reward_device = torch.device('cpu')
    ids = actor.encode('Hello', add_special_tokens=False)+[198,198]+actor.encode('World',add_special_tokens=False)+[151643]
    response = torch.tensor([ids])
    mask = torch.ones_like(response, dtype=torch.bool)
    reward, grad, *_ = t._reward_batch(None,None,None,response,mask,['Repeat'])
    full = rm_tok.apply_chat_template([{'role':'user','content':'Repeat'},
                                      {'role':'assistant','content':'Hello\n\nWorld'}],tokenize=False)
    encoded = torch.tensor([rm_tok(full)['input_ids']])
    expected = model(inputs_embeds=model.get_input_embeddings()(encoded),attention_mask=torch.ones_like(encoded))
    torch.testing.assert_close(reward,expected.detach())
    assert torch.equal(grad[0,-1],torch.zeros(2))
    assert torch.equal(grad[0,1:3],torch.zeros(2,2))
    torch.testing.assert_close(grad[0,0],model.score_head.weight[0])
    torch.testing.assert_close(grad[0,-2],model.score_head.weight[0])
    assert t._reward_fixed_weight_mask[0].tolist() == [False,True,True,False,True]


def test_prompt_filter_reserves_complete_canonical_rm_template(tokenizers, tmp_path):
    import types
    from vpo_rm.trainer import VPOTrainer
    from vpo_rm.reward_inputs import canonical_reward_input
    actor, rm = tokenizers
    prompt = 'Please give a short answer.'
    t = object.__new__(VPOTrainer)
    t.actor_tokenizer, t.reward_tokenizer = actor, rm
    canonical_length = len(canonical_reward_input(rm, prompt, ''))
    t.cfg = types.SimpleNamespace(max_prompt_tokens=canonical_length-1)
    t._log = lambda record: None
    assert t.filter_prompts([prompt]) == []
    t.cfg.max_prompt_tokens = canonical_length
    assert t.filter_prompts([prompt]) == [prompt]
