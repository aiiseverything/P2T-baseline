"""Actor and generation must preserve the exact SFT chat-token protocol."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scripts import profile_vllm_full as profile
from vpo_rm import token_policy
from vpo_rm.alignment import shared_output_mask
from vpo_rm.trainer import VPOTrainer


def tokenizer(*, with_bos=True, llama_pad=False):
    vocab = {'[UNK]': 0, '[BOS]': 1, '[USER]': 2, '[ASSISTANT]': 3,
             '[END]': 4, 'Hello': 5, 'world': 6, '[PAD]': 7}
    if llama_pad:
        vocab['<|finetune_right_pad_id|>'] = 8
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_bos:
        backend.post_processor = processors.TemplateProcessing(
            single='[BOS] $A', special_tokens=[('[BOS]', 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        bos_token='[BOS]', eos_token='[END]', pad_token='[PAD]',
        additional_special_tokens=['[USER]', '[ASSISTANT]'], padding_side='left')
    tok.chat_template = (("{{ bos_token }}" if with_bos else '')
        + "{% for m in messages %}[USER]{{ m['content'] }}[END]{% endfor %}"
          "{% if add_generation_prompt %}[ASSISTANT]{% endif %}")
    return tok


def prompt_encoder(tok, maximum=100):
    trainer = object.__new__(VPOTrainer)
    trainer.actor_tokenizer = tok
    trainer.actor_device = torch.device('cpu')
    trainer.cfg = SimpleNamespace(max_prompt_tokens=maximum)
    return trainer


@pytest.mark.parametrize('with_bos', [True, False])
def test_rendered_chat_encoder_preserves_native_prefix_and_left_padding(with_bos):
    trainer = prompt_encoder(tokenizer(with_bos=with_bos))
    batch, rendered = trainer._encode_prompts(['Hello', 'Hello world'])
    expected = [[1, 2, 5, 4, 3], [1, 2, 5, 6, 4, 3]] if with_bos else [
        [2, 5, 4, 3], [2, 5, 6, 4, 3]]
    assert [row[mask.bool()].tolist() for row, mask in zip(
        batch['input_ids'], batch['attention_mask'])] == expected
    assert batch['input_ids'][0, 0] == 7 and batch['attention_mask'][0, 0] == 0


def test_prompt_at_native_limit_keeps_assistant_header_and_overlimit_is_rejected():
    trainer = prompt_encoder(tokenizer(), maximum=5)
    batch, _ = trainer._encode_prompts(['Hello'])
    assert batch['input_ids'].tolist() == [[1, 2, 5, 4, 3]]
    with pytest.raises(ValueError, match='prompt|Prompt'):
        trainer._encode_prompts(['Hello world'])


@pytest.mark.parametrize('supports_thinking', [True, False])
def test_tokenized_chat_prompt_preserves_list_contract_with_tf5_default(supports_thinking):
    class DefaultDictTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                return_dict=True, **kwargs):
            if not supports_thinking and 'enable_thinking' in kwargs:
                raise TypeError('enable_thinking is unsupported')
            assert messages == [{'role': 'user', 'content': 'Hello'}]
            assert add_generation_prompt
            if not tokenize:
                return '[BOS][USER]Hello[END][ASSISTANT]'
            return {'input_ids': [1, 2, 5, 4, 3]} if return_dict else [1, 2, 5, 4, 3]
    tok = DefaultDictTokenizer()
    assert VPOTrainer._render_chat_prompt(tok, 'Hello', tokenize=True) == [1, 2, 5, 4, 3]
    assert VPOTrainer._render_chat_prompt(tok, 'Hello') == '[BOS][USER]Hello[END][ASSISTANT]'


def test_real_llama_actor_prefix_matches_native_sft_and_fixed_date():
    source = Path('/data/VPO-RM/models/Llama-3.1-8B-Instruct')
    if not (source / 'tokenizer.json').exists():
        pytest.skip('Local Llama tokenizer unavailable')
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tok.pad_token = '<|finetune_right_pad_id|>'
    batch, rendered = prompt_encoder(tok)._encode_prompts(['Say hello.'])
    ids = batch['input_ids'][0].tolist()
    assert len(ids) == 38 and ids.count(128000) == 1
    assert ids[-4:] == [128006, 78191, 128007, 271]
    assert 'Today Date: 26 Jul 2024' in rendered[0]
    assert ids == tok.apply_chat_template([{'role': 'user', 'content': 'Say hello.'}],
        tokenize=True, add_generation_prompt=True, return_dict=False)


def test_model_padding_uses_llama_dedicated_pad_and_keeps_qwen_shared_eos_contract():
    configure = getattr(token_policy, 'configure_model_padding', None)
    assert callable(configure), 'Model-aware RL padding helper is missing'
    llama = tokenizer(llama_pad=True)
    assert configure(llama) == 8
    assert llama.eos_token_id == 4
    assert not shared_output_mask(llama, 9)[8] and shared_output_mask(llama, 9)[4]
    actor, reward = tokenizer(with_bos=False), tokenizer(with_bos=False)
    reward.eos_token = '[BOS]'
    assert configure(actor) == 4
    assert configure(reward, fallback_token=actor.pad_token) == 4
    assert reward.eos_token_id == 1
    # A reward-owned dedicated pad always takes precedence over an actor fallback.
    assert configure(llama, fallback_token='[END]') == 8


def test_saved_actor_tokenizer_is_loaded_and_corruption_never_falls_back(tmp_path):
    load = getattr(token_policy, 'load_actor_tokenizer', None)
    assert callable(load), 'Saved initialization tokenizer loader is missing'
    base, saved = tmp_path / 'base', tmp_path / 'adapter'
    tokenizer(with_bos=False).save_pretrained(base)
    saved_tok = tokenizer(llama_pad=True)
    saved_tok.pad_token = '<|finetune_right_pad_id|>'
    saved_tok.save_pretrained(saved)
    actual = load(str(base), str(saved))
    assert actual.chat_template == saved_tok.chat_template
    assert actual.pad_token_id == 8
    (saved / 'sft_manifest.json').write_text(json.dumps({'token_protocol': {
        'bos_token_id': 1, 'pad_token_id': 8, 'response_eos_id': 4,
        'stop_token_ids': [4], 'chat_template_sha256': 'incorrect'}}))
    with pytest.raises(ValueError, match='token|template|protocol'):
        load(str(base), str(saved))
    (saved / 'sft_manifest.json').unlink()
    (saved / 'tokenizer_config.json').write_text('{invalid')
    with pytest.raises((ValueError, OSError)):
        load(str(base), str(saved))
    empty = tmp_path / 'empty-adapter'; empty.mkdir()
    assert load(str(base), str(empty)).chat_template == tokenizer(with_bos=False).chat_template


def test_explicit_tokenizer_source_and_initialization_source_reach_generation_server(tmp_path):
    saved = tmp_path / 'sft'; tokenizer().save_pretrained(saved)
    args = profile.parse_args(['--output-dir', 'unused', '--model', 'base',
                               '--init-adapter', str(saved)])
    command = profile.vllm_server_command(args, '/tmp/server.sock')
    assert '--tokenizer' in command, 'Generation server ignores the initialization tokenizer'
    assert command[command.index('--tokenizer') + 1] == str(saved)
    args = profile.parse_args(['--output-dir', 'unused', '--model', 'base',
                               '--init-adapter', str(saved), '--tokenizer', 'explicit'])
    assert profile.build_trainer_config(args, 'unused').tokenizer_name == 'explicit'
    command = profile.vllm_server_command(args, '/tmp/server.sock')
    assert command[command.index('--tokenizer') + 1] == 'explicit'


def test_generation_uses_exact_chat_ids_and_rejects_unbound_supplied_prefixes():
    prepare = getattr(token_policy, 'tokenize_rendered_prompts', None)
    assert callable(prepare), 'Explicit generation prompt helper is missing'
    tok = tokenizer()
    rendered = ['[BOS][USER]Hello[END][ASSISTANT]']
    expected = [{'prompt_token_ids': [1, 2, 5, 4, 3]}]
    assert prepare(tok, rendered) == expected
    assert prepare(tok, rendered, [[1, 2, 5, 4, 3]]) == expected
    for wrong in ([[1, 1, 2, 5, 4, 3]], [], [[]], [[1, True]], [[99]]):
        with pytest.raises(ValueError, match='prompt|token'):
            prepare(tok, rendered, wrong)


@pytest.mark.parametrize('engine_changes_prefix', [False, True])
def test_server_uses_saved_tokenizer_for_support_and_explicit_prompt_ids(
        tmp_path, monkeypatch, engine_changes_prefix):
    import io
    import sys
    from transformers import AutoConfig
    from scripts import vllm_generate_server as server

    tok = tokenizer(llama_pad=True)
    request = {'prompts': ['[BOS][USER]Hello[END][ASSISTANT]'],
               'prompt_token_ids': [[1, 2, 5, 4, 3]], 'adapter': 'saved',
               'adapter_id': 1, 'max_tokens': 5, 'min_tokens': 0,
               'group_size': 1, 'return_logprobs': True}
    sent, observed = [], {}
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def makefile(self, mode):
            return io.StringIO(json.dumps(request) + '\n' + json.dumps({'shutdown': True}) + '\n')
        def sendall(self, data): sent.append(json.loads(data))
    class Listener:
        def bind(self, address): pass
        def listen(self, size): pass
        def accept(self): return Connection(), None
        def close(self): pass
    class Engine:
        def __init__(self, **kwargs): observed['engine'] = kwargs
        def generate(self, prompts, params, lora_request):
            observed.update(prompts=prompts, sampling=vars(params))
            returned_prefix = [1, 1, 2, 5, 4, 3] if engine_changes_prefix else [1, 2, 5, 4, 3]
            return [SimpleNamespace(prompt_token_ids=returned_prefix, outputs=[
                SimpleNamespace(token_ids=[5, 4], finish_reason='stop', stop_reason=4,
                    logprobs=[{5: SimpleNamespace(logprob=-.1)}, {4: SimpleNamespace(logprob=-.2)}])])]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: tok)
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(vocab_size=9))
    monkeypatch.setattr(server.socket, 'socket', lambda *a: Listener())
    monkeypatch.setattr(sys, 'argv', ['server', '--model', 'base', '--tokenizer', 'saved',
                                     '--socket', str(tmp_path / 'server.sock')])
    if engine_changes_prefix:
        with pytest.raises(RuntimeError, match='prompt token'):
            server.main()
        return
    server.main()
    assert observed['prompts'] == [{'prompt_token_ids': [1, 2, 5, 4, 3]}]
    assert observed['engine']['tokenizer'] == 'saved'
    assert observed['sampling']['logit_bias'] == {8: -float('inf')}
    assert sent[0]['prompt_token_ids'] == [[1, 2, 5, 4, 3]]
    assert sent[0]['selected_logprobs'] == [[-.1, -.2]]


def test_preflight_compares_probabilities_on_exact_single_bos_prefix(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from scripts.preflight_training import WorkerChecks

    torch.manual_seed(9)
    actor = LlamaForCausalLM(LlamaConfig(vocab_size=9, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1)).eval()
    request = {'adapter_id': 1, 'adapter': 'initial', 'group_size': 1,
        'prompts': ['[BOS][USER]Hello[END][ASSISTANT]'],
        'prompt_token_ids': [[1, 2, 5, 4, 3]], 'min_tokens': 0, 'temperature': 1.}
    with torch.no_grad():
        logits = actor(torch.tensor([[1, 2, 5, 4, 3, 5, 4]])).logits[0, 4:6]
        expected = logits.log_softmax(-1).gather(1, torch.tensor([[5], [4]])).flatten().tolist()
    result = {'rows': [[5, 4]], 'selected_logprobs': [expected],
              'prompt_token_ids': [[1, 2, 5, 4, 3]], 'logprobs_mode': 'processed_logprobs'}
    checks = WorkerChecks(SimpleNamespace(worker_arm='grpo', output_dir=tmp_path))
    checks.trainer = SimpleNamespace(actor=actor, actor_tokenizer=tokenizer(),
        actor_device=torch.device('cpu'), output_mask=torch.ones(9, dtype=torch.bool), stop_token_ids=(4,))
    checks.check_probabilities(request, result)
    assert checks.checked_adapter_ids == {1}
    assert checks.report['probability_checks'][0]['max_abs_error'] < 1e-6


def test_preflight_selects_llama_family_without_mutating_qwen_launcher(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import preflight_training as preflight
    from scripts import corrected_rl_launcher as qwen
    original = qwen.common_config
    monkeypatch.setitem(sys.modules, 'scripts.llama_rl_launcher', SimpleNamespace(
        common_config=lambda root: {'model': 'llama-model', 'rm': 'llama-rm', 'init_adapter': 'llama-sft'},
        validation_identity=lambda root: {}))
    preflight.main(['--output-dir', str(tmp_path / 'out'), '--project-root', str(tmp_path),
                    '--runtime-image', 'image', '--experiment-family', 'llama', '--dry-run'])
    result = json.loads(capsys.readouterr().out)
    for argv in result['arms'].values():
        assert argv[argv.index('--model') + 1] == 'llama-model'
    assert qwen.common_config is original


def test_reward_batch_rejects_decode_expansion_before_scoring():
    from transformers import PreTrainedTokenizerFast
    source = Path('/data/VPO-RM/models/Skywork-Reward-Llama-3.1-8B-v0.2')
    if not (source / 'tokenizer.json').exists():
        pytest.skip('Local Llama tokenizer unavailable')
    tok = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    trainer = object.__new__(VPOTrainer)
    trainer.actor_tokenizer = trainer.reward_tokenizer = tok
    trainer.cfg = SimpleNamespace(method='grpo', microbatch_responses=1,
                                  max_prompt_tokens=2048, max_response_tokens=2048)
    trainer.reward_device = torch.device('cpu')
    trainer.reward = SimpleNamespace(get_input_embeddings=lambda: pytest.fail('Overflow reached RM scoring'))
    prompt = 'word ' * 1800
    prefix = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    assert len(prefix) == 1835  # The actor input is within its 2048-token budget.
    response = torch.full((1, 2048), 2275, dtype=torch.long)
    assert tok.decode([2275]) == 'о�'  # Retokenization expands this token to two.
    with pytest.raises(ValueError, match='[Cc]anonical RM.*4096|4096.*[Cc]anonical RM'):
        trainer._reward_batch(None, None, None, response, torch.ones_like(response, dtype=torch.bool), [prompt])


@pytest.mark.parametrize('native_limit,prompt_limit,response_limit', [(7, 2048, 2048), (100, 3, 4)])
def test_reward_batch_accepts_native_limit_and_rejects_one_token_over(
        native_limit, prompt_limit, response_limit):
    from torch import nn
    from vpo_rm.reward import LastTokenReward
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding = nn.Embedding(8, 2)
        def get_input_embeddings(self): return self.embedding
        def forward(self, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds)
    tok = tokenizer()
    tok.chat_template = "{{ bos_token }}{% for m in messages %}[USER]{{ m['content'] }}[END]{% endfor %}"
    tok.model_max_length = native_limit
    trainer = object.__new__(VPOTrainer)
    trainer.actor_tokenizer = trainer.reward_tokenizer = tok
    trainer.cfg = SimpleNamespace(method='grpo', microbatch_responses=1,
                                  max_prompt_tokens=2048, max_response_tokens=2048)
    trainer.reward_device = torch.device('cpu')
    trainer.cfg.max_prompt_tokens, trainer.cfg.max_response_tokens = prompt_limit, response_limit
    trainer.reward = LastTokenReward(Backbone(), nn.Linear(2, 1)).eval()
    response = torch.tensor([[5, 4]])
    scores, *_ = trainer._reward_batch(None, None, None, response,
        torch.ones_like(response, dtype=torch.bool), ['Hello'])
    assert scores.shape == (1,) and torch.isfinite(scores).all()
    assert trainer._reward_alignment_stats['rm_max_input_tokens'] == 7
    assert trainer._reward_alignment_stats['rm_input_token_budget'] == 7
    response = torch.tensor([[5, 6, 4]])
    with pytest.raises(ValueError, match='[Cc]anonical RM'):
        trainer._reward_batch(None, None, None, response,
            torch.ones_like(response, dtype=torch.bool), ['Hello'])
