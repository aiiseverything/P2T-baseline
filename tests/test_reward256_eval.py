"""Direct policy identities and single-H200 offline reward evaluation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_checkpoints as evaluation
from test_eval_policy import mock_generation_stack


def test_discover_bare_base():
    assert evaluation.discover_adapters(Path('none')) == [(0, Path('none'))]


@pytest.mark.parametrize('name,step', [('sft-native-eos-clean2k5e2', 0), ('checkpoint-250', 250)])
def test_discover_direct_adapter_preserves_checkpoint_identity(tmp_path, name, step):
    path = tmp_path / name
    path.mkdir()
    (path / 'adapter_config.json').write_text('{}')
    (path / 'adapter_model.safetensors').write_bytes(b'weights')
    assert evaluation.discover_adapters(path) == [(step, path)]


def test_direct_adapter_requires_weights(tmp_path):
    (tmp_path / 'adapter_config.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='weights'):
        evaluation.discover_adapters(tmp_path)


def test_single_gpu_base_cli_uses_no_lora_and_records_raw_scores(tmp_path, monkeypatch):
    constructors, requests = mock_generation_stack(monkeypatch)
    monkeypatch.setattr(evaluation.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(evaluation, 'load_validation_prompts', lambda *a: ['prompt'])
    def score(generations, prompts, temps, args):
        assert args.rm_device == 'cuda:0'
        assert generations == {('base', 0, 1.0): [[1, 2]]}
        return {('base', 0, 1.0): [-3.25]}
    monkeypatch.setattr(evaluation, 'score_all', score)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'data'
    dataset.write_bytes(b'dataset')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['eval', '--run', 'base=none', '--model', str(model),
        '--rm', str(model), '--dataset-path', str(dataset), '--temps', '1', '--seed', '42',
        '--rm-device', 'cuda:0', '--policy-head-dtype', 'float32', '--output', str(out)])
    evaluation.main()
    assert requests == [{'lora_request': None}]
    assert constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    manifest = json.loads((out / 'manifest.json').read_text())['config']
    assert manifest['adapters'] == {'base/0': None}
    assert manifest['reward_input_protocol'] == 'canonical_chat_v1'
    assert json.loads((out / 'eval.jsonl').read_text())['score'] == -3.25


@pytest.mark.parametrize('scores', [{}, {('base', 0, 1.): []}, {('base', 0, 1.): [float('nan')]}])
def test_missing_or_nonfinite_reward_cannot_be_summarized(scores):
    with pytest.raises(ValueError, match='scores|finite|coverage'):
        evaluation.validate_scores({('base', 0, 1.): [[1, 2]]}, scores, 1)


def chat_tokenizer(*, dedicated_pad=True):
    """Real tokenizer with the same rendered-BOS/automatic-BOS hazard as Llama."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, 'hello': 4, '<unk>': 5}
    vocab['<|finetune_right_pad_id|>' if dedicated_pad else '<other>'] = 3
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single='<bos> $A', special_tokens=[('<bos>', 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<bos>',
        eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
    tokenizer.chat_template = "<bos>{{ messages[0]['content'] }} {{ messages[1]['content'] }}<eos>"
    return tokenizer


def explicit_generation_stack(monkeypatch, *, echo='correct'):
    import transformers
    tokenizer = chat_tokenizer()
    loaded, engines, inputs = [], [], []
    def load(source, **kwargs):
        loaded.append(str(source))
        return tokenizer
    class Engine:
        def __init__(self, **kwargs):
            engines.append(kwargs)
        def generate(self, prompts, params, **kwargs):
            inputs.append(prompts)
            results = []
            for prompt in prompts:
                ids = (prompt['prompt_token_ids'] if isinstance(prompt, dict)
                       else tokenizer(prompt)['input_ids'])
                row = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 2])])
                if echo != 'missing':
                    row.prompt_token_ids = ids if echo == 'correct' else [1, *ids]
                results.append(row)
            return results
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', load)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(vocab_size=6))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    return loaded, engines, inputs


def test_reward_generation_sends_single_bos_ids_and_uses_saved_tokenizer(monkeypatch, tmp_path):
    loaded, engines, inputs = explicit_generation_stack(monkeypatch)
    source = str(tmp_path / 'saved-tokenizer')
    args = SimpleNamespace(model='actor', tokenizer=source, max_num_seqs=1,
                           seed=42, max_tokens=2048, policy_head_dtype='float32')
    result = evaluation.generate_all([('base', Path('none'))], ['<bos>hello'], [1.0], args)
    assert inputs == [[{'prompt_token_ids': [1, 4]}]]
    assert loaded and set(loaded) == {source}
    assert engines[0]['tokenizer'] == source
    assert result == {('base', 0, 1.0): [[4, 2]]}


@pytest.mark.parametrize('echo', ['missing', 'extra-bos'])
def test_reward_generation_rejects_unverified_prompt_echo(monkeypatch, echo):
    explicit_generation_stack(monkeypatch, echo=echo)
    args = SimpleNamespace(model='actor', max_num_seqs=1, seed=42, max_tokens=2048)
    with pytest.raises(ValueError, match='prompt token'):
        evaluation.generate_all([('base', Path('none'))], ['<bos>hello'], [1.0], args)


def test_reward_cache_binds_saved_tokenizer_files_and_prompt_ids(tmp_path, monkeypatch):
    from scripts.eval_artifacts import digest
    from vpo_rm.trainer import VPOTrainer
    loaded, engines, _ = explicit_generation_stack(monkeypatch)
    monkeypatch.setattr(evaluation.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(evaluation, 'load_validation_prompts', lambda *a: ['hello'])
    monkeypatch.setattr(VPOTrainer, '_render_chat_prompt', lambda tok, p: '<bos>' + p)
    monkeypatch.setattr(evaluation, 'score_all', lambda gs, *a: {key: [-3.25] for key in gs})
    model, source = tmp_path / 'model', tmp_path / 'saved-tokenizer'
    model.mkdir(); source.mkdir()
    (model / 'config.json').write_text('{}')
    tokenizer_file = source / 'tokenizer.json'
    tokenizer_file.write_text('{"version": 1}')
    (source / 'adapter_model.safetensors').write_bytes(b'large-weights')
    dataset = tmp_path / 'data'; dataset.write_bytes(b'dataset')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['eval', '--run', 'base=none', '--model', str(model),
        '--tokenizer', str(source), '--rm', str(model), '--dataset-path', str(dataset),
        '--temps', '1', '--seed', '42', '--rm-device', 'cuda:0', '--output', str(out)])
    evaluation.main()
    config = json.loads((out / 'manifest.json').read_text())['config']
    assert config['tokenizer']['source'] == str(source)
    assert config['prompt_token_ids_sha256'] == digest([[1, 4]])
    records = {row['name']: row for row in config['tokenizer']['fingerprint']['files']}
    assert 'sha256' in records['tokenizer.json']
    assert 'sha256' not in records['adapter_model.safetensors']
    evaluation.main()
    assert len(engines) == 1
    tokenizer_file.write_text('{"version": 2}')
    with pytest.raises(ValueError, match='different'):
        evaluation.main()
    assert len(engines) == 1


def test_reward_scoring_uses_rm_owned_pad_and_native_bf16_raw_scalar(monkeypatch):
    import torch
    import transformers
    actor, reward_tokenizer = chat_tokenizer(dedicated_pad=False), chat_tokenizer()
    loaded, calls, seen = [], [], []
    def tokenizer_load(source, **kwargs):
        loaded.append(source)
        return reward_tokenizer if source == 'reward' else actor
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 1, dtype=torch.bfloat16)
            self.embedding.weight.data[:, 0] = torch.arange(6)
        def get_input_embeddings(self):
            return self.embedding
        def forward(self, inputs_embeds, attention_mask, position_ids, **kwargs):
            seen.append((inputs_embeds.detach().clone(), attention_mask.clone(), position_ids.clone()))
            assert inputs_embeds.dtype == torch.bfloat16
            assert kwargs['use_cache'] is False
            return SimpleNamespace(last_hidden_state=inputs_embeds)
    class Reward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Backbone()
            self.score = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.score.weight.data.fill_(-1.625)
    def model_load(source, **kwargs):
        calls.append(kwargs)
        return Reward()
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', tokenizer_load)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', model_load)
    args = SimpleNamespace(model='actor', tokenizer='saved-tokenizer', rm='reward',
                           rm_device='cpu', rm_microbatch=2, policy_head_dtype='float32')
    scores = evaluation.score_all({('base', 0, 1.): [[4, 2], [4, 2]]},
                                  ['hello', 'hello hello'], [1.], args)
    assert reward_tokenizer.pad_token_id == 3
    assert loaded == ['reward', 'saved-tokenizer']
    assert scores == {('base', 0, 1.): [-3.25, -3.25]}
    assert calls == [{'torch_dtype': torch.bfloat16, 'trust_remote_code': True}]
    embeds, mask, positions = seen[0]
    assert embeds[:, :, 0].tolist() == [[1, 4, 4, 2, 3], [1, 4, 4, 4, 2]]
    assert mask.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert positions.tolist() == [[0, 1, 2, 3, 3], [0, 1, 2, 3, 4]]
