"""Arena generation preserves protocol, six-model identity, and offline caches."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_official_style_metadata_counts_code_fences_like_upstream():
    from scripts.eval_arena_hard import style_metadata
    simple = style_metadata('hello')
    assert simple['token_len'] == 1
    assert simple['header_count'] == {f'h{i}': 0 for i in range(1, 7)}
    assert simple['list_count'] == {'ordered': 0, 'unordered': 0}
    assert simple['bold_count'] == {'**': 0, '__': 0}
    value = style_metadata('# Visible\n- item\n**bold**\n```\n# Hidden\n- code\n**code**\n```')
    assert value['header_count']['h1'] == 1
    assert value['list_count']['unordered'] == 1
    assert value['bold_count']['**'] == 1


@pytest.mark.parametrize('change', ['duplicate', 'creative', 'missing_prompt', 'count'])
def test_question_validation_rejects_wrong_subset(change):
    from scripts.eval_arena_hard import validate_questions
    rows = [{'uid': 'a', 'category': 'hard_prompt', 'prompt': 'question'},
            {'uid': 'b', 'category': 'hard_prompt', 'prompt': 'question two'}]
    if change == 'duplicate': rows[1]['uid'] = 'a'
    elif change == 'creative': rows[1]['category'] = 'creative_writing'
    elif change == 'missing_prompt': rows[1]['prompt'] = ''
    else: rows.pop()
    with pytest.raises(ValueError):
        validate_questions(rows, expected_count=2)


@pytest.fixture
def generation_stack(tmp_path, monkeypatch):
    from scripts import eval_arena_hard as arena
    calls, engines = [], []

    class Tokenizer:
        eos_token_id = 2
        eos_token = '<eos>'
        pad_token_id = None
        all_special_ids = [2]
        def get_vocab(self): return {'a': 1, '<eos>': 2}
        def encode(self, text, add_special_tokens=False): return [1] * 3
        def convert_tokens_to_ids(self, token): return self.get_vocab().get(token)

    class Engine:
        def __init__(self, **kwargs): engines.append(kwargs)
        def generate(self, prompts, params, **kwargs):
            calls.append({'params': params, **kwargs})
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'], outputs=[SimpleNamespace(
                text='hello', token_ids=[1, 2], finish_reason='stop', stop_reason=2)]) for p in prompts]

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace(vocab_size=3))
    from vpo_rm.trainer import VPOTrainer
    monkeypatch.setattr(VPOTrainer, '_render_chat_prompt', lambda tok, text: text)
    monkeypatch.setattr(arena, 'validate_adapter_base', lambda *a: None)
    model = tmp_path / 'base'; model.mkdir(); (model / 'config.json').write_text('{}')
    adapter = tmp_path / 'train/vllm-adapters/step-250'; adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
    profile = adapter.parent.parent / 'profile_manifest.json'
    profile.write_text(json.dumps({'sampling': {'policy_head_dtype': 'float32'}}))
    dataset = tmp_path / 'questions.jsonl'
    questions = [{'uid': str(i), 'category': 'hard_prompt', 'prompt': f'question {i}'} for i in range(500)]
    dataset.write_text(''.join(json.dumps(q) + '\n' for q in questions))
    args = SimpleNamespace(model=str(model), dataset=str(dataset), output=str(tmp_path / 'out'),
        adapters=['base=none', f'rl={adapter}'], max_tokens=8, max_model_len=16,
        seed=42, policy_head_dtype='float32')
    return SimpleNamespace(arena=arena, args=args, calls=calls, engines=engines,
                           questions=questions, profile=profile, adapter=adapter)


def test_shared_engine_official_rows_and_verified_cache(generation_stack):
    stack = generation_stack
    result = stack.arena.generate_all(stack.args)
    assert set(result) == {'base', 'rl'}
    assert len(stack.engines) == 1 and len(stack.calls) == 2
    assert stack.engines[0]['hf_overrides'] == {'head_dtype': 'float32'}
    assert stack.engines[0]['max_model_len'] == 16
    assert stack.engines[0]['generation_config'] == 'vllm'
    assert stack.calls[0]['lora_request'] is None
    assert stack.calls[1]['lora_request'][2] == str(stack.adapter)
    params = stack.calls[0]['params']
    assert (params.temperature, params.top_p, params.top_k, params.n, params.seed, params.max_tokens) == (1., 1., -1, 1, 42, 8)
    out = Path(stack.args.output)
    rows = [json.loads(s) for s in (out / 'model_answer/rl.jsonl').read_text().splitlines()]
    assert len(rows) == 500
    assert rows[0]['uid'] == '0' and rows[0]['model'] == 'rl'
    assert rows[0]['messages'] == [{'role': 'user', 'content': 'question 0'},
                                  {'role': 'assistant', 'content': {'answer': 'hello'}}]
    assert rows[0]['metadata']['token_len'] == 1
    assert rows[0]['generation'] == {'response_tokens': 2, 'prompt_tokens': 3,
        'finish_reason': 'stop', 'stop_reason': 2, 'last_token_id': 2, 'stop_token_ids': [2],
        'ended_with_eos': True}
    config = json.loads((out / 'manifests/rl.json').read_text())['config']
    assert config['policy']['policy_head_dtype'] == 'float32'
    assert config['policy']['metadata'][0]['path'] == str(stack.profile)
    assert config['engine']['max_model_len'] == 16
    assert 'third_party/arena_hard/utils/add_markdown_info.py' in config['sources']
    from scripts.eval_artifacts import fingerprint
    assert stack.engines[0]['tokenizer'] == stack.args.model
    assert config['tokenizer'] == {'source': stack.args.model,
                                   'fingerprint': fingerprint(stack.args.model, full_weights=False)}
    assert config['tokenizer']['fingerprint'] == config['model']
    stack.arena.generate_all(stack.args)
    assert len(stack.engines) == 1 and len(stack.calls) == 2


@pytest.mark.parametrize('mutation', ['profile', 'tokens', 'context', 'output'])
def test_cache_rejects_mutated_generation_inputs_and_outputs(generation_stack, mutation):
    s = generation_stack
    s.arena.generate_all(s.args)
    if mutation == 'profile':
        s.profile.write_text(json.dumps({'policy_head_dtype': 'float32', 'changed': True}))
    elif mutation == 'tokens': s.args.max_tokens = 7
    elif mutation == 'context': s.args.max_model_len = 17
    else:
        (Path(s.args.output) / 'model_answer/base.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='different|changed'):
        s.arena.generate_all(s.args)
    assert len(s.calls) == 2


def test_uniform_response_budget_must_fit_every_prompt_before_engine(generation_stack):
    s = generation_stack
    s.args.max_model_len = 10  # three prompt tokens plus eight output tokens
    with pytest.raises(ValueError, match='context|budget'):
        s.arena.generate_all(s.args)
    assert s.engines == []


def test_implicit_mixed_head_run_is_rejected_before_engine(generation_stack):
    s = generation_stack
    s.args.policy_head_dtype = 'auto'
    with pytest.raises(ValueError, match='separate'):
        s.arena.generate_all(s.args)
    assert s.engines == []


def test_unicode_line_separators_survive_question_loading_and_cached_answers(generation_stack):
    s = generation_stack
    s.questions[0]['prompt'] = 'line one\u2028line two\u0085line three'
    Path(s.args.dataset).write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in s.questions))
    s.arena.generate_all(s.args)
    with (Path(s.args.output) / 'model_answer/base.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[0]['messages'][0]['content'] == s.questions[0]['prompt']
    s.arena.generate_all(s.args)
    assert len(s.calls) == 2


@pytest.mark.parametrize('mutation', ['uid', 'prompt', 'metadata', 'model'])
def test_answer_validation_rejects_wrong_binding(generation_stack, mutation):
    s = generation_stack
    s.arena.generate_all(s.args)
    rows = [json.loads(line) for line in
            (Path(s.args.output) / 'model_answer/rl.jsonl').read_text().splitlines()]
    if mutation == 'uid': rows[1]['uid'] = rows[0]['uid']
    elif mutation == 'prompt': rows[0]['messages'][0]['content'] = 'other prompt'
    elif mutation == 'metadata': rows[0]['metadata']['token_len'] = 999
    else: rows[0]['model'] = 'other model'
    with pytest.raises(ValueError):
        s.arena.validate_answers(rows, s.questions, 'rl')


def test_tokenizer_override_renders_prompts_and_binds_its_source(generation_stack, tmp_path, monkeypatch):
    """A base checkpoint without a chat template renders with its saved SFT tokenizer."""
    import transformers
    from scripts.eval_artifacts import fingerprint
    s = generation_stack
    saved = tmp_path / 'sft-adapter'; saved.mkdir()
    (saved / 'tokenizer_config.json').write_text('{"chat_template": "pinned"}')
    (saved / 'adapter_model.safetensors').write_bytes(b'sft weights')
    original = transformers.AutoTokenizer.from_pretrained
    sources = []
    def from_pretrained(source, *args, **kwargs):
        sources.append(str(source))
        return original(source, *args, **kwargs)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', from_pretrained)
    s.args.tokenizer = str(saved)
    s.arena.generate_all(s.args)
    assert sources == [str(saved)]
    assert s.engines[0]['model'] == s.args.model and s.engines[0]['tokenizer'] == str(saved)
    config = json.loads((Path(s.args.output) / 'manifests/base.json').read_text())['config']
    expected = fingerprint(saved, full_weights=False)
    assert config['tokenizer'] == {'source': str(saved), 'fingerprint': expected}
    weights = next(row for row in expected['files'] if row['name'] == 'adapter_model.safetensors')
    assert 'sha256' not in weights and weights['size'] == len(b'sft weights')
    assert config['tokenizer']['source'] != config['model']['path']
    s.arena.generate_all(s.args)
    assert len(s.calls) == 2
    (saved / 'tokenizer_config.json').write_text('{"chat_template": "changed"}')
    with pytest.raises(ValueError, match='different|changed'):
        s.arena.generate_all(s.args)
    assert len(s.calls) == 2


def test_offline_generation_config_matches_gpu_cache_identity(generation_stack):
    """Preparation scripts rebuild the exact cache identity with supplied runtime versions."""
    from types import SimpleNamespace as NS
    from scripts.eval_artifacts import fingerprint
    s = generation_stack
    s.arena.generate_all(s.args)
    manifest = json.loads((Path(s.args.output) / 'manifests/rl.json').read_text())['config']
    engine = s.arena.engine_arguments(s.args, 'float32', s.args.model)
    assert engine == s.engines[0]
    offline = s.arena.generation_config(
        NS(dataset=s.args.dataset, seed=42, max_tokens=8), 'rl', fingerprint(s.adapter),
        manifest['policy'], engine, model_fingerprint=fingerprint(s.args.model, full_weights=False),
        tokenizer_provenance=manifest['tokenizer'], stop_ids=[2], support_summary=manifest['output_support'],
        prompt_ids=[[1] * 3] * 500, runtime=manifest['runtime_versions'])
    assert offline == manifest
