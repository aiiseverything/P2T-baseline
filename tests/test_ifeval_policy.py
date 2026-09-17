"""IFEval's actual CLI must preserve actor precision and stop provenance."""
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pytest


def checkpoint(root, head=None, *, step=False):
    train = root / 'train'
    adapter = train / ('vllm-adapters/step-250' if step else 'checkpoint-250')
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
    if head is not None:
        metadata = train / 'profile_manifest.json' if step else adapter / 'run_manifest.json'
        metadata.write_text(json.dumps({'sampling': {'policy_head_dtype': head}}))
    return adapter


@pytest.fixture
def cli_stack(tmp_path, monkeypatch):
    from scripts import eval_ifeval
    constructors, generations = [], []

    class Engine:
        def __init__(self, **kwargs):
            constructors.append(kwargs)

        def generate(self, prompts, params, **kwargs):
            generations.append({'params': params, **kwargs})
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text='answer' if sample == 0 else ('capped' if row == 0 else ''),
                token_ids=[1, 2] if sample == 0 else ([1, 1] if row == 0 else []),
                finish_reason='stop' if sample == 0 else ('length' if row == 0 else 'stop'),
                stop_reason=2 if sample == 0 else None,
            ) for sample in range(params.n)]) for row, _ in enumerate(prompts)]

    class Tokenizer:
        eos_token_id = 2
        all_special_ids = [2]

        def get_vocab(self):
            return {'a': 1, '<eos>': 2}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)

    # Only the external generation stack and unrelated official checker are
    # substituted; main(), policy resolution, cache identity, and writes are real.
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(vocab_size=3))
    from vpo_rm.trainer import VPOTrainer
    monkeypatch.setattr(VPOTrainer, '_render_chat_prompt', lambda tok, prompt: prompt)
    monkeypatch.setattr(eval_ifeval, 'validate_adapter_base', lambda *a: None)

    def score(inp, responses):
        follows = bool(responses[inp.prompt])
        return SimpleNamespace(follow_all_instructions=follows, follow_instruction_list=[follows])

    monkeypatch.setitem(sys.modules, 'instruction_following_eval', SimpleNamespace(
        instructions_registry=SimpleNamespace(INSTRUCTION_DICT={'test:exists': object()}),
        evaluation_lib=SimpleNamespace(test_instruction_following_strict=score,
                                       test_instruction_following_loose=score)))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'data.jsonl'
    dataset.write_text(''.join(json.dumps({'key': i, 'prompt': f'question {i}',
        'instruction_id_list': ['test:exists'], 'kwargs': [{}]}) + '\n' for i in range(2)))
    output = tmp_path / 'out'
    argv = ['eval', '--model', str(model), '--dataset', str(dataset), '--output', str(output)]

    def run(*extra):
        monkeypatch.setattr(sys, 'argv', [*argv, *extra])
        eval_ifeval.main()

    return SimpleNamespace(run=run, output=output, constructors=constructors, generations=generations)


@pytest.mark.parametrize('step', [False, True])
def test_main_auto_uses_checkpoint_head_and_caches_metadata(tmp_path, cli_stack, step):
    adapter = checkpoint(tmp_path, 'float32', step=step)
    metadata = adapter.parent.parent / 'profile_manifest.json' if step else adapter / 'run_manifest.json'
    cli_stack.run('--adapters', f'rl={adapter}')
    assert cli_stack.constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    config = json.loads((cli_stack.output / 'rl/manifest_t1.0_n1.json').read_text())['config']
    assert config['policy']['policy_head_dtype'] == 'float32'
    assert config['policy']['metadata'] == [{'path': str(metadata),
        'sha256': hashlib.sha256(metadata.read_bytes()).hexdigest()}]
    cli_stack.run('--adapters', f'rl={adapter}')
    assert len(cli_stack.constructors) == len(cli_stack.generations) == 1
    metadata.write_text(json.dumps({'sampling': {'policy_head_dtype': 'float32'}, 'changed': True}))
    with pytest.raises(ValueError, match='different'):
        cli_stack.run('--adapters', f'rl={adapter}')
    assert len(cli_stack.generations) == 1


def test_main_auto_rejects_mixed_heads_before_engine(tmp_path, cli_stack):
    adapter = checkpoint(tmp_path, 'float32')
    with pytest.raises(ValueError, match='separate'):
        cli_stack.run('--adapters', 'base=none', f'rl={adapter}')
    assert cli_stack.constructors == []


def test_main_explicit_fp32_compares_base_legacy_and_rl_in_one_engine(tmp_path, cli_stack):
    legacy = checkpoint(tmp_path / 'legacy')
    adapter = checkpoint(tmp_path / 'rl', 'float32')
    cli_stack.run('--policy-head-dtype', 'float32', '--adapters',
                  'base=none', f'sft={legacy}', f'rl={adapter}')
    assert len(cli_stack.constructors) == 1
    assert cli_stack.constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    assert len(cli_stack.generations) == 3
    assert cli_stack.generations[0]['lora_request'] is None
    for tag in ['base', 'sft', 'rl']:
        config = json.loads((cli_stack.output / tag / 'manifest_t1.0_n1.json').read_text())['config']
        assert config['policy']['policy_head_dtype'] == 'float32'
        assert bool(config['policy']['metadata']) == (tag == 'rl')


def test_main_explicit_native_cannot_override_declared_fp32(tmp_path, cli_stack):
    adapter = checkpoint(tmp_path, 'float32')
    with pytest.raises(ValueError, match='conflict'):
        cli_stack.run('--policy-head-dtype', 'native', '--adapters', f'rl={adapter}')
    assert cli_stack.constructors == []


def test_main_unknown_manifest_head_fails_before_engine(tmp_path, cli_stack):
    adapter = checkpoint(tmp_path, 'float16')
    with pytest.raises(ValueError, match='Unknown policy_head_dtype'):
        cli_stack.run('--adapters', f'rl={adapter}')
    assert cli_stack.constructors == []


def test_main_native_cache_cannot_be_reused_for_fp32_baseline(cli_stack):
    cli_stack.run()
    assert 'hf_overrides' not in cli_stack.constructors[0]
    with pytest.raises(ValueError, match='different'):
        cli_stack.run('--policy-head-dtype', 'float32')
    assert len(cli_stack.generations) == 1


def test_main_keeps_sampling_defaults_and_records_single_sample_stop(cli_stack):
    cli_stack.run()
    params = cli_stack.generations[0]['params']
    assert (params.temperature, params.top_p, params.top_k, params.n,
            params.max_tokens, params.seed) == (1.0, 1.0, -1, 1, 1280, 42)
    row = json.loads((cli_stack.output / 'model/generations_t1.0_n1.jsonl').read_text().splitlines()[0])
    assert row['responses'] == ['answer']
    assert row['finish_reason'] == ['stop']
    assert row['stop_reason'] == [2]
    assert row['last_token_id'] == [2]


def test_main_stop_arrays_follow_each_prompt_and_sample_including_empty(cli_stack):
    cli_stack.run('--recipes', '1.0:2:1.0:-1')
    rows = [json.loads(line) for line in
            (cli_stack.output / 'model/generations_t1.0_n2.jsonl').read_text().splitlines()]
    assert rows[0]['responses'] == ['answer', 'capped']
    assert rows[0]['response_tokens'] == [2, 2]
    assert rows[0]['finish_reason'] == ['stop', 'length']
    assert rows[0]['stop_reason'] == [2, None]
    assert rows[0]['last_token_id'] == [2, 1]
    assert rows[1]['responses'] == ['answer', '']
    assert rows[1]['response_tokens'] == [2, 0]
    assert rows[1]['finish_reason'] == ['stop', 'stop']
    assert rows[1]['stop_reason'] == [2, None]
    assert rows[1]['last_token_id'] == [2, None]


def affected_symbol_inputs():
    from scripts import eval_ifeval
    data = [json.loads(line) for line in
            (eval_ifeval.ROOT / 'third_party/ifeval/input_data.jsonl').read_text().splitlines()]
    return eval_ifeval.make_inputs([row for row in data if row['key'] in (1122, 1129)])


def test_real_symbol_checker_reuses_parameters_for_strict_loose_and_models(monkeypatch):
    from scripts import eval_ifeval
    from instruction_following_eval import evaluation_lib, instructions
    original = instructions.LetterFrequencyChecker.build_description
    constructed = []

    def record(self, **kwargs):
        result = original(self, **kwargs)
        constructed.append(self.get_instruction_args())
        return result

    monkeypatch.setattr(instructions.LetterFrequencyChecker, 'build_description', record)
    inputs = affected_symbol_inputs()
    assert [inp.key for inp in inputs] == [1122, 1129]
    saved = random.getstate()
    try:
        random.seed(125)
        eval_ifeval.score_one_sample(inputs, ['', ''], evaluation_lib)
        first = constructed[:]
        assert first[:2] == first[2:]
        # Official symbol-to-letter behavior is preserved, not silently repaired.
        assert [args['letter'] for args in first] == ['o', 'h', 'o', 'h']
        constructed.clear()
        random.seed(999)
        eval_ifeval.score_one_sample(inputs, ['an english response', 'a second response'], evaluation_lib)
        assert constructed == first
    finally:
        random.setstate(saved)


def test_real_langdetect_uses_scoring_seed_and_restores_callers_state(monkeypatch):
    from scripts import eval_ifeval
    from instruction_following_eval import evaluation_lib
    from langdetect import DetectorFactory
    from langdetect.detector import Detector
    original = Detector.__init__
    seeds = []

    def record(self, factory):
        original(self, factory)
        seeds.append(self.seed)

    monkeypatch.setattr(Detector, '__init__', record)
    monkeypatch.setattr(DetectorFactory, 'seed', 319)
    saved = random.getstate()
    eval_ifeval.score_one_sample(affected_symbol_inputs()[:1],
        ['this is a complete english response with enough words for language detection.'],
        evaluation_lib, scoring_seed=17)
    assert len(seeds) >= 2 and set(seeds) == {17}
    assert random.getstate() == saved
    assert DetectorFactory.seed == 319


@pytest.mark.parametrize('failure_mode', ['strict', 'loose'])
def test_scoring_restores_random_and_detector_seed_when_checker_raises(monkeypatch, failure_mode):
    from scripts import eval_ifeval
    from langdetect import DetectorFactory
    monkeypatch.setattr(DetectorFactory, 'seed', 319)
    saved = random.getstate()

    def fail(*args):
        random.random()
        raise RuntimeError('checker failed')

    def succeed(*args):
        random.random()
        return object()

    lib = SimpleNamespace(test_instruction_following_strict=fail if failure_mode == 'strict' else succeed,
                          test_instruction_following_loose=fail)
    with pytest.raises(RuntimeError, match='checker failed'):
        eval_ifeval.score_one_sample(affected_symbol_inputs()[:1], ['response'], lib, scoring_seed=17)
    assert random.getstate() == saved
    assert DetectorFactory.seed == 319


def test_main_passes_scoring_seed_and_binds_scoring_protocol_to_cache(cli_stack, monkeypatch):
    from scripts import eval_ifeval
    original = eval_ifeval.score_one_sample
    seeds = []

    def record(*args, scoring_seed=42):
        seeds.append(scoring_seed)
        return original(*args, scoring_seed=scoring_seed)

    monkeypatch.setattr(eval_ifeval, 'score_one_sample', record)
    cli_stack.run('--seed', '17')
    assert seeds == [17]
    manifest = cli_stack.output / 'model/manifest_t1.0_n1.json'
    saved = json.loads(manifest.read_text())
    assert saved['config']['scoring'] == {
        'protocol': 'official_seeded_v1', 'seed': 17, 'langdetect_seed': 17}
    saved['config']['scoring']['seed'] = 18
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='different'):
        cli_stack.run('--seed', '17')
    assert len(cli_stack.generations) == 1 and seeds == [17]
