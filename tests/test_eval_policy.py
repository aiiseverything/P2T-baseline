"""Head protocol must survive loading a standalone evaluation adapter."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def checkpoint(tmp_path, head=None, *, step=False):
    train = tmp_path / 'train'
    adapter = train / ('vllm-adapters/step-250' if step else 'checkpoint-250')
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
    if head is not None:
        metadata = train / 'profile_manifest.json' if step else adapter / 'run_manifest.json'
        metadata.write_text(json.dumps({'sampling': {'policy_head_dtype': head}}))
    return adapter


@pytest.mark.parametrize('step', [False, True])
def test_auto_reads_checkpoint_or_step_parent_metadata(tmp_path, step):
    from scripts.eval_policy import resolve_policy_head, policy_engine_kwargs
    adapter = checkpoint(tmp_path, 'float32', step=step)
    policy = resolve_policy_head(adapter)
    assert policy['policy_head_dtype'] == 'float32'
    assert len(policy['metadata']) == 1
    assert len(policy['metadata'][0]['sha256']) == 64
    assert policy_engine_kwargs(policy['policy_head_dtype']) == {'hf_overrides': {'head_dtype': 'float32'}}


def test_legacy_defaults_native_but_allows_explicit_fp32_baseline(tmp_path):
    from scripts.eval_policy import resolve_policy_head, policy_engine_kwargs
    adapter = checkpoint(tmp_path)
    assert resolve_policy_head(adapter)['policy_head_dtype'] == 'native'
    assert resolve_policy_head('none')['policy_head_dtype'] == 'native'
    assert resolve_policy_head(adapter, 'float32')['policy_head_dtype'] == 'float32'
    assert policy_engine_kwargs('native') == {}


@pytest.mark.parametrize('head', ['float16', None, 32])
def test_unknown_manifest_head_fails_closed(tmp_path, head):
    from scripts.eval_policy import resolve_policy_head
    adapter = checkpoint(tmp_path)
    (adapter / 'run_manifest.json').write_text(json.dumps({'resolved_config': {'policy_head_dtype': head}}))
    with pytest.raises(ValueError, match='policy_head_dtype'):
        resolve_policy_head(adapter)


def test_conflicting_declarations_and_explicit_override_are_rejected(tmp_path):
    from scripts.eval_policy import resolve_policy_head
    adapter = checkpoint(tmp_path, 'float32')
    with pytest.raises(ValueError, match='conflict'):
        resolve_policy_head(adapter, 'native')
    (adapter.parent / 'profile_manifest.json').write_text(json.dumps({'config': {'policy_head_dtype': 'native'}}))
    with pytest.raises(ValueError, match='conflict'):
        resolve_policy_head(adapter)


@pytest.mark.parametrize('section', ['resolved_config', 'config', 'sampling', 'vllm_engine'])
def test_inconsistent_fields_in_one_manifest_are_rejected(tmp_path, section):
    from scripts.eval_policy import resolve_policy_head
    adapter = checkpoint(tmp_path)
    (adapter / 'run_manifest.json').write_text(json.dumps({
        'policy_head_dtype': 'float32', section: {'policy_head_dtype': 'native'}}))
    with pytest.raises(ValueError, match='conflict'):
        resolve_policy_head(adapter)


def test_mixed_engine_modes_rejected_before_generation(tmp_path):
    from scripts.eval_policy import resolve_shared_policy
    adapter = checkpoint(tmp_path, 'float32')
    with pytest.raises(ValueError, match='separate'):
        resolve_shared_policy(['none', adapter], 'auto')
    head, policies = resolve_shared_policy(['none', adapter], 'float32')
    assert head == 'float32' and len(policies) == 2


def mock_generation_stack(monkeypatch):
    constructors, generations = [], []
    class Engine:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
        def generate(self, prompts, params, **kwargs):
            generations.append(kwargs)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text='answer', token_ids=[1, 2], finish_reason='stop', stop_reason=2)]) for _ in prompts]
    class Tokenizer:
        eos_token_id = 2
        all_special_ids = [2]
        def get_vocab(self):
            return {'a': 1, '<eos>': 2}
        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(vocab_size=3))
    from vpo_rm.trainer import VPOTrainer
    monkeypatch.setattr(VPOTrainer, '_render_chat_prompt', lambda tok, prompt: prompt)
    return constructors, generations


def test_alpaca_constructor_cache_and_manifest_mutation(tmp_path, monkeypatch):
    from scripts import eval_alpaca
    constructors, generations = mock_generation_stack(monkeypatch)
    monkeypatch.setattr(eval_alpaca, 'validate_adapter_base', lambda *a: None)
    adapter = checkpoint(tmp_path, 'float32', step=True)
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'data.jsonl'; dataset.write_text('{"instruction":"question"}\n')
    output = tmp_path / 'out'
    argv = ['eval', '--model', str(model), '--adapters', f'test={adapter}',
            '--dataset', str(dataset), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    eval_alpaca.main()
    assert constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    saved = json.loads((output / 'test/manifest_t1.0_n1.json').read_text())['config']
    assert saved['policy']['policy_head_dtype'] == 'float32'
    assert saved['policy']['metadata'][0]['path'] == str(adapter.parent.parent / 'profile_manifest.json')
    eval_alpaca.main()
    assert len(constructors) == len(generations) == 1
    metadata = adapter.parent.parent / 'profile_manifest.json'
    metadata.write_text(json.dumps({'sampling': {'policy_head_dtype': 'float32'}, 'changed': True}))
    with pytest.raises(ValueError, match='different'):
        eval_alpaca.main()
    assert len(generations) == 1


def test_alpaca_explicit_baseline_mode_invalidates_cache(tmp_path, monkeypatch):
    from scripts import eval_alpaca
    constructors, generations = mock_generation_stack(monkeypatch)
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'data.jsonl'; dataset.write_text('{"instruction":"question"}\n')
    argv = ['eval', '--model', str(model), '--dataset', str(dataset), '--output', str(tmp_path / 'out')]
    monkeypatch.setattr(sys, 'argv', argv)
    eval_alpaca.main()
    assert 'hf_overrides' not in constructors[0]
    monkeypatch.setattr(sys, 'argv', [*argv, '--policy-head-dtype', 'float32'])
    with pytest.raises(ValueError, match='different'):
        eval_alpaca.main()
    assert len(generations) == 1


def test_checkpoint_generation_constructor_reads_parent_profile(tmp_path, monkeypatch):
    from scripts import eval_checkpoints
    constructors, _ = mock_generation_stack(monkeypatch)
    monkeypatch.setattr(eval_checkpoints, 'validate_adapter_base', lambda *a: None)
    adapter = checkpoint(tmp_path, 'float32', step=True)
    args = SimpleNamespace(model='model', max_num_seqs=1, seed=42, max_tokens=8, policy_head_dtype='auto')
    result = eval_checkpoints.generate_all([('test', adapter.parent.parent)], ['prompt'], [1.0], args)
    assert result == {('test', 250, 1.0): [[1, 2]]}
    assert constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}


def test_checkpoint_cli_cache_binds_profile_metadata(tmp_path, monkeypatch):
    from scripts import eval_checkpoints
    constructors, generations = mock_generation_stack(monkeypatch)
    monkeypatch.setattr(eval_checkpoints, 'validate_adapter_base', lambda *a: None)
    monkeypatch.setattr(eval_checkpoints.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(eval_checkpoints, 'load_validation_prompts', lambda *a: ['prompt'])
    monkeypatch.setattr(eval_checkpoints, 'score_all', lambda gs, *a: {key: [0.5] for key in gs})
    adapter = checkpoint(tmp_path, 'float32', step=True)
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text('{}')
    dataset = tmp_path / 'dataset'; dataset.write_bytes(b'data')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['eval', '--model', str(model), '--rm', str(model),
        '--run', f'test={adapter.parent.parent}', '--dataset-path', str(dataset),
        '--temps', '1', '--output', str(out), '--policy-head-dtype', 'auto'])
    eval_checkpoints.main()
    assert constructors[0]['hf_overrides'] == {'head_dtype': 'float32'}
    config = json.loads((out / 'manifest.json').read_text())['config']
    assert config['policies']['test/250']['policy_head_dtype'] == 'float32'
    eval_checkpoints.main()
    assert len(generations) == 1
    profile = adapter.parent.parent / 'profile_manifest.json'
    profile.write_text(json.dumps({'policy_head_dtype': 'float32', 'changed': True}))
    with pytest.raises(ValueError, match='different'):
        eval_checkpoints.main()
    assert len(generations) == 1


def test_rm_constructor_never_receives_actor_head_override(monkeypatch):
    from scripts import eval_checkpoints
    import transformers
    import vpo_rm.reward
    calls = []
    class Model:
        base_model = object()
        score = object()
        def to(self, device):
            return self
        def eval(self):
            return self
    def load(*args, **kwargs):
        calls.append(kwargs)
        return Model()
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', load)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(vpo_rm.reward, 'LastTokenReward', lambda *a: Model())
    args = SimpleNamespace(model='actor', rm='reward', policy_head_dtype='float32')
    assert eval_checkpoints.score_all({}, [], [], args) == {}
    assert calls == [{'torch_dtype': eval_checkpoints.torch.bfloat16, 'trust_remote_code': True}]


@pytest.mark.parametrize('entrypoint', ['alpaca', 'checkpoints'])
def test_actual_generation_rejects_mixed_modes_without_engine(tmp_path, monkeypatch, entrypoint):
    constructors, _ = mock_generation_stack(monkeypatch)
    adapter = checkpoint(tmp_path, 'float32', step=True)
    if entrypoint == 'alpaca':
        from scripts import eval_alpaca
        monkeypatch.setattr(sys, 'argv', ['eval', '--adapters', 'base=none', f'new={adapter}',
                                        '--output', str(tmp_path / 'out')])
        invoke = eval_alpaca.main
    else:
        from scripts import eval_checkpoints
        legacy = checkpoint(tmp_path / 'legacy', step=True)
        args = SimpleNamespace(policy_head_dtype='auto')
        invoke = lambda: eval_checkpoints.generate_all(
            [('new', adapter.parent.parent), ('old', legacy.parent.parent)], ['prompt'], [1.0], args)
    with pytest.raises(ValueError, match='separate'):
        invoke()
    assert constructors == []


def test_watcher_command_propagates_resolved_head(tmp_path):
    from scripts.watch_final_alpaca import generation_command
    adapter = checkpoint(tmp_path, 'float32')
    command = generation_command('/source', '/model', adapter, '/refs', '/out', 'test')
    assert command[command.index('--policy-head-dtype') + 1] == 'float32'


def test_watcher_rejects_changed_external_metadata(tmp_path):
    from scripts.eval_artifacts import commit_cache, fingerprint
    from scripts.eval_policy import resolve_policy_head
    from scripts.watch_final_alpaca import validate_generation
    adapter = checkpoint(tmp_path, 'float32', step=True)
    refs = tmp_path / 'refs.jsonl'
    refs.write_text(''.join(json.dumps({'instruction': str(i), 'reference_output': 'ref'}) + '\n' for i in range(805)))
    out = tmp_path / 'out'; out.mkdir()
    gen = out / 'generations_t1.0_n1.jsonl'
    gen.write_text(''.join(json.dumps({'instruction': str(i), 'response': 'answer', 'sample_idx': 0}) + '\n' for i in range(805)))
    config = {'adapter': fingerprint(adapter), 'policy': resolve_policy_head(adapter),
              'recipe': {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1},
              'max_tokens': 2048, 'seed': 42, 'dataset': fingerprint(refs)}
    commit_cache(out / 'manifest_t1.0_n1.json', config, [gen])
    assert len(validate_generation(out, refs, adapter)) == 805
    config['policy']['policy_head_dtype'] = 'native'
    commit_cache(out / 'manifest_t1.0_n1.json', config, [gen])
    with pytest.raises(ValueError, match='head|protocol'):
        validate_generation(out, refs, adapter)
    config['policy'] = resolve_policy_head(adapter)
    commit_cache(out / 'manifest_t1.0_n1.json', config, [gen])
    metadata = adapter.parent.parent / 'profile_manifest.json'
    metadata.write_text(json.dumps({'sampling': {'policy_head_dtype': 'float32'}, 'changed': True}))
    with pytest.raises(ValueError, match='head|protocol'):
        validate_generation(out, refs, adapter)
