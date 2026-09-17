"""Direct policy identities and single-H200 offline reward evaluation."""
import json
import sys
from pathlib import Path

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
