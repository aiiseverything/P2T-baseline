import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def launcher():
    return importlib.import_module('scripts.direct_rl_launcher')


@pytest.mark.parametrize('family', ['qwen', 'llama'])
def test_fresh_config_preserves_canonical_algorithm_and_final_only(family):
    m = launcher()
    old = m.baseline_config(family)
    new = m.build_config(family)
    assert new['init_adapter'] == ''
    assert new['checkpoint_interval'] > new['max_rollouts'] == 250
    assert new['keep_adapters_every'] == 250
    changed = {k for k in new if new[k] != old[k]}
    assert changed == {'init_adapter', 'checkpoint_interval'}
    assert new['kl_reference'] == 'init'


@pytest.mark.parametrize('family', ['qwen', 'llama'])
@pytest.mark.parametrize('arm', ['grpo', 'lam4'])
def test_commands_do_not_load_sft_or_none_and_use_shared_scale(family, arm, tmp_path):
    m = launcher()
    manifest = {'common_config': m.build_config(family), 'source_snapshot': str(tmp_path), 'arms': m.ARMS}
    cmd = m.training_command(manifest, arm, tmp_path / 'train', sigma0=2.5)
    assert 'None' not in cmd
    assert '--init-adapter' not in cmd or cmd[cmd.index('--init-adapter') + 1] == ''
    assert cmd[cmd.index('--length-reward-sigma0') + 1] == '2.5'
    assert cmd[cmd.index('--method') + 1] == ('grpo' if arm == 'grpo' else 'vpo_rm')
    assert ('--freeze-stop-tokens' in cmd) == (arm == 'lam4')
    assert cmd[cmd.index('--checkpoint-interval') + 1] == '251'


@pytest.mark.parametrize('bad', [None, 'None', '/some/sft'])
def test_fresh_config_rejects_any_nonempty_initialization(bad):
    m = launcher(); c = m.build_config('llama'); c['init_adapter'] = bad
    with pytest.raises(ValueError, match='initialization'):
        m.validate_config(c)


@pytest.mark.parametrize('key,value', [('checkpoint_interval', 50), ('keep_adapters_every', 0),
    ('keep_adapters_every', 50), ('kl_reference', 'rollout'), ('policy_head_dtype', 'native')])
def test_config_rejects_storage_or_reference_drift(key, value):
    m = launcher(); c = m.build_config('qwen'); c[key] = value
    with pytest.raises(ValueError): m.validate_config(c)


@pytest.mark.parametrize('sigma', [None, 0, -1, float('nan')])
def test_formal_command_requires_fresh_valid_shared_sigma(sigma, tmp_path):
    m = launcher(); manifest = {'common_config': m.build_config('qwen'), 'source_snapshot': str(tmp_path), 'arms': m.ARMS}
    with pytest.raises(ValueError, match='sigma'):
        m.training_command(manifest, 'grpo', tmp_path, sigma0=sigma)


def test_submission_is_exactly_one_three_gpu_replica(tmp_path):
    m = launcher()
    for family in ('qwen', 'llama'):
        for arm in m.ARMS:
            cmd = m.submit_command(tmp_path, family, arm)
            assert cmd[:2] == ['rjob', 'submit']
            assert cmd[cmd.index('--gpu') + 1] == '3'
            assert cmd[cmd.index('--cpu') + 1] == '48'
            assert '-P' not in cmd
            assert cmd[-2:] == [family, arm]


def test_changed_manifest_is_rejected(tmp_path):
    m = launcher(); path = tmp_path / 'experiment.json'
    path.write_text('{"common_config": {}}')
    (tmp_path / 'experiment.sha256').write_text('wrong\n')
    with pytest.raises(ValueError, match='manifest'):
        m.read_manifest(tmp_path)


def test_formal_bad_startup_terminates_running_training(monkeypatch, tmp_path):
    m = launcher()
    class Process:
        stopped = False
        def poll(self): return 1 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 1
    process = Process()
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: process)
    def reject(*args): raise ValueError('Bad startup mapping')
    monkeypatch.setattr(m, 'check_initial_rollouts', reject)
    with pytest.raises(ValueError, match='Bad startup'):
        m.supervise_formal(['fake'], tmp_path, tmp_path, {}, tmp_path)
    assert process.stopped


def test_formal_wrong_initial_weights_terminate_before_further_training(monkeypatch, tmp_path):
    m = launcher()
    class Process:
        stopped = False
        def poll(self): return 1 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 1
    process = Process()
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setattr(m, 'check_initial_rollouts', lambda *a: {'status': 'passed'})
    def reject(*args): raise ValueError('Formal initial LoRA differs')
    monkeypatch.setattr(m, 'verify_formal_initialization', reject)
    with pytest.raises(ValueError, match='initial LoRA'):
        m.supervise_formal(['fake'], tmp_path, tmp_path, {}, tmp_path)
    assert process.stopped
    assert not (tmp_path / 'startup_validation.json').exists()


@pytest.fixture
def completed_run(tmp_path):
    import hashlib
    import torch
    from safetensors.torch import save_file
    scheduled, rows = [], []
    for step in range(1, 251):
        prompts = [f'prompt {step}:{index}' for index in range(8)]
        scheduled.extend(prompts)
        (tmp_path / f'rollout-{step}-prompts.json').write_text(json.dumps(prompts))
        rows.append({'rollout': step, 'optimizer_steps': 1, 'skipped_rollout': False,
                     'input_prompt_groups': 8, 'kept_prompt_groups': 8, 'skipped_groups': 0,
                     'reward_count': 64, 'loss': 0., 'grad_norm': 1.})
    (tmp_path / 'profile_summary.json').write_text(json.dumps({'rollouts': rows}))
    checkpoint = tmp_path / 'checkpoint-250'
    checkpoint.mkdir()
    (checkpoint / 'run_manifest.json').write_text(json.dumps({'step': 250, 'resolved_config': {
        'init_adapter': '', 'model_name': 'actor', 'reward_model_name': 'rm',
        'kl_reference': 'init', 'method': 'grpo'}}))
    save_file({'base.layer.lora_B.weight': torch.ones(2, 2)}, str(checkpoint / 'adapter_model.safetensors'))
    digest = hashlib.sha256(json.dumps(scheduled, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    manifest = {'common_config': {'model': 'actor', 'rm': 'rm'},
                'cpu_protocol': {'formal_prompt_sha256': digest}}
    return tmp_path, manifest, rows, scheduled


@pytest.mark.parametrize('dropped', [1, 8])
def test_completion_accepts_canonical_partial_and_complete_group_skips(completed_run, dropped):
    import hashlib
    directory, manifest, rows, scheduled = completed_run
    kept = scheduled[dropped:8]
    (directory / 'rollout-1-prompts.json').write_text(json.dumps(kept))
    rows[0].update(kept_prompt_groups=len(kept), skipped_groups=dropped, reward_count=len(kept) * 8,
                   skipped_rollout=not kept, optimizer_steps=1 if kept else 0)
    (directory / 'profile_summary.json').write_text(json.dumps({'rollouts': rows}))
    result = launcher().validate_completion(directory, manifest, 'grpo')
    actual = kept + scheduled[8:]
    digest = hashlib.sha256(json.dumps(actual, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert result['actual_kept_prompt_sha256'] == digest
    assert result['kept_prompt_count'] == 2000 - dropped
    assert 'actual_formal_prompt_sha256' not in result
    assert result['optimizer_steps'] == (249 if dropped == 8 else 250)


@pytest.mark.parametrize('mutation', ['kept_count', 'skipped_count', 'reward_count', 'nonlist'])
def test_completion_rejects_prompt_evidence_inconsistent_with_metrics(completed_run, mutation):
    directory, manifest, rows, scheduled = completed_run
    if mutation == 'kept_count': rows[0]['kept_prompt_groups'] = 7
    elif mutation == 'skipped_count': rows[0]['skipped_groups'] = 1
    elif mutation == 'reward_count': rows[0]['reward_count'] = 56
    else: (directory / 'rollout-1-prompts.json').write_text(json.dumps({'prompt': 'x'}))
    (directory / 'profile_summary.json').write_text(json.dumps({'rollouts': rows}))
    with pytest.raises(ValueError, match='prompt|group|response'):
        launcher().validate_completion(directory, manifest, 'grpo')
