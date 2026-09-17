"""Six-GPU Arena launch isolation and parent completion gates, without CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SUITE = PROJECT / 'runs/arena-hard-v2-canonical-20260917'


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path[:])
    spec = importlib.util.spec_from_file_location('arena_parallel_test', SUITE / 'run_parallel_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mask,want', [
    (None, ['0', '1', '2', '3', '4', '5']),
    ('3,5,7,9,11,13', ['3', '5', '7', '9', '11', '13']),
    ('GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f', ['GPU-a','GPU-b','GPU-c','GPU-d','GPU-e','GPU-f']),
])
def test_gpu_assignment_preserves_inherited_physical_tokens(driver, mask, want):
    assert driver.gpu_tokens(6, mask) == want


@pytest.mark.parametrize('count,mask', [(5, None), (6, ''), (6, '0,1,2'), (6, '0,0,1,2,3,4')])
def test_bad_gpu_visibility_rejected(driver, count, mask):
    with pytest.raises(ValueError):
        driver.gpu_tokens(count, mask)


@pytest.fixture
def parent(tmp_path, monkeypatch, driver):
    suite = tmp_path / 'runs/suite'; suite.mkdir(parents=True)
    manifest = {'models': {tag: {'adapter': 'none' if tag == 'base' else f'/models/{tag}'}
                           for tag in driver.TAGS}, 'base_model': '/models/base', 'gpu_versions': {}}
    (suite / 'experiment.json').write_text(json.dumps(manifest))
    (suite / 'status.json').write_text('{"state":"submitted"}')
    monkeypatch.setattr(driver, 'SUITE', suite)
    monkeypatch.setattr(driver, 'SOURCE', suite / 'source')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5,7,9,11,13')
    monkeypatch.delenv('VLLM_PORT', raising=False)
    events, spawned, reaped = [], [], []
    codes = {tag: 0 for tag in driver.TAGS}

    def verify(m): events.append('verify'); return ['questions']
    def runtime(m, expected_devices):
        events.append('runtime')
        return {'device_count': expected_devices, 'gpus': ['NVIDIA H200'] * expected_devices}
    def validate(m, questions): events.append('validate'); return {'all_six': True}
    monkeypatch.setattr(driver, 'verify_inputs', verify)
    monkeypatch.setattr(driver, 'runtime_preflight', runtime)
    monkeypatch.setattr(driver, 'validate_results', validate)

    class Child:
        def __init__(self, command, **kwargs):
            self.tag = command[-1]
            self.pid = 100 + len(spawned)
            self.returncode = None
            events.append('spawn')
            spawned.append((command, kwargs))
        def poll(self): self.returncode = codes[self.tag]; return self.returncode
        def wait(self, timeout=None): reaped.append(self.tag); return codes[self.tag]
        def terminate(self): raise AssertionError('normal/failed children must all be collected')

    monkeypatch.setattr(driver.subprocess, 'Popen', Child)
    return SimpleNamespace(driver=driver, suite=suite, manifest=manifest, events=events,
                           spawned=spawned, reaped=reaped, codes=codes)


def test_all_six_start_isolated_before_wait_and_only_parent_completes(parent):
    p = parent
    p.driver.run_parent()
    assert p.events[:8] == ['verify', 'runtime'] + ['spawn'] * 6
    assert p.events[-1] == 'validate'
    assert p.reaped == list(p.driver.TAGS)
    for tag, token, (command, kwargs) in zip(p.driver.TAGS, ['3','5','7','9','11','13'], p.spawned):
        assert command[-2:] == ['--worker', tag]
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == token
        assert kwargs['env']['OMP_NUM_THREADS'] == '8'
        assert kwargs['env']['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
        assert kwargs['env']['VLLM_CACHE_ROOT'] == f'/tmp/arena-{p.suite.name}-{tag}/vllm'
        assert kwargs['env']['TORCHINDUCTOR_CACHE_DIR'] == f'/tmp/arena-{p.suite.name}-{tag}/inductor'
        assert 'VLLM_PORT' not in kwargs['env']
        assert kwargs['cwd'] == str(p.suite.parents[1])
        assert (p.suite / 'job' / tag / 'exit_code').read_text().strip() == '0'
    assert json.loads((p.suite / 'status.json').read_text())['state'] == 'generation_complete'
    assert json.loads((p.suite / 'generation_summary.json').read_text()) == {'all_six': True}
    assert (p.suite / 'generation_complete.json').exists()


def test_one_failure_still_collects_all_six_without_complete_marker(parent):
    p = parent
    p.codes['lam2'] = 1
    with pytest.raises(RuntimeError, match='lam2'):
        p.driver.run_parent()
    assert len(p.spawned) == len(p.reaped) == 6
    assert 'validate' not in p.events
    state = json.loads((p.suite / 'status.json').read_text())
    assert state['state'] == 'generation_failed'
    assert state['exit_codes']['lam2'] == 1
    assert not (p.suite / 'generation_complete.json').exists()


def test_preflight_dependency_failure_is_recorded_before_any_child(parent, monkeypatch):
    def fail(*a, **kw): raise ImportError('pandas is unavailable')
    monkeypatch.setattr(parent.driver, 'runtime_preflight', fail)
    with pytest.raises(ImportError, match='pandas'):
        parent.driver.run_parent()
    assert parent.spawned == []
    state = json.loads((parent.suite / 'status.json').read_text())
    assert state['state'] == 'generation_failed' and 'pandas' in state['error']


def test_invalid_final_results_cannot_mark_complete(parent, monkeypatch):
    def fail(*a): raise ValueError('coverage mismatch')
    monkeypatch.setattr(parent.driver, 'validate_results', fail)
    with pytest.raises(ValueError, match='coverage'):
        parent.driver.run_parent()
    assert len(parent.reaped) == 6
    assert not (parent.suite / 'generation_complete.json').exists()
    assert json.loads((parent.suite / 'status.json').read_text())['state'] == 'generation_failed'


def test_worker_executes_frozen_single_policy_cli(parent, monkeypatch):
    calls = []
    def execute(executable, command): calls.append((executable, command)); raise SystemExit(0)
    monkeypatch.setattr(parent.driver.os, 'execv', execute)
    with pytest.raises(SystemExit): parent.driver.run_worker('lam4')
    executable, command = calls[0]
    assert executable == sys.executable
    assert command[1] == str(parent.suite / 'source/scripts/eval_arena_hard.py')
    assert command[-2:] == ['--adapters', 'lam4=/models/lam4']
    for flag, value in [('--max-tokens','4096'),('--max-model-len','16384'),
                        ('--max-num-seqs','32'),('--policy-head-dtype','float32'),('--seed','42')]:
        assert command[command.index(flag)+1] == value
    assert parent.events == ['runtime']
    runtime = json.loads((parent.suite / 'job/lam4/runtime.json').read_text())
    assert runtime['device_count'] == 1


def test_shell_wrapper_preserves_mask_and_dependency_priority(tmp_path):
    suite = tmp_path / 'runs/suite'; suite.mkdir(parents=True)
    wrapper = suite / 'run_parallel_evaluation.sh'
    wrapper.write_bytes((SUITE / wrapper.name).read_bytes())
    bindir = tmp_path / 'bin'; bindir.mkdir()
    fake = bindir / 'python3'
    fake.write_text(f'#!{sys.executable}\nimport json,os\nfrom pathlib import Path\n'
                    "Path(os.environ['CAPTURE']).write_text(json.dumps(dict(os.environ)))\n")
    fake.chmod(0o755)
    capture = tmp_path / 'environment.json'
    env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', CAPTURE=str(capture),
               CUDA_VISIBLE_DEVICES='GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f', PYTHONPATH='/inherited')
    result = subprocess.run(['bash', str(wrapper)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads(capture.read_text())
    assert saved['PYTHONPATH'].split(':')[:3] == [str(suite/'source'), str(suite/'.arena-extra'), str(suite/'source/.vllm-extra')]
    assert saved['CUDA_VISIBLE_DEVICES'] == env['CUDA_VISIBLE_DEVICES']
    assert saved['OMP_NUM_THREADS'] == '8' and saved['TIKTOKEN_CACHE_DIR'] == str(suite/'.tiktoken-cache')
