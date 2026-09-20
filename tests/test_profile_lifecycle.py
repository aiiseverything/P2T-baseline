import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import profile_vllm_full as profile


def test_profile_provenance_binds_reward_implementations_and_protocol():
    manifest = profile.profile_source_manifest()
    assert manifest['reward_input_protocol'] == 'canonical_chat_v1'
    for name in ('vpo_rm/reward.py', 'vpo_rm/reward_inputs.py'):
        assert manifest['source_sha256'][name] == hashlib.sha256((profile.ROOT / name).read_bytes()).hexdigest()


def test_resolved_profile_rejects_changed_physical_reward_microbatch(tmp_path, monkeypatch):
    from dataclasses import replace
    original = profile.TrainerConfig.resolved
    monkeypatch.setattr(profile.TrainerConfig, 'resolved',
                        lambda self: replace(original(self), microbatch_responses=2))
    args = profile.parse_args(['--output-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='microbatch'):
        profile.build_trainer_config(args, tmp_path)


def test_profile_rejects_actual_trainer_microbatch_before_calibration(tmp_path, monkeypatch):
    args = profile.parse_args(['--output-dir', str(tmp_path)])
    monkeypatch.setattr(profile, 'parse_args', lambda: args)
    monkeypatch.setattr(profile, 'vllm_subprocess_environment', lambda *a: dict(os.environ))
    monkeypatch.setattr(profile.torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(profile.torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(profile, 'load_prompt_dataset', lambda *a, **k: (['p'] * 8, [], {}))
    monkeypatch.setattr(profile.VPOTrainer, 'from_pretrained',
                        lambda config: SimpleNamespace(cfg=SimpleNamespace(microbatch_responses=2)))
    with pytest.raises(ValueError, match='microbatch'):
        profile.main()


@pytest.mark.parametrize('checker', ['scripts/preflight_training.py',
                                    'scripts/preflight_quality.py',
                                    'scripts/check_ssh_capacity.py'])
def test_gpu_report_rejected_when_its_checker_source_changes(tmp_path, monkeypatch, checker):
    import json
    from scripts.corrected_rl_launcher import ARMS, RUNTIME, validate_report
    sources = set(profile.profile_source_manifest()['source_sha256']) | {
        'scripts/preflight_training.py', 'scripts/preflight_quality.py',
        'scripts/check_ssh_capacity.py'}
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((profile.ROOT / name).read_bytes())
    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    identity = profile.profile_source_manifest()
    report = tmp_path / 'gpu-validation.json'
    report.write_text(json.dumps({'status': 'passed', **identity,
        'arms': {arm: {'status': 'passed', 'rollouts': 2,
                      'runtime': {'versions': RUNTIME}} for arm in ARMS}}))
    validate_report(report, identity, gpu=True)
    with (tmp_path / checker).open('a') as stream:
        stream.write('\n# checker changed after validation\n')
    with pytest.raises(ValueError, match='source_sha256'):
        validate_report(report, profile.profile_source_manifest(), gpu=True)


def test_vllm_server_starts_in_its_own_process_group(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, 'vllm_server_command', lambda *args: [sys.executable, '-c', 'import time; time.sleep(60)'])
    server = profile.start_vllm_server(None, tmp_path / 'unused', dict(os.environ))
    try:
        assert os.getpgid(server.pid) == server.pid
        assert os.getpgid(server.pid) != os.getpgrp()
    finally:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=5)


def test_shutdown_uses_short_request_timeout_before_terminating_group(monkeypatch):
    requests, signals = [], []
    class Process:
        pid = 987654321
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            self.returncode = -signal.SIGTERM
            return self.returncode
    def request(payload, *, timeout):
        requests.append((payload, timeout))
        raise TimeoutError('unresponsive generation server')
    monkeypatch.setattr(profile.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    profile.shutdown_vllm_server(Process(), request)
    assert requests == [({'shutdown': True}, 5)]
    assert signals == [(987654321, signal.SIGTERM), (987654321, signal.SIGKILL)]


def test_shutdown_cleans_workers_even_when_server_already_exited(tmp_path):
    worker_pid = tmp_path / 'worker.pid'
    script = ('import pathlib,subprocess,sys; '
              'worker=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"]); '
              'pathlib.Path(sys.argv[1]).write_text(str(worker.pid))')
    server = subprocess.Popen([sys.executable, '-c', script, str(worker_pid)], start_new_session=True)
    server.wait(timeout=5)
    pid = int(worker_pid.read_text())
    try:
        profile.shutdown_vllm_server(server, lambda *a, **k: pytest.fail('exited server needs no request'))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('orphaned worker survived server cleanup')
    finally:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_sigterm_during_startup_runs_server_group_cleanup(monkeypatch, tmp_path):
    import json
    created = []
    args = profile.parse_args(['--output-dir', str(tmp_path / 'train'), '--max-rollouts', '1'])
    monkeypatch.setattr(profile, 'parse_args', lambda: args)
    monkeypatch.setattr(profile.torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(profile.torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(profile.torch.cuda, 'get_device_name', lambda i: 'test-device')
    monkeypatch.setattr(profile, 'vllm_subprocess_environment', lambda *a: dict(os.environ))
    monkeypatch.setattr(profile, 'load_prompt_dataset', lambda *a, **k: (['p'] * 8, [], {}))
    fake = SimpleNamespace(cfg=SimpleNamespace(microbatch_responses=1),
                           filter_prompts=lambda values: values, filtered_prompt_count=0,
                           sampling_manifest=lambda: {},
                           actor=SimpleNamespace(save_pretrained=lambda path: path.mkdir()))
    monkeypatch.setattr(profile.VPOTrainer, 'from_pretrained', lambda config: fake)
    def start(*args):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        created.append(process)
        return process
    monkeypatch.setattr(profile, 'start_vllm_server', start)
    monkeypatch.setattr(profile, 'time', SimpleNamespace(monotonic=time.monotonic,
                        sleep=lambda _: signal.raise_signal(signal.SIGTERM)))
    previous = signal.getsignal(signal.SIGTERM)
    # Suppress the real default termination in the pre-fix implementation so
    # the missing cleanup is an assertion failure rather than killing pytest.
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(RuntimeError('missing SIGTERM cleanup handler')))
    try:
        with pytest.raises(SystemExit) as result:
            profile.main()
        assert result.value.code == 128 + signal.SIGTERM
        assert created and created[0].poll() is not None
        manifest = json.loads((tmp_path / 'train/profile_manifest.json').read_text())
        assert manifest['reward_input_protocol'] == 'canonical_chat_v1'
        assert 'vpo_rm/reward_inputs.py' in manifest['source_sha256']
    finally:
        signal.signal(signal.SIGTERM, previous)
        for process in created:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def test_profile_provenance_binds_llama_launcher_and_protocol_checker(tmp_path, monkeypatch):
    for name in profile.profile_source_manifest()['source_sha256']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((profile.ROOT / name).read_bytes())
    names = ('scripts/llama_rl_launcher.py', 'scripts/check_llama_protocol.py')
    for name in names:
        (tmp_path / name).write_text('# independent Llama validation\n')
    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    before = profile.profile_source_manifest()['source_sha256']
    assert all(name in before for name in names)
    for name in names:
        (tmp_path / name).write_text('# changed validation\n')
    after = profile.profile_source_manifest()['source_sha256']
    assert all(before[name] != after[name] for name in names)
