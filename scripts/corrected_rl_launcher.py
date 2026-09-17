#!/usr/bin/env python3
"""Prepare a verified canonical-RM suite; never submit or delete cluster jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint

ARMS = {'grpo': {'method': 'grpo', 'credit_lambda': 1, 'freeze_stop_tokens': False, 'freeze_structural': False},
        **{f'lam{value}': {'method': 'vpo_rm', 'credit_lambda': value,
                          'freeze_stop_tokens': True, 'freeze_structural': True} for value in (2, 4, 8)}}
RUNTIME = {'torch': '2.13.0+cu129', 'transformers': '5.16.1',
           'vllm': '0.28.1rc1.dev199+g7c5dc571c.cu129', 'peft': '0.20.0', 'pyarrow': '21.0.0'}
PROTECTED_SFT_SHA256 = '21c0c7b9e75c640b03a3fddfc6bb1e7a478e02c6724111800d605187364761ad'
ROLLOUT_CORRECTION_PROTOCOL = 'detached_token_is_pg_and_kl_v1'


def common_config(project_root):
    root = Path(project_root).resolve()
    return {'model': str(root / 'models/Qwen3-14B-Base'),
            'rm': str(root / 'models/Skywork-Reward-V2-Qwen3-8B'),
            'dataset_path': str(root / 'datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet'),
            'init_adapter': str(root / 'models/sft-native-eos-clean2k5e2'),
            'max_rollouts': 250, 'max_response_tokens': 2048, 'generation_microbatch': 32,
            'keep_adapters_every': 250, 'checkpoint_interval': 250, 'learning_rate': 5e-5,
            'tau': 1, 'beta': .03, 'kl_reference': 'init', 'temperature': 1,
            'policy_epochs': 1, 'optimizer_minibatch_responses': 64, 'seed': 42,
            'generation_seed': 0, 'length_reward_mode': 'soft', 'min_response_tokens': 0,
            'short_response_threshold': 8, 'long_response_threshold': 1024,
            'short_penalty_strength': .5, 'long_penalty_strength': 2,
            'advantage_std_floor_fraction': .5, 'length_calibration_prompts': 128,
            'degenerate_newline_run': 32, 'length_penalty_slope': 0,
            'credit_microbatch_responses': 1, 'policy_head_dtype': 'float32',
            'vllm_gpu_memory_utilization': .45,
            'vllm_tensor_parallel_size': 1}


def validation_identity(project_root=ROOT):
    from scripts.profile_vllm_full import profile_source_manifest
    from vpo_rm.data import DEFAULT_BENCHMARK_PATHS, ROOT as DATA_ROOT
    project_root = Path(project_root).resolve()
    config = common_config(project_root)
    init_hash = file_hash(Path(config['init_adapter']) / 'adapter_model.safetensors')
    if init_hash != PROTECTED_SFT_SHA256:
        raise ValueError('Protected SFT initialization weights changed')
    return {**profile_source_manifest(),
            'config': {key: value for key, value in config.items()
                       if key not in {'max_rollouts', 'output_dir', 'length_reward_sigma0'}},
            'initialization_weights_sha256': init_hash,
            'dataset_sha256': file_hash(config['dataset_path']),
            'input_fingerprints': {
                'actor': fingerprint(config['model'], full_weights=False),
                'reward': fingerprint(config['rm'], full_weights=False),
                'initialization': fingerprint(config['init_adapter'], full_weights=True)},
            'benchmark_exclusion_sha256': {
                name: file_hash(project_root / path.relative_to(DATA_ROOT))
                for name, path in DEFAULT_BENCHMARK_PATHS.items()}}


def validate_report(path, identity, *, gpu=False):
    report = json.loads(Path(path).read_text())
    if report.get('status') != 'passed':
        raise ValueError(f'Required validation has not passed: {path}')
    for key, value in identity.items():
        if report.get(key) != value:
            raise ValueError(f'Validation {key} differs from current experiment: {path}')
    if gpu:
        arms = report.get('arms', {})
        if set(arms) != set(ARMS) or any(
                row.get('status') != 'passed' or row.get('rollouts') != 2 for row in arms.values()):
            raise ValueError('GPU validation must pass two rollouts for all four arms')
        if any(row.get('runtime', {}).get('versions') != RUNTIME for row in arms.values()):
            raise ValueError('GPU validation runtime versions differ from the formal runtime')
    return report


def validate_prepared_manifest(manifest, identity):
    if (manifest.get('common_config') != common_config(manifest['project_root'])
            or manifest.get('arms') != ARMS or manifest.get('validation_identity') != identity
            or manifest.get('reward_input_protocol') != 'canonical_chat_v1'):
        raise ValueError('Prepared experiment config or identity differs from validated inputs')


def training_command(manifest, arm, output, *, sigma0=None, pilot=False):
    config = dict(manifest['common_config'])
    selected = manifest['arms']['grpo' if pilot else arm]
    if pilot:
        config['max_rollouts'] = 2
    command = [sys.executable, str(Path(manifest['source_snapshot']) / 'scripts/profile_vllm_full.py'),
               '--output-dir', str(output), '--method', selected['method'],
               '--credit-lambda', str(selected['credit_lambda'])]
    for key, value in config.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    if sigma0 is not None:
        command += ['--length-reward-sigma0', str(sigma0)]
    for key in ('freeze_stop_tokens', 'freeze_structural'):
        if selected[key]:
            command.append('--' + key.replace('_', '-'))
    return command


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_calibration(record, config):
    expected = {'reward_format': 'canonical_chat_v1', 'mode': 'soft', 'source': 'initial_policy',
                'calibration_prompt_count': 128, 'group_size': 8, 'seed': config['seed'],
                'model': config['model'], 'reward_model': config['rm'], 'init_adapter': config['init_adapter']}
    expected.update({key: config[key] for key in ('short_response_threshold', 'long_response_threshold',
                    'short_penalty_strength', 'long_penalty_strength', 'advantage_std_floor_fraction',
                    'max_response_tokens')})
    sigma = record.get('sigma0')
    if (not isinstance(sigma, (int, float)) or isinstance(sigma, bool)
            or not math.isfinite(sigma) or sigma <= 0
            or any(record.get(key) != value for key, value in expected.items())):
        raise ValueError('Shared calibration does not match canonical inputs/configuration')
    sampling = record.get('sampling', {})
    if any(sampling.get(key) != value for key, value in
           {'temperature': 1, 'top_p': 1, 'top_k': 0, 'min_tokens': 0, 'presence_penalty': 0}.items()):
        raise ValueError('Shared calibration sampling configuration differs')
    if (config.get('policy_head_dtype') != 'float32'
            or sampling.get('policy_head_dtype') != config['policy_head_dtype']
            or sampling.get('rollout_correction') != ROLLOUT_CORRECTION_PROTOCOL):
        raise ValueError('Shared calibration policy precision or rollout correction differs')
    if len(record.get('responses', [])) != 1024 or len(set(record.get('prompt_sha256', []))) != 128:
        raise ValueError('Shared calibration must cover 128 distinct prompts x 8 responses')
    return record


def check_initial_rollouts(train_dir, manifest):
    train_dir = Path(train_dir)
    metrics_path = train_dir / 'profile_metrics.jsonl'
    if not metrics_path.exists():
        return None
    # A concurrently appended final line is not yet a committed metric record.
    lines = metrics_path.read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith('\n')]
    if len(rows) < 2:
        return None
    rows = rows[:2]
    if [row.get('rollout') for row in rows] != [1, 2] or any(row.get('skipped_rollout') for row in rows):
        raise ValueError('Initial two rollouts must perform training updates')
    profile = json.loads((train_dir / 'profile_manifest.json').read_text())
    if profile.get('reward_input_protocol') != 'canonical_chat_v1':
        raise ValueError('Initial rollouts used a different RM input protocol')
    corrected = 'policy_head_dtype' in manifest['common_config']
    if corrected and (
            manifest['common_config']['policy_head_dtype'] != 'float32'
            or profile.get('config', {}).get('policy_head_dtype') != 'float32'
            or profile.get('config', {}).get('rollout_importance_correction') is not True
            or profile.get('sampling', {}).get('policy_head_dtype') != 'float32'
            or profile.get('sampling', {}).get('rollout_correction') != ROLLOUT_CORRECTION_PROTOCOL):
        raise ValueError('Initial rollout policy precision or correction protocol differs')
    is_keys = ('rollout_is_min', 'rollout_is_mean', 'rollout_is_p01', 'rollout_is_p50',
               'rollout_is_p99', 'rollout_is_max', 'rollout_is_ess_ratio')
    probability_keys = ('rollout_logp_abs_error_mean', 'rollout_logp_abs_error_p99',
                        'rollout_logp_abs_error_max', 'rollout_direct_ratio_clip_fraction',
                        'initial_hf_logp_max_abs_error', 'initial_hf_ratio_clip_fraction')
    stop_ids = set(profile['sampling']['stop_token_ids'])
    maximum = manifest['common_config']['max_response_tokens']
    threshold = manifest.get('startup_max_unmapped_content_fraction', .25)
    summaries = []
    for row in rows:
        if corrected:
            if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0
                   for key in is_keys):
                raise ValueError('Initial importance sampling metrics must be finite and positive')
            if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                   for key in probability_keys):
                raise ValueError('Initial probability and HF consistency metrics must be finite')
            if row['initial_hf_ratio_clip_fraction'] != 0:
                raise ValueError('The initial HF ratio clip fraction must be zero')
        if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
               for key in ('loss', 'grad_norm', 'rm_unmapped_content_fraction')):
            raise ValueError('Initial loss, gradients and mapping metrics must be finite')
        if row['grad_norm'] <= 0 or row.get('rm_mapped_tokens', 0) <= 0 or not 0 <= row['rm_unmapped_content_fraction'] <= threshold:
            raise ValueError('Initial gradient/mapping coverage is outside validated bounds')
        step = row['rollout']
        tokens = json.loads((train_dir / f'rollout-{step}-tokens.json').read_text())
        rewards = json.loads((train_dir / f'rollout-{step}-rewards.json').read_text())
        if len(tokens) != row['reward_count'] or len(tokens) != len(rewards):
            raise ValueError('Initial response coverage differs from metrics')
        for ids, reward in zip(tokens, rewards):
            terminal = bool(ids) and ids[-1] in stop_ids
            stop_count = sum(token in stop_ids for token in ids)
            valid = (reward['finish_reason'] == 'stop' and terminal and stop_count == 1
                     or reward['finish_reason'] == 'length' and not stop_count and len(ids) == maximum)
            if not valid or reward['length'] != len(ids):
                raise ValueError('Initial stop/length termination metadata disagrees with tokens')
        if sum(map(len, tokens)) != row['response_tokens']:
            raise ValueError('Initial token count differs from metrics')
        summaries.append({key: row[key] for key in ('rollout', 'loss', 'grad_norm', 'rm_mapped_tokens',
                                                   'rm_unmapped_content_fraction', 'response_tokens')})
        if corrected:
            summaries[-1].update({key: row[key] for key in is_keys + probability_keys})
    return {'status': 'passed', 'rollouts': summaries, 'reward_input_protocol': 'canonical_chat_v1',
            'max_unmapped_content_fraction': threshold}


def prepare_suite(suite, cpu_validation, gpu_validation, *, project_root=ROOT, submit_template=None):
    project_root, suite = Path(project_root).resolve(), Path(suite).resolve()
    identity = validation_identity(project_root)
    validate_report(cpu_validation, identity)
    gpu = validate_report(gpu_validation, identity, gpu=True)
    config = common_config(project_root)
    calibration_path, calibration = None, None
    if gpu.get('calibration') is not None:
        calibration_path = Path(gpu['calibration']['path']).resolve()
        if file_hash(calibration_path) != gpu['calibration']['sha256']:
            raise ValueError('Preflight calibration hash changed')
        calibration = validate_calibration(json.loads(calibration_path.read_text()), config)
    if suite.exists():
        raise FileExistsError(f'Use a fresh suite directory: {suite}')
    free = shutil.disk_usage(project_root).free / 2**30
    if free < 60:
        raise RuntimeError(f'Need 60 GiB free before preparing this four-arm suite; have {free:.1f}')
    template_path = Path(submit_template) if submit_template else (
        project_root / 'runs/rl-soft-native-eos-20260917-003006/grpo/submit_command.json')
    template = json.loads(template_path.read_text())
    prefix = template[:template.index('--')]
    if prefix[:2] != ['rjob', 'submit']:
        raise ValueError('Expected a recorded rjob submit command template')
    for flag in ('--name', '--gpu', '--cpu', '--memory'):
        if flag not in prefix:
            raise ValueError(f'Submission template lacks {flag}')
    if '--image' in prefix and gpu.get('runtime_image') != prefix[prefix.index('--image') + 1]:
        raise ValueError('GPU validation runtime image differs from submission template image')
    suite.mkdir(parents=True)
    source = suite / 'source'
    source.mkdir()
    for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs'):
        shutil.copytree(project_root / directory, source / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('pyproject.toml',):
        shutil.copy2(project_root / name, source / name)
    for directory in ('models', 'datasets', '.vllm-extra'):
        if (project_root / directory).exists():
            (source / directory).symlink_to(project_root / directory, target_is_directory=True)
    for name, expected in identity['source_sha256'].items():
        if file_hash(source / name) != expected:
            raise ValueError(f'Source changed while freezing suite: {name}')
    snapshot = {str(path.relative_to(source)): file_hash(path)
                for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs')
                for path in (source / directory).rglob('*') if path.is_file()}
    snapshot['pyproject.toml'] = file_hash(source / 'pyproject.toml')
    validation = {}
    for kind, path in (('cpu', cpu_validation), ('gpu', gpu_validation)):
        target = suite / f'{kind}-validation.json'
        shutil.copy2(path, target)
        validation[kind] = {'path': str(target), 'sha256': file_hash(target)}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'suite': str(suite),
                'project_root': str(project_root), 'source_snapshot': str(source),
                'source_sha256': snapshot, 'validation_identity': identity, 'validation': validation,
                'common_config': config, 'arms': ARMS, 'runtime': RUNTIME,
                'reward_input_protocol': 'canonical_chat_v1',
                'startup_max_unmapped_content_fraction': .25,
                'storage_budget': {'startup_min_free_gib': 60, 'free_gib_at_creation': free,
                                   'retention': 'final checkpoint; step0/current/final adapters'},
                'calibration': {'mode': 'verified_gpu_preflight' if calibration else 'fresh_flock_pilot',
                                'max_wait_seconds': 3600},
                'initialization_weights_sha256': identity['initialization_weights_sha256'],
                'dataset_sha256': identity['dataset_sha256']}
    if calibration is not None:
        calibration = {**calibration, '_verification': {'identity': identity,
            'source_calibration_sha256': file_hash(calibration_path),
            'gpu_validation_sha256': validation['gpu']['sha256']}}
        write_json(suite / 'shared-calibration.json', calibration)
        manifest['calibration']['shared_sha256'] = file_hash(suite / 'shared-calibration.json')
    write_json(suite / 'experiment.json', manifest)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?arm required}"
case "$ARM" in grpo|lam2|lam4|lam8) ;; *) exit 2 ;; esac
export PYTHONPATH="$SUITE/source:$SUITE/source/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRESENCE_PENALTY=0.0 VLLM_WORKER_MULTIPROC_METHOD=spawn
exec > >(tee -a "$SUITE/$ARM/job.log") 2>&1
trap 'task_rc=$?; printf "%s\\n" "$task_rc" > "$SUITE/$ARM/exit_code"' EXIT
python3 "$SUITE/source/scripts/corrected_rl_launcher.py" run-arm --suite "$SUITE" --arm "$ARM"
'''
    atomic_text(suite / 'run_arm.sh', shell)
    names = {}
    suffix = hashlib.sha256(str(suite).encode()).hexdigest()[:10]
    for arm in ARMS:
        (suite / arm).mkdir()
        command = list(prefix)
        name = f'rl-can-{arm}-{suffix}'
        for flag, value in (('--name', name), ('--gpu', '3'), ('--cpu', '48'), ('--memory', '600000')):
            command[command.index(flag) + 1] = value
        command += ['--', 'bash', str(suite / 'run_arm.sh'), arm]
        write_json(suite / arm / 'submit_command.json', command)
        names[arm] = name
    write_json(suite / 'submission-plan.json', {'status': 'prepared_not_submitted', 'names': names,
                                              'experiment_sha256': file_hash(suite / 'experiment.json')})
    print(f'PREPARED_NOT_SUBMITTED suite={suite} manifest_sha256={file_hash(suite / "experiment.json")}', flush=True)
    return manifest


def run_training_process(command, project_root, train_dir, manifest, *, status_dir):
    """Supervise one newly started child and publish the first-two-rollout gate."""
    status_dir = Path(status_dir)
    process = subprocess.Popen(command, cwd=project_root)
    passed = None
    try:
        while True:
            if passed is None:
                passed = check_initial_rollouts(train_dir, manifest)
                if passed is not None:
                    write_json(status_dir / 'startup_validation.json', passed)
                    atomic_text(status_dir / 'stage', 'training_validated\n')
                    print('INITIAL_TWO_ROLLOUTS_PASSED ' + json.dumps(passed, allow_nan=False), flush=True)
            code = process.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f'Training subprocess exited with {code}')
                if passed is None:
                    # The child may finish just after the last metric read.
                    passed = check_initial_rollouts(train_dir, manifest)
                    if passed is None:
                        raise RuntimeError('Training exited without two validated rollout records')
                    write_json(status_dir / 'startup_validation.json', passed)
                return passed
            time.sleep(1)
    except BaseException as exc:
        write_json(status_dir / 'startup_or_training_failure.json', {'status': 'failed', 'error': repr(exc)})
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=75)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def shared_calibration(suite, arm, manifest):
    ready, failed = suite / 'shared-calibration.json', suite / 'calibration-failed.json'
    started = time.monotonic()
    with (suite / 'calibration.lock').open('a') as lock:
        while True:
            if failed.exists():
                raise RuntimeError(f'Calibration failed; inspect {failed}')
            if ready.exists():
                record = json.loads(ready.read_text())
                verification = record.get('_verification', {})
                if (verification.get('identity') != manifest['validation_identity']
                        or verification.get('gpu_validation_sha256') != manifest['validation']['gpu']['sha256']):
                    raise ValueError('Shared calibration identity differs from validated suite')
                expected = manifest['calibration'].get('shared_sha256')
                if expected and file_hash(ready) != expected:
                    raise ValueError('Prepared shared calibration changed')
                return validate_calibration(record, manifest['common_config'])
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() - started > manifest['calibration']['max_wait_seconds']:
                    raise TimeoutError('Shared calibration wait exceeded its time limit')
                time.sleep(5)
                continue
            try:
                if ready.exists():
                    continue
                if failed.exists():
                    raise RuntimeError('Shared calibration failed in another arm')
                pilot = suite / 'calibration-pilot'
                if pilot.exists():
                    raise FileExistsError('Incomplete prior calibration; refusing to overwrite')
                command = training_command(manifest, 'grpo', pilot, pilot=True)
                write_json(suite / 'calibration-command.json', command)
                atomic_text(suite / 'calibration-owner', arm + '\n')
                run_training_process(command, manifest['project_root'], pilot, manifest, status_dir=suite)
                record_path = pilot / 'length_reward_calibration.json'
                record = validate_calibration(json.loads(record_path.read_text()), manifest['common_config'])
                summary = json.loads((pilot / 'profile_summary.json').read_text())
                if [row['rollout'] for row in summary['rollouts']] != [1, 2]:
                    raise ValueError('Calibration pilot did not finish exactly two rollouts')
                record = {**record, '_verification': {'identity': manifest['validation_identity'],
                          'source_calibration_sha256': file_hash(record_path),
                          'gpu_validation_sha256': manifest['validation']['gpu']['sha256'],
                          'pilot_arm_owner': arm}}
                write_json(ready, record)
            except BaseException as exc:
                write_json(failed, {'status': 'failed', 'error': repr(exc), 'arm': arm})
                raise
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def run_arm(suite, arm):
    suite = Path(suite).resolve()
    manifest_path = suite / 'experiment.json'
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads((suite / 'submission-plan.json').read_text())
    if file_hash(manifest_path) != plan['experiment_sha256']:
        raise ValueError('Prepared experiment manifest changed')
    if arm not in ARMS or manifest['arms'] != ARMS:
        raise ValueError('Unknown or changed experiment arm')
    out = suite / arm
    if (out / 'train').exists() or (out / 'completion.json').exists():
        raise FileExistsError(f'Arm already has training output: {out}')
    os.chdir(manifest['project_root'])
    source = Path(manifest['source_snapshot'])
    if ROOT.resolve() != source.resolve():
        raise ValueError('Run the launcher from the frozen source snapshot')
    for name, expected in manifest['source_sha256'].items():
        if file_hash(source / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    identity = validation_identity(manifest['project_root'])
    validate_prepared_manifest(manifest, identity)
    for kind, record in manifest['validation'].items():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError(f'{kind} validation record changed')
        validate_report(record['path'], identity, gpu=kind == 'gpu')
    free = shutil.disk_usage(suite).free / 2**30
    if free < manifest['storage_budget']['startup_min_free_gib']:
        raise RuntimeError(f'Insufficient free storage for suite startup: {free:.1f} GiB')
    import torch
    if torch.cuda.device_count() != 3 or any(torch.cuda.get_device_properties(i).total_memory < 120 * 2**30 for i in range(3)):
        raise RuntimeError('Formal suite requires three H200-class GPUs')
    versions = {package: importlib.metadata.version(package) for package in manifest['runtime']}
    if versions != manifest['runtime']:
        raise RuntimeError(f'Runtime differs from validated software: {versions}')
    runtime = {'software': versions, 'gpus': [torch.cuda.get_device_name(i) for i in range(3)],
               'free_gib': free, 'experiment_sha256': file_hash(manifest_path),
               'source_snapshot': str(source), 'reward_input_protocol': identity['reward_input_protocol']}
    write_json(out / 'runtime.json', runtime)
    print(f'EXPERIMENT_MANIFEST path={manifest_path} sha256={file_hash(manifest_path)}', flush=True)
    print('VERIFIED_RUNTIME ' + json.dumps(runtime), flush=True)
    atomic_text(out / 'stage', 'calibration\n')
    calibration = shared_calibration(suite, arm, manifest)
    write_json(out / 'shared-calibration.json', calibration)
    print(f'SHARED_CALIBRATION path={suite / "shared-calibration.json"} '
          f'sha256={file_hash(suite / "shared-calibration.json")} sigma0={calibration["sigma0"]}', flush=True)
    command = training_command(manifest, arm, out / 'train', sigma0=calibration['sigma0'])
    write_json(out / 'command.json', command)
    print('TRAIN_COMMAND ' + json.dumps(command), flush=True)
    atomic_text(out / 'stage', 'training\n')
    started = time.monotonic()
    run_training_process(command, manifest['project_root'], out / 'train', manifest, status_dir=out)
    summary = json.loads((out / 'train/profile_summary.json').read_text())
    if [row['rollout'] for row in summary['rollouts']] != list(range(1, 251)):
        raise ValueError('Formal run did not finish all 250 rollouts')
    checkpoint = out / 'train/checkpoint-250'
    if json.loads((checkpoint / 'run_manifest.json').read_text())['step'] != 250 or not (checkpoint / 'adapter_model.safetensors').stat().st_size:
        raise ValueError('Final checkpoint is incomplete')
    write_json(out / 'completion.json', {'rollouts': 250, 'wall_seconds': time.monotonic() - started,
               'sigma0': calibration['sigma0'], 'reward_input_protocol': identity['reward_input_protocol'],
               'experiment_sha256': file_hash(manifest_path)})
    atomic_text(out / 'stage', 'complete\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('prepare', help='Freeze a validated suite and submission JSON; does not submit')
    prepare.add_argument('--suite', type=Path, required=True)
    prepare.add_argument('--cpu-validation', type=Path, required=True)
    prepare.add_argument('--gpu-validation', type=Path, required=True)
    prepare.add_argument('--project-root', type=Path, default=ROOT)
    prepare.add_argument('--submit-template', type=Path)
    run = commands.add_parser('run-arm', help='Run a prepared arm inside an explicitly submitted GPU job')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare_suite(args.suite, args.cpu_validation, args.gpu_validation,
                      project_root=args.project_root, submit_template=args.submit_template)
    else:
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        try:
            with (args.suite / args.arm / 'arm.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_arm(args.suite, args.arm)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
