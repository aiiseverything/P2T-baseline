#!/usr/bin/env python3
"""Freeze and run credit ablations of canonical VPO lambda4; never submit jobs.

Everything is inherited from the canonical suite ``runs/rl-fp32-is-canonical-20260917``:
actor, reward model, protected SFT initialization, dataset and prompt order,
shared sigma0, sampling, optimizer, KL and lambda-band settings. The only
difference is the token credit source: instead of reward-model input gradients,
the arm draws its token weights at random inside the same [1/lambda, lambda]
band (``random_direction`` feeds standard normal noise through the unchanged
allocator; ``random_band`` draws band-uniform weights and projects them onto the
response budget). ``shuffle`` permutes all valid response weights after normal RM allocation;
``norm_product`` replaces the logit-space gradient/score inner product with
the product of their L2 norms. Neither new arm skips reward input gradients.
``prepare`` proves this single-variable property by resolving
the actual training command and diffing it against the canonical arm's recorded
resolved configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import corrected_rl_launcher as corrected
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint

RUNTIME = corrected.RUNTIME
RUNTIME_IMAGE = 'registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest'
CANONICAL_RELATIVE = Path('runs/rl-fp32-is-canonical-20260917')
CANONICAL_EXPERIMENT_SHA256 = 'abb5f3fbd70115a35ceb165ccda054b2e0f44ab1013219aa8af640a63fe15073'
CANONICAL_ARM = 'lam4'
SOURCES = {'random_direction': 'randdir', 'random_band': 'randband',
           'shuffle': 'shuffle', 'norm_product': 'normprod'}
EXPECTED_DIFFERENCES = ('credit_source', 'output_dir')
write_json = corrected.write_json


def canonical_suite(project_root=None):
    """The canonical suite lives in the project root, never inside a frozen source snapshot."""
    return Path(ROOT if project_root is None else project_root).resolve() / CANONICAL_RELATIVE


def arm_spec(source):
    if source not in SOURCES:
        raise ValueError(f'Unknown random credit source: {source}')
    return {**corrected.ARMS[CANONICAL_ARM], 'credit_source': source}


def canonical_manifest(suite):
    path = Path(suite) / 'experiment.json'
    if file_hash(path) != CANONICAL_EXPERIMENT_SHA256:
        raise ValueError('Canonical experiment manifest changed')
    manifest = json.loads(path.read_text())
    if manifest['arms'].get(CANONICAL_ARM) != corrected.ARMS[CANONICAL_ARM]:
        raise ValueError('Canonical lambda4 arm definition differs')
    return manifest


def canonical_calibration(manifest, config, suite):
    path = Path(suite) / 'shared-calibration.json'
    if file_hash(path) != manifest['calibration']['shared_sha256']:
        raise ValueError('Canonical shared calibration changed')
    record = corrected.validate_calibration(json.loads(path.read_text()), config)
    return path, record


def canonical_resolved_config(suite):
    path = Path(suite) / CANONICAL_ARM / 'train/checkpoint-250/run_manifest.json'
    record = json.loads(path.read_text())
    if record.get('step') != 250 or not isinstance(record.get('resolved_config'), dict):
        raise ValueError('Canonical lambda4 final run manifest is incomplete')
    return path, record['resolved_config']


def training_command(manifest, arm, output, *, sigma0):
    if type(sigma0) not in (int, float) or isinstance(sigma0, bool) or not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError('Formal training requires the verified canonical sigma0')
    spec = manifest['arms'][arm]
    if spec.get('credit_source') not in SOURCES:
        raise ValueError('Ablation arm must declare a random credit source')
    return corrected.training_command(manifest, arm, output, sigma0=sigma0) + ['--credit-source', spec['credit_source']]


def resolved_from_command(command, output):
    from scripts.profile_vllm_full import build_trainer_config, parse_args
    return asdict(build_trainer_config(parse_args(command[2:]), output))


def verify_single_variable(command, output, canonical_resolved, source):
    """The resolved training configuration may differ from canonical lambda4 only in credit source and output."""
    resolved = resolved_from_command(command, output)
    canonical = {**canonical_resolved}
    canonical.setdefault('credit_source', 'rm_gradient')
    differences = sorted(key for key in set(resolved) | set(canonical) if resolved.get(key) != canonical.get(key))
    if tuple(differences) != EXPECTED_DIFFERENCES:
        raise ValueError(f'Ablation configuration differs from canonical lambda4 in {differences}; '
                         f'only {list(EXPECTED_DIFFERENCES)} may differ')
    if resolved['credit_source'] != source or canonical['credit_source'] != 'rm_gradient':
        raise ValueError('Ablation must replace rm_gradient credit by the requested random source')
    return {'status': 'passed', 'differences': differences, 'resolved_config': resolved}


def validation_identity(project_root=ROOT):
    identity = corrected.validation_identity(project_root)
    suite = canonical_suite(project_root)
    manifest = canonical_manifest(suite)
    expected = {key: value for key, value in manifest['common_config'].items() if key != 'max_rollouts'}
    if identity['config'] != expected:
        raise ValueError('Current common configuration differs from the canonical suite')
    calibration_path, calibration = canonical_calibration(manifest, corrected.common_config(project_root), suite)
    resolved_path, _ = canonical_resolved_config(suite)
    identity.update(experiment_kind='credit_ablation', canonical_arm=CANONICAL_ARM,
                    canonical_experiment_sha256=CANONICAL_EXPERIMENT_SHA256,
                    canonical_calibration_sha256=file_hash(calibration_path),
                    canonical_sigma0=calibration['sigma0'],
                    canonical_resolved_config_sha256=file_hash(resolved_path),
                    canonical_gpu_validation_sha256=file_hash(manifest['validation']['gpu']['path']))
    return identity


def submission_command(suite, arm, manifest, project_root=ROOT):
    template = json.loads((canonical_suite(project_root) / CANONICAL_ARM / 'submit_command.json').read_text())
    prefix = template[:template.index('--')]
    if prefix[:2] != ['rjob', 'submit'] or prefix[prefix.index('--image') + 1] != RUNTIME_IMAGE:
        raise ValueError('Canonical submission template is not the expected rjob envelope')
    suffix = hashlib.sha256(str(Path(suite).resolve()).encode()).hexdigest()[:10]
    command = list(prefix)
    for flag, value in (('--name', f'rl-abl-{SOURCES[manifest["arms"][arm]["credit_source"]]}-{suffix}'),
                        ('--gpu', '3'), ('--cpu', '48'), ('--memory', '600000')):
        command[command.index(flag) + 1] = value
    return command + ['--', '/usr/bin/env', 'OMP_NUM_THREADS=8', 'bash', str(Path(suite) / 'run_arm.sh'), arm]


def record_cpu_validation(output, project_root, pytest_log, pytest_summary):
    identity = validation_identity(project_root)
    log = Path(pytest_log)
    record = {'status': 'passed', **identity,
              'pytest': {'log': str(log.resolve()), 'log_sha256': file_hash(log), 'summary': pytest_summary},
              'recorded_at': datetime.now(timezone.utc).isoformat()}
    if 'failed' in pytest_summary or 'error' in pytest_summary or 'passed' not in pytest_summary:
        raise ValueError('CPU validation requires a passing pytest summary')
    write_json(output, record)
    return record


def prepare_suite(suite, source, cpu_validation, *, project_root=ROOT):
    project_root, suite = Path(project_root).resolve(), Path(suite).resolve()
    if source not in SOURCES:
        raise ValueError(f'Unknown random credit source: {source}')
    canonical_dir = canonical_suite(project_root)
    canonical = canonical_manifest(canonical_dir)
    config = corrected.common_config(project_root)
    if config != canonical['common_config']:
        raise ValueError('Current common configuration differs from the canonical suite')
    identity = validation_identity(project_root)
    cpu = corrected.validate_report(cpu_validation, identity)
    calibration_path, calibration = canonical_calibration(canonical, config, canonical_dir)
    resolved_path, canonical_resolved = canonical_resolved_config(canonical_dir)
    if suite.exists():
        raise FileExistsError(f'Use a fresh suite directory: {suite}')
    free = shutil.disk_usage(suite.parent if suite.parent.exists() else project_root).free / 2**30
    if free < 60:
        raise RuntimeError(f'Need 60 GiB free before preparing the ablation suite; have {free:.1f}')
    suite.mkdir(parents=True)
    source_dir = suite / 'source'
    source_dir.mkdir()
    for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs'):
        shutil.copytree(project_root / directory, source_dir / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.pytest_cache'))
    shutil.copy2(project_root / 'pyproject.toml', source_dir / 'pyproject.toml')
    for directory in ('models', 'datasets', '.vllm-extra'):
        (source_dir / directory).symlink_to(project_root / directory, target_is_directory=True)
    for name, expected in identity['source_sha256'].items():
        if file_hash(source_dir / name) != expected:
            raise ValueError(f'Source changed while freezing suite: {name}')
    snapshot = {str(path.relative_to(source_dir)): file_hash(path)
                for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs')
                for path in (source_dir / directory).rglob('*') if path.is_file()}
    snapshot['pyproject.toml'] = file_hash(source_dir / 'pyproject.toml')
    validation_target = suite / 'cpu-validation.json'
    shutil.copy2(cpu_validation, validation_target)
    arms = {source: arm_spec(source)}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'experiment_kind': 'credit_ablation',
                'suite': str(suite), 'project_root': str(project_root), 'source_snapshot': str(source_dir),
                'source_sha256': snapshot, 'validation_identity': identity,
                'validation': {'cpu': {'path': str(validation_target), 'sha256': file_hash(validation_target)}},
                'common_config': config, 'arms': arms, 'runtime': RUNTIME, 'runtime_image': RUNTIME_IMAGE,
                'reward_input_protocol': 'canonical_chat_v1', 'startup_max_unmapped_content_fraction': .25,
                'storage_budget': {'startup_min_free_gib': 60, 'free_gib_at_creation': free,
                                   'retention': 'final checkpoint; step0/current/final adapters'},
                'calibration': {'mode': 'inherited_canonical_shared_calibration', 'sigma0': calibration['sigma0'],
                                'source_path': str(calibration_path), 'source_sha256': file_hash(calibration_path)},
                'canonical': {'suite': str(canonical_dir), 'experiment_sha256': CANONICAL_EXPERIMENT_SHA256,
                              'arm': CANONICAL_ARM,
                              'command_sha256': file_hash(canonical_dir / CANONICAL_ARM / 'command.json'),
                              'resolved_config_path': str(resolved_path), 'resolved_config_sha256': file_hash(resolved_path),
                              'gpu_validation': {'path': canonical['validation']['gpu']['path'],
                                                 'sha256': file_hash(canonical['validation']['gpu']['path']),
                                                 'note': 'HF/vLLM probability, adapter-update and capacity gates of the '
                                                         'identical actor/RM/SFT/runtime; this ablation changes only '
                                                         'the credit source and is gated by its first two formal rollouts'}},
                'initialization_weights_sha256': identity['initialization_weights_sha256'],
                'dataset_sha256': identity['dataset_sha256']}
    command = training_command(manifest, source, suite / source / 'train', sigma0=calibration['sigma0'])
    manifest['single_variable_check'] = verify_single_variable(command, suite / source / 'train',
                                                               canonical_resolved, source)
    write_json(suite / 'shared-calibration.json', {**calibration, '_verification': {
        'identity': identity, 'source_calibration_sha256': file_hash(calibration_path),
        'inherited_from': str(calibration_path)}})
    manifest['calibration']['shared_sha256'] = file_hash(suite / 'shared-calibration.json')
    write_json(suite / 'experiment.json', manifest)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?arm required}"
case "$ARM" in ARM_CASES) ;; *) exit 2 ;; esac
export PYTHONPATH="$SUITE/source:$SUITE/source/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRESENCE_PENALTY=0.0 VLLM_WORKER_MULTIPROC_METHOD=spawn
exec > >(tee -a "$SUITE/$ARM/job.log") 2>&1
trap 'task_rc=$?; printf "%s\\n" "$task_rc" > "$SUITE/$ARM/exit_code"' EXIT
python3 "$SUITE/source/scripts/ablation_random_credit_launcher.py" run-arm --suite "$SUITE" --arm "$ARM"
'''.replace('ARM_CASES', '|'.join(arms))
    atomic_text(suite / 'run_arm.sh', shell)
    names = {}
    for arm in arms:
        (suite / arm).mkdir()
        command = submission_command(suite, arm, manifest, project_root)
        write_json(suite / arm / 'submit_command.json', command)
        names[arm] = command[command.index('--name') + 1]
    write_json(suite / 'submission-plan.json', {'status': 'prepared_not_submitted', 'names': names,
        'experiment_sha256': file_hash(suite / 'experiment.json'),
        'run_script_sha256': file_hash(suite / 'run_arm.sh'),
        'submit_command_sha256': {arm: file_hash(suite / arm / 'submit_command.json') for arm in arms}})
    print(f'PREPARED_NOT_SUBMITTED suite={suite} source={source} sigma0={calibration["sigma0"]}', flush=True)
    return manifest


def validate_completion(train_dir, manifest):
    import torch
    from safetensors import safe_open
    train_dir = Path(train_dir)
    summary = json.loads((train_dir / 'profile_summary.json').read_text())
    rows = summary.get('rollouts', [])
    if [row.get('rollout') for row in rows] != list(range(1, 251)):
        raise ValueError('Formal run did not complete all 250 rollouts')
    for row in rows:
        metric_keys = ('loss',) if row.get('skipped_rollout', False) and 'grad_norm' not in row else ('loss', 'grad_norm')
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in metric_keys):
            raise ValueError('Final rollout metrics are not finite')
        expected_updates = 0 if row.get('skipped_rollout', False) else 1
        if type(row.get('optimizer_steps')) is not int or row['optimizer_steps'] != expected_updates:
            raise ValueError('Formal rollout optimizer step count differs from the canonical policy')
    checkpoint = train_dir / 'checkpoint-250'
    run = json.loads((checkpoint / 'run_manifest.json').read_text())
    if run.get('step') != 250 or run.get('resolved_config', {}).get('credit_source') not in SOURCES:
        raise ValueError('Final checkpoint manifest does not describe the step-250 ablation')
    for name in ('adapter_model.safetensors', 'ref/adapter_model.safetensors', 'trainer_state.pt'):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Final checkpoint is incomplete: {name}')
    initial = Path(manifest['common_config']['init_adapter']) / 'adapter_model.safetensors'
    changed = 0
    with safe_open(initial, framework='pt', device='cpu') as original, \
            safe_open(checkpoint / 'adapter_model.safetensors', framework='pt', device='cpu') as trained, \
            safe_open(checkpoint / 'ref/adapter_model.safetensors', framework='pt', device='cpu') as reference:
        keys = set(original.keys())
        if not keys or keys != set(trained.keys()) or keys != set(reference.keys()):
            raise ValueError('Final adapter/reference tensor keys differ from SFT')
        for key in keys:
            before, after, frozen = original.get_tensor(key), trained.get_tensor(key), reference.get_tensor(key)
            if not torch.equal(before, frozen):
                raise ValueError(f'Final frozen reference adapter changed: {key}')
            if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
                raise ValueError(f'Final trained adapter shape/dtype/finiteness differs: {key}')
            changed += not torch.equal(before, after)
    if not changed:
        raise ValueError('Final trained adapter has not changed from SFT')
    return {'rollouts': 250, 'optimizer_steps': sum(row.get('optimizer_steps', 0) for row in rows),
            'skipped_rollouts': sum(bool(row.get('skipped_rollout')) for row in rows),
            'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': file_hash(checkpoint / 'run_manifest.json'),
            'adapter_sha256': file_hash(checkpoint / 'adapter_model.safetensors'),
            'changed_tensors': changed, 'reference_unchanged': True}


def run_arm(suite, arm):
    suite = Path(suite).resolve()
    manifest_path = suite / 'experiment.json'
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads((suite / 'submission-plan.json').read_text())
    if file_hash(manifest_path) != plan['experiment_sha256'] or file_hash(suite / 'run_arm.sh') != plan['run_script_sha256']:
        raise ValueError('Prepared experiment manifest or run script changed')
    if arm not in manifest['arms'] or manifest['arms'][arm] != arm_spec(arm):
        raise ValueError('Unknown or changed ablation arm')
    out = suite / arm
    try:
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
        if identity != manifest['validation_identity'] or manifest['common_config'] != corrected.common_config(manifest['project_root']):
            raise ValueError('Prepared inputs or configuration differ from validation')
        record = manifest['validation']['cpu']
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('CPU validation record changed')
        corrected.validate_report(record['path'], identity)
        if file_hash(suite / 'shared-calibration.json') != manifest['calibration']['shared_sha256']:
            raise ValueError('Inherited shared calibration changed')
        calibration = corrected.validate_calibration(json.loads((suite / 'shared-calibration.json').read_text()),
                                                     manifest['common_config'])
        if calibration['sigma0'] != manifest['calibration']['sigma0']:
            raise ValueError('Inherited sigma0 differs from the prepared manifest')
        free = shutil.disk_usage(suite).free / 2**30
        if free < manifest['storage_budget']['startup_min_free_gib']:
            raise RuntimeError(f'Insufficient free storage for suite startup: {free:.1f} GiB')
        import torch
        if torch.cuda.device_count() != 3 or any(torch.cuda.get_device_properties(i).total_memory < 120 * 2**30 for i in range(3)):
            raise RuntimeError('Formal ablation requires three H200-class GPUs')
        versions = {package: importlib.metadata.version(package) for package in manifest['runtime']}
        if versions != manifest['runtime']:
            raise RuntimeError(f'Runtime differs from validated software: {versions}')
        if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
            raise RuntimeError('vLLM multiprocessing must use spawn')
        write_json(out / 'runtime.json', {'software': versions, 'gpus': [torch.cuda.get_device_name(i) for i in range(3)],
            'free_gib': free, 'experiment_sha256': file_hash(manifest_path), 'source_snapshot': str(source)})
        command = training_command(manifest, arm, out / 'train', sigma0=calibration['sigma0'])
        _, canonical_resolved = canonical_resolved_config(canonical_suite(manifest['project_root']))
        verify_single_variable(command, out / 'train', canonical_resolved, arm)
        write_json(out / 'command.json', command)
        print('TRAIN_COMMAND ' + json.dumps(command), flush=True)
        atomic_text(out / 'stage', 'training\n')
        started = time.monotonic()
        corrected.run_training_process(command, manifest['project_root'], out / 'train', manifest, status_dir=out)
        completed = validate_completion(out / 'train', manifest)
        if validation_identity(manifest['project_root']) != identity:
            raise ValueError('Inputs or source changed during training')
        write_json(out / 'completion.json', {**completed, 'status': 'complete', 'wall_seconds': time.monotonic() - started,
            'sigma0': calibration['sigma0'], 'experiment_sha256': file_hash(manifest_path)})
        atomic_text(out / 'stage', 'complete\n')
    except BaseException as error:
        write_json(out / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
            'traceback': traceback.format_exc(), 'failed_at': datetime.now(timezone.utc).isoformat()})
        atomic_text(out / 'stage', 'failed\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cpu = commands.add_parser('record-cpu-validation')
    cpu.add_argument('--output', type=Path, required=True)
    cpu.add_argument('--pytest-log', type=Path, required=True)
    cpu.add_argument('--pytest-summary', required=True)
    cpu.add_argument('--project-root', type=Path, default=ROOT)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--suite', type=Path, required=True)
    prepare.add_argument('--credit-source', choices=sorted(SOURCES), required=True)
    prepare.add_argument('--cpu-validation', type=Path, required=True)
    prepare.add_argument('--project-root', type=Path, default=ROOT)
    run = commands.add_parser('run-arm')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--arm', choices=sorted(SOURCES), required=True)
    args = parser.parse_args(argv)
    if args.command == 'record-cpu-validation':
        record_cpu_validation(args.output, args.project_root, args.pytest_log, args.pytest_summary)
    elif args.command == 'prepare':
        prepare_suite(args.suite, args.credit_source, args.cpu_validation, project_root=args.project_root)
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
