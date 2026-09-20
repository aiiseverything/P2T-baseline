#!/usr/bin/env python3
"""Prepare and run the explicitly requested fresh-LoRA, final-only RL experiment.

Preparation is CPU-only. Each family's GRPO allocation runs an independent GPU
gate first; formal jobs never inherit gate weights or sampler state. Submission
is a separate action and never occurs implicitly while preparing or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import corrected_rl_launcher as canonical
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint

SHARED = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM')
RUNTIME = canonical.RUNTIME
IMAGE = 'registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest'
ARMS = {name: canonical.ARMS[name] for name in ('grpo', 'lam4')}
BASELINES = {'qwen': SHARED / 'runs/rl-fp32-is-canonical-20260917/experiment.json',
             'llama': Path('/data/VPO-RM/runs/llama31-rl-canonical-20260918-v3/experiment.json')}
PROMPT_SHA = '9c685ae9a4a652b8215ae50988614dd8723a6293222022e5381ab274f5f4f283'
check_initial_rollouts = canonical.check_initial_rollouts
write_json = canonical.write_json


def baseline_config(family):
    if family not in BASELINES:
        raise ValueError('Unknown family')
    return dict(json.loads(BASELINES[family].read_text())['common_config'])


def build_config(family):
    config = baseline_config(family)
    config.update(init_adapter='', checkpoint_interval=config['max_rollouts'] + 1)
    validate_config(config)
    return config


def validate_config(config):
    if config.get('init_adapter') != '':
        raise ValueError('Fresh initialization requires an explicit empty init_adapter')
    if (config.get('max_rollouts') != 250 or config.get('checkpoint_interval') != 251
            or config.get('keep_adapters_every') != 250):
        raise ValueError('Final-only storage requires 250 rollouts, interval251, retention250')
    if config.get('kl_reference') != 'init' or config.get('policy_head_dtype') != 'float32':
        raise ValueError('Frozen initial-base reference and FP32 head are required')
    if any(value is None for value in config.values()):
        raise ValueError('Configuration must not serialize None as a model/CLI value')


def read_manifest(family_dir):
    family_dir = Path(family_dir)
    path = family_dir / 'experiment.json'
    if file_hash(path) != (family_dir / 'experiment.sha256').read_text().strip():
        raise ValueError('Frozen experiment manifest changed')
    manifest = json.loads(path.read_text())
    validate_config(manifest['common_config'])
    if manifest.get('arms') != ARMS:
        raise ValueError('Only GRPO and VPO lambda4 are authorized in this suite')
    return manifest


def common_config(project_root):
    return dict(read_manifest(project_root)['common_config'])


def source_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): file_hash(path)
            for directory in ('scripts', 'vpo_rm')
            for path in sorted((root / directory).rglob('*.py'))
            if '__pycache__' not in path.parts}


def validation_identity(project_root):
    manifest = read_manifest(project_root)
    config = manifest['common_config']
    if source_hashes(ROOT) != manifest['source_sha256']:
        raise ValueError('Frozen training source differs from prepared snapshot')
    current = {role: fingerprint(config[key], full_weights=False)
               for role, key in (('actor', 'model'), ('reward', 'rm'))}
    if current != manifest['input_fingerprints']:
        raise ValueError('Model assets changed since preparation')
    if file_hash(config['dataset_path']) != manifest['dataset_sha256']:
        raise ValueError('Training dataset changed')
    for name, expected in manifest['benchmark_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Benchmark exclusion inputs changed')
    return {'experiment_family': manifest['family'],
            'initialization': 'fresh_lora_A_random_B_zero',
            'config': config, 'input_fingerprints': current,
            'dataset_sha256': manifest['dataset_sha256'],
            'source_sha256': manifest['source_sha256'],
            'reward_input_protocol': 'canonical_chat_v1',
            'ordered_formal_prompts_sha256': manifest['cpu_protocol']['formal_prompt_sha256']}


def training_command(manifest, arm, output, *, sigma0=None):
    validate_config(manifest['common_config'])
    if type(sigma0) not in (int, float) or not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError('Formal training requires a verified fresh shared sigma0')
    if arm not in ARMS:
        raise ValueError('Unknown direct RL arm')
    return canonical.training_command(manifest, arm, output, sigma0=sigma0)


def submit_command(suite, family, arm):
    if family not in BASELINES or arm not in ARMS:
        raise ValueError('Unknown family/arm')
    command = json.loads(Path('/data/VPO-RM/runs/llama31-rl-canonical-20260918-v3/lam4/submit_command.json').read_text())
    command = command[:command.index('--')]
    suffix = hashlib.sha256(str(Path(suite).resolve()).encode()).hexdigest()[:8]
    for flag, value in {'--name': f'direct-{family}-{arm}-{suffix}', '--gpu': '3',
                        '--cpu': '48', '--memory': '600000', '--image': IMAGE}.items():
        command[command.index(flag) + 1] = value
    return command + ['--', 'bash', str(Path(suite) / 'run_arm.sh'), family, arm]


def cpu_protocol(config):
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from vpo_rm.trainer import VPOTrainer, load_prompt_dataset
    from vpo_rm.token_policy import load_actor_tokenizer, configure_model_padding, get_stop_token_ids
    actor = load_actor_tokenizer(config['model'], '')
    reward = AutoTokenizer.from_pretrained(config['rm'], local_files_only=True)
    configure_model_padding(reward, fallback_token=actor.pad_token)
    if not actor.chat_template or not reward.chat_template:
        raise ValueError('This experiment requires actual model chat templates')
    example = VPOTrainer._render_chat_prompt(actor, 'Say hello.')
    if example == 'Say hello.' or 'assistant' not in example:
        raise ValueError('Chat rendering fell back to plain text')
    prompts, _, split = load_prompt_dataset('HuggingFaceH4/ultrafeedback_binarized',
        dataset_path=config['dataset_path'], exclude_benchmarks=True)
    # The filter only needs tokenizer/config/logging, not model weights.
    dummy = SimpleNamespace(actor_tokenizer=actor, reward_tokenizer=reward,
        cfg=SimpleNamespace(max_prompt_tokens=2048), _render_chat_prompt=VPOTrainer._render_chat_prompt,
        _log=lambda entry: None)
    filtered = VPOTrainer.filter_prompts(dummy, prompts)
    encoded = json.dumps(filtered[:2000], ensure_ascii=False, separators=(',', ':')).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != PROMPT_SHA or len(filtered[:2000]) != 2000:
        raise ValueError('Actual filtered training prompt order differs from canonical experiment')
    return {'status': 'passed', 'formal_prompt_count': 2000, 'formal_prompt_sha256': digest,
        'filtered_train_prompts': len(filtered), 'split': split,
        'actor_template_sha256': hashlib.sha256(actor.chat_template.encode()).hexdigest(),
        'reward_template_sha256': hashlib.sha256(reward.chat_template.encode()).hexdigest(),
        'actor_stop_ids': list(get_stop_token_ids(actor)), 'actor_pad_id': actor.pad_token_id,
        'reward_stop_ids': list(get_stop_token_ids(reward)), 'reward_pad_id': reward.pad_token_id,
        'example_prompt_ids': actor(example, add_special_tokens=False)['input_ids']}


def prepare(suite, outputs):
    suite, outputs = Path(suite).resolve(), Path(outputs).resolve()
    if suite.exists():
        raise FileExistsError('Use a fresh suite; no overwrites')
    if shutil.disk_usage('/data').free < 40 * 2**30:
        raise RuntimeError('Need 40 GiB free for four final checkpoints, transient files and concurrent download')
    suite.mkdir(parents=True)
    source = suite / 'source'
    source.mkdir()
    for directory in ('scripts', 'vpo_rm', 'tests', 'configs'):
        shutil.copytree(ROOT / directory, source / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Shared datasets are read-only inputs; individual required files are hashed.
    (source / 'datasets').symlink_to(SHARED / 'datasets', target_is_directory=True)
    (source / '.vllm-extra').symlink_to(SHARED / '.vllm-extra', target_is_directory=True)
    sources = source_hashes(source)
    from vpo_rm.data import DEFAULT_BENCHMARK_PATHS, ROOT as DATA_ROOT
    benchmarks = {str(path.relative_to(DATA_ROOT)): file_hash(path)
                  for path in DEFAULT_BENCHMARK_PATHS.values()}
    shell = '''#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:?family required}"
ARM="${2:?arm required}"
export PYTHONPATH="$SUITE/source:$SUITE/source/.vllm-extra"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PRESENCE_PENALTY=0.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec > >(tee -a "$SUITE/$FAMILY/$ARM/job.log") 2>&1
exec python3 "$SUITE/source/scripts/direct_rl_launcher.py" run-arm --family-dir "$SUITE/$FAMILY" --arm "$ARM"
'''
    atomic_text(suite / 'run_arm.sh', shell)
    manifests = {}
    for family in BASELINES:
        family_dir = suite / family
        family_dir.mkdir()
        config = build_config(family)
        assets = {role: fingerprint(config[key], full_weights=False)
                  for role, key in (('actor', 'model'), ('reward', 'rm'))}
        baseline = json.loads(BASELINES[family].read_text())
        prior = baseline['validation_identity']['input_fingerprints']
        if any(assets[role] != prior[role] for role in assets):
            raise ValueError(f'{family} assets differ from the successfully trained canonical suite')
        protocol = cpu_protocol(config)
        manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'family': family,
            'suite': str(suite), 'project_root': str(family_dir), 'output_root': str(outputs / family),
            'source_snapshot': str(source), 'source_sha256': sources, 'common_config': config,
            'arms': ARMS, 'runtime': RUNTIME, 'runtime_image': IMAGE,
            'reward_input_protocol': 'canonical_chat_v1', 'startup_max_unmapped_content_fraction': .25,
            'input_fingerprints': assets, 'dataset_sha256': file_hash(config['dataset_path']),
            'benchmark_sha256': benchmarks, 'cpu_protocol': protocol,
            'baseline': {'path': str(BASELINES[family]), 'sha256': file_hash(BASELINES[family])},
            'initialization': 'fresh_lora_A_random_B_zero', 'kl_reference': 'frozen_raw_actor',
            'storage': {'checkpoint_interval': 251, 'keep_adapters_every': 250,
                'final_checkpoint': 250, 'remove_transient_exports_after_verified_completion': True,
                'remove_preflight_weight_artifacts_after_verified_gate': True},
            'calibration_coverage_minimum': {'valid_responses': 512, 'eligible_groups': 64}}
        write_json(family_dir / 'experiment.json', manifest)
        atomic_text(family_dir / 'experiment.sha256', file_hash(family_dir / 'experiment.json') + '\n')
        write_json(family_dir / 'cpu-protocol.json', protocol)
        for arm in ARMS:
            (family_dir / arm).mkdir()
            write_json(family_dir / arm / 'submit_command.json', submit_command(suite, family, arm))
        manifests[family] = file_hash(family_dir / 'experiment.json')
    write_json(suite / 'submission-plan.json', {'status': 'prepared_not_submitted',
        'families': manifests, 'gpu_per_job': 3, 'max_jobs': 4,
        'order': 'GRPO allocations run independent family gates; submit each VPO only after its gate passes'})
    return manifests


def validate_gate(family_dir):
    manifest = read_manifest(family_dir)
    identity = validation_identity(family_dir)
    gate = json.loads((Path(family_dir) / 'shared-gate.json').read_text())
    if gate.get('status') != 'passed' or gate.get('identity') != identity:
        raise ValueError('Fresh GPU gate identity differs')
    if gate.get('experiment_sha256') != file_hash(Path(family_dir) / 'experiment.json'):
        raise ValueError('GPU gate was made for another manifest')
    for item in gate['evidence'].values():
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('GPU gate evidence changed')
    record = json.loads(Path(gate['evidence']['calibration']['path']).read_text())
    return canonical.validate_calibration(record, manifest['common_config'])


def run_gate(family_dir, manifest):
    from scripts.direct_rl_storage import cleanup_preflight_artifacts
    family_dir = Path(family_dir)
    output = Path(manifest['output_root'])
    preflight = output / 'gpu-preflight'
    if preflight.exists() or (family_dir / 'gate-started.json').exists():
        raise FileExistsError('Prior gate exists; never silently retry GPU work')
    write_json(family_dir / 'gate-started.json', {'status': 'started'})
    command = [sys.executable, str(ROOT / 'scripts/direct_rl_preflight.py'), '--output-dir',
        str(preflight), '--project-root', str(family_dir), '--runtime-image', IMAGE]
    write_json(family_dir / 'gpu-preflight-command.json', command)
    with (family_dir / 'gpu-preflight.log').open('w') as log:
        subprocess.run(command, cwd=family_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    gpu_path = preflight / 'gpu-validation.json'
    gpu = json.loads(gpu_path.read_text())
    identity = validation_identity(family_dir)
    if gpu.get('status') != 'passed' or any(gpu.get(key) != value for key, value in identity.items()):
        raise ValueError('GPU preflight failed or belongs to different inputs')
    if set(gpu.get('arms', {})) != set(ARMS):
        raise ValueError('Both direct-RL methods must pass the GPU gate')
    for arm, report in gpu['arms'].items():
        if (report.get('status') != 'passed' or report.get('rollouts') != 2
                or report.get('runtime', {}).get('versions') != RUNTIME):
            raise ValueError(f'Incomplete GPU gate: {arm}')
    if gpu['arms']['lam4'].get('capacity', {}).get('status') != 'passed':
        raise ValueError('VPO long-sequence capacity gate is required')
    if (gpu['arms']['grpo']['initial_adapters']['default_sha256'] !=
            gpu['arms']['lam4']['initial_adapters']['default_sha256']):
        raise ValueError('The two methods did not start from the same fresh LoRA')
    calibration_path = Path(gpu['calibration']['path'])
    if file_hash(calibration_path) != gpu['calibration']['sha256']:
        raise ValueError('Calibration evidence changed')
    calibration = canonical.validate_calibration(json.loads(calibration_path.read_text()), manifest['common_config'])
    valid = [row for row in calibration['responses'] if row.get('valid')]
    groups = {}
    for row in valid:
        group = row['group']; groups[group] = groups.get(group, 0) + 1
    coverage = {'valid_responses': len(valid), 'eligible_groups': sum(n >= 2 for n in groups.values())}
    if any(coverage[k] < v for k, v in manifest['calibration_coverage_minimum'].items()):
        raise ValueError('Insufficient fresh calibration coverage: ' + json.dumps(coverage))
    write_json(family_dir / 'calibration-coverage.json', coverage)
    shared = family_dir / 'shared-calibration.json'
    write_json(shared, calibration)
    cleanup = cleanup_preflight_artifacts(preflight, suite_dir=output)
    write_json(family_dir / 'preflight-cleanup.json', cleanup)
    write_json(family_dir / 'shared-gate.json', {'status': 'passed', 'identity': identity,
        'experiment_sha256': file_hash(family_dir / 'experiment.json'),
        'evidence': {'gpu': {'path': str(gpu_path), 'sha256': file_hash(gpu_path)},
                     'calibration': {'path': str(shared), 'sha256': file_hash(shared)}}})
    return validate_gate(family_dir)


def validate_completion(train_dir, manifest, arm):
    import torch
    from safetensors import safe_open
    train_dir = Path(train_dir)
    summary = json.loads((train_dir / 'profile_summary.json').read_text())
    rows = summary.get('rollouts', [])
    if [r.get('rollout') for r in rows] != list(range(1, 251)):
        raise ValueError('Training did not finish all 250 rollouts')
    # The trainer writes kept prompts after its canonical retry/drop policy.
    # The full scheduled order is bound by cpu_protocol and the frozen profile;
    # these artifacts may legitimately contain fewer than eight prompts.
    prompts = []
    for step, row in enumerate(rows, start=1):
        current = json.loads((train_dir / f'rollout-{step}-prompts.json').read_text())
        if (not isinstance(current, list) or len(current) > 8
                or any(not isinstance(item, str) for item in current)):
            raise ValueError('Formal rollout prompt evidence is incomplete')
        expected_counts = {'input_prompt_groups': 8, 'kept_prompt_groups': len(current),
                           'skipped_groups': 8 - len(current), 'reward_count': len(current) * 8}
        if any(key in row and (type(row[key]) is not int or row[key] != count)
               for key, count in expected_counts.items()):
            raise ValueError('Kept prompt/group/response counts differ from rollout metrics')
        if bool(row.get('skipped_rollout', False)) != (not current):
            raise ValueError('Kept prompt evidence differs from skipped-rollout status')
        prompts.extend(current)
    prompt_digest = hashlib.sha256(json.dumps(prompts, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()
    for row in rows:
        expected = 0 if row.get('skipped_rollout') else 1
        if row.get('optimizer_steps') != expected:
            raise ValueError('Unexpected optimizer update count')
        for key in ('loss', 'grad_norm'):
            if key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key])):
                raise ValueError('Nonfinite final training metrics')
    path = train_dir / 'checkpoint-250'
    record = json.loads((path / 'run_manifest.json').read_text())
    config = record['resolved_config']
    expected = manifest['common_config']
    if (record.get('step') != 250 or config.get('init_adapter') != ''
            or config.get('model_name') != expected['model'] or config.get('reward_model_name') != expected['rm']
            or config.get('kl_reference') != 'init' or config.get('method') != ARMS[arm]['method']
            or (path / 'ref').exists()):
        raise ValueError('Final checkpoint is not the requested fresh-LoRA experiment')
    nonzero_b = 0
    with safe_open(path / 'adapter_model.safetensors', framework='pt', device='cpu') as weights:
        for key in weights.keys():
            value = weights.get_tensor(key)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite final adapter')
            if '.lora_B.' in key and torch.count_nonzero(value):
                nonzero_b += 1
    if nonzero_b == 0:
        raise ValueError('No learned LoRA B matrices in final checkpoint')
    return {'rollouts': 250, 'optimizer_steps': sum(r['optimizer_steps'] for r in rows),
            'actual_kept_prompt_sha256': prompt_digest, 'kept_prompt_count': len(prompts),
            'nonzero_b_tensors': nonzero_b, 'final_adapter_sha256': file_hash(path / 'adapter_model.safetensors')}


def verify_formal_initialization(train_dir, manifest):
    from safetensors.torch import load_file
    from scripts.direct_rl_preflight import tensor_state_digest
    initial = load_file(str(Path(train_dir) / 'vllm-adapters/step-0/adapter_model.safetensors'))
    gate = json.loads((Path(manifest['output_root']) / 'gpu-preflight/gpu-validation.json').read_text())
    expected = gate['arms']['grpo']['initial_adapters']['default_sha256']
    if tensor_state_digest(initial) != expected:
        raise ValueError('Formal initial LoRA differs from the verified fresh initial weights')
    return expected


def supervise_formal(command, family_dir, train_dir, manifest, status):
    """Abort immediately on a bad first-two-rollout gate; never wait 250 steps."""
    process = subprocess.Popen(command, cwd=family_dir)
    checked = None
    try:
        while True:
            if checked is None:
                checked = check_initial_rollouts(train_dir, manifest)
                if checked is not None:
                    expected = verify_formal_initialization(train_dir, manifest)
                    checked['initial_adapter_sha256'] = expected
                    write_json(status / 'startup_validation.json', checked)
            code = process.poll()
            if code is not None:
                if code != 0:
                    raise subprocess.CalledProcessError(code, command)
                if checked is None:
                    raise RuntimeError('Formal training exited without two verified updates')
                return checked
            time.sleep(1)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=75)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)


def run_arm(family_dir, arm):
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    from scripts.direct_rl_preflight import runtime_report
    family_dir = Path(family_dir).resolve()
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    status = family_dir / arm
    with (status / 'arm.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_manifest(family_dir)
        train_dir = Path(manifest['output_root']) / arm / 'train'
        try:
            if train_dir.exists() or (status / 'command.json').exists():
                raise FileExistsError('Training already started; no automatic rerun')
            identity = validation_identity(family_dir)
            runtime = runtime_report(IMAGE)
            if (runtime['versions'] != RUNTIME or len(runtime['gpus']) != 3
                    or any(g['memory_bytes'] < 120 * 2**30 for g in runtime['gpus'])):
                raise ValueError('Formal job runtime/GPU allocation differs from validated setup')
            write_json(status / 'runtime.json', runtime)
            Path(manifest['output_root']).mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(manifest['output_root']).free < 30 * 2**30:
                raise RuntimeError('Need 30 GiB free for concurrent experiment outputs')
            atomic_text(status / 'stage', 'gpu_gate\n')
            if arm == 'grpo' and not (family_dir / 'shared-gate.json').exists():
                calibration = run_gate(family_dir, manifest)
            else:
                calibration = validate_gate(family_dir)
            command = training_command(manifest, arm, train_dir, sigma0=calibration['sigma0'])
            write_json(status / 'command.json', command)
            atomic_text(status / 'stage', 'training\n')
            # Separate process and separate vLLM instance: no pilot weights/RNG inherited.
            supervise_formal(command, family_dir, train_dir, manifest, status)
            completed = validate_completion(train_dir, manifest, arm)
            if validation_identity(family_dir) != identity:
                raise ValueError('Source/assets changed during training')
            cleanup = verify_and_cleanup_exports(train_dir, 250)
            write_json(status / 'completion.json', {'status': 'complete', **completed,
                'cleanup': cleanup, 'experiment_sha256': file_hash(family_dir / 'experiment.json'),
                'finished_at': datetime.now(timezone.utc).isoformat()})
            atomic_text(status / 'stage', 'complete\n')
        except BaseException as error:
            write_json(status / 'failure.json', {'status': 'failed', 'error': repr(error),
                'traceback': traceback.format_exc(), 'time': datetime.now(timezone.utc).isoformat()})
            atomic_text(status / 'stage', 'failed\n')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--outputs', required=True, type=Path)
    p = sub.add_parser('run-arm'); p.add_argument('--family-dir', required=True, type=Path)
    p.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        print(json.dumps(prepare(args.suite, args.outputs)))
    else:
        run_arm(args.family_dir, args.arm)


if __name__ == '__main__':
    main()
