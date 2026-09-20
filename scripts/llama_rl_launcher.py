#!/usr/bin/env python3
"""Freeze and run the audited Llama SFT-initialized RL experiment; never submit jobs.

The GRPO allocation first runs the shared GPU gate. Other allocations require
that gate's committed, hash-bound success record before loading any models.

Two actor profiles share every training setting and differ only in audited
inputs: ``instruct`` (Llama-3.1-8B-Instruct, EOT-terminated SFT, four arms) and
``base`` (pretrained Llama-3.1-8B, <|end_of_text|>-terminated SFT, GRPO and
VPO lambda4). The profile is recorded in each suite manifest.
"""
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
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import corrected_rl_launcher as corrected
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint

ARMS = corrected.ARMS
RUNTIME = corrected.RUNTIME
RUNTIME_IMAGE = 'registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest'
SHARED_PROJECT = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM')
BASELINE_MANIFEST = SHARED_PROJECT / 'runs/rl-fp32-is-canonical-20260917/experiment.json'
BASELINE_SHA256 = 'abb5f3fbd70115a35ceb165ccda054b2e0f44ab1013219aa8af640a63fe15073'
EXTRA_PACKAGES = SHARED_PROJECT / '.vllm-extra'
PROTECTED_SFT_SHA256 = '52a68bd14ecea9bf660e0bb04a78cb68e6d2c263eedcfb9c40c13acba01453cc'
DATASET_SHA256 = '0f951ca4502001d31f3e4c70716ae51d20e4ce4f847d12b6a6695a40d4d353a8'
STOP_IDS = [128001, 128008, 128009]
PROFILES = {
    'instruct': {
        'actor_dir': 'Llama-3.1-8B-Instruct',
        'init_adapter_dir': 'sft-llama31-8b-instruct-clean2k5e2-20260917',
        'protected_sft_sha256': PROTECTED_SFT_SHA256,
        'actor_response_eos_id': 128009,
        'arms': ('grpo', 'lam2', 'lam4', 'lam8'),
        'capacity_arm': 'lam8',
        'job_prefix': 'llama-rl',
        'quality_rule': 'strict',
        'sft_evidence_dir': 'runs/llama31-sft-aligned-20260917-attempt2',
        'asset_evidence': '.maintenance/llama-rl-20260918/assets_verified.json',
    },
    'base': {
        'actor_dir': 'Llama-3.1-8B',
        'init_adapter_dir': 'sft-llama31-8b-base-clean2k5e2-20260919',
        'protected_sft_sha256': '465d5f547dddf29269a7e98d66fae8a40e2e3d40dc6ca8a54d8e1937e0265096',
        'actor_response_eos_id': 128001,
        'arms': ('grpo', 'lam4'),
        'capacity_arm': 'lam4',
        'job_prefix': 'llama-base-rl',
        # Lightly SFT-tuned base answers verbosely ("...is Paris."); CPU greedy
        # replay of the canaries scored 5/8 strict and 7/8 final-word before launch.
        'quality_rule': 'final_word',
        'sft_evidence_dir': 'runs/llama31-base-sft-20260919',
        'asset_evidence': '.maintenance/llama-base-rl-20260919/assets_verified.json',
    },
}
ACTIVE_PROFILE = 'instruct'


def set_profile(name):
    """Select the audited actor profile; suites record it and re-select it on run."""
    global ACTIVE_PROFILE
    if name not in PROFILES:
        raise ValueError(f'Unknown Llama experiment profile: {name}')
    ACTIVE_PROFILE = name
    return PROFILES[name]


def profile():
    return PROFILES[ACTIVE_PROFILE]


def profile_arms():
    return {arm: ARMS[arm] for arm in profile()['arms']}
PROMPT_AUDIT = {
    'train_hash': 'ea85088a7d3777bad6f47d17452a06c20c2e00cf411860167cb2dc5d21a9b739',
    'validation_hash': '6473440de8839d9acc5bf7f6996bfd8aa247500ad5bb6429ddddfc9df4f6321c',
    'num_unique': 61097, 'validation_size': 2000,
    'filtered_train_prompts': 59061, 'dropped_train_prompts': 36,
    'formal_prompt_count': 2000,
    'ordered_formal_prompts_sha256': '9c685ae9a4a652b8215ae50988614dd8723a6293222022e5381ab274f5f4f283',
    'hash_serialization': 'UTF-8 JSON, ensure_ascii=False, separators=(comma,colon)',
    'same_formal_prompts_as_qwen': True,
}
training_command = corrected.training_command
check_initial_rollouts = corrected.check_initial_rollouts
run_training_process = corrected.run_training_process
validate_report = corrected.validate_report
write_json = corrected.write_json


def baseline_path():
    frozen = ROOT / 'configs/llama_rl_baseline.json'
    return frozen if frozen.is_file() else BASELINE_MANIFEST


def common_config(project_root):
    path = baseline_path()
    if file_hash(path) != BASELINE_SHA256:
        raise ValueError('Frozen Qwen baseline manifest changed')
    config = dict(json.loads(path.read_text())['common_config'])
    root = Path(project_root).resolve()
    models = root.parent / 'models'
    config.update(model=str(models / profile()['actor_dir']),
                  rm=str(models / 'Skywork-Reward-Llama-3.1-8B-v0.2'),
                  init_adapter=str(models / profile()['init_adapter_dir']),
                  dataset_path=str(root / 'datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet'))
    return config


def validate_model_contract(config):
    for key, role, architecture in (('model', 'actor', 'LlamaForCausalLM'),
                                     ('rm', 'reward', 'LlamaForSequenceClassification')):
        actual = json.loads((Path(config[key]) / 'config.json').read_text())
        if (actual.get('model_type') != 'llama' or actual.get('architectures') != [architecture]
                or actual.get('vocab_size') != 128256 or actual.get('tie_word_embeddings') is not False):
            raise ValueError(f'Wrong Llama {role} model architecture or vocabulary')
    adapter = json.loads((Path(config['init_adapter']) / 'adapter_config.json').read_text())
    base = Path(adapter.get('base_model_name_or_path', ''))
    if not base.is_absolute():
        base = ROOT / base
    expected = {'r': 64, 'lora_alpha': 128, 'lora_dropout': 0, 'bias': 'none', 'task_type': 'CAUSAL_LM'}
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if (base.resolve() != Path(config['model']).resolve()
            or any(adapter.get(key) != value for key, value in expected.items())
            or set(adapter.get('target_modules', [])) != targets):
        raise ValueError('Llama SFT adapter base or LoRA configuration differs')


def source_manifest():
    from scripts.profile_vllm_full import profile_source_manifest
    manifest = profile_source_manifest()
    manifest['source_sha256'].update({name: file_hash(ROOT / name) for name in
        ('scripts/llama_rl_launcher.py', 'scripts/check_llama_protocol.py')})
    return manifest


def sft_evidence(project_root, config):
    spec = profile()
    directory = Path(project_root).resolve().parent / spec['sft_evidence_dir']
    protected, eos = spec['protected_sft_sha256'], spec['actor_response_eos_id']
    if ACTIVE_PROFILE == 'instruct':
        paths = {name: directory / relative for name, relative in {
            'summary': 'summary.json', 'reload': 'job/final_reload.json',
            'completion': 'postcheck-recovery/completion_verified.json',
            'actor_integrity': 'job/actor_integrity.json'}.items()}
        records = {name: json.loads(path.read_text()) for name, path in paths.items()}
        complete = (records['completion'].get('status') == 'complete'
                    and records['completion'].get('postcheck_job_state') == 'Succeeded'
                    and records['completion'].get('summary_sha256') == file_hash(paths['summary'])
                    and records['completion'].get('final_reload_sha256') == file_hash(paths['reload']))
    else:
        paths = {name: directory / relative for name, relative in {
            'summary': 'summary.json', 'reload': 'job/final_reload.json',
            'completion': 'job/completion.json',
            'actor_integrity': 'job/actor_integrity.json'}.items()}
        records = {name: json.loads(path.read_text()) for name, path in paths.items()}
        complete = (records['completion'].get('rjob_final_state') == 'Succeeded'
                    and records['completion'].get('exit_code') == 0
                    and records['completion'].get('experiment_complete') is True
                    and records['completion'].get('final_reload') == 'passed'
                    and records['completion'].get('adapter_model_sha256') == protected
                    and records['summary'].get('expected_terminator') == eos
                    and records['reload'].get('response_eos_id') == eos)
    if (records['summary'].get('status') != 'complete'
            or records['summary'].get('optimizer_steps') != 157
            or records['summary'].get('adapter') != config['init_adapter']
            or records['reload'].get('status') != 'passed'
            or records['reload'].get('hashes', {}).get('adapter_model.safetensors') != protected
            or records['reload'].get('tokenizer', {}).get('response_eos_id') != eos
            or not complete
            or records['actor_integrity'].get('status') != 'passed'):
        raise ValueError('Completed and reloaded SFT evidence is missing or differs')
    reward_verification = Path(config['rm']) / 'DOWNLOAD_VERIFIED.json'
    actor_verification = Path(config['model']) / 'DOWNLOAD_MANIFEST.json'
    if (json.loads(reward_verification.read_text()).get('status') != 'complete_verified'
            or json.loads(actor_verification.read_text()).get('verified') is not True):
        raise ValueError('Actor or reward download verification is incomplete')
    paths.update(actor_download=actor_verification, reward_download=reward_verification)
    return {name: {'path': str(path), 'sha256': file_hash(path)} for name, path in paths.items()}


def validate_asset_evidence(path, config):
    """Bind the completed full-shard hash audit without rereading 31 GB per arm."""
    path = Path(path)
    record = json.loads(path.read_text())
    models = {Path(config[key]).name: Path(config[key]) for key in ('model', 'rm')}
    if record.get('status') != 'passed' or set(record.get('models', {})) != set(models):
        raise ValueError('Full asset verification is incomplete or names other models')
    for name, root in models.items():
        model = record['models'][name]
        if file_hash(model['manifest']) != model['manifest_sha256'] or not model.get('files'):
            raise ValueError('Verified asset download manifest changed')
        for entry in model['files']:
            current = root / entry['file']
            if Path(entry['file']).is_absolute() or '..' in Path(entry['file']).parts:
                raise ValueError('Invalid verified asset path')
            stat = current.stat()
            actual = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                      'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}
            if any(entry.get(key) != value for key, value in actual.items()):
                raise ValueError(f'Verified asset changed since full hashing: {current}')
            if current.suffix not in {'.safetensors', '.bin', '.pt'} and file_hash(current) != entry['sha256']:
                raise ValueError(f'Verified asset metadata changed: {current}')
    if not record.get('sft_hashes'):
        raise ValueError('Full asset verification lacks SFT hashes')
    for name, expected in record['sft_hashes'].items():
        current = (Path(config['model']) / 'config.json' if name == 'base_config.json'
                   else Path(config['init_adapter']) / name)
        if file_hash(current) != expected:
            raise ValueError(f'Protected SFT asset changed: {name}')
    return {'path': str(path.resolve()), 'sha256': file_hash(path),
            'base_weight_verification': 'full SHA256 audit bound to unchanged size/mtime/ctime/inode'}


def validation_identity(project_root=ROOT):
    # Check the protected adapter first, without ever loading model tensors.
    root = Path(project_root).resolve()
    init = root.parent / 'models' / profile()['init_adapter_dir']
    init_hash = file_hash(init / 'adapter_model.safetensors')
    if init_hash != profile()['protected_sft_sha256']:
        raise ValueError('Protected Llama SFT initialization weights changed')
    config = common_config(root)
    validate_model_contract(config)
    dataset_hash = file_hash(config['dataset_path'])
    if dataset_hash != DATASET_SHA256:
        raise ValueError('UltraFeedback dataset differs from frozen Qwen baseline')
    from vpo_rm.data import DEFAULT_BENCHMARK_PATHS, ROOT as DATA_ROOT
    return {**source_manifest(), 'experiment_family': 'llama',
            'experiment_profile': ACTIVE_PROFILE,
            'actor_response_eos_id': profile()['actor_response_eos_id'],
            'config': {key: value for key, value in config.items() if key != 'max_rollouts'},
            'baseline_manifest_sha256': BASELINE_SHA256,
            'initialization_weights_sha256': init_hash, 'dataset_sha256': dataset_hash,
            'input_fingerprints': {
                'actor': fingerprint(config['model'], full_weights=False),
                'reward': fingerprint(config['rm'], full_weights=False),
                'initialization': fingerprint(config['init_adapter'], full_weights=True)},
            'sft_evidence': sft_evidence(root, config),
            'full_asset_verification': validate_asset_evidence(
                root.parent / profile()['asset_evidence'], config),
            'benchmark_exclusion_sha256': {name: file_hash(root / path.relative_to(DATA_ROOT))
                for name, path in DEFAULT_BENCHMARK_PATHS.items()},
            'prompt_audit_expected': dict(PROMPT_AUDIT)}


def validate_protocol_report(record, config, *, mode, expected_identity=None):
    expected_paths = {'actor': config['model'], 'reward': config['rm'], 'init_adapter': config['init_adapter']}
    if (record.get('schema') != 'llama_protocol_v1' or record.get('status') != 'passed'
            or record.get('mode') != mode or record.get('paths') != expected_paths):
        raise ValueError('Llama protocol report failed or describes different mode/input paths')
    identity = record.get('identity', {})
    if set(identity) != set(expected_paths) or (expected_identity is not None and identity != expected_identity):
        raise ValueError('Llama protocol identity differs between CPU and GPU')
    if record.get('protocol', {}).get('actor_response_eos_id') != profile()['actor_response_eos_id']:
        raise ValueError('Llama protocol report audited a different actor response EOS')
    for role, files in identity.items():
        if not files:
            raise ValueError('Llama protocol input identity must not be empty')
        for name, expected in files.items():
            path = Path(expected_paths[role]) / name
            if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(path) != expected:
                raise ValueError(f'Llama protocol input changed: {role}/{name}')
    sources = record.get('source_sha256', {})
    if not sources:
        raise ValueError('Llama protocol report lacks source hashes')
    for name, expected in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(ROOT / name) != expected:
            raise ValueError(f'Llama protocol source changed: {name}')
    if mode == 'gpu':
        gpu, runtime = record.get('gpu'), record.get('runtime', {})
        if (not isinstance(gpu, dict) or gpu.get('status') != 'passed'
                or gpu.get('audit_protocol') != 'llama_reward_precision_audit_v2'
                or type(gpu.get('production_microbatch_responses')) is not int
                or gpu['production_microbatch_responses'] != 1):
            raise ValueError('GPU protocol lacks precision controls or the validated RM microbatch')
        finite = lambda value: type(value) in (int, float) and math.isfinite(value)
        for name, dtype, atol, invariance in (
                ('production_bf16', 'torch.bfloat16', .125, False),
                ('padding_fp32', 'torch.float32', .001, True)):
            phase = gpu.get(name)
            if (not isinstance(phase, dict) or phase.get('status') != 'passed'
                    or any(phase.get(key) is not True for key in
                           ('frozen_parameters', 'nonzero_input_gradient', 'terminal_eot_pooling'))
                    or phase.get('score_protocol') != 'raw_scalar_logit_no_sigmoid'
                    or phase.get('score_atol') != atol or phase.get('score_rtol') != 0
                    or phase.get('parameter_dtype') != dtype
                    or phase.get('attention_implementation') != 'sdpa'
                    or phase.get('batch_invariance_required') is not invariance
                    or phase.get('same_input_paths') != ['native', 'wrapper_no_grad', 'wrapper_input_gradients']
                    or phase.get('padding_sides') != ['left', 'right']
                    or phase.get('padding_gradient_max_abs') != 0):
                raise ValueError(f'GPU protocol {name} lacks raw-score, precision, padding, or gradient evidence')
            difference, gradient = phase.get('max_score_difference'), phase.get('mapped_gradient_norm_min')
            batch_difference, scores = phase.get('max_batch_score_difference'), phase.get('singleton_scores')
            if (not finite(difference) or not 0 <= difference <= atol
                    or not finite(batch_difference) or batch_difference < 0
                    or (invariance and batch_difference > atol)
                    or not finite(gradient) or gradient <= 0
                    or not isinstance(scores, list) or not scores or not all(map(finite, scores))):
                raise ValueError(f'GPU protocol {name} score or mapped gradient evidence is invalid')
            if invariance and (phase.get('tf32_disabled') is not True
                               or phase.get('float32_matmul_precision') != 'highest'):
                raise ValueError('GPU padding control must use full FP32 arithmetic without TF32')
        if (any(runtime.get(key) != RUNTIME[key] for key in ('torch', 'transformers'))
                or not isinstance(runtime.get('gpu_name'), str) or not runtime['gpu_name'].strip()):
            raise ValueError('GPU protocol runtime differs from pinned CUDA stack')
    return record


def validate_calibration(record, config):
    corrected.validate_calibration(record, config)
    if record['sampling'].get('stop_token_ids') != STOP_IDS:
        raise ValueError('Llama calibration stop tokens differ')
    return record


def audit_prompt_split(project_root):
    """Recompute tokenizer-dependent prompt filtering without loading weights."""
    from transformers import AutoTokenizer
    from vpo_rm.trainer import VPOTrainer, load_prompt_dataset
    from vpo_rm.reward_inputs import canonical_reward_input
    from vpo_rm.token_policy import load_actor_tokenizer
    config = common_config(project_root)
    # The trainer renders with the saved SFT tokenizer; a base checkpoint has no template.
    actor = load_actor_tokenizer(config['model'], config['init_adapter'])
    reward = AutoTokenizer.from_pretrained(config['rm'], local_files_only=True)
    prompts, _, split = load_prompt_dataset('HuggingFaceH4/ultrafeedback_binarized',
        dataset_path=config['dataset_path'], exclude_benchmarks=True)
    kept = [prompt for prompt in prompts if
        len(actor(VPOTrainer._render_chat_prompt(actor, prompt), add_special_tokens=False)['input_ids']) <= 2048
        and len(canonical_reward_input(reward, prompt, '')) <= 2048]
    ordered = hashlib.sha256(json.dumps(kept[:2000], ensure_ascii=False,
                                        separators=(',', ':')).encode()).hexdigest()
    actual = {key: split[key] for key in ('train_hash', 'validation_hash', 'num_unique', 'validation_size')}
    actual.update(filtered_train_prompts=len(kept), dropped_train_prompts=len(prompts) - len(kept),
                  formal_prompt_count=len(kept[:2000]), ordered_formal_prompts_sha256=ordered,
                  hash_serialization=PROMPT_AUDIT['hash_serialization'],
                  same_formal_prompts_as_qwen=ordered == PROMPT_AUDIT['ordered_formal_prompts_sha256'])
    if actual != PROMPT_AUDIT:
        raise ValueError('Llama prompt split/filter differs from the audited baseline comparison')
    return {**actual, 'benchmark_exclusion': split['benchmark_exclusion']}


def validate_runtime(report):
    if report.get('software') != RUNTIME:
        raise ValueError('Formal runtime package versions differ')
    gpus = report.get('gpus', [])
    if len(gpus) != 3 or any(row.get('memory_bytes', 0) < 120 * 2**30 for row in gpus):
        raise ValueError('Formal Llama suite requires three H200-class GPUs')
    if report.get('multiprocessing_method') != 'spawn':
        raise ValueError('vLLM multiprocessing must use spawn')
    return report


def runtime_report():
    import torch
    report = {'software': {name: importlib.metadata.version(name) for name in RUNTIME},
              'gpus': [{'name': torch.cuda.get_device_name(i),
                        'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                       for i in range(torch.cuda.device_count())],
              'multiprocessing_method': os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'python': sys.executable, 'cuda': torch.version.cuda}
    return validate_runtime(report)


def validate_report(path, identity, *, gpu=False):
    """Profile-aware GPU gate check: every profile arm must pass two rollouts."""
    report = corrected.validate_report(path, identity, gpu=False)
    if gpu:
        arms = report.get('arms', {})
        if set(arms) != set(profile()['arms']) or any(
                row.get('status') != 'passed' or row.get('rollouts') != 2 for row in arms.values()):
            raise ValueError('GPU validation must pass two rollouts for every profile arm')
        if any(row.get('runtime', {}).get('versions') != RUNTIME for row in arms.values()):
            raise ValueError('GPU validation runtime versions differ from the formal runtime')
        if any(row.get('quality', {}).get(stage, {}).get('rule', 'strict') != profile()['quality_rule']
               for row in arms.values() for stage in ('before', 'after')):
            raise ValueError('GPU validation quality gate used a different acceptance rule')
    return report


def submission_command(suite, arm):
    if arm not in profile_arms():
        raise ValueError('Unknown Llama experiment arm')
    suite = Path(suite).resolve()
    suffix = hashlib.sha256(str(suite).encode()).hexdigest()[:10]
    return ['rjob', 'submit', '--name', f"{profile()['job_prefix']}-{arm}-{suffix}", '--task-type', 'normal',
        '--priority', '9', '--enable-sshd', '--image', RUNTIME_IMAGE, '--image-pull-policy', 'IfNotPresent',
        '--gpu', '3', '--cpu', '48', '--memory', '600000', '--charged-group', 'ma4agismall_gpu',
        '--private-machine', 'group', '--namespace', 'ailab-ma4agismall',
        '--use-file-store', 'true', '--file-store-nfs-path', '10.68.62.222:/data:/data',
        '--mount', f'gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:{SHARED_PROJECT}',
        '--', 'bash', str(suite / 'run_arm.sh'), arm]


def prepare_suite(suite, cpu_validation, *, project_root=ROOT):
    suite, project_root = Path(suite).resolve(), Path(project_root).resolve()
    if suite.exists():
        raise FileExistsError(f'Use a fresh Llama suite directory: {suite}')
    config, identity = common_config(project_root), validation_identity(project_root)
    cpu = validate_protocol_report(json.loads(Path(cpu_validation).read_text()), config, mode='cpu')
    prompt_audit = audit_prompt_split(project_root)
    if not EXTRA_PACKAGES.is_dir():
        raise FileNotFoundError(f'Pinned extra packages directory missing: {EXTRA_PACKAGES}')
    free = shutil.disk_usage(suite.parent if suite.parent.exists() else project_root).free / 2**30
    if free < 60:
        raise RuntimeError(f'Need 60 GiB for shared preflight plus the profile arms; have {free:.1f}')
    suite.mkdir(parents=True)
    source = suite / 'source'; source.mkdir()
    for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs'):
        if (ROOT / directory).is_dir():
            shutil.copytree(ROOT / directory, source / directory,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    shutil.copy2(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    (source / 'configs').mkdir(exist_ok=True)
    shutil.copy2(baseline_path(), source / 'configs/llama_rl_baseline.json')
    for name, target in (('models', project_root.parent / 'models'),
                         ('datasets', project_root / 'datasets'), ('.vllm-extra', EXTRA_PACKAGES)):
        (source / name).symlink_to(target, target_is_directory=True)
    snapshot = {str(path.relative_to(source)): file_hash(path)
                for directory in ('vpo_rm', 'scripts', 'tests', 'third_party', 'configs')
                for path in (source / directory).rglob('*') if path.is_file()}
    snapshot['pyproject.toml'] = file_hash(source / 'pyproject.toml')
    if any(snapshot.get(name) != expected for name, expected in identity['source_sha256'].items()):
        raise ValueError('Source changed while freezing Llama suite')
    write_json(suite / 'cpu-protocol.json', cpu)
    write_json(suite / 'cpu-validation.json', {'status': 'passed', **identity,
        'protocol': {'path': str(suite / 'cpu-protocol.json'), 'sha256': file_hash(suite / 'cpu-protocol.json')}})
    write_json(suite / 'prompt-audit.json', prompt_audit)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'experiment_family': 'llama',
        'experiment_profile': ACTIVE_PROFILE, 'actor_response_eos_id': profile()['actor_response_eos_id'],
        'suite': str(suite), 'project_root': str(project_root), 'source_snapshot': str(source),
        'source_sha256': snapshot, 'validation_identity': identity, 'common_config': config,
        'arms': profile_arms(), 'runtime': RUNTIME, 'runtime_image': RUNTIME_IMAGE,
        'reward_input_protocol': 'canonical_chat_v1', 'startup_max_unmapped_content_fraction': .25,
        'baseline_manifest_sha256': BASELINE_SHA256,
        'initialization_weights_sha256': identity['initialization_weights_sha256'],
        'dataset_sha256': identity['dataset_sha256'],
        'validation': {'cpu': {'path': str(suite / 'cpu-validation.json'),
                              'sha256': file_hash(suite / 'cpu-validation.json')},
                       'cpu_protocol': {'path': str(suite / 'cpu-protocol.json'),
                                        'sha256': file_hash(suite / 'cpu-protocol.json')}},
        'prompt_audit': {'path': str(suite / 'prompt-audit.json'), 'sha256': file_hash(suite / 'prompt-audit.json')},
        'storage_budget': {'startup_min_free_gib': 60, 'free_gib_at_creation': free,
                           'retention': 'final checkpoint; step0/current/final adapters; shared preflight evidence'},
        'calibration': {'mode': 'fresh_shared_gpu_gate', 'owner': 'grpo', 'other_arms': 'require_completed_gate'}}
    write_json(suite / 'experiment.json', manifest)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?arm required}"
case "$ARM" in ARM_CASES) ;; *) exit 2 ;; esac
export PYTHONPATH="$SUITE/source:$SUITE/source/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRESENCE_PENALTY=0.0 VLLM_WORKER_MULTIPROC_METHOD=spawn
exec > >(tee -a "$SUITE/$ARM/job.log") 2>&1
trap 'task_rc=$?; printf "%s\\n" "$task_rc" > "$SUITE/$ARM/exit_code"' EXIT
python3 "$SUITE/source/scripts/llama_rl_launcher.py" run-arm --suite "$SUITE" --arm "$ARM"
'''.replace('ARM_CASES', '|'.join(profile_arms()))
    atomic_text(suite / 'run_arm.sh', shell)
    names = {}
    for arm in profile_arms():
        (suite / arm).mkdir()
        command = submission_command(suite, arm)
        write_json(suite / arm / 'submit_command.json', command)
        names[arm] = command[command.index('--name') + 1]
    write_json(suite / 'submission-plan.json', {'status': 'prepared_not_submitted', 'names': names,
        'launch_order': 'submit grpo first; submit other arms only after shared-gate.json status passed',
        'experiment_sha256': file_hash(suite / 'experiment.json'),
        'run_script_sha256': file_hash(suite / 'run_arm.sh'),
        'submit_command_sha256': {arm: file_hash(suite / arm / 'submit_command.json') for arm in profile_arms()}})
    print(f'PREPARED_NOT_SUBMITTED suite={suite} profile={ACTIVE_PROFILE}', flush=True)
    return manifest


def verify_source_snapshot(manifest):
    source = Path(manifest['source_snapshot'])
    for name, expected in manifest['source_sha256'].items():
        if file_hash(source / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')


def validate_prepared_suite(suite):
    suite = Path(suite)
    plan = json.loads((suite / 'submission-plan.json').read_text())
    if (file_hash(suite / 'experiment.json') != plan['experiment_sha256']
            or file_hash(suite / 'run_arm.sh') != plan['run_script_sha256']):
        raise ValueError('Prepared experiment manifest or run script changed')
    manifest = json.loads((suite / 'experiment.json').read_text())
    set_profile(manifest.get('experiment_profile', 'instruct'))
    if ROOT.resolve() != Path(manifest['source_snapshot']).resolve():
        raise ValueError('Run the Llama launcher from its frozen source snapshot')
    verify_source_snapshot(manifest)
    identity = validation_identity(manifest['project_root'])
    if (identity != manifest['validation_identity'] or manifest['arms'] != profile_arms()
            or manifest['common_config'] != common_config(manifest['project_root'])
            or manifest['runtime'] != RUNTIME or manifest['runtime_image'] != RUNTIME_IMAGE):
        raise ValueError('Prepared Llama inputs/configuration differ from validation')
    for record in [*manifest['validation'].values(), manifest['prompt_audit']]:
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Prepared CPU validation or prompt audit changed')
    validate_report(manifest['validation']['cpu']['path'], identity)
    validate_protocol_report(json.loads(Path(manifest['validation']['cpu_protocol']['path']).read_text()),
                             manifest['common_config'], mode='cpu')
    for arm, expected in plan['submit_command_sha256'].items():
        if file_hash(suite / arm / 'submit_command.json') != expected:
            raise ValueError('Prepared submission command changed')
    return manifest


def gate_commands(suite, manifest):
    suite, source = Path(suite), Path(manifest['source_snapshot'])
    config = manifest['common_config']
    protocol = [sys.executable, str(source / 'scripts/check_llama_protocol.py'),
        '--actor', config['model'], '--reward', config['rm'], '--init-adapter', config['init_adapter'],
        '--actor-eos', str(profile()['actor_response_eos_id']),
        '--output', str(suite / 'gpu-protocol.json'), '--gpu']
    preflight = [sys.executable, str(source / 'scripts/preflight_training.py'),
        '--output-dir', str(suite / 'gpu-preflight'), '--project-root', manifest['project_root'],
        '--runtime-image', manifest['runtime_image'], '--experiment-family', 'llama',
        '--experiment-profile', ACTIVE_PROFILE, '--arms', *profile()['arms'],
        '--capacity-arm', profile()['capacity_arm'], '--quality-rule', profile()['quality_rule']]
    return protocol, preflight


def read_shared_gate(suite, manifest):
    suite = Path(suite)
    gate = json.loads((suite / 'shared-gate.json').read_text())
    if (gate.get('status') != 'passed' or gate.get('identity') != manifest['validation_identity']
            or gate.get('experiment_sha256') != file_hash(suite / 'experiment.json')):
        raise ValueError('Shared Llama gate identity differs')
    required = {'gpu_validation', 'gpu_protocol', 'calibration'}
    if set(gate.get('evidence', {})) != required:
        raise ValueError('Shared gate evidence is incomplete')
    for record in gate['evidence'].values():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Shared gate evidence hash changed')
    gpu = validate_report(gate['evidence']['gpu_validation']['path'], manifest['validation_identity'], gpu=True)
    if (gpu.get('runtime_image') != manifest['runtime_image']
            or gpu['arms'][profile()['capacity_arm']].get('capacity', {}).get('status') != 'passed'):
        raise ValueError('GPU image or long-sequence capacity gate differs')
    cpu = json.loads(Path(manifest['validation']['cpu_protocol']['path']).read_text())
    validate_protocol_report(json.loads(Path(gate['evidence']['gpu_protocol']['path']).read_text()),
                             manifest['common_config'], mode='gpu', expected_identity=cpu['identity'])
    calibration = json.loads(Path(gate['evidence']['calibration']['path']).read_text())
    return validate_calibration(calibration, manifest['common_config'])


def shared_gate(suite, arm, manifest):
    suite = Path(suite)
    if (suite / 'shared-gate.json').is_file():
        return read_shared_gate(suite, manifest)
    if arm != 'grpo':
        raise FileNotFoundError('Shared Llama GPU gate must pass before starting another arm')
    with (suite / 'shared-gate.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another GRPO process owns the shared gate') from error
        if (suite / 'shared-gate.json').is_file():
            return read_shared_gate(suite, manifest)
        if any((suite / name).exists() for name in ('gate-started.json', 'gate-failure.json', 'gpu-preflight')):
            raise FileExistsError('Incomplete/failed prior gate exists; use a fresh suite')
        write_json(suite / 'gate-started.json', {'owner': arm, 'started_at': datetime.now(timezone.utc).isoformat()})
        try:
            protocol_command, preflight_command = gate_commands(suite, manifest)
            write_json(suite / 'gpu-protocol-command.json', protocol_command)
            with (suite / 'gpu-protocol.log').open('w') as log:
                subprocess.run(protocol_command, cwd=manifest['project_root'], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            cpu = json.loads(Path(manifest['validation']['cpu_protocol']['path']).read_text())
            validate_protocol_report(json.loads((suite / 'gpu-protocol.json').read_text()),
                manifest['common_config'], mode='gpu', expected_identity=cpu['identity'])
            write_json(suite / 'gpu-preflight-command.json', preflight_command)
            with (suite / 'gpu-preflight.log').open('w') as log:
                subprocess.run(preflight_command, cwd=manifest['project_root'], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            gpu_path = suite / 'gpu-preflight/gpu-validation.json'
            gpu = validate_report(gpu_path, manifest['validation_identity'], gpu=True)
            if (gpu.get('runtime_image') != manifest['runtime_image']
                    or gpu['arms'][profile()['capacity_arm']].get('capacity', {}).get('status') != 'passed'):
                raise ValueError('GPU image or long-sequence capacity gate differs')
            calibration_path = Path(gpu['calibration']['path'])
            if file_hash(calibration_path) != gpu['calibration']['sha256']:
                raise ValueError('Fresh calibration hash changed')
            calibration = validate_calibration(json.loads(calibration_path.read_text()), manifest['common_config'])
            if validation_identity(manifest['project_root']) != manifest['validation_identity']:
                raise ValueError('Inputs/source changed during shared GPU gate')
            verify_source_snapshot(manifest)
            write_json(suite / 'gpu-validation.json', gpu)
            write_json(suite / 'shared-calibration.json', {**calibration, '_verification': {
                'identity': manifest['validation_identity'], 'source_calibration_sha256': file_hash(calibration_path),
                'gpu_validation_sha256': file_hash(suite / 'gpu-validation.json')}})
            evidence = {name: {'path': str(suite / file), 'sha256': file_hash(suite / file)}
                        for name, file in [('gpu_validation', 'gpu-validation.json'),
                            ('gpu_protocol', 'gpu-protocol.json'), ('calibration', 'shared-calibration.json')]}
            # Publish last: readers must never observe a passed gate before its evidence exists.
            write_json(suite / 'shared-gate.json', {'status': 'passed', 'identity': manifest['validation_identity'],
                'experiment_sha256': file_hash(suite / 'experiment.json'), 'evidence': evidence,
                'completed_at': datetime.now(timezone.utc).isoformat()})
            return read_shared_gate(suite, manifest)
        except BaseException as error:
            write_json(suite / 'gate-failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(), 'failed_at': datetime.now(timezone.utc).isoformat()})
            raise


def validate_completion(train_dir, manifest):
    train_dir = Path(train_dir)
    summary = json.loads((train_dir / 'profile_summary.json').read_text())
    rows = summary.get('rollouts', [])
    if [row.get('rollout') for row in rows] != list(range(1, 251)):
        raise ValueError('Formal run did not complete all 250 rollouts')
    for row in rows:
        # A legitimate all-invalid rollout never runs backward and records no
        # gradient norm. Any gradient metric that is present must still be finite.
        metric_keys = ('loss',) if row.get('skipped_rollout', False) and 'grad_norm' not in row else ('loss', 'grad_norm')
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in metric_keys):
            raise ValueError('Final rollout metrics are not finite')
        # Preserve the baseline's explicit all-invalid-group skip policy. Every
        # rollout that keeps a group still performs exactly one optimizer step.
        expected_updates = 0 if row.get('skipped_rollout', False) else 1
        if type(row.get('optimizer_steps')) is not int or row['optimizer_steps'] != expected_updates:
            raise ValueError('Formal rollout optimizer step count differs from the baseline policy')
    checkpoint = train_dir / 'checkpoint-250'
    run = json.loads((checkpoint / 'run_manifest.json').read_text())
    if run.get('step') != 250:
        raise ValueError('Final checkpoint manifest does not describe step250')
    for name in ('adapter_model.safetensors', 'ref/adapter_model.safetensors', 'trainer_state.pt'):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Final checkpoint is incomplete: {name}')
    adapters = validate_checkpoint_adapters(checkpoint, manifest['common_config']['init_adapter'])
    return {'rollouts': 250, 'optimizer_steps': sum(row.get('optimizer_steps', 0) for row in rows),
            'skipped_rollouts': sum(bool(row.get('skipped_rollout')) for row in rows),
            'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': file_hash(checkpoint / 'run_manifest.json'),
            'adapter_sha256': file_hash(checkpoint / 'adapter_model.safetensors'), 'adapter_validation': adapters}


def validate_checkpoint_adapters(checkpoint, initial):
    import torch
    from safetensors import safe_open
    checkpoint, initial = Path(checkpoint), Path(initial)
    paths = [initial / 'adapter_model.safetensors', checkpoint / 'adapter_model.safetensors',
             checkpoint / 'ref/adapter_model.safetensors']
    changed = 0
    with safe_open(paths[0], framework='pt', device='cpu') as original, \
            safe_open(paths[1], framework='pt', device='cpu') as trained, \
            safe_open(paths[2], framework='pt', device='cpu') as reference:
        if not original.keys() or set(original.keys()) != set(trained.keys()) or set(original.keys()) != set(reference.keys()):
            raise ValueError('Final adapter/reference tensor keys differ from SFT')
        for key in original.keys():
            before, after, frozen = original.get_tensor(key), trained.get_tensor(key), reference.get_tensor(key)
            if not torch.equal(before, frozen):
                raise ValueError(f'Final frozen reference adapter changed: {key}')
            if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
                raise ValueError(f'Final trained adapter shape/dtype/finiteness differs: {key}')
            changed += not torch.equal(before, after)
    if not changed:
        raise ValueError('Final trained adapter has not changed from SFT')
    return {'status': 'passed', 'reference_unchanged': True, 'changed_tensors': changed}


def run_arm(suite, arm):
    suite = Path(suite).resolve(); out = suite / arm
    set_profile(json.loads((suite / 'experiment.json').read_text()).get('experiment_profile', 'instruct'))
    if arm not in profile_arms():
        raise ValueError('Unknown Llama experiment arm for this profile')
    with (out / 'arm.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this arm') from error
        try:
            if (out / 'train').exists() or (out / 'completion.json').exists():
                raise FileExistsError('Arm already has training output; no resume protocol')
            manifest = validate_prepared_suite(suite)
            os.chdir(manifest['project_root'])
            if arm != 'grpo':
                read_shared_gate(suite, manifest)
            free = shutil.disk_usage(suite).free / 2**30
            if free < manifest['storage_budget']['startup_min_free_gib']:
                raise RuntimeError(f'Insufficient suite storage: {free:.1f} GiB')
            runtime = runtime_report()
            write_json(out / 'runtime.json', {**runtime, 'free_gib': free,
                'experiment_sha256': file_hash(suite / 'experiment.json')})
            atomic_text(out / 'stage', 'shared_gpu_gate\n')
            calibration = shared_gate(suite, arm, manifest)
            # Gate probes may update their own adapters. Formal jobs always reload the protected SFT.
            manifest = validate_prepared_suite(suite)
            write_json(out / 'shared-calibration.json', calibration)
            command = training_command(manifest, arm, out / 'train', sigma0=calibration['sigma0'])
            write_json(out / 'command.json', command)
            print('TRAIN_COMMAND ' + json.dumps(command), flush=True)
            atomic_text(out / 'stage', 'training\n')
            started = time.monotonic()
            run_training_process(command, manifest['project_root'], out / 'train', manifest, status_dir=out)
            completed = validate_completion(out / 'train', manifest)
            if file_hash(Path(manifest['common_config']['init_adapter']) / 'adapter_model.safetensors') != profile()['protected_sft_sha256']:
                raise ValueError('Protected SFT changed during training')
            write_json(out / 'completion.json', {**completed, 'status': 'complete',
                'wall_seconds': time.monotonic() - started, 'sigma0': calibration['sigma0'],
                'experiment_sha256': file_hash(suite / 'experiment.json'),
                'shared_gate_sha256': file_hash(suite / 'shared-gate.json')})
            atomic_text(out / 'stage', 'complete\n')
        except BaseException as error:
            write_json(out / 'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(), 'failed_at': datetime.now(timezone.utc).isoformat()})
            atomic_text(out / 'stage', 'failed\n')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--suite', type=Path, required=True)
    prepare.add_argument('--cpu-validation', type=Path, required=True, help='Passed check_llama_protocol.py CPU report')
    prepare.add_argument('--project-root', type=Path, default=ROOT)
    prepare.add_argument('--profile', choices=sorted(PROFILES), default='instruct',
                         help='Audited actor profile: instruct (four arms) or base (grpo and lam4)')
    run = commands.add_parser('run-arm')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        set_profile(args.profile)
        prepare_suite(args.suite, args.cpu_validation, project_root=args.project_root)
    else:
        run_arm(args.suite, args.arm)


if __name__ == '__main__':
    main()
