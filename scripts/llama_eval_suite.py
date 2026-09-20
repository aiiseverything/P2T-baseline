#!/usr/bin/env python3
"""Freeze the requested Llama evaluations and summarize verified worker results.

This module prepares submission commands but never submits or retries jobs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, digest, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head

TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
SEEDS = [42, 43, 44, 45, 46]
METRICS = ('prompt_strict', 'prompt_loose', 'inst_strict', 'inst_loose')
SHARED = Path('/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM')
TRAINING = Path('/data/VPO-RM/runs/llama31-rl-canonical-20260918-v3')
SFT_SHA = '52a68bd14ecea9bf660e0bb04a78cb68e6d2c263eedcfb9c40c13acba01453cc'
IFEVAL_SHA = '67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49'
PROMPTS_SHA = '1111707a265f6ff9b8927fb9a85cb3d00a8ab7095a01c319bc1781f804d2c5f7'
IMAGE = 'registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_training_acceptance(training, acceptance):
    require(acceptance.get('status') == 'passed' and acceptance.get('inputs_sha256'),
            'Missing final training acceptance audit')
    for path, expected in acceptance['inputs_sha256'].items():
        require(file_hash(path) == expected, f'Final acceptance audit input changed: {path}')
    required = [Path(training) / 'experiment.json', Path(training) / 'shared-gate.json']
    required.extend(Path(training) / tag / 'completion.json' for tag in TAGS[2:])
    require(all(str(p) in acceptance['inputs_sha256'] for p in required), 'Incomplete acceptance audit binding')
    for tag in TAGS[2:]:
        completed = json.loads((Path(training) / tag / 'completion.json').read_text())
        require(acceptance['arms'][tag]['adapter_sha256'] == completed['adapter_sha256'],
                f'Final acceptance adapter hash differs: {tag}')


def model_inputs(training, manifest, *, expected_sft_sha=SFT_SHA):
    """Bind existing exports to accepted final checkpoint hashes, without copying."""
    training = Path(training)
    sft = Path(manifest['common_config']['init_adapter'])
    require(file_hash(sft / 'adapter_model.safetensors') == expected_sft_sha,
            'Protected SFT weights changed')
    adapters = {'base': 'none', 'sft-init': str(sft)}
    for tag in TAGS[2:]:
        complete = json.loads((training / tag / 'completion.json').read_text())
        require(complete.get('status') == 'complete' and complete.get('rollouts') == 250
                and complete.get('optimizer_steps') == 250 and complete.get('skipped_rollouts') == 0,
                f'Unaccepted final checkpoint: {tag}')
        adapter = training / tag / 'train/vllm-adapters/step-250'
        require(file_hash(adapter / 'adapter_model.safetensors') == complete['adapter_sha256'],
                f'Export differs from accepted final checkpoint: {tag}')
        adapters[tag] = str(adapter)
    result = {}
    for tag, adapter in adapters.items():
        policy = resolve_policy_head(adapter, 'float32')
        paths = [] if adapter == 'none' else [Path(adapter) / name for name in
                    ('adapter_config.json', 'adapter_model.safetensors')]
        paths.extend(Path(row['path']) for row in policy['metadata'])
        result[tag] = {'adapter': adapter, 'files_sha256': {str(p): file_hash(p) for p in paths},
                       'policy': policy}
    return result


def submission_command(suite, tag):
    suite = Path(suite).resolve()
    require(tag in TAGS and suite.name in ('reward256', 'ifeval5'), 'Unknown benchmark or model')
    suffix = hashlib.sha256(str(suite).encode()).hexdigest()[:8]
    return ['rjob', 'submit', '--name', f'llama-{suite.name}-{tag}-{suffix}',
            '--task-type', 'normal', '--priority', '9', '--enable-sshd', '--image', IMAGE,
            '--image-pull-policy', 'IfNotPresent', '--gpu', '1', '--cpu', '8', '--memory', '128000',
            '--charged-group', 'ma4agismall_gpu', '--private-machine', 'group',
            '--namespace', 'ailab-ma4agismall', '--use-file-store', 'true',
            '--file-store-nfs-path', '10.68.62.222:/data:/data',
            '--mount', f'gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:{SHARED}',
            '--', 'bash', str(suite / 'run_worker.sh'), tag]


def worker_shell():
    return '''#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:?model tag required}"
case "$TAG" in base|sft-init|grpo|lam2|lam4|lam8) ;; *) exit 2 ;; esac
SHARED=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
mkdir -p "$SUITE/job/$TAG"
exec > >(tee -a "$SUITE/job/$TAG/worker.log") 2>&1
trap 'task_rc=$?; printf "%s\\n" "$task_rc" > "$SUITE/job/$TAG/exit_code"' EXIT
TASK_CACHE="$(mktemp -d "/tmp/llama-eval-${TAG}.XXXXXX")"
export HF_HOME="$TASK_CACHE/hf" HF_DATASETS_CACHE="$TASK_CACHE/datasets"
export VLLM_CACHE_ROOT="$TASK_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor"
export TRITON_CACHE_DIR="$TASK_CACHE/triton" XDG_CACHE_HOME="$TASK_CACHE/xdg"
export PYTHONPATH="$SUITE/source:$SHARED/runs/ifeval-final-canonical-20260917/.ifeval-extra:$SHARED/runs/arena-hard-v2-canonical-20260917/.arena-extra:$SHARED/.vllm-extra:$SUITE/source/third_party/ifeval"
export NLTK_DATA="$SUITE/source/third_party/ifeval/nltk_data"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 EVAL_TOPP=1.0 EVAL_PP=0.0
cd "$SUITE/source"
python3 "$SUITE/source/scripts/llama_eval_worker.py" --suite "$SUITE" --tag "$TAG"
'''


def prepare(campaign, validation, training=TRAINING):
    campaign, training = Path(campaign).resolve(), Path(training).resolve()
    require(not campaign.exists(), 'Use a fresh evaluation output directory')
    check = json.loads(Path(validation).read_text())
    require(check.get('status') == 'passed' and check.get('source_sha256'), 'Missing CPU validation')
    for name, expected in check['source_sha256'].items():
        require(file_hash(ROOT / name) == expected, f'Source changed after CPU validation: {name}')
    require(shutil.disk_usage(campaign.parent).free > 10 * 2**30, 'Need 10 GiB free for evaluation artifacts')
    training_manifest = json.loads((training / 'experiment.json').read_text())
    acceptance = json.loads((training / 'audit/final-root-review.json').read_text())
    validate_training_acceptance(training, acceptance)
    config = training_manifest['common_config']
    models = model_inputs(training, training_manifest)
    # The completed full-shard audit binds prior SHA checks and current file stats.
    from scripts.llama_rl_launcher import validate_asset_evidence
    evidence = training_manifest['validation_identity']['full_asset_verification']
    asset_path = evidence['path']
    require(file_hash(asset_path) == evidence['sha256'], 'Training asset audit changed')
    validate_asset_evidence(asset_path, config)
    qreward = SHARED / 'runs/reward256-canonical-20260917'
    qifeval = SHARED / 'runs/ifeval-five-seeds-canonical-20260917'
    reward_baseline = json.loads((qreward / 'experiment.json').read_text())
    ifeval_baseline = json.loads((qifeval / 'experiment.json').read_text())
    require(file_hash(qreward / 'eval_prompts.json') == PROMPTS_SHA, 'Qwen reward prompt set changed')
    require(file_hash(qifeval / 'input_data.jsonl') == IFEVAL_SHA, 'Canonical IFEval data changed')
    require(file_hash(config['dataset_path']) == reward_baseline['dataset_sha256'], 'Reward dataset changed')
    from scripts.eval_checkpoints import load_validation_prompts
    from vpo_rm.token_policy import load_actor_tokenizer, tokenize_rendered_prompts, get_stop_token_ids
    from vpo_rm.trainer import VPOTrainer
    tokenizer = load_actor_tokenizer(config['model'], tokenizer_name=config['init_adapter'])
    require(list(get_stop_token_ids(tokenizer)) == [128001, 128008, 128009]
            and tokenizer.pad_token_id == 128004, 'Llama token policy mismatch')
    reward_prompts = json.loads((qreward / 'eval_prompts.json').read_text())
    require(load_validation_prompts(config['dataset_path'], 256) == reward_prompts,
            'Llama reward prompts differ from Qwen')
    with (qifeval / 'input_data.jsonl').open() as stream:
        ifeval_data = [json.loads(line) for line in stream if line.strip()]
    require(len(ifeval_data) == 541 and sum(len(r['instruction_id_list']) for r in ifeval_data) == 834,
            'Canonical IFEval dimensions changed')
    fingerprints = {key: fingerprint(config[field], full_weights=False) for key, field in
                    [('base_model', 'model'), ('reward_model', 'rm'), ('tokenizer', 'init_adapter')]}
    campaign.mkdir(parents=True)
    for benchmark, baseline, prompts in [('reward256', reward_baseline, reward_prompts),
                                         ('ifeval5', ifeval_baseline, [r['prompt'] for r in ifeval_data])]:
        suite = campaign / benchmark; suite.mkdir()
        source = suite / 'source'; source.mkdir()
        for name in ('scripts', 'vpo_rm'):
            shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'third_party/ifeval', source / 'third_party/ifeval',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        (source / 'datasets').symlink_to(ROOT / 'datasets', target_is_directory=True)
        (source / 'models').symlink_to(Path(config['model']).parent, target_is_directory=True)
        shutil.copy2(ROOT / 'pyproject.toml', source / 'pyproject.toml')
        shutil.copy2(validation, suite / 'cpu-validation.json')
        if benchmark == 'reward256':
            shutil.copy2(qreward / 'eval_prompts.json', suite / 'eval_prompts.json')
            dataset = config['dataset_path']
        else:
            shutil.copy2(qifeval / 'input_data.jsonl', suite / 'input_data.jsonl')
            dataset = str(suite / 'input_data.jsonl')
        encoded = tokenize_rendered_prompts(tokenizer, [VPOTrainer._render_chat_prompt(tokenizer, p) for p in prompts])
        rows = [row['prompt_token_ids'] for row in encoded]
        require(all(row[0] == 128000 and row.count(128000) == 1 and len(row) + 2048 <= 4096
                    for row in rows), 'Llama prompt protocol or context budget failed')
        manifest = {'created_at': now(), 'benchmark': benchmark, 'base_model': config['model'],
            'reward_model': config['rm'], 'tokenizer': config['init_adapter'],
            **{key + '_fingerprint': value for key, value in fingerprints.items()},
            'models': models, 'dataset': dataset, 'dataset_sha256': file_hash(dataset),
            'gpu_versions': baseline['gpu_versions'], 'scoring_versions': ifeval_baseline['scoring_versions'],
            'seeds': [42] if benchmark == 'reward256' else SEEDS, 'scoring_seed': 42,
            'expected_stop_token_ids': [128001, 128008, 128009], 'actor_pad_token_id': 128004,
            'prompt_token_ids_sha256': digest(rows), 'max_prompt_tokens': max(map(len, rows)),
            'training_suite': str(training), 'training_manifest_sha256': file_hash(training / 'experiment.json'),
            'training_acceptance_sha256': file_hash(training / 'audit/final-root-review.json'),
            'qwen_baseline': str(qreward if benchmark == 'reward256' else qifeval),
            'qwen_baseline_manifest_sha256': file_hash((qreward if benchmark == 'reward256' else qifeval) / 'experiment.json'),
            'generation': {'temperature': 1., 'top_p': 1., 'top_k': -1, 'n': 1, 'min_tokens': 0,
                           'max_tokens': 2048, 'max_model_len': 4096, 'policy_head_dtype': 'float32'},
            'reward_protocol': {'serialization': 'canonical_chat_v1', 'metric': 'raw_scalar_no_length_or_kl_penalty',
                                'head_dtype': 'native_bfloat16', 'microbatch': 1,
                                'qwen_microbatch': 4, 'reason': 'Llama single-row RM protocol validated during training'},
            'resources_per_job': {'gpu': 1, 'cpu': 8, 'memory_mib': 128000, 'image': IMAGE}}
        atomic_text(suite / 'run_worker.sh', worker_shell())
        manifest['files_sha256'] = {str(p.relative_to(suite)): file_hash(p) for p in source.rglob('*')
                                   if p.is_file() and 'datasets' not in p.relative_to(source).parts
                                   and 'models' not in p.relative_to(source).parts}
        for name in ('run_worker.sh', 'cpu-validation.json', 'eval_prompts.json' if benchmark == 'reward256' else 'input_data.jsonl'):
            manifest['files_sha256'][name] = file_hash(suite / name)
        for name, expected in check['source_sha256'].items():
            require(file_hash(source / name) == expected, f'Source changed during freeze: {name}')
        save(suite / 'experiment.json', manifest)
        save(suite / 'submission-plan.json', {tag: submission_command(suite, tag) for tag in TAGS})
    save(campaign / 'preparation.json', {'status': 'prepared_not_submitted', 'created_at': now(),
        'benchmarks': ['reward256', 'ifeval5'], 'models': list(TAGS), 'n_jobs': 12,
        'reward_evaluations': 6, 'ifeval_evaluations': 30})
    return campaign


def aggregate_ifeval(rows):
    require(len(rows) == 30 and {(r['model'], r['seed']) for r in rows}
            == {(tag, seed) for tag in TAGS for seed in SEEDS}, 'Expected all 30 model/seed pairs, exact coverage')
    individual, means = [], []
    for row in rows:
        require(all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS), 'Invalid IFEval metrics')
        individual.append({'model': row['model'], 'seed': row['seed'],
            **{k + '_pct': row[k] * 100 for k in METRICS},
            'four_metric_mean_pct': statistics.mean(row[k] * 100 for k in METRICS),
            'mean_tokens': row['mean_tokens'], 'truncated_count': row['capped']})
    for tag in TAGS:
        records = [row for row in individual if row['model'] == tag]
        summary = {'model': tag, 'n_evaluations': 5}
        for key in (*METRICS, 'four_metric_mean'):
            values = [row[key + '_pct'] for row in records]
            summary.update({key + '_mean_pct': statistics.mean(values),
                            key + '_sd_pp': statistics.stdev(values),
                            key + '_min_pct': min(values), key + '_max_pct': max(values)})
        means.append(summary)
    return individual, means


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def summarize(suite):
    from scripts import llama_eval_worker as worker
    suite = Path(suite)
    manifest = json.loads((suite / 'experiment.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        require(file_hash(suite / name) == expected, f'Frozen source changed: {name}')
    for tag in TAGS:
        state = json.loads((suite / 'job' / tag / 'status.json').read_text())
        require(state['state'] == 'complete' and (suite / 'job' / tag / 'exit_code').read_text().strip() == '0',
                f'Worker not complete: {tag}')
    if manifest['benchmark'] == 'ifeval5':
        rows = [dict(worker.validate_seed_result(suite / 'results', tag, seed, manifest), model=tag)
                for tag in TAGS for seed in SEEDS]
        individual, means = aggregate_ifeval(rows)
        report = {'status': 'complete', 'n_evaluations': 30, 'seeds': SEEDS, 'scoring_seed': 42,
            'individual_results': individual, 'model_summary': means, 'verification': rows,
            'custom_metric': {'name': 'four_metric_mean', 'official': False,
                             'formula': '(prompt_strict + prompt_loose + inst_strict + inst_loose) / 4'},
            'interpretation': 'Five generation seeds of each fixed model, not pass@5 or five training seeds.'}
        atomic_text(suite / 'results_30.csv', csv_text(individual))
        atomic_text(suite / 'model_summary.csv', csv_text(means))
    else:
        rows = [dict(worker.validate_reward_result(suite / 'results' / tag, tag, manifest), model=tag) for tag in TAGS]
        report = {'status': 'complete', 'n_evaluations': 6, 'n_prompts_per_model': 256,
                  'seed': 42, 'results': rows, 'metric': 'Raw Llama RM scalar; no length or KL penalty'}
    report.update(verified_at=now(), experiment_sha256=file_hash(suite / 'experiment.json'))
    save(suite / 'summary.json', report)
    save(suite / 'completion.json', {'status': 'complete', 'verified_at': now(),
                                    'summary_sha256': file_hash(suite / 'summary.json')})
    return report


def worker_states(suite):
    values = []
    for tag in TAGS:
        job = Path(suite) / 'job' / tag
        status, exit_path = job / 'status.json', job / 'exit_code'
        if exit_path.exists() and exit_path.read_text().strip() != '0':
            values.append('failed')
        else:
            values.append(json.loads(status.read_text()).get('state') if status.exists() else 'pending')
    return values


def watch(campaign):
    campaign = Path(campaign)
    while True:
        states = {}
        for benchmark in ('reward256', 'ifeval5'):
            suite = campaign / benchmark
            if (suite / 'completion.json').exists():
                states[benchmark] = 'complete'; continue
            values = worker_states(suite)
            if 'failed' in values:
                states[benchmark] = 'failed'
            elif all(value == 'complete' for value in values) and all((suite / 'job' / tag / 'exit_code').exists() for tag in TAGS):
                try:
                    summarize(suite); states[benchmark] = 'complete'
                except Exception as error:
                    save(suite / 'summary-failure.json', {'status': 'failed', 'at': now(), 'error': repr(error)})
                    states[benchmark] = 'failed'
            else:
                states[benchmark] = 'running'
        save(campaign / 'watch-status.json', {'updated_at': now(), 'benchmarks': states})
        if all(state in ('complete', 'failed') for state in states.values()):
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--campaign', required=True); prep.add_argument('--validation', required=True)
    summary = sub.add_parser('summarize'); summary.add_argument('--suite', required=True)
    watcher = sub.add_parser('watch'); watcher.add_argument('--campaign', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.campaign, args.validation))
    elif args.command == 'summarize':
        print(json.dumps(summarize(args.suite)))
    else:
        watch(args.campaign)


if __name__ == '__main__':
    main()
