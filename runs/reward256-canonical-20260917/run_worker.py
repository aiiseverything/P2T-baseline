"""Run one immutable six-model reward experiment arm on one H200."""
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SUITE = Path(__file__).resolve().parent
PROJECT = SUITE.parent.parent
SOURCE = SUITE / 'source'
sys.path.insert(0, str(SOURCE))
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint, digest


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    tag = sys.argv[1]
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    if tag not in manifest['models']:
        raise ValueError(f'Unknown model {tag}')
    job = SUITE / 'job' / tag
    job.mkdir(parents=True, exist_ok=True)
    state = {'tag': tag, 'state': 'preflight', 'started_at': now()}
    save(job / 'status.json', state)
    try:
        for name, expected in manifest['files_sha256'].items():
            if file_hash(SUITE / name) != expected:
                raise ValueError(f'Frozen experiment file changed: {name}')
        for path, expected in manifest['models'][tag]['files_sha256'].items():
            if file_hash(path) != expected:
                raise ValueError(f'Model input changed: {path}')
        for key in ('base_model', 'reward_model'):
            if fingerprint(manifest[key], full_weights=False) != manifest[key + '_fingerprint']:
                raise ValueError(f'{key} fingerprint changed')
        if file_hash(manifest['dataset']) != manifest['dataset_sha256']:
            raise ValueError('Dataset changed')
        versions = {name: importlib.metadata.version(name) for name in manifest['gpu_versions']}
        if versions != manifest['gpu_versions']:
            raise RuntimeError(f'GPU runtime drift: {versions}')
        for name in ('peft', 'pyarrow', 'pandas'):
            importlib.import_module(name)
        import torch
        if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
            raise RuntimeError('This protocol requires exactly one H200 per rjob')
        from scripts.eval_checkpoints import load_validation_prompts
        from scripts.eval_policy import resolve_policy_head
        from scripts.eval_artifacts import validate_adapter_base
        from transformers import AutoTokenizer
        from vpo_rm.trainer import VPOTrainer
        prompts = load_validation_prompts(manifest['dataset'], 256)
        if prompts != json.loads((SUITE / 'eval_prompts.json').read_text()):
            raise ValueError('Evaluation prompts differ from the audited frozen 256')
        adapter = manifest['models'][tag]['adapter']
        validate_adapter_base(adapter, manifest['base_model'])
        if resolve_policy_head(adapter, 'float32') != manifest['models'][tag]['policy']:
            raise ValueError('Actor precision/provenance changed')
        tokenizer = AutoTokenizer.from_pretrained(manifest['base_model'], trust_remote_code=True)
        lengths = [len(tokenizer.encode(VPOTrainer._render_chat_prompt(tokenizer, p),
                                       add_special_tokens=False)) for p in prompts]
        if min(lengths) < 1 or max(lengths) + 2048 > 4096:
            raise ValueError('Full prompt plus uniform output budget exceeds context')
        runtime = {'versions': versions, 'python': sys.version, 'checked_at': now(),
            'gpu': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'free_total_bytes': list(torch.cuda.mem_get_info(0)),
            'frozen_prompts_digest': digest(prompts), 'max_prompt_tokens': max(lengths),
            'dependency_modules': {name: importlib.import_module(name).__file__
                                   for name in ('peft', 'pyarrow', 'pandas')}}
        save(job / 'runtime.json', runtime)
        output = SUITE / 'results' / tag
        command = [sys.executable, str(SOURCE / 'scripts/eval_checkpoints.py'),
            '--run', f'{tag}={adapter}', '--model', manifest['base_model'],
            '--rm', manifest['reward_model'], '--dataset-path', manifest['dataset'],
            '--num-prompts', '256', '--temps', '1.0', '--max-tokens', '2048', '--min-tokens', '0',
            '--max-num-seqs', '32', '--rm-microbatch', '4', '--rm-device', 'cuda:0',
            '--seed', '42', '--policy-head-dtype', 'float32', '--output', str(output)]
        save(job / 'command.json', command)
        state.update(state='evaluating', runtime_ready_at=now())
        save(job / 'status.json', state)
        print(json.dumps({'runtime': runtime, 'command': command}), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)
        result_manifest = json.loads((output / 'manifest.json').read_text())
        for name, expected in result_manifest['outputs'].items():
            if file_hash(output / name) != expected:
                raise ValueError(f'Output integrity failure: {name}')
        if json.loads((output / 'eval_prompts.json').read_text()) != prompts:
            raise ValueError('Output prompt identity failure')
        rows = [json.loads(line) for line in (output / 'eval.jsonl').read_text().splitlines()]
        step = 0 if tag in ('base', 'sft-init') else 250
        if (len(rows) != 256 or [r['prompt'] for r in rows] != list(range(256))
                or any(r['run'] != tag or r['step'] != step or r['temp'] != 1.0
                       or not math.isfinite(r['score']) or not 0 <= r['response_tokens'] <= 2048 for r in rows)):
            raise ValueError('Result coverage/value failure')
        cfg = result_manifest['config']
        if (cfg['policy_head_dtype'] != 'float32' or cfg['top_p'] != 1.0
                or cfg['presence_penalty'] != 0.0 or cfg['args']['seed'] != 42
                or cfg['reward_input_protocol'] != 'canonical_chat_v1'):
            raise ValueError('Result protocol mismatch')
        state.update(state='complete', finished_at=now(), n_scores=256,
                     summary=json.loads((output / 'summary.json').read_text()))
        save(job / 'status.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(state='failed', failed_at=now(), error=repr(error))
        save(job / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
