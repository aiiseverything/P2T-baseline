#!/usr/bin/env python3
"""Frozen six-policy Alpaca generation and one bounded GPT-4.1 judge invocation."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from scripts.eval_artifacts import atomic_text, cache_matches, digest, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head
from scripts.llama_eval_worker import (require, save, now, read_jsonl, close_number,
    validate_adapter_result, validate_tokenizer_result, validate_prompt_protocol)
from scripts import judge_alpaca as judge

TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
STOPS = [128001, 128008, 128009]
RECIPE = {'temp': 1., 'n': 1, 'top_p': 1., 'top_k': -1}
GEN_FILE = 'generations_t1.0_n1.jsonl'
JUDGE_SHA = '9fa7f4cbf110b21bb520805a9e913cc2ccbf8da7b539ca184963429231ef0dce'
GEN_SOURCES = {'scripts/eval_artifacts.py', 'scripts/eval_alpaca.py',
    'vpo_rm/token_policy.py', 'vpo_rm/model_identity.py', 'vpo_rm/trainer.py',
    'vpo_rm/integration.py', 'vpo_rm/alignment.py'}


def validate_manifest(manifest):
    require(manifest['benchmark'] == 'alpaca805' and manifest['seeds'] == [42], 'Alpaca protocol changed')
    require(set(manifest['models']) == set(TAGS), 'Expected all six policies')
    require(manifest['tokenizer'] == manifest['base_model'] and Path(manifest['base_model']).is_absolute(),
            'Expected the absolute Base tokenizer source')
    require(manifest['expected_stop_token_ids'] == STOPS and manifest['actor_pad_token_id'] == 128004,
            'Actor stop/pad protocol changed')
    require(manifest['judge'] == {'model': 'gpt-4.1', 'workers': 16, 'budget_cny': 30,
                                 'source_sha256': JUDGE_SHA}, 'Judge protocol changed')
    for tag, item in manifest['models'].items():
        require((item['adapter'] == 'none') == (tag == 'base'), 'Only Base may omit its adapter')
        require(item['policy']['policy_head_dtype'] == 'float32', 'Expected FP32 actor head')
        if tag not in ('base', 'sft-init'):
            require(Path(item['adapter']).name in ('step-250', 'checkpoint-250'), 'Expected final RL step 250')


def references(suite):
    rows = read_jsonl(Path(suite) / 'references.jsonl')
    judge.validate_references(rows)
    require(len(rows) == 805, 'Expected 805 reference instructions')
    return rows


def verify_inputs(suite, manifest, tag=None):
    suite = Path(suite).resolve(); validate_manifest(manifest)
    required = {'source/' + name for name in GEN_SOURCES | {
        'scripts/llama_alpaca_campaign.py', 'scripts/llama_eval_worker.py',
        'scripts/eval_policy.py', 'scripts/judge_alpaca.py'}}
    required.update(('references.jsonl', 'judge_template.txt'))
    require(required <= set(manifest['files_sha256']), 'Missing frozen input bindings')
    for relative, expected in manifest['files_sha256'].items():
        path = suite / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(suite), 'Frozen path escapes suite')
        require(file_hash(path) == expected, f'Frozen input hash changed: {relative}')
    require(file_hash(suite / 'source/scripts/judge_alpaca.py') == JUDGE_SHA
            and file_hash(judge.__file__) == JUDGE_SHA, 'Judge source changed')
    for name in ('base_model', 'tokenizer'):
        require(fingerprint(manifest[name], full_weights=False) == manifest[name + '_fingerprint'],
                f'{name} fingerprint changed')
    for label in ([tag] if tag is not None else TAGS):
        item = manifest['models'][label]
        for path, expected in item['files_sha256'].items():
            require(file_hash(path) == expected, f'Adapter input changed: {path}')
        require(resolve_policy_head(item['adapter'], 'float32') == item['policy'], 'Actor precision identity changed')
    references(suite)


def generation_command(suite, manifest, tag):
    suite = Path(suite); validate_manifest(manifest)
    return [sys.executable, str(suite / 'source/scripts/eval_alpaca.py'),
        '--model', manifest['base_model'], '--dataset', str(suite / 'references.jsonl'),
        '--adapters', f"{tag}={manifest['models'][tag]['adapter']}",
        '--output', str(suite / 'generations'), '--recipes', '1.0:1:1.0:-1',
        '--seed', '42', '--max-tokens', '2048', '--policy-head-dtype', 'float32']


def runtime_preflight(suite, manifest, tag):
    from scripts.eval_artifacts import validate_adapter_base
    from vpo_rm.token_policy import load_actor_tokenizer
    import torch
    versions = {name: importlib.metadata.version(name) for name in manifest['gpu_versions']}
    require(versions == manifest['gpu_versions'], 'Generation runtime version drift')
    require(torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0), 'Expected one H200')
    validate_adapter_base(manifest['models'][tag]['adapter'], manifest['base_model'])
    tokenizer = load_actor_tokenizer(manifest['base_model'])
    ids = validate_prompt_protocol(tokenizer, [row['instruction'] for row in references(suite)], manifest)
    return {'checked_at': now(), 'gpu': torch.cuda.get_device_name(0), 'gpu_versions': versions,
            'prompt_token_ids_sha256': digest(ids), 'max_prompt_tokens': max(map(len, ids)), 'n_prompts': 805}


def validate_generation(suite, tag, manifest):
    suite = Path(suite); validate_manifest(manifest); refs = references(suite)
    directory = suite / 'generations' / tag; path = directory / GEN_FILE
    cache_path = directory / 'manifest_t1.0_n1.json'; cache = json.loads(cache_path.read_text())
    require(cache['outputs'] == {GEN_FILE: file_hash(path)}, 'Generation hash mismatch')
    cfg = cache['config']; item = manifest['models'][tag]
    require(cfg['protocol'] == 2 and cfg['scorer'] == 'alpaca' and cfg['recipe'] == RECIPE
            and cfg['seed'] == 42 and cfg['max_tokens'] == 2048
            and cfg['engine']['dtype'] == 'bfloat16' and cfg['engine']['max_model_len'] == 4096,
            'Generation recipe/engine protocol changed')
    require(cfg['model'] == manifest['base_model_fingerprint'] and cfg['policy'] == item['policy']
            and cfg['dataset'] == fingerprint(suite / 'references.jsonl') and cfg['stop_token_ids'] == STOPS,
            'Generation model/policy/dataset/stop identity changed')
    validate_adapter_result(cfg['adapter'], item); validate_tokenizer_result(cfg, manifest)
    if 'expected_output_support' in manifest:
        require(cfg.get('output_support') == manifest['expected_output_support'],
                'Generation output support changed')
    require(cfg['runtime_versions'] == {name: manifest['gpu_versions'][name] for name in ('transformers', 'vllm')},
            'Generation runtime identity changed')
    require(GEN_SOURCES <= set(cfg['sources']) and all(
        manifest['files_sha256'].get('source/' + name) == value for name, value in cfg['sources'].items()),
        'Generation source identity changed')
    rows = read_jsonl(path); require(len(rows) == 805, 'Incomplete generation coverage')
    for i, (row, ref) in enumerate(zip(rows, refs)):
        require(type(row.get('idx')) is int and row['idx'] == i
                and type(row.get('sample_idx')) is int and row['sample_idx'] == 0
                and row['instruction'] == ref['instruction'], 'Generation row identity/coverage mismatch')
        require(isinstance(row.get('response'), str), 'Candidate response must be text')
        require(all(key in row for key in ('response_tokens', 'finish_reason', 'stop_reason', 'last_token_id')),
                'Missing finish metadata')
        require(type(row['response_tokens']) is int and 0 < row['response_tokens'] <= 2048
                and row['finish_reason'] in ('stop', 'length')
                and type(row['last_token_id']) is int and 0 <= row['last_token_id'] < 128256
                and row['stop_reason'] in (None, *STOPS), 'Invalid finish metadata')
        require((row['finish_reason'] != 'length' or row['response_tokens'] == 2048)
                and (row['finish_reason'] != 'stop' or row['last_token_id'] in STOPS), 'Inconsistent finish metadata')
    lengths = sorted(row['response_tokens'] for row in rows)
    return {'tag': tag, 'n': 805, 'mean_tokens': sum(lengths) / 805, 'p95_tokens': lengths[int(.95 * 805)],
            'capped': sum(row['finish_reason'] == 'length' for row in rows),
            'generation_sha256': file_hash(path), 'manifest_sha256': file_hash(cache_path)}


def worker(suite, tag):
    suite = Path(suite).resolve(); job = suite / 'job' / tag; job.mkdir(parents=True, exist_ok=True)
    with (job / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'tag': tag, 'state': 'preflight', 'started_at': now()}; save(job / 'status.json', state)
        try:
            require(Path(__file__).resolve() == suite / 'source/scripts/llama_alpaca_campaign.py', 'Run the frozen worker')
            experiment = suite / 'experiment.json'; experiment_sha = file_hash(experiment)
            manifest = json.loads(experiment.read_text()); verify_inputs(suite, manifest, tag)
            runtime = runtime_preflight(suite, manifest, tag); save(job / 'runtime.json', runtime)
            command = generation_command(suite, manifest, tag); save(job / 'command.json', command)
            state.update(state='evaluating', experiment_sha256=experiment_sha, runtime_ready_at=now())
            save(job / 'status.json', state)
            subprocess.run(command, cwd=suite / 'source', check=True)
            result = validate_generation(suite, tag, manifest); verify_inputs(suite, manifest, tag)
            require(file_hash(experiment) == experiment_sha, 'Experiment changed during generation')
            state.update(state='complete', finished_at=now(), result=result)
            save(job / 'completion.json', state); save(job / 'status.json', state)
            return state
        except BaseException as error:
            state.update(state='failed', failed_at=now(), error=repr(error)); save(job / 'status.json', state)
            raise


def judge_command(suite, judge_python):
    suite = Path(suite)
    return [str(judge_python), str(suite / 'source/scripts/judge_alpaca.py'),
        '--gens-root', str(suite / 'generations'), '--refs', str(suite / 'references.jsonl'),
        '--template', str(suite / 'judge_template.txt'), '--tags', *TAGS, '--workers', '16', '--budget-cny', '30']


def judge_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
        env[name] = 'http://127.0.0.1:17891'
    env.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost', PYTHONDONTWRITEBYTECODE='1')
    return env


def validate_judgment(suite, tag, manifest):
    suite = Path(suite); directory = suite / 'generations' / tag; gen = directory / GEN_FILE
    rows = read_jsonl(gen); refs = references(suite); template = (suite / 'judge_template.txt').read_text()
    result_path = directory / 'results_judged.json'; cache_path = result_path.with_suffix('.manifest.json')
    cfg = judge.judge_result_config(gen, refs, template, 0, GEN_FILE)
    require(cache_matches(cache_path, cfg, [result_path]), 'Missing judge result cache')
    result = json.loads(result_path.read_text())
    protocol = digest({'rows': rows, 'refs': refs, 'template': template, 'judge': 'gpt-4.1',
                       'protocol': 'md5-order-logprob-v2', 'source': JUDGE_SHA})
    annotation = directory / f'annotations_{protocol[:16]}.json'
    require(result['tag'] == tag and result['judge_model'] == 'gpt-4.1'
            and result.get('limit') is None and result['judge_protocol'] == protocol
            and Path(result['annotations']).resolve() == annotation.resolve(), 'Judge result identity mismatch')
    records = judge.validated_checkpoint_records(json.loads(annotation.read_text()), protocol, rows)
    require(len(rows) == len(records) == 805 and set(records) == {digest(row) for row in rows},
            'Incomplete annotation coverage')
    require(all(records[digest(row)]['chars'] == len(row['response']) for row in rows), 'Annotation character count mismatch')
    valid = [record for record in records.values() if record['preference'] is not None]
    n = len(valid); require(result['n_judged'] == n and result['n_failed_parse'] == 805 - n, 'Judge parse counts mismatch')
    recomputed = {'weighted_win_rate': sum(row['preference'] for row in valid) / n if n else None,
                  'win_rate': sum(row['preference'] > .5 for row in valid) / n if n else None,
                  'mean_candidate_chars': sum(row['chars'] for row in valid) / n if n else None}
    for key, expected in recomputed.items():
        if expected is None: require(result[key] is None, f'Invalid empty {key}')
        else: close_number(result[key], expected, key)
    require(type(result['spent_cny']) in (float, int) and math.isfinite(result['spent_cny'])
            and result['spent_cny'] >= 0, 'Invalid cumulative spend')
    return {'tag': tag, 'n_judged': n, 'n_failed_parse': 805 - n, **recomputed,
            'cumulative_spent_cny': result['spent_cny'], 'result_sha256': file_hash(result_path),
            'result_manifest_sha256': file_hash(cache_path), 'annotations_sha256': file_hash(annotation),
            'annotations': str(annotation)}


def watch_once(suite, judge_python):
    suite = Path(suite).resolve(); directory = suite / 'judge'; directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'state.json'
    with (suite / 'judge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {'state': 'waiting'}
        # An interrupted invocation may already have incurred charges. Only an
        # explicit separate recovery procedure may ever dispatch again.
        if state['state'] in ('judge_started', 'failed'):
            return state
        try:
            experiment = suite / 'experiment.json'; experiment_sha = file_hash(experiment)
            manifest = json.loads(experiment.read_text()); verify_inputs(suite, manifest)
            if state['state'] == 'complete':
                require(state['experiment_sha256'] == experiment_sha, 'Completed experiment changed')
                completion = json.loads((suite / 'completion.json').read_text())
                for name, expected in completion['files_sha256'].items():
                    require(file_hash(suite / name) == expected, f'Completed artifact changed: {name}')
                return state
            ready = []
            for tag in TAGS:
                job = suite / 'job' / tag
                worker_state = json.loads((job / 'status.json').read_text()) if (job / 'status.json').exists() else {}
                exit_code = (job / 'exit_code').read_text().strip() if (job / 'exit_code').exists() else None
                require(worker_state.get('state') != 'failed' and exit_code in (None, '0'), f'Generation failed: {tag}')
                complete = worker_state.get('state') == 'complete' and exit_code == '0'
                if complete: require(worker_state.get('experiment_sha256') == experiment_sha, 'Worker experiment mismatch')
                ready.append(complete)
            if not all(ready):
                state.update(state='waiting', checked_at=now(), experiment_sha256=experiment_sha)
                save(state_path, state); return state
            generations = {tag: validate_generation(suite, tag, manifest) for tag in TAGS}
            require(not any((suite / 'generations').glob('*/annotations_*.json'))
                    and not any((suite / 'generations').glob('*/results_judged*.json')),
                    'Prior judge artifacts require manual inspection')
            command = judge_command(suite, judge_python); save(directory / 'command.json', command)
            state.update(state='judge_started', started_at=now(), experiment_sha256=experiment_sha,
                         generation_validation=generations, budget_cny=30, workers=16)
            save(state_path, state)
            with (directory / 'judge.log').open('w') as log:
                proc = subprocess.run(command, cwd=suite / 'source', env=judge_environment(),
                                      stdout=log, stderr=subprocess.STDOUT)
            require(proc.returncode == 0, 'Judge failed or budget guard stopped dispatch; no automatic retry')
            verify_inputs(suite, manifest)
            require(file_hash(experiment) == experiment_sha, 'Experiment changed during judging')
            require({tag: validate_generation(suite, tag, manifest) for tag in TAGS} == generations,
                    'Generations changed during judging')
            results = [validate_judgment(suite, tag, manifest) for tag in TAGS]
            combined = json.loads((suite / 'generations/judged_summary.json').read_text())
            require(set(combined) == set(TAGS) and all(combined[tag] == json.loads(
                (suite / 'generations' / tag / 'results_judged.json').read_text()) for tag in TAGS),
                'Combined judge summary mismatch')
            summary = {'status': 'complete', 'verified_at': now(), 'experiment_sha256': experiment_sha,
                'benchmark': 'Internal AlpacaEval 805, GPT-4.1 judge; not official LC',
                'n_annotations': 6 * 805, 'n_failed_parse': sum(row['n_failed_parse'] for row in results),
                'models': results, 'generation_validation': generations,
                'spent_cny_session_delta': results[-1]['cumulative_spent_cny'],
                'budget_policy': 'One judge process; 30 CNY dispatch guard, in-flight/delayed billing may exceed it.'}
            save(suite / 'summary.json', summary)
            stream = io.StringIO(); columns = ('tag', 'n_judged', 'n_failed_parse', 'weighted_win_rate', 'win_rate', 'mean_candidate_chars')
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
            atomic_text(suite / 'summary.csv', stream.getvalue())
            artifacts = [suite / name for name in ('summary.json', 'summary.csv', 'generations/judged_summary.json')]
            artifacts.extend(path for path in (suite / 'generations').rglob('*') if path.is_file())
            save(suite / 'completion.json', {'status': 'complete', 'verified_at': now(),
                'experiment_sha256': experiment_sha, 'files_sha256': {str(path.relative_to(suite)): file_hash(path) for path in artifacts}})
            state.update(state='complete', finished_at=now()); save(state_path, state)
            return state
        except BaseException as error:
            state.update(state='failed', failed_at=now(), error=repr(error)); save(state_path, state)
            if isinstance(error, (KeyboardInterrupt, SystemExit)): raise
            return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    work = sub.add_parser('worker'); work.add_argument('--suite', required=True, type=Path); work.add_argument('--tag', choices=TAGS, required=True)
    watch = sub.add_parser('watch'); watch.add_argument('--suite', required=True, type=Path)
    watch.add_argument('--judge-python', default='/root/.venvs/alpacaeval/bin/python')
    args = parser.parse_args(argv)
    if args.command == 'worker': worker(args.suite, args.tag); return 0
    while True:
        state = watch_once(args.suite, args.judge_python)
        print(json.dumps(state), flush=True)
        if state['state'] != 'waiting': return 0 if state['state'] == 'complete' else 1
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
