#!/usr/bin/env python3
"""Run and verify the frozen random-credit (VPO lambda-4, independent random direction) evaluation on one H200.

Single policy: the step-250 checkpoint of runs/rl-ablation-random-credit-20260919
(credit_source=random_direction), evaluated under the canonical Qwen3-14B-Base
protocol so every number is directly comparable with the published
rl-fp32-is-canonical-20260917 arms: prompts render with the base model's own
tokenizer exactly as the canonical evaluation scripts did, stops are
endoftext/im_end, and there is no BOS.

Validation follows the Llama-base and Qwen-instruct workers. Base/RM weights use
frozen filesystem fingerprints; adapter files use full SHA256. No model weights
are loaded by this worker's CPU preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = SOURCE
sys.path.insert(0, str(SOURCE))
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint, digest

TAGS = ('randdir',)
ADAPTER_FREE = ()
STEP0 = ()
SEEDS = [42, 43, 44, 45, 46]
# Qwen3-14B-Base: no BOS, native <|endoftext|> pad/eos, endoftext+im_end stops —
# exactly the protocol the canonical evaluation scripts rendered with.
STOPS = [151643, 151645]
PAD = 151643
EOS = 151643
VOCAB = 151936
RECIPE = {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}
REWARD_METRIC = 'raw_skywork_scalar_no_length_or_kl_penalty'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    # U+2028/U+2029 inside JSON strings are data, not record separators.
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_manifest(manifest, tag):
    require(set(manifest['models']) == set(TAGS) and tag in TAGS,
            'Expected only the random-credit VPO lambda-4 policy')
    benchmark = manifest['benchmark']
    require(benchmark in ('reward256', 'ifeval5'), 'Unknown benchmark')
    require(manifest['seeds'] == ([42] if benchmark == 'reward256' else SEEDS)
            and manifest['scoring_seed'] == 42, 'Generation/scoring seed protocol changed')
    require(manifest['expected_stop_token_ids'] == STOPS and manifest['actor_pad_token_id'] == PAD,
            'Qwen base stop/pad protocol changed')
    require(isinstance(manifest['prompt_token_ids_sha256'], str)
            and len(manifest['prompt_token_ids_sha256']) == 64, 'Missing frozen prompt ID digest')
    for name in ('base_model', 'reward_model', 'tokenizer', 'dataset'):
        require(Path(manifest[name]).is_absolute(), f'Expected absolute {name} path')
    require(manifest['tokenizer'] == manifest['base_model'],
            'The canonical evaluation renders with the base model tokenizer')
    for label, item in manifest['models'].items():
        require(item['adapter'] != 'none', 'The random-credit policy must bind its checkpoint adapter')
        require(Path(item['adapter']).name in ('step-250', 'checkpoint-250'), 'Evaluation requires final step 250')
        require(item['policy']['policy_head_dtype'] == 'float32', 'The actor head must use FP32')


def generation_command(suite, manifest, tag):
    validate_manifest(manifest, tag)
    suite = Path(suite)
    common = ['--model', manifest['base_model'], '--tokenizer', manifest['tokenizer'],
              '--max-tokens', '2048', '--policy-head-dtype', 'float32']
    adapter = manifest['models'][tag]['adapter']
    if manifest['benchmark'] == 'reward256':
        return [sys.executable, str(suite / 'source/scripts/eval_checkpoints.py'), *common,
            '--run', f'{tag}={adapter}', '--rm', manifest['reward_model'],
            '--dataset-path', manifest['dataset'], '--num-prompts', '256', '--temps', '1.0',
            '--min-tokens', '0', '--max-num-seqs', '32', '--rm-microbatch', '1',
            '--rm-device', 'cuda:0', '--seed', '42', '--output', str(suite / 'results' / tag)]
    return [sys.executable, str(suite / 'source/scripts/eval_ifeval.py'), *common,
        '--dataset', manifest['dataset'], '--output', str(suite / 'results'),
        '--recipes', '1.0:1:1.0:-1', '--seeds', *map(str, SEEDS), '--scoring-seed', '42',
        '--adapters', f'{tag}={adapter}']


def child_environment(environment=None):
    result = dict(os.environ if environment is None else environment)
    result.update(EVAL_TOPP='1.0', EVAL_PP='0.0', PYTHONDONTWRITEBYTECODE='1')
    return result


def dataset_rows(manifest):
    rows = read_jsonl(manifest['dataset'])
    require(len(rows) == len({r['key'] for r in rows}) == len({r['prompt'] for r in rows}) == 541,
            'Expected 541 unique IFEval keys and prompts')
    require(sum(len(r['instruction_id_list']) for r in rows) == 834,
            'Expected exactly 834 IFEval instructions')
    return rows


def close_number(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12, f'Incorrect {label}')


def validate_tokenizer_result(config, manifest):
    require(config.get('tokenizer') == {'source': manifest['tokenizer'],
            'fingerprint': manifest['tokenizer_fingerprint']}, 'Result tokenizer identity mismatch')
    require(config.get('prompt_token_ids_sha256') == manifest['prompt_token_ids_sha256'],
            'Result prompt token IDs changed')


def validate_adapter_result(value, item):
    if item['adapter'] == 'none':
        require(value is None, 'Base must have a null adapter fingerprint')
        return
    require(isinstance(value, dict) and value.get('path') == item['adapter'], 'Result adapter path mismatch')
    files = {row['name']: row for row in value['files']}
    for path, expected in item['files_sha256'].items():
        if Path(path).is_relative_to(item['adapter']):
            relative = str(Path(path).relative_to(item['adapter']))
            require(files.get(relative, {}).get('sha256') == expected,
                    f'Result adapter fingerprint mismatch: {relative}')


def validate_reward_result(output, tag, manifest):
    validate_manifest(manifest, tag)
    require(manifest['benchmark'] == 'reward256', 'Expected reward256 manifest')
    output = Path(output)
    files = [output / name for name in ('eval_prompts.json', 'eval.jsonl', 'summary.json', 'generations.jsonl')]
    cache = json.loads((output / 'manifest.json').read_text())
    require(cache['outputs'] == {p.name: file_hash(p) for p in files}, 'Incomplete or changed reward output manifest')
    cfg = cache['config']; item = manifest['models'][tag]
    step = 250
    require(cfg['policy_head_dtype'] == 'float32' and cfg['policies'] == {f'{tag}/{step}': item['policy']}
            and cfg['top_p'] == 1.0 and cfg['top_k'] == -1 and cfg['samples_per_prompt'] == 1
            and cfg['presence_penalty'] == 0.0 and cfg['reward_input_protocol'] == 'canonical_chat_v1'
            and cfg['reward_metric'] == REWARD_METRIC, 'Result reward protocol mismatch')
    expected_args = dict(seed=42, num_prompts=256, temps=[1.0], max_tokens=2048, min_tokens=0,
        max_num_seqs=32, rm_microbatch=1, rm_device='cuda:0', model=manifest['base_model'],
        rm=manifest['reward_model'], tokenizer=manifest['tokenizer'], dataset_path=manifest['dataset'])
    require(all(cfg['args'].get(k) == v for k, v in expected_args.items()), 'Result generation/RM arguments changed')
    require(cfg['model'] == manifest['base_model_fingerprint'] and cfg['rm'] == manifest['reward_model_fingerprint']
            and cfg['dataset'] == fingerprint(manifest['dataset']), 'Result model or dataset identity mismatch')
    validate_tokenizer_result(cfg, manifest)
    require(set(cfg['adapters']) == {f'{tag}/{step}'}, 'Unexpected adapter result coverage')
    validate_adapter_result(cfg['adapters'][f'{tag}/{step}'], item)
    prompts = json.loads((output / 'eval_prompts.json').read_text())
    require(len(prompts) == 256 and prompts == cfg['prompts'], 'Reward prompt identity failure')
    rows = read_jsonl(output / 'eval.jsonl'); generations = read_jsonl(output / 'generations.jsonl')
    require(len(rows) == len(generations) == 256, 'Incomplete reward coverage')
    for i, (row, generated) in enumerate(zip(rows, generations)):
        identity = dict(run=tag, step=step, temp=1.0, prompt=i)
        require(all(row.get(k) == v and generated.get(k) == v for k, v in identity.items()), 'Reward row identity mismatch')
        require(type(row['score']) in (float, int) and math.isfinite(row['score']), 'Nonfinite reward')
        ids = generated['token_ids']
        require(isinstance(ids, list) and all(type(t) is int and 0 <= t < VOCAB for t in ids)
                and type(row['response_tokens']) is int and row['response_tokens'] == len(ids)
                and 0 <= len(ids) <= 2048, 'Reward response token coverage mismatch')
    summary = json.loads((output / 'summary.json').read_text())
    require(set(summary) == {tag} and set(summary[tag]) == {str(step)}
            and set(summary[tag][str(step)]) == {'1.0'}, 'Unexpected reward summary coverage')
    values = summary[tag][str(step)]['1.0']
    require(values['n'] == 256 and values['reward_metric'] == REWARD_METRIC, 'Reward summary protocol mismatch')
    close_number(values['mean'], sum(r['score'] for r in rows) / 256, 'mean reward')
    close_number(values['mean_response_tokens'], sum(r['response_tokens'] for r in rows) / 256, 'mean reward length')
    close_number(values['at_token_cap_rate'], sum(r['response_tokens'] >= 2048 for r in rows) / 256, 'reward cap rate')
    require(isinstance(values['ci95'], list) and len(values['ci95']) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in values['ci95'])
            and values['ci95'][0] <= values['ci95'][1], 'Invalid reward confidence interval')
    return dict(values, tag=tag, step=step, seed=42, n_scores=256, summary=summary,
                result=str(output / 'summary.json'), result_sha256=file_hash(output / 'summary.json'),
                generation_sha256=file_hash(output / 'generations.jsonl'))


def validate_seed_result(output, tag, seed, manifest):
    validate_manifest(manifest, tag)
    require(manifest['benchmark'] == 'ifeval5', 'Expected ifeval5 manifest')
    require(tag in TAGS and tag in manifest['models'] and seed in SEEDS,
            'Unexpected result policy or generation seed')
    directory = Path(output) / f'seed-{seed}' / tag
    data = dataset_rows(manifest)
    item = manifest['models'][tag]
    result_path = directory / 'results_t1.0_n1.json'
    gen_path = directory / 'generations_t1.0_n1.jsonl'
    cache = json.loads((directory / 'manifest_t1.0_n1.json').read_text())
    require(cache['outputs'] == {p.name: file_hash(p) for p in (result_path, gen_path)},
            f'Incomplete or changed output manifest: {tag}/{seed}')
    config = cache['config']
    require(config['policy'] == item['policy'], 'Result policy identity mismatch')
    require(config['scoring'] == {'protocol': 'official_seeded_v1', 'seed': 42, 'langdetect_seed': 42},
            'Scoring seed/protocol mismatch')
    require(config['recipe'] == RECIPE and config['seed'] == seed
            and config['max_tokens'] == 2048 and config['engine']['seed'] == 42
            and config['engine']['dtype'] == 'bfloat16' and config['engine']['max_model_len'] == 4096,
            'Generation seed/engine/recipe mismatch')
    require(config['dataset'] == fingerprint(manifest['dataset']), 'Result dataset fingerprint mismatch')
    require(config['model'] == manifest['base_model_fingerprint'], 'Result base model fingerprint mismatch')
    validate_adapter_result(config['adapter'], item)
    validate_tokenizer_result(config, manifest)
    require(config['stop_token_ids'] == STOPS, 'Qwen stop IDs changed')
    generated = read_jsonl(gen_path)
    require(len(generated) == 541, 'Incomplete generated prompt coverage')
    for expected, actual in zip(data, generated):
        require((expected['key'], expected['prompt']) == (actual['key'], actual['prompt']),
                'Generated key/prompt identity mismatch')
        require(all(isinstance(actual[k], list) and len(actual[k]) == 1 for k in
                    ('responses', 'response_tokens', 'finish_reason', 'stop_reason', 'last_token_id')),
                'Expected exactly one generated response per prompt')
        require(isinstance(actual['responses'][0], str)
                and type(actual['response_tokens'][0]) is int and 0 <= actual['response_tokens'][0] <= 2048
                and actual['finish_reason'][0] in ('stop', 'length'), 'Invalid generation metadata')
    result = json.loads(result_path.read_text())
    require(result['tag'] == tag and result['adapter'] == item['adapter']
            and result['seed'] == seed and result['recipe'] == RECIPE and result['max_tokens'] == 2048,
            'Result policy/seed identity mismatch')
    details = result['details']
    require(len(details) == 541, 'Incomplete scored prompt coverage')
    for expected, detail in zip(data, details):
        require(detail['key'] == expected['key'] and detail['sample'] == 0,
                'Score key/sample identity mismatch')
        for prefix in ('strict', 'loose'):
            flags = detail[prefix + '_list']
            require(isinstance(flags, list) and len(flags) == len(expected['instruction_id_list'])
                    and all(type(flag) is bool for flag in flags), 'Incomplete instruction flags')
            require(type(detail[prefix + '_all']) is bool and detail[prefix + '_all'] == all(flags),
                    'Prompt pass does not match instruction flags')
    metrics = {}
    for prefix in ('strict', 'loose'):
        metrics['prompt_' + prefix] = sum(d[prefix + '_all'] for d in details) / 541
        metrics['inst_' + prefix] = sum(sum(d[prefix + '_list']) for d in details) / 834
    for key, expected in metrics.items():
        close_number(result[key], expected, key)
    lengths = sorted(row['response_tokens'][0] for row in generated)
    mean_tokens = sum(lengths) / 541
    close_number(result['response_length_mean'], mean_tokens, 'mean response length')
    close_number(result['response_length_p95'], lengths[int(.95 * 541)], 'p95 response length')
    return dict(metrics, seed=seed, scoring_seed=42, prompts=541, instructions=834,
                mean_tokens=mean_tokens, capped=sum(row['finish_reason'][0] == 'length' for row in generated),
                result=str(result_path), result_sha256=file_hash(result_path),
                generation_sha256=file_hash(gen_path))


def validate_training_identity(manifest, tag):
    """Bind the checkpoint to the random-credit ablation of canonical lambda-4."""
    item = manifest['models'][tag]
    training = json.loads((Path(item['adapter']) / 'run_manifest.json').read_text())
    config = training['resolved_config']
    require(training['step'] == 250 and config['model_name'] == manifest['base_model']
            and config['policy_head_dtype'] == 'float32' and config['method'] == 'vpo_rm'
            and config['credit_lambda'] == 4.0
            and config.get('credit_source') == 'random_direction',
            'Expected the step-250 random-credit VPO lambda-4 ablation with an FP32 head')
    require(config['init_adapter'].endswith('sft-native-eos-clean2k5e2'),
            'Expected the canonical native-EOS SFT initialization')


def verify_inputs(suite, manifest, tag):
    validate_manifest(manifest, tag)
    suite = Path(suite).resolve()
    entry = 'eval_checkpoints.py' if manifest['benchmark'] == 'reward256' else 'eval_ifeval.py'
    required = {'source/scripts/' + name for name in ('randdir_eval_worker.py', entry, 'eval_artifacts.py', 'eval_policy.py')}
    required.update('source/vpo_rm/' + name for name in ('trainer.py', 'token_policy.py', 'integration.py',
                    'alignment.py', 'model_identity.py', 'reward.py', 'reward_inputs.py'))
    require(required <= set(manifest['files_sha256']), 'Missing frozen source bindings')
    for relative, expected in manifest['files_sha256'].items():
        path = suite / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(suite), 'Frozen file escapes suite')
        require(file_hash(path) == expected, f'Frozen experiment file changed: {relative}')
    for path, expected in manifest['models'][tag]['files_sha256'].items():
        require(file_hash(path) == expected, f'Model input changed: {path}')
    for name in ('base_model', 'reward_model', 'tokenizer'):
        require(fingerprint(manifest[name], full_weights=False) == manifest[name + '_fingerprint'],
                f'{name} fingerprint changed')
    require(file_hash(manifest['dataset']) == manifest['dataset_sha256'], 'Dataset changed')
    validate_training_identity(manifest, tag)


def validate_prompt_protocol(tokenizer, prompts, manifest):
    from vpo_rm.trainer import VPOTrainer
    from vpo_rm.token_policy import get_stop_token_ids, tokenize_rendered_prompts
    require(tokenizer.bos_token_id is None and tokenizer.pad_token_id == PAD
            and tokenizer.eos_token_id == EOS and list(get_stop_token_ids(tokenizer)) == STOPS,
            'Actor tokenizer native BOS/pad/EOS/stop protocol changed')
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, prompt) for prompt in prompts]
    rows = [row['prompt_token_ids'] for row in tokenize_rendered_prompts(tokenizer, rendered)]
    # Qwen3 prepends no BOS and its chat template appends an empty reasoning block.
    require(all(row and 151643 not in row for row in rows), 'Qwen prompts must contain no BOS/endoftext token')
    require(all(t.endswith('<think>\n\n</think>\n\n') for t in rendered), 'Qwen3 empty-reasoning suffix missing')
    require(max(map(len, rows)) + 2048 <= 4096, 'Full prompt and uniform output budget exceed context')
    require(digest(rows) == manifest['prompt_token_ids_sha256'], 'Frozen prompt token IDs changed')
    return rows


def validate_runtime_versions(manifest):
    required = {'torch', 'transformers', 'vllm', 'peft', 'pyarrow', 'pandas'}
    require(required <= set(manifest.get('gpu_versions', {})), 'Missing GPU runtime version bindings')
    if manifest['benchmark'] == 'ifeval5':
        require({'absl-py', 'immutabledict', 'langdetect', 'nltk', 'click', 'joblib', 'tqdm', 'six'}
                <= set(manifest.get('scoring_versions', {})), 'Missing scoring runtime version bindings')
    versions = {}
    for group in ('gpu_versions', 'scoring_versions'):
        expected = manifest.get(group, {})
        versions[group] = {name: importlib.metadata.version(name) for name in expected}
        require(versions[group] == expected, f'Runtime version drift: {group}')
    return versions


def runtime_preflight(suite, manifest, tag):
    from scripts.eval_policy import resolve_policy_head
    from scripts.eval_artifacts import validate_adapter_base
    item = manifest['models'][tag]
    validate_adapter_base(item['adapter'], manifest['base_model'])
    require(resolve_policy_head(item['adapter'], 'float32') == item['policy'], 'Actor precision/provenance changed')
    versions = validate_runtime_versions(manifest)
    modules = {name: importlib.import_module(name) for name in ('peft', 'pyarrow', 'pandas')}
    extra = {}
    if manifest['benchmark'] == 'ifeval5':
        modules.update({name: importlib.import_module(name) for name in ('nltk', 'langdetect')})
        require(modules['nltk'].sent_tokenize('One sentence. Another sentence.') == ['One sentence.', 'Another sentence.'],
                'Frozen NLTK resource smoke check failed')
        sys.path.insert(0, str(Path(suite) / 'source/third_party/ifeval'))
        from instruction_following_eval import evaluation_lib
        extra.update(official_scoring_module=evaluation_lib.__file__, nltk_data=os.environ.get('NLTK_DATA'))
        data = dataset_rows(manifest)
        prompts = [row['prompt'] for row in data]
    else:
        from scripts.eval_checkpoints import load_validation_prompts
        prompts = load_validation_prompts(manifest['dataset'], 256)
        require(prompts == json.loads((Path(suite) / 'eval_prompts.json').read_text()), 'Frozen reward 256 prompts changed')
    import torch
    require(torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0),
            'This protocol requires exactly one H200 per rjob')
    from vpo_rm.token_policy import load_actor_tokenizer
    tokenizer = load_actor_tokenizer(manifest['base_model'], tokenizer_name=manifest['tokenizer'])
    rows = validate_prompt_protocol(tokenizer, prompts, manifest)
    return dict(versions, python=sys.version, checked_at=now(), gpu=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), free_total_bytes=list(torch.cuda.mem_get_info(0)),
        prompts_digest=digest(prompts), prompt_token_ids_sha256=digest(rows), max_prompt_tokens=max(map(len, rows)),
        dependency_modules={name: module.__file__ for name, module in modules.items()},
        weight_identity='Adapter files: full SHA256; base/RM weights: frozen size/inode/mtime/ctime fingerprints.', **extra)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--tag', required=True, choices=TAGS)
    args = parser.parse_args(argv)
    suite, tag = args.suite.resolve(), args.tag
    job = suite / 'job' / tag
    job.mkdir(parents=True, exist_ok=True)
    with (job / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(tag=tag, state='preflight', started_at=now())
        save(job / 'status.json', state)
        try:
            require(Path(__file__).resolve() == suite / 'source/scripts/randdir_eval_worker.py',
                    'Execute the frozen worker inside this suite/source')
            experiment = suite / 'experiment.json'
            experiment_sha = file_hash(experiment)
            manifest = json.loads(experiment.read_text())
            verify_inputs(suite, manifest, tag)
            command = generation_command(suite, manifest, tag)
            runtime = runtime_preflight(suite, manifest, tag)
            save(job / 'runtime.json', runtime)
            save(job / 'command.json', command)
            state.update(state='evaluating', runtime_ready_at=now(), experiment_sha256=experiment_sha,
                         benchmark=manifest['benchmark'], seeds=manifest['seeds'], scoring_seed=42)
            save(job / 'status.json', state)
            subprocess.run(command, cwd=suite / 'source', env=child_environment(), check=True)
            if manifest['benchmark'] == 'reward256':
                results = validate_reward_result(suite / 'results' / tag, tag, manifest)
                require(json.loads((suite / 'results' / tag / 'eval_prompts.json').read_text())
                        == json.loads((suite / 'eval_prompts.json').read_text()), 'Output reward prompt identity changed')
                state.update(n_scores=256, evaluations=1, total_prompts=256, results=results, summary=results['summary'])
            else:
                results = {str(seed): validate_seed_result(suite / 'results', tag, seed, manifest) for seed in SEEDS}
                state.update(evaluations=5, total_prompts=2705, results=results)
            verify_inputs(suite, manifest, tag)
            require(file_hash(experiment) == experiment_sha, 'Experiment changed during evaluation')
            state.update(state='complete', finished_at=now())
            save(job / 'completion.json', state)
            save(job / 'status.json', state)
            print(json.dumps(state, allow_nan=False), flush=True)
            return state
        except BaseException as error:
            state.update(state='failed', failed_at=now(), error=repr(error))
            save(job / 'status.json', state)
            raise


if __name__ == '__main__':
    main()
