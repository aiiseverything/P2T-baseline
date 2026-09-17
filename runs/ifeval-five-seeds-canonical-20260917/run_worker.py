"""One immutable policy, one H200 engine, five independently seeded IFEval runs."""
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
TAGS = ('sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
SEEDS = [42, 43, 44, 45, 46]
RECIPE = {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}
sys.path.insert(0, str(SOURCE))
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint, digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    # Unicode line/paragraph separators inside JSON strings are ordinary data.
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dataset_rows(manifest):
    rows = read_jsonl(manifest['dataset'])
    require(len(rows) == len({r['key'] for r in rows}) == len({r['prompt'] for r in rows}) == 541,
            'Expected 541 unique IFEval keys and prompts')
    require(sum(len(r['instruction_id_list']) for r in rows) == 834,
            'Expected exactly 834 IFEval instructions')
    return rows


def generation_command(manifest, tag):
    require(tag in TAGS and tag in manifest['models'], 'Unknown IFEval policy')
    require(manifest['seeds'] == SEEDS and manifest['scoring_seed'] == 42,
            'Expected generation seeds 42..46 and fixed scoring seed 42')
    return [sys.executable, str(SOURCE / 'scripts/eval_ifeval.py'),
            '--model', manifest['base_model'], '--dataset', manifest['dataset'],
            '--output', str(SUITE / 'results'), '--recipes', '1.0:1:1.0:-1',
            '--max-tokens', '2048', '--seeds', *map(str, manifest['seeds']),
            '--scoring-seed', '42', '--policy-head-dtype', 'float32',
            '--adapters', f"{tag}={manifest['models'][tag]['adapter']}"]


def close_number(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12, f'Incorrect {label}')


def validate_seed_result(output, tag, seed, manifest):
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
    require(config['adapter']['path'] == item['adapter'], 'Result adapter path mismatch')
    files = {row['name']: row for row in config['adapter']['files']}
    for path, expected in item['files_sha256'].items():
        relative = Path(path).relative_to(item['adapter']) if Path(path).is_relative_to(item['adapter']) else None
        if relative is not None:
            require(files.get(str(relative), {}).get('sha256') == expected,
                    f'Result adapter fingerprint mismatch: {relative}')
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


def main():
    tag = sys.argv[1]
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    require(set(manifest['models']) == set(TAGS) and tag in TAGS, 'Expected the five requested policies')
    command = generation_command(manifest, tag)
    job = SUITE / 'job' / tag
    job.mkdir(parents=True, exist_ok=True)
    state = {'tag': tag, 'state': 'preflight', 'started_at': now()}
    save(job / 'status.json', state)
    try:
        for relative, expected in manifest['files_sha256'].items():
            require(file_hash(SUITE / relative) == expected, f'Frozen experiment file changed: {relative}')
        for path, expected in manifest['models'][tag]['files_sha256'].items():
            require(file_hash(path) == expected, f'Model input changed: {path}')
        require(fingerprint(manifest['base_model'], full_weights=False) == manifest['base_model_fingerprint'],
                'Base model fingerprint changed')
        require(file_hash(manifest['dataset']) == manifest['dataset_sha256'], 'Dataset changed')
        data = dataset_rows(manifest)
        from scripts.eval_policy import resolve_policy_head
        from scripts.eval_artifacts import validate_adapter_base
        item = manifest['models'][tag]
        validate_adapter_base(item['adapter'], manifest['base_model'])
        require(resolve_policy_head(item['adapter'], 'float32') == item['policy'], 'Actor precision/provenance changed')
        versions = {}
        for group in ('gpu_versions', 'scoring_versions'):
            versions[group] = {name: importlib.metadata.version(name) for name in manifest[group]}
            require(versions[group] == manifest[group], f'Runtime version drift: {group}')
        modules = {name: importlib.import_module(name) for name in ('peft', 'pyarrow', 'pandas', 'nltk', 'langdetect')}
        require(modules['nltk'].sent_tokenize('One sentence. Another sentence.') == ['One sentence.', 'Another sentence.'],
                'Frozen NLTK resource smoke check failed')
        from instruction_following_eval import evaluation_lib
        import torch
        require(torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0),
                'This protocol requires exactly one H200 per rjob')
        from transformers import AutoTokenizer
        from vpo_rm.trainer import VPOTrainer
        tokenizer = AutoTokenizer.from_pretrained(manifest['base_model'], trust_remote_code=True)
        lengths = [len(tokenizer.encode(VPOTrainer._render_chat_prompt(tokenizer, row['prompt']),
                    add_special_tokens=False)) for row in data]
        require(min(lengths) >= 1 and max(lengths) + 2048 <= 4096, 'Full prompt and output budget exceed context')
        runtime = dict(versions, python=sys.version, checked_at=now(), gpu=torch.cuda.get_device_name(0),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), free_total_bytes=list(torch.cuda.mem_get_info(0)),
            dataset_digest=digest(data), max_prompt_tokens=max(lengths),
            dependency_modules={name: module.__file__ for name, module in modules.items()},
            official_scoring_module=evaluation_lib.__file__, nltk_data=os.environ.get('NLTK_DATA'))
        save(job / 'runtime.json', runtime)
        save(job / 'command.json', command)
        state.update(state='evaluating', runtime_ready_at=now(), seeds=manifest['seeds'], scoring_seed=42)
        save(job / 'status.json', state)
        print(json.dumps({'runtime': runtime, 'command': command}), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)
        results = {str(seed): validate_seed_result(SUITE / 'results', tag, seed, manifest) for seed in SEEDS}
        state.update(state='complete', finished_at=now(), seeds=SEEDS,
                     evaluations=5, total_prompts=2705, results=results)
        save(job / 'completion.json', state)
        save(job / 'status.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(state='failed', failed_at=now(), error=repr(error))
        save(job / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
