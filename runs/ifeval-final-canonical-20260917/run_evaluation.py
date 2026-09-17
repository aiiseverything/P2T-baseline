"""Run the six fixed IFEval policies and verify their saved rule scores."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime

SUITE = Path(__file__).resolve().parent
SOURCE = SUITE / 'source'
EXPECTED_TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
sys.path[:0] = [str(SOURCE), str(SUITE / '.ifeval-extra'),
                str(SOURCE / '.vllm-extra'), str(SOURCE / 'third_party/ifeval')]
os.environ['NLTK_DATA'] = str(SOURCE / 'third_party/ifeval/nltk_data')

from scripts.eval_artifacts import atomic_text, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head
from scripts.watch_final_humaneval import final_checkpoint_ready


def verify_inputs(manifest):
    assert tuple(manifest['models']) == EXPECTED_TAGS, 'Expected exactly the six requested models in fixed order'
    for relative, expected in manifest['files_sha256'].items():
        assert file_hash(SUITE / relative) == expected, relative
    data = [json.loads(line) for line in (SUITE / 'input_data.jsonl').read_text().splitlines() if line.strip()]
    assert len(data) == len({r['key'] for r in data}) == len({r['prompt'] for r in data}) == 541
    assert sum(len(r['instruction_id_list']) for r in data) == 834
    for tag, item in manifest['models'].items():
        assert resolve_policy_head(item['adapter'], 'float32') == item['policy'], tag
        for path, expected in item['files_sha256'].items():
            assert file_hash(path) == expected, (tag, path)
        if tag in ('grpo', 'lam2', 'lam4', 'lam8'):
            assert str(final_checkpoint_ready(Path(manifest['training_suite']) / tag)) == item['adapter'], tag
    versions = {p: importlib.metadata.version(p) for p in manifest['scoring_versions']}
    assert versions == manifest['scoring_versions'], versions
    return data


def validate_results(manifest, data):
    assert tuple(manifest['models']) == EXPECTED_TAGS
    summaries = {}
    for tag, item in manifest['models'].items():
        directory = SUITE / 'results' / tag
        result_path = directory / 'results_t1.0_n1.json'
        gen_path = directory / 'generations_t1.0_n1.jsonl'
        cache = json.loads((directory / 'manifest_t1.0_n1.json').read_text())
        assert cache['outputs'] == {p.name: file_hash(p) for p in (result_path, gen_path)}, tag
        config = cache['config']
        assert config['policy'] == item['policy'], tag
        assert config['scoring'] == {'protocol': 'official_seeded_v1', 'seed': 42, 'langdetect_seed': 42}, tag
        assert config['recipe'] == {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}, tag
        assert config['seed'] == 42 and config['max_tokens'] == 2048, tag
        assert config['dataset'] == fingerprint(SUITE / 'input_data.jsonl'), tag
        assert config['model']['path'] == manifest['base_model'], tag
        if item['adapter'] == 'none':
            assert config['adapter'] is None, tag
        else:
            assert config['adapter']['path'] == item['adapter'], tag
            files = {row['name']: row for row in config['adapter']['files']}
            assert files['adapter_model.safetensors']['sha256'] == item['files_sha256'][str(Path(item['adapter']) / 'adapter_model.safetensors')], tag
        generated = [json.loads(line) for line in gen_path.read_text().splitlines() if line.strip()]
        assert len(generated) == 541, tag
        for expected, actual in zip(data, generated):
            assert (expected['key'], expected['prompt']) == (actual['key'], actual['prompt']), tag
            assert all(len(actual[k]) == 1 for k in ('responses', 'response_tokens', 'finish_reason', 'stop_reason', 'last_token_id')), tag
            assert 0 <= actual['response_tokens'][0] <= 2048, tag
            assert actual['finish_reason'][0] in ('stop', 'length'), tag
        result = json.loads(result_path.read_text())
        assert result['tag'] == tag and result['adapter'] == item['adapter'], tag
        details = result['details']
        assert len(details) == 541, tag
        for expected, detail in zip(data, details):
            assert detail['key'] == expected['key'] and detail['sample'] == 0, tag
            for prefix in ('strict', 'loose'):
                flags = detail[prefix + '_list']
                assert len(flags) == len(expected['instruction_id_list']), tag
                assert all(isinstance(flag, bool) for flag in flags), tag
                assert detail[prefix + '_all'] == all(flags), tag
        metrics = {}
        for prefix in ('strict', 'loose'):
            metrics['prompt_' + prefix] = sum(d[prefix + '_all'] for d in details) / 541
            metrics['inst_' + prefix] = sum(sum(d[prefix + '_list']) for d in details) / 834
        for key, value in metrics.items():
            assert abs(value - result[key]) < 1e-12, (tag, key)
        mean_tokens = sum(row['response_tokens'][0] for row in generated) / 541
        assert abs(mean_tokens - result['response_length_mean']) < 1e-12, tag
        summaries[tag] = dict(metrics, prompts=541, instructions=834, mean_tokens=mean_tokens,
                              capped=sum(row['finish_reason'][0] == 'length' for row in generated),
                              result=str(result_path))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    data = verify_inputs(manifest)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    random.seed(42)
    if args.preflight:
        import nltk
        assert nltk.sent_tokenize('One sentence. Another sentence.') == ['One sentence.', 'Another sentence.']
        print('Preflight passed: six model identities, frozen source/dependencies, 541 prompts, FP32 policies, NLTK resources.', flush=True)
        return
    if args.validate_only:
        print(json.dumps(validate_results(manifest, data), indent=2), flush=True)
        return
    runtime = {p: importlib.metadata.version(p) for p in manifest['gpu_versions']}
    assert runtime == manifest['gpu_versions'], runtime
    atomic_text(SUITE / 'job/runtime.json', json.dumps(runtime, indent=2))
    command = ['eval_ifeval', '--model', manifest['base_model'], '--dataset', str(SUITE / 'input_data.jsonl'),
               '--output', str(SUITE / 'results'), '--recipes', '1.0:1:1.0:-1', '--max-tokens', '2048',
               '--seed', '42', '--policy-head-dtype', 'float32', '--adapters']
    command += [f"{tag}={item['adapter']}" for tag, item in manifest['models'].items()]
    atomic_text(SUITE / 'job/command.json', json.dumps(command, indent=2))
    state_path = SUITE / 'status.json'
    state = json.loads(state_path.read_text())
    state.update(state='running', started_at=datetime.now().astimezone().isoformat())
    atomic_text(state_path, json.dumps(state, indent=2))
    try:
        from scripts import eval_ifeval
        sys.argv = command
        eval_ifeval.main()
        summaries = validate_results(manifest, data)
        atomic_text(SUITE / 'summary.json', json.dumps(summaries, indent=2))
        state.update(state='complete', finished_at=datetime.now().astimezone().isoformat(), models=summaries)
        atomic_text(SUITE / 'completion.json', json.dumps(state, indent=2))
        print('All six IFEval results verified.', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), failed_at=datetime.now().astimezone().isoformat())
        raise
    finally:
        atomic_text(state_path, json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
