"""Run and validate six frozen policies on the fixed Arena-Hard v2 hard500."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys

SUITE = Path(__file__).resolve().parent
SOURCE = SUITE / 'source'
sys.path[:0] = [str(SOURCE), str(SOURCE / '.vllm-extra')]
from scripts.eval_artifacts import atomic_text, file_hash, fingerprint
from scripts.eval_policy import resolve_policy_head
from scripts.watch_final_humaneval import final_checkpoint_ready

TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def verify_inputs(manifest):
    assert tuple(manifest['models']) == TAGS
    for relative, expected in manifest['files_sha256'].items():
        assert file_hash(SUITE / relative) == expected, relative
    data = [json.loads(line) for line in (SUITE / 'question.jsonl').open()]
    assert len(data) == len({q['uid'] for q in data}) == 500
    assert all(q['category'] == 'hard_prompt' for q in data)
    assert fingerprint(manifest['base_model'], full_weights=False) == manifest['base_identity']
    for tag, item in manifest['models'].items():
        assert resolve_policy_head(item['adapter'], 'float32') == item['policy'], tag
        for path, expected in item['files_sha256'].items():
            assert file_hash(path) == expected, (tag, path)
        if tag in TAGS[2:]:
            assert str(final_checkpoint_ready(Path(manifest['training_suite']) / tag)) == item['adapter'], tag
    return data


def validate_results(manifest, questions):
    summaries = {}
    for tag, item in manifest['models'].items():
        path = SUITE / 'model_answer' / f'{tag}.jsonl'
        cache = json.loads((SUITE / 'manifests' / f'{tag}.json').read_text())
        assert cache['outputs'] == {path.name:file_hash(path)}, tag
        config = cache['config']
        assert config['model_tag'] == tag and config['policy'] == item['policy'], tag
        assert config['dataset'] == fingerprint(SUITE/'question.jsonl'), tag
        assert config['model'] == manifest['base_identity'], tag
        assert config['recipe'] == {'temp':1.0,'n':1,'top_p':1.0,'top_k':-1}, tag
        assert config['seed'] == 42 and config['max_tokens'] == 4096, tag
        assert config['engine']['max_model_len'] == 16384, tag
        assert config['engine']['hf_overrides'] == {'head_dtype':'float32'}, tag
        if item['adapter'] == 'none':
            assert config['adapter'] is None, tag
        else:
            assert config['adapter'] == fingerprint(item['adapter']), tag
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == 500, tag
        lengths = []
        for q, row in zip(questions, rows):
            assert (row['uid'], row['model']) == (q['uid'], tag), tag
            assert row['messages'][0] == {'role': 'user', 'content': q['prompt']}, tag
            assert isinstance(row['messages'][-1]['content']['answer'], str)
            generation = row['generation']
            assert 0 <= generation['response_tokens'] <= 4096, tag
            assert generation['finish_reason'] in ('stop', 'length'), tag
            assert generation['prompt_tokens'] + 4096 <= 16384, tag
            assert set(generation['stop_token_ids']) == {151643,151645}, tag
            lengths.append(generation['response_tokens'])
        summaries[tag] = {'count':500,'mean_qwen_tokens':sum(lengths)/500,
            'max_qwen_tokens':max(lengths), 'capped':sum(r['generation']['finish_reason']=='length' for r in rows),
            'empty_answers':sum(not r['messages'][-1]['content']['answer'].strip() for r in rows),
            'answer_sha256':file_hash(path)}
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((SUITE / 'experiment.json').read_text())
    questions = verify_inputs(manifest)
    if args.preflight:
        print('Preflight passed: six exact policies, pinned question/baseline/code identities, no prompt truncation.', flush=True)
        return
    if args.validate_only:
        print(json.dumps(validate_results(manifest, questions), indent=2), flush=True)
        return
    runtime = {p:importlib.metadata.version(p) for p in manifest['gpu_versions']}
    assert runtime == manifest['gpu_versions'], runtime
    import torch
    assert torch.cuda.device_count() == 1
    runtime['gpu'] = torch.cuda.get_device_name(0)
    runtime['extra_versions'] = {p:importlib.metadata.version(p) for p in ['tiktoken','pandas','peft']}
    save(SUITE/'job/runtime.json', runtime)
    state = json.loads((SUITE/'status.json').read_text())
    state.update(state='generating', started_at=datetime.now().astimezone().isoformat())
    save(SUITE/'status.json',state)
    command = ['eval_arena_hard', '--model', manifest['base_model'], '--dataset', str(SUITE/'question.jsonl'),
      '--output', str(SUITE), '--max-tokens', '4096','--max-model-len','16384',
      '--seed','42','--policy-head-dtype','float32','--adapters']
    command.extend(f"{tag}={item['adapter']}" for tag,item in manifest['models'].items())
    save(SUITE/'job/command.json',command)
    try:
        from scripts import eval_arena_hard
        sys.argv = command
        eval_arena_hard.main()
        summary = validate_results(manifest, questions)
        save(SUITE/'generation_summary.json',summary)
        state.update(state='generation_complete',generation_finished_at=datetime.now().astimezone().isoformat())
        save(SUITE/'generation_complete.json',state)
        print('All six sets of 500 answers verified.',flush=True)
    except BaseException as exc:
        state.update(state='generation_failed',error=repr(exc))
        raise
    finally:
        save(SUITE/'status.json',state)

if __name__ == '__main__':
    main()
