#!/usr/bin/env python3
"""Freeze the 17-model paper campaign: creative completion and Qwen-Instruct seed43."""
import ast
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/data/VPO-RM/runs/formal-arena750-20260920')
OLD = Path('/data/VPO-RM/runs')


def read(p):
    return json.loads(Path(p).read_text())


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def rows(p):
    return [json.loads(s) for s in Path(p).open() if s.strip()]


def write_rows(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = OUT / 'source'
    prior = OLD / 'llama-base-evals-20260919/arena'
    shutil.copytree(prior / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'scripts/judge_arena_hard.py', source / 'scripts/judge_arena_hard.py')
    judge_source = OLD / 'qwen-instruct-creative250-20260920/judge_pipeline/source/third_party/arena_hard'
    for name in ('config/arena-hard-v2.0.yaml', 'utils/judge_utils.py', 'gen_judgment.py'):
        target = source / 'third_party/arena_hard' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(judge_source / name, target)
    # Retain the already-used Llama tokenizer override and allow the two subsets/all750.
    p = source / 'scripts/eval_arena_hard.py'
    text = p.read_text()
    start = text.index('def validate_questions(')
    end = text.index('\n\ndef validate_answers(', start)
    text = text[:start] + '''def validate_questions(rows, expected_count=500, category='hard_prompt'):
    counts = {'hard_prompt': 500, 'creative_writing': 250, 'all': 750}
    if category not in counts or len(rows) != expected_count or expected_count != counts[category]:
        raise ValueError('Wrong category/question count')
    seen = set()
    for row in rows:
        if (not isinstance(row.get('uid'), str) or not row['uid'] or row['uid'] in seen
                or not isinstance(row.get('prompt'), str) or not row['prompt'].strip()
                or row.get('category') not in ('hard_prompt', 'creative_writing')
                or (category != 'all' and row['category'] != category)):
            raise ValueError('Question identity/category mismatch')
        seen.add(row['uid'])
    if category == 'all' and sum(r['category'] == 'hard_prompt' for r in rows) != 500:
        raise ValueError('Expected exactly hard500 + creative250')
    return rows
''' + text[end:]
    text = text.replace('questions = validate_questions(read_jsonl(args.dataset))',
        "category = getattr(args, 'category', 'hard_prompt')\n    questions = validate_questions(read_jsonl(args.dataset), {'hard_prompt': 500, 'creative_writing': 250, 'all': 750}[category], category)")
    text = text.replace('tokenizer = load_actor_tokenizer(args.model, tokenizer_name=tokenizer_name)',
        "if tokenizer_name:\n        tokenizer = load_actor_tokenizer(args.model, tokenizer_name=tokenizer_name)\n    else:\n        from transformers import AutoTokenizer\n        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)")
    text = text.replace("parser.add_argument('--output', required=True)",
        "parser.add_argument('--category', choices=('hard_prompt','creative_writing','all'), default='hard_prompt')\n    parser.add_argument('--output', required=True)")
    ast.parse(text)
    p.write_text(text)
    shutil.copytree(OLD / 'qwen-instruct-creative250-20260920/.tiktoken-cache', OUT / '.tiktoken-cache')
    for filename in ('resilient_judge.py', 'arena_exclusion_policy.py', 'pipeline.py'):
        target = 'legacy_pipeline.py' if filename == 'pipeline.py' else filename
        shutil.copy2(OLD / 'qwen-instruct-creative250-20260920/judge_pipeline' / filename, OUT / target)

    questions = rows(ROOT / 'third_party/arena_hard/data/arena-hard-v2.0/question.jsonl')
    refs = rows(ROOT / 'runs/arena-hard-v2-gpt4o-judge-20260917/reference/official-gpt-4o-mini-2024-07-18-750.jsonl')
    ref = {r['uid']: r for r in refs}
    assert len(questions) == len(ref) == 750
    for category, count in [('all', 750), ('hard_prompt', 500), ('creative_writing', 250)]:
        selected = [q for q in questions if category == 'all' or q['category'] == category]
        assert len(selected) == count
        for q in selected:
            assert ref[q['uid']]['messages'][0]['content'] == q['prompt']
        write_rows(OUT / 'data' / f'{category}.jsonl', selected)
        write_rows(OUT / 'reference' / f'{category}.jsonl', [ref[q['uid']] for q in selected])

    models = {}
    groups = {
        'qb': dict(actor='Qwen3-14B-Base', rm='Skywork-Qwen3-8B',
                   base_model=str(ROOT / 'models/Qwen3-14B-Base'), tokenizer='',
                   sft=str(ROOT / 'models/sft-native-eos-clean2k5e2')),
    }
    for tag in ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8'):
        old = ROOT / 'runs/arena-hard-v2-canonical-20260917/manifests' / f'{tag}.json'
        config = read(old)['config']
        models['qb-' + tag] = dict(family='qb', old_tag=tag, adapter=(config['adapter'] or {}).get('path', 'none'),
            old_generation_manifest=str(old), old_hard_provenance=str(OLD / 'arena-hard-gpt4o-retry5-gpt41-20260918/scores/mixed_gpt4o_gpt41/game_provenance.jsonl'),
            seed=42, category='creative_writing', questions=250, action='complete_creative')
    sources = {
        'qb-random': ('qb', OLD / 'randdir-evals-20260919/arena', ['randdir']),
        'lb': ('lb', prior, ['base', 'sft', 'grpo', 'lam4']),
        'li': ('li', OLD / 'direct-llama-reward-arena-20260918/arena_hard', ['instruct', 'grpo', 'lam4']),
        'qi': ('qi', OLD / 'qwen-instruct-evals-20260919/arena', ['instruct', 'grpo', 'lam4']),
    }
    for name, (family, campaign, tags) in sources.items():
        old_manifest = campaign / 'generation_manifest.json'
        old = read(old_manifest)
        if family != 'qb':
            groups[family] = dict(actor={'lb':'Llama-3.1-8B-Base','li':'Llama-3.1-8B-Instruct','qi':'Qwen3-14B-Instruct'}[family],
                rm='Skywork-Qwen3-8B' if family == 'qi' else 'Skywork-Llama-3.1-8B-v0.2',
                base_model=old['base_model'], tokenizer=old.get('tokenizer') or '',
                sft=old.get('tokenizer') if family == 'lb' else '')
        for tag in tags:
            key = 'qb-random' if name == 'qb-random' else family + '-' + tag
            reuse = family == 'qi' and tag == 'instruct'
            rerun = family == 'qi' and not reuse
            models[key] = dict(family=family, old_tag=tag, adapter=old['models'][tag]['adapter'],
                old_generation_manifest=str(campaign / 'manifests' / f'{tag}.json'),
                old_hard_provenance=str(campaign / 'judge_pipeline/scores/mixed_gpt4o_gpt41/game_provenance.jsonl'),
                seed=43 if rerun else 42, category='all' if rerun else 'creative_writing',
                questions=0 if reuse else (750 if rerun else 250),
                action='reuse_all750' if reuse else ('rerun_all750' if rerun else 'complete_creative'))
    assert len(models) == 17 and sum(m['questions'] for m in models.values()) == 5000
    plan = dict(schema='formal_arena750_v1', groups=groups, models=models,
        gpu_versions=old['gpu_versions'], reference='gpt-4o-mini-2024-07-18',
        judge_policy=dict(primary='gpt-4o', primary_output_attempts=5, fallback='gpt-4.1', fallback_output_attempts=1,
            stop_at_first_valid=True, orders=2, temperature=0, max_tokens=16000),
        generation=dict(temperature=1.0, top_p=1.0, top_k=-1, n=1, max_tokens=4096,
            max_model_len=16384, max_num_seqs=32, policy_head_dtype='float32'),
        new_answers=5000, new_judge_games=10000,
        scoring='Pool the individual two-order verdicts across both categories. Strong verdicts weight3; ties0.5. Common valid UIDs within each actor family; same Qwen-base value in both paper tables.',
        qi_reuse_creative=str(OLD / 'qwen-instruct-creative250-20260920'),
        non_arena_columns='Preserve exactly as supplied by user.',
        authorization='User requested creative250 for the remaining14 models and seed43 evaluation of existing Qwen-Instruct GRPO/VPO checkpoints on all750. Never stop/requeue/resize existing jobs.')
    save(OUT / 'plan.json', plan)
    template = read(OLD / 'qwen-instruct-creative250-20260920/job/lam4/submit_command.json')
    for key, item in models.items():
        if not item['questions']:
            continue
        command = template[:template.index('--')]
        command[command.index('--name') + 1] = 'ar750-' + key + '-s' + str(item['seed']) + '-0920'
        command += ['--', 'bash', str(OUT / 'run_generation.sh'), key]
        save(OUT / 'jobs' / key / 'submit_command.json', command)
    print(json.dumps(dict(root=str(OUT), models=len(models), generation_jobs=16, answers=5000, judge_games=10000)))


if __name__ == '__main__':
    main()
