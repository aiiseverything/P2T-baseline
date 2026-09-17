#!/usr/bin/env python3
"""Audit immutable official GPT-4o-mini responses and select canonical hard500."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
from collections import Counter
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'runs/arena-hard-v2-canonical-20260917'
REVISION = '15f3746e21432264ce9b453999bde4f3c946d2e6'
MODEL = 'gpt-4o-mini-2024-07-18'
REPO = 'lmarena-ai/arena-hard-auto'
TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
OFFICIAL_LFS_SHA256 = {
    'official_questions': '6cfe75abb0e09cd39e7f9b0b18ea96d41be2d4d9ddf76dc1d07d78b6accb2039',
    'official_answers': '7c02a9479dcba4b5facf73f599cd352acb63475a798a2316dc80f3601c9a1711',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique(items, label):
    result = {row['uid']: row for row in items}
    assert len(result) == len(items), f'Duplicate UID in {label}'
    return result


def write_new_or_identical(path, body):
    if path.exists():
        assert path.read_bytes() == body, f'Refusing to replace different file: {path}'
    else:
        path.write_bytes(body)


def main():
    source_question_path = HERE / 'official-question-750.jsonl'
    source_answer_path = HERE / f'official-{MODEL}-750.jsonl'
    canonical_question_path = OLD / 'question.jsonl'
    assert sha(source_question_path) == OFFICIAL_LFS_SHA256['official_questions']
    assert sha(source_answer_path) == OFFICIAL_LFS_SHA256['official_answers']
    questions = rows(canonical_question_path)
    official_questions = rows(source_question_path)
    official_answers = rows(source_answer_path)
    question_map = unique(questions, 'canonical questions')
    official_question_map = unique(official_questions, 'official questions')
    answer_map = unique(official_answers, 'official answers')
    assert len(questions) == 500
    assert len(official_questions) == len(official_answers) == 750
    assert set(answer_map) == set(official_question_map)
    assert all(q['category'] == 'hard_prompt' for q in questions)
    assert {q['uid'] for q in official_questions if q['category'] == 'hard_prompt'} == set(question_map)
    for question in questions:
        official_question = official_question_map[question['uid']]
        assert {key: official_question[key] for key in question} == question, f'Official question mismatch: {question["uid"]}'
        assert set(official_question) - set(question) == {'language'}
        assert isinstance(official_question['language'], str) and official_question['language']

    os.environ['TIKTOKEN_CACHE_DIR'] = str(OLD / '.tiktoken-cache')
    import tiktoken
    encoding = tiktoken.encoding_for_model('gpt-4o')
    style_path = OLD / 'source/third_party/arena_hard/utils/add_markdown_info.py'
    spec = importlib.util.spec_from_file_location('_frozen_official_arena_style', style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    pattern = re.compile('```([^`]*)```')

    def validate_answer(row, question, model):
        assert row['model'] == model, f'Model mismatch: {row["uid"]}'
        messages = row['messages']
        assert len(messages) == 2, f'Unexpected messages: {row["uid"]}'
        assert messages[0] == {'role': 'user', 'content': question['prompt']}, f'Answer prompt mismatch: {row["uid"]}'
        assert messages[1]['role'] == 'assistant'
        answer = messages[1]['content']['answer']
        assert isinstance(answer, str) and answer.strip(), f'Empty answer: {row["uid"]}'
        metadata = {'token_len': len(encoding.encode(answer, disallowed_special=()))}
        metadata |= style.count_markdown_elements(style.remove_pattern(answer, pattern), suffix='')
        assert row['metadata'] == metadata, f'Style metadata mismatch: {model}/{row["uid"]}'

    for answer in official_answers:
        validate_answer(answer, official_question_map[answer['uid']], MODEL)
    selected = [answer_map[q['uid']] for q in questions]
    candidate_audits = {}
    for tag in TAGS:
        path = OLD / 'model_answer' / f'{tag}.jsonl'
        candidates = rows(path)
        candidate_map = unique(candidates, tag)
        assert len(candidates) == 500 and set(candidate_map) == set(question_map)
        for answer in candidates:
            validate_answer(answer, question_map[answer['uid']], tag)
        candidate_audits[tag] = {
            'path': str(path), 'sha256': sha(path), 'answers': len(candidates),
            'unique_uids': len(candidate_map), 'hard500_coverage': 500,
            'missing_uids': [], 'extra_uids': [], 'nonempty_answers': 500,
            'exact_prompt_matches': 500, 'recomputed_style_matches': 500,
        }
    filtered_path = HERE / f'{MODEL}.jsonl'
    # Preserve each original object; the only transformation is selection/order.
    filtered_bytes = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected).encode('utf-8')
    write_new_or_identical(filtered_path, filtered_bytes)
    original_lines = {json.loads(line)['uid']: line for line in source_answer_path.read_text().split('\n') if line.strip()}
    assert all(json.loads(original_lines[row['uid']]) == row for row in rows(filtered_path))
    sources = {}
    for name, path, remote in (
        ('official_questions', source_question_path, 'question.jsonl'),
        ('official_answers', source_answer_path, f'model_answer/{MODEL}.jsonl'),
    ):
        sources[name] = {
            'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'url': f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/arena-hard-v2.0/{remote}',
            'official_lfs_sha256': OFFICIAL_LFS_SHA256[name],
            'rows': 750,
        }
    audit = {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'reference_model': MODEL,
        'official_dataset': {'repo_id': REPO, 'revision': REVISION},
        'sources': sources,
        'canonical_questions': {'path': str(canonical_question_path), 'sha256': sha(canonical_question_path), 'rows': 500},
        'output': {'path': str(filtered_path), 'sha256': sha(filtered_path), 'rows': 500},
        'checks': {
            'official_unique_question_uids': 750, 'official_unique_answer_uids': 750,
            'official_complete_question_coverage': True,
            'canonical_question_fields_identical_to_official': 500,
            'official_additional_question_fields': ['language'],
            'official_hard500_language_counts': dict(Counter(official_question_map[q['uid']]['language'] for q in questions)),
            'official_answer_prompt_matches': 750, 'official_nonempty_answers': 750,
            'official_answer_model_matches': 750, 'official_recomputed_style_matches': 750,
            'hard500_coverage': 500, 'missing_uids': [], 'extra_uids': [],
            'selected_order_matches_canonical_questions': True,
            'selected_objects_unchanged': True,
        },
        'candidate_coverage': candidate_audits,
        'style_validation': {
            'source': str(style_path), 'source_sha256': sha(style_path),
            'tokenizer_model': 'gpt-4o', 'encoding': encoding.name,
            'versions': {name: importlib.metadata.version(name) for name in ('tiktoken', 'pandas', 'tqdm')},
            'cache_files': {p.name: sha(p) for p in (OLD / '.tiktoken-cache').iterdir() if p.is_file()},
        },
        'preparation_script': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
        'notes': ['Reference replacement defines a custom Arena-Hard-v2 hard500 protocol.',
                  'Official HF question rows additionally contain a language field; all canonical uid/category/subcategory/prompt fields match exactly.',
                  'The official questions were fetched through an hf-mirror URL redirecting to Hugging Face after direct TLS failures; their bytes match the pinned official LFS SHA256.',
                  'All six existing canonical candidate outputs are reused without generation.',
                  'No old inputs, responses or judgments were modified.'],
    }
    (HERE / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    provenance = {key: audit[key] for key in ('reference_model', 'official_dataset', 'sources', 'canonical_questions', 'output', 'preparation_script')}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'official_revision': REVISION, 'output': audit['output'], 'candidate_models': list(candidate_audits), 'total_style_metadata_verified': 3750}))


if __name__ == '__main__':
    main()
