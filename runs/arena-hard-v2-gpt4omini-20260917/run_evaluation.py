"""Validate byte-identical reuse of completed generation; never generate answers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parent
TAGS = ('base', 'sft-init', 'grpo', 'lam2', 'lam4', 'lam8')
BASELINE = 'gpt-4o-mini-2024-07-18'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_copy(source, target, expected):
    require(file_hash(source) == expected, f'Original source changed: {source}')
    require(file_hash(target) == expected, f'Reused copy changed: {target}')


def verify_reused_generation(suite=SUITE):
    suite = Path(suite).resolve()
    experiment = read_json(suite / 'experiment.json')
    require(tuple(experiment['models']) == TAGS, 'Candidate set/order changed')
    require(experiment['judge']['baseline'] == BASELINE
            and experiment['judge']['uses_official_baseline'] is False,
            'Custom baseline is not explicitly identified')
    for relative, expected in experiment['files_sha256'].items():
        require(file_hash(suite / relative) == expected, f'Frozen input/source changed: {relative}')
    reuse = experiment['generation_reuse']
    require(reuse['new_rollouts'] == 0, 'Only reuse is supported')
    original = Path(reuse['source_suite'])
    prior = read_json(suite / 'original_generation/experiment.json')
    completion = read_json(suite / 'original_generation/evaluation_complete.json')
    require(completion['state'] == 'evaluation_complete'
            and completion['verification']['status'] == 'passed'
            and completion['verification']['current_games'] == 6000,
            'Original generation lacks completed verification')
    for relative, expected in reuse['original_files_sha256'].items():
        destination = suite / ('original_generation/' + relative if relative in
                              ('experiment.json', 'evaluation_complete.json', 'generation_complete.json')
                              else relative)
        verify_copy(original / relative, destination, expected)
    require(experiment['models'] == prior['models'], 'Reused model identities changed')
    require(experiment['base_identity'] == prior['base_identity'], 'Reused base identity changed')
    require(experiment['generation'] == prior['generation'], 'Reused generation recipe changed')
    old_hashes = completion['verification']['input_files_sha256']
    questions = read_jsonl(suite / 'question.jsonl')
    require(len(questions) == len({q['uid'] for q in questions}) == 500
            and all(q['category'] == 'hard_prompt' for q in questions), 'Expected unique hard500')
    require(file_hash(suite / 'question.jsonl') == old_hashes[str(original / 'question.jsonl')],
            'Question bytes differ from original verified evaluation')
    require({p.stem for p in (suite / 'model_answer').glob('*.jsonl')} == {*TAGS, BASELINE},
            'Unexpected model-answer coverage')
    for tag in TAGS:
        path = suite / 'model_answer' / f'{tag}.jsonl'
        answer_hash = file_hash(path)
        require(answer_hash == old_hashes[str(original / 'model_answer' / path.name)],
                f'Answer bytes differ from original verified evaluation: {tag}')
        cache = read_json(suite / 'manifests' / f'{tag}.json')
        config = cache['config']
        require(cache['outputs'] == {path.name: answer_hash}, f'Generation cache output mismatch: {tag}')
        require(config['model_tag'] == tag and config['policy'] == experiment['models'][tag]['policy'],
                f'Generation policy mismatch: {tag}')
        require(config['model'] == prior['base_identity'], f'Base model provenance mismatch: {tag}')
        require(config['recipe'] == {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}
                and config['seed'] == 42 and config['max_tokens'] == 4096
                and config['engine']['max_model_len'] == 16384
                and config['engine']['hf_overrides'] == {'head_dtype': 'float32'},
                f'Generation parameters mismatch: {tag}')
        rows = read_jsonl(path)
        require(len(rows) == 500, f'Answer count mismatch: {tag}')
        for q, row in zip(questions, rows):
            require(row['uid'] == q['uid'] and row['model'] == tag
                    and row['messages'][0] == {'role': 'user', 'content': q['prompt']},
                    f'Answer identity/prompt/order mismatch: {tag}')
            require(isinstance(row['messages'][-1]['content']['answer'], str)
                    and row['messages'][-1]['content']['answer'].strip(), f'Empty answer: {tag}')
    provenance = read_json(suite / 'reference/provenance.json')
    audit = read_json(suite / 'reference/audit.json')
    require(audit['status'] == 'passed' and provenance['reference_model'] == BASELINE,
            'Reference audit incomplete or wrong model')
    require(file_hash(suite / 'model_answer' / f'{BASELINE}.jsonl') == provenance['output']['sha256'],
            'Reference bytes differ from audited official subset')
    reference = read_jsonl(suite / 'model_answer' / f'{BASELINE}.jsonl')
    require(len(reference) == 500, 'Reference coverage mismatch')
    for q, row in zip(questions, reference):
        require(row['uid'] == q['uid'] and row['model'] == BASELINE
                and row['messages'][0] == {'role': 'user', 'content': q['prompt']}
                and row['messages'][-1]['content']['answer'].strip(), 'Reference prompt/order mismatch')
    return {'status': 'passed', 'candidates': list(TAGS), 'questions': 500, 'answers': 3000,
            'new_rollouts': 0, 'baseline_model': BASELINE, 'source_suite': str(original)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(verify_reused_generation(), indent=2))


if __name__ == '__main__':
    main()
