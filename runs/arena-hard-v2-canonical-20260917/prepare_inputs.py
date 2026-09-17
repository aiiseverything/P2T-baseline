"""Pin Arena-Hard hard500 inputs and check normalized exact training overlaps."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, file_hash
from vpo_rm.data import normalize_prompt


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def main():
    import numpy as np
    import tiktoken
    from transformers import AutoTokenizer
    from vpo_rm.trainer import VPOTrainer
    upstream = ROOT / 'third_party/arena_hard'
    data = upstream / 'data/arena-hard-v2.0'
    questions = [json.loads(l) for l in (data / 'question.jsonl').open()]
    hard = [q for q in questions if q['category'] == 'hard_prompt']
    ids = [q['uid'] for q in hard]
    assert len(hard) == len(set(ids)) == 500
    baseline = [json.loads(l) for l in (data / 'model_answer/o3-mini-2025-01-31.jsonl').open()]
    by_id = {r['uid']: r for r in baseline}
    assert len(by_id) == len(baseline) == 750
    baseline = [by_id[uid] for uid in ids]
    for q, b in zip(hard, baseline):
        assert b['model'] == 'o3-mini-2025-01-31'
        assert b['messages'][0] == {'role': 'user', 'content': q['prompt']}
        assert isinstance(b['messages'][-1]['content']['answer'], str)
    atomic_text(SUITE / 'question.jsonl', ''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in hard))
    atomic_text(SUITE / 'model_answer/o3-mini-2025-01-31.jsonl', ''.join(json.dumps(b, ensure_ascii=False) + '\n' for b in baseline))
    os.environ['TIKTOKEN_CACHE_DIR'] = str(SUITE / '.tiktoken-cache')
    encoder = tiktoken.encoding_for_model('gpt-4o')
    helper_spec = importlib.util.spec_from_file_location('arena_style', upstream / 'utils/add_markdown_info.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    import re
    for b in baseline:
        answer = b['messages'][-1]['content']['answer']
        metadata = {'token_len': len(encoder.encode(answer, disallowed_special=()))}
        metadata.update(helper.count_markdown_elements(helper.remove_pattern(answer, re.compile('```([^`]*)```')), suffix=''))
        assert metadata == b['metadata'], b['uid']
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3-14B-Base', local_files_only=True)
    lengths = [len(VPOTrainer._render_chat_prompt(tokenizer, q['prompt'], tokenize=True)) for q in hard]
    assert max(lengths) + 4096 <= 16384
    normalized = {normalize_prompt(q['prompt']): q['uid'] for q in hard}
    assert len(normalized) == 500
    overlap = {}
    for tag in ['grpo', 'lam2', 'lam4', 'lam8']:
        directory = ROOT / 'runs/rl-fp32-is-canonical-20260917' / tag / 'train'
        paths = [directory / f'rollout-{i}-prompts.json' for i in range(1,251)]
        prompts = [p for path in paths for p in json.loads(path.read_text())]
        assert len(prompts) == 2000
        hits = sorted({normalized[normalize_prompt(p)] for p in prompts if normalize_prompt(p) in normalized})
        overlap[tag] = {'used_prompt_records': len(prompts), 'unique_prompts': len(set(prompts)), 'normalized_exact_overlap_uids': hits}
    # Reconstruct the historical SFT sample with its frozen, original sampler.
    sft_file = ROOT / 'runs/sft-native-eos-clean2k5e2/source/scripts/sft_init.py'
    spec = importlib.util.spec_from_file_location('historical_sft', sft_file)
    old_sft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_sft)
    pairs, split = old_sft.load_chosen_pairs(str(ROOT / 'datasets/sft_v2/sft_clean.parquet'), 2000)
    pairs = random.Random(42).sample(pairs, 2500)
    sample_digest = hashlib.sha256('\n'.join(f'{p}|{c[:64]}' for p, c in pairs).encode()).hexdigest()
    manifest = json.loads((ROOT / 'models/sft-native-eos-clean2k5e2/sft_manifest.json').read_text())
    assert sample_digest == manifest['data_sha256']
    assert split == manifest['split']
    overlap['sft-init'] = {'used_prompt_records': len(pairs), 'sample_sha256': sample_digest,
        'normalized_exact_overlap_uids': sorted({normalized[normalize_prompt(p)] for p,_ in pairs if normalize_prompt(p) in normalized})}
    info = {'benchmark': 'arena-hard-v2.0', 'subset': 'hard_prompt', 'count':500,
      'upstream_revision':'196f6b826783b3da7310e361a805fa36f0be83f3',
      'source_sha256': {str(p.relative_to(upstream)):file_hash(p) for p in [data/'question.jsonl',data/'model_answer/o3-mini-2025-01-31.jsonl']},
      'subcategory_counts':dict(collections.Counter(q.get('subcategory') for q in hard)),
      'prompt_tokens_qwen':dict(zip(['min','median','p90','p95','p99','max'],np.percentile(lengths,[0,50,90,95,99,100]).tolist())),
      'generation': {'max_tokens':4096,'max_model_len':16384,'temperature':1.0,'top_p':1.0,'top_k':-1,'seed':42,'policy_head_dtype':'float32'},
      'overlap_check':{'method':'normalized exact prompt equality; not a semantic or pretraining-contamination audit','models':overlap}}
    # Fixed random pilot chosen without observing any candidates or judgments.
    pilot = sorted(random.Random(42).sample(ids, 5))
    write_json(SUITE/'pilot_uids.json',pilot)
    write_json(SUITE/'inputs.json',info)
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()
