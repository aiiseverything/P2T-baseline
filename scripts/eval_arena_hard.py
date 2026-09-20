#!/usr/bin/env python3
"""Generate official Arena-Hard v2 answer files with a shared Qwen/LoRA engine."""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_alpaca import parse_adapters
from scripts.eval_artifacts import (atomic_text, cache_matches, commit_cache, digest,
                                    eval_config, file_hash, fingerprint, runtime_versions,
                                    validate_adapter_base, validate_outputs)
from scripts.eval_policy import policy_engine_kwargs, resolve_shared_policy

GENERATION_SOURCES = ('scripts/eval_alpaca.py', 'scripts/eval_policy.py',
                      'third_party/arena_hard/gen_answer.py',
                      'third_party/arena_hard/utils/add_markdown_info.py')


def read_jsonl(path):
    # splitlines() also splits U+2028/U+0085 inside valid JSON string literals.
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


@lru_cache(maxsize=1)
def _style_tools():
    import tiktoken
    path = ROOT / 'third_party/arena_hard/utils/add_markdown_info.py'
    spec = importlib.util.spec_from_file_location('_arena_hard_style', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tiktoken.encoding_for_model('gpt-4o'), module


def style_metadata(text):
    """Use upstream GPT-4o length and its exact code-fence/Markdown counting."""
    encoding, upstream = _style_tools()
    return {'token_len': len(encoding.encode(text, disallowed_special=()))} | upstream.count_markdown_elements(
        upstream.remove_pattern(text, re.compile('```([^`]*)```')), suffix='')


def validate_questions(rows, expected_count=500, category='hard_prompt'):
    if category not in ('hard_prompt', 'creative_writing'):
        raise ValueError('Unsupported Arena category')
    if not isinstance(rows, list) or len(rows) != expected_count or not rows:
        raise ValueError(f'Expected exactly {expected_count} hard questions')
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('uid'), str) or not row['uid']
                or row['uid'] in seen or row.get('category') != category
                or not isinstance(row.get('prompt'), str) or not row['prompt'].strip()):
            raise ValueError('Questions require unique uid, hard_prompt category and nonempty prompt')
        seen.add(row['uid'])
    return rows


def validate_answers(rows, questions, model_tag):
    """Validate exact question/model binding and independently recompute style."""
    if not isinstance(rows, list) or len(rows) != len(questions):
        raise ValueError('Incomplete Arena answer coverage')
    by_uid = {question['uid']: question for question in questions}
    seen = set()
    for row in rows:
        try:
            uid = row['uid']
            messages = row['messages']
            answer = messages[-1]['content']['answer']
            generation = row['generation']
            valid = (uid in by_uid and uid not in seen and row['model'] == model_tag
                and isinstance(answer, str)
                and messages == [{'role': 'user', 'content': by_uid[uid]['prompt']},
                                 {'role': 'assistant', 'content': {'answer': answer}}]
                and isinstance(row['ans_id'], str) and bool(row['ans_id'])
                and isinstance(row['tstamp'], (float, int)) and math.isfinite(row['tstamp'])
                and row['metadata'] == style_metadata(answer)
                and generation['finish_reason'] in ('stop', 'length')
                and all(type(generation[k]) is int and generation[k] >= 0
                        for k in ('response_tokens', 'prompt_tokens'))
                and isinstance(generation['stop_token_ids'], list)
                and all(type(token) is int and token >= 0 for token in generation['stop_token_ids'])
                and (generation['last_token_id'] is None if generation['response_tokens'] == 0
                     else type(generation['last_token_id']) is int and generation['last_token_id'] >= 0)
                and type(generation['ended_with_eos']) is bool
                and generation['ended_with_eos'] == (generation['last_token_id'] in generation['stop_token_ids']))
        except (KeyError, TypeError, IndexError):
            valid = False
        if not valid:
            raise ValueError('Invalid Arena answer, metadata, or question/model binding')
        seen.add(uid)
    if seen != set(by_uid):
        raise ValueError('Incomplete Arena answer coverage')
    return rows


def _summary(rows, answer_path, manifest_path):
    return {'answer_path': str(answer_path), 'manifest_path': str(manifest_path),
            'n_answers': len(rows),
            'mean_response_tokens': sum(r['generation']['response_tokens'] for r in rows) / len(rows),
            'n_length': sum(r['generation']['finish_reason'] == 'length' for r in rows),
            'n_empty': sum(not r['messages'][-1]['content']['answer'].strip() for r in rows)}


def engine_arguments(args, head, tokenizer_source):
    """Shared vLLM engine identity; prompts are explicit IDs, the tokenizer only detokenizes."""
    return dict(model=args.model, tokenizer=tokenizer_source, dtype='bfloat16', trust_remote_code=True,
        generation_config='vllm', enable_lora=True, max_lora_rank=64, max_loras=1,
        max_model_len=args.max_model_len, max_num_seqs=getattr(args, 'max_num_seqs', 32),
        gpu_memory_utilization=0.80, tensor_parallel_size=1, seed=args.seed,
        **policy_engine_kwargs(head))


def generation_config(args, tag, adapter_fingerprint, policy, engine_kwargs, *, model_fingerprint,
                      tokenizer_provenance, stop_ids, support_summary, prompt_ids, runtime=None):
    """Cache identity of one answer file; offline callers supply the GPU runtime versions."""
    recipe = {'temp': 1.0, 'n': 1, 'top_p': 1.0, 'top_k': -1}
    config = eval_config(args, recipe, model_fingerprint, adapter_fingerprint, stop_ids,
                         'arena_hard', support_summary)
    config.update(model_tag=tag, policy=policy, engine=dict(engine_kwargs),
                  tokenizer=dict(tokenizer_provenance), prompt_token_ids_sha256=digest(prompt_ids),
                  style_protocol='arena_hard_v2_gpt4o_upstream_v1')
    config['runtime_versions'] = (runtime_versions(('transformers', 'vllm', 'tiktoken', 'pandas'))
                                  if runtime is None else dict(runtime))
    for source in GENERATION_SOURCES:
        config['sources'][source] = file_hash(ROOT / source)
    return config


def generate_all(args):
    """Evaluate all args.adapters sequentially in one lazy engine; never judge."""
    adapters = parse_adapters(args.adapters)
    head, policies = resolve_shared_policy([path for _, path in adapters], args.policy_head_dtype)
    if args.max_tokens < 1 or args.max_model_len < 1 or getattr(args, 'max_num_seqs', 32) < 1:
        raise ValueError('Token/context budgets and max_num_seqs must be positive')
    category = getattr(args, 'category', 'hard_prompt')
    questions = validate_questions(read_jsonl(args.dataset),
                                  250 if category == 'creative_writing' else 500, category)
    for _, path in adapters:
        validate_adapter_base(path, args.model)

    from transformers import AutoConfig
    from vpo_rm.integration import checked_sampling_params, sampling_summary, vllm_support_kwargs
    from vpo_rm.token_policy import (get_stop_token_ids, load_actor_tokenizer,
                                     resolve_actor_tokenizer_source)
    from vpo_rm.trainer import VPOTrainer
    # A base checkpoint without a chat template renders with its saved SFT tokenizer,
    # exactly as the RL rollouts did; the model's own tokenizer remains the default.
    tokenizer_name = getattr(args, 'tokenizer', '') or ''
    tokenizer = load_actor_tokenizer(args.model, tokenizer_name=tokenizer_name)
    tokenizer_source = resolve_actor_tokenizer_source(args.model, tokenizer_name=tokenizer_name)
    tokenizer_provenance = {'source': tokenizer_source,
                            'fingerprint': fingerprint(tokenizer_source, full_weights=False)}
    stop_ids = get_stop_token_ids(tokenizer)
    if not stop_ids:
        raise ValueError('No registered EOS/stop tokens for actor')
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, q['prompt']) for q in questions]
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in rendered]
    if any(not ids or len(ids) + args.max_tokens > args.max_model_len for ids in prompt_ids):
        raise ValueError('A full prompt plus the uniform response budget exceeds the context limit')
    prompts = [{'prompt_token_ids': ids} for ids in prompt_ids]
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    support_kwargs = vllm_support_kwargs(tokenizer, vocab_size)
    support_summary = sampling_summary(support_kwargs)
    model_fingerprint = fingerprint(args.model, full_weights=False)
    engine_kwargs = engine_arguments(args, head, tokenizer_source)
    plans = []
    # Validate every existing cache before allocating a GPU engine or overwriting anything.
    for i, (tag, path) in enumerate(adapters):
        answer_path = Path(args.output) / 'model_answer' / f'{tag}.jsonl'
        manifest_path = Path(args.output) / 'manifests' / f'{tag}.json'
        config = generation_config(args, tag, None if path == 'none' else fingerprint(path),
            policies[i], engine_kwargs, model_fingerprint=model_fingerprint,
            tokenizer_provenance=tokenizer_provenance, stop_ids=stop_ids,
            support_summary=support_summary, prompt_ids=prompt_ids)
        cached = cache_matches(manifest_path, config, [answer_path])
        if cached:
            validate_answers(read_jsonl(answer_path), questions, tag)
        plans.append((tag, path, answer_path, manifest_path, config, cached))
    # Fail on missing tokenization/style dependencies before GPU allocation.
    _style_tools()
    llm, summaries = None, {}
    for i, (tag, path, answer_path, manifest_path, config, cached) in enumerate(plans):
        if cached:
            rows = read_jsonl(answer_path)
        else:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            if llm is None:
                llm = LLM(**engine_kwargs)
            params = checked_sampling_params(SamplingParams, temperature=1.0, top_p=1.0, top_k=-1,
                n=1, max_tokens=args.max_tokens, seed=args.seed, stop_token_ids=list(stop_ids), **support_kwargs)
            lora = None if path == 'none' else LoRARequest(f'lora-{tag}', i + 1, str(Path(path)))
            outputs = llm.generate(prompts, params, lora_request=lora)
            validate_outputs(outputs, len(questions), 1)
            rows = []
            for question, ids, output in zip(questions, prompt_ids, outputs):
                if list(output.prompt_token_ids) != ids:
                    raise ValueError('vLLM prompt order/token IDs mismatch')
                sample = output.outputs[0]
                if len(sample.token_ids) > args.max_tokens:
                    raise ValueError('vLLM response exceeded configured token budget')
                last = sample.token_ids[-1] if sample.token_ids else None
                rows.append({'uid': question['uid'], 'ans_id': uuid.uuid4().hex,
                    'model': tag, 'tstamp': time.time(),
                    'messages': [{'role': 'user', 'content': question['prompt']},
                                 {'role': 'assistant', 'content': {'answer': sample.text}}],
                    'metadata': style_metadata(sample.text),
                    'generation': {'response_tokens': len(sample.token_ids), 'prompt_tokens': len(ids),
                        'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                        'last_token_id': last, 'stop_token_ids': list(stop_ids),
                        'ended_with_eos': last in stop_ids}})
            validate_answers(rows, questions, tag)
            atomic_text(answer_path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
            commit_cache(manifest_path, config, [answer_path])
        summaries[tag] = _summary(rows, answer_path, manifest_path)
        print(json.dumps({'tag': tag, 'cached': cached, **summaries[tag]}), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3-14B-Base')
    parser.add_argument('--dataset', required=True, help='Exactly 500 hard_prompt JSONL rows')
    parser.add_argument('--category', choices=('hard_prompt', 'creative_writing'), default='hard_prompt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapters', nargs='+', required=True, help='TAG=PATH; base=none')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy-head-dtype', choices=('auto', 'native', 'float32'), default='float32')
    parser.add_argument('--tokenizer', default='',
                        help='Tokenizer directory that renders the chat prompts, e.g. a saved SFT '
                             'adapter when the base checkpoint ships without a chat template '
                             "(default: the model's own tokenizer)")
    generate_all(parser.parse_args())


if __name__ == '__main__':
    main()
