#!/usr/bin/env python3
"""Offline RM eval of saved LoRA adapters on a frozen prompt set.

Evaluates every vllm-adapters/step-* checkpoint of one or more training runs on
the SAME 256 validation prompts (first 256 of the SHA256-ordered 2000-prompt
validation split) so training progress is comparable across runs, checkpoints,
and temperatures. Also accepts a direct adapter directory or LABEL=none for
the bare base model. Generation runs on cuda:0 via in-process vLLM; scoring on
--rm-device (default cuda:1) using the full user/assistant reward chat template.
On an H200, --rm-device cuda:0 shares the card with the 0.45-budget vLLM engine.

Outputs:
  eval.jsonl   one row per (run, step, temp, prompt): score + response length
  summary.json per (run, step, temp): mean, bootstrap 95% CI, length stats

Usage (2-GPU rjob):
  python3 scripts/eval_checkpoints.py \
    --run grpo=runs/formal-skywork-grpo-.../ \
    --run vpo_rm=runs/formal-skywork-vpo_rm-.../ \
    --output runs/eval-p9d
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.eval_artifacts import (atomic_text, digest, fingerprint, file_hash, cache_matches,
                                    commit_cache, runtime_versions,
                                    validate_adapter_base, validate_outputs)
from scripts.eval_policy import resolve_shared_policy, policy_engine_kwargs
from vpo_rm.token_policy import (configure_model_padding, get_stop_token_ids,
                                 load_actor_tokenizer, resolve_actor_tokenizer_source,
                                 tokenize_rendered_prompts)


def load_validation_prompts(dataset_path: str, num_prompts: int) -> list[str]:
    from vpo_rm.trainer import load_prompt_dataset, split_prompts
    prompts, valid, _ = load_prompt_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", dataset_path=dataset_path)
    if len(valid) < num_prompts:
        raise RuntimeError(f"validation split has {len(valid)} prompts < {num_prompts}")
    return valid[:num_prompts]


def discover_adapters(run_dir: Path) -> list[tuple[int, Path]]:
    if str(run_dir) == 'none':
        return [(0, run_dir)]
    if (run_dir / 'adapter_config.json').is_file():
        if not any((run_dir / name).is_file() for name in
                   ('adapter_model.safetensors', 'adapter_model.bin')):
            raise FileNotFoundError(f'No adapter weights under {run_dir}')
        step = 0
        if run_dir.name.startswith(('checkpoint-', 'step-')):
            step = int(run_dir.name.rsplit('-', 1)[1])
        return [(step, run_dir)]
    root = run_dir / "vllm-adapters"
    if not root.is_dir():
        raise FileNotFoundError(f"no vllm-adapters/ under {run_dir}")
    steps = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("step-"):
            continue
        try:
            step = int(d.name.split("-", 1)[1])
        except ValueError:
            continue
        if any(d.glob("*.safetensors")):
            steps.append((step, d))
    return sorted(steps)


def _banned_ids(model_path: str) -> dict[int, float]:
    """Reserve-row ban shared with the training server (see vllm_generate_server)."""
    from transformers import AutoConfig, AutoTokenizer
    known = set(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True).get_vocab().values())
    vocab_size = AutoConfig.from_pretrained(model_path, trust_remote_code=True).vocab_size
    return {i: -100.0 for i in range(vocab_size) if i not in known}


def _stop_ids(model_path):
    from transformers import AutoTokenizer
    from vpo_rm.token_policy import get_stop_token_ids
    return list(get_stop_token_ids(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)))


def _presence_penalty() -> float:
    """Matched-conditions eval: PP-trained arms are evaluated with their
    training-time presence penalty (EVAL_PP env); PP-free arms default to 0."""
    import os
    return float(os.environ.get("EVAL_PP", "0.0"))


def _top_p() -> float:
    """EVAL_TOPP overrides the 0.9 default (1.0 replicates training rollouts)."""
    import os
    return float(os.environ.get("EVAL_TOPP", "0.9"))


def validate_temperature(temperature: float) -> float:
    """Reject values vLLM would clamp or reject instead of changing protocol."""
    if (not math.isfinite(temperature)
            or (temperature != 0 and not 0.01 <= temperature <= 2.0)):
        raise ValueError("temperature must be 0 or in [0.01, 2.0]")
    return temperature


def generate_all(runs, rendered, temps, args):
    """In-process vLLM: one pass over (run, step, temp); returns token-id lists."""
    head, _ = resolve_shared_policy(
        [adapter for _, run_dir in runs for _, adapter in discover_adapters(run_dir)],
        getattr(args, 'policy_head_dtype', 'auto'))
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vpo_rm.integration import checked_sampling_params, vllm_support_kwargs
    tokenizer_source = resolve_actor_tokenizer_source(
        args.model, tokenizer_name=getattr(args, 'tokenizer', ''))
    tokenizer = load_actor_tokenizer(args.model, tokenizer_name=tokenizer_source)
    token_prompts = tokenize_rendered_prompts(tokenizer, rendered)
    llm = LLM(model=args.model, tokenizer=tokenizer_source,
              dtype="bfloat16", trust_remote_code=True, generation_config="vllm",
              enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=4096, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=0.45, tensor_parallel_size=1, seed=args.seed,
              **policy_engine_kwargs(head))
    from transformers import AutoConfig
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    support_kwargs = vllm_support_kwargs(tokenizer, vocab_size)
    params = {t: checked_sampling_params(SamplingParams, temperature=validate_temperature(t),
                                top_p=1.0 if t == 0 else _top_p(),
                                top_k=-1, n=1,
                                max_tokens=args.max_tokens, seed=args.seed,
                                min_tokens=getattr(args, 'min_tokens', 0),
                                stop_token_ids=list(get_stop_token_ids(tokenizer)),
                                **support_kwargs,
                                presence_penalty=_presence_penalty())
              for t in temps}
    generations = {}
    adapter_id = 0
    for label, run_dir in runs:
        for step, adapter in discover_adapters(run_dir):
            validate_adapter_base(adapter, args.model)
            adapter_id += 1
            lora = (None if str(adapter) == 'none' else
                    LoRARequest(f"{label}-step-{step}", adapter_id, str(adapter)))
            for t in temps:
                t0 = time.monotonic()
                outs = llm.generate(token_prompts, params[t], lora_request=lora)
                validate_outputs(outs, len(rendered), 1)
                if any(getattr(output, 'prompt_token_ids', None) != prompt['prompt_token_ids']
                       for output, prompt in zip(outs, token_prompts)):
                    raise ValueError('vLLM prompt token IDs differ from the rendered chat protocol')
                toks = [list(o.token_ids) if hasattr(o, "token_ids")
                        else list(o.outputs[0].token_ids) for o in outs]
                generations[(label, step, t)] = toks
                print(f"gen {label} step={step} temp={t}: {len(toks)} responses "
                      f"mean_len={statistics.mean(map(len, toks)):.0f} "
                      f"({time.monotonic()-t0:.0f}s)", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return generations


def reward_rows(actor_tokenizer, reward_tokenizer, prompts, responses):
    """Serialize decoded completions through the shared canonical RM protocol."""
    if len(prompts) != len(responses):
        raise ValueError("RM responses must cover all prompts exactly once")
    from vpo_rm.reward_inputs import canonical_reward_input
    return [canonical_reward_input(reward_tokenizer, str(prompt),
                actor_tokenizer.decode(response, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False))
            for prompt, response in zip(prompts, responses)]


def score_all(generations, prompts, temps, args):
    """Unshaped Skywork RM scores with canonical complete conversations."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from vpo_rm.reward import LastTokenReward
    device = getattr(args, 'rm_device', 'cuda:1')
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        print(f'RM loading on {device}: free={free / 2**30:.1f} GiB '
              f'total={total / 2**30:.1f} GiB', flush=True)
    rtok = AutoTokenizer.from_pretrained(args.rm, padding_side="right", trust_remote_code=True)
    atok = load_actor_tokenizer(args.model, tokenizer_name=getattr(args, 'tokenizer', ''))
    configure_model_padding(rtok, fallback_token=getattr(rtok, 'pad_token', None))
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.rm, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    backbone = getattr(rm_base, "base_model", None) or getattr(rm_base, "model", None)
    reward = LastTokenReward(backbone, rm_base.score).to(device).eval()
    scores = {}
    with torch.no_grad():
        for (label, step, t), responses in sorted(generations.items()):
            rows = reward_rows(atok, rtok, prompts, responses)
            order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
            vals = [0.0] * len(rows)
            micro = args.rm_microbatch
            for lo in range(0, len(order), micro):
                idx = order[lo:lo + micro]
                width = max(len(rows[i]) for i in idx)
                ids = torch.full((len(idx), width), rtok.pad_token_id, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, i in enumerate(idx):
                    ids[j, :len(rows[i])] = torch.tensor(rows[i], dtype=torch.long)
                    mask[j, :len(rows[i])] = 1
                out = reward(inputs_embeds=reward.get_input_embeddings()(ids.to(device)),
                             attention_mask=mask.to(device))
                if out.shape != (len(idx),) or not torch.isfinite(out).all():
                    raise ValueError('RM must return finite scalar scores for every response')
                for j, i in enumerate(idx):
                    vals[i] = float(out[j])
            scores[(label, step, t)] = vals
            print(f"rm  {label} step={step} temp={t}: mean={statistics.mean(vals):.3f}", flush=True)
    return scores


def validate_scores(generations, scores, num_prompts):
    if not generations or set(generations) != set(scores):
        raise ValueError('RM scores must cover every generated policy/temperature')
    for key, values in scores.items():
        if len(values) != num_prompts or len(generations[key]) != num_prompts:
            raise ValueError('Incomplete per-prompt reward coverage')
        if not all(math.isfinite(value) for value in values):
            raise ValueError('RM scores must be finite')


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot) - 1]
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                   help='Training run, direct adapter directory, or LABEL=none for Base')
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument('--tokenizer', default='',
                   help='Actor tokenizer artifacts; defaults to --model')
    p.add_argument("--rm", default="models/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--dataset-path",
                   default="datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--temps", type=float, nargs="+", default=[0.7, 0.0])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--min-tokens", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--rm-microbatch", type=int, default=8)
    p.add_argument('--rm-device', choices=('cuda:0', 'cuda:1'), default='cuda:1')
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument('--policy-head-dtype', choices=('auto', 'native', 'float32'), default='auto',
                   help='Actor head precision; auto follows manifests, or native for legacy adapters')
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        args.temps = [validate_temperature(t) for t in args.temps]
    except ValueError as error:
        p.error(str(error))
    required_gpus = 1 if args.rm_device == 'cuda:0' else 2
    if torch.cuda.device_count() < required_gpus:
        raise RuntimeError(f'eval needs {required_gpus} GPUs with RM on {args.rm_device}')
    if min(args.num_prompts, args.max_tokens, args.max_num_seqs, args.rm_microbatch) < 1:
        p.error('Prompt count, token budget, sequence count and microbatch must be positive')
    runs = []
    for spec in args.run:
        if '=' not in spec:
            p.error('Expected --run LABEL=PATH')
        label, path = spec.split("=", 1)
        if not label or not path or any(old == label for old, _ in runs):
            p.error("run labels and paths must be nonempty, with unique labels")
        runs.append((label, Path(path)))

    from vpo_rm.trainer import VPOTrainer
    from transformers import AutoConfig
    from vpo_rm.integration import sampling_summary, vllm_support_kwargs
    args.tokenizer = str(Path(resolve_actor_tokenizer_source(
        args.model, tokenizer_name=args.tokenizer)).resolve())
    atok = load_actor_tokenizer(args.model, tokenizer_name=args.tokenizer)
    actor_vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    support_summary = sampling_summary(vllm_support_kwargs(atok, actor_vocab_size))
    prompts = load_validation_prompts(args.dataset_path, args.num_prompts)
    rendered = [VPOTrainer._render_chat_prompt(atok, p) for p in prompts]
    prompt_token_ids = [row['prompt_token_ids'] for row in tokenize_rendered_prompts(atok, rendered)]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = [(label, step, adapter) for label, run_dir in runs for step, adapter in discover_adapters(run_dir)]
    if not checkpoints:
        raise ValueError('No adapter checkpoints discovered')
    for _, _, adapter in checkpoints:
        validate_adapter_base(adapter, args.model)
    head, policies = resolve_shared_policy([adapter for _, _, adapter in checkpoints], args.policy_head_dtype)
    config = {'args': vars(args), 'top_p': _top_p(), 'presence_penalty': _presence_penalty(),
              'model': fingerprint(args.model, full_weights=False),
              'tokenizer': {'source': args.tokenizer,
                            'fingerprint': fingerprint(args.tokenizer, full_weights=False)},
              'prompt_token_ids_sha256': digest(prompt_token_ids),
              'rm': fingerprint(args.rm, full_weights=False), 'dataset': fingerprint(args.dataset_path),
              'adapters': {f'{label}/{step}': None if str(adapter) == 'none' else fingerprint(adapter)
                           for label, step, adapter in checkpoints},
              'prompts': prompts, 'source_sha256': file_hash(__file__),
              'eval_artifacts_sha256': file_hash(ROOT / 'scripts/eval_artifacts.py'),
              'trainer_sha256': file_hash(ROOT / 'vpo_rm/trainer.py'),
              'token_policy_sha256': file_hash(ROOT / 'vpo_rm/token_policy.py'),
              'integration_sha256': file_hash(ROOT / 'vpo_rm/integration.py'),
              'alignment_sha256': file_hash(ROOT / 'vpo_rm/alignment.py'),
              'reward_inputs_sha256': file_hash(ROOT / 'vpo_rm/reward_inputs.py'),
              'reward_sha256': file_hash(ROOT / 'vpo_rm/reward.py'),
              'eval_policy_sha256': file_hash(ROOT / 'scripts/eval_policy.py'),
              'model_identity_sha256': file_hash(ROOT / 'vpo_rm/model_identity.py'),
              'reward_input_protocol': 'canonical_chat_v1',
              'reward_metric': 'raw_skywork_scalar_no_length_or_kl_penalty',
              'top_k': -1, 'samples_per_prompt': 1,
              'runtime_versions': runtime_versions()}
    config['policy_head_dtype'] = head
    config['policies'] = {f'{label}/{step}': policy
                          for (label, step, _), policy in zip(checkpoints, policies)}
    config['output_support'] = support_summary
    files = [out / name for name in ('eval_prompts.json', 'eval.jsonl', 'summary.json', 'generations.jsonl')]
    manifest = out / 'manifest.json'
    if cache_matches(manifest, config, files):
        print('Verified cached RM evaluation; skipping', flush=True)
        return
    atomic_text(out / 'eval_prompts.json', json.dumps(prompts))
    print(f"eval set: {len(prompts)} frozen validation prompts, temps={args.temps}", flush=True)

    generations = generate_all(runs, rendered, args.temps, args)
    scores = score_all(generations, prompts, args.temps, args)
    validate_scores(generations, scores, len(prompts))

    eval_rows, generation_rows = [], []
    for (label, step, t), vals in sorted(scores.items()):
        toks = generations[(label, step, t)]
        for i, (score, resp) in enumerate(zip(vals, toks)):
            identity = {'run': label, 'step': step, 'temp': t, 'prompt': i}
            eval_rows.append({**identity, 'score': score, 'response_tokens': len(resp)})
            generation_rows.append({**identity, 'token_ids': resp})
    atomic_text(out / 'eval.jsonl', '\n'.join(json.dumps(row) for row in eval_rows))
    atomic_text(out / 'generations.jsonl', '\n'.join(json.dumps(row) for row in generation_rows))
    summary = {}
    for (label, step, t), vals in sorted(scores.items()):
        toks = generations[(label, step, t)]
        lo, hi = bootstrap_ci(vals)
        summary.setdefault(label, {}).setdefault(str(step), {})[str(t)] = {
            "mean": statistics.mean(vals), "ci95": [lo, hi],
            "mean_response_tokens": statistics.mean(map(len, toks)),
            "at_token_cap_rate": sum(len(ids) >= args.max_tokens for ids in toks) / len(toks),
            "reward_metric": "raw_skywork_scalar_no_length_or_kl_penalty",
            "n": len(vals)}
    atomic_text(out / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
    commit_cache(manifest, config, files)
    print(f"wrote {out}/eval.jsonl and summary.json", flush=True)


if __name__ == "__main__":
    main()
