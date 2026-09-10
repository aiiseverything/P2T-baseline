#!/usr/bin/env python3
"""Offline RM eval of saved LoRA adapters on a frozen prompt set.

Evaluates every vllm-adapters/step-* checkpoint of one or more training runs on
the SAME 256 validation prompts (first 256 of the SHA256-ordered 2000-prompt
validation split) so training progress is comparable across runs, checkpoints,
and temperatures.  Generation runs on cuda:0 via in-process vLLM; scoring on
cuda:1 with the Skywork RM using the trainer's exact chat-template prefixing.

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
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def load_validation_prompts(dataset_path: str, num_prompts: int) -> list[str]:
    from vpo_rm.trainer import load_prompt_dataset, split_prompts
    prompts, valid, _ = load_prompt_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", dataset_path=dataset_path)
    if len(valid) < num_prompts:
        raise RuntimeError(f"validation split has {len(valid)} prompts < {num_prompts}")
    return valid[:num_prompts]


def discover_adapters(run_dir: Path) -> list[tuple[int, Path]]:
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


def generate_all(runs, rendered, temps, args):
    """In-process vLLM: one pass over (run, step, temp); returns token-id lists."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=4096, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=0.45, tensor_parallel_size=1, seed=args.seed)
    params = {t: SamplingParams(temperature=t, top_p=0.9 if t > 0 else 1.0,
                                max_tokens=args.max_tokens, seed=args.seed,
                                min_tokens=16, stop_token_ids=[151643, 151645])
              for t in temps}
    generations = {}
    for label, run_dir in runs:
        for step, adapter in discover_adapters(run_dir):
            lora = LoRARequest(f"step-{step}", step + 1, str(adapter))
            for t in temps:
                t0 = time.monotonic()
                outs = llm.generate(rendered, params[t], lora_request=lora)
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


def score_all(generations, prompts, temps, args):
    """Skywork RM scoring on cuda:1 with the trainer's prefix format."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from vpo_rm.reward import LastTokenReward
    from vpo_rm.trainer import VPOTrainer
    rtok = AutoTokenizer.from_pretrained(args.rm, padding_side="right", trust_remote_code=True)
    if rtok.pad_token_id is None:
        rtok.pad_token = rtok.eos_token
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.rm, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:1").eval()
    backbone = getattr(rm_base, "base_model", None) or getattr(rm_base, "model", None)
    reward = LastTokenReward(backbone, rm_base.score).to("cuda:1").eval()
    prefixes = [VPOTrainer._render_chat_prompt(rtok, str(p), tokenize=True) for p in prompts]
    if prefixes and hasattr(prefixes[0], "input_ids"):
        prefixes = [p.input_ids for p in prefixes]
    prefixes = [list(map(int, p)) for p in prefixes]

    scores = {}
    with torch.no_grad():
        for (label, step, t), responses in sorted(generations.items()):
            rows = [prefix + resp for prefix, resp in zip(prefixes, responses)]
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
                out = reward(inputs_embeds=reward.get_input_embeddings()(ids.to("cuda:1")),
                             attention_mask=mask.to("cuda:1"))
                for j, i in enumerate(idx):
                    vals[i] = float(out[j])
            scores[(label, step, t)] = vals
            print(f"rm  {label} step={step} temp={t}: mean={statistics.mean(vals):.3f}", flush=True)
    return scores


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot) - 1]
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--rm", default="models/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--dataset-path",
                   default="datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--temps", type=float, nargs="+", default=[0.7, 0.0])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--rm-microbatch", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("eval needs two GPUs: cuda:0 generation, cuda:1 RM")
    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs.append((label, Path(path)))

    from vpo_rm.trainer import VPOTrainer
    from transformers import AutoTokenizer
    atok = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    prompts = load_validation_prompts(args.dataset_path, args.num_prompts)
    rendered = [VPOTrainer._render_chat_prompt(atok, p) for p in prompts]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_prompts.json").write_text(json.dumps(prompts))
    print(f"eval set: {len(prompts)} frozen validation prompts, temps={args.temps}", flush=True)

    generations = generate_all(runs, rendered, args.temps, args)
    scores = score_all(generations, prompts, args.temps, args)

    with (out / "eval.jsonl").open("w") as f:
        for (label, step, t), vals in sorted(scores.items()):
            toks = generations[(label, step, t)]
            for i, (score, resp) in enumerate(zip(vals, toks)):
                f.write(json.dumps({"run": label, "step": step, "temp": t,
                                    "prompt": i, "score": score,
                                    "response_tokens": len(resp)}) + "\n")
    summary = {}
    for (label, step, t), vals in sorted(scores.items()):
        toks = generations[(label, step, t)]
        lo, hi = bootstrap_ci(vals)
        summary.setdefault(label, {}).setdefault(str(step), {})[str(t)] = {
            "mean": statistics.mean(vals), "ci95": [lo, hi],
            "mean_response_tokens": statistics.mean(map(len, toks)),
            "n": len(vals)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out}/eval.jsonl and summary.json", flush=True)


if __name__ == "__main__":
    main()
