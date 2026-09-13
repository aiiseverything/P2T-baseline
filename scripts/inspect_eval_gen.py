#!/usr/bin/env python3
"""Decode what a checkpoint actually generates on eval prompts.

Writes plain-text generations (both PP variants) for qualitative inspection —
the eval pipeline only stores scores, this recovers the behavior behind them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run dir (has vllm-adapters/)")
    p.add_argument("--step", type=int, default=500)
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--eval-prompts", default="runs/eval-grpo-p9g/eval_prompts.json")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--temps", type=float, nargs="+", default=[0.7])
    p.add_argument("--pps", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vpo_rm.trainer import VPOTrainer

    atok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    known = set(atok.get_vocab().values())
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    banned = {i: -100.0 for i in range(vocab_size) if i not in known}

    prompts = json.loads(Path(args.eval_prompts).read_text())[:args.num_prompts]
    rendered = [VPOTrainer._render_chat_prompt(atok, x) for x in prompts]

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=4096, max_num_seqs=16,
              gpu_memory_utilization=0.45, tensor_parallel_size=1, seed=1234)
    adapter = Path(args.run) / "vllm-adapters" / f"step-{args.step}"
    lora = LoRARequest("inspect", 1, str(adapter))

    out = {}
    for pp in args.pps:
        for t in args.temps:
            params = SamplingParams(temperature=t, top_p=0.9 if t > 0 else 1.0,
                                    max_tokens=2048, seed=1234, min_tokens=16,
                                    stop_token_ids=[151643, 151645],
                                    logit_bias=banned, presence_penalty=pp)
            outs = llm.generate(rendered, params, lora_request=lora)
            key = f"pp{pp}_t{t}"
            out[key] = [o.outputs[0].text for o in outs]
            print(f"[{key}] mean_len={sum(len(o.outputs[0].token_ids) for o in outs)/len(outs):.0f}", flush=True)

    Path(args.output).write_text(json.dumps(
        {"prompts": prompts, "generations": out}, indent=1, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
