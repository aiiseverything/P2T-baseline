#!/usr/bin/env python3
"""Generate once with vLLM, write token IDs, then exit to release GPU memory."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vpo_rm.integration import (vllm_sampling_kwargs, generation_payload,
                                checked_sampling_params, sampling_summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", required=True, help="JSON list of rendered prompts")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoConfig, AutoTokenizer

    prompts = json.loads(Path(args.prompts).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    sampling = vllm_sampling_kwargs(tokenizer, vocab_size, {
        "max_tokens": args.max_tokens, "min_tokens": args.min_tokens,
        "temperature": args.temperature, "group_size": args.group_size,
        "probe": args.probe, "presence_penalty": float(os.environ.get("PRESENCE_PENALTY", "0"))})
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=1,
              seed=args.seed, generation_config="vllm",
              max_model_len=4096, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=0.45, tensor_parallel_size=1)
    params = checked_sampling_params(SamplingParams, **sampling)
    request = LoRARequest("vpo-once", 1, args.adapter)
    generated = llm.generate(prompts, params, lora_request=request)
    payload = generation_payload(generated, sampling["stop_token_ids"])
    payload["sampling"] = sampling_summary(sampling)
    Path(args.output).write_text(json.dumps(payload))
    print(f"vllm_generation=OK requests={len(payload['rows'])} tokens={payload['tokens']}")


if __name__ == "__main__":
    main()
