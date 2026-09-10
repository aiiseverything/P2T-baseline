#!/usr/bin/env python3
"""Generate once with vLLM, write token IDs, then exit to release GPU memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", required=True, help="JSON list of rendered prompts")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = json.loads(Path(args.prompts).read_text())
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=4096, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=0.45, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0 if args.probe else 1.0,
                            top_p=1.0, top_k=0, max_tokens=args.max_tokens,
                            n=1 if args.probe else args.group_size)
    request = LoRARequest("vpo-once", 1, args.adapter)
    generated = llm.generate(prompts, params, lora_request=request)
    rows = []
    for result in generated:
        rows.extend([list(x.token_ids) for x in result.outputs])
    Path(args.output).write_text(json.dumps(rows))
    print(f"vllm_generation=OK requests={len(rows)} tokens={sum(map(len, rows))}")


if __name__ == "__main__":
    main()
