#!/usr/bin/env python3
"""IFEval benchmark: generate responses with vLLM, score with Google's official code.

Uses the exact evaluation logic from google-research/instruction_following_eval
(downloaded to third_party/ifeval/), matching the paper's metrics.

Usage (in rjob):
  python3 scripts/eval_ifeval.py \
    --adapter models/sft-init-qwen3-14b-base \
    --output runs/ifeval-sft-init
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ifeval"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--adapter", required=True)
    p.add_argument("--dataset", default="datasets/ifeval/ifeval_input_data.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7])
    p.add_argument("--max-tokens", type=int, default=1280)
    args = p.parse_args()

    # Install check: official evaluation code needs absl, immutabledict, langdetect
    from instruction_following_eval import instructions_registry
    from instruction_following_eval import evaluation_lib
    print(f"Official code loaded, {len(instructions_registry.INSTRUCTION_DICT)} constraint types", flush=True)

    # Load IFEval data
    data = [json.loads(l) for l in open(args.dataset)]
    print(f"Loaded {len(data)} prompts", flush=True)

    # Render prompts
    from transformers import AutoTokenizer
    from vpo_rm.trainer import VPOTrainer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, row["prompt"]) for row in data]

    # vLLM generation
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoConfig
    known = set(tokenizer.get_vocab().values())
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    banned = {i: -100.0 for i in range(vocab_size) if i not in known}

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=1,
              max_model_len=4096, max_num_seqs=32,
              gpu_memory_utilization=0.45, tensor_parallel_size=1, seed=42)
    lora = LoRARequest("ifeval", 1, str(Path(args.adapter)))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for temp in args.temps:
        params = SamplingParams(temperature=temp,
                                top_p=1.0 if temp == 0 else 0.9,
                                max_tokens=args.max_tokens, seed=42,
                                stop_token_ids=[151643, 151645],
                                logit_bias=banned)
        outputs = llm.generate(rendered, params, lora_request=lora)
        responses = [o.outputs[0].text for o in outputs]
        print(f"Generated {len(responses)} responses at temp={temp}", flush=True)

        # Build prompt_to_response dict for official eval
        prompt_to_response = {row["prompt"]: resp for row, resp in zip(data, responses)}

        # Run official evaluation (strict + loose)
        # Create InputExample-like objects
        class Inp:
            def __init__(self, row):
                self.prompt = row["prompt"]
                self.instruction_id_list = row["instruction_id_list"]
                self.kwargs = row["kwargs"]
                self.key = row["key"]

        inp_list = [Inp(row) for row in data]

        # Strict evaluation (exact from evaluation_lib.py)
        strict_results = []
        for inp in inp_list:
            out = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
            strict_results.append(out)

        # Loose evaluation (exact from evaluation_lib.py)
        loose_results = []
        for inp in inp_list:
            out = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
            loose_results.append(out)

        # Compute metrics
        n = len(strict_results)
        prompt_strict_acc = sum(r.follow_all_instructions for r in strict_results) / n
        prompt_loose_acc = sum(r.follow_all_instructions for r in loose_results) / n
        all_strict = [f for r in strict_results for f in r.follow_instruction_list]
        all_loose = [f for r in loose_results for f in r.follow_instruction_list]
        inst_strict_acc = sum(all_strict) / len(all_strict)
        inst_loose_acc = sum(all_loose) / len(all_loose)

        # Per-constraint-type breakdown
        per_type = {}
        for inp, sr, lr in zip(inp_list, strict_results, loose_results):
            for inst_id, s, l in zip(inp.instruction_id_list,
                                      sr.follow_instruction_list,
                                      lr.follow_instruction_list):
                key = inst_id.split(":")[0]
                per_type.setdefault(key, []).append({"strict": s, "loose": l})

        print(f"\n{'='*55}")
        print(f"  IFEval (temp={temp}) — adapter: {args.adapter}")
        print(f"{'='*55}")
        print(f"  Prompt strict:  {prompt_strict_acc:.4f}  ({sum(r.follow_all_instructions for r in strict_results)}/{n})")
        print(f"  Prompt loose:   {prompt_loose_acc:.4f}")
        print(f"  Instr strict:   {inst_strict_acc:.4f}")
        print(f"  Instr loose:    {inst_loose_acc:.4f}")
        print(f"\n  By constraint type:")
        for key in sorted(per_type):
            vals = per_type[key]
            s_acc = sum(v["strict"] for v in vals) / len(vals)
            print(f"    {key:<30s} strict={s_acc:.3f} ({sum(v['strict'] for v in vals)}/{len(vals)})")

        # Save
        (out_dir / f"results_temp{temp}.json").write_text(json.dumps({
            "adapter": str(args.adapter), "temperature": temp,
            "prompt_strict": prompt_strict_acc, "prompt_loose": prompt_loose_acc,
            "inst_strict": inst_strict_acc, "inst_loose": inst_loose_acc,
            "per_constraint": {k: {"strict": sum(v["strict"] for v in vs)/len(vs),
                                    "loose": sum(v["loose"] for v in vs)/len(vs),
                                    "count": len(vs)}
                                for k, vs in per_type.items()},
            "details": [{"key": inp.key,
                          "prompt": inp.prompt[:100],
                          "strict_all": sr.follow_all_instructions,
                          "loose_all": lr.follow_all_instructions,
                          "strict_list": sr.follow_instruction_list,
                          "loose_list": lr.follow_instruction_list}
                         for inp, sr, lr in zip(inp_list, strict_results, loose_results)]
        }, indent=1, ensure_ascii=False))
        print(f"  Saved to {out_dir}/results_temp{temp}.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
