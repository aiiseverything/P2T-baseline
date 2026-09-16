#!/usr/bin/env python3
"""AlpacaEval 2.0 generation sweep (judge runs separately off-cluster).

Generates one response per official 805 instruction for each TAG=PATH
adapter inside one shared vLLM engine ("none" = bare base model). No
scoring here: pairwise judging against the gpt-4-turbo reference happens
in scripts/judge_alpaca.py on a networked machine.

Usage (in rjob):
  python3 scripts/eval_alpaca.py \
    --output runs/alpacaeval-evals \
    --adapters base=none sft-init=runs/.../step-0 ... \
    --recipes 1.0:1:1.0:-1

  python3 scripts/eval_alpaca.py --selftest   # offline checks, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_recipe(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"recipe '{spec}' must be temp:n_samples:top_p:top_k")
    temp, n, top_p, top_k = float(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
    if temp < 0 or n < 1 or not (0 < top_p <= 1.0) or top_k == 0 or top_k < -1:
        raise ValueError(f"out-of-range values in recipe '{spec}'")
    if temp == 0.0 and n > 1:
        raise ValueError(f"greedy (temp 0) cannot draw {n} distinct samples: '{spec}'")
    return {"temp": temp, "n": n, "top_p": top_p, "top_k": top_k}


def parse_adapters(specs):
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"adapter spec '{spec}' must be TAG=PATH (or TAG=none)")
        tag, path = spec.split("=", 1)
        if not tag or not path:
            raise ValueError(f"empty tag or path in '{spec}'")
        out.append((tag, path))
    if not out:
        raise ValueError("no adapters given")
    return out


def run_selftest(args) -> None:
    for bad in ["0.7", "0.0:5:1.0:-1"]:
        try:
            parse_recipe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"recipe '{bad}' should have been rejected")
    try:
        parse_adapters(["=x"])
        raise AssertionError("adapters '=x' should have been rejected")
    except ValueError:
        pass
    data = [json.loads(l) for l in open(args.dataset)]
    assert len(data) == 805, f"expected 805 instructions, got {len(data)}"
    assert all("instruction" in r and r["reference_output"] for r in data), "missing fields"
    try:
        from vllm import SamplingParams
    except ImportError:
        print("selftest OK (no vllm; recipe construction not checked)")
        return
    r = parse_recipe("1.0:1:1.0:-1")
    SamplingParams(temperature=r["temp"], top_p=r["top_p"],
                   top_k=r["top_k"], n=r["n"], max_tokens=16,
                   stop_token_ids=[151643, 151645])
    print("selftest OK (805 instructions; recipes construct under installed vLLM)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--adapters", nargs="+", default=[],
                   help="Adapter entries TAG=PATH (or TAG=none for the bare base model)")
    p.add_argument("--dataset", default="datasets/alpacaeval/eval_gpt4turbo_reference.jsonl")
    p.add_argument("--output", default="",
                   help="output dir (required unless --selftest)")
    p.add_argument("--recipes", nargs="+", default=["1.0:1:1.0:-1"],
                   help="Sampling recipes temp:n_samples:top_p:top_k")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="Run the offline selftest and exit (no GPU)")
    args = p.parse_args()
    recipes = [parse_recipe(s) for s in args.recipes]

    if args.selftest:
        run_selftest(args)
        return

    adapters = parse_adapters(args.adapters) if args.adapters else [("model", "none")]
    if not args.output:
        p.error("--output is required unless --selftest")

    data = [json.loads(l) for l in open(args.dataset)]
    print(f"Loaded {len(data)} AlpacaEval instructions", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer, AutoConfig
    from vpo_rm.trainer import VPOTrainer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, row["instruction"]) for row in data]

    known = set(tokenizer.get_vocab().values())
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    banned = {i: -100.0 for i in range(vocab_size) if i not in known}

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              enable_lora=True, max_lora_rank=64, max_loras=4,
              max_model_len=4096, max_num_seqs=64,
              gpu_memory_utilization=0.80, tensor_parallel_size=1, seed=args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (tag, path) in enumerate(adapters):
        lora = None if path == "none" else LoRARequest(f"lora-{tag}", i + 1, str(Path(path)))
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        for recipe in recipes:
            rectag = f"t{recipe['temp']}_n{recipe['n']}"
            gen_path = tag_dir / f"generations_{rectag}.jsonl"
            if gen_path.exists():
                print(f"[{tag}/{rectag}] generations exist, skipping", flush=True)
                continue
            params = SamplingParams(
                temperature=recipe["temp"], top_p=recipe["top_p"],
                top_k=recipe["top_k"], n=recipe["n"],
                max_tokens=args.max_tokens, seed=args.seed,
                stop_token_ids=[151643, 151645], logit_bias=banned)
            outputs = llm.generate(rendered, params, lora_request=lora)
            n_actual = len(outputs[0].outputs)
            if n_actual != recipe["n"]:
                raise RuntimeError(f"vLLM returned {n_actual} samples, expected {recipe['n']}")

            lens = [len(o.outputs[0].token_ids) for o in outputs]
            n_trunc = sum(t >= args.max_tokens for t in lens)
            print(f"\n{'=' * 50}")
            print(f"  AlpacaEval [{tag} / {rectag}] max_tokens={args.max_tokens} — adapter: {path}")
            print(f"  response tokens: mean {sum(lens)/len(lens):.0f}, "
                  f"pinned at cap {n_trunc}/{len(lens)} ({n_trunc/len(lens):.1%})")

            gen_path.write_text("\n".join(json.dumps({
                "idx": j,
                "instruction": data[j]["instruction"],
                "response": outputs[j].outputs[0].text,
                "response_tokens": lens[j],
            }, ensure_ascii=False) for j in range(len(data))))
            print(f"  Saved to {gen_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
