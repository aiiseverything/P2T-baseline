#!/usr/bin/env python3
"""GSM8K benchmark with rule-based scoring (same sweep architecture as IFEval).

Each adapter entry TAG=PATH is evaluated inside one shared vLLM engine; the
literal path "none" evaluates the bare base model. 0-shot prompting through
the trainer's chat template, asking for the final answer after '####';
scoring extracts that number (fallback: last number in the response) and
compares to the gold answer with 0.1% relative tolerance.

Usage (in rjob):
  python3 scripts/eval_gsm8k.py \
    --output runs/gsm8k-evals/vpo-p11-lam8struct \
    --adapters step-50=runs/.../vllm-adapters/step-50 ... \
    --recipes 1.0:1:1.0:-1

  python3 scripts/eval_gsm8k.py --selftest   # offline scorer check, no GPU
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_SUFFIX = (" Solve the problem step by step. "
                 "End your response with the final numeric answer after '#### '.")


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


_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?%?"


def extract_number(text: str):
    """Final answer: prefer the last '#### N', else the last number written."""
    hits = re.findall(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if not hits:
        hits = re.findall(_NUM, text)
    if not hits:
        return None
    cand = hits[-1].replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(cand)
    except ValueError:
        return None


def gold_number(answer_field: str):
    hits = re.findall(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", answer_field)
    if not hits:
        raise ValueError(f"no '#### N' in gold answer: {answer_field[:80]!r}")
    return float(hits[-1].replace(",", "").replace("$", ""))


def is_correct(pred, gold):
    return pred is not None and abs(pred - gold) <= max(1e-4, abs(gold) * 1e-3)


def run_selftest(args) -> None:
    cases = [
        ("...so she makes $18.\n#### 18", 18.0, True),
        ("The answer is 42.\n#### 42", 42.0, True),
        ("...total is 1,234 dollars\n#### 1234", 1234.0, True),      # comma inside ####
        ("#### 3.50", 3.5, True),                                      # decimals
        ("blah #### 7", 8.0, False),                                   # wrong value
        ("I think it is 5 apples and 3 more, total eight", 8.0, False),  # word number, no digits->5? last num is 3
        ("no numbers here at all", 8.0, False),                        # None pred
        ("answer: -12\n#### -12", -12.0, True),                        # negative
        ("#### 72.", 72.0, True),                                      # trailing period -> num regex takes 72
        ("final answer:\n#### 100%\n", 100.0, True),                   # percent sign stripped
    ]
    for text, gold, expect in cases:
        pred = extract_number(text)
        got = is_correct(pred, gold)
        assert got == expect, f"{text!r}: pred={pred} gold={gold} expected {expect}"
    # fallback path: no #### but numbers present -> last number
    assert extract_number("first 3 then 4 then 6") == 6.0
    assert gold_number("reasoning...\n#### 1,000") == 1000.0
    # recipes / adapters validation shared with eval_ifeval
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
    try:
        from vllm import SamplingParams
    except ImportError:
        print("selftest OK (no vllm; recipe construction not checked)")
        return
    for spec in ["1.0:1:1.0:-1", "0.7:5:0.8:20"]:
        r = parse_recipe(spec)
        SamplingParams(temperature=r["temp"], top_p=r["top_p"],
                       top_k=r["top_k"], n=r["n"], max_tokens=16,
                       stop_token_ids=[151643, 151645])
    print("selftest OK (recipes construct under installed vLLM)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--adapters", nargs="+", default=[],
                   help="Adapter entries TAG=PATH (or TAG=none for the bare base model)")
    p.add_argument("--dataset", default="datasets/gsm8k/test.jsonl")
    p.add_argument("--output", default="",
                   help="output dir (required unless --selftest)")
    p.add_argument("--recipes", nargs="+", default=["1.0:1:1.0:-1"],
                   help="Sampling recipes temp:n_samples:top_p:top_k")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="Run the offline scorer selftest and exit (no GPU)")
    args = p.parse_args()
    recipes = [parse_recipe(s) for s in args.recipes]

    if args.selftest:
        run_selftest(args)
        return

    adapters = parse_adapters(args.adapters) if args.adapters else [("model", "none")]
    if not args.output:
        p.error("--output is required unless --selftest")

    data = [json.loads(l) for l in open(args.dataset)]
    golds = [gold_number(row["answer"]) for row in data]
    print(f"Loaded {len(data)} GSM8K problems", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer, AutoConfig
    from vpo_rm.trainer import VPOTrainer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, row["question"] + PROMPT_SUFFIX)
                for row in data]

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
            result_path = tag_dir / f"results_{rectag}.json"
            if result_path.exists():
                print(f"[{tag}/{rectag}] results exist, skipping", flush=True)
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

            per_sample_acc, details = [], []
            tokens_by_prompt = [sorted(len(o.outputs[s].token_ids) for s in range(n_actual))
                                for o in outputs]
            for s in range(n_actual):
                correct = 0
                for j, o in enumerate(outputs):
                    text = o.outputs[s].text
                    ok = is_correct(extract_number(text), golds[j])
                    correct += ok
                    details.append({"sample": s, "idx": j,
                                    "gold": golds[j], "pred": extract_number(text),
                                    "correct": ok})
                per_sample_acc.append(correct / len(outputs))
            acc = sum(per_sample_acc) / n_actual
            mean_len = sum(sum(l) / len(l) for l in tokens_by_prompt) / len(tokens_by_prompt)

            print(f"\n{'=' * 50}")
            print(f"  GSM8K [{tag} / {rectag}] — adapter: {path}")
            print(f"  accuracy: {acc:.4f} (mean@{n_actual}; per-sample "
                  f"{', '.join(f'{a:.3f}' for a in per_sample_acc)})")
            print(f"  response length mean={mean_len:.0f} tokens")

            result_path.write_text(json.dumps({
                "adapter": str(path), "tag": tag,
                "recipe": recipe, "seed": args.seed,
                "accuracy": acc, "per_sample": per_sample_acc,
                "response_length_mean": mean_len,
                "n_problems": len(data),
                "details": details,
            }, indent=1, ensure_ascii=False))
            print(f"  Saved to {result_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
