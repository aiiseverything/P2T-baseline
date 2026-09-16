#!/usr/bin/env python3
"""Probe P(stop token) at answer-end contexts: base vs SFT adapters.

For each (prompt, gold answer) pair, teacher-force prompt+gold and read the
next-token distribution right after the gold answer's last content token.
Reports P(<|endoftext|>), P(<|im_end|>), and the argmax token, per model.

Usage (in rjob, 1 GPU, ~10 min):
  python3 scripts/probe_stop_probs.py --adapters base=none e2=models/sftv2-clean-2k5e2 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="TAG=PATH entries (none = bare base)")
    p.add_argument("--train", default="datasets/sft_v2/train25.jsonl")
    p.add_argument("--test", default="datasets/sft_v2/test25.jsonl")
    p.add_argument("--gold-map", default="datasets/sft_v2/sft_clean.parquet")
    p.add_argument("--out", default="runs/stop-probe/results.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import pyarrow.parquet as pq

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    END, IMEND = tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")

    gold = {str(r["prompt"]): str(r["chosen"]) for r in
            pq.read_table(args.gold_map).to_pylist()}
    prompts = [json.loads(l)["instruction"] for l in open(args.train)]
    prompts += [json.loads(l)["instruction"] for l in open(args.test)]

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

    results = {}
    for spec in args.adapters:
        tag, path = spec.split("=", 1)
        model = base
        if path != "none":
            model = PeftModel.from_pretrained(base, str(ROOT / path),
                                              adapter_name=tag).eval()
        pe, pi, top, topid = [], [], [], []
        with torch.no_grad():
            for q in prompts:
                g = gold.get(q)
                if not g:
                    continue
                text = tok.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": g}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=False)
                ids = tok(text, add_special_tokens=False,
                          return_tensors="pt").input_ids.cuda()
                # position right after the answer's last CONTENT token:
                # drop the template tail (<|im_end|> and trailing \n)
                n_tail = 2
                logits = model(ids[:, :-n_tail]).logits[0, -1].float()
                pr = torch.softmax(logits, -1)
                pe.append(float(pr[END])); pi.append(float(pr[IMEND]))
                topid.append(int(pr.argmax()))
        n = len(pe)
        from collections import Counter
        results[tag] = {
            "n": n,
            "P_endoftext_mean": sum(pe) / n, "P_im_end_mean": sum(pi) / n,
            "P_either_mean": (sum(a + b for a, b in zip(pe, pi))) / n,
            "argmax_top3": Counter(
                tok.decode([i]) for i in topid).most_common(3),
        }
        if path != "none":
            model = model.unload()
        print(tag, json.dumps(results[tag]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
