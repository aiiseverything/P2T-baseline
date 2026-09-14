#!/usr/bin/env python3
"""IFEval benchmark: generate responses with vLLM, score with Google's official code.

Uses the exact evaluation logic from google-research/instruction_following_eval
(downloaded to third_party/ifeval/), matching the paper's metrics.

Each "adapter" entry is TAG=PATH and is evaluated inside one shared vLLM engine,
so checkpoint sweeps amortize the model load. The literal path "none" evaluates
the bare base model without LoRA. Sampling is configured per "recipe"
temp:n_samples:top_p:top_k; multi-sample recipes report mean@N.

Usage (in rjob):
  python3 scripts/eval_ifeval.py \
    --output runs/ifeval-evals/vpo-p11-lam4struct \
    --adapters step-50=runs/.../vllm-adapters/step-50 step-100=... \
    --recipes 1.0:1:1.0:-1

  python3 scripts/eval_ifeval.py --selftest   # offline scoring check, no GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ifeval"))


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


def make_inputs(data):
    class Inp:
        """Official evaluation_lib only reads these four attributes."""

        def __init__(self, row):
            self.prompt = row["prompt"]
            self.instruction_id_list = row["instruction_id_list"]
            self.kwargs = row["kwargs"]
            self.key = row["key"]

    return [Inp(row) for row in data]


def score_one_sample(inp_list, responses, evaluation_lib):
    """Run the official strict + loose checkers on one response per prompt."""
    prompt_to_response = {inp.prompt: resp for inp, resp in zip(inp_list, responses)}
    strict = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inp_list]
    loose = [evaluation_lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inp_list]
    return strict, loose


def aggregate(inp_list, strict, loose) -> dict:
    n = len(strict)
    all_s = [f for r in strict for f in r.follow_instruction_list]
    all_l = [f for r in loose for f in r.follow_instruction_list]
    per_type = {}
    for inp, sr, lr in zip(inp_list, strict, loose):
        for iid, s, l in zip(inp.instruction_id_list,
                             sr.follow_instruction_list, lr.follow_instruction_list):
            key = iid.split(":")[0]
            per_type.setdefault(key, []).append((s, l))
    return {
        "prompt_strict": sum(r.follow_all_instructions for r in strict) / n,
        "prompt_loose": sum(r.follow_all_instructions for r in loose) / n,
        "inst_strict": sum(all_s) / len(all_s),
        "inst_loose": sum(all_l) / len(all_l),
        "per_constraint": {k: {"strict": sum(s for s, _ in v) / len(v),
                                "loose": sum(l for _, l in v) / len(v),
                                "count": len(v)}
                           for k, v in per_type.items()},
    }


def mean_over_samples(metrics_list) -> dict:
    """Average scalar and per-constraint metrics across samples (same prompts,
    so per-constraint counts are identical and a plain mean is exact)."""
    k = len(metrics_list)
    out = {key: sum(m[key] for m in metrics_list) / k
           for key in ["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"]}
    keys = set().union(*(m["per_constraint"] for m in metrics_list))
    out["per_constraint"] = {
        key: {field: sum(m["per_constraint"].get(key, {}).get(field, 0.0) for m in metrics_list) / k
              for field in ["strict", "loose"]}
        for key in keys
    }
    return out


def run_selftest(args) -> None:
    """Exercise the scoring/aggregation path offline with canned responses."""
    from instruction_following_eval import evaluation_lib

    data = [json.loads(l) for l in open(args.dataset)]
    inp_list = make_inputs(data[:40])
    # Two deliberately different "samples": near-empty vs long single line.
    samples = [["Hello."] * len(inp_list), ["word " * 120] * len(inp_list)]
    per_sample = []
    for responses in samples:
        strict, loose = score_one_sample(inp_list, responses, evaluation_lib)
        per_sample.append(aggregate(inp_list, strict, loose))
    mean = mean_over_samples(per_sample)
    for tag, m in zip(["sample0", "sample1"], per_sample):
        print(f"{tag}: strict={m['prompt_strict']:.4f} loose={m['prompt_loose']:.4f} "
              f"inst={m['inst_strict']:.4f}")
    print(f"mean@2:  strict={mean['prompt_strict']:.4f} loose={mean['prompt_loose']:.4f}")
    for m in per_sample:
        for key in ["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"]:
            assert 0.0 <= m[key] <= 1.0
    assert abs(mean["prompt_strict"]
               - sum(m["prompt_strict"] for m in per_sample) / 2) < 1e-12
    assert set(mean["per_constraint"]) == set(per_sample[0]["per_constraint"])
    # Bad recipes / adapter specs must be rejected.
    for bad in ["0.7", "0.0:5:1.0:-1", "-0.1:1:1.0:-1", "0.7:0:1.0:-1", "0.7:1:0:-1"]:
        try:
            parse_recipe(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"recipe '{bad}' should have been rejected")
    for bad in [["step-50"], ["=runs/x"], ["tag="], []]:
        try:
            parse_adapters(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"adapters {bad} should have been rejected")
    assert parse_adapters(["base=none", "step-50=/abs/path"])[0][1] == "none"
    assert parse_adapters(["a=x", "b=y"])[1] == ("b", "y")
    print("selftest OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--adapters", nargs="+", default=[],
                   help="Adapter entries TAG=PATH (or TAG=none for the bare base model)")
    p.add_argument("--dataset", default="datasets/ifeval/ifeval_input_data.jsonl")
    p.add_argument("--output", default="",
                   help="output dir (required unless --selftest)")
    p.add_argument("--recipes", nargs="+", default=["1.0:1:1.0:-1"],
                   help="Sampling recipes temp:n_samples:top_p:top_k "
                        "(default: training temperature, single sample)")
    p.add_argument("--max-tokens", type=int, default=1280)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="Run the offline scoring selftest and exit (no GPU)")
    args = p.parse_args()
    recipes = [parse_recipe(s) for s in args.recipes]

    if args.selftest:
        run_selftest(args)
        return

    adapters = parse_adapters(args.adapters) if args.adapters else [("model", "none")]
    if not args.output:
        p.error("--output is required unless --selftest")

    # Install check: official evaluation code needs absl, immutabledict, langdetect
    from instruction_following_eval import instructions_registry
    from instruction_following_eval import evaluation_lib
    print(f"Official code loaded, {len(instructions_registry.INSTRUCTION_DICT)} constraint types", flush=True)

    # Load IFEval data
    data = [json.loads(l) for l in open(args.dataset)]
    print(f"Loaded {len(data)} prompts", flush=True)
    inp_list = make_inputs(data)

    # vLLM generation
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer, AutoConfig
    from vpo_rm.trainer import VPOTrainer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rendered = [VPOTrainer._render_chat_prompt(tokenizer, row["prompt"]) for row in data]

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
                top_k=None if recipe["top_k"] == -1 else recipe["top_k"],
                n=recipe["n"], max_tokens=args.max_tokens, seed=args.seed,
                stop_token_ids=[151643, 151645], logit_bias=banned)
            outputs = llm.generate(rendered, params, lora_request=lora)
            n_actual = len(outputs[0].outputs)
            if n_actual != recipe["n"]:
                raise RuntimeError(f"vLLM returned {n_actual} samples, expected {recipe['n']}")
            # sample-major: responses[s][i] = sample s for prompt i
            responses_by_sample = [[o.outputs[s].text for o in outputs] for s in range(n_actual)]
            tokens_by_prompt = [sorted(len(o.outputs[s].token_ids) for s in range(n_actual)) for o in outputs]
            print(f"[{tag}/{rectag}] generated {len(outputs)} prompts x {n_actual} samples "
                  f"(top_p={recipe['top_p']}, top_k={recipe['top_k']})", flush=True)

            per_sample, details = [], []
            for s, responses in enumerate(responses_by_sample):
                strict, loose = score_one_sample(inp_list, responses, evaluation_lib)
                per_sample.append(aggregate(inp_list, strict, loose))
                for inp, sr, lr in zip(inp_list, strict, loose):
                    details.append({"sample": s, "key": inp.key, "prompt": inp.prompt[:100],
                                    "strict_all": sr.follow_all_instructions,
                                    "loose_all": lr.follow_all_instructions,
                                    "strict_list": sr.follow_instruction_list,
                                    "loose_list": lr.follow_instruction_list})
            mean_metrics = mean_over_samples(per_sample) if n_actual > 1 else per_sample[0]

            mean_len = sum(sum(l) / len(l) for l in tokens_by_prompt) / len(tokens_by_prompt)
            p95_len = sorted(l[-1] for l in tokens_by_prompt)[int(0.95 * len(tokens_by_prompt))]
            print(f"\n{'=' * 55}")
            print(f"  IFEval [{tag} / {rectag}] — adapter: {path}")
            print(f"{'=' * 55}")
            for s, m in enumerate(per_sample):
                print(f"  sample {s}: strict={m['prompt_strict']:.4f} loose={m['prompt_loose']:.4f}")
            print(f"  mean@{n_actual}: strict={mean_metrics['prompt_strict']:.4f} "
                  f"loose={mean_metrics['prompt_loose']:.4f} "
                  f"inst_strict={mean_metrics['inst_strict']:.4f} "
                  f"inst_loose={mean_metrics['inst_loose']:.4f}")
            print(f"  response length: mean={mean_len:.0f} p95={p95_len} tokens")
            print(f"\n  By constraint type (strict, mean@{n_actual}):")
            for key in sorted(mean_metrics["per_constraint"]):
                print(f"    {key:<30s} {mean_metrics['per_constraint'][key]['strict']:.3f}")

            result_path.write_text(json.dumps({
                "adapter": str(path), "tag": tag,
                "recipe": recipe, "seed": args.seed, "max_tokens": args.max_tokens,
                **mean_metrics,
                "per_sample": per_sample,
                "response_length_mean": mean_len, "response_length_p95": p95_len,
                "details": details,
            }, indent=1, ensure_ascii=False))
            (tag_dir / f"generations_{rectag}.jsonl").write_text("\n".join(json.dumps({
                "key": inp.key, "prompt": inp.prompt,
                "responses": [responses_by_sample[s][i] for s in range(n_actual)],
                "response_tokens": tokens_by_prompt[i],
            }) for i, inp in enumerate(inp_list)))
            print(f"  Saved to {result_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
