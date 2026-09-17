#!/usr/bin/env python3
"""Generate raw HumanEval function completions; never execute generated code."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import (atomic_text, file_hash, fingerprint,
                                    runtime_versions, validate_adapter_base, validate_outputs)


def load_problems(path, expected_count=164):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate HumanEval task IDs")
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} HumanEval tasks, found {len(rows)}")
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key]
               for key in ("task_id", "prompt", "test", "entry_point")):
            raise ValueError("Missing HumanEval problem fields")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-14B-Base")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", default="datasets/humaneval/HumanEval.jsonl.gz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-step", type=int, default=250)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 2048:
        parser.error("max-tokens must be in [1, 2048]")
    problems = load_problems(args.dataset)
    adapter = Path(args.adapter).resolve()
    recorded = json.loads((adapter / "run_manifest.json").read_text())
    if recorded.get("step") != args.expected_step:
        raise ValueError("Checkpoint step does not match the requested final checkpoint")
    validate_adapter_base(adapter, args.model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a fresh generation output directory: {output}")

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vpo_rm.integration import checked_sampling_params, sampling_summary, vllm_support_kwargs
    from vpo_rm.token_policy import get_stop_token_ids
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    stop_ids = list(get_stop_token_ids(tokenizer))
    support = vllm_support_kwargs(tokenizer, config.vocab_size)
    # HumanEval's native completion prompt is already the function prefix.
    # No chat formatting, answer extraction, test leakage, or whitespace stripping.
    prompts = [problem["prompt"] for problem in problems]
    if any(len(tokenizer.encode(p, add_special_tokens=False)) + args.max_tokens > 4096 for p in prompts):
        raise ValueError("HumanEval prompt exceeds the configured context window")
    provenance = {"benchmark":"HumanEval", "tasks":len(problems), "tag":args.tag,
        "protocol":"native_function_completion_raw_v1", "postprocessing":"none",
        "temperature":0.0, "n":1, "top_p":1.0, "top_k":-1,
        "max_tokens":args.max_tokens, "min_tokens":0, "seed":args.seed,
        "stop_token_ids":stop_ids, "output_support":sampling_summary(support),
        "dataset_sha256":file_hash(args.dataset), "model":fingerprint(args.model, full_weights=False),
        "adapter":str(adapter), "checkpoint_step":args.expected_step,
        "adapter_weights_sha256":file_hash(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256":file_hash(adapter / "adapter_config.json"),
        "runtime_versions":runtime_versions(("torch", "transformers", "vllm", "peft")),
        "source_sha256":{name:file_hash(ROOT / name) for name in
            ("scripts/eval_humaneval.py", "scripts/eval_artifacts.py", "vpo_rm/integration.py", "vpo_rm/token_policy.py")}}
    atomic_text(output / "generation_config.json", json.dumps(provenance, indent=2))
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=False,
        generation_config="vllm", enable_lora=True, max_lora_rank=64, max_loras=1,
        max_model_len=4096, max_num_seqs=32, gpu_memory_utilization=.80,
        tensor_parallel_size=1, seed=args.seed)
    params = checked_sampling_params(SamplingParams, temperature=0.0, n=1, top_p=1.0,
        top_k=-1, max_tokens=args.max_tokens, min_tokens=0, seed=args.seed,
        stop_token_ids=stop_ids, **support)
    outputs = llm.generate(prompts, params, lora_request=LoRARequest(args.tag, 1, str(adapter)))
    validate_outputs(outputs, len(problems), 1)
    samples, raw = [], []
    for problem, result in zip(problems, outputs):
        generated = result.outputs[0]
        samples.append({"task_id":problem["task_id"], "completion":generated.text})
        raw.append({**samples[-1], "response_tokens":len(generated.token_ids),
            "token_ids":list(generated.token_ids), "finish_reason":generated.finish_reason,
            "stop_reason":generated.stop_reason})
    atomic_text(output / "samples.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in samples))
    atomic_text(output / "generations.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in raw))
    atomic_text(output / "generation_complete.json", json.dumps({"tasks":len(samples),
        "samples_sha256":file_hash(output / "samples.jsonl"),
        "generations_sha256":file_hash(output / "generations.jsonl"),
        "response_tokens_mean":sum(x["response_tokens"] for x in raw)/len(raw),
        "truncated":sum(x["finish_reason"] == "length" for x in raw)}, indent=2))
    print(f"HUMANEVAL_GENERATION_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
