#!/usr/bin/env python3
"""Resident vLLM LoRA generation server on dedicated generation GPUs.

Port of the parent project's ``scripts/vllm_generate_server.py``.  The process
owns its GPU allocation for its whole lifetime and serves newline-delimited JSON
over a Unix domain socket, so the engine -- and its CUDA graphs -- survive across
rollouts instead of being rebuilt every step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from .tokens import load_actor_tokenizer, resolve_actor_tokenizer_source
from .vllm import (checked_sampling_params, generation_payload, sampling_summary,
                   tokenize_rendered_prompts, vllm_sampling_kwargs)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="", help="Actor tokenizer source; defaults to the base model")
    p.add_argument("--socket", required=True)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--policy-head-dtype", choices=("native", "float32"), default="float32")
    p.add_argument("--disable-custom-all-reduce", action="store_true",
                   help="Route tensor-parallel all-reduce through NCCL. Required on "
                        "PCIe-bridge topologies without NVLink, where vLLM's custom "
                        "all-reduce raises a CUDA 'invalid argument' and kills the engine.")
    args = p.parse_args(argv)
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        p.error("--gpu-memory-utilization must be in (0, 1]")
    if args.tensor_parallel_size < 1 or args.max_num_seqs < 1:
        p.error("--tensor-parallel-size and --max-num-seqs must be positive")
    return args


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Qwen3 reserves 271 embedding rows beyond the tokenizer's vocabulary.  vLLM
    # samples over all rows, but the trainer only admits realized tokens, so ban
    # the reserved ids here: generation and training must share one support.
    from transformers import AutoConfig
    tokenizer_source = resolve_actor_tokenizer_source(args.model, tokenizer_name=args.tokenizer)
    tokenizer = load_actor_tokenizer(args.model, tokenizer_name=tokenizer_source)
    vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    if float(os.environ.get("PRESENCE_PENALTY", "0.0")) != 0:
        raise ValueError("PRESENCE_PENALTY is unsupported by the on-policy training protocol")

    llm = LLM(
        model=args.model, tokenizer=tokenizer_source, dtype="bfloat16", trust_remote_code=True,
        enable_lora=True, max_lora_rank=64, max_loras=2, max_cpu_loras=2,
        seed=args.seed,
        generation_config="vllm",
        hf_overrides={"head_dtype": "float32"} if args.policy_head_dtype == "float32" else {},
        logprobs_mode="processed_logprobs",
        max_model_len=4096, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    # Fail closed if the backend quietly ignored the request: a silent
    # re-enable is what took the first pilot's engine core down.  The lookup
    # itself is best effort -- an internal attribute path that moves between
    # vLLM versions must not become a new startup crash of its own.
    resolved = None
    try:
        resolved = llm.llm_engine.vllm_config.parallel_config.disable_custom_all_reduce
    except AttributeError:
        pass
    if args.disable_custom_all_reduce and resolved is False:
        raise RuntimeError("vLLM ignored --disable-custom-all-reduce; its custom "
                           "all-reduce is unsafe on this PCIe-bridge topology")
    print(f"custom_all_reduce_disabled={resolved}", flush=True)
    sock_path = Path(args.socket)
    sock_path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(1)
    print("vllm_server_ready", flush=True)
    try:
        while True:
            conn, _ = listener.accept()
            with conn:
                reader = conn.makefile("r")
                for line in reader:
                    if not line.strip():
                        continue
                    request = json.loads(line)
                    if request.get("shutdown"):
                        conn.sendall(b'{"ok":true}\n')
                        return
                    try:
                        prompts = tokenize_rendered_prompts(
                            tokenizer, request["prompts"], request.get("prompt_token_ids"))
                        adapter = request["adapter"]
                        adapter_id = int(request["adapter_id"])
                        sampling = vllm_sampling_kwargs(tokenizer, vocab_size, request)
                        params = checked_sampling_params(SamplingParams, **sampling)
                        lora_request = LoRARequest(f"p2t-step-{adapter_id}", adapter_id, adapter)
                        started = time.monotonic()
                        generated = llm.generate(prompts, params, lora_request=lora_request)
                        actual_prompt_ids = [list(result.prompt_token_ids) for result in generated]
                        if actual_prompt_ids != [p["prompt_token_ids"] for p in prompts]:
                            raise RuntimeError("vLLM changed the explicit prompt token IDs")
                        payload = {"ok": True,
                                   **generation_payload(generated, sampling["stop_token_ids"],
                                                        include_logprobs=bool(request.get("return_logprobs", False))),
                                   "prompt_token_ids": actual_prompt_ids,
                                   "sampling": sampling_summary(sampling),
                                   "logprobs_mode": "processed_logprobs",
                                   "engine_generation_sec": time.monotonic() - started,
                                   "server_pid": os.getpid(), "adapter_id": adapter_id}
                    except Exception as error:  # keep the server alive across request faults
                        payload = {"ok": False, "error": f"{type(error).__name__}: {error}",
                                   "traceback": traceback.format_exc()}
                    conn.sendall((json.dumps(payload) + "\n").encode())
    finally:
        listener.close()
        sock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
