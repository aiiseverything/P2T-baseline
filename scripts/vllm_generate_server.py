#!/usr/bin/env python3
"""Persistent single-GPU vLLM LoRA generation server.

The process owns one GPU for its entire lifetime and serves newline-delimited
JSON requests over a Unix domain socket.  Keeping the engine alive avoids
reloading the base model and CUDA graphs for every rollout.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Qwen3 reserves 271 embedding rows beyond the tokenizer's vocab (model
    # vocab 151936 vs 151665 entries).  vLLM samples over all rows, but the
    # trainer's output support only admits realized tokens, and a rare hit on
    # a reserved id crashed the credit integrity check (p9e rollout 85, p9g
    # smoke rollout 1).  Ban them here so generation and training share the
    # same support, per the contract in vpo_rm/alignment.py.
    from transformers import AutoConfig, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _known = set(_tok.get_vocab().values())
    _vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
    _banned = {i: -100.0 for i in range(_vocab_size) if i not in _known}
    # p9h VPO fix: suppress exact-repetition attractors (the p9g newline-spam
    # entropy collapse).  Off by default; enabled per job via PRESENCE_PENALTY.
    _presence = float(os.environ.get("PRESENCE_PENALTY", "0.0"))

    llm = LLM(
        model=args.model, dtype="bfloat16", trust_remote_code=True,
        enable_lora=True, max_lora_rank=64, max_loras=2, max_cpu_loras=2,
        seed=args.seed,
        max_model_len=4096, max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.45, tensor_parallel_size=1,
    )
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
                    req = json.loads(line)
                    if req.get("shutdown"):
                        conn.sendall(b'{"ok":true}\n')
                        return
                    prompts = req["prompts"]
                    adapter = req["adapter"]
                    adapter_id = int(req["adapter_id"])
                    probe = bool(req.get("probe", False))
                    params = SamplingParams(
                        temperature=0.0 if probe else 1.0,
                        top_p=1.0, top_k=0,
                        max_tokens=int(req["max_tokens"]),
                        n=1 if probe else int(req.get("group_size", 8)),
                        # Qwen3-Base's generation_config stops on <|endoftext|>
                        # (151643) only; the chat template's turn end is
                        # <|im_end|> (151645).  Accept both so a base model
                        # mimicking the template still stops at turn end
                        # instead of rolling into a new turn.
                        stop_token_ids=[151643, 151645],
                        # Anti-reward-hacking layer 1: p9d4 collapsed to
                        # single-stop-token answers within ~60 rollouts because
                        # the RM scores empty responses 7.7.  Forcing a floor
                        # length makes the degenerate policy unreachable.
                        min_tokens=min(16, int(req["max_tokens"])),
                        logit_bias=_banned,
                        presence_penalty=_presence,
                    )
                    request = LoRARequest(
                        f"vpo-step-{adapter_id}", adapter_id, adapter,
                    )
                    t0 = time.monotonic()
                    generated = llm.generate(prompts, params, lora_request=request)
                    rows = []
                    for result in generated:
                        rows.extend([list(x.token_ids) for x in result.outputs])
                    payload = {"ok": True, "rows": rows,
                               "tokens": sum(map(len, rows)),
                               "engine_generation_sec": time.monotonic() - t0,
                               "server_pid": os.getpid(), "adapter_id": adapter_id}
                    conn.sendall((json.dumps(payload) + "\n").encode())
    finally:
        listener.close()
        sock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
