#!/usr/bin/env python3
"""On-target CUDA/BF16, exact-mask kernel and real native-EOS LoRA smoke test."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_rows(payload, support, stop_ids, maximum):
    rows, reasons = payload["rows"], payload["finish_reasons"]
    if not rows or len(rows) != len(reasons):
        raise ValueError("Missing responses or finish metadata")
    for row, reason in zip(rows, reasons):
        if not row or len(row) > maximum or any(token not in support for token in row):
            raise ValueError("Invalid response length or output support")
        if reason == "stop":
            if row[-1] not in stop_ids or any(token in stop_ids for token in row[:-1]):
                raise ValueError("Stop reason does not match terminal EOS")
        elif reason == "length":
            if len(row) != maximum or any(token in stop_ids for token in row):
                raise ValueError("Length reason does not match actual truncation")
        else:
            raise ValueError("Unknown finish reason")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, help="Fresh directory; no model weights are saved")
    p.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    args = p.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if out.exists():
        p.error("Use a fresh output directory")
    if not 0 < args.gpu_memory_utilization <= 1:
        p.error("gpu-memory-utilization must be in (0, 1]")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vpo_rm.alignment import shared_output_mask
    from vpo_rm.integration import checked_sampling_params, generation_payload, vllm_sampling_kwargs
    from vpo_rm.trainer import VPOTrainer

    if torch.cuda.device_count() != args.tensor_parallel_size:
        raise RuntimeError("Expose exactly tensor-parallel-size GPUs via CUDA_VISIBLE_DEVICES")
    expected = {"torch": "2.13.0+cu129", "transformers": "5.16.1", "peft": "0.20.0",
                "vllm": "0.28.1rc1.dev199+g7c5dc571c.cu129", "pyarrow": "21.0.0"}
    software = {name: importlib.metadata.version(name) for name in expected}
    if software != expected:
        raise RuntimeError(f"Install requirements/ssh-a6000-cu129.txt before this version-specific gate: {software}")
    out.mkdir(parents=True)
    devices = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability(index) < (8, 0):
                raise RuntimeError("This deployment requires native BF16 on Ampere or newer")
            x = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
            if not torch.all(x @ x == 16):
                raise RuntimeError("BF16 CUDA matrix multiply failed")
            torch.cuda.synchronize(index)
            props = torch.cuda.get_device_properties(index)
            devices.append({"name": props.name, "capability": torch.cuda.get_device_capability(index),
                            "total_memory_gib": props.total_memory / 2**30})
            del x
    runtime = {"python": platform.python_version(), "software": software, "gpus": devices,
               "torch_cuda": torch.version.cuda, "tensor_parallel_size": args.tensor_parallel_size}
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    model = str(ROOT / "models/Qwen3-14B-Base")
    adapter = str(ROOT / "models/sft-native-eos-clean2k5e2")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    vocab = AutoConfig.from_pretrained(model, local_files_only=True).vocab_size
    kernel_results = []
    for minimum in (0, 8):
        kwargs = vllm_sampling_kwargs(tok, vocab,
            {"temperature": 1., "min_tokens": minimum, "max_tokens": 64, "group_size": 2})
        params = checked_sampling_params(SamplingParams, **kwargs)
        state = LogitBiasState(1, torch.device("cuda:0"))
        state.add_request(0, 1, params)
        state.apply_staged_writes()
        logits = torch.zeros((1, vocab), device="cuda:0")
        banned = list(kwargs["logit_bias"])
        logits[:, banned] = 1000
        mapping = torch.tensor([0], device="cuda:0", dtype=torch.int32)
        state.apply_logit_bias(logits, mapping, np.array([0], dtype=np.int32),
                              torch.tensor([0], device="cuda:0"))
        torch.cuda.synchronize()
        if not torch.isneginf(logits[:, banned]).all():
            raise RuntimeError("Backend did not apply an exact negative-infinity vocabulary mask")
        stopped = logits[:, kwargs["stop_token_ids"]]
        if not (torch.isneginf(stopped).all() if minimum else torch.isfinite(stopped).all()):
            raise RuntimeError("Backend minimum-length EOS mask disagrees with training")
        kernel_results.append({"min_tokens": minimum, "banned_count": len(banned), "passed": True})
        del state, logits, mapping
    (out / "kernel.json").write_text(json.dumps(kernel_results, indent=2))
    torch.cuda.empty_cache()
    engine = LLM(model=model, dtype="bfloat16", generation_config="vllm", enable_lora=True,
                 max_lora_rank=64, max_loras=2, max_model_len=4096, max_num_seqs=4,
                 tensor_parallel_size=args.tensor_parallel_size,
                 gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    instructions = ["What is 2 + 2? Give only the number.", "Reply with only OK."]
    prompts = [VPOTrainer._render_chat_prompt(tok, text) for text in instructions]
    kwargs = vllm_sampling_kwargs(tok, vocab,
        {"temperature": 1., "min_tokens": 0, "max_tokens": 64, "group_size": 2})
    outputs = engine.generate(prompts, checked_sampling_params(SamplingParams, **kwargs),
                              lora_request=LoRARequest("native-eos-sft", 1, adapter))
    payload = generation_payload(outputs, kwargs["stop_token_ids"])
    support = set(shared_output_mask(tok, vocab).nonzero().flatten().tolist())
    validate_rows(payload, support, set(kwargs["stop_token_ids"]), 64)
    if len(payload["rows"]) != 4:
        raise RuntimeError("Expected two responses per prompt")
    payload["decoded"] = [tok.decode(row, skip_special_tokens=False) for row in payload["rows"]]
    payload["instructions"] = instructions
    (out / "generations.json").write_text(json.dumps(payload, indent=2))
    (out / "result.json").write_text(json.dumps({"status": "passed", "rows": 4,
        "scope": "CUDA/BF16, exact masking kernel and real LoRA generation; not training capacity"}, indent=2))
    print("SSH_GPU_SMOKE_PASSED", out, flush=True)


if __name__ == "__main__":
    main()
