#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
python3 - <<'PY'
import time
import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

model_path = "/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/models/Qwen3-14B"
lora_path = "/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/smoke-skywork10/checkpoint-1"
prompts = [
    "Explain why the sky appears blue in one short paragraph.",
    "Give three practical tips for learning a new language.",
    "Write a concise Python function that reverses a string.",
    "What are two benefits of regular exercise?",
    "Summarize the water cycle for a high school student.",
    "Suggest a simple dinner recipe using vegetables and rice.",
    "Explain the difference between precision and recall.",
    "Write a polite two-sentence email asking to reschedule a meeting.",
]
prompts = [p for p in prompts for _ in range(8)]
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, vllm={vllm.__version__}")
llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
          enable_lora=True, max_lora_rank=64, max_loras=1,
          max_model_len=4096, max_num_seqs=64,
          gpu_memory_utilization=0.80)
params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=32)
request = LoRARequest("skywork-smoke", 1, lora_path)

# First call includes any remaining lazy compilation; report it separately.
t0 = time.perf_counter()
first = llm.generate(prompts, params, lora_request=request)
first_sec = time.perf_counter() - t0
first_tokens = sum(len(x.outputs[0].token_ids) for x in first)
print(f"first_batch_seconds={first_sec:.3f} tokens={first_tokens} tok_per_sec={first_tokens/first_sec:.2f}")

# Second call measures steady-state batch throughput.
t0 = time.perf_counter()
second = llm.generate(prompts, params, lora_request=request)
second_sec = time.perf_counter() - t0
second_tokens = sum(len(x.outputs[0].token_ids) for x in second)
print(f"steady_batch_seconds={second_sec:.3f} tokens={second_tokens} tok_per_sec={second_tokens/second_sec:.2f}")
print("lora_batch_generation=OK")
PY
