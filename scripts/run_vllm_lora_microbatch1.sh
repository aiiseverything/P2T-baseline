#!/usr/bin/env bash
set -u

PROJECT=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
export PROJECT
cd "$PROJECT"

echo '==> Installing the two dependencies missing from the vLLM image'
EXTRA_SITE="$PROJECT/.vllm-extra"
export PYTHONPATH="$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"
if python3 -c 'import peft, pyarrow' 2>/dev/null; then
  echo "==> Reusing cached dependencies from $EXTRA_SITE"
else
  mkdir -p "$EXTRA_SITE"
  python3 -m pip install --target "$EXTRA_SITE" --upgrade --no-cache-dir --no-deps \
    --retries 8 --timeout 180 \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    'peft==0.20.0' 'pyarrow>=15,<22' || \
    echo 'WARNING: dependency installation failed; continuing with the benchmark'
fi

python3 - <<'PY'
import importlib
for name in ('peft', 'pyarrow'):
    try:
        mod = importlib.import_module(name)
        print(f'{name}=OK {getattr(mod, "__version__", "")}')
    except Exception as exc:
        print(f'{name}=MISSING {type(exc).__name__}: {exc}')
PY

python3 - <<'PY'
import time
import os
import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

project = os.environ['PROJECT']
model_path = f'{project}/models/Qwen3-14B'
lora_path = f'{project}/runs/smoke-skywork10/checkpoint-1'
base_prompts = [
    'Explain why the sky appears blue in one short paragraph.',
    'Give three practical tips for learning a new language.',
    'Write a concise Python function that reverses a string.',
    'What are two benefits of regular exercise?',
    'Summarize the water cycle for a high school student.',
    'Suggest a simple dinner recipe using vegetables and rice.',
    'Explain the difference between precision and recall.',
    'Write a polite two-sentence email asking to reschedule a meeting.',
]
# Same 64-request workload as the batch-64 benchmark.
prompts = [p for p in base_prompts for _ in range(8)]
print(f'torch={torch.__version__}, cuda={torch.version.cuda}, vllm={vllm.__version__}')
max_tokens = int(os.environ.get('VPO_MAX_TOKENS', '32'))
print(f'workload_requests={len(prompts)}, micro_batch={os.environ.get("VPO_MICROBATCH", "1")}, max_tokens={max_tokens}')
llm = LLM(model=model_path, dtype='bfloat16', trust_remote_code=True,
          enable_lora=True, max_lora_rank=64, max_loras=1,
          max_model_len=4096, max_num_seqs=int(os.environ.get('VPO_MICROBATCH', '1')),
          gpu_memory_utilization=0.80)
params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=max_tokens)
request = LoRARequest('skywork-smoke', 1, lora_path)

for label in ('warmup', 'steady'):
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, lora_request=request)
    elapsed = time.perf_counter() - t0
    tokens = sum(len(x.outputs[0].token_ids) for x in outputs)
    print(f'{label}_seconds={elapsed:.3f} requests={len(outputs)} tokens={tokens} tok_per_sec={tokens/elapsed:.2f}')
print(f'lora_microbatch{os.environ.get("VPO_MICROBATCH", "1")}_generation=OK')
PY
