#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM

VENV=/tmp/vllm-check-venv
python -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip --quiet
"$VENV/bin/python" -m pip install -e '.[data,train]' --quiet --retries 8 --timeout 180
"$VENV/bin/python" -m pip install vllm --quiet --retries 8 --timeout 180

"$VENV/bin/python" - <<'PY'
import torch
import vllm
print(f"vllm={vllm.__version__}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, devices={torch.cuda.device_count()}")
from vllm import LLM, SamplingParams
model = LLM(model="models/Qwen3-14B", dtype="bfloat16",
            max_model_len=4096, max_num_seqs=8,
            gpu_memory_utilization=0.80, trust_remote_code=True)
out = model.generate(["Say hello in one short sentence."],
                     SamplingParams(temperature=0.0, max_tokens=8))
print("startup_and_generation=OK")
print(out[0].outputs[0].text)
PY
