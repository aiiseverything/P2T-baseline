#!/usr/bin/env bash
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if ! python3 -c "import peft" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn peft
fi

# 1) create a ZERO-INIT LoRA adapter (B=0 -> identical outputs to base) using
#    the exact same LoraConfig geometry as sft_init.py, then verify B==0.
python3 - <<'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import safetensors.torch as st

model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen3-14B-Base", torch_dtype=torch.bfloat16, trust_remote_code=True)
cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
get_peft_model(model, cfg).save_pretrained("models/zero-init-lora")
t = st.load_file("models/zero-init-lora/adapter_model.safetensors")
bkeys = [k for k in t if "lora_B" in k]
amax = max(float(t[k].abs().max()) for k in bkeys)
print(f"zero-init adapter saved: {len(t)} tensors, {len(bkeys)} lora_B, "
      f"max|B|={amax:.2e} (must be 0)")
assert amax == 0.0, "lora_B is not zero!"
EOF

# 2) roll it through the SAME LoRA inference path as v1/e2
python3 scripts/eval_alpaca.py \
  --output runs/tvt-zero --max-tokens 2048 --recipes 1.0:1:1.0:-1 \
  --dataset datasets/sft_v2/train25.jsonl \
  --adapters zero-train=$R/models/zero-init-lora

python3 scripts/eval_alpaca.py \
  --output runs/tvt-zero --max-tokens 2048 --recipes 1.0:1:1.0:-1 \
  --dataset datasets/sft_v2/test25.jsonl \
  --adapters zero-test=$R/models/zero-init-lora

echo "=== length summary ==="
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/tvt-zero/*/generations_t*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    lens = sorted(r["response_tokens"] for r in rows)
    n = len(lens); pin = sum(t >= 2048 for t in lens)
    print(f"{f.split('/')[-2]:12s} n={n}  p10 {lens[n//10]:4d}  median {lens[n//2]:4d}  "
          f"mean {sum(lens)/n:5.0f}  | pinned@2048 {100*pin/n:3.0f}%")
EOF