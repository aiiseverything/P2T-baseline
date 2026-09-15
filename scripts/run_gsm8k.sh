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

# Offline gate on the scorer path before any GPU work.
python3 scripts/eval_gsm8k.py --selftest

# Checkpoint sweep: space-separated TAG=PATH entries ("none" = bare base model).
ADAPTERS="${GSM8K_ADAPTERS:?GSM8K_ADAPTERS=tag=path,... required}"
RECIPES="${GSM8K_RECIPES:-1.0:1:1.0:-1}"

exec python3 scripts/eval_gsm8k.py \
  --output "${GSM8K_OUT:?required}" \
  --recipes $RECIPES \
  --adapters $ADAPTERS
