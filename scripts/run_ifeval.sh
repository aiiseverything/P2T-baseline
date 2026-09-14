#!/usr/bin/env bash
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra:$R/third_party/ifeval"
export PYTHONUNBUFFERED=1

# Official IFEval code dependencies (internal mirror, fast on GPU nodes)
if ! python3 -c "import absl, immutabledict, langdetect" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    absl-py immutabledict langdetect
fi
if ! python3 -c "import peft" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn peft
fi

exec python3 scripts/eval_ifeval.py \
  --adapter "${IFEVAL_ADAPTER:?required}" \
  --output "${IFEVAL_OUT:?required}" \
  --temps 0.0 0.7
