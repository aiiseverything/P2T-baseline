#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The derived vpo-rm-vllm-train image could not be pulled on some nodes
# (ImagePullBackOff); run on the cluster vLLM base image instead and install
# the two missing packages from the internal mirror (no-op when present).
if ! python3 -c "import peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    "peft==0.20.0" "pyarrow>=15,<22"
fi

# Mode via env var (rjob passes -e cleanly; positional args get mangled)
# or positional for local runs.
mode="${SFT_MODE:-${1:-}}"
case "$mode" in
  smoke)
    # Fast validation pass: a few hundred examples, still exercises rendering,
    # prefix assertions, LoRA forward/backward, adapter save.
    exec python3 scripts/sft_init.py --limit 300 --micro-batch 2 --grad-accum 4 \
      --output "runs/sft-init-smoke-${JOB_ID:-local}" ;;
  full)
    # Classmate's sizing: small subset, a couple of epochs — initialization
    # only needs to burn in format/stopping conventions, not peak quality.
    exec python3 scripts/sft_init.py \
      --max-examples 10000 --epochs 2 \
      --output models/sft-init-qwen3-14b-base ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
