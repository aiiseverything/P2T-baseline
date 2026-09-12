#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! python3 -c "import peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    "peft==0.20.0" "pyarrow>=15,<22"
fi

# EVAL_RUNS: semicolon-separated label=path pairs (rjob-safe, no spaces needed)
# EVAL_OUT: output dir under runs/
exec python3 scripts/eval_checkpoints.py \
  ${EVAL_MODEL:+--model "$EVAL_MODEL"} \
  --run "${EVAL_RUNS:?EVAL_RUNS=label=path required}" \
  --output "${EVAL_OUT:?EVAL_OUT required}"
