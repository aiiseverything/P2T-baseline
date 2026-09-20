#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:?family required}"
ARM="${2:?arm required}"
[[ "$FAMILY" == qwen ]]
[[ "$ARM" == grpo || "$ARM" == lam4 ]]
export PYTHONPATH="/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/runs/direct-rl-qwen-instruct-v3-20260918/source:/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM/.vllm-extra"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PRESENCE_PENALTY=0.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec > >(tee -a "$SUITE/$FAMILY/$ARM/job.log") 2>&1
exec python3 "$SUITE/run_direct.py" "$ARM"
