#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Both OOM failures showed multiple GiB reserved-but-unallocated; large vocab
# tensors of varying width make fragmentation a real failure mode over 500
# rollouts.  The vLLM engine runs in a separate process and is unaffected.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

method="${1:?usage: run_skywork_500.sh grpo|vpo_rm}"
case "$method" in
  grpo|vpo_rm) ;;
  *) echo "unsupported method: $method" >&2; exit 2 ;;
esac

exec python3 scripts/profile_vllm_full.py \
  --method "$method" \
  --max-rollouts 500 \
  --max-response-tokens 2048 \
  --generation-microbatch 32 \
  --keep-adapters-every 100 \
  --output-dir "runs/formal-skywork-${method}-${JOB_ID:?}"
