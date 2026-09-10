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

# Gate on the unit tests before committing GPU hours (no torch on the login
# node, so tests can only run inside the job image).
python3 -m pytest tests/test_core.py tests/test_integration.py tests/test_trainer.py -q

# p9d: Qwen3-14B-Base actor (paper headroom), LoRA lr 1e-4 (p9c's 1e-6 was the
# paper's full-FT rate), Plan B credit (standardized, weight-capped), adapters
# every 50 for the offline 256-prompt eval curve.
exec python3 scripts/profile_vllm_full.py \
  --method "$method" \
  --model models/Qwen3-14B-Base \
  --learning-rate 1e-4 \
  --tau 1.0 \
  --weight-cap 20.0 \
  --max-rollouts 500 \
  --max-response-tokens 2048 \
  --generation-microbatch 32 \
  --keep-adapters-every 50 \
  --output-dir "runs/formal-skywork-${method}-${JOB_ID:?}"
