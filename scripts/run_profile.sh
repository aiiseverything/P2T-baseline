#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
python -m pip install -e '.[data,train]' --quiet
python scripts/check_readiness.py --setting ultrafeedback_skywork_8b \
  --require-deps --require-data-reader --require-gpu \
  --json runs/readiness-profile10.json
python scripts/train_skywork.py \
  --model models/Qwen3-14B \
  --rm models/Skywork-Reward-V2-Qwen3-8B \
  --dataset-path datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet \
  --output-dir runs/profile-skywork-10b \
  --method vpo_rm --max-rollouts 10 --prompts-per-rollout 8 \
  --group-size 8 --max-response-tokens 32
