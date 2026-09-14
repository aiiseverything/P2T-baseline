#!/usr/bin/env bash
# Submit the 2 LLM x 2 RM matrix: SFT init for Qwen3-8B, RM-4B artifact
# calibration, and six train->eval chain jobs (GRPO + VPO per new cell).
# The (14B, RM-8B) cell is already covered by p10/p11.
# Idempotent (rjob registry-poll dedup); everything queues at priority 9.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest

submit() { # entry name gpus cpu mem [ENV=VAL...]
  local entry="$1" name="$2" gpus="$3" cpu="$4" mem="$5"
  shift 5
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"
    return
  fi
  local env_args=(-e "DISTRIBUTED_JOB=true")
  for kv in "$@"; do env_args+=(-e "$kv"); done
  echo "SUBMIT $name (${gpus}GPU) <- $entry $*"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu "$gpus" --cpu "$cpu" --memory "$mem" \
    --charged-group ma4agismall_gpu --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    "${env_args[@]}" \
    -- bash -exc "bash $R/scripts/$entry" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

B14=models/Qwen3-14B-Base
B8=models/Qwen3-8B-Base
RM8=models/Skywork-Reward-V2-Qwen3-8B
RM4=models/Skywork-Reward-V2-Qwen3-4B
S14=models/sft-init-qwen3-14b-base
S8=models/sft-init-qwen3-8b-base
CALIB=runs/rm4b-calib
VPO_ENV="CREDIT_LAMBDA=4.0 FREEZE_STOP_TOKENS=1 FREEZE_STRUCTURAL=1"

# --- prerequisites ---
submit run_sft_init.sh sft-init-8b 1 16 64000 \
  "SFT_MODE=full" "SFT_MODEL=$R/$B8" "SFT_OUTPUT=$R/$S8"
submit run_rm_calib.sh calib-rm4b 3 48 600000 \
  "CALIB_MODEL=$R/$B14" "CALIB_RM=$R/$RM4" "CALIB_OUT=$R/$CALIB" "CALIB_INIT=$R/$S14"

# --- cell: 14B x RM-4B ---
submit run_train_eval_chain.sh mx-14b-rm4b-grpo 3 48 600000 \
  "CHAIN_NAME=mx-14b-rm4b-grpo" "CHAIN_METHOD=grpo" \
  "CHAIN_MODEL=$R/$B14" "CHAIN_RM=$R/$RM4" "CHAIN_INIT=$R/$S14" \
  "MAX_ROLLOUTS=250" "SEED=42"
submit run_train_eval_chain.sh mx-14b-rm4b-vpo 3 48 600000 \
  "CHAIN_NAME=mx-14b-rm4b-vpo" "CHAIN_METHOD=vpo_rm" \
  "CHAIN_MODEL=$R/$B14" "CHAIN_RM=$R/$RM4" "CHAIN_INIT=$R/$S14" \
  "CHAIN_WAIT_FILE=$R/$CALIB/DONE" \
  "MAX_ROLLOUTS=250" "SEED=42" $VPO_ENV

# --- cell: 8B x RM-8B ---
submit run_train_eval_chain.sh mx-8b-rm8b-grpo 3 48 600000 \
  "CHAIN_NAME=mx-8b-rm8b-grpo" "CHAIN_METHOD=grpo" \
  "CHAIN_MODEL=$R/$B8" "CHAIN_RM=$R/$RM8" "CHAIN_INIT=$R/$S8" \
  "MAX_ROLLOUTS=250" "SEED=42"
submit run_train_eval_chain.sh mx-8b-rm8b-vpo 3 48 600000 \
  "CHAIN_NAME=mx-8b-rm8b-vpo" "CHAIN_METHOD=vpo_rm" \
  "CHAIN_MODEL=$R/$B8" "CHAIN_RM=$R/$RM8" "CHAIN_INIT=$R/$S8" \
  "MAX_ROLLOUTS=250" "SEED=42" $VPO_ENV

# --- cell: 8B x RM-4B ---
submit run_train_eval_chain.sh mx-8b-rm4b-grpo 3 48 600000 \
  "CHAIN_NAME=mx-8b-rm4b-grpo" "CHAIN_METHOD=grpo" \
  "CHAIN_MODEL=$R/$B8" "CHAIN_RM=$R/$RM4" "CHAIN_INIT=$R/$S8" \
  "MAX_ROLLOUTS=250" "SEED=42"
submit run_train_eval_chain.sh mx-8b-rm4b-vpo 3 48 600000 \
  "CHAIN_NAME=mx-8b-rm4b-vpo" "CHAIN_METHOD=vpo_rm" \
  "CHAIN_MODEL=$R/$B8" "CHAIN_RM=$R/$RM4" "CHAIN_INIT=$R/$S8" \
  "CHAIN_WAIT_FILE=$R/$CALIB/DONE" \
  "MAX_ROLLOUTS=250" "SEED=42" $VPO_ENV

echo
echo "=== matrix jobs ==="
rjob list 2>/dev/null | grep -E "showname=(sft-init-8b|calib-rm4b|mx-)" || true
