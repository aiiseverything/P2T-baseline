#!/usr/bin/env bash
# Submit the SFT v3 three-arm grid (user-approved 2026-09-16):
#   lowlr (lr 1e-5) / eosw (EOS weight 20) / both — vs existing control
#   sftv2-clean-2k5e2. Each arm: clean 2.5k x 2ep + post-train tvt rollout.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
D=$R/datasets/sft_v2

submit() { # name lr eosw
  local name="$1" lr="$2" w="$3"
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"; return
  fi
  echo "SUBMIT $name (lr=$lr eos_weight=$w)"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 1 --cpu 16 --memory 200000 \
    --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "SFT_MODE=full" -e "SFT_DATA=$D/sft_clean.parquet" \
    -e "SFT_OUTPUT=$R/models/$name" \
    -e "SFT_MAX_EXAMPLES=2500" -e "SFT_EPOCHS=2" -e "SFT_MONITOR_EVERY=50" \
    -e "SFT_LR=$lr" -e "SFT_EOS_WEIGHT=$w" -e "SFT_TVT=1" \
    -- bash -exc "bash $R/scripts/run_sft_v2.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

submit sftv3-lowlr 1e-5 1.0
submit sftv3-eosw  1e-4 20.0
submit sftv3-both  1e-5 20.0

echo
echo "=== reconcile ==="
rjob list 2>/dev/null | grep "showname=sftv3" | grep -v Stopped | \
  sed 's/.*(showname=\([^)]*\)): /\1/' | sort | uniq -c
