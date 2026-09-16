#!/usr/bin/env bash
# Submit the SFT v2 2x2 grid (user-approved 2026-09-15):
#   {raw, clean} x {10k x 2ep (=v1 params), 2.5k x 8ep (equal 625 steps)}
# Every arm carries the as_text echo fix + free-running monitor.
# Submit smoke first; the 4 full jobs go out once smoke Succeeds.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
D=$R/datasets/sft_v2
O=$R/models

submit() { # name data max_examples epochs mode
  local name="$1" data="$2" mx="$3" ep="$4" mode="$5"
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"; return
  fi
  echo "SUBMIT $name"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 1 --cpu 16 --memory 200000 \
    --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "SFT_MODE=$mode" -e "SFT_DATA=$data" -e "SFT_OUTPUT=$O/$name" \
    -e "SFT_MAX_EXAMPLES=$mx" -e "SFT_EPOCHS=$ep" \
    -- bash -exc "bash $R/scripts/run_sft_v2.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

# Default to smoke only — the grid must never go out before smoke Succeeds
# (the chain watcher calls this with an explicit stage).
STAGE="${1:-smoke}"

if [ "$STAGE" = smoke ] || [ "$STAGE" = all ]; then
  submit sftv2-smoke "$D/sft_clean.parquet" 0 2 smoke
fi
if [ "$STAGE" = grid ] || [ "$STAGE" = all ]; then
  submit sftv2-raw-10k    "$D/sft_raw.parquet"   10000 2 full
  submit sftv2-clean-10k  "$D/sft_clean.parquet" 10000 2 full
  submit sftv2-raw-2k5    "$D/sft_raw.parquet"   2500  8 full
  submit sftv2-clean-2k5  "$D/sft_clean.parquet" 2500  8 full
fi

echo
echo "=== reconcile ==="
rjob list 2>/dev/null | grep "showname=sftv2" | grep -v Stopped | \
  sed 's/.*(showname=\([^)]*\)): /\1/' | sort | uniq -c
