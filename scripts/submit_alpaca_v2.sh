#!/usr/bin/env bash
# Submit AlpacaEval generation for the four SFT v2 adapters (post-grid,
# user-approved 2026-09-15). One 1-GPU job, four tags, resume-safe.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
OUTROOT=$R/runs/alpacaeval-evals
mkdir -p "$OUTROOT"

ADAPTERS=""
for t in sftv2-raw-10k sftv2-clean-10k sftv2-raw-2k5 sftv2-clean-2k5; do
  [ -f "$R/models/$t/adapter_config.json" ] && ADAPTERS="$ADAPTERS $t=$R/models/$t"
done
[ -n "$ADAPTERS" ] || { echo "no v2 adapters found under models/"; exit 1; }
echo "tags:$(wc -w <<<"$ADAPTERS")"

NAME=alpacaeval-gen-v2
if rjob list 2>/dev/null | grep -q "showname=$NAME)"; then
  echo "SKIP: rjob already exists"; exit 0
fi

rjob submit --name "$NAME" --task-type normal --priority 9 --enable-sshd \
  --image "$IMAGE" --image-pull-policy IfNotPresent \
  --gpu 1 --cpu 16 --memory 200000 \
  --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
  --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
  -e "ALPACA_ADAPTERS=$ADAPTERS" \
  -e "ALPACA_OUT=$OUTROOT" \
  -- bash -exc "bash $R/scripts/run_alpaca.sh" 2>&1 | tail -1

for _ in $(seq 1 12); do
  rjob list 2>/dev/null | grep -q "showname=$NAME)" && break
  sleep 5
done
rjob list 2>/dev/null | grep "showname=$NAME" || echo "WARNING: not visible after 60s"
