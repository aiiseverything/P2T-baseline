#!/usr/bin/env bash
# Submit the AlpacaEval generation sweep (user-approved 2026-09-15, 2x3 GPUs):
#   base + SFT init + GRPO p10 (s50-250) + 3 VPO p11 lambda arms (s50-250) = 22 tags
# TWO 3-GPU jobs; tags alternate between them; inside each job 3 single-GPU
# vLLM engines shard its 11 tags (~1-1.5h).
# Judging (gpt-4.1 via linkapi relay) runs afterwards on a networked machine.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
OUTROOT=$R/runs/alpacaeval-evals
mkdir -p "$OUTROOT"

G=runs/formal-skywork-grpo-p10-grpo-baseline2-69550359/vllm-adapters
L2=runs/formal-skywork-vpo_rm-p11-vpo-lam2-0-struct-56479229-7e9e9/vllm-adapters
L4=runs/formal-skywork-vpo_rm-p11-vpo-lam4-0-struct-57689603-1f5f5/vllm-adapters
L8=runs/formal-skywork-vpo_rm-p11-vpo-lam8-0-struct-58650746-d4780/vllm-adapters

ADAPTERS="base=none sft-init=$G/step-0"
for s in 50 100 150 200 250; do
  ADAPTERS="$ADAPTERS grpo-s$s=$G/step-$s"
done
for s in 50 100 150 200 250; do
  ADAPTERS="$ADAPTERS vpo-lam2struct-s$s=$L2/step-$s vpo-lam4struct-s$s=$L4/step-$s vpo-lam8struct-s$s=$L8/step-$s"
done

# verify every adapter dir exists before submitting
missing=0
for spec in $ADAPTERS; do
  path="${spec#*=}"
  [ "$path" = "none" ] && continue
  [ -f "$R/$path/adapter_config.json" ] || { echo "MISSING: $path"; missing=1; }
done
[ $missing -eq 0 ] || { echo "aborting: missing adapters"; exit 1; }

# alternate tags between the two jobs (i % 2) for balance
A=""; B=""; i=0
for spec in $ADAPTERS; do
  if [ $((i % 2)) -eq 0 ]; then A="$A $spec"; else B="$B $spec"; fi
  i=$((i + 1))
done
echo "job A:$(wc -w <<<"$A") tags, job B:$(wc -w <<<"$B") tags"

submit() { # name adapters
  local name="$1" adapters="$2"
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"; return
  fi
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 3 --cpu 48 --memory 600000 \
    --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "ALPACA_ADAPTERS=$adapters" \
    -e "ALPACA_OUT=$OUTROOT" \
    -e "ALPACA_FANOUT=3" \
    -- bash -exc "bash $R/scripts/run_alpaca_fanout.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

submit alpacaeval-gen-a "$A"
submit alpacaeval-gen-b "$B"

echo
echo "=== reconcile (expect 2 jobs) ==="
rjob list 2>/dev/null | grep "showname=alpacaeval-gen" | grep -v Stopped | \
  sed 's/.*(showname=\([^)]*\)): /\1/' | sort | uniq -c
