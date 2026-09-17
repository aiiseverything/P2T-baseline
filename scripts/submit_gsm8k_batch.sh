#!/usr/bin/env bash
# Submit GSM8K checkpoint-sweep jobs (user-approved set):
#   GRPO p10 (incl. step-0 = shared SFT init) + the three p11 lambda arms.
# 1 GPU each, priority 9, private-machine group (see .skills/rjob).
# Re-run at a higher generation cap:
#   GSM8K_SUFFIX=-3072 GSM8K_MAX_TOKENS=3072 bash scripts/submit_gsm8k_batch.sh
# (suffix goes into job names + output dirs so the 1024 results stay intact)
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
OUTROOT=$R/runs/gsm8k-evals
MAXTOK="${GSM8K_MAX_TOKENS:-1024}"
SUF="${GSM8K_SUFFIX:-}"
mkdir -p "$OUTROOT"

submit() { # name adapters
  local name="$1" adapters="$2"
  [ -n "$adapters" ] || { echo "SKIP $name: no adapters"; return; }
  local jobs
  jobs="$(rjob list)" || return "$?"
  if grep -Fq "showname=$name)" <<< "$jobs"; then
    echo "SKIP $name: rjob already exists"
    return
  fi
  if find "$OUTROOT/$name" -name "results_t*.json" 2>/dev/null | grep -q .; then
    echo "SKIP $name: results already on disk"
    return
  fi
  echo "SUBMIT $name"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 1 --cpu 16 --memory 200000 \
    --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "GSM8K_ADAPTERS=$adapters" \
    -e "GSM8K_OUT=$OUTROOT/$name" \
    -e "GSM8K_MAX_TOKENS=$MAXTOK" \
    -- bash -exc "bash $R/scripts/run_gsm8k.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

sweep() { # run-dir steps...
  local root="$1"; shift
  local adapters="" s
  for s in "$@"; do
    [ -f "$R/$root/step-$s/adapter_config.json" ] && \
      adapters="$adapters step-$s=$R/$root/step-$s"
  done
  echo "$adapters"
}

G=runs/formal-skywork-grpo-p10-grpo-baseline2-69550359/vllm-adapters
L2=runs/formal-skywork-vpo_rm-p11-vpo-lam2-0-struct-56479229-7e9e9/vllm-adapters
L4=runs/formal-skywork-vpo_rm-p11-vpo-lam4-0-struct-57689603-1f5f5/vllm-adapters
L8=runs/formal-skywork-vpo_rm-p11-vpo-lam8-0-struct-58650746-d4780/vllm-adapters

submit "gsm8k${SUF}-grpo-p10"     "$(sweep "$G" 0 50 100 150 200 250)"
submit "gsm8k${SUF}-p11-lam2struct" "$(sweep "$L2" 50 100 150 200 250)"
submit "gsm8k${SUF}-p11-lam4struct" "$(sweep "$L4" 50 100 150 200 250)"
submit "gsm8k${SUF}-p11-lam8struct" "$(sweep "$L8" 50 100 150 200 250)"

echo
echo "=== reconcile (expect exactly 4 gsm8k jobs) ==="
rjob list 2>/dev/null | grep "showname=gsm8k" | grep -v Stopped | sed 's/.*(showname=\([^)]*\)): /\1/' | sort | uniq -c
