#!/usr/bin/env bash
# Batch-submit RM eval jobs (256 frozen validation prompts, Skywork scoring,
# temps 1.0 + 0.7 — matches the p10 protocol) for the four p11 VPO arms.
# p10 GRPO + VPO arms already have RM evals in runs/eval-p10-*.
# Idempotent: skips jobs already in rjob or with results on disk; waits for
# rjob registration between submits (registry lag caused duplicates before).
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest

submit() { # name run-label run-path
  local name="$1" label="$2" path="$3"
  if [ -f "$R/runs/$name/summary.json" ]; then
    echo "SKIP $name: results already on disk"
    return
  fi
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"
    return
  fi
  echo "SUBMIT $name  <-  $path"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 2 --cpu 32 --memory 400000 \
    --charged-group ma4agismall_gpu --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "EVAL_RUNS=$label=$R/$path" \
    -e "EVAL_TEMPS=1.0 0.7" \
    -e "EVAL_OUT=$R/runs/$name" \
    -- bash -exc "bash $R/scripts/run_eval.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

submit eval-p11-lam2struct  lam2struct  runs/formal-skywork-vpo_rm-p11-vpo-lam2-0-struct-56479229-7e9e9
submit eval-p11-lam4struct  lam4struct  runs/formal-skywork-vpo_rm-p11-vpo-lam4-0-struct-57689603-1f5f5
submit eval-p11-lam8struct  lam8struct  runs/formal-skywork-vpo_rm-p11-vpo-lam8-0-struct-58650746-d4780
submit eval-p11-lam4eos-s43 lam4eos-s43 runs/formal-skywork-vpo_rm-p11-vpo-lam4-eos-seed43-59647788

echo
echo "=== registered RM eval jobs ==="
rjob list 2>/dev/null | grep -E "showname=eval-p11" || true
