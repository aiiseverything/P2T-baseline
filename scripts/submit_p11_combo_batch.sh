#!/usr/bin/env bash
# Submit the four combined 3-GPU per-arm eval jobs for the p11 VPO arms
# (RM eval on cards 0+1 in parallel with the IFEval sweep on card 2).
# Idempotent: skips jobs already in rjob or with results on disk; waits for
# rjob registration between submits.
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest

submit() { # name run-label run-path ifeval-adapters
  local name="$1" label="$2" path="$3" adapters="$4"
  if [ -f "$R/runs/eval-$name/summary.json" ]; then
    echo "SKIP $name: RM eval results already on disk"
    return
  fi
  local jobs
  jobs="$(rjob list)" || return "$?"
  if grep -Fq "showname=$name)" <<< "$jobs"; then
    echo "SKIP $name: rjob already exists"
    return
  fi
  echo "SUBMIT $name  <-  $path"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 3 --cpu 48 --memory 600000 \
    --charged-group ma4agismall_gpu --private-machine group --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "EVAL_RUNS=$label=$R/$path" \
    -e "EVAL_OUT=$R/runs/eval-$name" \
    -e "IFEVAL_ADAPTERS=$adapters" \
    -e "IFEVAL_OUT=$R/runs/ifeval-evals/$name" \
    -- bash -exc "bash $R/scripts/run_arm_eval_combo.sh" 2>&1 | tail -1
  for _ in $(seq 1 12); do
    rjob list 2>/dev/null | grep -q "showname=$name)" && break
    sleep 5
  done
}

# Adapter lists expanded at submit time; step-250 exists for all p11 arms now.
mk_adapters() { # run-dir
  local root="$1/vllm-adapters" adapters="" s
  for s in 50 100 150 200 250; do
    [ -f "$R/$root/step-$s/adapter_config.json" ] && \
      adapters="$adapters step-$s=$R/$root/step-$s"
  done
  echo "$adapters"
}

L2=runs/formal-skywork-vpo_rm-p11-vpo-lam2-0-struct-56479229-7e9e9
L4=runs/formal-skywork-vpo_rm-p11-vpo-lam4-0-struct-57689603-1f5f5
L8=runs/formal-skywork-vpo_rm-p11-vpo-lam8-0-struct-58650746-d4780
S43=runs/formal-skywork-vpo_rm-p11-vpo-lam4-eos-seed43-59647788

submit evalcombo-p11-lam2struct  lam2struct  "$L2"  "$(mk_adapters "$L2")"
submit evalcombo-p11-lam4struct  lam4struct  "$L4"  "$(mk_adapters "$L4")"
submit evalcombo-p11-lam8struct  lam8struct  "$L8"  "$(mk_adapters "$L8")"
submit evalcombo-p11-lam4eos-s43 lam4eos-s43 "$S43" "$(mk_adapters "$S43")"

echo
echo "=== registered combo jobs ==="
rjob list 2>/dev/null | grep -E "showname=evalcombo" || true
