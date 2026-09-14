#!/usr/bin/env bash
# Batch-submit IFEval checkpoint-sweep jobs at temp=1.0 (training temperature):
#   base model, SFT init, GRPO (p10-baseline2), and VPO p10 + p11 arms.
# Each run gets ONE job that sweeps its step-{50..250} adapters inside a single
# vLLM engine. Step-0 equals the shared SFT init for every arm, so the dedicated
# ifeval-sft-init job (already queued) provides that point for all curves.
# Idempotent: skips jobs already in rjob or with results on disk; checkpoints
# that do not exist yet are dropped from the sweep (p11 step-250 appears when
# the run finishes).
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
IMAGE=registry.h.pjlab.org.cn/ailab/vllm-openai-cu129-nightly-x86_64:latest
OUTROOT=$R/runs/ifeval-evals
mkdir -p "$OUTROOT"

submit() { # name adapter-specs...
  local name="$1"
  shift
  local adapters="$*"
  [ -n "$adapters" ] || { echo "SKIP $name: no checkpoints available"; return; }
  if rjob list 2>/dev/null | grep -q "showname=$name)"; then
    echo "SKIP $name: rjob already exists"
    return
  fi
  if find "$OUTROOT/$name" -name "results_t*.json" 2>/dev/null | grep -q .; then
    echo "SKIP $name: results already on disk"
    return
  fi
  echo "SUBMIT $name  [$adapters]"
  rjob submit --name "$name" --task-type normal --priority 9 --enable-sshd \
    --image "$IMAGE" --image-pull-policy IfNotPresent \
    --gpu 1 --cpu 16 --memory 64000 \
    --charged-group ma4agismall_gpu --namespace ailab-ma4agismall \
    --mount "gpfs://gpfs1/ma4agi-gpu/suminle/interests/VPO-RM:$R" \
    -e "IFEVAL_ADAPTERS=$adapters" \
    -e "IFEVAL_OUT=$OUTROOT/$name" \
    -- bash -exc "bash $R/scripts/run_ifeval.sh" 2>&1 | tail -1
  sleep 5
}

sweep() { # name adapter_root steps...
  local name="$1" root="$2"
  shift 2
  local adapters="" s
  for s in "$@"; do
    if [ -f "$R/$root/step-$s/adapter_config.json" ]; then
      adapters="$adapters step-$s=$R/$root/step-$s"
    else
      echo "  (note) $name: step-$s missing, excluded"
    fi
  done
  submit "$name" $adapters
}

# --- base model (no LoRA) and baselines: can start immediately ---
submit ifeval-base "base=none"
# SFT init (= step-0 for every curve) is covered by the queued ifeval-sft-init job.

P10G=runs/formal-skywork-grpo-p10-grpo-baseline2-69550359/vllm-adapters
sweep ifeval-grpo-p10 "$P10G" 50 100 150 200 250

# --- VPO p10 generation (EOS freeze) ---
sweep ifeval-p10-lam2eos runs/formal-skywork-vpo_rm-p10-vpo-lam2-0-eosfrz2-71149830-c39f5/vllm-adapters 50 100 150 200 250
sweep ifeval-p10-lam4eos runs/formal-skywork-vpo_rm-p10-vpo-lam4-0-eosfrz2-72089082-23d54/vllm-adapters 50 100 150 200 250
sweep ifeval-p10-lam8eos runs/formal-skywork-vpo_rm-p10-vpo-lam8-0-eosfrz2-72959074-c37a8/vllm-adapters 50 100 150 200 250

# --- VPO p11 generation (structural freeze + seed replication) ---
sweep ifeval-p11-lam2struct runs/formal-skywork-vpo_rm-p11-vpo-lam2-0-struct-56479229-7e9e9/vllm-adapters 50 100 150 200 250
sweep ifeval-p11-lam4struct runs/formal-skywork-vpo_rm-p11-vpo-lam4-0-struct-57689603-1f5f5/vllm-adapters 50 100 150 200 250
sweep ifeval-p11-lam8struct runs/formal-skywork-vpo_rm-p11-vpo-lam8-0-struct-58650746-d4780/vllm-adapters 50 100 150 200 250
sweep ifeval-p11-lam4eos-s43 runs/formal-skywork-vpo_rm-p11-vpo-lam4-eos-seed43-59647788/vllm-adapters 50 100 150 200 250

echo
echo "=== registered ifeval jobs ==="
rjob list 2>/dev/null | grep -i ifeval || true
