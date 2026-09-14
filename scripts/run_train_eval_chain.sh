#!/usr/bin/env bash
# Train -> eval chain inside ONE 3-GPU rjob allocation. The cards train for
# MAX_ROLLOUTS steps, then — without releasing them — run the per-arm combo
# eval (RM eval on cards 0+1 in parallel with the IFEval sweep on card 2).
# Re-queuing a separate eval job after training cost us 3h of queue time once
# (project quota got taken in between); this layout makes that impossible.
#
# Required env:
#   CHAIN_NAME   job/output tag, e.g. mx-8b-rm4b-vpo
#   CHAIN_METHOD grpo | vpo_rm
#   CHAIN_MODEL  policy base model dir
#   CHAIN_RM     reward model dir
#   CHAIN_INIT   SFT init adapter dir (waited for if absent)
# Optional env:
#   CHAIN_WAIT_FILE  marker file gating training start (e.g. RM calibration
#                    DONE); waited for up to CHAIN_WAIT_HOURS (default 12)
#   LR / CREDIT_LAMBDA / FREEZE_STOP_TOKENS / FREEZE_STRUCTURAL / SEED /
#   MAX_ROLLOUTS   pass through to run_skywork_500.sh
set -uo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
CHAIN_NAME="${CHAIN_NAME:?required}"
CHAIN_METHOD="${CHAIN_METHOD:?grpo|vpo_rm}"
export JOB_ID="${JOB_ID:-${CHAIN_NAME}-local}"

wait_for() { # path label hours
  local path="$1" label="$2" hours="${3:-12}"
  local deadline=$(( $(date +%s) + hours * 3600 ))
  while [ ! -e "$path" ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[chain] FATAL: $label still missing after ${hours}h: $path" >&2
      exit 3
    fi
    echo "[chain] waiting for $label ..."
    sleep 300
  done
}

[ -n "${CHAIN_WAIT_FILE:-}" ] && wait_for "$CHAIN_WAIT_FILE" "gate marker" "${CHAIN_WAIT_HOURS:-12}"
wait_for "${CHAIN_MODEL:?}/config.json" "policy model" 12
wait_for "${CHAIN_RM:?}/config.json" "reward model" 12
wait_for "${CHAIN_INIT:?}/adapter_config.json" "SFT init adapter" 12

# ---- stage 1: training on all three cards ----
MODEL="$CHAIN_MODEL" RM="$CHAIN_RM" INIT_ADAPTER="$CHAIN_INIT" \
  bash scripts/run_skywork_500.sh "$CHAIN_METHOD"
RUN_DIR="runs/formal-skywork-${CHAIN_METHOD}-${JOB_ID}"
[ -f "$RUN_DIR/profile_summary.json" ] || {
  echo "[chain] FATAL: training did not complete: $RUN_DIR" >&2
  exit 3
}
echo "[chain] training done: $RUN_DIR"

# ---- stage 2: combo eval in the SAME allocation ----
# cards 0+1: RM eval (all saved checkpoints, temps 1.0/0.7)
# card 2:    IFEval sweep (steps 50..250)
adapters=""
for s in 50 100 150 200 250; do
  [ -f "$RUN_DIR/vllm-adapters/step-$s/adapter_config.json" ] && \
    adapters="$adapters step-$s=$R/$RUN_DIR/vllm-adapters/step-$s"
done

EVAL_RUNS="$CHAIN_NAME=$R/$RUN_DIR" \
EVAL_OUT="$R/runs/eval-$CHAIN_NAME" \
IFEVAL_ADAPTERS="$adapters" \
IFEVAL_OUT="$R/runs/ifeval-evals/$CHAIN_NAME" \
  bash scripts/run_arm_eval_combo.sh
echo "[chain] complete: eval-$CHAIN_NAME + ifeval-evals/$CHAIN_NAME"
