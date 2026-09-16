#!/usr/bin/env bash
# RM artifact calibration: run 2 vpo_rm rollouts under a NEW reward model,
# then measure EOS / structural-token |d_t| inflation from the credit dumps.
# Writes $CALIB_OUT/DONE on success — the VPO arms using this RM gate on that
# marker so training never starts with an unverified freeze list.
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/model_defaults.sh"
cd "$R"
export JOB_ID="${JOB_ID:-rm-calib-local}"
CALIB_MODEL="${CALIB_MODEL:-models/Qwen3-14B-Base}"
CALIB_RM="${CALIB_RM:?e.g. models/Skywork-Reward-V2-Qwen3-4B}"
CALIB_OUT="${CALIB_OUT:?e.g. runs/rm4b-calib}"
CALIB_INIT="${CALIB_INIT:-$(default_initial_adapter "$CALIB_MODEL")}"

for f in "$CALIB_MODEL/config.json" "$CALIB_RM/config.json" "$CALIB_INIT/adapter_config.json"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 3; }
done

MAX_ROLLOUTS=2 MODEL="$CALIB_MODEL" RM="$CALIB_RM" INIT_ADAPTER="$CALIB_INIT" \
CREDIT_LAMBDA=4.0 FREEZE_STOP_TOKENS=1 FREEZE_STRUCTURAL=1 \
  bash scripts/run_skywork_500.sh vpo_rm

RUN_DIR="runs/formal-skywork-vpo_rm-${JOB_ID}"
mkdir -p "$CALIB_OUT"
python3 scripts/analyze_rm_artifacts.py --run "$RUN_DIR" --out "$CALIB_OUT" --tokenizer "$CALIB_MODEL"
touch "$CALIB_OUT/DONE"
echo "calibration complete: $CALIB_OUT/summary.json"
