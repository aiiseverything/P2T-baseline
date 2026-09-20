#!/usr/bin/env bash
# Live monitor for a detached run: redraw the curves on every completed step.
#
# Runs as its own detached process so it outlives any terminal, and never touches
# the trainer: it only reads metrics.jsonl and writes into report_dir.
#
#   setsid nohup bash scripts/watch_run.sh p2t250 30 >/dev/null 2>&1 &
#
# The argument is a *poll* interval, not a plot interval.  Plotting is triggered
# by a new rollout appearing in metrics.jsonl, so there is exactly one redraw per
# training step regardless of how long a step takes.  Each redraw also archives a
# numbered copy under report_dir/steps/, giving a per-step time series rather than
# only the latest picture.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: watch_run.sh <run-name> [poll-seconds]}"
INTERVAL="${2:-30}"
CONFIG="${ROOT}/configs/${RUN}.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/$(${PY} -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"
REPORT="${ROOT}/$(${PY} -c "import json,sys;print(json.load(open(sys.argv[1]))['report_dir'])" "$CONFIG")"
TOTAL="$(${PY} -c "import json,sys;print(json.load(open(sys.argv[1]))['rollout_iterations'])" "$CONFIG")"
LOG="${OUT}/train.log"
PID_FILE="${OUT}/train.pid"
HEALTH="${REPORT}/health.log"
STEPS_DIR="${REPORT}/steps"
mkdir -p "$REPORT" "$STEPS_DIR"

echo "=== watcher started $(date -Is) for ${RUN}: poll ${INTERVAL}s, one figure per step ===" >> "$HEALTH"

last=-1
while true; do
  if [ -f "${REPORT}/metrics.jsonl" ]; then
    # Count completed rollouts.  grep -c exits 1 on zero matches and this script
    # runs without -e, but the fallback keeps `count` a number either way.
    count="$(grep -c '"rollout"' "${REPORT}/metrics.jsonl" 2>/dev/null || echo 0)"
    if [ "${count:-0}" -gt 0 ] && [ "${count:-0}" -gt "$last" ]; then
      echo "--- step ${count}/${TOTAL} at $(date -Is) ---" >> "$HEALTH"
      "${PY}" "${ROOT}/scripts/plot_reward.py" --run "$REPORT" \
        --title "P2T ${RUN} — step ${count}/${TOTAL}" >>"$HEALTH" 2>&1
      # Archive this step's figure alongside the live one.  Zero-padded so the
      # directory sorts chronologically.
      if [ -f "${REPORT}/reward.png" ]; then
        cp "${REPORT}/reward.png" "$(printf '%s/step-%04d.png' "$STEPS_DIR" "$count")"
      fi
      "${PY}" "${ROOT}/scripts/check_run_health.py" --report "$REPORT" >> "$HEALTH" 2>&1
      last="$count"
    fi
  fi
  if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "=== training process gone at $(date -Is) ===" >> "$HEALTH"
    tail -5 "$LOG" >> "$HEALTH" 2>/dev/null
    break
  fi
  sleep "$INTERVAL"
done
