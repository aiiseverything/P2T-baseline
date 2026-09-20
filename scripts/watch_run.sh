#!/usr/bin/env bash
# Live monitor for a detached run: redraw the curves and check for trouble.
#
# Runs as its own detached process so it outlives any terminal, and never
# touches the trainer: it only reads metrics.jsonl and writes into report_dir.
#
#   setsid nohup bash scripts/watch_run.sh p2t250 300 >/dev/null 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: watch_run.sh <run-name> [interval-seconds]}"
INTERVAL="${2:-300}"
CONFIG="${ROOT}/configs/${RUN}.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/$(${PY} -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"
REPORT="${ROOT}/$(${PY} -c "import json,sys;print(json.load(open(sys.argv[1]))['report_dir'])" "$CONFIG")"
LOG="${OUT}/train.log"
PID_FILE="${OUT}/train.pid"
HEALTH="${REPORT}/health.log"
mkdir -p "$REPORT"

echo "=== watcher started $(date -Is) for ${RUN}, every ${INTERVAL}s ===" >> "$HEALTH"
while true; do
  if [ -f "${REPORT}/metrics.jsonl" ]; then
    "${PY}" "${ROOT}/scripts/plot_reward.py" --run "$REPORT" >>"$HEALTH" 2>&1
    "${PY}" "${ROOT}/scripts/check_run_health.py" --report "$REPORT" >> "$HEALTH" 2>&1
  fi
  if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "=== training process gone at $(date -Is) ===" >> "$HEALTH"
    tail -5 "$LOG" >> "$HEALTH" 2>/dev/null
    break
  fi
  sleep "$INTERVAL"
done
