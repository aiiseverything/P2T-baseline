#!/usr/bin/env bash
# Detached monitor for a RED run: redraws the curves once per rollout and runs the
# health check, appending everything to <report_dir>/health.log.
#
#   setsid nohup bash red/scripts/watch_red.sh red250 30 &
#
# Mirrors the sibling arm's watcher but drives the RED plotter and checker, and
# lives here so a RED run never depends on a file the other arm owns.  Exits when
# the run's pid file stops naming a live process.
set -uo pipefail

RUN="${1:?usage: watch_red.sh <run-name> [interval-seconds]}"
INTERVAL="${2:-30}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/$RUN.json"
if [ ! -f "$CONFIG" ]; then
  echo "no config at $CONFIG" >&2
  exit 1
fi
read -r OUTPUT_DIR REPORT_DIR < <(python3 -c "
import json
c = json.load(open('$CONFIG'))
print(c['output_dir'], c.get('report_dir') or (c['output_dir'].rstrip('/') + '/report'))
")
PID_FILE="$ROOT/$OUTPUT_DIR/train.pid"
REPORT="$ROOT/$REPORT_DIR"
LOG="$REPORT/health.log"
mkdir -p "$REPORT" "$REPORT/steps"

echo "=== watcher started $(date --iso-8601=seconds) for $RUN (every ${INTERVAL}s) ===" >> "$LOG"
last=0
while true; do
  if [ ! -f "$PID_FILE" ] || ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "=== train.pid is gone; watcher stopping $(date --iso-8601=seconds) ===" >> "$LOG"
    break
  fi
  step="$(python3 -c "
import json
try:
    rows=[json.loads(l) for l in open('$REPORT/metrics.jsonl') if l.strip()]
    print(max((r['rollout'] for r in rows if 'rollout' in r), default=0))
except FileNotFoundError:
    print(0)
" 2>/dev/null || echo 0)"
  if [ "${step:-0}" -gt "$last" ]; then
    last="$step"
    printf -v padded "%04d" "$step"
    "$ROOT/.venv/bin/python" "$ROOT/red/scripts/plot_red.py" \
      --run "$REPORT" --out "$REPORT/steps/step-$padded.png" >> "$LOG" 2>&1
    "$ROOT/.venv/bin/python" "$ROOT/red/scripts/plot_red.py" --run "$REPORT" >> "$LOG" 2>&1
    "$ROOT/.venv/bin/python" "$ROOT/red/scripts/check_red_health.py" \
      --report "$REPORT" >> "$LOG" 2>&1
  fi
  sleep "$INTERVAL"
done
