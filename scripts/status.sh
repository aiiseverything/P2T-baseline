#!/usr/bin/env bash
# Progress of a detached P2T run: liveness, tail of the log, latest metrics.
#
#   bash scripts/status.sh smoke10
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: status.sh <run-name>}"
CONFIG="${ROOT}/configs/${RUN}.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

PY="${ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY=python
OUT="${ROOT}/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"
REPORT="${ROOT}/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('report_dir') or '')" "$CONFIG")"
[ -n "$REPORT" ] || REPORT="${OUT}/report"
LOG="${OUT}/train.log"
PID_FILE="${OUT}/train.pid"

echo "run:       ${RUN}"
echo "output:    ${OUT}"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "status:    RUNNING (pid $(cat "$PID_FILE"))"
else
  echo "status:    not running"
fi
for gpu in 0 1 2 3; do
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$gpu" 2>/dev/null | sed 's/^/gpu /'
done

echo
echo "--- last metrics ---"
if [ -f "${REPORT}/metrics.jsonl" ]; then
  tail -1 "${REPORT}/metrics.jsonl" | python -m json.tool 2>/dev/null || tail -1 "${REPORT}/metrics.jsonl"
else
  echo "(no metrics yet)"
fi

echo
echo "--- last git push ---"
if [ -f "${REPORT}/git_push.log" ]; then tail -2 "${REPORT}/git_push.log"; else echo "(none yet)"; fi

echo
echo "--- train.log tail ---"
tail -15 "$LOG" 2>/dev/null || echo "(no log)"
