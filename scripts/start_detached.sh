#!/usr/bin/env bash
# Start a P2T run detached from every terminal.
#
# tmux and screen are not installed on this host, so the run is made
# terminal-independent with setsid + nohup: setsid puts the process in its own
# session with no controlling terminal, which is what survives closing the
# laptop, an SSH drop, or the agent session that launched it.
#
#   bash scripts/start_detached.sh smoke10 [extra run_train.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: start_detached.sh <run-name> [extra args...]}"
shift || true
CONFIG="${ROOT}/configs/${RUN}.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

OUT_DIR="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"
OUT="${ROOT}/${OUT_DIR}"
mkdir -p "$OUT"
LOG="${OUT}/train.log"
PID="${OUT}/train.pid"

if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running: pid $(cat "$PID")" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Actor on 0, reward model on 1, vLLM on 2 and 3. The server subprocess
# re-exports its own CUDA_VISIBLE_DEVICES from the physical ids.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== run ${RUN} started $(date -Is) ===" >> "$LOG"
setsid nohup python "${ROOT}/scripts/run_train.py" --config "$CONFIG" "$@" \
  >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PID"
sleep 2
if kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "started ${RUN}: pid $(cat "$PID"), log ${LOG}"
else
  echo "failed to start; see ${LOG}" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
