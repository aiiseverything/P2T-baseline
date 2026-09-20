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
# Actor on 0, reward model on 1, vLLM on 2 and 3.  Set explicitly rather than
# inherited: the trainer's actor/reward indices are relative to this list while
# vllm_gpus names physical cards, so an inherited value would silently run the
# actor on one set of GPUs and generation on another.
export CUDA_VISIBLE_DEVICES="${P2T_GPUS:-0,1,2,3}"

# Refuse to start onto occupied cards: an OOM 90 seconds into model loading is
# a worse failure than not starting.
python - <<'CHECK'
import os, subprocess, sys
visible = [g.strip() for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
used = {}
for line in out.strip().splitlines():
    index, memory = (part.strip() for part in line.split(","))
    used[index] = int(memory)
busy = [(g, used.get(g, 0)) for g in visible if used.get(g, 0) > 2048]
if busy:
    print(f"refusing to start: GPUs {busy} already hold more than 2 GiB", file=sys.stderr)
    raise SystemExit(1)
print("gpu preflight ok:", {g: used.get(g, 0) for g in visible})
CHECK

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
