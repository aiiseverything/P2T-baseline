#!/usr/bin/env bash
# Launch a RED run detached, so it survives a closed terminal.
#
#   bash red/scripts/start_red.sh red250 [extra run_red.py args...]
#
# A self-contained copy of the sibling arm's launcher on purpose: nothing here
# touches the shared scripts/ directory, so a RED run can never be blocked by, or
# accidentally disturb, a P2T run that is already using those files.
set -euo pipefail

RUN="${1:?usage: start_red.sh <run-name> [args...]}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/$RUN.json"
if [ ! -f "$CONFIG" ]; then
  echo "no config at $CONFIG" >&2
  exit 1
fi

OUTPUT_DIR="$(python3 -c "import json,sys; print(json.load(open('$CONFIG'))['output_dir'])")"
PID_FILE="$OUTPUT_DIR/train.pid"
LOG_FILE="$OUTPUT_DIR/train.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running: pid $(cat "$PID_FILE")" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Two ways to place the three roles, and the config must agree with whichever is
# used:
#
#   RED_GPUS=4,5,6   --1,2   remaps physical cards to 0,1,2, so the config must
#                            name actor_device cuda:0, reward_device cuda:1 and
#                            vllm_gpus as *physical* ids (["6"] for a 3-card run,
#                            because vllm.py hands vllm_gpus straight to the
#                            subprocess as its own CUDA_VISIBLE_DEVICES, and with
#                            tp=1 vLLM takes the first of them).
#   RED_GPUS unset    --     nothing is remapped, so the config names absolute
#                            ids (cuda:4 / cuda:5 / vllm_gpus ["6"]).  The
#                            trainer's device-plan check is skipped in this mode
#                            because it keys off a non-empty CUDA_VISIBLE_DEVICES.
#
# Either way every card the run will touch must be free before it starts.
if [ -n "${RED_GPUS:-}" ]; then
  GPUS="$RED_GPUS"
  export CUDA_VISIBLE_DEVICES="$GPUS"
else
  GPUS=""
fi
# The cards to check are the ones the config actually names: with remapping the
# actor/reward entries are indices into RED_GPUS, without it they are absolute.
CHECK="$(python3 - "$CONFIG" "$GPUS" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
remap = [g for g in sys.argv[2].split(",") if g]
def index(dev):
    return dev.split(":")[-1]
cards = []
for role in ("actor_device", "reward_device"):
    value = index(cfg[role])
    cards.append(remap[int(value)] if remap else value)
cards += [str(g) for g in cfg["vllm_gpus"]]
print(" ".join(dict.fromkeys(cards)))
PY
)"
for gpu in $CHECK; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0)"
  if [ "${used:-0}" -gt 2048 ]; then
    echo "GPU $gpu is already holding ${used} MiB; refusing to launch" >&2
    exit 1
  fi
done
echo "device plan: config names [$CHECK]"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "=== red run $RUN started $(date --iso-8601=seconds) on GPUs $GPUS ==="
} >> "$LOG_FILE"

setsid nohup "$ROOT/.venv/bin/python" red/scripts/run_red.py --config "configs/$RUN.json" "$@" \
  >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "started $RUN: pid $(cat "$PID_FILE") -> $LOG_FILE"
