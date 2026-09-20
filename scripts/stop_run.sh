#!/usr/bin/env bash
# Stop a detached P2T run and its generation server, then confirm the GPUs are free.
#
#   bash scripts/stop_run.sh formal250
#
# Why this is a script and not an inline command: `pkill -f <pattern>` matches
# against whole command lines, so an inline `pkill -f "watch_run.sh formal250"`
# also matches the shell that is running it and kills itself mid-sequence.  That
# happened once here, at the cost of a half-executed stop.  Every match below is
# filtered against this script's own process tree, and the trainer is addressed by
# its recorded pid rather than by pattern.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: stop_run.sh <run-name>}"
cd "$ROOT"

CONFIG="configs/${RUN}.json"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"

# Anything in this script's own ancestry must never be a kill target.
SELF_TREE=" $$ $PPID "
safe_kill() {  # safe_kill <signal> <pid...>
  local sig="$1"; shift
  for pid in "$@"; do
    [ -n "$pid" ] || continue
    case "$SELF_TREE" in *" $pid "*) echo "  skip $pid (self)"; continue;; esac
    kill "-$sig" "$pid" 2>/dev/null && echo "  sent $sig to $pid" || true
  done
}

echo "=== 1. watcher ==="
# -f matches full command lines; exclude our own tree explicitly.
safe_kill TERM $(pgrep -f "watch_run\.sh ${RUN}" 2>/dev/null || true)

echo "=== 2. trainer (SIGINT first: its finally block closes the vLLM child) ==="
TP=""
[ -f "${OUT}/train.pid" ] && TP="$(cat "${OUT}/train.pid")"
if [ -n "$TP" ] && kill -0 "$TP" 2>/dev/null; then
  safe_kill INT "$TP"
  for i in $(seq 1 36); do
    kill -0 "$TP" 2>/dev/null || { echo "  trainer exited after ~$((i*5))s"; break; }
    sleep 5
  done
  if kill -0 "$TP" 2>/dev/null; then echo "  still up -> SIGKILL"; safe_kill KILL "$TP"; sleep 10; fi
else
  echo "  trainer not running"
fi

echo "=== 3. leftover generation processes ==="
for _ in 1 2; do
  ORPHANS="$(pgrep -f "p2t\.vllm_server" 2>/dev/null || true)"
  [ -n "$ORPHANS" ] || { echo "  none"; break; }
  safe_kill KILL $ORPHANS
  sleep 12
done

echo "=== 4. GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
BUSY="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$1 < 4 && $2 > 2048 {print $1}' | tr '\n' ' ')"
if [ -n "$BUSY" ]; then
  echo "WARNING: GPUs ${BUSY}still hold >2 GiB; a relaunch will refuse to start" >&2
  exit 1
fi
echo "GPUs 0-3 are free"
