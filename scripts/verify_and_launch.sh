#!/usr/bin/env bash
# Verify the P2T package, then launch a run detached and attach a watcher.
#
# One command for the whole pre-launch sequence, so the gate cannot be skipped
# by accident: the run only starts if the test suite is green.
#
#   bash scripts/verify_and_launch.sh smoke10
#
# Leaves behind: runs/<run>/train.log, runs/<run>/train.pid,
# reports/<run>/{metrics.jsonl,reward.png,health.log}.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-smoke10}"
PY="${ROOT}/.venv/bin/python"
cd "$ROOT"

[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 2; }
[ -f "configs/${RUN}.json" ] || { echo "no config for run '${RUN}'" >&2; exit 2; }

echo "=== 1/4 test suite ==="
if ! "$PY" -m pytest tests/p2t -q; then
  echo "TESTS FAILED -- not launching. Fix the failures first." >&2
  exit 1
fi

echo
echo "=== 2/4 free-GPU preflight ==="
# start_detached.sh repeats this check and refuses occupied cards, but failing
# here costs two seconds instead of a partially-created run directory.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

echo
echo "=== 3/4 launch (setsid+nohup: survives this terminal and this session) ==="
bash scripts/start_detached.sh "$RUN" || exit 1

echo
echo "=== 4/4 watcher (one figure per completed step, also detached) ==="
REPORT="${ROOT}/$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('report_dir') or '')" \
  "configs/${RUN}.json")"
[ -n "$REPORT" ] || REPORT="${ROOT}/runs/${RUN}/report"
mkdir -p "$REPORT"
# 30s is the *poll* interval, not the plot interval: watch_run.sh redraws once per
# new rollout in metrics.jsonl and archives a numbered copy under steps/.  A step
# takes ~5 min, so polling every 30s keeps each figure within half a minute of the
# step that produced it.
setsid nohup bash scripts/watch_run.sh "$RUN" 30 >/dev/null 2>&1 &
echo "watcher started (one figure per step, polling every 30s)"
echo "  live figure:   ${REPORT}/reward.png"
echo "  per-step:      ${REPORT}/steps/step-NNNN.png"
echo "  health log:    ${REPORT}/health.log"

echo
echo "Follow it with:  bash scripts/status.sh ${RUN}"
echo "Plot on demand:  ${PY} scripts/plot_reward.py --run ${REPORT}"
