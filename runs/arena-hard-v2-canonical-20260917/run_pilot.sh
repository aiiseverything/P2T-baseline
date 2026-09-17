#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$SUITE/job/pilot.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/job/pilot_exit_code"' EXIT
/root/.venvs/alpacaeval/bin/python "$SUITE/source/scripts/judge_arena_hard.py" \
  --questions "$SUITE/question.jsonl" \
  --baseline "$SUITE/model_answer/o3-mini-2025-01-31.jsonl" \
  --answers-dir "$SUITE/model_answer" \
  --output-dir "$SUITE/model_judgment/gpt-4.1" \
  --tags base sft-init grpo lam2 lam4 lam8 \
  --uids-file "$SUITE/pilot_uids.json" --workers 12 --budget-cny 20
