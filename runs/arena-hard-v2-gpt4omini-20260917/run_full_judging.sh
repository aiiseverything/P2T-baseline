#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_BUDGET_CNY="${1:?require measured pilot-based cumulative budget}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export TIKTOKEN_CACHE_DIR="$SUITE/.tiktoken-cache"
exec > >(tee -a "$SUITE/job/full_judging.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/job/full_judging_exit_code"' EXIT
/root/miniconda3/envs/sml/bin/python "$SUITE/run_evaluation.py" --validate-only
/root/.venvs/alpacaeval/bin/python "$SUITE/source/scripts/judge_arena_hard.py" \
  --baseline-model gpt-4o-mini-2024-07-18 \
  --questions "$SUITE/question.jsonl" \
  --baseline "$SUITE/model_answer/gpt-4o-mini-2024-07-18.jsonl" \
  --answers-dir "$SUITE/model_answer" \
  --output-dir "$SUITE/model_judgment/gpt-4.1" \
  --tags base sft-init grpo lam2 lam4 lam8 --workers 32 --budget-cny "$TASK_BUDGET_CNY"
/root/miniconda3/envs/sml/bin/python "$SUITE/source/scripts/score_arena_hard.py" \
  --baseline-model gpt-4o-mini-2024-07-18 \
  --questions "$SUITE/question.jsonl" --answers-dir "$SUITE/model_answer" \
  --judgments-dir "$SUITE/model_judgment/gpt-4.1" --output "$SUITE/scores" --seed 42 --threads 1
