#!/usr/bin/env bash
# Combined per-arm eval inside ONE 3-GPU rjob allocation:
#   cards 0+1 -> RM eval   (vLLM generation + Skywork scoring, run_eval.sh)
#   card 2    -> IFEval sweep (all checkpoints, run_ifeval.sh)
# Both run concurrently as independent processes; the job exits nonzero if
# either fails. Chaining evals into one held allocation avoids a second queue
# wait (bit us 2026-09-15: 14 separate eval jobs sat Inqueue 3h after training
# freed the GPUs, because the quota had been taken by other lab members).
#
# Required env: EVAL_RUNS, EVAL_OUT, IFEVAL_ADAPTERS, IFEVAL_OUT
# Optional env: EVAL_TEMPS (default "1.0 0.7"), IFEVAL_RECIPES (default "1.0:1:1.0:-1")
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra:$R/third_party/ifeval"
export NLTK_DATA="$R/third_party/ifeval/nltk_data"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Serialize dependency installs ONCE up front so the two concurrent eval
# processes don't race pip inside the shared image env.
if ! python3 -c "import absl, immutabledict, langdetect, nltk, peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    absl-py immutabledict langdetect nltk "peft==0.20.0" "pyarrow>=15,<22"
fi

: "${EVAL_RUNS:?EVAL_RUNS=label=path required}"
: "${EVAL_OUT:?EVAL_OUT required}"
: "${IFEVAL_ADAPTERS:?IFEVAL_ADAPTERS=tag=path,... required}"
: "${IFEVAL_OUT:?IFEVAL_OUT required}"

mkdir -p "$(dirname "$EVAL_OUT")" "$(dirname "$IFEVAL_OUT")"
rm_eval_log="$EVAL_OUT.combo.log"
ifeval_log="$IFEVAL_OUT.combo.log"

echo "[combo] RM eval   -> cards 0,1  log: $rm_eval_log"
echo "[combo] IFEval    -> card 2    log: $ifeval_log"

CUDA_VISIBLE_DEVICES=0,1 \
EVAL_TEMPS="${EVAL_TEMPS:-1.0 0.7}" \
  bash "$R/scripts/run_eval.sh" > "$rm_eval_log" 2>&1 &
rm_pid=$!

CUDA_VISIBLE_DEVICES=2 \
IFEVAL_RECIPES="${IFEVAL_RECIPES:-1.0:1:1.0:-1}" \
  bash "$R/scripts/run_ifeval.sh" > "$ifeval_log" 2>&1 &
if_pid=$!

fail=0
wait "$rm_pid" || { echo "[combo] RM eval FAILED (tail of $rm_eval_log):"; tail -5 "$rm_eval_log"; fail=1; }
wait "$if_pid" || { echo "[combo] IFEval FAILED (tail of $ifeval_log):"; tail -5 "$ifeval_log"; fail=1; }

if [ "$fail" = 0 ]; then
  echo "[combo] both evals succeeded"
fi
exit "$fail"
