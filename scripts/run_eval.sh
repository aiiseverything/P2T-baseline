#!/usr/bin/env bash
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$R"
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! python3 -c "import peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    "peft==0.20.0" "pyarrow>=15,<22"
fi

# EVAL_RUNS: semicolon-separated label=path pairs (rjob-safe, no spaces needed)
# EVAL_OUT: output dir under runs/
# EVAL_TEMPS: space-separated temperatures (default "0.7 0.0")
args=(scripts/eval_checkpoints.py --model "${EVAL_MODEL:-models/Qwen3-14B-Base}"
      --rm "${EVAL_RM:-models/Skywork-Reward-V2-Qwen3-8B}")
IFS=';' read -r -a run_specs <<< "${EVAL_RUNS:?EVAL_RUNS=label=path required}"
for spec in "${run_specs[@]}"; do
  [[ "$spec" == *=* && -n "${spec%%=*}" && -n "${spec#*=}" ]] || {
    echo "Invalid EVAL_RUNS entry: $spec (expected label=path)" >&2; exit 2;
  }
  args+=(--run "$spec")
done
read -r -a temperatures <<< "${EVAL_TEMPS:-0.7 0.0}"
exec python3 "${args[@]}" --temps "${temperatures[@]}" \
  --output "${EVAL_OUT:?EVAL_OUT required}"
