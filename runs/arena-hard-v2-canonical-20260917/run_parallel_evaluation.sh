#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SUITE/source:$SUITE/.arena-extra:$SUITE/source/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="$SUITE/.tiktoken-cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
mkdir -p "$SUITE/job"
exec > >(tee -a "$SUITE/job/parallel.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/job/parallel_exit_code"' EXIT
cd "$SUITE/../.."
python3 "$SUITE/run_parallel_evaluation.py"
