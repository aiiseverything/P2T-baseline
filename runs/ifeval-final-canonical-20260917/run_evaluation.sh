#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SUITE/source:$SUITE/.ifeval-extra:$SUITE/source/.vllm-extra:$SUITE/source/third_party/ifeval${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="$SUITE/source/third_party/ifeval/nltk_data"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
exec > >(tee -a "$SUITE/job/job.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/job/exit_code"' EXIT
python3 "$SUITE/run_evaluation.py"
