#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?arm required}"
case "$ARM" in grpo|lam2|lam4|lam8) ;; *) exit 2 ;; esac
export PYTHONPATH="$SUITE/source:$SUITE/source/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
exec > >(tee -a "$SUITE/$ARM/job.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/$ARM/exit_code"' EXIT
python3 "$SUITE/run_generation.py" "$ARM"
