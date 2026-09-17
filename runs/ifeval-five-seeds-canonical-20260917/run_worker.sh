#!/usr/bin/env bash
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SUITE/../.." && pwd)"
TAG="${1:?model tag required}"
case "$TAG" in sft-init|grpo|lam2|lam4|lam8) ;; *) exit 2 ;; esac
mkdir -p "$SUITE/job/$TAG"
exec > >(tee -a "$SUITE/job/$TAG/worker.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$SUITE/job/$TAG/exit_code"' EXIT
TASK_CACHE="$(mktemp -d "/tmp/ifeval5-${TAG}.XXXXXX")"
export HF_HOME="$TASK_CACHE/hf" HF_DATASETS_CACHE="$TASK_CACHE/datasets"
export VLLM_CACHE_ROOT="$TASK_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor"
export TRITON_CACHE_DIR="$TASK_CACHE/triton" XDG_CACHE_HOME="$TASK_CACHE/xdg"
export PYTHONPATH="$SUITE/source:$PROJECT/runs/ifeval-final-canonical-20260917/.ifeval-extra:$PROJECT/runs/arena-hard-v2-canonical-20260917/.arena-extra:$PROJECT/.vllm-extra:$SUITE/source/third_party/ifeval"
export NLTK_DATA="$SUITE/source/third_party/ifeval/nltk_data"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PROJECT"
python3 "$SUITE/run_worker.py" "$TAG"
