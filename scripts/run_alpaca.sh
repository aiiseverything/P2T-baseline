#!/usr/bin/env bash
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if ! python3 -c "import peft" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn peft
fi

# Offline gate (dataset size, recipe/adapters validation) before any GPU work.
python3 scripts/eval_alpaca.py --selftest

# Checkpoint sweep: space-separated TAG=PATH entries ("none" = bare base model).
ADAPTERS="${ALPACA_ADAPTERS:?ALPACA_ADAPTERS=tag=path,... required}"
RECIPES="${ALPACA_RECIPES:-1.0:1:1.0:-1}"

# Optional sharding for multi-GPU fan-out: ALPACA_SHARD=k/N keeps adapters at
# indices i where i%N==k and pins the process to GPU k.
if [ -n "${ALPACA_SHARD:-}" ]; then
  k="${ALPACA_SHARD%%/*}"; n="${ALPACA_SHARD##*/}"
  export CUDA_VISIBLE_DEVICES="$k"
  filtered=(); i=0
  for spec in $ADAPTERS; do
    [ $((i % n)) -eq "$k" ] && filtered+=("$spec")
    i=$((i + 1))
  done
  ADAPTERS="${filtered[*]}"
  echo "shard $ALPACA_SHARD: ${#filtered[@]} tags, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  [ -n "$ADAPTERS" ] || { echo "shard empty, nothing to do"; exit 0; }
fi

exec python3 scripts/eval_alpaca.py \
  --model "${ALPACA_MODEL:-${EVAL_MODEL:-models/Qwen3-14B-Base}}" \
  --output "${ALPACA_OUT:?required}" \
  --max-tokens "${ALPACA_MAX_TOKENS:-2048}" \
  --recipes $RECIPES \
  --adapters $ADAPTERS
