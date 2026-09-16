#!/usr/bin/env bash
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra:$R/third_party/ifeval"
export NLTK_DATA="$R/third_party/ifeval/nltk_data"
export PYTHONUNBUFFERED=1

# Official IFEval code dependencies (internal mirror, fast on GPU nodes).
# punkt_tab ships inside the repo (downloaded once via proxy) so GPU nodes
# need no NLTK download.
if ! python3 -c "import absl, immutabledict, langdetect, nltk" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    absl-py immutabledict langdetect nltk
fi
if ! python3 -c "import peft" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ peft
fi

# Offline gate on the scoring/aggregation path before any GPU work.
python3 scripts/eval_ifeval.py --selftest

# Checkpoint sweep: space-separated TAG=PATH entries ("none" = bare base model).
# Single-adapter jobs keep the legacy IFEVAL_ADAPTER env var.
ADAPTERS="${IFEVAL_ADAPTERS:-}"
if [ -z "$ADAPTERS" ] && [ -n "${IFEVAL_ADAPTER:-}" ]; then
  ADAPTERS="$(basename "${IFEVAL_ADAPTER%/}")=${IFEVAL_ADAPTER}"
fi

# Sampling recipe (default: training temperature 1.0, single sample).
RECIPES="${IFEVAL_RECIPES:-1.0:1:1.0:-1}"

exec python3 scripts/eval_ifeval.py \
  --model "${IFEVAL_MODEL:-${EVAL_MODEL:-models/Qwen3-14B-Base}}" \
  --output "${IFEVAL_OUT:?required}" \
  --recipes $RECIPES \
  ${ADAPTERS:+--adapters $ADAPTERS}
