#!/usr/bin/env bash
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
export PYTHONPATH="$R:$R/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if ! python3 -c "import peft" 2>/dev/null; then
  python3 -m pip install --no-cache-dir --quiet \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn peft
fi

# TVT_ADAPTERS: space-separated TAG=PATH entries; one engine per dataset,
# each sweeping all adapters.
ADAPTERS="${TVT_ADAPTERS:?required}"
OUT="${TVT_OUT:?required}"
RECIPES="${TVT_RECIPES:-1.0:1:1.0:-1}"

TRAIN_A=""; TEST_A=""
for spec in $ADAPTERS; do
  tag="${spec%%=*}"; path="${spec#*=}"
  TRAIN_A="$TRAIN_A ${tag}-train=$path"
  TEST_A="$TEST_A ${tag}-test=$path"
done

python3 scripts/eval_alpaca.py \
  --output "$OUT" --max-tokens 2048 --recipes "$RECIPES" \
  --dataset datasets/sft_v2/train25.jsonl \
  --adapters $TRAIN_A

python3 scripts/eval_alpaca.py \
  --output "$OUT" --max-tokens 2048 --recipes "$RECIPES" \
  --dataset datasets/sft_v2/test25.jsonl \
  --adapters $TEST_A

echo "=== length summary ==="
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/tvt-dual/*/generations_t*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    lens = sorted(r["response_tokens"] for r in rows)
    n = len(lens)
    pin = sum(t >= 2048 for t in lens)
    print(f"{f.split('/')[-2]:24s} n={n}  p10 {lens[n//10]:4d}  median {lens[n//2]:4d}  "
          f"mean {sum(lens)/n:5.0f}  | pinned@2048 {100*pin/n:3.0f}%")
EOF