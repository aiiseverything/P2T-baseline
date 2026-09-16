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

ADAPTER="${TVT_ADAPTER:?required}"   # models/sftv2-clean-2k5
OUT="${TVT_OUT:-runs/train-vs-test-clean2k5}"
TRAIN_DATA="${TVT_TRAIN:-datasets/sft_v2/train100_clean2k5.jsonl}"
TEST_DATA="${TVT_TEST:-datasets/sft_v2/test100_clean2k5.jsonl}"
RECIPES="${TVT_RECIPES:-1.0:1:1.0:-1}"

# two engine startups (one per dataset)
python3 scripts/eval_alpaca.py \
  --output "$OUT" --max-tokens 2048 --recipes "$RECIPES" \
  --dataset "$TRAIN_DATA" \
  --adapters "clean2k5-train=$ADAPTER"

python3 scripts/eval_alpaca.py \
  --output "$OUT" --max-tokens 2048 --recipes "$RECIPES" \
  --dataset "$TEST_DATA" \
  --adapters "clean2k5-test=$ADAPTER"

echo "=== length summary ==="
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/train-vs-test-clean2k5*/**/generations_*.jsonl", recursive=True)):
    rows = [json.loads(l) for l in open(f)]
    lens = sorted(r["response_tokens"] for r in rows)
    n = len(lens)
    pin = sum(t >= 2048 for t in lens)
    tag = f.split('/')[-2].replace('clean2k5-','')
    print(f"{f.split('/')[1]}/{tag:10s} n={n}  tokens: p10 {lens[n//10]}  median {lens[n//2]}  "
          f"p90 {lens[9*n//10]}  mean {sum(lens)/n:.0f}  | pinned@2048: {100*pin/n:.0f}%")
EOF