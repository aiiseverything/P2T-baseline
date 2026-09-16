#!/usr/bin/env bash
set -euo pipefail
R=/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
cd "$R"
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! python3 -c "import peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn "peft==0.20.0" "pyarrow>=15,<22"
fi

# SFT v2: fixed assistant-only extraction, optional cleaned data, free-running
# monitor. Grid (2026-09-15): {raw,clean} x {10k x 2ep (=v1 params), 2.5k x 8ep
# (equal 625 steps)}. All arms carry the echo fix.
mode="${SFT_MODE:-smoke}"
SFT_MODEL="${SFT_MODEL:-models/Qwen3-14B-Base}"
SFT_DATA="${SFT_DATA:-datasets/sft_v2/sft_raw.parquet}"
SFT_OUTPUT="${SFT_OUTPUT:-runs/sftv2-dev}"
SFT_MAX_EXAMPLES="${SFT_MAX_EXAMPLES:-10000}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_MONITOR_EVERY="${SFT_MONITOR_EVERY:-100}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_EOS_WEIGHT="${SFT_EOS_WEIGHT:-1.0}"
SFT_TVT="${SFT_TVT:-0}"   # 1 = roll 25 train + 25 test after training (same job)

case "$mode" in
  smoke)
    exec python3 scripts/sft_init.py \
      --model "$SFT_MODEL" --dataset-path "$SFT_DATA" \
      --limit 300 --micro-batch 2 --grad-accum 4 \
      --monitor-every 5 \
      --output "$SFT_OUTPUT" ;;
  full)
    python3 scripts/sft_init.py \
      --model "$SFT_MODEL" --dataset-path "$SFT_DATA" \
      --max-examples "$SFT_MAX_EXAMPLES" --epochs "$SFT_EPOCHS" \
      --learning-rate "$SFT_LR" --eos-weight "$SFT_EOS_WEIGHT" \
      --monitor-every "$SFT_MONITOR_EVERY" \
      --output "$SFT_OUTPUT"
    if [ "$SFT_TVT" = "1" ]; then
      # post-training rollout through the identical tvt protocol
      TVT_ADAPTER="$SFT_OUTPUT" TVT_OUT="runs/tvt-$(basename "$SFT_OUTPUT")" \
        TVT_TRAIN="$R/datasets/sft_v2/train25.jsonl" \
        TVT_TEST="$R/datasets/sft_v2/test25.jsonl" \
        bash "$R/scripts/run_train_vs_test.sh"
    fi ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
