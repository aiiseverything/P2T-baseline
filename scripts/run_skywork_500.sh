#!/usr/bin/env bash
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$(dirname "${BASH_SOURCE[0]}")/model_defaults.sh"
cd "$R"
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Both OOM failures showed multiple GiB reserved-but-unallocated; large vocab
# tensors of varying width make fragmentation a real failure mode over 500
# rollouts.  The vLLM engine runs in a separate process and is unaffected.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The derived training image is not pullable on every node; the base vLLM
# image plus an idempotent internal-mirror install works everywhere.
if ! python3 -c "import peft, pyarrow" 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    "peft==0.20.0" "pyarrow>=15,<22"
fi
MAX_ROLLOUTS="${MAX_ROLLOUTS:-500}"
MODEL="${MODEL:-models/Qwen3-14B-Base}"
RM="${RM:-models/Skywork-Reward-V2-Qwen3-8B}"
INIT_ADAPTER="${INIT_ADAPTER:-$(default_initial_adapter "$MODEL")}"
# Per-arm overrides: p9h VPO runs at LR=3e-5 (credit concentration acts as a
# ~10-20x effective-lr multiplier on hot tokens; see vpo坍缩分析-p9g.md).
LR="${LR:-5e-5}"
CREDIT_LAMBDA="${CREDIT_LAMBDA:-2.0}"
SEED="${SEED:-42}"
FREEZE_FLAG=""
[ "${FREEZE_STOP_TOKENS:-0}" = "1" ] && FREEZE_FLAG="--freeze-stop-tokens"
[ "${FREEZE_STRUCTURAL:-0}" = "1" ] && FREEZE_FLAG="$FREEZE_FLAG --freeze-structural"
# Present on cloned jobs; rjob injects it only with -e DISTRIBUTED_JOB=true.
JOB_ID="${JOB_ID:-local-$(date +%Y%m%d%H%M%S)}"; export JOB_ID

method="${1:?usage: run_skywork_500.sh grpo|vpo_rm}"
case "$method" in
  grpo|vpo_rm) ;;
  *) echo "unsupported method: $method" >&2; exit 2 ;;
esac

# Gate on the unit tests before committing GPU hours (no torch on the login
# node, so tests can only run inside the job image).
python3 -m pytest tests/test_core.py tests/test_token_policy.py tests/test_integration.py tests/test_trainer.py -q

# p9g: shared SFT init (stage 0), init-anchored KL (beta 0.03), calibrated
# length debias (5.06e-3/token below 600, overlong floored), guard at 64,
# lr 5e-5, entropy monitoring.  Rationale: p9f proved the RM's short-answer
# bias is lr-insensitive, so the fixes target the reward landscape.
exec python3 scripts/profile_vllm_full.py \
  --method "$method" \
  --model "$MODEL" \
  --rm "$RM" \
  --learning-rate "$LR" \
  --seed "$SEED" \
  --tau 1.0 \
  --credit-lambda "$CREDIT_LAMBDA" \
  $FREEZE_FLAG \
  --beta 0.03 \
  --init-adapter "$INIT_ADAPTER" \
  --kl-reference init \
  --length-penalty-slope 0.00506 \
  --length-penalty-anchor 600 \
  --min-response-tokens 64 \
  --max-rollouts "$MAX_ROLLOUTS" \
  --max-response-tokens 2048 \
  --generation-microbatch 32 \
  --keep-adapters-every 50 \
  --output-dir "runs/formal-skywork-${method}-${JOB_ID:?}"
