#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM
export PYTHONPATH="$PWD:$PWD/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Use dependencies already in the image and shared cache; no network install.
python3 - <<'CHECK'
import importlib.util, runpy, subprocess, sys
if importlib.util.find_spec("pytest"):
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_trainer.py",
                    "tests/test_core.py", "tests/test_integration.py", "-q"], check=True)
else:
    ns = runpy.run_path("tests/test_trainer.py")
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}", flush=True)
CHECK
nvidia-smi --query-gpu=name,memory.total --format=csv
for method in grpo vpo_rm; do
    python3 scripts/profile_vllm_full.py \
        --method "$method" --max-rollouts 3 \
        --output-dir "runs/profile-compare-${JOB_ID:?}-${method}" \
        --max-response-tokens 2048 --generation-microbatch 32
done
