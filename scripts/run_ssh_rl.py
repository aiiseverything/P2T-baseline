#!/usr/bin/env python3
"""Launch one native-EOS experiment on explicitly selected SSH-server GPUs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("grpo", "lam2", "lam4", "lam8"), required=True)
    p.add_argument("--output-dir", required=True, help="Fresh directory for this run")
    p.add_argument("--gpus", default="0,1,2,3", help="Exactly 3 or 4 physical GPU indices/UUIDs")
    p.add_argument("--max-rollouts", type=int, default=250)
    p.add_argument("--length-reward-mode", choices=("soft", "legacy"), default="soft")
    p.add_argument("--sigma0", type=float, help="Shared initial-policy calibration; omit to calibrate")
    p.add_argument("--calibration-prompts", type=int, default=128)
    p.add_argument("--generation-microbatch", type=int, default=4)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=.85)
    p.add_argument("--dry-run", action="store_true", help="Print the command without loading any model")
    return p


def build_command(args):
    gpus = [x.strip() for x in args.gpus.split(",")]
    if len(gpus) not in (3, 4) or not all(gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Select 3 or 4 distinct GPUs: actor, reward, then 1 or 2 generation GPUs")
    if args.max_rollouts < 1 or args.calibration_prompts < 1 or args.generation_microbatch < 1:
        raise ValueError("Rollout, calibration and generation counts must be positive")
    if not math.isfinite(args.vllm_gpu_memory_utilization) or not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("vLLM memory utilization must be in (0, 1]")
    if args.sigma0 is not None and (not math.isfinite(args.sigma0) or args.sigma0 <= 0):
        raise ValueError("sigma0 must be finite and positive")
    if args.length_reward_mode == "legacy" and args.sigma0 is not None:
        raise ValueError("sigma0 belongs to soft length rewards, not the legacy protocol")
    arm_lambda = {"grpo": "1", "lam2": "2", "lam4": "4", "lam8": "8"}[args.arm]
    command = [sys.executable, str(ROOT / "scripts/profile_vllm_full.py"),
        "--model", str(ROOT / "models/Qwen3-14B-Base"),
        "--rm", str(ROOT / "models/Skywork-Reward-V2-Qwen3-8B"),
        "--init-adapter", str(ROOT / "models/sft-native-eos-clean2k5e2"),
        "--dataset-path", str(ROOT / "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet"),
        "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
        "--method", "grpo" if args.arm == "grpo" else "vpo_rm",
        "--credit-lambda", arm_lambda, "--credit-microbatch-responses", "1",
        "--max-rollouts", str(args.max_rollouts), "--max-response-tokens", "2048",
        "--learning-rate", "5e-5", "--beta", "0.03", "--tau", "1.0",
        "--kl-reference", "init", "--temperature", "1.0",
        "--seed", "42", "--generation-seed", "0", "--policy-epochs", "1",
        "--optimizer-minibatch-responses", "64", "--keep-adapters-every", "50",
        "--generation-microbatch", str(args.generation_microbatch),
        "--vllm-tensor-parallel-size", str(len(gpus) - 2),
        "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--length-reward-mode", args.length_reward_mode]
    if args.arm != "grpo":
        command += ["--freeze-stop-tokens", "--freeze-structural"]
    if args.length_reward_mode == "soft":
        command += ["--length-calibration-prompts", str(args.calibration_prompts)]
        if args.sigma0 is not None:
            command += ["--length-reward-sigma0", str(args.sigma0)]
    else:
        command += ["--min-response-tokens", "64", "--length-penalty-slope", "0.00506",
                    "--length-penalty-anchor", "600"]
    environment = {"CUDA_VISIBLE_DEVICES": ",".join(gpus), "PYTHONUNBUFFERED": "1",
                   "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    return command, environment


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        command, overrides = build_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"cwd": str(ROOT), "command": command, "environment": overrides}, indent=2), flush=True)
    if args.dry_run:
        return
    os.chdir(ROOT)
    os.execvpe(command[0], command, {**os.environ, **overrides})


if __name__ == "__main__":
    main()
