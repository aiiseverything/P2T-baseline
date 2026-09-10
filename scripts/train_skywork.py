#!/usr/bin/env python
"""Launch Skywork GRPO/VPO-RM training without submitting a cluster job.

Examples:
  python scripts/train_skywork.py --smoke --prompts-file prompts.txt
  python scripts/train_skywork.py --max-rollouts 500 --prompts-per-rollout 8 --group-size 8
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Allow execution directly from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vpo_rm.trainer import TrainerConfig, VPOTrainer, load_prompt_dataset, split_prompts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--rm", default="Skywork/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--output-dir", default="runs/skywork")
    p.add_argument("--method", choices=("grpo", "vpo_rm"), default="vpo_rm")
    p.add_argument("--prompts-file", help="newline-delimited prompts; otherwise load UltraFeedback")
    p.add_argument("--dataset-path", help="local train_prefs parquet; avoids dataset download")
    p.add_argument("--max-rollouts", type=int, default=TrainerConfig.rollout_iterations)
    p.add_argument("--prompts-per-rollout", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-response-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--actor-device", default="cuda:0")
    p.add_argument("--reward-device", default="cuda:1")
    p.add_argument("--smoke", action="store_true", help="hard cap to <=1 rollout, 2 prompts, 2 responses/group")
    args = p.parse_args(argv)
    if args.prompts_file:
        prompts = [x.strip() for x in Path(args.prompts_file).read_text().splitlines() if x.strip()]
        prompts, valid, hashes = split_prompts(prompts, validation_size=2000)
        if not prompts:
            prompts = valid
    else:
        prompts, valid, hashes = load_prompt_dataset(
            "HuggingFaceH4/ultrafeedback_binarized", validation_size=2000,
            dataset_path=args.dataset_path)
    cfg = TrainerConfig(model_name=args.model, reward_model_name=args.rm, output_dir=args.output_dir,
                        method=args.method, rollout_iterations=args.max_rollouts,
                        prompts_per_rollout=args.prompts_per_rollout, group_size=args.group_size,
                        max_response_tokens=args.max_response_tokens,
                        seed=args.seed, top_k=args.top_k,
                        smoke=args.smoke, actor_device=args.actor_device,
                        reward_device=args.reward_device)
    cfg = cfg.resolved()
    # Record split information before loading potentially large checkpoints.
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "data_split.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))
    trainer = VPOTrainer.from_pretrained(cfg)
    prompts = trainer.filter_prompts(prompts)
    if not prompts:
        raise ValueError("No prompts remain after actor/RM template length filtering")
    split_meta = dict(hashes, filtered_train_prompts=len(prompts),
                      dropped_train_prompts=trainer.filtered_prompt_count)
    trainer.data_split = split_meta
    (out / "data_split.json").write_text(json.dumps(split_meta, indent=2, sort_keys=True))
    trainer.train(prompts)
    print(json.dumps({"output_dir": str(out), "rollouts": trainer.rollout_index,
                      "total_tokens": trainer.total_tokens, "split": split_meta}, sort_keys=True))


if __name__ == "__main__":
    main()
