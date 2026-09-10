#!/usr/bin/env python3
"""Full GRPO/VPO rollouts with Actor/RM/vLLM on three separate GPUs.

Reports startup separately from repeated generation, training and LoRA saving.
The first batch matches the earlier one-rollout profile; later batches advance
through the same deterministic prompt split as the main trainer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTRA = ROOT / ".vllm-extra"
if EXTRA.exists():
    sys.path.insert(0, str(EXTRA))

import torch
from vpo_rm.trainer import TrainerConfig, VPOTrainer, load_prompt_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B")
    p.add_argument("--rm", default="models/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--dataset-path", default="datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", choices=["grpo", "vpo_rm"], default="vpo_rm")
    p.add_argument("--max-rollouts", type=int, default=3)
    p.add_argument("--max-response-tokens", type=int, default=2048)
    p.add_argument("--generation-microbatch", type=int, default=32)
    p.add_argument("--keep-adapters-every", type=int, default=0,
                   help="Keep step-0, every Nth adapter, and the current adapter; 0 keeps all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generation-seed", type=int, default=0,
                   help="Matches the earlier profile's vLLM default seed")
    args = p.parse_args()
    if args.max_rollouts < 1:
        p.error("--max-rollouts must be positive")
    if args.keep_adapters_every < 0:
        p.error("--keep-adapters-every must be nonnegative")
    if torch.cuda.device_count() != 3:
        raise RuntimeError("This profile requires exactly three visible GPUs")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    started = time.monotonic()
    prompts, _, split = load_prompt_dataset("HuggingFaceH4/ultrafeedback_binarized", dataset_path=args.dataset_path)
    cfg = TrainerConfig(
        model_name=args.model, reward_model_name=args.rm, output_dir=str(out),
        method=args.method, rollout_iterations=args.max_rollouts,
        prompts_per_rollout=8, group_size=8,
        max_response_tokens=args.max_response_tokens,
        generation_microbatch_responses=args.generation_microbatch,
        microbatch_responses=1, seed=args.seed, smoke=False,
        actor_device="cuda:0", reward_device="cuda:1",
    ).resolved()
    # Seed before PEFT initializes LoRA A matrices, not only in trainer.__init__.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trainer = VPOTrainer.from_pretrained(cfg)
    prompts = trainer.filter_prompts(prompts)
    if len(prompts) < cfg.prompts_per_rollout * args.max_rollouts:
        raise RuntimeError("not enough prompts after filtering")
    trainer.data_split = split
    (out / "data_split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    (out / "profile_manifest.json").write_text(json.dumps({
        "config": asdict(cfg), "generation_seed": args.generation_seed,
        "gpu_count": 3, "gpu_names": [torch.cuda.get_device_name(i) for i in range(3)],
        "vllm_device": "cuda:2", "job_id": os.environ.get("JOB_ID"),
        "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                          for f in ["vpo_rm/trainer.py", "vpo_rm/integration.py",
                                    "scripts/profile_vllm_full.py", "scripts/vllm_generate_server.py"]},
    }, indent=2))
    adapter_root = out / "vllm-adapters"
    adapter_root.mkdir()
    adapter_path = adapter_root / "step-0"
    adapter_id = 1
    trainer.actor.save_pretrained(adapter_path)
    setup_sec = time.monotonic() - started
    print(f"devices actor=cuda:0 rm=cuda:1 vllm=cuda:2 method={cfg.method} "
          f"requests=64 max_tokens={cfg.max_response_tokens} setup_sec={setup_sec:.2f}", flush=True)
    tmp = tempfile.TemporaryDirectory(prefix="vpo-vllm-")
    socket_path = Path(tmp.name) / "server.sock"
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
    env["CUDA_VISIBLE_DEVICES"] = visible[2]
    env["PYTHONUNBUFFERED"] = "1"
    server_started = time.monotonic()
    server = subprocess.Popen([
        sys.executable, str(ROOT / "scripts/vllm_generate_server.py"),
        "--model", args.model, "--socket", str(socket_path),
        "--max-num-seqs", str(args.generation_microbatch),
        "--seed", str(args.generation_seed)], env=env)

    def request(payload):
        if server.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {server.returncode}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(1800)
            conn.connect(str(socket_path))
            conn.sendall((json.dumps(payload) + "\n").encode())
            with conn.makefile("r") as reader:
                line = reader.readline()
        if not line:
            raise RuntimeError("vLLM server closed without returning a result")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(f"vLLM failed: {result}")
        return result

    def sync_training_devices():
        for device in [trainer.actor_device, trainer.reward_device]:
            torch.cuda.synchronize(device)

    generation_stats = {}

    @torch.no_grad()
    def rollout_vllm(batch_prompts):
        nonlocal generation_stats
        trainer.actor.eval()
        batch, rendered = trainer._encode_prompts(batch_prompts)
        prompt_width = batch["input_ids"].shape[1]
        result = request({"prompts": rendered, "adapter": str(adapter_path),
                          "adapter_id": adapter_id, "max_tokens": cfg.max_response_tokens,
                          "group_size": cfg.group_size})
        rows = result.pop("rows")
        expected = len(batch_prompts) * cfg.group_size
        if len(rows) != expected or any(not row for row in rows):
            raise RuntimeError("vLLM returned missing or empty responses")
        lengths = [len(row) for row in rows]
        if max(lengths) > cfg.max_response_tokens:
            raise RuntimeError("vLLM exceeded response token limit")
        width = max(lengths)
        responses = torch.full((expected, width), trainer.actor_tokenizer.pad_token_id,
                               dtype=torch.long, device=trainer.actor_device)
        for i, row in enumerate(rows):
            responses[i, :len(row)] = torch.tensor(row, device=trainer.actor_device)
        rmask = torch.arange(width, device=trainer.actor_device)[None, :] < torch.tensor(
            lengths, device=trainer.actor_device)[:, None]
        expanded_input = batch["input_ids"].repeat_interleave(cfg.group_size, dim=0)
        prompt_mask = batch["attention_mask"].repeat_interleave(cfg.group_size, dim=0)
        input_ids = torch.cat([expanded_input, responses], dim=1)
        full_mask = torch.cat([prompt_mask, rmask.to(prompt_mask.dtype)], dim=1)
        positions = torch.arange(prompt_width, input_ids.shape[1], device=trainer.actor_device).expand(expected, -1)
        generation_stats = {**result, "response_lengths": lengths,
                            "mean_response_tokens": sum(lengths) / expected,
                            "padded_response_width": width, "prompt_width": prompt_width,
                            "truncation_rate": sum(n == cfg.max_response_tokens for n in lengths) / expected}
        (out / f"rollout-{trainer.rollout_index + 1}-tokens.json").write_text(json.dumps(rows))
        sync_training_devices()
        return input_ids, full_mask, positions, responses, rmask, [x for x in rendered for _ in range(cfg.group_size)]

    records = []
    try:
        while not socket_path.exists():
            if server.poll() is not None:
                raise RuntimeError(f"vLLM startup exited with code {server.returncode}")
            if time.monotonic() - server_started > 600:
                raise TimeoutError("vLLM startup timed out")
            time.sleep(1)
        startup_sec = time.monotonic() - server_started
        print(f"vllm_startup_sec={startup_sec:.2f} server_pid={server.pid}", flush=True)
        trainer.rollout = rollout_vllm
        for step in range(args.max_rollouts):
            chosen = prompts[step * 8:(step + 1) * 8]
            (out / f"rollout-{step + 1}-prompts.json").write_text(json.dumps(chosen))
            sync_training_devices()
            for dev in [0, 1]:
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.monotonic()
            metrics = trainer.train_rollout(chosen)
            sync_training_devices()
            training_sec = time.monotonic() - t0
            adapter_path = adapter_root / f"step-{step + 1}"
            ts = time.monotonic()
            trainer.actor.save_pretrained(adapter_path)
            adapter_id = step + 2
            save_sec = time.monotonic() - ts
            record = {**metrics, "method": cfg.method,
                      "profile_training_sec": training_sec, "adapter_save_sec": save_sec,
                      "profile_rollout_with_save_sec": time.monotonic() - t0,
                      "generation": generation_stats,
                      "peak_allocated_gib": [torch.cuda.max_memory_allocated(i) / 2**30 for i in [0, 1]],
                      "allocated_gpu_hours_since_setup": (time.monotonic() - started) * 3 / 3600}
            records.append(record)
            with (out / "profile_metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if args.keep_adapters_every:
                current_step = step + 1
                for old in adapter_root.glob("step-*"):
                    try:
                        old_step = int(old.name.split("-", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    if old_step not in {0, current_step} and old_step % args.keep_adapters_every:
                        shutil.rmtree(old, ignore_errors=True)
        # Keep the Actor on its GPU; this also checks independent GPU residency.
        tp = time.monotonic()
        probe = request({"prompts": [trainer._render_chat_prompt(trainer.actor_tokenizer, prompts[0])],
                         "adapter": str(adapter_path), "adapter_id": adapter_id,
                         "probe": True, "max_tokens": 8})
        probe_sec = time.monotonic() - tp
        if not probe["rows"] or not probe["rows"][0]:
            raise RuntimeError("post-update LoRA probe generated no tokens")
        print(f"adapter_sync=OK server_pid={probe['server_pid']} adapter_id={adapter_id} probe_sec={probe_sec:.2f}", flush=True)
        tc = time.monotonic()
        trainer.save_checkpoint(trainer.rollout_index)
        checkpoint_sec = time.monotonic() - tc
        (out / "profile_summary.json").write_text(json.dumps({
            "method": cfg.method, "setup_sec": setup_sec, "vllm_startup_sec": startup_sec,
            "rollouts": records, "final_adapter_probe_sec": probe_sec,
            "checkpoint_sec": checkpoint_sec, "server_pid": server.pid,
            "measured_total_sec": time.monotonic() - started,
            "estimate_note": "First rollout may include kernel warmup; later rollouts include next adapter load in generation. Different prompt/response lengths affect comparison.",
        }, indent=2))
    finally:
        if server.poll() is None:
            try:
                request({"shutdown": True})
                server.wait(timeout=45)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        tmp.cleanup()
    if server.returncode != 0:
        raise RuntimeError(f"vLLM server shutdown failed: {server.returncode}")


if __name__ == "__main__":
    main()
