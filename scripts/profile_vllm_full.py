#!/usr/bin/env python3
"""Full GRPO/VPO rollouts with separate Actor/RM and vLLM GPU allocations.

Reports startup separately from repeated generation, training and LoRA saving.
The first batch matches the earlier one-rollout profile; later batches advance
through the same deterministic prompt split as the main trainer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import signal
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
from vpo_rm.length_reward_cli import add_length_reward_args, length_reward_config_kwargs
from vpo_rm.reward_inputs import REWARD_INPUT_PROTOCOL
from vpo_rm.token_policy import resolve_actor_tokenizer_source


def profile_source_manifest():
    sources = ["vpo_rm/trainer.py", "vpo_rm/integration.py", "vpo_rm/core.py",
               "vpo_rm/token_policy.py", "vpo_rm/alignment.py", "vpo_rm/data.py",
               "vpo_rm/reward.py", "vpo_rm/reward_inputs.py", "vpo_rm/model_identity.py",
               "vpo_rm/length_reward.py", "vpo_rm/rollout_selection.py",
               "vpo_rm/length_reward_cli.py", "scripts/profile_vllm_full.py",
               "scripts/vllm_generate_server.py", "scripts/corrected_rl_launcher.py",
               "scripts/eval_artifacts.py", "scripts/preflight_training.py",
               "scripts/preflight_quality.py", "scripts/check_ssh_capacity.py",
               "vpo_rm/policy_precision.py", "scripts/llama_rl_launcher.py",
               "scripts/check_llama_protocol.py"]
    return {"reward_input_protocol": REWARD_INPUT_PROTOCOL,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in sources}}


def write_profile_calibration_manifest(manifest_path: Path, calibration_path: Path,
                                       trainer_config: TrainerConfig) -> None:
    """Record the trainer's post-calibration config and exact calibration file."""
    manifest = json.loads(manifest_path.read_text())
    calibration_bytes = calibration_path.read_bytes()
    manifest["config"] = asdict(trainer_config)
    manifest["length_reward_calibration"] = {
        "data": json.loads(calibration_bytes),
        "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-14B")
    p.add_argument("--rm", default="models/Skywork-Reward-V2-Qwen3-8B")
    p.add_argument("--tokenizer", default="", help="Actor tokenizer; defaults to saved SFT tokenizer, then base")
    p.add_argument("--dataset-path", default="datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", choices=["grpo", "vpo_rm"], default="vpo_rm")
    p.add_argument("--credit-source", choices=["rm_gradient", "random_direction", "random_band", "shuffle", "norm_product"],
                   default="rm_gradient",
                   help="VPO credit: RM dot product, random, post-allocation shuffle, or gradient norm product")
    p.add_argument("--max-rollouts", type=int, default=3)
    p.add_argument("--max-response-tokens", type=int, default=2048)
    p.add_argument("--generation-microbatch", type=int, default=32)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=.45)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                   help="Dedicated generation GPUs after the Actor and RM GPUs")
    p.add_argument("--credit-microbatch-responses", type=int, default=0,
                   help="Credit-cache response microbatch; 0 retains the full-batch path")
    p.add_argument("--policy-head-dtype", choices=("native", "float32"), default="float32",
                   help="Output projection precision shared by HF and vLLM")
    p.add_argument("--keep-adapters-every", type=int, default=0,
                   help="Keep step-0, every Nth adapter, and the current adapter; 0 keeps all")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Full checkpoint interval; the final checkpoint is always saved")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="LoRA learning rate; p9c used the paper's full-FT 1e-6, ~100x too small")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Dimensionless softmax temperature over standardized credit (Plan B)")
    p.add_argument("--credit-lambda", type=float, default=2.0,
                   help="Credit band: adaptive tau caps any token weight at lambda x uniform")
    p.add_argument("--freeze-stop-tokens", action="store_true", default=False,
                   help="Pin stop-token (EOS/im_end) credit weight at exactly 1, removing the RM scoring-position gradient artifact")
    p.add_argument("--freeze-structural", action="store_true", default=False,
                   help="Additionally pin newline/whitespace tokens at weight 1")
    p.add_argument("--beta", type=float, default=0.01, help="KL coefficient")
    p.add_argument("--init-adapter", default="",
                   help="Shared SFT initialization for both arms (stage 0 output)")
    p.add_argument("--kl-reference", choices=["rollout", "init"], default="init",
                   help="KL anchored to the SFT init ('init') or per-step rollout policy")
    add_length_reward_args(p)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--policy-epochs", type=int, default=1)
    p.add_argument("--optimizer-minibatch-responses", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generation-seed", type=int, default=0,
                   help="Matches the earlier profile's vLLM default seed")
    args = p.parse_args(argv)
    if args.max_rollouts < 1:
        p.error("--max-rollouts must be positive")
    if args.keep_adapters_every < 0:
        p.error("--keep-adapters-every must be nonnegative")
    if args.checkpoint_interval < 1:
        p.error("--checkpoint-interval must be positive")
    if not math.isfinite(args.vllm_gpu_memory_utilization) or not 0 < args.vllm_gpu_memory_utilization <= 1:
        p.error("--vllm-gpu-memory-utilization must be in (0, 1]")
    if args.vllm_tensor_parallel_size < 1:
        p.error("--vllm-tensor-parallel-size must be positive")
    if args.credit_microbatch_responses < 0:
        p.error("--credit-microbatch-responses must be nonnegative")
    if args.generation_microbatch < 1:
        p.error("--generation-microbatch must be positive")
    return args


def require_physical_reward_microbatch(config):
    # RM batching changes BF16 arithmetic even for native HF scoring. The
    # calibration, preflight and formal profile all use the same single-row path.
    if type(config.microbatch_responses) is not int or config.microbatch_responses != 1:
        raise ValueError("Validated profile requires physical RM microbatch_responses=1")
    return config


def build_trainer_config(args, output_dir):
    config = TrainerConfig(
        model_name=args.model, reward_model_name=args.rm, output_dir=str(output_dir),
        tokenizer_name=args.tokenizer,
        method=args.method, credit_source=args.credit_source, rollout_iterations=args.max_rollouts,
        prompts_per_rollout=8, group_size=8,
        max_response_tokens=args.max_response_tokens,
        generation_microbatch_responses=args.generation_microbatch,
        credit_microbatch_responses=args.credit_microbatch_responses,
        microbatch_responses=1, seed=args.seed, smoke=False,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate, tau=args.tau,
        credit_lambda=args.credit_lambda, beta=args.beta,
        freeze_stop_tokens=args.freeze_stop_tokens,
        freeze_structural=args.freeze_structural,
        init_adapter=args.init_adapter, kl_reference=args.kl_reference,
        temperature=args.temperature, policy_epochs_per_rollout=args.policy_epochs,
        optimizer_minibatch_responses=args.optimizer_minibatch_responses,
        rollout_importance_correction=True,
        policy_head_dtype=args.policy_head_dtype,
        actor_device="cuda:0", reward_device="cuda:1",
        allocated_gpu_count=2 + args.vllm_tensor_parallel_size,
        **length_reward_config_kwargs(args),
    ).resolved()
    return require_physical_reward_microbatch(config)


def unpadded_prompt_token_ids(batch):
    """Serialize exactly the attended HF prompt IDs for generation RPCs."""
    return [ids[mask.bool()].tolist()
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]


def pack_vllm_rollout(result, batch, rendered, *, group_size, max_response_tokens,
                     pad_token_id, adapter_id):
    """Bind one RPC's chosen-token probabilities to its exact HF rollout rows."""
    if result.get("logprobs_mode") != "processed_logprobs":
        raise RuntimeError("vLLM rollout requires processed logprobs")
    if type(result.get("adapter_id")) is not int or result["adapter_id"] != adapter_id:
        raise RuntimeError("vLLM rollout adapter identity differs from the request")
    expected_prompts = unpadded_prompt_token_ids(batch)
    if len(rendered) != len(expected_prompts) or result.get("prompt_token_ids") != expected_prompts:
        raise RuntimeError("vLLM prompt token IDs differ from unpadded HF prompts")
    rows, finish_reasons = result.get("rows", []), result.get("finish_reasons", [])
    expected = len(rendered) * group_size
    if (not expected or len(rows) != expected or len(finish_reasons) != expected
            or any(not row for row in rows)):
        raise RuntimeError("vLLM returned missing or empty responses")
    lengths = [len(row) for row in rows]
    if max(lengths) > max_response_tokens:
        raise RuntimeError("vLLM exceeded response token limit")
    probabilities = result.get("selected_logprobs")
    if (not isinstance(probabilities, list) or len(probabilities) != expected
            or any(not isinstance(values, list) or len(values) != length
                   for values, length in zip(probabilities, lengths))):
        raise RuntimeError("vLLM selected logprobs must cover every response token")
    flat_probabilities = [value for values in probabilities for value in values]
    if any(type(value) not in (float, int) or not math.isfinite(value) or value > 0
           for value in flat_probabilities):
        raise RuntimeError("vLLM selected logprobs must be finite and nonpositive")
    device, width = batch["input_ids"].device, max(lengths)
    prompt_width = batch["input_ids"].shape[1]
    responses = torch.full((expected, width), pad_token_id, dtype=torch.long, device=device)
    rollout_logprobs = torch.zeros((expected, width), dtype=torch.float32, device=device)
    for i, (row, values) in enumerate(zip(rows, probabilities)):
        responses[i, :len(row)] = torch.tensor(row, device=device)
        rollout_logprobs[i, :len(row)] = torch.tensor(values, dtype=torch.float32, device=device)
    if not torch.isfinite(rollout_logprobs).all():
        raise RuntimeError("vLLM selected logprobs are not representable as finite float32")
    rmask = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    expanded_input = batch["input_ids"].repeat_interleave(group_size, dim=0)
    prompt_mask = batch["attention_mask"].repeat_interleave(group_size, dim=0)
    input_ids = torch.cat([expanded_input, responses], dim=1)
    full_mask = torch.cat([prompt_mask, rmask.to(prompt_mask.dtype)], dim=1)
    positions = torch.arange(prompt_width, input_ids.shape[1], device=device).expand(expected, -1)
    # Raw probability/prompt arrays belong to the returned rollout, never the
    # console metrics or a mutable "last RPC" cache used by later minibatches.
    summary = {key: value for key, value in result.items()
               if key not in {"rows", "selected_logprobs", "prompt_token_ids"}}
    summary.update(scope="last_generation_request", response_lengths=lengths,
                   mean_response_tokens=sum(lengths) / expected,
                   padded_response_width=width, prompt_width=prompt_width,
                   truncation_rate=sum(reason == "length" for reason in finish_reasons) / expected,
                   selected_logprobs_count=len(flat_probabilities),
                   mean_selected_logprob=sum(flat_probabilities) / len(flat_probabilities),
                   min_selected_logprob=min(flat_probabilities),
                   max_selected_logprob=max(flat_probabilities))
    rollout = (input_ids, full_mask, positions, responses, rmask,
               [text for text in rendered for _ in range(group_size)], finish_reasons, rollout_logprobs)
    return rollout, summary


def vllm_subprocess_environment(tensor_parallel_size, device_count, environ=None):
    """Map logical generation GPUs back to the parent's physical IDs or UUIDs."""
    expected = 2 + tensor_parallel_size
    if device_count != expected:
        raise RuntimeError(f"This profile requires exactly {expected} visible GPUs "
                           f"(Actor + RM + {tensor_parallel_size} vLLM); got {device_count}")
    env = dict(os.environ if environ is None else environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = [str(i) for i in range(device_count)] if visible is None else [
        value.strip() for value in visible.split(",")]
    if len(devices) != expected or any(not value for value in devices):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify every detected visible GPU")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[2:])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def vllm_server_command(args, socket_path):
    return [sys.executable, str(ROOT / "scripts/vllm_generate_server.py"),
            "--model", args.model, "--socket", str(socket_path),
            "--tokenizer", resolve_actor_tokenizer_source(args.model, args.init_adapter, args.tokenizer),
            "--max-num-seqs", str(args.generation_microbatch),
            "--seed", str(args.generation_seed),
            "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
            "--tensor-parallel-size", str(args.vllm_tensor_parallel_size),
            "--policy-head-dtype", args.policy_head_dtype]


def start_vllm_server(args, socket_path, env):
    # Workers inherit this dedicated group, so cleanup cannot signal the actor.
    return subprocess.Popen(vllm_server_command(args, socket_path), env=env,
                            start_new_session=True)


def shutdown_vllm_server(server, request):
    """Bound the shutdown request and reap the server plus any surviving workers."""
    try:
        if server.poll() is None:
            try:
                request({"shutdown": True}, timeout=5)
                server.wait(timeout=45)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                pass
    finally:
        # Even an already-exited parent can leave live vLLM worker descendants.
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if server.poll() is None:
                server.wait(timeout=10)


def _terminate_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def main():
    args = parse_args()
    gpu_count = 2 + args.vllm_tensor_parallel_size
    env = vllm_subprocess_environment(args.vllm_tensor_parallel_size, torch.cuda.device_count())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    started = time.monotonic()
    prompts, _, split = load_prompt_dataset("HuggingFaceH4/ultrafeedback_binarized", dataset_path=args.dataset_path,
                                          exclude_benchmarks=True)
    cfg = build_trainer_config(args, out)
    # Seed before PEFT initializes LoRA A matrices, not only in trainer.__init__.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trainer = VPOTrainer.from_pretrained(cfg)
    require_physical_reward_microbatch(trainer.cfg)
    prompts = trainer.filter_prompts(prompts)
    if len(prompts) < cfg.prompts_per_rollout * args.max_rollouts:
        raise RuntimeError("not enough prompts after filtering")
    split = dict(split, filtered_train_prompts=len(prompts),
                 dropped_train_prompts=trainer.filtered_prompt_count)
    trainer.data_split = split
    (out / "data_split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    manifest_path = out / "profile_manifest.json"
    manifest_path.write_text(json.dumps({
        "config": asdict(cfg), "generation_seed": args.generation_seed,
        "sampling": trainer.sampling_manifest(), "resume_supported": False,
        "gpu_hours_definition": "elapsed_since_trainer_initialization_times_allocated_gpu_count",
        "gpu_count": gpu_count, "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "vllm_device": "cuda:2", "job_id": os.environ.get("JOB_ID"),
        "vllm_devices": [f"cuda:{i}" for i in range(2, gpu_count)],
        "policy_head_dtype": cfg.policy_head_dtype,
        "rollout_importance_correction": cfg.rollout_importance_correction,
        "vllm_engine": {"tensor_parallel_size": args.vllm_tensor_parallel_size,
                        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                        "max_num_seqs": args.generation_microbatch, "max_model_len": 4096,
                        "policy_head_dtype": cfg.policy_head_dtype},
        **profile_source_manifest(),
    }, indent=2))
    print(f"PROFILE_MANIFEST path={manifest_path} "
          f"sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()} "
          f"reward_input_protocol={REWARD_INPUT_PROTOCOL}", flush=True)
    adapter_root = out / "vllm-adapters"
    adapter_root.mkdir()
    adapter_path = adapter_root / "step-0"
    adapter_id = 1
    trainer.actor.save_pretrained(adapter_path)
    setup_sec = time.monotonic() - started
    print(f"devices actor=cuda:0 rm=cuda:1 vllm={list(range(2, gpu_count))} method={cfg.method} "
          f"requests=64 max_tokens={cfg.max_response_tokens} setup_sec={setup_sec:.2f}", flush=True)
    tmp = tempfile.TemporaryDirectory(prefix="vpo-vllm-")
    socket_path = Path(tmp.name) / "server.sock"
    server_started = time.monotonic()
    server = start_vllm_server(args, socket_path, env)

    def request(payload, *, timeout=1800):
        if server.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {server.returncode}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
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
    generation_requests = []

    @torch.no_grad()
    def rollout_vllm(batch_prompts):
        nonlocal generation_stats
        trainer.actor.eval()
        batch, rendered = trainer._encode_prompts(batch_prompts)
        result = request({"prompts": rendered, "prompt_token_ids": unpadded_prompt_token_ids(batch),
                          "adapter": str(adapter_path),
                          "adapter_id": adapter_id, "max_tokens": cfg.max_response_tokens,
                          "group_size": cfg.group_size, "temperature": cfg.temperature,
                          "min_tokens": cfg.min_response_tokens, "top_p": cfg.top_p,
                          "top_k": cfg.top_k, "presence_penalty": 0.0,
                          "return_logprobs": True})
        rollout, generation_stats = pack_vllm_rollout(
            result, batch, rendered, group_size=cfg.group_size,
            max_response_tokens=cfg.max_response_tokens,
            pad_token_id=trainer.actor_tokenizer.pad_token_id, adapter_id=adapter_id)
        generation_requests.append(dict(generation_stats))
        sync_training_devices()
        return rollout

    records = []
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_on_signal)
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
        if cfg.length_reward_mode == "soft":
            trainer.prepare_length_reward(prompts)
            write_profile_calibration_manifest(
                manifest_path, out / "length_reward_calibration.json", trainer.cfg)
        for step in range(args.max_rollouts):
            generation_requests.clear()
            chosen = prompts[step * 8:(step + 1) * 8]
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
                      "generation_requests": list(generation_requests),
                      "peak_allocated_gib": [torch.cuda.max_memory_allocated(i) / 2**30 for i in [0, 1]],
                      "allocated_gpu_hours_since_setup": (time.monotonic() - started) * gpu_count / 3600}
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
        probe_batch, probe_rendered = trainer._encode_prompts(prompts[:1])
        probe = request({"prompts": probe_rendered,
                         "prompt_token_ids": unpadded_prompt_token_ids(probe_batch),
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
        try:
            shutdown_vllm_server(server, request)
        finally:
            tmp.cleanup()
            signal.signal(signal.SIGTERM, previous_sigterm)
    if server.returncode != 0:
        raise RuntimeError(f"vLLM server shutdown failed: {server.returncode}")


if __name__ == "__main__":
    main()
