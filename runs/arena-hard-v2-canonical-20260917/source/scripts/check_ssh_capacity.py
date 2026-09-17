#!/usr/bin/env python3
"""Two synthetic 64x2048 VPO capacity steps on two visible Actor/RM GPUs.

This loads the real Base-14B actor, Skywork-8B reward model and native-EOS SFT
adapter. It executes real RM input gradients, credit and two Adam updates, but
does not run generation or save model checkpoints. The fixed sigma0=1 is ONLY
for capacity testing: these synthetic rewards are not calibration or quality
results. Use a fresh, separate output directory and CUDA_VISIBLE_DEVICES with
exactly two GPUs. A separate vLLM startup/generation test is still required.
The second step tests memory with persistent Adam states; this remains a
two-step capacity check, not a guarantee for arbitrarily long training runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAPACITY_STEPS = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="New directory dedicated to this synthetic capacity test")
    return parser.parse_args(argv)


def build_config(output_dir):
    from vpo_rm.trainer import TrainerConfig

    return TrainerConfig(
        model_name=str(ROOT / "models/Qwen3-14B-Base"),
        reward_model_name=str(ROOT / "models/Skywork-Reward-V2-Qwen3-8B"),
        init_adapter=str(ROOT / "models/sft-native-eos-clean2k5e2"),
        output_dir=str(output_dir), actor_device="cuda:0", reward_device="cuda:1",
        allocated_gpu_count=2, method="vpo_rm", credit_lambda=4.,
        freeze_stop_tokens=True, freeze_structural=True,
        prompts_per_rollout=8, group_size=8, max_prompt_tokens=2048,
        max_response_tokens=2048, microbatch_responses=1,
        credit_microbatch_responses=1, optimizer_minibatch_responses=64,
        rollout_iterations=CAPACITY_STEPS, policy_epochs_per_rollout=1,
        learning_rate=5e-5, beta=.03, tau=1., temperature=1.,
        kl_reference="init", length_reward_mode="soft", length_reward_sigma0=1.,
        min_response_tokens=0, checkpoint_interval=100,
    ).resolved()


def check_gpu_count(device_count):
    if device_count != 2:
        raise RuntimeError("Capacity testing requires exactly two visible GPUs (Actor and RM); "
                           "set CUDA_VISIBLE_DEVICES before running this script")


def create_output_dir(path):
    path = Path(path).expanduser().resolve()
    # Even an existing empty directory is rejected, so reruns cannot silently
    # mix a failed capacity attempt with another run.
    path.mkdir(parents=True, exist_ok=False)
    return path


def prompt_lengths(actor_tokenizer, reward_tokenizer, prompt):
    """Measure Actor generation input and canonical empty-response RM input."""
    from vpo_rm.trainer import VPOTrainer
    from vpo_rm.reward_inputs import canonical_reward_input

    text = VPOTrainer._render_chat_prompt(actor_tokenizer, prompt, tokenize=False)
    actor_ids = actor_tokenizer(text, add_special_tokens=True)["input_ids"]
    reward_ids = canonical_reward_input(reward_tokenizer, prompt, "")
    return len(actor_ids), len(reward_ids)


def make_synthetic_prompts(actor_tokenizer, reward_tokenizer, *, count=8,
                           max_prompt_tokens=2048):
    """Fill real user content to the largest joint template length that fits.

    Whole words separated by spaces avoid an artificial token-ID prefix. A
    binary search finds the longest repeated-word count under both limits;
    actual lengths are measured again and included in the capacity report.
    """
    if count < 1 or max_prompt_tokens < 1:
        raise ValueError("Prompt count and prompt length limit must be positive")
    prompts = []
    for index in range(count):
        base = f"Capacity check prompt {index}. Repeat these words:"

        def candidate(repetitions):
            return base + " alpha" * repetitions

        def fits(repetitions):
            return max(prompt_lengths(actor_tokenizer, reward_tokenizer,
                                      candidate(repetitions))) <= max_prompt_tokens

        if not fits(0):
            raise ValueError("The synthetic prompt template exceeds the prompt length limit")
        lo, hi = 0, max_prompt_tokens
        while fits(hi):
            lo, hi = hi, hi * 2
            if hi > max_prompt_tokens * 64:
                raise ValueError("Repeated words did not increase the tokenizer's prompt length")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid
        prompt = candidate(lo)
        if max(prompt_lengths(actor_tokenizer, reward_tokenizer, prompt)) > max_prompt_tokens:
            raise ValueError("Synthetic prompt exceeds the prompt length limit")
        prompts.append(prompt)
    return prompts


def _ordinary_response_tokens(tokenizer, output_mask, stop_token_ids):
    excluded = set(tokenizer.all_special_ids) | set(stop_token_ids)
    token_ids = []
    for word in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "omega"):
        for token_id in tokenizer(" " + word, add_special_tokens=False)["input_ids"]:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if (token_id not in excluded and token_id not in token_ids
                    and 0 <= token_id < output_mask.numel() and bool(output_mask[token_id])
                    and decoded.strip().isalpha() and "\n" not in decoded):
                token_ids.append(token_id)
    if len(token_ids) < 8:
        raise ValueError("Capacity responses require at least eight supported ordinary word tokens")
    return token_ids[:8]


def build_synthetic_rollout(trainer, prompts):
    """Preserve normal Actor/RM chat preparation while supplying fixed responses."""
    import torch
    from vpo_rm.alignment import check_response_tokens
    from vpo_rm.reward_inputs import canonical_reward_input

    lengths = [prompt_lengths(trainer.actor_tokenizer, trainer.reward_tokenizer, prompt)
               for prompt in prompts]
    if not lengths or max(max(pair) for pair in lengths) > trainer.cfg.max_prompt_tokens:
        raise ValueError("Synthetic prompt exceeds the prompt length limit")
    batch, rendered = trainer._encode_prompts(prompts)
    if batch["attention_mask"].sum(-1).tolist() != [pair[0] for pair in lengths]:
        raise ValueError("Actor encoding changed the measured untruncated prompt lengths")
    native_eos = trainer.actor_tokenizer.eos_token_id
    if (not isinstance(native_eos, int) or native_eos not in trainer.stop_token_ids
            or not bool(trainer.output_mask[native_eos])):
        raise ValueError("Actor native EOS must be a supported stop token")
    word_ids = _ordinary_response_tokens(trainer.actor_tokenizer, trainer.output_mask,
                                        trainer.stop_token_ids)
    group_size, response_width = trainer.cfg.group_size, trainer.cfg.max_response_tokens
    batch_size = len(prompts) * group_size
    device = trainer.actor_device
    words = torch.tensor(word_ids, device=device)
    offsets = torch.arange(batch_size, device=device)[:, None]
    body = words[(torch.arange(response_width - 1, device=device)[None, :] + offsets) % len(words)]
    eos = torch.full((batch_size, 1), native_eos, dtype=torch.long, device=device)
    responses = torch.cat((body, eos), dim=1)
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    prefixes = batch["input_ids"].repeat_interleave(group_size, dim=0)
    prompt_mask = batch["attention_mask"].repeat_interleave(group_size, dim=0)
    input_ids = torch.cat((prefixes, responses), dim=1)
    full_mask = torch.cat((prompt_mask, response_mask.to(prompt_mask.dtype)), dim=1)
    prompt_width = prefixes.shape[1]
    positions = torch.arange(prompt_width, input_ids.shape[1], device=device).expand(batch_size, -1)
    check_response_tokens(input_ids, full_mask, positions, responses, response_mask)
    trainer.actor.eval()
    rollout = (input_ids, full_mask, positions, responses, response_mask,
               [text for text in rendered for _ in range(group_size)], ["stop"] * batch_size)
    # Canonical serialization removes Actor special tokens, retokenizes the
    # visible answer, and appends RM template suffixes. Its actual sequence
    # length cannot be inferred by adding Actor response width to a prefix.
    response_texts = [trainer.actor_tokenizer.decode(row, skip_special_tokens=True,
                       clean_up_tokenization_spaces=False) for row in responses.detach().cpu().tolist()]
    reward_lengths = [len(canonical_reward_input(trainer.reward_tokenizer,
                       prompts[index // group_size], text))
                      for index, text in enumerate(response_texts)]
    shape = {
        "prompt_count": len(prompts), "group_size": group_size,
        "response_shape": list(responses.shape), "vocab_size": trainer.output_mask.numel(),
        "actor_input_shape": list(input_ids.shape),
        "requested_prompt_limit": trainer.cfg.max_prompt_tokens,
        "actor_prompt_lengths": [pair[0] for pair in lengths],
        "reward_prompt_lengths": [pair[1] for pair in lengths],
        "actor_padded_prompt_width": prompt_width,
        "reward_sequence_lengths": reward_lengths,
        "reward_max_sequence_length": max(reward_lengths),
        "native_eos_id": native_eos, "ordinary_response_token_ids": word_ids,
        "finish_reason": "stop", "valid_response_tokens": int(response_mask.sum()),
    }
    return rollout, shape


def initial_report(config):
    return {
        "status": "started", "synthetic_capacity_test_only": True,
        "sigma0_testonly": 1., "valid_for_reward_calibration": False,
        "valid_for_quality_comparison": False, "vllm_tested": False,
        "allocated_gpu_count": 2, "config": asdict(config),
        "steps": CAPACITY_STEPS, "completed_steps": 0, "step_metrics": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "memory_scope": "PyTorch peaks from model loading through two worst-shape optimizer steps, "
                        "including persistent Adam state in step two; not total process/NVML "
                        "usage, not vLLM generation, and not an unlimited-training guarantee",
        "expected_response_shape": [64, 2048],
    }


def _memory_report(torch):
    cards = []
    for device in range(2):
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        cards.append({"device": f"cuda:{device}", "name": properties.name,
                      "total_bytes": total, "free_bytes_after_test": free,
                      "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                      "peak_allocated_gib": allocated / 2**30,
                      "peak_reserved_gib": reserved / 2**30})
    return cards


def run_capacity_steps(trainer, prompts, rollout, report):
    def fixed_rollout(requested_prompts):
        if list(requested_prompts) != prompts:
            raise RuntimeError("Synthetic complete groups unexpectedly requested resampling")
        trainer.actor.eval()
        return rollout

    trainer.rollout = fixed_rollout
    batch_size = len(prompts) * trainer.cfg.group_size
    expected_tokens = batch_size * trainer.cfg.max_response_tokens
    # Fixed sigma0 bypasses calibration generation. Each iteration still uses
    # real RM gradients, new old-policy logits, VPO credit and an Adam update.
    # No model reload or peak reset occurs between steps, retaining Adam state.
    for index in range(CAPACITY_STEPS):
        metrics = trainer.train_rollout(prompts)
        report["step_metrics"].append(metrics)
        if (metrics["optimizer_steps"] != 1 or metrics["response_tokens"] != expected_tokens
                or metrics["reward_count"] != batch_size or metrics["skipped_rollout"]
                or metrics["resampled_groups"] or metrics["skipped_groups"]):
            raise RuntimeError("Capacity test did not execute the complete synthetic update")
        report["completed_steps"] = index + 1


def main(argv=None):
    args = parse_args(argv)
    import torch
    from vpo_rm.trainer import VPOTrainer

    check_gpu_count(torch.cuda.device_count())
    output = create_output_dir(args.output_dir)
    config = build_config(output)
    report = initial_report(config)
    report_path = output / "capacity-report.json"
    report_path.write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    try:
        for device in range(2):
            torch.cuda.reset_peak_memory_stats(device)
        trainer = VPOTrainer.from_pretrained(config)
        prompts = make_synthetic_prompts(trainer.actor_tokenizer, trainer.reward_tokenizer,
                                         count=8, max_prompt_tokens=2048)
        rollout, shape = build_synthetic_rollout(trainer, prompts)
        report["shape"] = shape
        (output / "synthetic-prompts.json").write_text(json.dumps(prompts, indent=2))

        run_capacity_steps(trainer, prompts, rollout, report)
        for device in range(2):
            torch.cuda.synchronize(device)
        report.update(status="passed")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_sec"] = time.monotonic() - started
        try:
            report["gpus"] = _memory_report(torch)
        except Exception as error:
            report["memory_report_error"] = f"{type(error).__name__}: {error}"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"capacity_report": str(report_path), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
