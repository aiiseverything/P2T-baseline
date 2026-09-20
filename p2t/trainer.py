"""P2T training loop: rollouts, Eq. (1)-(5), and a GRPO update that never changes.

Credit assignment is the only thing this file does differently from the parent
project's trainer.  Everything around it -- prompt rendering, the vLLM sampling
protocol, the reward-model input mapping, the soft length window, the degeneracy
guard, the group standardisation, the clipped surrogate, the KL-to-init term, the
optimizer settings and the metric names -- is mirrored from ``vpo_rm`` so a P2T
run and a VPO-RM run differ in exactly one thing.

Where P2T is *cheaper* than VPO-RM: it needs no policy logits to build credit, so
the ``[B, T, V]`` response-logits tensor never exists here.  The actor is only
forwarded for the importance ratio and the KL term, one response microbatch at a
time.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from .attribution import P2T_APPROXIMATION, null_token_attribution
from .autopush import AutoPusher
from .data import load_prompt_dataset, normalize_prompt, split_prompts
from .length_reward import (calibrate_reward_scale, guard_degenerate_rewards,
                            response_degeneracy, soft_length_penalties)
from .loss import grpo_policy_loss, kl_from_logp
from .mapping import REWARD_INPUT_PROTOCOL
from .policy import (encode_prompts, render_chat_prompt, response_logits,
                     rollout_logp_microbatch, selected_logp_from_logits,
                     sampling_logits, stop_token_ids_for)
from .reward import P2T_ALPHA_SHORT_COT, P2T_OMEGA, group_advantages, p2t_credit
from .rm import LastTokenReward, build_rm_batch, score_responses
from .tokens import (check_tokenizer_identity, get_special_token_ids,
                     load_actor_tokenizer, shared_output_mask)
from .vllm import GenerationServer, pack_rollout, unpadded_prompt_token_ids


@dataclass
class TrainerConfig:
    model_name: str = "models/Qwen3-14B-Base"
    reward_model_name: str = "models/Skywork-Reward-V2-Qwen3-8B"
    tokenizer_name: str = ""
    init_adapter: str = ""
    output_dir: str = "runs/p2t"
    report_dir: str = ""
    run_name: str = "p2t"
    actor_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    vllm_gpus: list[str] = field(default_factory=lambda: ["2", "3"])
    vllm_gpu_memory_utilization: float = .85
    vllm_tensor_parallel_size: int = 2
    vllm_max_num_seqs: int = 32
    seed: int = 42
    generation_seed: int = 0
    group_size: int = 8
    prompts_per_rollout: int = 8
    rollout_iterations: int = 10
    optimizer_minibatch_responses: int = 64
    microbatch_responses: int = 1
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_response_tokens: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = .01
    max_grad_norm: float = 1.0
    clip_eps: float = .2
    beta: float = .03
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True
    # --- P2T, paper constants -------------------------------------------------
    omega: float = P2T_OMEGA
    alpha: float = P2T_ALPHA_SHORT_COT
    null_token_id: int | None = None  # None -> the reward model's pad token
    # --- length reward, mirrored from the project ----------------------------
    length_threshold_short: int = 8
    length_threshold_long: int = 1024
    short_penalty_strength: float = .5
    long_penalty_strength: float = 2.
    advantage_std_floor_fraction: float = .5
    sigma0: float | None = None
    calibration_prompts: int = 0  # 0 -> derive from the run config; smoke uses 8
    degenerate_newline_run: int = 32
    degenerate_penalty: float = 1.0
    # --- bookkeeping ----------------------------------------------------------
    checkpoint_interval: int = 0  # 0 -> only the final adapter
    keep_adapters_every: int = 1
    token_chunk_size: int = 128
    validation_size: int = 2000
    fit_prompt_filter: bool = True
    push_every: int = 0  # 0 disables autopush
    push_remote: str = "origin"
    push_branch: str = "p2t-baseline"
    dry_run: bool = False

    def resolved(self) -> "TrainerConfig":
        c = TrainerConfig(**asdict(self))
        for name in ("group_size", "prompts_per_rollout", "rollout_iterations",
                     "optimizer_minibatch_responses", "microbatch_responses",
                     "max_prompt_tokens", "max_response_tokens", "vllm_max_num_seqs",
                     "vllm_tensor_parallel_size", "lora_r", "lora_alpha",
                     "token_chunk_size", "degenerate_newline_run"):
            value = getattr(c, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm", "weight_decay", "clip_eps",
                     "short_penalty_strength", "long_penalty_strength",
                     "advantage_std_floor_fraction", "degenerate_penalty"):
            value = getattr(c, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("learning_rate", "max_grad_norm"):
            if getattr(c, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(c.beta) or c.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if c.microbatch_responses != 1:
            raise ValueError("The physical reward/actor microbatch must be one")
        if c.optimizer_minibatch_responses % c.microbatch_responses:
            raise ValueError("optimizer_minibatch_responses must be a multiple of the microbatch")
        if not 0 < c.clip_eps < 1:
            raise ValueError("clip_eps must lie in (0,1)")
        if c.top_p != 1.0 or c.top_k != 0:
            raise ValueError("training supports top_p=1 and top_k=0 only")
        if not math.isfinite(c.temperature) or not .01 <= c.temperature <= 2.0:
            raise ValueError("temperature must be in [0.01, 2] to avoid backend clamping")
        if c.lora_dropout != 0:
            raise ValueError("training requires zero dropout for matching rollout probabilities")
        if c.init_adapter and not c.lora:
            raise ValueError("init_adapter requires lora=True")
        # P2T-specific validation: the paper has no allocator, so VPO-only knobs
        # must be rejected rather than silently ignored.
        if not math.isfinite(c.omega) or c.omega < 0:
            raise ValueError("omega must be finite and nonnegative")
        if not math.isfinite(c.alpha) or c.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if c.null_token_id is not None and (
                isinstance(c.null_token_id, bool) or not isinstance(c.null_token_id, int)
                or c.null_token_id < 0):
            raise ValueError("null_token_id must be a nonnegative Python int or null")
        if not 0 < c.length_threshold_short <= c.length_threshold_long < c.max_response_tokens:
            raise ValueError("require 0 < short <= long threshold < max_response_tokens")
        if c.sigma0 is not None and (not math.isfinite(c.sigma0) or c.sigma0 <= 0):
            raise ValueError("sigma0 must be finite and positive")
        if not 0 <= c.min_response_tokens <= c.max_response_tokens:
            raise ValueError("require 0 <= min_response_tokens <= max_response_tokens")
        if c.push_every < 0:
            raise ValueError("push_every must be nonnegative")
        if len(c.vllm_gpus) != c.vllm_tensor_parallel_size:
            raise ValueError("vllm_gpus must list exactly tensor_parallel_size devices")
        return c


class P2TTrainer:
    """Actor on one device, frozen reward model on another, generation on its own."""

    def __init__(self, actor, actor_tokenizer, reward, reward_tokenizer, config: TrainerConfig,
                 generation: GenerationServer | None = None):
        self.cfg = config.resolved()
        self.actor = actor
        self.actor_tokenizer = actor_tokenizer
        self.reward = reward
        self.reward_tokenizer = reward_tokenizer
        self.generation = generation
        self.actor_device = torch.device(self.cfg.actor_device)
        self.reward_device = torch.device(self.cfg.reward_device)
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(self.cfg.report_dir or (self.output_dir / "report"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.report_dir / "metrics.jsonl"
        self.adapter_root = self.output_dir / "vllm-adapters"
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        self.null_token_id = (self.cfg.null_token_id if self.cfg.null_token_id is not None
                              else int(self.reward_tokenizer.pad_token_id))
        self.stop_token_ids = stop_token_ids_for(self.actor_tokenizer)
        self.output_mask = shared_output_mask(
            self.actor_tokenizer, self.actor.get_output_embeddings().weight.shape[0],
            self.actor_device)
        self.vocab_size = int(self.actor.get_output_embeddings().weight.shape[0])
        self.special_token_ids = torch.tensor(get_special_token_ids(self.actor_tokenizer),
                                              device=self.actor_device)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.actor.parameters() if p.requires_grad),
            lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        self.rollout_index = 0
        self.total_tokens = 0
        self.adapter_id = 0
        self.sigma0 = self.cfg.sigma0
        self.started = time.monotonic()
        self.pusher = AutoPusher(enabled=bool(self.cfg.push_every), every=self.cfg.push_every,
                                 remote=self.cfg.push_remote, branch=self.cfg.push_branch,
                                 report_dir=self.report_dir, repo_root=Path(__file__).resolve().parents[1])
        self._last_rollout_logprobs = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, config: TrainerConfig, *, start_generation: bool = True) -> "P2TTrainer":
        c = config.resolved()
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        random.seed(c.seed)
        torch.manual_seed(c.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c.seed)
        dtype = torch.bfloat16 if c.actor_device.startswith("cuda") else torch.float32
        actor_tokenizer = load_actor_tokenizer(c.model_name, c.init_adapter, c.tokenizer_name)
        reward_tokenizer = AutoTokenizer.from_pretrained(
            c.reward_model_name, padding_side="right", trust_remote_code=True)
        from .tokens import configure_model_padding
        configure_model_padding(reward_tokenizer, fallback_token=actor_tokenizer.pad_token)
        actor = AutoModelForCausalLM.from_pretrained(c.model_name, torch_dtype=dtype,
                                                     trust_remote_code=True)
        rm_base = AutoModelForSequenceClassification.from_pretrained(
            c.reward_model_name, torch_dtype=dtype, trust_remote_code=True)
        actor.config.pad_token_id = actor_tokenizer.pad_token_id
        rm_base.config.pad_token_id = reward_tokenizer.pad_token_id
        backbone = getattr(rm_base, "base_model", None) or getattr(rm_base, "model", None)
        if backbone is None or not hasattr(rm_base, "score"):
            raise ValueError("Reward checkpoint must expose a decoder backbone and scalar score head")
        reward = LastTokenReward(backbone, rm_base.score)
        check_tokenizer_identity(actor_tokenizer, reward_tokenizer,
                                 actor.get_output_embeddings().weight.shape[0],
                                 reward.get_input_embeddings().weight.shape[0])
        if c.lora:
            from peft import LoraConfig, PeftModel, get_peft_model
            if c.init_adapter:
                actor = PeftModel.from_pretrained(actor, c.init_adapter, is_trainable=True)
                # The SFT checkpoint is also the KL reference; mount it a second
                # time as a frozen adapter instead of copying the backbone.
                actor.load_adapter(c.init_adapter, adapter_name="ref", is_trainable=False)
                actor.set_adapter("default")
            else:
                actor = get_peft_model(actor, LoraConfig(
                    r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                    bias="none", task_type="CAUSAL_LM", target_modules=c.target_modules))
        if c.gradient_checkpointing:
            try:
                actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                actor.gradient_checkpointing_enable()
        generation = None
        if start_generation:
            generation = GenerationServer(
                model=c.model_name, tokenizer_source=c.tokenizer_name or c.model_name,
                socket_path=Path(c.output_dir) / "vllm.sock", gpus=c.vllm_gpus,
                max_num_seqs=c.vllm_max_num_seqs, seed=c.generation_seed,
                gpu_memory_utilization=c.vllm_gpu_memory_utilization,
                tensor_parallel_size=c.vllm_tensor_parallel_size,
                log_path=Path(c.output_dir) / "vllm_server.log")
        trainer = cls(actor, actor_tokenizer, reward, reward_tokenizer, c, generation)
        trainer.actor.to(trainer.actor_device)
        trainer.reward.to(trainer.reward_device).eval()
        for parameter in trainer.reward.parameters():
            parameter.requires_grad_(False)
        return trainer

    # ------------------------------------------------------------- generation
    def save_adapter(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(path)
        return path

    @torch.no_grad()
    def rollout(self, prompts):
        """One prompt group per prompt, sampled by the resident vLLM server."""
        if self.generation is None:
            raise RuntimeError("No generation server attached")
        self.actor.eval()
        batch, rendered = encode_prompts(self.actor_tokenizer, prompts, self.actor_device,
                                         self.cfg.max_prompt_tokens)
        adapter_path = self.adapter_root / "live"
        self.save_adapter(adapter_path)
        self.adapter_id += 1
        result = self.generation.request({
            "prompts": rendered,
            "prompt_token_ids": unpadded_prompt_token_ids(batch),
            "adapter": str(adapter_path), "adapter_id": self.adapter_id,
            "max_tokens": self.cfg.max_response_tokens, "group_size": self.cfg.group_size,
            "temperature": self.cfg.temperature, "top_p": self.cfg.top_p, "top_k": self.cfg.top_k,
            "min_tokens": self.cfg.min_response_tokens, "presence_penalty": 0.0,
            "return_logprobs": True})
        rollout, summary = pack_rollout(
            result, batch, rendered, group_size=self.cfg.group_size,
            max_response_tokens=self.cfg.max_response_tokens,
            pad_token_id=self.actor_tokenizer.pad_token_id, adapter_id=self.adapter_id)
        return rollout, summary

    # ------------------------------------------------------------- calibration
    @torch.no_grad()
    def prepare_sigma0(self, prompts) -> float:
        """Freeze the initial-policy reward scale before any optimizer update.

        The project's protocol calibrates on 128 prompts and shares one sigma0
        across every arm, so the length window is identical everywhere.  A short
        smoke may use fewer prompts, which must never be presented as the formal
        calibration.
        """
        if self.sigma0 is not None:
            return self.sigma0
        if self.rollout_index:
            raise RuntimeError("sigma0 calibration must precede all training updates")
        count = self.cfg.calibration_prompts or 8
        unique = list(dict.fromkeys(prompts))
        if len(unique) < count:
            raise ValueError(f"Calibration requires {count} distinct prompts; got {len(unique)}")
        chosen = random.Random(self.cfg.seed).sample(unique, count)
        scores, groups, valid_rows = [], [], []
        for start in range(0, count, self.cfg.prompts_per_rollout):
            batch = chosen[start:start + self.cfg.prompts_per_rollout]
            rollout, _ = self.rollout(batch)
            input_ids, full_mask, positions, responses, rmask, _, reasons, _ = rollout
            rewards, _ = self._raw_rewards(batch, input_ids, responses, rmask)
            empty, repeated = self._degeneracy(responses, rmask)
            valid = torch.tensor([reason == "stop" for reason in reasons],
                                 device=rewards.device) & ~(empty | repeated)
            for j, (reward, ok) in enumerate(zip(rewards.tolist(), valid.tolist())):
                scores.append(reward)
                groups.append(start + j // self.cfg.group_size)
                valid_rows.append(ok)
        sigma0 = calibrate_reward_scale(torch.tensor(scores), torch.tensor(groups),
                                        torch.tensor(valid_rows, dtype=torch.bool))
        self.sigma0 = sigma0
        record = {"mode": "p2t_soft_length", "sigma0": sigma0,
                  "calibration_prompt_count": count,
                  "formal_calibration": count >= 128,
                  "group_size": self.cfg.group_size, "seed": self.cfg.seed,
                  "model": self.cfg.model_name, "reward_model": self.cfg.reward_model_name,
                  "prompt_sha256": [__import__("hashlib").sha256(p.encode()).hexdigest() for p in chosen]}
        (self.output_dir / "length_reward_calibration.json").write_text(
            json.dumps(record, indent=2, allow_nan=False))
        return sigma0

    # ------------------------------------------------------------- reward side
    def _degeneracy(self, responses, rmask):
        stop = set(self.stop_token_ids)
        texts = [self.actor_tokenizer.decode(
            [x for x in row[valid].detach().cpu().tolist() if x not in stop],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for row, valid in zip(responses, rmask)]
        empty, repeated = response_degeneracy(texts, self.cfg.degenerate_newline_run)
        return (torch.tensor(empty, device=self.reward_device, dtype=torch.bool),
                torch.tensor(repeated, device=self.reward_device, dtype=torch.bool))

    def _raw_rewards(self, prompts, input_ids, responses, rmask):
        """Forward-only RM scoring, used by the calibration pass."""
        rows, mapped, _, stats = build_rm_batch(
            self.actor_tokenizer, self.reward_tokenizer,
            [p for p in prompts for _ in range(self.cfg.group_size)],
            responses, rmask, self.cfg.max_prompt_tokens, self.cfg.max_response_tokens)
        rewards, _ = score_responses(self.reward, self.reward_tokenizer, rows, mapped,
                                     responses, rmask, device=self.reward_device, microbatch=1)
        return rewards.to(self.reward_device), stats

    def score_and_attribute(self, prompts, responses, rmask):
        """Raw RM scores plus Eq. (2) attributions for every response.

        ``prompts`` is one entry per *prompt group*; the RM rows are per response,
        so each prompt is repeated ``group_size`` times -- the same expansion the
        parent project performs before scoring.
        """
        group_prompts = [prompt for prompt in prompts for _ in range(self.cfg.group_size)]
        rows, mapped, fixed_weight_mask, stats = build_rm_batch(
            self.actor_tokenizer, self.reward_tokenizer, group_prompts, responses, rmask,
            self.cfg.max_prompt_tokens, self.cfg.max_response_tokens)
        rewards, grads = score_responses(self.reward, self.reward_tokenizer, rows, mapped,
                                         responses, rmask, device=self.reward_device,
                                         microbatch=self.cfg.microbatch_responses)
        rewards = rewards.to(self.reward_device)
        grads = grads.to(self.reward_device)
        attribution = null_token_attribution(
            grads, responses.to(self.reward_device),
            self.reward.get_input_embeddings().weight, self.null_token_id,
            rmask.to(self.reward_device), token_chunk_size=self.cfg.token_chunk_size)
        stats = dict(stats)
        stats["p2t_unmapped_share_mean"] = float(
            fixed_weight_mask.sum() / rmask.sum().clamp_min(1))
        stats["p2t_null_token_id"] = self.null_token_id
        del grads
        return rewards, attribution, fixed_weight_mask.to(self.actor_device), stats

    # ------------------------------------------------------------ training step
    def train_rollout(self, prompts) -> dict:
        timings = {}

        def phase(name, start):
            for device in (self.actor_device, self.reward_device):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            timings[name] = time.monotonic() - start
            return time.monotonic()

        started = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        if self.actor_device.type == "cuda":
            torch.cuda.empty_cache()
        if self.sigma0 is None:
            raise RuntimeError("Call prepare_sigma0(training_prompts) before training")

        clock = phase("generation_sec", started)
        rollout, generation_summary = self.rollout(prompts)
        input_ids, full_mask, positions, responses, rmask, _, finish_reasons, rollout_logprobs = rollout
        self._validate_rollout(rollout, len(prompts))

        clock = phase("reward_model_gradient_sec", clock)
        raw_rewards, attribution, _, alignment = self.score_and_attribute(prompts, responses, rmask)
        batch = raw_rewards.shape[0]
        group_ids = torch.arange(batch, device=self.reward_device) // self.cfg.group_size
        lengths = rmask.sum(-1).to(self.reward_device)
        truncated = torch.tensor([reason == "length" for reason in finish_reasons],
                                 device=self.reward_device, dtype=torch.bool)

        short_penalty, long_penalty = soft_length_penalties(
            lengths, self.sigma0,
            short_threshold=self.cfg.length_threshold_short,
            long_threshold=self.cfg.length_threshold_long,
            max_length=self.cfg.max_response_tokens,
            short_strength=self.cfg.short_penalty_strength,
            long_strength=self.cfg.long_penalty_strength)
        empty, repeated = self._degeneracy(responses, rmask)
        shaped = raw_rewards - short_penalty - long_penalty
        flags = empty | repeated
        shaped = shaped.clone()
        for group in group_ids.unique():
            members = group_ids == group
            flagged = members & flags
            if bool(flagged.any()):
                good = members & ~flags
                if not bool(good.any()):
                    raise RuntimeError("An entirely degenerate group reached the update")
                shaped[flagged] = shaped[good].min() - self.cfg.degenerate_penalty
        shaped, n_degenerate = guard_degenerate_rewards(
            shaped, lengths, group_ids, 1, self.cfg.degenerate_penalty)
        advantages, scales = group_advantages(
            shaped, group_ids,
            std_floor=self.cfg.advantage_std_floor_fraction * self.sigma0)

        clock = phase("credit_sec", clock)
        credit = p2t_credit(raw_rewards, attribution, advantages, rmask.to(self.reward_device),
                            omega=self.cfg.omega, alpha=self.cfg.alpha)
        # Only the three [B, T] credit fields cross to the actor device; the
        # attribution and the RM embedding stay on the reward device.
        credit = replace(credit, advantage=credit.advantage.to(self.actor_device),
                         direction=credit.direction.to(self.actor_device),
                         weight=credit.weight.to(self.actor_device), tau_used=None)

        clock = phase("actor_old_logp_sec", clock)
        entropy = torch.zeros(responses.shape, dtype=torch.float32, device=self.actor_device)
        old_logp = rollout_logp_microbatch(
            self.actor, input_ids, full_mask, positions, responses, rmask, self.output_mask,
            temperature=self.cfg.temperature, microbatch=self.cfg.microbatch_responses,
            min_response_tokens=self.cfg.min_response_tokens,
            stop_token_ids=self.stop_token_ids, token_chunk_size=self.cfg.token_chunk_size,
            entropy_out=entropy)
        clock = phase("ref_logp_sec", clock)
        reference_adapter = "ref" if self.cfg.init_adapter else "base"
        ref_logp = rollout_logp_microbatch(
            self.actor, input_ids, full_mask, positions, responses, rmask, self.output_mask,
            temperature=self.cfg.temperature, microbatch=self.cfg.microbatch_responses,
            adapter=reference_adapter, min_response_tokens=self.cfg.min_response_tokens,
            stop_token_ids=self.stop_token_ids, token_chunk_size=self.cfg.token_chunk_size)

        # Sampler-to-HF correction: the actor re-forward is the clipping anchor,
        # so the ratio must carry the sampler's own chosen-token probability.
        importance = (old_logp.detach() - rollout_logprobs).exp().masked_fill(~rmask, 1)
        if not (torch.isfinite(importance) & (importance > 0)).all():
            raise ValueError("Rollout importance weights must be positive and finite")

        clock = phase("actor_update_sec", clock)
        self.actor.train()
        minibatch = self.cfg.optimizer_minibatch_responses
        micro = max(1, self.cfg.microbatch_responses)
        loss_value, optimizer_steps, grad_norm = 0.0, 0, 0.0
        for start in range(0, batch, minibatch):
            end = min(batch, start + minibatch)
            self.optimizer.zero_grad(set_to_none=True)
            for lo in range(start, end, micro):
                hi = min(end, lo + micro)
                sl = slice(lo, hi)
                logits = response_logits(self.actor, input_ids[sl], full_mask[sl],
                                         positions[sl], rmask[sl], output_mask=self.output_mask)
                logits = sampling_logits(logits, min_response_tokens=self.cfg.min_response_tokens,
                                         stop_token_ids=self.stop_token_ids, inplace=True)
                new_logp = selected_logp_from_logits(
                    logits, responses[sl], rmask[sl],
                    policy_temperature=self.cfg.temperature,
                    token_chunk_size=self.cfg.token_chunk_size)
                chunk = grpo_policy_loss(new_logp, old_logp[sl], credit.advantage[sl], rmask[sl],
                                         self.cfg.clip_eps, importance_weights=importance[sl])
                if self.cfg.beta:
                    chunk = chunk + self.cfg.beta * kl_from_logp(
                        ref_logp[sl], new_logp, rmask[sl], importance_weights=importance[sl])
                if not torch.isfinite(chunk):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise ValueError("Non-finite actor loss; optimizer step cancelled")
                (chunk * (hi - lo) / (end - start)).backward()
                loss_value += float(chunk.detach()) * (hi - lo) / batch
                del chunk, new_logp, logits
            norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            if not torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError("Non-finite actor gradients; optimizer step cancelled")
            grad_norm = max(grad_norm, float(norm))
            self.optimizer.step()
            optimizer_steps += 1

        self.total_tokens += int(rmask.sum().item())
        self.rollout_index += 1
        clock = phase("logging_sec", clock)

        mask = rmask.to(self.reward_device)
        with torch.no_grad():
            delta = (ref_logp - old_logp).masked_fill(~rmask, 0)
            kl_to_init = float((torch.expm1(delta) - delta).masked_fill(~rmask, 0).sum()
                               / rmask.sum())
        metrics = {
            "rollout": self.rollout_index,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "optimizer_steps": optimizer_steps,
            "reward_count": batch,
            "raw_reward_mean": float(raw_rewards.mean()),
            "raw_reward_std": float(raw_rewards.std(correction=0)),
            "raw_reward_min": float(raw_rewards.min()),
            "raw_reward_max": float(raw_rewards.max()),
            "raw_reward_negative_fraction": float((raw_rewards < 0).float().mean()),
            "shaped_reward_mean": float(shaped.mean()),
            "reward_mean": float(shaped.mean()),
            "reward_std": float(shaped.std(correction=0)),
            "group_sigma_mean": float(scales.mean()),
            "group_sigma_min": float(scales.min()),
            "group_sigma_max": float(scales.max()),
            "advantage_abs_mean": float(advantages.abs().mean()),
            "mean_response_tokens": float(lengths.float().mean()),
            "response_tokens": int(rmask.sum()),
            "truncated_responses": int(truncated.sum()),
            "degenerate_responses": int(n_degenerate),
            "empty_responses": int(empty.sum()),
            "newline_degenerate_responses": int(repeated.sum()),
            "short_penalty_mean": float(short_penalty.mean()),
            "long_penalty_mean": float(long_penalty.mean()),
            "response_entropy": float((entropy.sum() / rmask.sum().clamp_min(1)).item()),
            "kl_to_init": kl_to_init,
            "sigma0": float(self.sigma0),
            "elapsed_sec": time.monotonic() - started,
            "gpu_hours": (time.monotonic() - self.started) * (
                2 + self.cfg.vllm_tensor_parallel_size) / 3600,
        }
        metrics.update(self._p2t_diagnostics(credit, attribution, raw_rewards, advantages, mask))
        metrics.update(alignment)
        metrics.update({f"phase_{key}": value for key, value in timings.items()})
        metrics.update({k: v for k, v in generation_summary.items()
                        if isinstance(v, (int, float, str, bool))})
        if self.actor_device.type == "cuda":
            metrics["actor_peak_gb"] = torch.cuda.max_memory_allocated(self.actor_device) / 2 ** 30
        if self.reward_device.type == "cuda":
            metrics["reward_peak_gb"] = torch.cuda.max_memory_allocated(self.reward_device) / 2 ** 30
        self._log(metrics)
        self._dump_credit(credit, attribution, raw_rewards)
        if self.cfg.checkpoint_interval and self.rollout_index % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint(self.rollout_index)
        self.pusher.maybe_push(self.rollout_index, metrics)
        return metrics

    # ----------------------------------------------------------------- reports
    def _p2t_diagnostics(self, credit, attribution, raw_rewards, advantages, mask):
        """Observation only: nothing here feeds back into the update."""
        number = mask.sum(-1).float().clamp_min(1)
        weight = credit.weight.to(self.reward_device).masked_fill(~mask, 0)
        share = weight / number[:, None]
        valid_share = share[mask]
        flat = (share.amax(-1) <= 1.0 / number + 1e-3).float().mean()
        onehot = (share.amax(-1) >= 0.9).float().mean()
        ess = 1.0 / (share.square().sum(-1) * number).clamp_min(1e-12)
        valid_attr = attribution.to(self.reward_device)[mask]
        bonus = credit.direction.to(self.reward_device)[mask]
        advantage_scale = advantages.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
        quantiles = torch.quantile(valid_attr.float(),
                                   torch.tensor([.01, .5, .99], device=valid_attr.device))
        return {
            "p2t_omega": self.cfg.omega,
            "p2t_alpha": self.cfg.alpha,
            "p2t_protocol": P2T_APPROXIMATION,
            "p2t_attribution_mean": float(valid_attr.mean()),
            "p2t_attribution_std": float(valid_attr.std(correction=0)),
            "p2t_attribution_p01": float(quantiles[0]),
            "p2t_attribution_p50": float(quantiles[1]),
            "p2t_attribution_p99": float(quantiles[2]),
            "p2t_share_max_mean": float(share.amax(-1).mean()),
            "p2t_share_ess_mean": float(ess.mean()),
            "p2t_flat_response_fraction": float(flat),
            "p2t_onehot_response_fraction": float(onehot),
            "p2t_bonus_abs_mean": float(bonus.abs().mean()),
            "p2t_bonus_over_advantage": float(bonus.abs().mean() / advantage_scale),
            # VPO-parity names so the existing plotting/analysis code keeps working.
            "credit_w_mean": float(weight[mask].mean()),
            "credit_w_std": float(weight[mask].std(unbiased=False)),
            "credit_w_max": float(weight[mask].max()),
            "credit_ess_ratio": float(ess.mean()),
        }

    def _dump_credit(self, credit, attribution, raw_rewards):
        # A small, per-step artifact: the attribution, the token bonus and the
        # final advantage behind every response token.
        torch.save({"w": credit.weight.half().cpu(),
                    "d": credit.direction.half().cpu(),
                    "i": attribution.half().cpu(),
                    "advantage": credit.advantage.half().cpu(),
                    "raw_reward": raw_rewards.float().cpu(),
                    "protocol": P2T_APPROXIMATION},
                   self.output_dir / f"rollout-{self.rollout_index}-credit.pt")

    def _validate_rollout(self, rollout, prompt_count):
        input_ids, full_mask, positions, responses, rmask, rendered, reasons, logprobs = rollout
        expected = prompt_count * self.cfg.group_size
        if len(responses) != expected or len(reasons) != expected:
            raise ValueError("Rollout must contain complete prompt groups")
        if any(reason not in {"stop", "length"} for reason in reasons):
            raise ValueError("Every response requires an explicit stop or length finish reason")
        lengths = rmask.sum(-1)
        if bool(((lengths < 1) | (lengths > self.cfg.max_response_tokens)).any()):
            raise ValueError("Invalid response length")
        valid_ids = responses[rmask]
        if ((valid_ids < 0) | (valid_ids >= self.output_mask.numel())).any() \
                or not self.output_mask[valid_ids].all():
            raise ValueError("Sampled tokens must belong to the policy output support")
        if not torch.isfinite(logprobs[rmask]).all():
            raise ValueError("Sampler log probabilities must be finite")

    def _log(self, metrics):
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)

    def save_checkpoint(self, step: int):
        path = self.output_dir / f"checkpoint-{step}"
        self.actor.save_pretrained(path)
        self.actor_tokenizer.save_pretrained(path)
        (path / "run_manifest.json").write_text(json.dumps({
            "resolved_config": asdict(self.cfg), "step": step,
            "reward_input_protocol": REWARD_INPUT_PROTOCOL,
            "p2t_protocol": P2T_APPROXIMATION, "total_tokens": self.total_tokens,
            "null_token_id": self.null_token_id, "resume_supported": False},
            indent=2, default=str))
        return path

    # ------------------------------------------------------------------ driver
    def train(self, prompts):
        prompts = list(prompts)
        if not prompts:
            raise ValueError("No prompts supplied")
        self.prepare_sigma0(prompts)
        for _ in range(self.cfg.rollout_iterations):
            start = (self.rollout_index * self.cfg.prompts_per_rollout) % len(prompts)
            chosen = [prompts[(start + i) % len(prompts)]
                      for i in range(self.cfg.prompts_per_rollout)]
            self.train_rollout(chosen)
        final = self.save_checkpoint(self.rollout_index)
        self.pusher.push(self.rollout_index, {"status": "finished"})
        return final

    def close(self):
        if self.generation is not None:
            self.generation.close()


def load_config(path) -> TrainerConfig:
    payload = json.loads(Path(path).read_text())
    known = {f.name for f in TrainerConfig.__dataclass_fields__.values()}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainerConfig(**payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="P2T baseline training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-every", type=int, default=None)
    parser.add_argument("--calibration-prompts", type=int, default=None)
    parser.add_argument("--sigma0", type=float, default=None)
    parser.add_argument("--max-rollouts", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for name, value in (("output_dir", args.output_dir), ("push_every", args.push_every),
                        ("calibration_prompts", args.calibration_prompts),
                        ("sigma0", args.sigma0), ("rollout_iterations", args.max_rollouts)):
        if value is not None:
            setattr(config, name, value)
    if not config.report_dir:
        config.report_dir = str(Path(config.output_dir) / "report")
    if config.dry_run:
        print(json.dumps({"resolved_config": asdict(config.resolved())}, indent=2, default=str))
        return
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError("P2T training needs at least two visible GPUs: actor and reward model")
    dataset_path = os.environ.get("P2T_DATASET_PATH") or str(
        Path(__file__).resolve().parents[1]
        / "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(f"training parquet not found: {dataset_path}; "
                                f"run scripts/prepare_assets.py --download")
    train_prompts, valid_prompts, split = load_prompt_dataset(dataset_path=dataset_path)
    trainer = P2TTrainer.from_pretrained(config)
    try:
        if trainer.generation is not None:
            trainer.generation.wait_until_ready()
        (trainer.output_dir / "data_split.json").write_text(json.dumps(split, indent=2, default=str))
        trainer.train(train_prompts)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
