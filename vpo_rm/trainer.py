"""Runnable two-device GRPO/VPO-RM trainer.

The implementation intentionally keeps orchestration in one dependency-light module.  It
can be used with real Qwen/Skywork checkpoints or tiny local Transformers models for a
smoke run::

    python -m vpo_rm.trainer --model ... --rm ... --smoke

GPU 0 owns the actor and optimizer; GPU 1 owns the frozen reward model.  Inputs crossing
the boundary are detached and copied explicitly.  No job is submitted by this module.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .alignment import check_response_tokens, check_tokenizers, shared_output_mask
from .core import Credit, grpo_policy_loss
from .integration import (RolloutCache, actor_response_logits, build_credit_cache,
                          response_reward_gradients, selected_logp_from_logits,
                          sampling_logits)
from .reward import LastTokenReward
from .token_policy import get_stop_token_ids, get_structural_token_ids


def normalize_prompt(text: str) -> str:
    """Canonical prompt used for deduplication and reproducible splitting."""
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def split_prompts(prompts: Iterable[str], validation_size: int = 2000):
    # Sort by the normalized key while retaining the first original spelling.  This
    # makes split hashes stable without silently rewriting user-visible prompts.
    by_key = {}
    for p in prompts:
        original, key = str(p), normalize_prompt(p)
        if key and key not in by_key:
            by_key[key] = original
    keys = sorted(by_key, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    rows = [by_key[k] for k in keys]
    n = min(max(0, int(validation_size)), len(rows))
    valid = rows[:n]
    train = rows[n:]
    payload = lambda xs: hashlib.sha256(
        "\n".join(hashlib.sha256(normalize_prompt(x).encode()).hexdigest() for x in xs).encode()
    ).hexdigest()
    return train, valid, {"train_hash": payload(train), "validation_hash": payload(valid),
                          "num_unique": len(rows), "validation_size": len(valid)}


def check_fresh_output(output_dir):
    """Reject existing training artifacts; this trainer has no resume protocol."""
    path = Path(output_dir)
    markers = ("metrics.jsonl", "credit_stats.jsonl", "profile_manifest.json",
               "profile_metrics.jsonl", "vllm-adapters", "adapter_config.json",
               "length_reward_calibration.json")
    if path.is_file() or any((path / name).exists() for name in markers) or any(path.glob("checkpoint-*")):
        raise FileExistsError(f"Use a fresh output directory; existing training artifacts found: {path}")


@dataclass
class TrainerConfig:
    model_name: str = "Qwen/Qwen3-14B"
    reward_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-8B"
    output_dir: str = "runs/skywork"
    actor_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    allocated_gpu_count: int = 2  # Reserved devices; zero for CPU-only runs.
    seed: int = 42
    group_size: int = 8
    prompts_per_rollout: int = 8
    rollout_iterations: int = 500
    policy_epochs_per_rollout: int = 1
    optimizer_minibatch_responses: int = 64
    microbatch_responses: int = 1
    credit_microbatch_responses: int = 0  # Zero preserves dense old-policy VPO credit.
    generation_microbatch_responses: int = 1
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    learning_rate: float = 1e-6
    weight_decay: float = .01
    max_grad_norm: float = 1.0
    clip_eps: float = .2
    beta: float = .01
    tau: float = 1.0
    credit_lambda: float = 2.0
    freeze_stop_tokens: bool = False
    freeze_structural: bool = False
    min_response_tokens: int | None = None  # Resolved to 0 for soft rewards, 8 for legacy.
    degenerate_penalty: float = 1.0
    init_adapter: str = ""
    kl_reference: str = "init"
    length_penalty_slope: float = 0.0
    length_penalty_anchor: int = 600
    length_reward_mode: str = "legacy"
    length_reward_sigma0: float | None = None
    short_response_threshold: int = 8
    long_response_threshold: int = 1024
    short_penalty_strength: float = .5
    long_penalty_strength: float = 2.
    advantage_std_floor_fraction: float = .5
    length_calibration_prompts: int = 128
    degenerate_newline_run: int = 32
    method: str = "vpo_rm"
    checkpoint_interval: int = 100
    token_chunk_size: int = 128
    vocab_chunk_size: int = 8192
    validation_size: int = 2000
    smoke: bool = False
    max_smoke_rollouts: int = 1
    max_smoke_prompts: int = 2
    max_smoke_group_size: int = 2
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True

    def resolved(self) -> "TrainerConfig":
        c = TrainerConfig(**asdict(self))
        if c.length_reward_mode not in {"legacy", "soft"}:
            raise ValueError("length_reward_mode must be legacy or soft")
        if c.min_response_tokens is None:
            c.min_response_tokens = 0 if c.length_reward_mode == "soft" else 8
        if c.method not in {"grpo", "vpo_rm"}:
            raise ValueError("method must be grpo or vpo_rm")
        if c.kl_reference not in {"rollout", "init"}:
            raise ValueError("kl_reference must be rollout or init")
        if c.init_adapter and not c.lora:
            raise ValueError("init_adapter requires lora=True")
        if c.top_p != 1.0 or c.top_k != 0:
            raise ValueError("training supports top_p=1 and top_k=0 only")
        if not math.isfinite(c.temperature) or not .01 <= c.temperature <= 2.0:
            raise ValueError("temperature must be in [0.01, 2] to avoid backend clamping")
        if c.lora_dropout != 0:
            raise ValueError("training requires zero dropout for matching rollout probabilities")
        if c.smoke:
            c.rollout_iterations = min(c.rollout_iterations, c.max_smoke_rollouts)
            c.prompts_per_rollout = min(c.prompts_per_rollout, c.max_smoke_prompts)
            c.group_size = min(c.group_size, c.max_smoke_group_size)
            c.max_response_tokens = min(c.max_response_tokens, 32)
        if c.group_size < 2 or c.prompts_per_rollout < 1:
            raise ValueError("group_size must be >=2 and prompts_per_rollout positive")
        if c.policy_epochs_per_rollout < 1 or c.optimizer_minibatch_responses < 1:
            raise ValueError("policy_epochs_per_rollout and optimizer_minibatch_responses must be positive")
        if c.max_response_tokens < 1 or not 0 <= c.min_response_tokens <= c.max_response_tokens:
            raise ValueError("require 0 <= min_response_tokens <= max_response_tokens and a positive maximum")
        if c.length_reward_mode == "soft":
            if c.min_response_tokens != 0 or c.length_penalty_slope != 0:
                raise ValueError("soft length rewards require min_response_tokens=0 and length_penalty_slope=0")
            if not 0 < c.short_response_threshold <= c.long_response_threshold < c.max_response_tokens:
                raise ValueError("require 0 < short_response_threshold <= long_response_threshold < max_response_tokens")
            if c.length_reward_sigma0 is not None and (
                    not math.isfinite(c.length_reward_sigma0) or c.length_reward_sigma0 <= 0):
                raise ValueError("length_reward_sigma0 must be finite and positive")
            if any(not math.isfinite(x) or x < 0 for x in
                   (c.short_penalty_strength, c.long_penalty_strength)):
                raise ValueError("length penalty strengths must be finite and nonnegative")
            if not math.isfinite(c.advantage_std_floor_fraction) or c.advantage_std_floor_fraction <= 0:
                raise ValueError("advantage_std_floor_fraction must be finite and positive")
            if not math.isfinite(c.degenerate_penalty) or c.degenerate_penalty <= 0:
                raise ValueError("soft degenerate_penalty must be finite and positive")
            if c.length_calibration_prompts < 1 or c.degenerate_newline_run < 1:
                raise ValueError("length_calibration_prompts and degenerate_newline_run must be positive")
        if c.checkpoint_interval < 1 or c.microbatch_responses < 1:
            raise ValueError("checkpoint_interval and microbatch_responses must be positive")
        if type(c.credit_microbatch_responses) is not int or c.credit_microbatch_responses < 0:
            raise ValueError("credit_microbatch_responses must be a nonnegative integer")
        if not isinstance(c.allocated_gpu_count, int) or c.allocated_gpu_count < 0:
            raise ValueError("allocated_gpu_count must be a nonnegative integer")
        return c


class VPOTrainer:
    """Minimal production trainer; actor and RM are always placed on separate devices."""
    def __init__(self, actor, actor_tokenizer, reward, reward_tokenizer, config: TrainerConfig):
        self.cfg = config.resolved()
        check_fresh_output(self.cfg.output_dir)
        # Seed every RNG used by prompt sampling and generation.  The config's seed
        # must be effective for reproducible rollouts across formal runs.
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        self.actor, self.actor_tokenizer = actor, actor_tokenizer
        self.reward, self.reward_tokenizer = reward, reward_tokenizer
        self.actor_device = torch.device(self.cfg.actor_device)
        self.reward_device = torch.device(self.cfg.reward_device)
        self.actor.to(self.actor_device)
        self.reward.to(self.reward_device).eval()
        for p in self.reward.parameters():
            p.requires_grad_(False)
        self.output_mask = shared_output_mask(self.actor_tokenizer,
                                               self.actor.get_output_embeddings().weight.shape[0],
                                               self.actor_device)
        self.stop_token_ids = get_stop_token_ids(self.actor_tokenizer)
        self.structural_token_ids = (get_structural_token_ids(self.actor_tokenizer)
                                     if self.cfg.freeze_structural else ())
        if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in self.actor.modules()):
            raise ValueError("Actor dropout must be zero to match generation and training probabilities")
        for name in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
            if float(getattr(getattr(self.actor, "config", None), name, 0.0) or 0.0) != 0:
                raise ValueError("Actor dropout must be zero to match generation and training probabilities")
        self.reference_actor = None
        self.reference_adapter = None
        if self.cfg.kl_reference == "init":
            if hasattr(self.actor, "disable_adapter"):
                self.reference_adapter = "ref" if self.cfg.init_adapter else "base"
            else:
                self.reference_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW((p for p in self.actor.parameters() if p.requires_grad),
                                           lr=self.cfg.learning_rate,
                                           weight_decay=self.cfg.weight_decay)
        self.rollout_index = 0
        self.total_tokens = 0
        self._started = time.monotonic()
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(self.cfg.output_dir) / "metrics.jsonl"
        self.filtered_prompt_count = 0
        self.data_split = None
        self.length_reward_calibration = None

    @staticmethod
    def _render_chat_prompt(tokenizer, prompt: str, tokenize: bool = False):
        """Render one user prompt using a model's template, with a plain fallback."""
        messages = [{"role": "user", "content": str(prompt)}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=tokenize, add_generation_prompt=True,
                    enable_thinking=False)
            except (ImportError, TypeError, ValueError):
                try:
                    return tokenizer.apply_chat_template(
                        messages, tokenize=tokenize, add_generation_prompt=True)
                except (ImportError, TypeError, ValueError):
                    pass
        if tokenize:
            return tokenizer(str(prompt), add_special_tokens=True)["input_ids"]
        return str(prompt)

    @classmethod
    def from_pretrained(cls, config: TrainerConfig) -> "VPOTrainer":
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        c = config.resolved()
        check_fresh_output(c.output_dir)
        # PEFT initializes random LoRA A weights before trainer.__init__ runs.
        random.seed(c.seed)
        torch.manual_seed(c.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c.seed)
        dtype = torch.bfloat16 if c.actor_device.startswith("cuda") else torch.float32
        atok = AutoTokenizer.from_pretrained(c.model_name, padding_side="left", trust_remote_code=True)
        rtok = AutoTokenizer.from_pretrained(c.reward_model_name, padding_side="right", trust_remote_code=True)
        # Skywork checkpoints sometimes carry 151654 as pad; experiment contract uses EOS.
        eos = atok.eos_token_id
        for tok in (atok, rtok):
            if eos is not None:
                tok.pad_token = atok.eos_token
                tok.pad_token_id = eos
        actor = AutoModelForCausalLM.from_pretrained(c.model_name, torch_dtype=dtype,
                                                     trust_remote_code=True)
        rm_base = AutoModelForSequenceClassification.from_pretrained(c.reward_model_name,
                                                                       torch_dtype=dtype,
                                                                       trust_remote_code=True)
        if eos is not None:
            actor.config.pad_token_id = eos
            if getattr(actor, "generation_config", None) is not None:
                actor.generation_config.pad_token_id = eos
            rm_base.config.pad_token_id = eos
        # Keep only the decoder backbone and scalar score head.  This also works for Qwen3.
        backbone = getattr(rm_base, "base_model", None)
        if backbone is None:
            backbone = getattr(rm_base, "model", None)
        if backbone is None or not hasattr(rm_base, "score"):
            raise ValueError("Reward checkpoint must expose a decoder backbone and scalar score head")
        reward = LastTokenReward(backbone, rm_base.score)
        check_tokenizers(atok, rtok,
                         actor.get_output_embeddings().weight.shape[0],
                         reward.get_input_embeddings().weight.shape[0])
        if c.lora:
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
                if c.init_adapter:
                    # p9g: shared SFT initialization for both arms.  The same
                    # checkpoint is mounted a second time as a frozen "ref"
                    # adapter for the init-anchored KL; no extra backbone copy.
                    actor = PeftModel.from_pretrained(actor, c.init_adapter, is_trainable=True)
                    if c.kl_reference == "init":
                        actor.load_adapter(c.init_adapter, adapter_name="ref", is_trainable=False)
                    actor.set_adapter("default")
                else:
                    actor = get_peft_model(actor, LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha,
                        lora_dropout=c.lora_dropout, bias="none", task_type="CAUSAL_LM",
                        target_modules=c.target_modules))
            except ImportError:
                if c.smoke:
                    # tiny local smoke environments often omit peft; train the model directly.
                    pass
                else:
                    raise RuntimeError("peft is required for a non-smoke LoRA run")
        if c.gradient_checkpointing:
            # The experiment contract has always specified activation checkpointing
            # for the Actor update; the non-reentrant variant also works with
            # frozen embeddings under LoRA.  Eval/no-grad forwards are unaffected.
            try:
                actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                actor.gradient_checkpointing_enable()
        return cls(actor, atok, reward, rtok, c)

    def _encode_prompts(self, prompts: Sequence[str]):
        texts = [self._render_chat_prompt(self.actor_tokenizer, p, tokenize=False)
                 for p in prompts]
        batch = self.actor_tokenizer(texts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.cfg.max_prompt_tokens)
        return {k: v.to(self.actor_device) for k, v in batch.items()}, texts

    def filter_prompts(self, prompts: Sequence[str]) -> list[str]:
        """Apply the experiment's post-template prompt length policy.

        Both model-specific chat templates are measured before training.  This avoids
        silently truncating a prompt for the actor while feeding a different prefix to
        the reward model.
        """
        kept = []
        for prompt in prompts:
            actor_text = self._render_chat_prompt(self.actor_tokenizer, prompt, tokenize=False)
            actor_ids = self.actor_tokenizer(actor_text, add_special_tokens=False)["input_ids"]
            reward_ids = self._render_chat_prompt(self.reward_tokenizer, prompt, tokenize=True)
            if len(actor_ids) <= self.cfg.max_prompt_tokens and len(reward_ids) <= self.cfg.max_prompt_tokens:
                kept.append(prompt)
        self.filtered_prompt_count = len(prompts) - len(kept)
        if self.filtered_prompt_count:
            self._log({"event": "prompt_filter", "input_prompts": len(prompts),
                       "kept_prompts": len(kept), "dropped_prompts": self.filtered_prompt_count,
                       "max_prompt_tokens": self.cfg.max_prompt_tokens})
        return kept

    @torch.no_grad()
    def rollout(self, prompts: Sequence[str]):
        from transformers import GenerationConfig
        batch, rendered_prompts = self._encode_prompts(prompts)
        self.actor.eval()
        # Generate one prompt at a time to bound KV-cache memory.  The configured
        # generation microbatch is a number of responses, so with the main-run
        # value of 1 this deliberately makes one generate call per answer.
        prompt_width = batch["input_ids"].shape[1]
        response_chunks = []
        generation_mb = max(1, int(self.cfg.generation_microbatch_responses))
        for prompt_idx in range(len(prompts)):
            sub = {k: v[prompt_idx:prompt_idx + 1] for k, v in batch.items()}
            for group_start in range(0, self.cfg.group_size, generation_mb):
                n = min(generation_mb, self.cfg.group_size - group_start)
                # stop at BOTH end tokens: base generation_config only lists
                # <|endoftext|>, but SFT teaches <|im_end|> — without this the
                # rollout continues past the answer end (2026-09-16 audit).
                generated = self.actor.generate(
                    generation_config=GenerationConfig(),
                    **sub, do_sample=True, temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p, top_k=self.cfg.top_k,
                    max_new_tokens=self.cfg.max_response_tokens,
                    min_new_tokens=self.cfg.min_response_tokens,
                    num_return_sequences=n,
                    eos_token_id=list(self.stop_token_ids),
                    suppress_tokens=(~self.output_mask).nonzero().flatten().tolist(),
                    pad_token_id=self.actor_tokenizer.pad_token_id,
                    return_dict_in_generate=False)
                response_chunks.append(generated[:, prompt_width:])
        max_response_width = max(x.shape[1] for x in response_chunks)
        pad_id = self.actor_tokenizer.pad_token_id
        response_chunks = [torch.nn.functional.pad(
            x, (0, max_response_width - x.shape[1]), value=pad_id)
            for x in response_chunks]
        responses = torch.cat(response_chunks, dim=0)
        expanded_input = batch["input_ids"].repeat_interleave(self.cfg.group_size, dim=0)
        out = torch.cat([expanded_input, responses], dim=1)
        # HF repeats each input contiguously when num_return_sequences>1.
        prompt_mask = batch["attention_mask"].repeat_interleave(self.cfg.group_size, dim=0)
        rendered_prompts = [p for p in rendered_prompts for _ in range(self.cfg.group_size)]
        # Generation pads after EOS; retain EOS itself as a valid response token.
        is_stop = torch.zeros_like(responses, dtype=torch.bool)
        for stop_id in self.stop_token_ids:
            is_stop |= responses.eq(stop_id)
        # Include exactly the first stop token, excluding any later pad/eos.
        seen_before = is_stop.long().cumsum(-1) - is_stop.long()
        rmask = seen_before.eq(0)
        finish_reasons = ["stop" if stopped else "length"
                          for stopped in is_stop.any(-1).tolist()]
        full_mask = torch.cat([prompt_mask, rmask.to(prompt_mask.dtype)], dim=1)
        positions = torch.arange(prompt_width, out.shape[1], device=self.actor_device).expand(out.shape[0], -1)
        return out, full_mask, positions, responses, rmask, rendered_prompts, finish_reasons

    def _sampling_logits(self, logits):
        # Keep the full tensor in model precision. Each probability/credit
        # reduction divides by temperature only after promoting its FP32 chunk.
        return sampling_logits(logits,
                               min_response_tokens=getattr(self.cfg, "min_response_tokens", 0),
                               stop_token_ids=getattr(self, "stop_token_ids", ()), inplace=True)

    def sampling_manifest(self):
        support = self.output_mask.nonzero().flatten().tolist()
        return {"protocol": "full_softmax_with_minimum_stop_mask_v1",
                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
                "top_k": self.cfg.top_k, "min_tokens": self.cfg.min_response_tokens,
                "presence_penalty": 0.0, "stop_token_ids": list(self.stop_token_ids),
                "support_sha256": hashlib.sha256(json.dumps(support).encode()).hexdigest()}

    def _reward_batch(self, input_ids, full_mask, positions, responses, rmask, prompts=None):
        """Build Skywork-formatted RM inputs while preserving generated token IDs."""
        if prompts is None:
            raise ValueError("prompts are required to apply the reward-model chat template")
        rows, width = [], []
        for prompt, response, valid in zip(prompts, responses, rmask):
            prefix = self._render_chat_prompt(self.reward_tokenizer, str(prompt), tokenize=True)
            if hasattr(prefix, "input_ids"):
                prefix = prefix.input_ids
            prefix = list(prefix)
            answer = response[valid].detach().cpu().tolist()
            rows.append(prefix + answer)
            width.append((len(prefix), len(answer)))
        # The experiment's physical response microbatch is one.  In particular,
        # RM input gradients must be computed in chunks too; retaining the
        # backward graph for all 64 responses can exceed a 140GB H200.
        micro = max(1, int(self.cfg.microbatch_responses))
        reward_parts, grad_parts = [], []
        pad = self.reward_tokenizer.pad_token_id
        for chunk_start in range(0, len(rows), micro):
            chunk_end = min(len(rows), chunk_start + micro)
            chunk_rows = rows[chunk_start:chunk_end]
            max_len = max(map(len, chunk_rows))
            rid = torch.full((len(chunk_rows), max_len), pad, dtype=torch.long,
                             device=self.reward_device)
            rmask_full = torch.zeros_like(rid, dtype=torch.long)
            rpos = torch.full((len(chunk_rows), responses.shape[1]), -1,
                              dtype=torch.long, device=self.reward_device)
            for j, row in enumerate(chunk_rows):
                rid[j, :len(row)] = torch.tensor(row, dtype=torch.long,
                                                 device=self.reward_device)
                rmask_full[j, :len(row)] = 1
                start, length = width[chunk_start + j]
                rpos[j, :length] = torch.arange(start, start + length,
                                                device=self.reward_device)
            toks = responses[chunk_start:chunk_end].to(self.reward_device)
            valid = rmask[chunk_start:chunk_end].to(self.reward_device)
            if self.cfg.method == "grpo":
                # GRPO uses only sequence rewards.  Avoid constructing the input
                # gradient graph (which is the expensive VPO-RM operation).
                with torch.no_grad():
                    emb = self.reward.get_input_embeddings()(rid)
                    reward = self.reward(inputs_embeds=emb, attention_mask=rmask_full)
                    del emb
                if reward.ndim != 1 or not torch.isfinite(reward).all():
                    raise ValueError("Reward adapter must return finite scalar scores")
                grad = None
            else:
                reward, grad = response_reward_gradients(
                    self.reward, rid, rmask_full, rpos, toks, valid)
            reward_parts.append(reward.cpu())
            if grad is not None:
                grad_parts.append(grad.cpu())
            del rid, rmask_full, rpos, toks, valid, reward, grad
        rewards = torch.cat(reward_parts).to(self.reward_device)
        grads = torch.cat(grad_parts).to(self.reward_device) if grad_parts else None
        return rewards, grads, None, None

    @torch.no_grad()
    def _old_logp_microbatch(self, input_ids, attention_mask, positions, responses, response_mask,
                             adapter=None, entropy_out=None):
        """Cache log-probabilities without a full B×T×V allocation.

        adapter="ref" evaluates the frozen init adapter (anchored-KL reference)
        by switching the active LoRA and restoring the trainable one after.
        entropy_out optionally accumulates per-token response entropy.
        """
        B, T = responses.shape
        result = torch.zeros((B, T), dtype=torch.float32, device=self.actor_device)
        micro = max(1, int(self.cfg.microbatch_responses))
        switched = adapter not in {None, "base", "initial_model"}
        if switched:
            self.actor.set_adapter(adapter)
        model = (self.reference_actor if adapter == "initial_model" else self.actor)
        context = self.actor.disable_adapter() if adapter == "base" else nullcontext()
        try:
            with context:
                for start in range(0, B, micro):
                    end = min(B, start + micro)
                    logits = actor_response_logits(model, input_ids[start:end], attention_mask[start:end],
                                                   positions[start:end], response_mask[start:end],
                                                   output_mask=self.output_mask)
                    logits = self._sampling_logits(logits)
                    z = logits.float() / getattr(self.cfg, "temperature", 1.0)
                    safe = responses[start:end].masked_fill(~response_mask[start:end], 0)
                    result[start:end] = z.gather(-1, safe[..., None]).squeeze(-1) - z.logsumexp(-1)
                    if entropy_out is not None:
                        logz = z.logsumexp(-1, keepdim=True)
                        lp = z - logz
                        p = lp.exp()
                        entropy_out[start:end] = -torch.where(
                            p > 0, p * lp, torch.zeros_like(p)).sum(-1).masked_fill(
                            ~response_mask[start:end], 0)
                        del p, lp, logz
                    del logits, z
        finally:
            if switched:
                self.actor.set_adapter("default")
        return result

    def _reference_logp(self, *args):
        adapter = self.reference_adapter if self.reference_actor is None else "initial_model"
        return self._old_logp_microbatch(*args, adapter=adapter)

    @torch.no_grad()
    def _old_policy_entropy(self, logits, response_mask):
        """Response entropy from FP32 token blocks of model-precision logits."""
        entropy = torch.zeros(response_mask.shape, dtype=torch.float32, device=logits.device)
        rows, times = response_mask.bool().nonzero(as_tuple=True)
        chunk = max(1, int(self.cfg.token_chunk_size)) * 4
        for lo in range(0, rows.numel(), chunk):
            r, t = rows[lo:lo + chunk], times[lo:lo + chunk]
            z = logits[r, t].float() / self.cfg.temperature
            logz = z.logsumexp(-1, keepdim=True)
            lp = z - logz
            p = lp.exp()
            entropy[r, t] = -torch.where(p > 0, p * lp, torch.zeros_like(p)).sum(-1)
        return entropy

    @torch.no_grad()
    def _credit_cache_microbatch(self, input_ids, full_mask, positions, responses,
                                response_mask, grads, advantages, scales):
        """Build VPO caches without retaining a full rollout's vocabulary logits.

        Rewards, advantages and scales must already describe complete prompt
        groups. Only independent response attribution is partitioned here;
        optimizer batches and per-response credit normalization stay unchanged.
        The RM embedding is copied once, and response gradients move to the
        actor device one microbatch at a time.
        """
        def finish_phase(start):
            if self.actor_device.type == "cuda":
                torch.cuda.synchronize(self.actor_device)
            return time.monotonic() - start

        start = time.monotonic()
        rm_weight = self.reward.get_input_embeddings().weight.detach().to(self.actor_device)
        advantages, scales = advantages.to(self.actor_device), scales.to(self.actor_device)
        credit_seconds = finish_phase(start)
        actor_seconds = 0.
        caches, entropies = [], []
        micro = self.cfg.credit_microbatch_responses
        for lo in range(0, responses.shape[0], micro):
            sl = slice(lo, lo + micro)
            start = time.monotonic()
            logits = actor_response_logits(self.actor, input_ids[sl], full_mask[sl],
                                           positions[sl], response_mask[sl],
                                           output_mask=self.output_mask)
            logits = self._sampling_logits(logits)
            actor_seconds += finish_phase(start)
            start = time.monotonic()
            input_grads = grads[sl].to(self.actor_device)
            caches.append(build_credit_cache(
                logits, responses[sl], input_grads, rm_weight,
                advantages[sl], scales[sl], response_mask[sl], self.cfg.tau,
                credit_lambda=self.cfg.credit_lambda,
                freeze_stop_tokens=self.cfg.freeze_stop_tokens,
                freeze_structural=self.cfg.freeze_structural,
                stop_token_ids=self.stop_token_ids,
                structural_token_ids=self.structural_token_ids,
                policy_temperature=self.cfg.temperature,
                min_response_tokens=self.cfg.min_response_tokens,
                token_chunk_size=self.cfg.token_chunk_size,
                vocab_chunk_size=self.cfg.vocab_chunk_size))
            entropies.append(self._old_policy_entropy(logits, response_mask[sl]))
            del logits, input_grads
            credit_seconds += finish_phase(start)
        start = time.monotonic()
        credit = Credit(
            torch.cat([cache.credit.advantage for cache in caches]),
            torch.cat([cache.credit.direction for cache in caches]),
            torch.cat([cache.credit.weight for cache in caches]),
            torch.cat([cache.credit.tau_used for cache in caches]))
        cache = RolloutCache(torch.cat([cache.old_logp for cache in caches]), credit,
                             self.cfg.temperature, self.cfg.min_response_tokens,
                             tuple(self.stop_token_ids))
        entropy = torch.cat(entropies)
        credit_seconds += finish_phase(start)
        return cache, entropy, {"actor_old_logits_sec": actor_seconds,
                                "credit_cache_sec": credit_seconds}

    def _response_degeneracy(self, responses, rmask):
        from .length_reward import response_degeneracy
        stop = set(self.stop_token_ids)
        texts = [self.actor_tokenizer.decode(
            [x for x in row[valid].detach().cpu().tolist() if x not in stop],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for row, valid in zip(responses, rmask)]
        empty, repeated = response_degeneracy(texts, self.cfg.degenerate_newline_run)
        return (torch.tensor(empty, device=self.reward_device, dtype=torch.bool),
                torch.tensor(repeated, device=self.reward_device, dtype=torch.bool))

    @torch.no_grad()
    def prepare_length_reward(self, prompts: Sequence[str]):
        """Freeze an initial-policy RM scale before the first optimizer update.

        Calibration uses the same sampler and exact RM token formatting as training.
        It consumes the sampler RNG stream, but performs no updates and does not
        increment the training rollout/token counters. Explicit sigma0 lets all
        comparison arms share one independently recorded calibration.
        """
        if self.cfg.length_reward_mode != "soft":
            return None
        if self.length_reward_calibration is not None:
            return self.length_reward_calibration
        if self.rollout_index:
            raise RuntimeError("Length reward calibration must precede all training updates")
        from .length_reward import calibrate_reward_scale
        started = time.monotonic()
        record = {"mode": "soft", "sigma0": self.cfg.length_reward_sigma0,
                  "source": "explicit" if self.cfg.length_reward_sigma0 is not None else "initial_policy",
                  "calibration_prompt_count": 0, "group_size": self.cfg.group_size,
                  "seed": self.cfg.seed, "model": self.cfg.model_name,
                  "init_adapter": self.cfg.init_adapter, "reward_model": self.cfg.reward_model_name,
                  "sampling": self.sampling_manifest(),
                  "reward_format": "reward_chat_generation_prefix_plus_exact_response_tokens_v1",
                  "length_counting": "generated tokens including EOS, excluding prompt and padding",
                  "rng_policy": "calibration consumes sampler RNG before training",
                  "short_response_threshold": self.cfg.short_response_threshold,
                  "long_response_threshold": self.cfg.long_response_threshold,
                  "max_response_tokens": self.cfg.max_response_tokens,
                  "short_penalty_strength": self.cfg.short_penalty_strength,
                  "long_penalty_strength": self.cfg.long_penalty_strength,
                  "advantage_std_floor_fraction": self.cfg.advantage_std_floor_fraction}
        if self.cfg.length_reward_sigma0 is None:
            unique = list(dict.fromkeys(prompts))
            count = self.cfg.length_calibration_prompts
            if len(unique) < count:
                raise ValueError(f"Calibration requires {count} distinct training prompts; got {len(unique)}")
            chosen = random.Random(self.cfg.seed).sample(unique, count)
            scores, groups, valid_rows, details = [], [], [], []
            old_method = self.cfg.method
            was_training = self.actor.training
            try:
                # Even VPO needs only forward RM scores during calibration.
                self.cfg.method = "grpo"
                for start in range(0, count, self.cfg.prompts_per_rollout):
                    batch_prompts = chosen[start:start + self.cfg.prompts_per_rollout]
                    rollout = self.rollout(batch_prompts)
                    ids, mask, positions, responses, rmask, _, reasons = rollout
                    self._validate_rollout(rollout, len(batch_prompts))
                    rewards, *_ = self._reward_batch(ids, mask, positions, responses, rmask,
                        [p for p in batch_prompts for _ in range(self.cfg.group_size)])
                    empty, repeated = self._response_degeneracy(responses, rmask)
                    valid = torch.tensor([x == "stop" for x in reasons], device=rewards.device) & ~(empty | repeated)
                    for j, (reward, is_valid) in enumerate(zip(rewards.tolist(), valid.tolist())):
                        group = start + j // self.cfg.group_size
                        scores.append(reward); groups.append(group); valid_rows.append(is_valid)
                        details.append({"group": group, "reward": reward, "valid": is_valid,
                                        "length": int(rmask[j].sum()), "finish_reason": reasons[j]})
                sigma0 = calibrate_reward_scale(torch.tensor(scores), torch.tensor(groups),
                                                torch.tensor(valid_rows, dtype=torch.bool))
            finally:
                self.cfg.method = old_method
                self.actor.train(was_training)
            self.cfg.length_reward_sigma0 = sigma0
            record.update(sigma0=sigma0, calibration_prompt_count=count, responses=details,
                          prompt_sha256=[hashlib.sha256(p.encode()).hexdigest() for p in chosen])
        record["elapsed_sec"] = time.monotonic() - started
        (Path(self.cfg.output_dir) / "length_reward_calibration.json").write_text(
            json.dumps(record, indent=2, allow_nan=False))
        self.length_reward_calibration = record
        return record

    def _validate_rollout(self, rollout, prompt_count):
        ids, mask, positions, responses, rmask, rendered, reasons = rollout
        expected = prompt_count * self.cfg.group_size
        if len(responses) != expected or len(rendered) != expected or len(reasons) != expected:
            raise ValueError("Rollout must contain complete prompt groups")
        if any(x not in {"stop", "length"} for x in reasons):
            raise ValueError("Every response requires an explicit stop or length finish reason")
        lengths = rmask.sum(-1)
        if bool(((lengths < 1) | (lengths > self.cfg.max_response_tokens)).any()):
            raise ValueError("Invalid response length")
        check_response_tokens(ids, mask, positions, responses, rmask)
        valid_ids = responses[rmask]
        if ((valid_ids < 0) | (valid_ids >= self.output_mask.numel())).any() or not self.output_mask[valid_ids].all():
            raise ValueError("Sampled tokens must belong to the policy output support")

    def _write_rollout_artifacts(self, prompts, rollout):
        out = Path(self.cfg.output_dir)
        step = self.rollout_index + 1
        rows = [] if rollout is None else [row[valid].detach().cpu().tolist()
                                           for row, valid in zip(rollout[3], rollout[4])]
        (out / f"rollout-{step}-tokens.json").write_text(json.dumps(rows))
        (out / f"rollout-{step}-prompts.json").write_text(json.dumps(list(prompts)))

    def train_rollout(self, prompts: Sequence[str]) -> dict[str, float]:
        def elapsed_phase(start):
            # Attribute completed CUDA work to the correct stage, not a later
            # host read/copy that happens to synchronize the stream.
            for device in {self.actor_device, self.reward_device}:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            return time.monotonic() - start

        t0 = time.monotonic()
        phase = {}
        soft = self.cfg.length_reward_mode == "soft"
        if soft:
            if self.cfg.length_reward_sigma0 is None:
                raise RuntimeError("Call prepare_length_reward(training_prompts) before training to calibrate sigma0")
            self.prepare_length_reward([])
        # Release gradients from the preceding optimizer step before allocating
        # rollout logits/KV caches (critical for long 64-response runs).
        self.optimizer.zero_grad(set_to_none=True)
        if self.actor_device.type == "cuda":
            torch.cuda.empty_cache()
        tp = time.monotonic()
        if soft:
            from .rollout_selection import select_training_rollout
            rollout, prompts, selection_stats = select_training_rollout(self, prompts)
        else:
            rollout = self.rollout(prompts)
            selection_stats = {"resampled_groups": 0, "skipped_groups": 0,
                               "input_prompt_groups": len(prompts), "kept_prompt_groups": len(prompts),
                               "generated_response_tokens": int(rollout[4].sum())}
        self._write_rollout_artifacts(prompts, rollout)
        if rollout is None:
            # No response reaches the loss, including KL, Adam moments or weight decay.
            self.rollout_index += 1
            metrics = {"rollout": self.rollout_index, "skipped_rollout": True,
                       "optimizer_steps": 0, "response_tokens": 0, "reward_count": 0,
                       "loss": 0., "elapsed_sec": time.monotonic() - t0,
                       "gpu_hours": (time.monotonic() - self._started) * self.cfg.allocated_gpu_count / 3600,
                       "phase_generation_sec": elapsed_phase(tp), **selection_stats}
            self._log(metrics)
            if self.rollout_index % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(self.rollout_index)
            return metrics
        input_ids, full_mask, positions, responses, rmask, rendered, finish_reasons = rollout
        if len(finish_reasons) != responses.shape[0] or any(x not in {"stop", "length"} for x in finish_reasons):
            raise ValueError("Every response requires an explicit stop or length finish reason")
        valid_ids = responses[rmask]
        if ((valid_ids < 0) | (valid_ids >= self.output_mask.numel())).any() or not self.output_mask[valid_ids].all():
            raise ValueError("Sampled tokens must belong to the policy output support")
        phase["generation_sec"] = elapsed_phase(tp)
        old_logits = None
        if self.cfg.method == "vpo_rm" and self.cfg.credit_microbatch_responses == 0:
            tp = time.monotonic()
            with torch.no_grad():
                old_logits = actor_response_logits(self.actor, input_ids, full_mask, positions, rmask,
                                                   output_mask=self.output_mask)
                old_logits = self._sampling_logits(old_logits)
            phase["actor_old_logits_sec"] = elapsed_phase(tp)
        tp = time.monotonic()
        rewards, grads, rid, rmask_full = self._reward_batch(
            input_ids, full_mask, positions, responses, rmask,
            [p for p in prompts for _ in range(self.cfg.group_size)])
        phase["reward_model_gradient_sec" if self.cfg.method == "vpo_rm"
              else "reward_model_forward_sec"] = elapsed_phase(tp)
        B = rewards.shape[0]
        group_ids = torch.arange(B, device=self.reward_device) // self.cfg.group_size
        from .core import group_advantages, guard_degenerate_rewards
        lengths_dev = rmask.sum(-1).to(rewards.device)
        raw_rewards = rewards.detach().clone()
        short_penalty = torch.zeros_like(rewards)
        long_penalty = torch.zeros_like(rewards)
        truncated = torch.tensor([reason == "length" for reason in finish_reasons],
                                 device=rewards.device, dtype=torch.bool)
        empty = torch.zeros_like(truncated)
        repeated = torch.zeros_like(truncated)
        if soft:
            from .length_reward import soft_length_penalties
            short_penalty, long_penalty = soft_length_penalties(
                lengths_dev, self.cfg.length_reward_sigma0,
                short_threshold=self.cfg.short_response_threshold,
                long_threshold=self.cfg.long_response_threshold,
                max_length=self.cfg.max_response_tokens,
                short_strength=self.cfg.short_penalty_strength,
                long_strength=self.cfg.long_penalty_strength)
            rewards = rewards - short_penalty - long_penalty
            empty, repeated = self._response_degeneracy(responses, rmask)
            # Only actual degeneration gets a hard floor; short nonempty answers
            # and true truncations retain their softly shaped reward.
            flags = empty | repeated
            n_degenerate = int(flags.sum())
            rewards = rewards.clone()
            for group in group_ids.unique():
                members = group_ids == group
                flagged = members & flags
                if bool(flagged.any()):
                    good = members & ~flags
                    if not bool(good.any()):
                        raise RuntimeError("An entirely degenerate group escaped rollout selection")
                    rewards[flagged] = rewards[good].min() - self.cfg.degenerate_penalty
            advantages, scales = group_advantages(
                rewards, group_ids,
                std_floor=self.cfg.advantage_std_floor_fraction * self.cfg.length_reward_sigma0)
        else:
            # Preserve the historical experiment's below-anchor cost and floor.
            if self.cfg.length_penalty_slope > 0:
                short_penalty = self.cfg.length_penalty_slope * (
                    self.cfg.length_penalty_anchor - lengths_dev).clamp_min(0).float()
                rewards = rewards - short_penalty
            rewards, n_degenerate = guard_degenerate_rewards(
                rewards, lengths_dev, group_ids,
                max(1, self.cfg.min_response_tokens), self.cfg.degenerate_penalty,
                also_floor=truncated if self.cfg.length_penalty_slope > 0 else None)
            advantages, scales = group_advantages(rewards, group_ids)
        length_penalty = short_penalty + long_penalty
        if n_degenerate:
            print(f"[reward-guard] {n_degenerate}/{B} flagged responses floored", flush=True)
        mean_len = float(lengths_dev.float().mean())
        if mean_len < self.cfg.min_response_tokens:
            print(f"[reward-guard] WARNING mean response length {mean_len:.1f} below "
                  f"{self.cfg.min_response_tokens} — degenerate policy suspected", flush=True)
        reward_rows = [{"length": int(lengths_dev[i]), "finish_reason": finish_reasons[i],
                        "raw_reward": float(raw_rewards[i]), "short_penalty": float(short_penalty[i]),
                        "long_penalty": float(long_penalty[i]), "reward": float(rewards[i]),
                        "advantage": float(advantages[i]), "scale": float(scales[i]),
                        "empty": bool(empty[i]), "newline_degenerate": bool(repeated[i])}
                       for i in range(B)]
        (Path(self.cfg.output_dir) / f"rollout-{self.rollout_index + 1}-rewards.json").write_text(
            json.dumps(reward_rows, allow_nan=False))
        ref_logp = None
        entropy = None
        if self.cfg.method == "vpo_rm":
            if self.cfg.credit_microbatch_responses:
                cache, entropy, credit_phase = self._credit_cache_microbatch(
                    input_ids, full_mask, positions, responses, rmask, grads, advantages, scales)
                phase.update(credit_phase)
            else:
                tp = time.monotonic()
                rm_weight = self.reward.get_input_embeddings().weight.detach().to(self.actor_device)
                cache = build_credit_cache(old_logits, responses, grads.to(self.actor_device), rm_weight,
                                           advantages.to(self.actor_device), scales.to(self.actor_device),
                                           rmask, self.cfg.tau, credit_lambda=self.cfg.credit_lambda,
                                           freeze_stop_tokens=self.cfg.freeze_stop_tokens,
                                           freeze_structural=self.cfg.freeze_structural,
                                           stop_token_ids=self.stop_token_ids,
                                           structural_token_ids=self.structural_token_ids,
                                           policy_temperature=self.cfg.temperature,
                                           min_response_tokens=self.cfg.min_response_tokens,
                                           token_chunk_size=self.cfg.token_chunk_size,
                                           vocab_chunk_size=self.cfg.vocab_chunk_size)
                phase["credit_cache_sec"] = elapsed_phase(tp)
            if self.cfg.kl_reference == "init":
                tp = time.monotonic()
                ref_logp = self._reference_logp(input_ids, full_mask, positions, responses, rmask)
                phase["ref_logp_sec"] = elapsed_phase(tp)
            # Rollout-policy entropy for collapse monitoring, from the already
            # materialized old logits (no extra forward).
            if entropy is None:
                entropy = self._old_policy_entropy(old_logits, rmask)
        else:
            tp = time.monotonic()
            entropy = torch.zeros(responses.shape, dtype=torch.float32, device=self.actor_device)
            old_logp = self._old_logp_microbatch(input_ids, full_mask, positions, responses, rmask,
                                                 entropy_out=entropy)
            if self.cfg.kl_reference == "init":
                ref_logp = self._reference_logp(input_ids, full_mask, positions, responses, rmask)
            phase["actor_old_logp_sec"] = elapsed_phase(tp)
            phase["credit_cache_sec"] = 0.0
            token_adv = advantages.to(self.actor_device)[:, None].expand_as(rmask)
            # Build a compatible cache with zero direction; GRPO uses sequence advantages.
            from .core import Credit
            with torch.no_grad():
                cache = type("Cache", (), {"old_logp": old_logp, "credit": Credit(token_adv, torch.zeros_like(token_adv), torch.ones_like(token_adv))})
        # old_logits is only needed while constructing old_logp/credit.  Keeping
        # the full [B,T,V] tensor alive would consume ~37 GiB at 2048 tokens and
        # leave no room for the per-microbatch Actor update forward.
        del old_logits
        if self.actor_device.type == "cuda":
            torch.cuda.empty_cache()
        tp = time.monotonic()
        self.actor.train()
        # Group advantages, attribution and old/reference probabilities remain
        # frozen across all configured optimizer minibatches and policy epochs.
        micro = max(1, int(self.cfg.microbatch_responses))
        minibatch = self.cfg.optimizer_minibatch_responses
        loss_value = 0.0
        optimizer_steps = 0
        grad_norm = 0.0
        check_response_tokens(input_ids, full_mask, positions, responses, rmask)
        if not torch.isfinite(cache.old_logp[rmask]).all():
            raise ValueError("Sampled tokens must have finite probability in the sampling support")
        if ref_logp is not None and not torch.isfinite(ref_logp[rmask]).all():
            raise ValueError("Initial reference log probabilities must be finite on sampled support")
        for _epoch in range(self.cfg.policy_epochs_per_rollout):
            for batch_start in range(0, B, minibatch):
                batch_end = min(B, batch_start + minibatch)
                self.optimizer.zero_grad(set_to_none=True)
                for start in range(batch_start, batch_end, micro):
                    end = min(batch_end, start + micro)
                    sl = slice(start, end)
                    new_logits = actor_response_logits(
                        self.actor, input_ids[sl], full_mask[sl], positions[sl], rmask[sl],
                        output_mask=self.output_mask)
                    new_logits = self._sampling_logits(new_logits)
                    new_logp = selected_logp_from_logits(new_logits, responses[sl], rmask[sl],
                                                         policy_temperature=self.cfg.temperature)
                    chunk_loss = grpo_policy_loss(new_logp, cache.old_logp[sl],
                                                  cache.credit.advantage[sl], rmask[sl],
                                                  self.cfg.clip_eps)
                    if self.cfg.beta:
                        base_logp = ref_logp[sl] if ref_logp is not None else cache.old_logp[sl]
                        delta = (base_logp.masked_fill(~rmask[sl], 0)
                                 - new_logp.masked_fill(~rmask[sl], 0))
                        kl = torch.expm1(delta) - delta
                        chunk_loss = chunk_loss + self.cfg.beta * (kl.sum(-1) / rmask[sl].sum(-1)).mean()
                    if not torch.isfinite(chunk_loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise ValueError("Non-finite actor loss; optimizer step cancelled")
                    weight = (end - start) / (batch_end - batch_start)
                    (chunk_loss * weight).backward()
                    loss_value += float(chunk_loss.detach()) * (end - start) / B / self.cfg.policy_epochs_per_rollout
                    del chunk_loss, new_logp, new_logits
                    if self.cfg.beta:
                        del delta, kl
                # Never let Adam moments or actor parameters absorb a nonfinite
                # gradient, including one produced by an otherwise finite loss.
                norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise ValueError("Non-finite actor gradients; optimizer step cancelled")
                grad_norm = max(grad_norm, float(norm))
                self.optimizer.step()
                optimizer_steps += 1
        phase["actor_update_sec"] = elapsed_phase(tp)
        self.total_tokens += int(rmask.sum().item())
        self.rollout_index += 1
        reference_kl = 0.0
        if ref_logp is not None:
            delta = (ref_logp - cache.old_logp).masked_fill(~rmask, 0)
            reference_kl = float((torch.expm1(delta) - delta).sum() / rmask.sum())
        metrics = {"rollout": self.rollout_index, "loss": loss_value,
                   "skipped_rollout": False, "reward_count": B,
                   "optimizer_steps": optimizer_steps,
                   "raw_reward_mean": float(raw_rewards.mean()),
                   "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(correction=0)),
                   "response_tokens": int(rmask.sum()), "grad_norm": grad_norm,
                   "degenerate_responses": n_degenerate,
                   "mean_response_tokens": mean_len,
                   "length_penalty_mean": float(length_penalty.mean()),
                   "short_penalty_mean": float(short_penalty.mean()),
                   "long_penalty_mean": float(long_penalty.mean()),
                   "empty_responses": int(empty.sum()), "newline_degenerate_responses": int(repeated.sum()),
                   "truncated_responses": int(truncated.sum()),
                   "response_entropy": float((entropy.sum() / rmask.sum().clamp_min(1)).item())
                   if entropy is not None else 0.0,
                   "kl_to_init": reference_kl,
                   "group_sigma_mean": float(scales.mean()), "group_sigma_min": float(scales.min()),
                   "group_sigma_max": float(scales.max()),
                   "elapsed_sec": time.monotonic() - t0,
                   "gpu_hours": (time.monotonic() - self._started) * self.cfg.allocated_gpu_count / 3600}
        metrics.update(selection_stats)
        if soft:
            metrics["length_reward_sigma0"] = self.cfg.length_reward_sigma0
        metrics.update({f"phase_{k}": v for k, v in phase.items()})
        if self.cfg.method == "vpo_rm":
            w = cache.credit.weight.float().masked_fill(~rmask, 0)
            lengths = rmask.sum(-1).float()
            ess = lengths.square() / w.square().sum(-1).clamp_min(1e-12)
            metrics["credit_ess_ratio"] = float((ess / lengths.clamp_min(1)).mean())
            # Per-rollout weight-distribution summary plus a full histogram in
            # credit_stats.jsonl (fixed 0.05-wide bins over [0, 3]) with the
            # per-response adaptive taus — the analysis artifacts for the
            # lambda-band dose-response study.
            # Distribution stats on a CPU copy: aten::histogram with tensor
            # bins has no CUDA kernel in this build, and the transfer of
            # ~B*T floats is negligible.
            wv = w[rmask].cpu()
            qs = torch.quantile(wv, torch.tensor([.05, .25, .5, .75, .95], device=wv.device))
            metrics["credit_w_mean"] = float(wv.mean())
            metrics["credit_w_std"] = float(wv.std(unbiased=False))
            metrics["credit_w_p05"] = float(qs[0])
            metrics["credit_w_p50"] = float(qs[2])
            metrics["credit_w_p95"] = float(qs[4])
            metrics["credit_w_max"] = float(wv.max())
            tau_used = cache.credit.tau_used
            metrics["credit_tau_adaptive_mean"] = float(tau_used.mean())
            metrics["credit_lambda_binding"] = float(
                (tau_used > self.cfg.tau).float().mean())
            hist = torch.histogram(wv, bins=torch.linspace(0, 3.0, 61, device=wv.device))
            with (Path(self.cfg.output_dir) / "credit_stats.jsonl").open("a") as f:
                f.write(json.dumps({"rollout": self.rollout_index,
                                    "tau": [round(float(t), 4) for t in tau_used],
                                    "w_hist": [int(c) for c in hist.hist],
                                    "bin_width": 0.05}) + "\n")
            # Per-token credit dump (~0.5 MB/step) for case studies: token ids
            # live in rollout-N-tokens.json; this adds the direction d_t and
            # the final weight w_t behind every one of them.
            torch.save({"w": cache.credit.weight.half().cpu(),
                        "d": cache.credit.direction.half().cpu(),
                        "tau": cache.credit.tau_used.cpu()},
                       Path(self.cfg.output_dir) / f"rollout-{self.rollout_index}-credit.pt")
            # Raw (pre-standardization) utility spread: the p9c scale-mismatch
            # gauge.  Healthy Plan B operation shows ESS well below one while
            # this stays near the p9c value; a return of ESS ~0.998 with this
            # gauge near zero means the direction signal itself vanished.
            seq_adv = cache.credit.advantage.sum(-1) / lengths.clamp_min(1)
            raw = (seq_adv[:, None] * cache.credit.direction.float()).masked_fill(~rmask, 0)
            raw_mean = raw.sum(-1) / lengths.clamp_min(1)
            raw_var = (raw.square().sum(-1) / lengths.clamp_min(1) - raw_mean.square()).clamp_min(0)
            metrics["credit_raw_utility_std"] = float(raw_var.sqrt().mean())
        self._log(metrics)
        if self.rollout_index % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint(self.rollout_index)
        return metrics

    def _log(self, metrics):
        with self.log_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")

    def save_checkpoint(self, step: int | None = None):
        step = self.rollout_index if step is None else step
        path = Path(self.cfg.output_dir) / f"checkpoint-{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(path)
        self.actor_tokenizer.save_pretrained(path)
        state = {"optimizer": self.optimizer.state_dict(), "step": step,
                 "config": asdict(self.cfg), "rng_state": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(state, path / "trainer_state.pt")
        manifest = {"resolved_config": asdict(self.cfg),
            "model": self.cfg.model_name, "reward_model": self.cfg.reward_model_name,
            "step": step, "total_tokens": self.total_tokens,
            "sampling": self.sampling_manifest(), "resume_supported": False,
            "gpu_hours_definition": "elapsed_since_trainer_initialization_times_allocated_gpu_count"}
        if self.data_split is not None:
            manifest["data_split"] = self.data_split
        try:
            import transformers
            manifest["software_versions"] = {"torch": torch.__version__,
                                               "transformers": transformers.__version__}
        except ImportError:
            manifest["software_versions"] = {"torch": torch.__version__}
        (path / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        return path

    def train(self, prompts: Sequence[str]):
        prompts = list(prompts)
        if not prompts:
            raise ValueError("No prompts supplied")
        self.prepare_length_reward(prompts)
        for _ in range(self.cfg.rollout_iterations):
            k = self.cfg.prompts_per_rollout
            start = (self.rollout_index * k) % len(prompts)
            chosen = [prompts[(start + i) % len(prompts)] for i in range(k)]
            self.train_rollout(chosen)
        return self.save_checkpoint(self.rollout_index)


def load_prompt_dataset(name: str, split: str = "train_prefs", field: str = "prompt",
                        validation_size: int = 2000, dataset_path: str | None = None,
                        exclude_benchmarks: bool = False, benchmark_paths=None):
    # Some inference-oriented images expose a namespace-only ``datasets``
    # package.  Local parquet input should remain usable without requiring the
    # full Hugging Face datasets stack (and without network access).
    if dataset_path:
        try:
            from datasets import load_dataset
        except (ImportError, AttributeError):
            load_dataset = None
        if load_dataset is not None:
            ds = load_dataset("parquet", data_files=dataset_path, split="train")
            values = (row[field] for row in ds)
        else:
            import pyarrow.parquet as pq
            values = (x for x in pq.read_table(dataset_path, columns=[field]).column(field).to_pylist())
    else:
        from datasets import load_dataset
        ds = load_dataset(name, split=split)
        values = (row[field] for row in ds)
    exclusion = None
    if exclude_benchmarks:
        from .data import exclude_benchmark_prompts
        values, exclusion = exclude_benchmark_prompts(values, benchmark_paths=benchmark_paths)
    train, valid, hashes = split_prompts(values, validation_size)
    if exclusion is not None:
        hashes["benchmark_exclusion"] = exclusion
    return train, valid, hashes


def main(argv=None):
    from .length_reward_cli import add_length_reward_args, length_reward_config_kwargs

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TrainerConfig.model_name); p.add_argument("--rm", default=TrainerConfig.reward_model_name)
    p.add_argument("--output-dir", default="runs/skywork"); p.add_argument("--method", choices=["grpo", "vpo_rm"], default="vpo_rm")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--prompts-file")
    p.add_argument("--dataset-path", help="local train_prefs parquet (avoids network download)")
    p.add_argument("--init-adapter", default="", help="SFT LoRA adapter used to initialize the actor")
    add_length_reward_args(p)
    args = p.parse_args(argv)
    cfg = TrainerConfig(model_name=args.model, reward_model_name=args.rm, output_dir=args.output_dir,
                        method=args.method, smoke=args.smoke, init_adapter=args.init_adapter,
                        **length_reward_config_kwargs(args))
    if args.prompts_file:
        prompts = [x.strip() for x in Path(args.prompts_file).read_text().splitlines() if x.strip()]
        from .data import exclude_benchmark_prompts
        prompts, exclusion = exclude_benchmark_prompts(prompts)
        validation_size = min(cfg.validation_size, max(0, len(set(map(normalize_prompt, prompts))) - 1))
        prompts, _, split_meta = split_prompts(prompts, validation_size=validation_size)
        split_meta["benchmark_exclusion"] = exclusion
    else:
        prompts, _, split_meta = load_prompt_dataset("HuggingFaceH4/ultrafeedback_binarized",
                                            dataset_path=args.dataset_path, exclude_benchmarks=True)
    trainer = VPOTrainer.from_pretrained(cfg)
    prompts = trainer.filter_prompts(prompts)
    trainer.data_split = dict(split_meta, filtered_train_prompts=len(prompts),
                              dropped_train_prompts=trainer.filtered_prompt_count)
    (Path(cfg.output_dir) / "data_split.json").write_text(json.dumps(trainer.data_split, indent=2))
    trainer.train(prompts)


if __name__ == "__main__":
    main()
