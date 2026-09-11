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
from .core import grpo_policy_loss
from .integration import (actor_response_logits, build_credit_cache,
                          response_reward_gradients, selected_logp_from_logits)
from .reward import LastTokenReward


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


@dataclass
class TrainerConfig:
    model_name: str = "Qwen/Qwen3-14B"
    reward_model_name: str = "Skywork/Skywork-Reward-V2-Qwen3-8B"
    output_dir: str = "runs/skywork"
    actor_device: str = "cuda:0"
    reward_device: str = "cuda:1"
    seed: int = 42
    group_size: int = 8
    prompts_per_rollout: int = 8
    rollout_iterations: int = 500
    policy_epochs_per_rollout: int = 1
    optimizer_minibatch_responses: int = 64
    microbatch_responses: int = 1
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
    min_response_tokens: int = 8
    degenerate_penalty: float = 1.0
    init_adapter: str = ""
    kl_reference: str = "rollout"
    length_penalty_slope: float = 0.0
    length_penalty_anchor: int = 600
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
        if c.method not in {"grpo", "vpo_rm"}:
            raise ValueError("method must be grpo or vpo_rm")
        if c.kl_reference not in {"rollout", "init"}:
            raise ValueError("kl_reference must be rollout or init")
        if c.kl_reference == "init" and not c.init_adapter:
            raise ValueError("kl_reference=init requires init_adapter")
        if c.smoke:
            c.rollout_iterations = min(c.rollout_iterations, c.max_smoke_rollouts)
            c.prompts_per_rollout = min(c.prompts_per_rollout, c.max_smoke_prompts)
            c.group_size = min(c.group_size, c.max_smoke_group_size)
            c.max_response_tokens = min(c.max_response_tokens, 32)
        if c.group_size < 2 or c.prompts_per_rollout < 1:
            raise ValueError("group_size must be >=2 and prompts_per_rollout positive")
        if c.policy_epochs_per_rollout < 1 or c.optimizer_minibatch_responses < 1:
            raise ValueError("policy_epochs_per_rollout and optimizer_minibatch_responses must be positive")
        return c


class VPOTrainer:
    """Minimal production trainer; actor and RM are always placed on separate devices."""
    def __init__(self, actor, actor_tokenizer, reward, reward_tokenizer, config: TrainerConfig):
        self.cfg = config.resolved()
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
                generated = self.actor.generate(
                    **sub, do_sample=True, temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p, top_k=self.cfg.top_k,
                    max_new_tokens=self.cfg.max_response_tokens,
                    min_new_tokens=self.cfg.min_response_tokens,
                    num_return_sequences=n,
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
        rmask = torch.ones_like(responses, dtype=torch.bool)
        eos = self.actor_tokenizer.eos_token_id
        pad = self.actor_tokenizer.pad_token_id
        stop_id = eos if eos is not None else pad
        if stop_id is not None:
            seen = responses.eq(stop_id).cumsum(-1)
            rmask = seen.eq(0) | responses.eq(stop_id)
            # A malformed generation containing only padding still has one token for
            # stable credit pooling; this is filtered by max-response policy upstream.
            rmask[:, 0] = True
        full_mask = torch.cat([prompt_mask, rmask.to(prompt_mask.dtype)], dim=1)
        positions = torch.arange(prompt_width, out.shape[1], device=self.actor_device).expand(out.shape[0], -1)
        return out, full_mask, positions, responses, rmask, rendered_prompts

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
        switched = adapter is not None
        if switched:
            self.actor.set_adapter(adapter)
        try:
            for start in range(0, B, micro):
                end = min(B, start + micro)
                logits = actor_response_logits(self.actor, input_ids[start:end], attention_mask[start:end],
                                               positions[start:end], response_mask[start:end],
                                               output_mask=self.output_mask)
                z = logits.float()
                safe = responses[start:end].masked_fill(~response_mask[start:end], 0)
                result[start:end] = z.gather(-1, safe[..., None]).squeeze(-1) - z.logsumexp(-1)
                if entropy_out is not None:
                    logz = z.logsumexp(-1, keepdim=True)
                    lp = z - logz
                    p = lp.exp()
                    # p==0 at -inf-masked vocab rows; 0*(-inf) would be NaN.
                    entropy_out[start:end] = -torch.where(
                        p > 0, p * lp, torch.zeros_like(p)).sum(-1).masked_fill(
                        ~response_mask[start:end], 0)
                    del p, lp, logz
                del logits, z
        finally:
            if switched:
                self.actor.set_adapter("default")
        return result

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
        # Release gradients from the preceding optimizer step before allocating
        # rollout logits/KV caches (critical for long 64-response runs).
        self.optimizer.zero_grad(set_to_none=True)
        if self.actor_device.type == "cuda":
            torch.cuda.empty_cache()
        tp = time.monotonic()
        input_ids, full_mask, positions, responses, rmask, rendered = self.rollout(prompts)
        phase["generation_sec"] = elapsed_phase(tp)
        old_logits = None
        if self.cfg.method == "vpo_rm":
            tp = time.monotonic()
            with torch.no_grad():
                old_logits = actor_response_logits(self.actor, input_ids, full_mask, positions, rmask,
                                                   output_mask=self.output_mask)
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
        # Anti-reward-hacking layer 2: under-length responses (below
        # min_response_tokens, unreachable in normal operation because the
        # vLLM server enforces min_tokens) are floored below their group
        # minimum so they can never earn positive advantage.  p9d4 showed the
        # Skywork RM scores a bare stop token 7.7 — above real answers.
        lengths_dev = rmask.sum(-1).to(rewards.device)
        # Calibrated length debias (p9g): the RM pays ~3.4 points per 1000
        # tokens of shortness on real answers (regression over the healthy
        # phases of p9d4/p9e/p9f, n=140).  Subtract 1.5x that slope below the
        # 600-token anchor so the short-answer slide stops paying; above the
        # anchor nothing changes.  Truncated (overlong) responses are floored
        # below their group minimum alongside the under-length guard.
        length_penalty = torch.zeros_like(rewards)
        if self.cfg.length_penalty_slope > 0:
            shortfall = (self.cfg.length_penalty_anchor - lengths_dev).clamp_min(0)
            length_penalty = self.cfg.length_penalty_slope * shortfall.float()
            rewards = rewards - length_penalty
        truncated = lengths_dev >= (self.cfg.max_response_tokens - 1)
        rewards, n_degenerate = guard_degenerate_rewards(
            rewards, lengths_dev, group_ids,
            self.cfg.min_response_tokens, self.cfg.degenerate_penalty,
            also_floor=truncated if self.cfg.length_penalty_slope > 0 else None)
        if n_degenerate:
            print(f"[reward-guard] {n_degenerate}/{B} responses under "
                  f"{self.cfg.min_response_tokens} tokens floored below group min", flush=True)
        mean_len = float(lengths_dev.float().mean())
        if mean_len < self.cfg.min_response_tokens:
            print(f"[reward-guard] WARNING mean response length {mean_len:.1f} below "
                  f"{self.cfg.min_response_tokens} — degenerate policy suspected", flush=True)
        advantages, scales = group_advantages(rewards, group_ids)
        ref_logp = None
        entropy = None
        if self.cfg.method == "vpo_rm":
            tp = time.monotonic()
            rm_weight = self.reward.get_input_embeddings().weight.detach().to(self.actor_device)
            cache = build_credit_cache(old_logits, responses, grads.to(self.actor_device), rm_weight,
                                       advantages.to(self.actor_device), scales.to(self.actor_device),
                                       rmask, self.cfg.tau, credit_lambda=self.cfg.credit_lambda,
                                       token_chunk_size=self.cfg.token_chunk_size,
                                       vocab_chunk_size=self.cfg.vocab_chunk_size)
            phase["credit_cache_sec"] = elapsed_phase(tp)
            if self.cfg.kl_reference == "init":
                tp = time.monotonic()
                ref_logp = self._old_logp_microbatch(input_ids, full_mask, positions,
                                                     responses, rmask, adapter="ref")
                phase["ref_logp_sec"] = elapsed_phase(tp)
            # Rollout-policy entropy for collapse monitoring, from the already
            # materialized old logits (no extra forward).
            with torch.no_grad():
                entropy = torch.zeros_like(cache.credit.direction)
                rows, times = rmask.bool().nonzero(as_tuple=True)
                chunk = max(1, int(self.cfg.token_chunk_size)) * 4
                for lo in range(0, rows.numel(), chunk):
                    r, t = rows[lo:lo + chunk], times[lo:lo + chunk]
                    z = old_logits[r, t].float()
                    logz = z.logsumexp(-1, keepdim=True)
                    lp = z - logz
                    p = lp.exp()
                    entropy[r, t] = -torch.where(
                        p > 0, p * lp, torch.zeros_like(p)).sum(-1)
                    del p, lp, logz
        else:
            tp = time.monotonic()
            entropy = torch.zeros(responses.shape, dtype=torch.float32, device=self.actor_device)
            old_logp = self._old_logp_microbatch(input_ids, full_mask, positions, responses, rmask,
                                                 entropy_out=entropy)
            if self.cfg.kl_reference == "init":
                ref_logp = self._old_logp_microbatch(input_ids, full_mask, positions,
                                                     responses, rmask, adapter="ref")
            phase["actor_old_logp_sec"] = elapsed_phase(tp)
            phase["credit_cache_sec"] = 0.0
            token_adv = advantages.to(self.actor_device)[:, None].expand_as(rmask)
            # Build a compatible cache with zero direction; GRPO uses sequence advantages.
            from .core import Credit
            with torch.no_grad():
                cache = type("Cache", (), {"old_logp": old_logp, "credit": Credit(token_adv, torch.zeros_like(token_adv), torch.ones_like(token_adv))})
        # old_logits is only needed while constructing old_logp/credit.  Keeping
        # the full [B,T,V] tensor alive would consume ~33 GiB at 2048 tokens and
        # leave no room for the per-microbatch Actor update forward.
        del old_logits
        if self.actor_device.type == "cuda":
            torch.cuda.empty_cache()
        tp = time.monotonic()
        self.actor.train(); self.optimizer.zero_grad(set_to_none=True)
        # Accumulate gradients over response microbatches while keeping the complete
        # prompt group in the cached credit.  Weighting by chunk size reproduces the
        # configured mean-over-responses reduction exactly, then performs one update.
        micro = max(1, int(self.cfg.microbatch_responses))
        loss_value = 0.0
        check_response_tokens(input_ids, full_mask, positions, responses, rmask)
        for start in range(0, B, micro):
            end = min(B, start + micro)
            sl = slice(start, end)
            # One teacher-forced forward feeds both the clipped policy loss and the
            # KL estimator.  A separate KL forward recomputed identical logits
            # (dropout is zero) while keeping a second activation graph alive.
            new_logits = actor_response_logits(
                self.actor, input_ids[sl], full_mask[sl], positions[sl], rmask[sl],
                output_mask=self.output_mask)
            new_logp = selected_logp_from_logits(new_logits, responses[sl], rmask[sl])
            chunk_loss = grpo_policy_loss(new_logp, cache.old_logp[sl],
                                          cache.credit.advantage[sl], rmask[sl],
                                          self.cfg.clip_eps)
            if self.cfg.beta:
                # KL reference: the rollout policy (per-step stabilizer, legacy)
                # or the frozen SFT initialization (p9g anchor against drift
                # and degenerate modes).  The k3 estimator stays >= 0.
                base_logp = (ref_logp[sl] if ref_logp is not None
                             else cache.old_logp[sl].detach())
                delta = base_logp - new_logp
                kl = (delta.exp() - delta - 1).masked_fill(~rmask[sl], 0)
                chunk_loss = chunk_loss + self.cfg.beta * (kl.sum(-1) / rmask[sl].sum(-1)).mean()
            weight = (end - start) / B
            (chunk_loss * weight).backward()
            loss_value += float(chunk_loss.detach()) * weight
            del chunk_loss, new_logp
            if self.cfg.beta:
                del delta, kl
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm))
        self.optimizer.step()
        phase["actor_update_sec"] = elapsed_phase(tp)
        self.total_tokens += int(rmask.sum().item())
        self.rollout_index += 1
        metrics = {"rollout": self.rollout_index, "loss": loss_value,
                   "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(correction=0)),
                   "response_tokens": int(rmask.sum()), "grad_norm": grad_norm,
                   "degenerate_responses": n_degenerate,
                   "mean_response_tokens": mean_len,
                   "length_penalty_mean": float(length_penalty.mean()),
                   "truncated_responses": int(truncated.sum()),
                   "response_entropy": float((entropy.sum() / rmask.sum().clamp_min(1)).item())
                   if entropy is not None else 0.0,
                   "kl_to_init": float((ref_logp - cache.old_logp).masked_fill(~rmask, 0).sum()
                                       / rmask.sum().clamp_min(1)) if ref_logp is not None else 0.0,
                   "group_sigma_mean": float(scales.mean()), "group_sigma_min": float(scales.min()),
                   "group_sigma_max": float(scales.max()),
                   "elapsed_sec": time.monotonic() - t0,
                   "gpu_hours": (time.monotonic() - self._started) * 2 / 3600}
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
            wv = w[rmask]
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
            "step": step, "total_tokens": self.total_tokens}
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
        for _ in range(self.cfg.rollout_iterations):
            k = self.cfg.prompts_per_rollout
            start = (self.rollout_index * k) % len(prompts)
            chosen = [prompts[(start + i) % len(prompts)] for i in range(k)]
            self.train_rollout(chosen)
        return self.save_checkpoint(self.rollout_index)


def load_prompt_dataset(name: str, split: str = "train_prefs", field: str = "prompt",
                        validation_size: int = 2000, dataset_path: str | None = None):
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
    train, valid, hashes = split_prompts(values, validation_size)
    return train, valid, hashes


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TrainerConfig.model_name); p.add_argument("--rm", default=TrainerConfig.reward_model_name)
    p.add_argument("--output-dir", default="runs/skywork"); p.add_argument("--method", choices=["grpo", "vpo_rm"], default="vpo_rm")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--prompts-file")
    p.add_argument("--dataset-path", help="local train_prefs parquet (avoids network download)")
    args = p.parse_args(argv)
    cfg = TrainerConfig(model_name=args.model, reward_model_name=args.rm, output_dir=args.output_dir,
                        method=args.method, smoke=args.smoke)
    if args.prompts_file:
        prompts = [x.strip() for x in Path(args.prompts_file).read_text().splitlines() if x.strip()]
    else:
        prompts, _, _ = load_prompt_dataset("HuggingFaceH4/ultrafeedback_binarized",
                                            dataset_path=args.dataset_path)
    trainer = VPOTrainer.from_pretrained(cfg)
    trainer.train(prompts)


if __name__ == "__main__":
    main()
