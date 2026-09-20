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
import shutil
import time

import torch

from .attribution import P2T_APPROXIMATION, null_token_attribution
from .autopush import AutoPusher
from .data import load_prompt_dataset, normalize_prompt, split_prompts
from .length_reward import (calibrate_reward_scale, guard_degenerate_rewards,
                            response_degeneracy, soft_length_penalties)
from .loss import grpo_policy_loss, kl_from_logp
from .mapping import REWARD_INPUT_PROTOCOL, canonical_reward_input
from .policy import (encode_prompts, render_chat_prompt, response_logits,
                     rollout_logp_microbatch, selected_logp_from_logits,
                     sampling_logits, stop_token_ids_for)
from .reward import P2T_ALPHA_SHORT_COT, P2T_OMEGA, group_advantages, p2t_credit
from .rollout import select_training_rollout, validate_response_termination
from .rm import LastTokenReward, build_rm_batch, score_responses
from .tokens import check_tokenizer_identity, load_actor_tokenizer, shared_output_mask
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
    policy_head_dtype: str = "float32"
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
    gate_initial_hf_clip: bool = True  # abort if the first re-forward disagrees
    max_unmapped_content_fraction: float = .25  # the project's startup bound
    checkpoint_interval: int = 0  # 0 -> only the final adapter
    keep_adapters: int = 2  # most recent per-step vLLM adapters retained on disk
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
        if c.min_response_tokens:
            # The parent's soft mode forbids a minimum length, and a nonzero one
            # would additionally apply the legacy degeneracy floor that the
            # sibling arms never apply.
            raise ValueError("the soft length protocol requires min_response_tokens=0")
        if c.push_every < 0:
            raise ValueError("push_every must be nonnegative")
        if c.policy_head_dtype not in {"native", "float32"}:
            raise ValueError("policy_head_dtype must be native or float32")
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
        self.optimizer = torch.optim.AdamW(
            (p for p in self.actor.parameters() if p.requires_grad),
            lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        self.rollout_index = 0
        self.total_tokens = 0
        self.adapter_id = 0
        self.sigma0 = self.cfg.sigma0
        self.filtered_prompt_count = 0
        self.init_adapter_sha256 = ""
        self._adapter_identity = {}
        self.started = time.monotonic()
        self.pusher = AutoPusher(enabled=bool(self.cfg.push_every), every=self.cfg.push_every,
                                 remote=self.cfg.push_remote, branch=self.cfg.push_branch,
                                 report_dir=self.report_dir, repo_root=Path(__file__).resolve().parents[1])

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
        if c.policy_head_dtype == "float32":
            from .policy_precision import enable_fp32_output_head
            enable_fp32_output_head(trainer.actor)
        trainer.reward.to(trainer.reward_device).eval()
        for parameter in trainer.reward.parameters():
            parameter.requires_grad_(False)
        trainer._refuse_nonzero_dropout()
        if c.init_adapter:
            trainer._verify_init_adapter()
        return trainer

    def _verify_init_adapter(self):
        """Bind the SFT initialization to the actor it claims to fine-tune.

        PEFT will happily mount an adapter on the wrong base and the run would
        look healthy while training something nobody asked for.  The project
        records the adapter's weight hash in every manifest; we do the same, so
        the P2T arm can be shown to have started from the same bytes as its
        siblings.
        """
        import hashlib
        directory = Path(self.cfg.init_adapter)
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"init adapter has no adapter_config.json: {directory}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"init adapter has no adapter_model.safetensors: {directory}")
        declared = json.loads(config_path.read_text()).get("base_model_name_or_path") or ""
        if declared and Path(declared).name != Path(self.cfg.model_name).name:
            raise ValueError(
                f"init adapter was fitted to {declared!r}, but the actor is "
                f"{self.cfg.model_name!r}")
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        self.init_adapter_sha256 = digest
        self._adapter_identity = {"init_adapter": str(directory),
                                  "init_adapter_base": declared,
                                  "init_adapter_sha256": digest}

    def _refuse_nonzero_dropout(self):
        """Sampler and trainer probabilities must match, so no dropout anywhere."""
        if any(isinstance(module, torch.nn.Dropout) and module.p > 0
               for module in self.actor.modules()):
            raise ValueError("Actor dropout must be zero to match generation and training probabilities")
        for name in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob",
                     "attention_probs_dropout_prob"):
            if float(getattr(getattr(self.actor, "config", None), name, 0.0) or 0.0) != 0:
                raise ValueError("Actor dropout must be zero to match generation and training probabilities")

    def filter_prompts(self, prompts):
        """Apply the experiment's post-template prompt length policy.

        Both chat templates are measured before training so a prompt cannot be
        silently truncated for the actor while a different prefix is fed to the
        reward model.  Mirrors ``VPOTrainer.filter_prompts``; the parent applies
        it before training, and the surviving list is what every arm shares.
        """
        kept = []
        for prompt in prompts:
            actor_text = render_chat_prompt(self.actor_tokenizer, prompt, tokenize=False)
            actor_ids = self.actor_tokenizer(actor_text, add_special_tokens=False)["input_ids"]
            reward_ids = canonical_reward_input(self.reward_tokenizer, str(prompt), "")
            if (len(actor_ids) <= self.cfg.max_prompt_tokens
                    and len(reward_ids) <= self.cfg.max_prompt_tokens):
                kept.append(prompt)
        self.filtered_prompt_count = len(prompts) - len(kept)
        if self.filtered_prompt_count:
            self._log({"event": "prompt_filter", "input_prompts": len(prompts),
                       "kept_prompts": len(kept), "dropped_prompts": self.filtered_prompt_count,
                       "max_prompt_tokens": self.cfg.max_prompt_tokens})
        return kept

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
        # A distinct path per step, matching the parent project.  Reusing one
        # path would let the server keep serving a cached adapter and quietly
        # turn the rollout off-policy.
        adapter_path = self.adapter_root / f"step-{self.rollout_index}"
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
        self._prune_adapters()
        return rollout, summary

    def _prune_adapters(self):
        """Keep only the newest adapters: each one is ~0.5 GiB on disk."""
        paths = sorted(self.adapter_root.glob("step-*"),
                       key=lambda path: int(path.name.split("-")[1]))
        for stale in paths[:max(0, len(paths) - self.cfg.keep_adapters)]:
            shutil.rmtree(stale, ignore_errors=True)

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
        if count < 128 and self.cfg.rollout_iterations > 20:
            print("[sigma0] WARNING: this run calibrated its own length-reward scale on "
                  f"{count} prompts, but the formal protocol uses 128 and every arm must "
                  "share one sigma0. The length window here will differ from the GRPO and "
                  "VPO-RM arms.", flush=True)
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
        # The project's startup gate bounds how much *real text* failed to map.
        # If a large share of content tokens carry no reward-model gradient, the
        # attribution is mostly structural zeros and Eq. (3) is being driven by
        # positions the reward model never read.
        if (self.rollout_index == 0
                and stats["rm_unmapped_content_fraction"] > self.cfg.max_unmapped_content_fraction):
            raise ValueError(
                f"Unmapped content tokens are {stats['rm_unmapped_content_fraction']:.3f} of "
                f"the response, above the {self.cfg.max_unmapped_content_fraction} bound; "
                f"the actor-to-RM token mapping is not holding")
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
        # `self.rollout` returns (rollout, generation_summary); the selector wants
        # the tuple alone, but the sampling provenance still has to reach the log.
        sampling_summary = {}

        def rollout_for_selection(batch):
            rollout, summary = self.rollout(batch)
            if not sampling_summary:
                sampling_summary.update(summary)
            return rollout

        rollout, prompts, selection = select_training_rollout(
            rollout_for_selection, prompts, group_size=self.cfg.group_size,
            pad_token_id=self.actor_tokenizer.pad_token_id,
            flag_degenerate=self._degeneracy, device=self.actor_device,
            stop_token_ids=self.stop_token_ids,
            max_response_tokens=self.cfg.max_response_tokens)
        self._write_rollout_artifacts(prompts, rollout)
        if rollout is None:
            # No response reaches the loss, including KL, Adam moments or decay.
            self.rollout_index += 1
            metrics = {"rollout": self.rollout_index, "skipped_rollout": True,
                       "optimizer_steps": 0, "response_tokens": 0, "reward_count": 0,
                       "loss": 0.0, "elapsed_sec": time.monotonic() - started,
                       "gpu_hours": (time.monotonic() - self.started)
                       * (2 + self.cfg.vllm_tensor_parallel_size) / 3600,
                       "phase_generation_sec": clock - started, **selection}
            self._log(metrics)
            self.pusher.maybe_push(self.rollout_index, metrics)
            return metrics
        input_ids, full_mask, positions, responses, rmask, _, finish_reasons, rollout_logprobs = rollout
        # Every mask consumer below writes ``~mask``; a long tensor there is a
        # bitwise complement, not a boolean negation, and the failure is silent
        # until an unrelated masked_fill raises.  Normalise once.
        rmask = rmask.bool()
        self._validate_rollout((input_ids, full_mask, positions, responses, rmask, _,
                                finish_reasons, rollout_logprobs), len(prompts))

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
        # The flagged set is what was floored above; report that count rather than
        # the guard's, which is a no-op once min_length is at its soft-mode floor.
        n_degenerate = int(flags.sum())
        shaped, _ = guard_degenerate_rewards(
            shaped, lengths, group_ids, max(1, self.cfg.min_response_tokens),
            self.cfg.degenerate_penalty)
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
        # The sampler and the trainer forward the same weights under the same
        # FP32 head, so this delta should be numerically zero.  Whatever it is,
        # the importance weights would otherwise absorb it silently.
        logp_delta = (old_logp - rollout_logprobs)[rmask]
        is_values = importance[rmask]
        is_quantiles = torch.quantile(is_values, torch.tensor([.01, .5, .99], device=is_values.device))
        error_quantiles = torch.quantile(logp_delta.abs(), torch.tensor([.5, .99],
                                                                       device=logp_delta.device))
        # The project's startup gate reads this exact vocabulary, so a run that
        # omits a key cannot be validated by the same tooling as its siblings.
        probability_metrics = {
            "rollout_logp_abs_error_mean": float(logp_delta.abs().mean()),
            "rollout_logp_abs_error_p50": float(error_quantiles[0]),
            "rollout_logp_abs_error_p99": float(error_quantiles[1]),
            "rollout_logp_abs_error_max": float(logp_delta.abs().max()),
            "rollout_is_min": float(is_values.min()),
            "rollout_is_mean": float(is_values.mean()),
            "rollout_is_p01": float(is_quantiles[0]),
            "rollout_is_p50": float(is_quantiles[1]),
            "rollout_is_p99": float(is_quantiles[2]),
            "rollout_is_max": float(is_values.max()),
            "rollout_is_ess_ratio": float(
                is_values.sum().square() / (is_values.numel() * is_values.square().sum())),
            "rollout_direct_ratio_clip_fraction": float(
                ((importance < 1 - self.cfg.clip_eps)
                 | (importance > 1 + self.cfg.clip_eps))[rmask].float().mean()),
        }

        clock = phase("actor_update_sec", clock)
        self.actor.train()
        minibatch = self.cfg.optimizer_minibatch_responses
        micro = max(1, self.cfg.microbatch_responses)
        loss_value, optimizer_steps, grad_norm = 0.0, 0, 0.0
        initial_delta, initial_clip_fraction = None, 0.0
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
                if initial_delta is None:
                    # The first trainable forward must reproduce the cached old
                    # policy log-probabilities at theta = theta_old.  Checked
                    # here, before backward, so a protocol mismatch aborts the
                    # run instead of updating once under the broken protocol and
                    # reporting it afterwards.
                    initial_delta = (new_logp.detach() - old_logp[sl])[rmask[sl]]
                    initial_clip_fraction = float(
                        ((initial_delta < math.log1p(-self.cfg.clip_eps))
                         | (initial_delta > math.log1p(self.cfg.clip_eps))).float().mean())
                    if (self.cfg.gate_initial_hf_clip and self.rollout_index == 0
                            and initial_clip_fraction):
                        raise ValueError(
                            f"First re-forward disagrees with the sampler "
                            f"(max |delta| {float(initial_delta.abs().max()):.3e}, "
                            f"clip fraction {initial_clip_fraction:.3f}); the sampler "
                            f"and trainer probability protocols differ")
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
            # Same estimator and weighting as the project's reference-KL metric.
            delta = (ref_logp - old_logp).masked_fill(~rmask, 0)
            kl_values = (torch.expm1(delta) - delta) * importance.detach()
            kl_to_init = float(kl_values.masked_fill(~rmask, 0).sum() / rmask.sum())
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
        metrics.update(selection)
        metrics.update(self._adapter_identity)
        metrics.update(probability_metrics)
        # Sampling provenance: temperature, top-p/k, suppressed-token digest,
        # engine timing, adapter id.  The other arms record the equivalent, and
        # its absence would be silent.
        metrics.update({key: value for key, value in sampling_summary.items()
                        if isinstance(value, (int, float, str, bool))})
        if initial_delta is not None:
            metrics["initial_hf_logp_max_abs_error"] = float(initial_delta.abs().max())
            metrics["initial_hf_ratio_clip_fraction"] = initial_clip_fraction
        metrics.update({f"phase_{key}": value for key, value in timings.items()})
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
        """Observation only: nothing here feeds back into the update.

        Eq. (3) adds `alpha * R * (1 + omega * p_i)` to every token, so the plain
        bonus-to-advantage ratio is dominated by a per-response *constant* and
        barely separates a flat attribution softmax from a one-hot one.  The
        varying part is `alpha * omega * R * (p_i - 1/T)`, and that is what
        `p2t_varying_bonus_over_advantage` measures.
        """
        number = mask.sum(-1).float().clamp_min(1)
        weight = credit.weight.to(self.reward_device).masked_fill(~mask, 0)
        share = (weight / number[:, None]).masked_fill(~mask, 0)
        flat = (share.amax(-1) <= 1.0 / number + 1e-3).float().mean()
        onehot = (share.amax(-1) >= 0.9).float().mean()
        # ESS/T = 1/(T * sum p^2): 1 is flat (inert), 1/T is one-hot.  Same
        # direction as the VPO arms' credit ESS, not the inverse.
        ess = 1.0 / (share.square().sum(-1) * number).clamp_min(1e-12)
        valid_attr = attribution.to(self.reward_device)[mask]
        valid_advantage = advantages[:, None].expand_as(mask)[mask]
        valid_bonus = credit.direction.to(self.reward_device)[mask]
        advantage_scale = advantages.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
        # Exactly alpha * omega * R * (share - 1/T): the per-response constant
        # alpha * R * (1 + omega/T) is subtracted off, leaving only the part that
        # can move credit between tokens.  `number` is [B]; it needs a column to
        # broadcast against [B, T] rather than forming a [B, B] outer product.
        constant = (self.cfg.alpha * raw_rewards.to(self.reward_device)[:, None]
                    * (1 + self.cfg.omega / number[:, None])).expand_as(mask)
        varying = valid_bonus - constant[mask]
        # A real flip is A^hat + bonus changing sign, not merely the bonus
        # opposing A^hat: a bonus that opposes but is smaller leaves the
        # response's direction intact.
        nonzero = valid_advantage != 0
        sign_flip = (torch.sign((valid_advantage + valid_bonus)[nonzero])
                     != torch.sign(valid_advantage[nonzero]))
        # Share mass landing on tokens whose attribution is exactly zero, in the
        # same row-major gather order as valid_attr.
        valid_share = share[mask]
        zero_mass = valid_share[valid_attr == 0].sum() / valid_share.sum().clamp_min(1e-12)
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
            "p2t_flat_response_fraction": float(flat),
            "p2t_onehot_response_fraction": float(onehot),
            "p2t_bonus_abs_mean": float(valid_bonus.abs().mean()),
            "p2t_bonus_over_advantage": float(valid_bonus.abs().mean() / advantage_scale),
            "p2t_varying_bonus_over_advantage": float(varying.abs().mean() / advantage_scale),
            "p2t_sign_flip_fraction": float(sign_flip.float().mean()) if nonzero.any() else 0.0,
            "p2t_zero_attribution_share_mass": float(zero_mass),
            # VPO-parity names so the project's existing analysis keeps working.
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
        # A "stop" row must end in a stop token and a "length" row must fill the
        # cap; otherwise the truncation count, the long-length penalty and the
        # degeneracy flags all silently describe the wrong response.
        validate_response_termination(responses, rmask, reasons, self.stop_token_ids,
                                      self.cfg.max_response_tokens)
        lengths = rmask.sum(-1)
        if bool(((lengths < 1) | (lengths > self.cfg.max_response_tokens)).any()):
            raise ValueError("Invalid response length")
        valid_ids = responses[rmask]
        if ((valid_ids < 0) | (valid_ids >= self.output_mask.numel())).any() \
                or not self.output_mask[valid_ids].all():
            raise ValueError("Sampled tokens must belong to the policy output support")
        if not torch.isfinite(logprobs[rmask]).all():
            raise ValueError("Sampler log probabilities must be finite")

    def _write_rollout_artifacts(self, prompts, rollout):
        """Persist the exact prompts and token IDs a step was trained on."""
        step = self.rollout_index + 1
        rows = [] if rollout is None else [row[valid].detach().cpu().tolist()
                                           for row, valid in zip(rollout[3], rollout[4])]
        (self.output_dir / f"rollout-{step}-tokens.json").write_text(json.dumps(rows))
        (self.output_dir / f"rollout-{step}-prompts.json").write_text(json.dumps(list(prompts)))
        if rollout is not None:
            lengths = rollout[4].sum(-1).tolist()
            rows_path = self.output_dir / f"rollout-{step}-rewards.json"
            rows_path.write_text(json.dumps([
                {"length": int(length), "finish_reason": reason}
                for length, reason in zip(lengths, rollout[6])]))

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
            "null_token_id": self.null_token_id, "resume_supported": False,
            **self._adapter_identity},
            indent=2, default=str))
        return path

    # ------------------------------------------------------------------ driver
    def train(self, prompts):
        prompts = list(prompts)
        if not prompts:
            raise ValueError("No prompts supplied")
        if self.cfg.fit_prompt_filter:
            prompts = self.filter_prompts(prompts)
            if not prompts:
                raise ValueError("Every prompt exceeded the token budget")
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


def check_fresh_output(output_dir) -> None:
    """Reject a directory that already holds training artifacts.

    This trainer has no resume protocol, so relaunching into an existing
    directory would interleave two runs' metrics under duplicated rollout
    numbers, overwrite the credit dumps and adapters of the first attempt, and
    leave no marker saying the result is unanalysable.
    """
    path = Path(output_dir)
    if not path.exists():
        return
    markers = ("metrics.jsonl", "length_reward_calibration.json", "data_split.json",
               "vllm-adapters", "train.pid")
    found = [name for name in markers if (path / name).exists()]
    found += [entry.name for entry in path.glob("checkpoint-*")]
    found += [entry.name for entry in path.glob("rollout-*-credit.pt")]
    if found:
        raise FileExistsError(
            f"{path} already contains training artifacts {sorted(found)[:5]}; this trainer "
            f"cannot resume, so choose a fresh --output-dir or delete the old run")


def load_config(path) -> TrainerConfig:
    payload = json.loads(Path(path).read_text())
    # Keys beginning with "_" are documentation for a human reader; JSON has no
    # comment syntax and the shipped configs explain their provenance inline.
    payload = {key: value for key, value in payload.items() if not key.startswith("_")}
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
    check_fresh_output(config.output_dir)
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError("P2T training needs at least two visible GPUs: actor and reward model")
    # `actor_device`/`reward_device` are indices into CUDA_VISIBLE_DEVICES, while
    # `vllm_gpus` names physical cards.  If the two disagree the trainer would
    # run on one set of GPUs and generation on another, possibly another job's.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    expected = [str(index) for index in range(2 + config.vllm_tensor_parallel_size)]
    if visible and [item.strip() for item in visible.split(",")] != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} does not match the device plan "
            f"{expected}; vllm_gpus={config.vllm_gpus} are physical ids while "
            f"actor_device/reward_device are indices into the visible list")
    dataset_path = os.environ.get("P2T_DATASET_PATH") or str(
        Path(__file__).resolve().parents[1]
        / "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(f"training parquet not found: {dataset_path}; "
                                f"run scripts/prepare_assets.py --download")
    train_prompts, valid_prompts, split = load_prompt_dataset(
        dataset_path=dataset_path, validation_size=config.validation_size)
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
