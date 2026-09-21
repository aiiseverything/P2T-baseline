"""RED training loop: rollouts, Eq. (6)-(8), and a REINFORCE-style RLOO update.

Credit assignment is the only thing this file does differently from the parent
project's trainer.  Everything around it -- prompt rendering, the vLLM sampling
protocol, the reward-model input mapping, the soft length window, the degeneracy
guard, the optimizer settings and the shared metric names -- is mirrored from
``vpo_rm`` so a RED run and a VPO-RM run differ in exactly one thing.

Two things do *not* mirror the project, both of them forced by the paper:

* The KL penalty lives inside the reward (Eq. 8), so no separate KL loss term is
  applied here and the credit step runs *after* the reference log-probs rather
  than before.
* The surrogate is REINFORCE with RLOO's leave-one-out baseline.  Nothing is ever
  clamped, which is what RLOO's own paper prescribes.  ``clip_eps`` is still
  accepted from the shared config and is read in three places, none of which clips
  the objective: the config validation, the ``rollout_direct_ratio_clip_fraction``
  metric, and the startup gate that checks the trainer's re-forward reproduces the
  sampler's probabilities.  Setting it to 0 or 1 to mean "no clipping" fails the
  validation rather than disabling anything.
* The one importance-like factor is the sampler-to-trainer correction
  ``exp(old_logp - rollout_logprob)``, which multiplies the objective.  It is a
  numerical fix for running the sampler and the trainer in different kernels --
  the startup gate asserts it is ~1 -- not a clipping device, and it is not part
  of RLOO's Eq. (11).  RED_REPRO_NOTES 2.8 records it.

Where RED is *cheaper* than VPO-RM: it needs no policy logits and no gradient
through the reward model to build credit -- one forward pass over the canonical
reward-model row yields every prefix score Eq. (6) differences.  The actor is only
forwarded for the selected log-probs and the entropy, one response microbatch at a
time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch

from .autopush import AutoPusher
from .data import load_prompt_dataset, normalize_prompt, split_prompts
from .length_reward import (calibrate_reward_scale, guard_degenerate_rewards,
                            response_degeneracy, soft_length_penalties)
from .loss import kl_metric, rloo_policy_loss
from .mapping import REWARD_INPUT_PROTOCOL, canonical_reward_input
from .policy import (encode_prompts, render_chat_prompt, response_logits,
                     rollout_logp_microbatch, selected_logp_from_logits,
                     sampling_logits, stop_token_ids_for)
from .reward import (RED_BETA_C_DEFAULT, RED_PROTOCOL, RLOO_ADVANTAGE_RULE,
                     group_sigma, red_final_reward, red_kl_reward, rloo_baseline,
                     rloo_red_credit, sequence_returns, sequence_reward_at_eos)
from .rollout import select_training_rollout, validate_response_termination
from .rm import LastTokenReward, build_rm_batch, score_prefixes
from .tokens import check_tokenizer_identity, load_actor_tokenizer, shared_output_mask
from .vllm import GenerationServer, pack_rollout, unpadded_prompt_token_ids


@dataclass
class TrainerConfig:
    model_name: str = "models/Qwen3-14B-Base"
    reward_model_name: str = "models/Skywork-Reward-V2-Qwen3-8B"
    tokenizer_name: str = ""
    init_adapter: str = ""
    output_dir: str = "runs/red"
    report_dir: str = ""
    run_name: str = "red"
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
    # Kept for config parity with the sibling arms.  RED's surrogate is
    # REINFORCE, which has no ratio to clip, so this value is never read; the
    # trainer stamps the fact into the run manifest as ``red_clipping``.
    clip_eps: float = .2
    beta: float = .03
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True
    # --- RED, paper constants -------------------------------------------------
    beta_c: float = RED_BETA_C_DEFAULT
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
    keep_checkpoints: int = 2  # most recent checkpoints kept; < 1 keeps every one
    token_chunk_size: int = 128
    validation_size: int = 2000
    fit_prompt_filter: bool = True
    push_every: int = 0  # 0 disables autopush
    push_remote: str = "origin"
    push_branch: str = "main"         # branch name on the remote
    push_local_branch: str = ""       # branch this checkout must be on
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
        # RED-specific validation: the paper has no allocator, so VPO-only knobs
        # must be rejected rather than silently ignored.
        if not math.isfinite(c.beta_c) or not 0 <= c.beta_c <= 1:
            raise ValueError("beta_c must be finite and within [0, 1]")
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


class REDTrainer:
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
                                 local_branch=self.cfg.push_local_branch,
                                 report_dir=self.report_dir, repo_root=Path(__file__).resolve().parents[1])

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, config: TrainerConfig, *, start_generation: bool = True) -> "REDTrainer":
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
        the RED arm can be shown to have started from the same bytes as its
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
        """Keep only the newest adapters: each one is ~2.0 GiB on disk.

        Measured at 2.0 GiB per snapshot, four times the 0.5 GiB this docstring
        used to claim, so two retained snapshots cost 3.9 GiB rather than 1 GiB.

        Two, not one, is deliberate: vLLM loads an adapter by directory path, so
        keeping the previous snapshot leaves a buffer across the write-new,
        switch-server, drop-old sequence. Pruning to a single snapshot would race
        the generation server against a directory it may still be referencing.
        """
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
        record = {"mode": "red_soft_length", "sigma0": sigma0,
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
        rewards, _ = score_prefixes(self.reward, self.reward_tokenizer, rows, mapped,
                                    responses, rmask, device=self.reward_device, microbatch=1)
        return rewards.to(self.reward_device), stats

    def score_and_redistribute(self, prompts, responses, rmask):
        """Pooled RM scores plus the Eq. (6) prefix differences for every response.

        ``prompts`` is one entry per *prompt group*; the RM rows are per response,
        so each prompt is repeated ``group_size`` times -- the same expansion the
        parent project performs before scoring.

        One forward pass per row yields the score head at *every* position, so the
        redistributed rewards and the sequence score come out of the same call and
        are consistent by construction: ``sum_t r~_t = R_phi(x, y) - R_phi(x, empty)``.
        Nothing here needs a gradient through the reward model.
        """
        group_prompts = [prompt for prompt in prompts for _ in range(self.cfg.group_size)]
        rows, mapped, fixed_weight_mask, stats = build_rm_batch(
            self.actor_tokenizer, self.reward_tokenizer, group_prompts, responses, rmask,
            self.cfg.max_prompt_tokens, self.cfg.max_response_tokens)
        rewards, token_rewards = score_prefixes(
            self.reward, self.reward_tokenizer, rows, mapped, responses, rmask,
            device=self.reward_device, microbatch=self.cfg.microbatch_responses)
        rewards = rewards.to(self.reward_device)
        token_rewards = token_rewards.to(self.reward_device)
        stats = dict(stats)
        stats["red_unmapped_share_mean"] = float(
            fixed_weight_mask.sum() / rmask.sum().clamp_min(1))
        # The project's startup gate bounds how much *real text* failed to map.
        # If a large share of content tokens carry no reward-model reading, the
        # redistribution is mostly structural zeros and the advantages are being
        # driven by positions the reward model never read.
        if (self.rollout_index == 0
                and stats["rm_unmapped_content_fraction"] > self.cfg.max_unmapped_content_fraction):
            raise ValueError(
                f"Unmapped content tokens are {stats['rm_unmapped_content_fraction']:.3f} of "
                f"the response, above the {self.cfg.max_unmapped_content_fraction} bound; "
                f"the actor-to-RM token mapping is not holding")
        return rewards, token_rewards, fixed_weight_mask.to(self.actor_device), stats

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

        # `phase(name, start)` records the work between `start` and now, so every
        # call has to come *after* the work it names.  This one used to sit here,
        # before generation, which shifted every label by one: `generation_sec`
        # reported ~0 and the real generation cost was filed under
        # `reward_model_gradient_sec`.  The plan's time budget is backfilled from
        # these numbers, so the labels have to mean what they say.
        clock = started
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
        clock = phase("generation_sec", clock)
        self._write_rollout_artifacts(prompts, rollout)
        if rollout is None:
            # No response reaches the loss, including KL, Adam moments or decay.
            self.rollout_index += 1
            metrics = {"rollout": self.rollout_index, "skipped_rollout": True,
                       "optimizer_steps": 0, "response_tokens": 0, "reward_count": 0,
                       "loss": 0.0, "elapsed_sec": time.monotonic() - started,
                       "gpu_hours": (time.monotonic() - self.started)
                       * (2 + self.cfg.vllm_tensor_parallel_size) / 3600,
                       "phase_generation_sec": timings["generation_sec"], **selection}
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

        raw_rewards, token_rewards, _, alignment = self.score_and_redistribute(
            prompts, responses, rmask)
        clock = phase("reward_model_prefix_sec", clock)
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
        # Diagnostics only.  RED does not standardise -- RLOO's baseline is the
        # leave-one-out mean -- but the group spread is what the sibling arms
        # divide by, so it is reported to keep the reward scales comparable.
        scales = group_sigma(shaped, group_ids)

        entropy = torch.zeros(responses.shape, dtype=torch.float32, device=self.actor_device)
        old_logp = rollout_logp_microbatch(
            self.actor, input_ids, full_mask, positions, responses, rmask, self.output_mask,
            temperature=self.cfg.temperature, microbatch=self.cfg.microbatch_responses,
            min_response_tokens=self.cfg.min_response_tokens,
            stop_token_ids=self.stop_token_ids, token_chunk_size=self.cfg.token_chunk_size,
            entropy_out=entropy)
        clock = phase("actor_old_logp_sec", clock)
        reference_adapter = "ref" if self.cfg.init_adapter else "base"
        ref_logp = rollout_logp_microbatch(
            self.actor, input_ids, full_mask, positions, responses, rmask, self.output_mask,
            temperature=self.cfg.temperature, microbatch=self.cfg.microbatch_responses,
            adapter=reference_adapter, min_response_tokens=self.cfg.min_response_tokens,
            stop_token_ids=self.stop_token_ids, token_chunk_size=self.cfg.token_chunk_size)
        clock = phase("ref_logp_sec", clock)

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

        # ---------------------------------------------------- credit, Eq. (6)-(8)
        # Eq. (8) subtracts the KL from the reward, so credit can only be
        # assembled once the reference log-probs exist -- the opposite of the
        # sibling arms, which build credit first and add the KL as a separate loss
        # term afterwards.  The consequence is that no KL term appears below.
        #
        # Everything here lives on the actor device, including the group ids, which
        # are otherwise built on the reward device.  The sibling arm can build
        # credit on the reward device and move the three [B, T] fields afterwards;
        # RED cannot, because Eq. (8) needs the reference log-probs.  Mixing the
        # two is a crash on the first rollout that no CPU test can see, so the
        # credit functions validate their own device arguments.
        credit_group_ids = group_ids.to(self.actor_device)
        actor_mask = rmask.to(self.actor_device)
        kl_reward = red_kl_reward(old_logp, ref_logp, actor_mask)
        sequence_reward = sequence_reward_at_eos(raw_rewards.to(self.actor_device), actor_mask)
        final_reward = red_final_reward(token_rewards.to(self.actor_device), sequence_reward,
                                        kl_reward, actor_mask, beta_c=self.cfg.beta_c,
                                        beta=self.cfg.beta)
        # RLOO's baseline is a leave-one-out mean of the *sequence return*, and
        # that return is KL-shaped (RLOO Eq. 3), so the same KL that rides inside
        # r^final also enters the baseline.  The project's shaped reward supplies
        # the length control there, which is how the length window keeps acting on
        # the advantage in an arm that does not standardise.
        returns = sequence_returns(shaped.to(self.actor_device), kl_reward, actor_mask,
                                   self.cfg.beta)
        baseline = rloo_baseline(returns, credit_group_ids)
        credit = rloo_red_credit(final_reward, baseline, actor_mask)
        del kl_reward, sequence_reward, final_reward, returns
        clock = phase("credit_sec", clock)

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
                chunk = rloo_policy_loss(new_logp, credit.advantage[sl], rmask[sl],
                                         importance_weights=importance[sl])
                # No KL term: Eq. (8) already subtracted beta * r^KL from every
                # token reward, so the constraint reaches the policy through the
                # advantage.  Adding the project's KL term here would double-count
                # it -- see RED_REPRO_NOTES.md 2.7.
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

        clock = phase("actor_update_sec", clock)
        self.total_tokens += int(rmask.sum().item())
        self.rollout_index += 1

        with torch.no_grad():
            # Same estimator and weighting as the project's reference-KL metric, so
            # the drift curve is comparable with the sibling arms'.  Note this is
            # *not* the KL that Eq. (8) subtracts -- the reward uses the signed log
            # ratio (see RED_REPRO_NOTES 2.5).
            kl_to_init = float(kl_metric(ref_logp, old_logp, rmask,
                                         importance_weights=importance))
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
            "advantage_abs_mean": float(credit.advantage.abs().mean()),
            "rloo_baseline_mean": float(baseline.mean()),
            "rloo_baseline_std": float(baseline.std(correction=0)),
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
        metrics.update(self._red_diagnostics(credit, baseline, actor_mask))
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
        # Everything after the optimizer step is metric computation, the RED
        # diagnostics included, so this is the phase that closes the step.
        clock = phase("logging_sec", clock)
        metrics.update({f"phase_{key}": value for key, value in timings.items()})
        if self.actor_device.type == "cuda":
            metrics["actor_peak_gb"] = torch.cuda.max_memory_allocated(self.actor_device) / 2 ** 30
        if self.reward_device.type == "cuda":
            metrics["reward_peak_gb"] = torch.cuda.max_memory_allocated(self.reward_device) / 2 ** 30
        self._log(metrics)
        self._dump_credit(credit, raw_rewards, token_rewards, baseline)
        if self.cfg.checkpoint_interval and self.rollout_index % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint(self.rollout_index)
        self.pusher.maybe_push(self.rollout_index, metrics)
        return metrics

    # ----------------------------------------------------------------- reports
    def _red_diagnostics(self, credit, baseline, mask):
        """Observation only: nothing here feeds back into the update.

        RED's redistributed rewards are signed differences, so unlike the sibling
        arm's attribution softmax they do not form a distribution over tokens.
        The share used here is the *positive* part of the reward normalised to sum
        to one, which keeps ``credit_ess_ratio`` and the flat/one-hot fractions
        readable and comparable across arms; rows with no positive credit at all
        fall back to a uniform share instead of dividing by zero.
        """
        number = mask.sum(-1).float().clamp_min(1)
        weight = credit.weight.masked_fill(~mask, 0)
        share = (weight / number[:, None]).masked_fill(~mask, 0)
        flat = (share.amax(-1) <= 1.0 / number + 1e-3).float().mean()
        onehot = (share.amax(-1) >= 0.9).float().mean()
        # ESS/T = 1/(T * sum p^2): 1 is flat (inert), 1/T is one-hot.  Same
        # direction as the VPO arms' credit ESS, not the inverse.
        ess = 1.0 / (share.square().sum(-1) * number).clamp_min(1e-12)
        valid_reward = credit.direction[mask]
        valid_advantage = credit.advantage[mask]
        quantiles = torch.quantile(valid_reward.float(),
                                   torch.tensor([.01, .5, .99], device=valid_reward.device))
        # The token-varying part of the credit: the within-response spread of
        # r^final, relative to the sequence scale the leave-one-out baseline sets.
        # This is the analogue of the sibling arm's varying-bonus ratio.
        row_mean = (credit.direction.masked_fill(~mask, 0).sum(-1, keepdim=True) / number[:, None])
        varying = (credit.direction - row_mean)[mask]
        baseline_scale = baseline.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
        # How often the redistribution flips a token's update direction relative
        # to the baseline alone -- i.e. how far this arm's gradient is from plain
        # RLOO's, which is the quantity the whole experiment turns on.
        baseline_only = (-baseline[:, None].expand_as(mask))[mask]
        nonzero = baseline_only != 0
        flip = (torch.sign(valid_advantage[nonzero]) != torch.sign(baseline_only[nonzero]))
        positive = valid_reward.clamp_min(0).sum()
        mass = valid_reward.abs().sum().clamp_min(1e-12)
        return {
            "red_beta_c": self.cfg.beta_c,
            # The KL coefficient also rides inside the reward (Eq. 8), so it is a
            # method constant and is stamped per row like beta_c.
            "red_beta": self.cfg.beta,
            "red_protocol": RED_PROTOCOL,
            "red_advantage_rule": RLOO_ADVANTAGE_RULE,
            "red_token_reward_mean": float(valid_reward.mean()),
            "red_token_reward_std": float(valid_reward.std(correction=0)),
            "red_token_reward_abs_mean": float(valid_reward.abs().mean()),
            "red_token_reward_p01": float(quantiles[0]),
            "red_token_reward_p50": float(quantiles[1]),
            "red_token_reward_p99": float(quantiles[2]),
            "red_positive_mass_fraction": float(positive / mass),
            "red_share_max_mean": float(share.amax(-1).mean()),
            "red_flat_response_fraction": float(flat),
            "red_onehot_response_fraction": float(onehot),
            "red_varying_over_baseline": float(varying.abs().mean() / baseline_scale),
            "red_advantage_flip_fraction": float(flip.float().mean()) if nonzero.any() else 0.0,
            # VPO-parity names so the project's existing analysis keeps working.
            "credit_w_mean": float(weight[mask].mean()),
            "credit_w_std": float(weight[mask].std(unbiased=False)),
            "credit_w_max": float(weight[mask].max()),
            "credit_ess_ratio": float(ess.mean()),
        }

    def _dump_credit(self, credit, raw_rewards, token_rewards, baseline):
        # A small, per-step artifact: the redistributed token reward, the
        # leave-one-out baseline and the final advantage behind every response
        # token.  ``d``/``advantage`` keep the sibling arms' field names so the
        # same readers work; ``i`` has no RED counterpart and is omitted rather
        # than filled with zeros that would read as real attribution.
        torch.save({"w": credit.weight.half().cpu(),
                    "d": credit.direction.half().cpu(),
                    "advantage": credit.advantage.half().cpu(),
                    "raw_reward": raw_rewards.float().cpu(),
                    "token_reward": token_rewards.float().cpu(),
                    "baseline": baseline.float().cpu(),
                    "protocol": RED_PROTOCOL,
                    "advantage_rule": RLOO_ADVANTAGE_RULE,
                    "beta_c": self.cfg.beta_c},
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

    def _prune_checkpoints(self):
        """Keep only the newest checkpoints: each one is ~2 GiB on disk.

        ``keep_checkpoints < 1`` keeps every checkpoint.  The final checkpoint is
        always the newest, so it is never the one pruned.
        """
        if self.cfg.keep_checkpoints < 1:
            return
        paths = []
        for path in self.output_dir.glob("checkpoint-*"):
            try:
                paths.append((int(path.name.split("-")[1]), path))
            except (IndexError, ValueError):
                continue  # not a step-numbered checkpoint; leave it alone
        paths.sort()
        for _, stale in paths[:max(0, len(paths) - self.cfg.keep_checkpoints)]:
            shutil.rmtree(stale, ignore_errors=True)

    def save_checkpoint(self, step: int):
        path = self.output_dir / f"checkpoint-{step}"
        self.actor.save_pretrained(path)
        self.actor_tokenizer.save_pretrained(path)
        (path / "run_manifest.json").write_text(json.dumps({
            "resolved_config": asdict(self.cfg), "step": step,
            "reward_input_protocol": REWARD_INPUT_PROTOCOL,
            "red_protocol": RED_PROTOCOL, "red_advantage_rule": RLOO_ADVANTAGE_RULE,
            "red_beta_c": self.cfg.beta_c,
            # RLOO has no importance ratio, so there is nothing to clip.  Stamped
            # explicitly so a reader of the manifest is never left wondering
            # whether the shared clip_eps silently applied.
            "red_clipping": "none (reinforce)",
            "total_tokens": self.total_tokens,
            # There is no in-place resume: the optimizer moments are not saved and
            # `check_fresh_output` refuses to reuse a run directory.  What a
            # checkpoint *does* support is relaunching as a new run seeded from
            # these weights.  Spelling out what does and does not carry over, so a
            # restart is not silently a different experiment.
            "resume_supported": False,
            "restart": {
                "recipe": "point a new config's init_adapter at this directory and "
                          "give it a fresh output_dir/report_dir",
                "carries_over": ["LoRA weights (default + ref)", "tokenizer",
                                 "the resolved config recorded here"],
                "does_not_carry_over": [
                    "AdamW moment estimates: the first steps after a restart take a "
                    "larger effective step than they otherwise would",
                    "the KL reference identity: beta-KL is measured against "
                    "init_adapter, so a restart re-anchors it to these weights "
                    "instead of the original SFT initialisation",
                    "the prompt cursor: prompts are walked as "
                    "(rollout_index * prompts_per_rollout) % len(prompts), so a "
                    "restart begins again at the corpus start",
                ],
            },
            **self._adapter_identity},
            indent=2, default=str))
        self._prune_checkpoints()
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
    # Not train.pid: the launcher writes that before the trainer starts, so
    # including it would make every legitimate launch look like a reused run.
    markers = ("metrics.jsonl", "length_reward_calibration.json", "data_split.json",
               "vllm-adapters", "run_manifest.json")
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
    parser = argparse.ArgumentParser(description="RED baseline training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-every", type=int, default=None)
    parser.add_argument("--calibration-prompts", type=int, default=None)
    parser.add_argument("--sigma0", type=float, default=None)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configuration and exit without loading a model")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for name, value in (("output_dir", args.output_dir), ("push_every", args.push_every),
                        ("calibration_prompts", args.calibration_prompts),
                        ("sigma0", args.sigma0), ("rollout_iterations", args.max_rollouts)):
        if value is not None:
            setattr(config, name, value)
    if args.dry_run:
        config.dry_run = True
    if not config.report_dir:
        config.report_dir = str(Path(config.output_dir) / "report")
    if config.dry_run:
        print(json.dumps({"resolved_config": asdict(config.resolved())}, indent=2, default=str))
        return
    check_fresh_output(config.output_dir)
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError("RED training needs at least two visible GPUs: actor and reward model")
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
    dataset_path = os.environ.get("RED_DATASET_PATH") or str(
        Path(__file__).resolve().parents[1]
        / "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet")
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(f"training parquet not found: {dataset_path}; "
                                f"run scripts/prepare_assets.py --download")
    train_prompts, valid_prompts, split = load_prompt_dataset(
        dataset_path=dataset_path, validation_size=config.validation_size)
    trainer = REDTrainer.from_pretrained(config)
    try:
        if trainer.generation is not None:
            trainer.generation.wait_until_ready()
        (trainer.output_dir / "data_split.json").write_text(json.dumps(split, indent=2, default=str))
        trainer.train(train_prompts)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
