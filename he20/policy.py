"""Actor-side probability protocol: logits, selected log-probs, entropy.

he20 never needs the policy's full-vocabulary response logits for a reward: it
needs them only for the GRPO surrogate, the KL term, and the per-token entropy
that ``mask.py`` thresholds (the 80/20 paper's Eq. (6)).  So every reduction here
runs in token blocks and the trainer keeps a per-response microbatch.  That is a
real saving over the VPO-RM arms, which must hold ``[B, T, V]`` at once to compute
their softmax-over-vocabulary direction.

Ported from ``vpo_rm/integration.py`` and ``vpo_rm/trainer.py`` so both methods
share one probability protocol: same support mask, same minimum-stop suppression,
same temperature placement, same FP32 blocking.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .tensors import gather_response, position_ids_from_mask
from .tokens import get_stop_token_ids


def render_chat_prompt(tokenizer, prompt: str, tokenize: bool = False):
    """Render one user prompt with thinking disabled, as the experiment contract
    requires; falls back to the plain prompt if the template rejects the flag."""
    messages = [{"role": "user", "content": str(prompt)}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=True,
                enable_thinking=False, return_dict=False)
        except (ImportError, TypeError, ValueError):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=tokenize, add_generation_prompt=True, return_dict=False)
            except (ImportError, TypeError, ValueError):
                pass
    if tokenize:
        return tokenizer(str(prompt), add_special_tokens=True)["input_ids"]
    return str(prompt)


@torch.no_grad()
def encode_prompts(tokenizer, prompts, device, max_prompt_tokens: int):
    texts = [render_chat_prompt(tokenizer, prompt, tokenize=False) for prompt in prompts]
    batch = tokenizer(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False, truncation=False)
    if (batch["attention_mask"].sum(-1) > max_prompt_tokens).any():
        raise ValueError("Actor prompt exceeds max_prompt_tokens after chat formatting; filter prompts first")
    return {k: v.to(device) for k, v in batch.items()}, texts


def sampling_logits(logits: Tensor, *, min_response_tokens: int = 0,
                    stop_token_ids=(), inplace: bool = False) -> Tensor:
    """Suppress stop tokens before the minimum response length, as generation does."""
    if min_response_tokens < 0:
        raise ValueError("min_response_tokens must be nonnegative")
    if not stop_token_ids or min_response_tokens == 0:
        return logits
    z = logits if inplace else logits.clone()
    z[:, :min_response_tokens, list(stop_token_ids)] = -torch.inf
    return z


_warned_full_logits = False


def _warn_full_logits_once():
    """``logits_to_keep`` is what keeps the forward inside a 48 GB card.

    If the checkpoint's forward rejects it we fall back to materializing the
    whole ``[B, T, V]`` tensor, which is survivable only because the physical
    microbatch is one.  Say so loudly rather than let the memory claim in the
    README quietly become false.
    """
    global _warned_full_logits
    if not _warned_full_logits:
        _warned_full_logits = True
        print("[policy] WARNING: the actor rejected logits_to_keep; falling back to a "
              "full [B,T,V] logits tensor. Memory use is much higher than the "
              "microbatch-1 budget assumes.", flush=True)


def response_logits(actor: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                    response_positions: Tensor, response_mask: Tensor,
                    output_mask: Tensor | None = None) -> Tensor:
    """Logits for each response token, read from its predecessor position.

    ``response_positions`` indexes the tokens themselves, so their predictions
    sit at ``position - 1``.  Only the needed slice is materialized: a full
    ``[B, T, V]`` tensor at 2048 tokens is ~37 GiB in bf16, which is exactly the
    allocation the 48 GB cards cannot afford.
    """
    valid = response_mask.bool()
    if (response_positions[valid] < 1).any():
        raise ValueError("Each response token needs a preceding Actor context position")
    predecessors = response_positions - 1
    if not gather_response(attention_mask, predecessors, valid)[valid].bool().all():
        raise ValueError("The prediction position must belong to the Actor context")
    position_ids = position_ids_from_mask(attention_mask)
    start = int(predecessors[valid].min())
    width = predecessors.shape[1]
    keep = torch.arange(start, start + width, device=input_ids.device)
    contiguous = torch.equal(predecessors[0], keep) and all(
        torch.equal(row, predecessors[0]) for row in predecessors[1:])
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, use_cache=False, return_dict=True)
    if contiguous:
        try:
            logits = actor(**kwargs, logits_to_keep=keep).logits
        except TypeError:
            _warn_full_logits_once()
            logits = gather_response(actor(**kwargs).logits, predecessors, valid)
    else:
        logits = gather_response(actor(**kwargs).logits, predecessors, valid)
    if output_mask is not None:
        if output_mask.shape != (logits.shape[-1],) or not output_mask.bool().any():
            raise ValueError("output_mask must define a nonempty vocabulary support")
        logits = logits.masked_fill(~output_mask.bool(), -torch.inf)
    return logits


def selected_logp_from_logits(logits: Tensor, token_ids: Tensor, response_mask: Tensor,
                              token_chunk_size: int = 128,
                              policy_temperature: float = 1.0) -> Tensor:
    """Selected-token log-probs without a second full-vocabulary FP32 copy."""
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be finite and positive")
    safe_ids = token_ids.masked_fill(~response_mask.bool(), 0)
    parts = []
    chunk = max(1, int(token_chunk_size))
    for lo in range(0, logits.shape[1], chunk):
        hi = min(logits.shape[1], lo + chunk)
        z = logits[:, lo:hi].float() / policy_temperature
        parts.append(z.gather(-1, safe_ids[:, lo:hi, None]).squeeze(-1) - z.logsumexp(-1))
    return torch.cat(parts, dim=1).masked_fill(~response_mask.bool(), 0)


@torch.no_grad()
def response_entropy(logits: Tensor, response_mask: Tensor, policy_temperature: float,
                     token_chunk_size: int = 128) -> Tensor:
    """Per-token response entropy, from FP32 blocks of model-precision logits."""
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be finite and positive")
    entropy = torch.zeros(response_mask.shape, dtype=torch.float32, device=logits.device)
    rows, times = response_mask.bool().nonzero(as_tuple=True)
    chunk = max(1, int(token_chunk_size))
    for lo in range(0, rows.numel(), chunk):
        r, t = rows[lo:lo + chunk], times[lo:lo + chunk]
        z = logits[r, t].float() / policy_temperature
        log_z = z.logsumexp(-1, keepdim=True)
        log_p = z - log_z
        p = log_p.exp()
        entropy[r, t] = -torch.where(p > 0, p * log_p, torch.zeros_like(p)).sum(-1)
    return entropy


@torch.no_grad()
def rollout_logp_microbatch(actor: nn.Module, input_ids, attention_mask, positions,
                            responses, response_mask, output_mask, *, temperature,
                            microbatch: int = 1, adapter: str | None = None,
                            min_response_tokens: int = 0, stop_token_ids=(),
                            token_chunk_size: int = 128, entropy_out=None) -> Tensor:
    """Cache log-probabilities one response microbatch at a time.

    ``adapter="base"`` evaluates the frozen initialization by switching the LoRA
    off -- at step 0 a fresh adapter is exactly the base model, so this is the
    KL reference the experiment contract calls ``init``.  ``adapter`` may also
    name a second mounted adapter when a saved SFT checkpoint is the reference.
    """
    rows = responses.shape[0]
    result = torch.zeros((rows, responses.shape[1]), dtype=torch.float32,
                         device=input_ids.device)
    micro = max(1, int(microbatch))
    switched = adapter not in (None, "base")
    if switched:
        actor.set_adapter(adapter)
    from contextlib import nullcontext
    # "base" disables the LoRA adapter.  With no adapter fitted the model already
    # is the initialisation, so the reference is the identity path.
    context = (actor.disable_adapter() if adapter == "base" and hasattr(actor, "disable_adapter")
               else nullcontext())
    try:
        with context:
            for start in range(0, rows, micro):
                end = min(rows, start + micro)
                logits = response_logits(actor, input_ids[start:end], attention_mask[start:end],
                                         positions[start:end], response_mask[start:end],
                                         output_mask=output_mask)
                logits = sampling_logits(logits, min_response_tokens=min_response_tokens,
                                         stop_token_ids=stop_token_ids, inplace=True)
                result[start:end] = selected_logp_from_logits(
                    logits, responses[start:end], response_mask[start:end],
                    policy_temperature=temperature, token_chunk_size=token_chunk_size)
                if entropy_out is not None:
                    entropy_out[start:end] = response_entropy(
                        logits, response_mask[start:end], temperature, token_chunk_size)
                del logits
    finally:
        if switched:
            actor.set_adapter("default")
    return result


def stop_token_ids_for(tokenizer):
    return get_stop_token_ids(tokenizer)
