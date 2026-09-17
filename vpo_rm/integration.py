"""Rollout cache construction and differentiable Actor loss for a GRPO trainer."""
from dataclasses import dataclass
import math
import hashlib
import json
import torch
from torch import Tensor, nn
from .alignment import gather_response, check_response_tokens
from .core import Credit, compute_credit, grpo_policy_loss
from .reward import reward_input_gradients, position_ids_from_mask
from .token_policy import get_stop_token_ids


def sampling_logits(logits: Tensor, *, min_response_tokens: int = 0, stop_token_ids=(),
                    inplace: bool = False) -> Tensor:
    """Apply the minimum-length stop support shared by both generation backends.

    Vocabulary support has already been applied by actor_response_logits.
    Position zero is the first generated token; stopping is allowed after the
    configured number of generated tokens. Temperature is applied separately
    inside FP32 probability reductions, preserving BF16 full-tensor storage.
    """
    if min_response_tokens < 0:
        raise ValueError("min_response_tokens must be nonnegative")
    if not stop_token_ids or min_response_tokens == 0:
        return logits
    z = logits if inplace else logits.clone()
    if stop_token_ids and min_response_tokens:
        z[:, :min_response_tokens, list(stop_token_ids)] = -torch.inf
    return z


def vllm_sampling_kwargs(tokenizer, vocab_size: int, request: dict) -> dict:
    """One explicit on-policy protocol shared by server and one-shot generation."""
    probe = bool(request.get("probe", False))
    temperature = float(request.get("temperature", 1.0))
    if not math.isfinite(temperature) or not .01 <= temperature <= 2.0:
        raise ValueError("training temperature must be in [0.01, 2] to avoid vLLM clamping")
    if request.get("top_p", 1.0) != 1.0 or request.get("top_k", 0) != 0:
        raise ValueError("training supports top_p=1 and top_k=0 only")
    if float(request.get("presence_penalty", 0.0)) != 0:
        raise ValueError("presence_penalty is unsupported by the training probability protocol")
    maximum = int(request["max_tokens"])
    minimum = 0 if probe else int(request.get("min_tokens", 8))
    if maximum < 1 or not 0 <= minimum <= maximum:
        raise ValueError("generation requires 0 <= min_tokens <= max_tokens")
    return {"temperature": 0.0 if probe else temperature,
            "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
            "max_tokens": maximum, "min_tokens": minimum,
            "n": 1 if probe else int(request.get("group_size", 8)),
            "stop_token_ids": list(get_stop_token_ids(tokenizer)),
            **({"logprobs": 1} if request.get("return_logprobs", False) else {}),
            **vllm_support_kwargs(tokenizer, vocab_size)}


def vllm_support_kwargs(tokenizer, vocab_size: int) -> dict:
    """Exact compact output ban, avoiding vLLM's 1024-entry allow-list limit.

    Use the SamplingParams constructor directly through checked_sampling_params;
    the OpenAI-style from_optional helper clamps infinite bias to -100.
    """
    from .alignment import shared_output_mask
    banned = (~shared_output_mask(tokenizer, vocab_size)).nonzero().flatten().tolist()
    if len(banned) > 1024:
        raise ValueError("This vLLM protocol supports at most 1024 suppressed vocabulary IDs")
    return {"logit_bias": {token_id: -math.inf for token_id in banned}}


def checked_sampling_params(sampling_params_class, **kwargs):
    """Fail closed if a backend changes exact support or clamps temperature."""
    params = sampling_params_class(**kwargs)
    expected = kwargs.get("logit_bias") or {}
    actual = getattr(params, "logit_bias", None) or {}
    for token_id, value in expected.items():
        if value == -math.inf and actual.get(token_id) != -math.inf:
            raise RuntimeError("vLLM modified the exact output support; refusing approximate sampling")
    if "temperature" in kwargs and getattr(params, "temperature", kwargs["temperature"]) != kwargs["temperature"]:
        raise RuntimeError("vLLM modified sampling temperature; probability protocol would differ")
    return params


def sampling_summary(kwargs):
    """JSON-safe sampling provenance; never serialize nonstandard Infinity."""
    summary = {k:v for k,v in kwargs.items() if k not in {"logit_bias", "allowed_token_ids"}}
    suppressed = sorted((kwargs.get("logit_bias") or {}).keys())
    summary["suppressed_token_count"] = len(suppressed)
    summary["suppressed_token_ids_sha256"] = hashlib.sha256(json.dumps(suppressed).encode()).hexdigest()
    return summary


def generation_payload(generated, stop_token_ids, *, include_logprobs=False) -> dict:
    """Keep engine termination metadata; an emitted final stop is complete at cap."""
    rows, reasons, engine_reasons, stop_reasons = [], [], [], []
    selected_logprobs, prompt_token_ids = [], []
    for result in generated:
        if include_logprobs:
            prompt_token_ids.append(list(result.prompt_token_ids))
        for output in result.outputs:
            ids = list(output.token_ids)
            reason = output.finish_reason
            if not ids or reason not in {"stop", "length"}:
                raise ValueError("Generation must return nonempty responses with stop/length metadata")
            rows.append(ids)
            engine_reasons.append(reason)
            reasons.append("stop" if ids[-1] in stop_token_ids else reason)
            stop_reasons.append(output.stop_reason)
            if include_logprobs:
                probabilities = getattr(output, "logprobs", None)
                if probabilities is None or len(probabilities) != len(ids):
                    raise ValueError("Requested chosen-token logprobs are missing or misaligned")
                values = []
                for token, options in zip(ids, probabilities):
                    selected = options.get(token)
                    if selected is None or not math.isfinite(selected.logprob):
                        raise ValueError("Requested chosen-token logprob is missing or nonfinite")
                    values.append(float(selected.logprob))
                selected_logprobs.append(values)
    return {"rows": rows, "finish_reasons": reasons,
            "engine_finish_reasons": engine_reasons, "stop_reasons": stop_reasons,
            "tokens": sum(map(len, rows)),
            **({"selected_logprobs": selected_logprobs, "prompt_token_ids": prompt_token_ids}
               if include_logprobs else {})}


def validate_response_termination(responses: Tensor, response_mask: Tensor,
                                  finish_reasons, stop_token_ids,
                                  max_response_tokens: int) -> None:
    """Require terminal EOS for stopped rows and a full cap for truncations.

    Responses are right padded; padding may share the EOS token ID. An EOS
    emitted exactly at the cap is complete and must be labelled ``stop``.
    """
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("Response termination requires matching [B,T] tensors")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("Response termination requires a binary response mask")
    if len(finish_reasons) != len(responses) or any(
            reason not in {"stop", "length"} for reason in finish_reasons):
        raise ValueError("Every response requires a stop or length finish reason")
    valid = response_mask.bool()
    lengths = valid.sum(-1)
    if ((lengths < 1) | (lengths > max_response_tokens)).any():
        raise ValueError("Invalid response termination length")
    expected = torch.arange(responses.shape[1], device=responses.device)[None, :] < lengths[:, None]
    if not torch.equal(valid, expected):
        raise ValueError("Response termination requires right-padded response tokens")
    stop_ids = torch.as_tensor(tuple(stop_token_ids), dtype=responses.dtype, device=responses.device)
    stopped = torch.isin(responses, stop_ids) & valid
    stop_count = stopped.sum(-1)
    terminal_stop = stopped.gather(1, (lengths - 1)[:, None]).squeeze(1)
    expects_stop = torch.tensor([reason == "stop" for reason in finish_reasons],
                                device=responses.device)
    consistent = torch.where(expects_stop, (stop_count == 1) & terminal_stop,
                             (stop_count == 0) & (lengths == max_response_tokens))
    if not consistent.all():
        raise ValueError("Response termination stop/EOS or length metadata is inconsistent")


@dataclass(frozen=True)
class RolloutCache:
    old_logp: Tensor
    credit: Credit
    policy_temperature: float = 1.0
    min_response_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()


def actor_response_logits(actor: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                          response_positions: Tensor, response_mask: Tensor,
                          position_ids: Tensor | None = None,
                          output_mask: Tensor | None = None) -> Tensor:
    """response_positions indexes actual tokens; their predictions are at t-1."""
    valid = response_mask.bool()
    if (response_positions[valid] < 1).any():
        raise ValueError("Each response token needs a preceding Actor context position")
    predecessors = response_positions - 1
    if not gather_response(attention_mask, predecessors, valid)[valid].bool().all():
        raise ValueError("The prediction position must belong to the Actor context")
    if position_ids is None:
        position_ids = position_ids_from_mask(attention_mask)
    # The trainer pads every row to a common prompt width and appends the
    # response contiguously.  Qwen3/Transformers supports logits_to_keep, so
    # request only the predecessor positions needed for the response.  This
    # avoids materializing a full-sequence logits tensor and then copying a
    # second [B,T,V] response tensor (which is prohibitive at 2048 tokens).
    start = int(predecessors[valid].min())
    # The keep range must span the whole padded response width.  Deriving the end
    # from valid positions alone compares a width-T row against a length-L range,
    # which silently disables logits_to_keep for every response shorter than the
    # batch maximum (63 of 64 microbatch forwards in the main configuration).
    width = predecessors.shape[1]
    keep = torch.arange(start, start + width, device=input_ids.device)
    contiguous = torch.equal(predecessors[0], keep) and all(
        torch.equal(row, predecessors[0]) for row in predecessors[1:])
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, use_cache=False, return_dict=True)
    if contiguous:
        try:
            outputs = actor(**kwargs, logits_to_keep=keep)
            logits = outputs.logits
        except TypeError:
            outputs = actor(**kwargs)
            logits = gather_response(outputs.logits, predecessors, valid)
    else:
        outputs = actor(**kwargs)
        logits = gather_response(outputs.logits, predecessors, valid)
    if output_mask is not None:
        if output_mask.shape != (logits.shape[-1],) or not output_mask.bool().any():
            raise ValueError("output_mask must define a nonempty vocabulary support")
        logits = logits.masked_fill(~output_mask.bool(), -torch.inf)
    return logits


def selected_logp_from_logits(logits: Tensor, token_ids: Tensor,
                              response_mask: Tensor, token_chunk_size: int = 128,
                              policy_temperature: float = 1.0) -> Tensor:
    """Compute selected-token log-probs without a full FP32 B×T×V copy.

    The model output remains BF16.  FP32 reduction is performed one token block
    at a time, which keeps the numerical reduction used by the loss while
    avoiding a second full-vocabulary allocation at long response lengths.
    """
    if not math.isfinite(policy_temperature) or policy_temperature <= 0:
        raise ValueError("policy_temperature must be finite and positive")
    safe_ids = token_ids.masked_fill(~response_mask.bool(), 0)
    parts = []
    for lo in range(0, logits.shape[1], max(1, int(token_chunk_size))):
        hi = min(logits.shape[1], lo + max(1, int(token_chunk_size)))
        z = logits[:, lo:hi].float() / policy_temperature
        parts.append(z.gather(-1, safe_ids[:, lo:hi, None]).squeeze(-1)
                     - z.logsumexp(-1))
    return torch.cat(parts, dim=1).masked_fill(~response_mask.bool(), 0)


@torch.no_grad()
def build_credit_cache(old_logits: Tensor, token_ids: Tensor, input_grads: Tensor,
                       rm_weight: Tensor, advantage: Tensor, reward_scale: Tensor,
                       response_mask: Tensor, tau: float, credit_lambda: float = 2.0,
                       freeze_stop_tokens: bool = False,
                       freeze_structural: bool = False,
                       stop_token_ids=None, structural_token_ids=None,
                       policy_temperature: float = 1.0,
                       min_response_tokens: int = 0,
                       fixed_weight_mask: Tensor | None = None,
                       **chunk_sizes) -> RolloutCache:
    """Cache attribution and the probability protocol after group aggregation.

    old_logits must already include the vocabulary and minimum-stop support
    masks. Temperature is applied inside FP32 chunks. The public loss helper
    reuses the cached protocol rather than requiring duplicate configuration.
    """
    stop_ids = tuple(stop_token_ids or ())
    if min_response_tokens < 0:
        raise ValueError("min_response_tokens must be nonnegative")
    if min_response_tokens and stop_ids:
        early = old_logits[:, :min_response_tokens, list(stop_ids)]
        if not torch.isneginf(early).all():
            raise ValueError("old_logits must already include the minimum-stop support mask")
    credit = compute_credit(old_logits, token_ids, input_grads, rm_weight,
                            advantage, reward_scale, response_mask, tau,
                            credit_lambda=credit_lambda,
                            freeze_stop_tokens=freeze_stop_tokens,
                            freeze_structural=freeze_structural,
                            stop_token_ids=stop_token_ids,
                            structural_token_ids=structural_token_ids,
                            fixed_weight_mask=fixed_weight_mask,
                            policy_temperature=policy_temperature, **chunk_sizes)
    # Token blocks avoid allocating an additional full [B,T,V] log-softmax.
    old_logp = torch.zeros_like(credit.direction)
    rows, times = response_mask.bool().nonzero(as_tuple=True)
    chunk = chunk_sizes.get("token_chunk_size", 128)
    for lo in range(0, rows.numel(), chunk):
        r, t = rows[lo:lo+chunk], times[lo:lo+chunk]
        z = old_logits[r, t].float() / policy_temperature
        old_logp[r, t] = z.gather(-1, token_ids[r, t, None]).squeeze(-1) - z.logsumexp(-1)
    return RolloutCache(old_logp, credit, policy_temperature, min_response_tokens, stop_ids)


def response_reward_gradients(reward_model: nn.Module, rm_input_ids: Tensor,
                              rm_attention_mask: Tensor, rm_response_positions: Tensor,
                              token_ids: Tensor, response_mask: Tensor,
                              **score_kwargs) -> tuple[Tensor, Tensor]:
    check_response_tokens(rm_input_ids, rm_attention_mask, rm_response_positions,
                          token_ids, response_mask)
    rewards, all_grads = reward_input_gradients(reward_model, rm_input_ids,
                                              rm_attention_mask, **score_kwargs)
    return rewards, gather_response(all_grads, rm_response_positions, response_mask)


def actor_policy_loss(actor: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                      response_positions: Tensor, token_ids: Tensor,
                      response_mask: Tensor, cache: RolloutCache,
                      clip_eps: float = .2, position_ids: Tensor | None = None,
                      output_mask: Tensor | None = None) -> Tensor:
    check_response_tokens(input_ids, attention_mask, response_positions,
                          token_ids, response_mask)
    logits = actor_response_logits(actor, input_ids, attention_mask, response_positions,
                                   response_mask, position_ids, output_mask)
    logits = sampling_logits(logits, min_response_tokens=cache.min_response_tokens,
                             stop_token_ids=cache.stop_token_ids, inplace=True)
    # FP32 reductions retain stable log probabilities with bf16 model weights,
    # while token blocking avoids materializing a second full [B,T,V] tensor.
    new_logp = selected_logp_from_logits(logits, token_ids, response_mask,
                                         policy_temperature=cache.policy_temperature)
    return grpo_policy_loss(new_logp, cache.old_logp, cache.credit.advantage,
                            response_mask, clip_eps)
