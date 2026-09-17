"""Retry and select complete prompt groups before soft-reward training."""
from dataclasses import dataclass

import torch
from torch import Tensor

from .alignment import check_response_tokens
from .length_reward import response_degeneracy
from .integration import validate_response_termination


@dataclass
class _Row:
    prefix: Tensor
    response: Tensor
    rendered: str
    finish_reason: str
    degenerate: bool
    rollout_logp: Tensor | None


def validate_rollout_logprobs(rollout, required: bool = False) -> Tensor | None:
    """Validate optional sampling probabilities on actual response tokens only.

    Missing probabilities are allowed for the original seven-field HF rollout.
    Padding values carry no probability information and are deliberately ignored.
    """
    if not isinstance(rollout, (tuple, list)) or len(rollout) not in (7, 8):
        raise ValueError('rollout must contain seven fields and optional rollout logprobabilities')
    logp = rollout[7] if len(rollout) == 8 else None
    if logp is None:
        if required:
            raise ValueError('rollout logprobabilities are required for importance correction')
        return None
    responses, mask = rollout[3:5]
    if (not isinstance(logp, Tensor) or not logp.is_floating_point()
            or not isinstance(responses, Tensor) or not isinstance(mask, Tensor)
            or logp.ndim != 2 or logp.shape != responses.shape or logp.shape != mask.shape):
        raise ValueError('rollout logprobabilities must be floating tensors matching response shape')
    if logp.device != responses.device or logp.device != mask.device:
        raise ValueError('rollout logprobabilities must share the response device')
    selected = logp[mask.bool()]
    if not torch.isfinite(selected).all() or (selected > 0).any():
        raise ValueError('valid rollout logprobabilities must be finite and nonpositive')
    return logp


def _validate_sample(trainer, rollout, prompt_count: int) -> list[_Row]:
    """Validate the original positions and support before any sample is dropped."""
    if not isinstance(rollout, (tuple, list)) or len(rollout) not in (7, 8):
        raise ValueError("rollout must contain seven fields and optional rollout logprobabilities")
    ids, attention, positions, responses, mask, rendered, reasons = rollout[:7]
    tensors = (ids, attention, positions, responses, mask)
    if any(not isinstance(x, Tensor) or x.ndim != 2 for x in tensors):
        raise ValueError("rollout tensors must be two-dimensional")
    batch = prompt_count * trainer.cfg.group_size
    if (any(x.shape[0] != batch for x in tensors) or attention.shape != ids.shape
            or positions.shape != responses.shape or mask.shape != responses.shape):
        raise ValueError("rollout tensor shapes must preserve complete prompt groups")
    if ids.dtype != torch.long or responses.dtype != torch.long or positions.dtype != torch.long:
        raise ValueError("rollout token IDs and response positions must be int64")
    if any(x.device != ids.device for x in tensors):
        raise ValueError("rollout tensors must share a device")
    if not isinstance(rendered, (list, tuple)) or len(rendered) != batch or any(
            not isinstance(x, str) for x in rendered):
        raise ValueError("rollout rendered prompts must contain one string per response")
    if not isinstance(reasons, (list, tuple)) or len(reasons) != batch or any(
            x not in ("stop", "length") for x in reasons):
        raise ValueError("rollout finish reasons must be stop or length, one per response")
    if any(not ((x == 0) | (x == 1)).all() for x in (attention, mask)):
        raise ValueError("rollout attention and response masks must be binary")
    valid = mask.bool()
    rollout_logp = validate_rollout_logprobs(rollout,
        required=getattr(trainer.cfg, 'rollout_importance_correction', False))
    lengths = valid.sum(-1)
    if ((lengths < 1) | (lengths > trainer.cfg.max_response_tokens)).any():
        raise ValueError("rollout responses must be nonempty and within max_response_tokens")
    expected_mask = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(valid, expected_mask):
        raise ValueError("rollout responses must be right padded")
    validate_response_termination(responses, valid, reasons, trainer.stop_token_ids,
                                  trainer.cfg.max_response_tokens)
    support = trainer.output_mask
    if not isinstance(support, Tensor) or support.ndim != 1 or support.dtype != torch.bool:
        raise ValueError("trainer output_mask must be a one-dimensional boolean tensor")
    selected = responses[valid]
    if ((selected < 0) | (selected >= support.numel())).any():
        raise ValueError("rollout response token outside vocabulary")
    if not support.to(selected.device)[selected].all():
        raise ValueError("rollout response token outside output support")
    # Check the sampler's original representation, never only the rebuilt one.
    check_response_tokens(ids, attention, positions, responses, valid)
    stops = set(trainer.stop_token_ids)
    cpu_responses = responses.detach().cpu()
    cpu_valid = valid.detach().cpu()
    texts = [trainer.actor_tokenizer.decode(
        [token for token in row[row_valid].tolist() if token not in stops],
        skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for row, row_valid in zip(cpu_responses, cpu_valid)]
    empty, repeated = response_degeneracy(texts, newline_run=trainer.cfg.degenerate_newline_run)
    rows = []
    for i in range(batch):
        selected_positions = positions[i, valid[i]]
        start = int(selected_positions[0])
        length = int(lengths[i])
        if not torch.equal(selected_positions, torch.arange(start, start + length, device=ids.device)):
            raise ValueError("rollout response positions must be contiguous")
        prefix_mask = attention[i, :start].bool()
        prefix = ids[i, :start][prefix_mask]
        if prefix.numel() == 0:
            raise ValueError("rollout prompt prefix must be nonempty")
        expected_attention = torch.zeros_like(attention[i], dtype=torch.bool)
        expected_attention[start - prefix.numel():start + length] = True
        if not torch.equal(attention[i].bool(), expected_attention):
            raise ValueError("rollout must contain a left-padded prompt and right-padded response")
        if i % trainer.cfg.group_size and not torch.equal(prefix, rows[-1].prefix):
            raise ValueError("rollout prompt prefix differs within a prompt group")
        rows.append(_Row(prefix, responses[i, valid[i]], rendered[i], reasons[i],
                         empty[i] or repeated[i],
                         rollout_logp[i, valid[i]] if rollout_logp is not None else None))
    return rows


def _bad_group(rows: list[_Row]) -> bool:
    return all(row.finish_reason == "length" or row.degenerate for row in rows)


def _pack_rows(trainer, rows: list[_Row], original):
    """Rebuild common widths from actual tokens, never from previous padding."""
    prompt_width = max(row.prefix.numel() for row in rows)
    response_width = max(row.response.numel() for row in rows)
    batch = len(rows)
    pad_id = trainer.actor_tokenizer.pad_token_id
    ids = original[0].new_full((batch, prompt_width + response_width), pad_id)
    attention = original[1].new_zeros(ids.shape)
    responses = original[3].new_full((batch, response_width), pad_id)
    mask = original[4].new_zeros(responses.shape)
    available_logps = [row.rollout_logp for row in rows if row.rollout_logp is not None]
    if available_logps and len(available_logps) != batch:
        raise ValueError('cannot repack mixed present and missing rollout logprobabilities')
    logp = None
    if available_logps:
        dtype = available_logps[0].dtype
        for values in available_logps[1:]:
            dtype = torch.promote_types(dtype, values.dtype)
        logp = torch.zeros(responses.shape, device=ids.device, dtype=dtype)
    for i, row in enumerate(rows):
        length = row.response.numel()
        ids[i, prompt_width - row.prefix.numel():prompt_width] = row.prefix
        ids[i, prompt_width:prompt_width + length] = row.response
        attention[i, prompt_width - row.prefix.numel():prompt_width + length] = 1
        responses[i, :length] = row.response
        mask[i, :length] = 1
        if logp is not None:
            logp[i, :length] = row.rollout_logp
    positions = torch.arange(prompt_width, prompt_width + response_width,
                             device=ids.device).expand(batch, -1)
    result = (ids, attention, positions, responses, mask,
              [row.rendered for row in rows], [row.finish_reason for row in rows])
    if logp is not None or len(original) == 8:
        result = (*result, logp)
    check_response_tokens(*result[:5])
    return result


def select_training_rollout(trainer, prompts):
    """Sample all groups, retry bad groups once, then keep complete good groups.

    Called by soft-length training only. A group is bad when each completion
    is truncated or degenerate, including groups mixing both failure types. Every sampler output
    is validated before filtering. Statistics count all generated response
    tokens, including original/retry samples that are subsequently discarded.
    This function does not write rollout artifacts or update trainer counters.
    """
    prompts = list(prompts)
    if not prompts or trainer.cfg.group_size < 1:
        raise ValueError("rollout selection requires prompts and a positive group_size")
    group_size = trainer.cfg.group_size
    original = trainer.rollout(prompts)
    rows = _validate_sample(trainer, original, len(prompts))
    groups = [rows[i:i + group_size] for i in range(0, len(rows), group_size)]
    bad_indices = [i for i, group in enumerate(groups) if _bad_group(group)]
    stats = {"input_prompt_groups": len(prompts), "kept_prompt_groups": len(prompts),
             "resampled_groups": len(bad_indices), "skipped_groups": 0,
             "generated_response_tokens": sum(row.response.numel() for row in rows)}
    if not bad_indices:
        return original, prompts, stats
    retry = trainer.rollout([prompts[i] for i in bad_indices])
    retry_rows = _validate_sample(trainer, retry, len(bad_indices))
    stats["generated_response_tokens"] += sum(row.response.numel() for row in retry_rows)
    for retry_index, original_index in enumerate(bad_indices):
        replacement = retry_rows[retry_index * group_size:(retry_index + 1) * group_size]
        if not torch.equal(groups[original_index][0].prefix, replacement[0].prefix):
            raise ValueError("rollout retry changed the prompt token prefix")
        if _bad_group(replacement):
            groups[original_index] = []
            stats["skipped_groups"] += 1
        else:
            groups[original_index] = replacement
    kept_prompts = [prompt for prompt, group in zip(prompts, groups) if group]
    stats["kept_prompt_groups"] = len(kept_prompts)
    if not kept_prompts:
        return None, kept_prompts, stats
    merged = _pack_rows(trainer, [row for group in groups for row in group], original)
    return merged, kept_prompts, stats
