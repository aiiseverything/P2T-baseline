"""Rollout-level validation and selection, mirroring the parent project.

Two things the parent does that change which data a run actually trains on:

1. ``validate_response_termination`` -- a row whose finish reason says "stop"
   must really end in a stop token, and a "length" row must really fill the cap.
   Without it a mis-labelled row silently mis-scores the truncation count, the
   long-length penalty and the degeneracy flags.
2. ``select_training_rollout`` -- a prompt group in which *every* response is
   truncated or degenerate carries no usable signal, so it is resampled once and
   dropped if it is still bad.  Training on such a group would also make the RED
   arm see a different prompt population than the GRPO and VPO-RM arms it is
   compared against.

Port of ``vpo_rm/integration.py:validate_response_termination`` and
``vpo_rm/rollout_selection.py:select_training_rollout``.
"""
from __future__ import annotations

import torch

ROLLOUT_FIELDS = ("input_ids", "full_mask", "positions", "responses", "response_mask",
                  "rendered", "finish_reasons", "rollout_logprobs")


def validate_response_termination(responses: torch.Tensor, response_mask: torch.Tensor,
                                  finish_reasons, stop_token_ids, max_response_tokens: int) -> None:
    """Require terminal EOS for stopped rows and a full cap for truncations."""
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
    stop_ids = torch.as_tensor(tuple(stop_token_ids), dtype=responses.dtype,
                               device=responses.device)
    stopped = torch.isin(responses, stop_ids) & valid
    stop_count = stopped.sum(-1)
    terminal_stop = stopped.gather(1, (lengths - 1)[:, None]).squeeze(1)
    expects_stop = torch.tensor([reason == "stop" for reason in finish_reasons],
                                device=responses.device)
    consistent = torch.where(expects_stop, (stop_count == 1) & terminal_stop,
                             (stop_count == 0) & (lengths == max_response_tokens))
    if not consistent.all():
        raise ValueError("Response termination stop/EOS or length metadata is inconsistent")


def _usable_rows(rollout, flag_degenerate):
    """A response is usable only if it stopped on its own and is not degenerate."""
    responses, rmask, reasons = rollout[3], rollout[4], rollout[6]
    empty, repeated = flag_degenerate(responses, rmask)
    finished = torch.tensor([reason == "stop" for reason in reasons], device=empty.device)
    return finished & ~(empty | repeated)


def _check_sources(rollout, stop_token_ids, max_response_tokens) -> None:
    """Validate a rollout before any of it can be dropped.

    ``_pack`` re-densifies the mask, so a source with an interior hole would be
    silently repaired rather than rejected -- check the sources, not just the
    merged result.  ``_pack`` assumes right-padded rows, so this is the check
    that makes that assumption hold; it is required, not optional.
    """
    if stop_token_ids is None or max_response_tokens is None:
        raise ValueError("select_training_rollout requires stop_token_ids and "
                         "max_response_tokens so sources can be validated")
    validate_response_termination(rollout[3], rollout[4], rollout[6],
                                  stop_token_ids, max_response_tokens)


def _usable_groups(usable, prompt_count: int, group_size: int) -> list[bool]:
    groups = torch.arange(prompt_count * group_size, device=usable.device) // group_size
    return [bool(usable[groups == index].any()) for index in range(prompt_count)]


def _group_rows(group_indices, group_size: int) -> list[int]:
    """Expand kept *group* indices into the *row* indices of a rollout tuple."""
    return [group * group_size + offset for group in group_indices for offset in range(group_size)]


def _needed_width(rollout, indices) -> int:
    rows = torch.as_tensor(indices, device=rollout[4].device)
    return int(rollout[4][rows].sum(-1).max().item())


def _pack(rollout, indices, width: int, prompt_width: int, pad_token_id: int, device):
    """Rebuild an 8-field rollout from selected rows.

    Response columns are trimmed or zero-padded to ``width`` and the prompt block
    is **left**-padded to the batch-common ``prompt_width``: the actor tokenizer
    pads on the left, and a resampled batch holds fewer prompts so its own
    prompt block is narrower.  Without this the pieces cannot be concatenated.

    ``width`` is the widest piece's need, so a piece whose own response block is
    *narrower* than ``width`` must be padded up to it, not sliced to it: its rows
    can only supply as many columns as the block holds, and a slice narrower than
    ``keep`` is a shape error rather than a trim.  The pad tail lands outside
    every row's mask, so the loss already ignores it.
    """
    input_ids, full_mask, positions, responses, response_mask, rendered, reasons, logprobs = rollout
    rows = torch.as_tensor(indices, device=device)
    selected = responses[rows]
    selected_mask = response_mask[rows]
    usable = min(width, responses.shape[1])
    keep = torch.arange(usable, device=device)[None, :] < selected_mask[:, :usable].sum(-1, keepdim=True)
    packed_mask = torch.zeros((len(indices), width), dtype=response_mask.dtype, device=device)
    packed_mask[:, :usable] = (selected_mask[:, :usable].bool() & keep).to(response_mask.dtype)
    packed_responses = torch.full((len(indices), width), pad_token_id, dtype=responses.dtype,
                                  device=device)
    packed_responses[:, :usable] = selected[:, :usable]
    packed_logprobs = torch.zeros((len(indices), width), dtype=torch.float32, device=device)
    packed_logprobs[:, :usable] = logprobs[rows][:, :usable]

    source_width = input_ids.shape[1] - responses.shape[1]
    if source_width > prompt_width:
        raise ValueError("Common prompt width is narrower than a source rollout's")
    deficit = prompt_width - source_width
    prompt_block = input_ids[rows][:, :source_width]
    prompt_mask = full_mask[rows][:, :source_width]
    if deficit:
        prompt_block = torch.nn.functional.pad(prompt_block, (deficit, 0), value=pad_token_id)
        prompt_mask = torch.nn.functional.pad(prompt_mask, (deficit, 0), value=0)
    packed_input = torch.cat([prompt_block, packed_responses], dim=1)
    packed_full = torch.cat([prompt_mask, packed_mask.to(full_mask.dtype)], dim=1)
    packed_positions = torch.arange(prompt_width, packed_input.shape[1],
                                    device=device).expand(len(indices), -1)
    return (packed_input, packed_full, packed_positions, packed_responses, packed_mask,
            [rendered[i] for i in indices], [reasons[i] for i in indices], packed_logprobs)


@torch.no_grad()
def select_training_rollout(rollout_fn, prompts, *, group_size: int, pad_token_id: int,
                            flag_degenerate, device, stop_token_ids,
                            max_response_tokens: int):
    """Sample prompt groups, resample wholly-bad groups once, drop what remains.

    ``rollout_fn(prompts)`` must return the bare 8-field rollout tuple, not a
    ``(rollout, summary)`` pair.  Returns ``(rollout_or_None, prompts, stats)``;
    a ``None`` rollout means no group survived, in which case the caller must
    skip the update entirely -- including the KL term, the Adam moments and
    weight decay.
    """
    rollout = rollout_fn(prompts)
    _check_sources(rollout, stop_token_ids, max_response_tokens)
    usable = _usable_rows(rollout, flag_degenerate)
    good = _usable_groups(usable, len(prompts), group_size)
    stats = {"input_prompt_groups": len(prompts),
             "resampled_groups": sum(1 for ok in good if not ok),
             "skipped_groups": 0,
             "generated_response_tokens": int(rollout[4].sum())}

    kept_groups = [index for index, ok in enumerate(good) if ok]
    pieces = [(rollout, _group_rows(kept_groups, group_size),
               [prompts[index] for index in kept_groups])]
    if stats["resampled_groups"]:
        retry_prompts = [prompt for prompt, ok in zip(prompts, good) if not ok]
        retry = rollout_fn(retry_prompts)
        _check_sources(retry, stop_token_ids, max_response_tokens)
        retry_usable = _usable_rows(retry, flag_degenerate)
        retry_good = _usable_groups(retry_usable, len(retry_prompts), group_size)
        stats["skipped_groups"] = sum(1 for ok in retry_good if not ok)
        kept_retry = [index for index, ok in enumerate(retry_good) if ok]
        pieces.append((retry, _group_rows(kept_retry, group_size),
                       [retry_prompts[index] for index in kept_retry]))

    pieces = [(source, indices, kept) for source, indices, kept in pieces if indices]
    if not pieces:
        stats["kept_prompt_groups"] = 0
        return None, [], stats

    width = max(_needed_width(source, indices) for source, indices, _ in pieces)
    # A retry batch holds fewer prompts, so its own chat padding is narrower;
    # every piece is left-padded to the widest prompt block before concatenation.
    # Its response block is narrower too -- a resampled batch only spans the
    # groups that failed, and those are short whenever the failure is length
    # collapse -- so ``width`` can legitimately exceed a piece's own block and
    # ``_pack`` pads that piece up to it rather than slicing past it.
    prompt_width = max(source[0].shape[1] - source[3].shape[1] for source, _, _ in pieces)
    packed = [_pack(source, indices, width, prompt_width, pad_token_id, device)
              for source, indices, _ in pieces]
    merged = tuple(
        sum((piece[position] for piece in packed), []) if position in (5, 6)
        else torch.cat([piece[position] for piece in packed], dim=0)
        for position in range(len(ROLLOUT_FIELDS)))
    merged_prompts = [prompt for _, _, kept in pieces for prompt in kept]
    stats["kept_prompt_groups"] = len(merged_prompts)
    stats["generated_response_tokens"] = int(merged[4].sum())
    return merged, merged_prompts, stats
