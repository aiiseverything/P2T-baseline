"""The project's soft length window, mirrored so the he20 arm is comparable.

``soft_length_penalties`` is a line-for-line port of
``vpo_rm/length_reward.py:soft_length_penalties``: costs are expressed in frozen
reward-model units scaled by the initial-policy calibration ``sigma0``, so the
length shaping is invariant to the absolute RM scale.  A response shorter than
``short_threshold`` is charged up to ``short_strength * sigma0``, one longer than
``long_threshold`` up to ``long_strength * sigma0``, and nothing in between.

These costs change the *sequence* reward that the GRPO advantage is computed from,
exactly as in the sibling arms.  he20 has no per-token reward, so they are never
redistributed across the response and never enter the entropy mask.  See
``HE20_REPRO_NOTES.md``.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral, Real

import torch
from torch import Tensor


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@torch.no_grad()
def soft_length_penalties(lengths: Tensor, sigma0: float, *, short_threshold=8,
                          long_threshold=1024, max_length=2048,
                          short_strength=.5, long_strength=2.) -> tuple[Tensor, Tensor]:
    """Nonnegative short/long costs in frozen RM-scale units."""
    sigma0 = _finite_number(sigma0, "sigma0")
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive")
    short_threshold = _finite_number(short_threshold, "short_threshold")
    long_threshold = _finite_number(long_threshold, "long_threshold")
    max_length = _finite_number(max_length, "max_length")
    if not 0 < short_threshold <= long_threshold < max_length:
        raise ValueError("Require 0 < short_threshold <= long_threshold < max_length")
    short_strength = _finite_number(short_strength, "short_strength")
    long_strength = _finite_number(long_strength, "long_strength")
    if short_strength < 0 or long_strength < 0:
        raise ValueError("Penalty strengths must be nonnegative")
    if not isinstance(lengths, Tensor) or lengths.ndim != 1 or lengths.is_complex():
        raise ValueError("lengths must be a real one-dimensional tensor")
    if not torch.isfinite(lengths).all() or ((lengths < 0) | (lengths > max_length)).any():
        raise ValueError("lengths must be finite and between zero and max_length")
    values = lengths if lengths.dtype == torch.float64 else lengths.float()
    short = ((short_threshold - values) / short_threshold).clamp(0, 1) * (short_strength * sigma0)
    long = ((values - long_threshold) / (max_length - long_threshold)).clamp(0, 1) * (long_strength * sigma0)
    if not torch.isfinite(short).all() or not torch.isfinite(long).all():
        raise ValueError("Length penalties overflow the tensor's numeric range")
    return short, long


def response_degeneracy(texts: Sequence[str], newline_run=32) -> tuple[list[bool], list[bool]]:
    """Flag empty/whitespace text and runs of literal newlines, separately."""
    if isinstance(newline_run, bool) or not isinstance(newline_run, Integral) or newline_run < 1:
        raise ValueError("newline_run must be a positive integer")
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ValueError("texts must be a sequence of strings")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must contain only strings")
    repeated_newlines = "\n" * newline_run
    return ([text.strip() == "" for text in texts],
            [repeated_newlines in text for text in texts])


@torch.no_grad()
def calibrate_reward_scale(rewards: Tensor, group_ids: Tensor, valid: Tensor) -> float:
    """Median population std across groups with at least two valid completions.

    This is also the direct, empirical measurement of the RM's score scale that
    the paper never states: the he20 diagnostics report it next to the raw reward
    so the reader can see what scale the length penalties -- expressed in
    ``sigma0`` units -- and the group-relative advantages live on.
    """
    if any(not isinstance(x, Tensor) for x in (rewards, group_ids, valid)):
        raise ValueError("rewards, group_ids and valid must be tensors")
    if rewards.ndim != 1 or group_ids.shape != rewards.shape or valid.shape != rewards.shape:
        raise ValueError("rewards, group_ids and valid must have shape [B]")
    if group_ids.device != rewards.device or valid.device != rewards.device:
        raise ValueError("Calibration tensors must be on the same device")
    if rewards.is_complex() or group_ids.is_complex() or valid.is_complex():
        raise ValueError("Calibration tensors must be real")
    if not torch.isfinite(group_ids).all():
        raise ValueError("group_ids must be finite")
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("valid must be a binary mask")
    selected = valid.bool()
    if not torch.isfinite(rewards[selected]).all():
        raise ValueError("Valid calibration rewards must be finite")
    values = rewards.double()
    scales = []
    for group in group_ids.unique():
        sample = values[selected & (group_ids == group)]
        if sample.numel() >= 2:
            scales.append(sample.std(correction=0))
    if not scales:
        raise ValueError("Calibration needs a group with at least two valid completions")
    scales = torch.stack(scales).sort().values
    if not torch.isfinite(scales).all():
        raise ValueError("Calibration group standard deviations must be finite")
    middle = scales.numel() // 2
    sigma0 = (scales[middle] if scales.numel() % 2
              else scales[middle - 1] / 2 + scales[middle] / 2).item()
    if not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("Calibrated sigma0 must be finite and positive")
    return sigma0


@torch.no_grad()
def guard_degenerate_rewards(rewards: Tensor, lengths: Tensor, group_ids: Tensor,
                             min_length: int = 1, penalty: float = 1.0):
    """Floor flagged responses to their group minimum minus ``penalty``.

    Port of ``vpo_rm/core.py:guard_degenerate_rewards``.  The floored value is
    cross-response coupled, which is why it is applied to the shaped *sequence*
    reward and never to the reward model's own score: R stays the frozen reward
    model's output for every response.
    """
    if rewards.ndim != 1 or lengths.shape != rewards.shape or group_ids.shape != rewards.shape:
        raise ValueError("rewards, lengths and group_ids must share shape [B]")
    if min_length < 1 or penalty < 0:
        raise ValueError("min_length must be positive and penalty non-negative")
    rewards = rewards.clone()
    flagged = lengths < min_length
    if not bool(flagged.any()):
        return rewards, 0
    for group in group_ids[flagged].unique():
        selected = group_ids == group
        floor = rewards[selected].min() - penalty
        rewards[selected & flagged] = floor
    return rewards, int(flagged.sum())
