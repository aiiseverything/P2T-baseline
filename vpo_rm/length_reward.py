"""Calibrated response-length costs and explicit text-degeneracy checks."""
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
    """Return nonnegative short/long costs in frozen RM-scale units.

    Lengths count generated tokens including EOS, excluding padding. They do
    not determine generation's minimum length or whether a response is empty.
    """
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
    """Separately flag empty/whitespace text and consecutive literal newlines.

    The caller removes stop tokens before decoding. Neither normal newlines nor
    short nonempty answers are classified as degenerate by this function.
    """
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

    Invalid completions are excluded, including their scores. Eligible constant
    groups contribute a zero std; they are not silently removed from the median.
    An absent, nonfinite, or nonpositive calibration is an explicit failure.
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
    # Double precision avoids avoidable loss in the frozen calibration statistic.
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
