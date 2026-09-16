"""Shared command-line options for legacy and soft response-length rewards."""
from __future__ import annotations

import argparse


_CONFIG_FIELDS = (
    "length_reward_mode", "length_reward_sigma0", "short_response_threshold",
    "long_response_threshold", "short_penalty_strength", "long_penalty_strength",
    "advantage_std_floor_fraction", "length_calibration_prompts",
    "degenerate_newline_run", "min_response_tokens", "length_penalty_slope",
    "length_penalty_anchor",
)


def add_length_reward_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("response-length reward")
    group.add_argument("--length-reward-mode", choices=("legacy", "soft"), default="legacy")
    group.add_argument("--length-reward-sigma0", type=float,
                       help="Shared initial RM reward standard deviation; omit to calibrate soft mode")
    group.add_argument("--short-response-threshold", type=int, default=8)
    group.add_argument("--long-response-threshold", type=int, default=1024)
    group.add_argument("--short-penalty-strength", type=float, default=.5)
    group.add_argument("--long-penalty-strength", type=float, default=2.)
    group.add_argument("--advantage-std-floor-fraction", type=float, default=.5)
    group.add_argument("--length-calibration-prompts", type=int, default=128)
    group.add_argument("--degenerate-newline-run", type=int, default=32)
    group.add_argument("--min-response-tokens", type=int,
                       help="Defaults to 8 in legacy mode and 0 in soft mode")
    group.add_argument("--length-penalty-slope", type=float, default=0.,
                       help="Legacy linear reward debias below --length-penalty-anchor")
    group.add_argument("--length-penalty-anchor", type=int, default=600)


def length_reward_config_kwargs(args: argparse.Namespace) -> dict:
    """Pass CLI values to TrainerConfig; TrainerConfig.resolved validates modes."""
    return {name: getattr(args, name) for name in _CONFIG_FIELDS}
