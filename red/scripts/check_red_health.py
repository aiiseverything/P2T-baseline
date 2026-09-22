"""Health check for a RED run, reading only ``metrics.jsonl``.

    python red/scripts/check_red_health.py --report reports/red250 [--window 10]
    python red/scripts/check_red_health.py --report reports/red250 \
        --stop-on-problem --pidfile runs/red250/train.pid

Written for this arm rather than reusing the sibling's checker because the
thresholds differ: RED reports no attribution, and the two quantities the P2T
checker flags as expected-but-noteworthy (`credit_ess_ratio` near 1, a high flat
fraction) have different meanings here.  RED's credit share is the positive part
of a signed redistribution, so a flat share means the reward is spread evenly and
a concentrated one means a few tokens carry it -- neither is a failure.

Exit code 1 iff a problem was found, so a watcher can branch on it.  The
collapse signatures the retired advantage rule produced -- every token pushed the
same way, the redistributed term shrinking to nothing against the sequence
advantage, and the flip fraction sitting flat at zero -- are problems here, not
notes: ``red250`` reported all three from rollout 26 onward for 40 more rollouts
before it crashed, and a checker that only notes them does not stop that.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
from pathlib import Path

# The protocol stamps, mirrored from ``red/reward.py`` rather than imported: this
# script is invoked standalone (``python red/scripts/check_red_health.py``), which
# puts ``red/scripts`` on sys.path and not the repository root, and it is kept
# dependency-free for the same reason the paper extraction script is.
# ``tests/red/test_check_red_health.py`` asserts the mirror, so drift between the
# two fails the suite instead of silently mis-certifying a run.
RED_PROTOCOL = "prefix_difference_eq6"
RLOO_ADVANTAGE_RULE = "r4_seq_advantage_plus_centered_credit"
RETIRED_RLOO_ADVANTAGE_RULE = "loo_scalar_baseline_r3"

# Keys that must be finite on every row, or the run is not trustworthy.
FINITE = ("loss", "grad_norm", "raw_reward_mean", "response_entropy", "kl_to_init",
          "mean_response_tokens", "red_token_reward_abs_mean", "credit_ess_ratio",
          "rloo_baseline_mean", "advantage_abs_mean", "group_sigma_mean")
# Keys the sibling arms also report; their absence means the arms' metric
# vocabularies have drifted apart and cross-arm plots would silently break.
SHARED = ("raw_reward_mean", "mean_response_tokens", "response_entropy", "kl_to_init",
          "group_sigma_mean", "credit_ess_ratio", "reward_count", "optimizer_steps",
          "truncated_responses")


def rate(rows, field, window):
    """Half-vs-half mean shift over the last ``window`` rows.

    A window's own scatter is the bar a shift has to clear: comparing the first
    half against the second and requiring the gap to exceed the window's standard
    deviation tests the trend against noise instead of against a fixed slope.
    """
    values = [row.get(field) for row in rows[-window:] if isinstance(row.get(field), (int, float))]
    if len(values) < max(2, window // 2):
        return None
    half = len(values) // 2
    early = sum(values[:half]) / half
    late = sum(values[half:]) / (len(values) - half)
    spread = (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5
    return (late - early) if abs(late - early) > spread else 0.0


def stop_run(pidfile, problems):
    """Ask the training process to stop, after a problem it cannot recover from.

    Only ever reached through ``--stop-on-problem``.  Killing a run is
    destructive enough that the launcher has to ask for it explicitly, and the
    reason is printed so the health log records why a run was stopped next to the
    run itself.  This stops the process; it does not remove anything, so the most
    recent checkpoint stays on disk and the failure stays inspectable.
    """
    path = Path(pidfile)
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        print(f"  stop-on-problem: no usable pid file at {path}; leaving the run alone")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"  stop-on-problem: pid {pid} is already gone")
    except PermissionError:
        print(f"  stop-on-problem: not permitted to signal pid {pid}; leaving the run alone")
    else:
        print(f"  stop-on-problem: sent SIGTERM to pid {pid} for: {problems[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RED run health check")
    parser.add_argument("--report", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stop-on-problem", action="store_true",
                        help="SIGTERM the run named by --pidfile when a problem is found")
    parser.add_argument("--pidfile",
                        help="the run's train.pid, required with --stop-on-problem")
    args = parser.parse_args()
    if args.stop_on_problem and not args.pidfile:
        parser.error("--stop-on-problem needs --pidfile so it cannot signal the wrong process")

    path = Path(args.report) / "metrics.jsonl"
    if not path.is_file():
        print(f"PROBLEM: no metrics at {path}")
        return 1
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print("PROBLEM: metrics.jsonl has a malformed line")
            return 1
        if "rollout" in row:
            rows.append(row)
    if not rows:
        print("PROBLEM: no completed rollouts yet")
        return 1

    problems, notes = [], []
    last = rows[-1]

    if last.get("optimizer_steps", 0) == 0:
        problems.append("the last rollout took no optimizer step")
    for key in SHARED:
        if key not in last:
            problems.append(f"shared metric {key} is missing; cross-arm comparison would break")
    for key in FINITE:
        value = last.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"{key} is not a finite number ({value!r})")

    # Sampler-vs-trainer agreement: judged on the typical token, because the tail
    # of that difference is bf16 numerics across two kernels by nature.
    if last.get("rollout_logp_abs_error_p50", 0) > 0.05:
        problems.append(f"systematic sampler/trainer log-prob disagreement "
                        f"(p50 {last['rollout_logp_abs_error_p50']:.4f})")
    if last.get("rollout_is_ess_ratio", 1) < 0.9:
        problems.append(f"importance weights are degenerate "
                        f"(ESS ratio {last['rollout_is_ess_ratio']:.4f})")
    if last.get("initial_hf_logp_max_abs_error", 0) > 1e-4:
        problems.append("the trainer's first re-forward disagrees with the sampler")

    # RED-specific: the protocol stamps must survive into every row, or an
    # artifact could be read later as a different method.  The retired rule is
    # named explicitly rather than left as an unknown string, because a red250-era
    # artifact reaching this check is a real and expected situation: its advantage
    # collapsed onto a single sign, so it must not be certified as healthy.
    if last.get("red_advantage_rule") == RETIRED_RLOO_ADVANTAGE_RULE:
        problems.append(f"advantage rule {RETIRED_RLOO_ADVANTAGE_RULE!r} is retired; "
                        f"its estimator collapsed under all-negative groups "
                        f"(see RED_REPRO_NOTES 2.2) and the run is not comparable")
    elif last.get("red_advantage_rule") != RLOO_ADVANTAGE_RULE:
        problems.append(f"unexpected advantage rule {last.get('red_advantage_rule')!r}")
    if last.get("red_protocol") != RED_PROTOCOL:
        problems.append(f"unexpected redistribution protocol {last.get('red_protocol')!r}")

    length = rate(rows, "mean_response_tokens", args.window)
    if length is not None and length < -20:
        problems.append(f"response length falling {length:.0f} tokens/rollout")
    entropy = rate(rows, "response_entropy", args.window)
    if entropy is not None and entropy < -0.02:
        problems.append(f"entropy falling {entropy:.4f} nats/rollout (collapse?)")
    if last.get("kl_to_init", 0) > 5:
        problems.append(f"KL to init is {last['kl_to_init']:.2f}; the policy has drifted far")
    count = last.get("reward_count", 0)
    if count and last.get("truncated_responses", 0) > 0.5 * count:
        problems.append(f"{last['truncated_responses']}/{count} responses were truncated")
    if last.get("group_sigma_mean", 1) <= 1e-6:
        problems.append("every prompt group has near-zero reward spread")

    # The collapse signatures the retired rule produced.  A sequence-contrast
    # estimator keeps the share of upward-pushed tokens near a half, and RED's term
    # near the scale of the sequence advantage it rides on; when the advantage
    # collapses onto one signed constant both extremes are reached at once, which
    # is exactly what red250 reported from rollout 26 onward.
    pushed = last.get("red_positive_advantage_fraction")
    if isinstance(pushed, (int, float)) and not 0.05 < pushed < 0.95:
        problems.append(f"{pushed:.3f} of tokens are pushed the same way; the advantage "
                        f"has collapsed onto one sign (red250 reached 1.000)")
    bonus = last.get("red_bonus_over_advantage")
    if isinstance(bonus, (int, float)) and not 0.02 < bonus < 50:
        problems.append(f"the redistributed term is {bonus:.4g}x the sequence advantage; "
                        f"one of the two terms has stopped mattering")
    # Judged over a window rather than on the last row, because a single quiet
    # rollout is ordinary.  This is the signal red250 actually showed: from rollout
    # 26 on, the redistribution stopped changing any token's update direction.
    flip_values = [row.get("red_advantage_flip_fraction") for row in rows[-args.window:]]
    flip_values = [value for value in flip_values if isinstance(value, (int, float))]
    quiet = sum(value < 0.01 for value in flip_values)
    if len(flip_values) >= max(2, args.window // 2) and quiet > len(flip_values) / 2:
        problems.append(f"the redistribution has stopped moving the update direction "
                        f"(flip fraction below 0.01 on {quiet}/{len(flip_values)} recent "
                        f"rollouts); this is the red250 collapse signature")

    # Notes, not problems.  A flat credit share means the redistributed reward is
    # spread evenly across the response; a concentrated one means a few tokens
    # carry it.  RED's rewards are signed differences, so unlike the sibling arm's
    # softmax neither end is a failure -- it is what the redistribution does.
    share = last.get("red_share_max_mean")
    if isinstance(share, (int, float)) and share < 0.02:
        notes.append(f"redistributed credit is very flat (mean max share {share:.4f})")
    flat = last.get("red_flat_response_fraction")
    if isinstance(flat, (int, float)) and flat >= 0.99:
        notes.append("nearly every response has a flat credit share")
    flip = last.get("red_advantage_flip_fraction")
    if isinstance(flip, (int, float)) and flip < 0.01:
        notes.append(f"RED barely changes the update direction vs plain RLOO "
                     f"(flip fraction {flip:.4f}); see RED_REPRO_NOTES 2.6")

    print(f"rollout {last.get('rollout')}: raw_reward {last.get('raw_reward_mean')}, "
          f"loss {last.get('loss')}, grad_norm {last.get('grad_norm')}, "
          f"tokens {last.get('mean_response_tokens')}, "
          f"token_reward_abs_mean {last.get('red_token_reward_abs_mean')}, "
          f"flip {last.get('red_advantage_flip_fraction')}, "
          f"pushed-up {last.get('red_positive_advantage_fraction')}, "
          f"bonus/advantage {last.get('red_bonus_over_advantage')}")
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if not problems:
        print("  ok")
    if problems and args.stop_on_problem:
        stop_run(args.pidfile, problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
