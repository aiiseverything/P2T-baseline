"""Health check for a RED run, reading only ``metrics.jsonl``.

    python red/scripts/check_red_health.py --report reports/red250 [--window 10]

Written for this arm rather than reusing the sibling's checker because the
thresholds differ: RED reports no attribution, and the two quantities the P2T
checker flags as expected-but-noteworthy (`credit_ess_ratio` near 1, a high flat
fraction) have different meanings here.  RED's credit share is the positive part
of a signed redistribution, so a flat share means the reward is spread evenly and
a concentrated one means a few tokens carry it -- neither is a failure.

Exit code 1 iff a problem was found, so a watcher can branch on it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="RED run health check")
    parser.add_argument("--report", required=True)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

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
    # artifact could be read later as a different method.
    if last.get("red_advantage_rule") != "loo_scalar_baseline_r3":
        problems.append(f"unexpected advantage rule {last.get('red_advantage_rule')!r}")
    if last.get("red_protocol") != "prefix_difference_eq6":
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
          f"flip {last.get('red_advantage_flip_fraction')}")
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if not problems:
        print("  ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
