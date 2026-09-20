#!/usr/bin/env python3
"""Health checks over a live run's metrics, so trouble is noticed while it runs.

Reads only ``metrics.jsonl``.  Reports every condition that the audits named as
either a silent-degradation risk or a lost-run risk, and exits non-zero if any
of them is active, so a watcher can act on it.

    python scripts/check_run_health.py --report reports/p2t250
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rate(rows, field, window):
    values = [row.get(field) for row in rows[-window:] if isinstance(row.get(field), (int, float))]
    if len(values) < 2:
        return None
    return (values[-1] - values[0]) / (len(values) - 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args(argv)

    path = Path(args.report) / "metrics.jsonl"
    if not path.is_file():
        print("no metrics yet")
        return 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        print("metrics file is empty")
        return 0
    recent = rows[-args.window:]
    last = rows[-1]
    problems = []

    # 1. Training is actually updating.
    if last.get("skipped_rollout"):
        problems.append("last rollout was skipped (no group survived)")
    if last.get("optimizer_steps") in (0, None):
        problems.append(f"no optimizer step on the last rollout: {last.get('optimizer_steps')}")

    # 2. Nothing has gone non-finite.
    for field in ("loss", "grad_norm", "raw_reward_mean", "response_entropy",
                  "kl_to_init", "mean_response_tokens"):
        value = last.get(field)
        if value is None or not math.isfinite(value):
            problems.append(f"{field} is not finite: {value}")

    # 3. The sampler and the trainer are running the same protocol.
    #
    # Judge this on the *typical* token, not the worst one.  The trainer's own
    # re-forward is the thing that must be exact, and it is checked separately
    # below; the sampler-to-HF delta is bf16 numerics across two different
    # kernels and has a heavy tail by nature.  Measured on rollout 1: p50 0.0015
    # nats, p99 0.106, max 0.88 over 40,364 tokens, with the importance weights
    # at ESS 0.9993 and a 0.2% clip fraction -- i.e. one outlier token, not a
    # protocol break.  A real break shifts the median and collapses the ESS, so
    # those are what is gated.
    for field, limit in (("rollout_logp_abs_error_p50", 0.05),
                         ("rollout_logp_abs_error_mean", 0.10)):
        value = last.get(field)
        if isinstance(value, (int, float)) and value > limit:
            problems.append(f"systematic sampler/trainer log-prob disagreement: "
                            f"{field}={value:.3e} > {limit}")
    is_ess = last.get("rollout_is_ess_ratio")
    if isinstance(is_ess, (int, float)) and is_ess < 0.9:
        problems.append(f"rollout importance weights are degenerate (ESS {is_ess:.4f})")
    # The trainable forward at theta = theta_old must reproduce the cached old
    # log-probabilities exactly.  This one has no numerical excuse.
    own = last.get("initial_hf_logp_max_abs_error")
    if isinstance(own, (int, float)) and own > 1e-4:
        problems.append(f"trainer re-forward disagrees with its own cache: {own:.3e}")

    # 4. Credit assignment is doing something.  This is the p9c failure mode:
    #    a flat attribution softmax makes Eq. (3) a per-response constant.
    ess = last.get("credit_ess_ratio")
    flat = last.get("p2t_flat_response_fraction")
    if isinstance(ess, (int, float)) and ess > 0.99:
        problems.append(f"attribution softmax is flat (credit_ess_ratio {ess:.5f}); "
                        f"the token term is inert")
    if isinstance(flat, (int, float)) and flat >= 0.99:
        problems.append(f"{flat:.0%} of responses have a flat attribution softmax")

    # 5. The run is not collapsing or running away.
    length = rate(rows, "mean_response_tokens", args.window)
    entropy = rate(rows, "response_entropy", args.window)
    if length is not None and length < -20:
        problems.append(f"response length falling {length:.0f} tokens/rollout")
    if entropy is not None and entropy < -0.02:
        problems.append(f"entropy falling {entropy:.4f} nats/rollout (collapse?)")
    kl = last.get("kl_to_init")
    if isinstance(kl, (int, float)) and kl > 5:
        problems.append(f"KL to init is {kl:.2f}, policy has drifted far")
    if isinstance(last.get("truncated_responses"), int) and \
            last["truncated_responses"] > 0.5 * max(1, last.get("reward_count", 1)):
        problems.append(f"{last['truncated_responses']}/{last.get('reward_count')} responses "
                        f"hit the length cap")

    # 6. The reward is not degenerating to a constant (no group signal).
    sigma = last.get("group_sigma_mean")
    if isinstance(sigma, (int, float)) and sigma <= 1e-6 + 0:
        problems.append(f"mean group spread is {sigma:.2e}; advantages are ~0")

    summary = (f"rollout {last.get('rollout')}: raw_reward {last.get('raw_reward_mean')}, "
               f"loss {last.get('loss')}, grad_norm {last.get('grad_norm')}, "
               f"tokens {last.get('mean_response_tokens')}, ESS/T {ess}, "
               f"varying/adv {last.get('p2t_varying_bonus_over_advantage')}")
    print(summary)
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
