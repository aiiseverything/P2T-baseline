#!/usr/bin/env python3
"""Training curves for a P2T run.

Panels: the reward-model score the actor is being pushed toward (the headline
curve), the length-shaped reward that Eq. (4) actually standardises, response
length, policy entropy, the ratio of the token bonus to the sequence advantage,
and the attribution softmax's effective sample size.

The last two are diagnostics, not objectives: a flat softmax drives the ESS
toward 1 -- the inert end, same reading as the VPO arms' credit ESS -- and
reduces Eq. (3) to a per-response constant, which is the failure mode that makes
a P2T arm silently behave like GRPO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PANELS = [
    ("raw_reward_mean", "RM raw reward (Eq. 3 R)", "raw reward"),
    ("reward_mean", "length-shaped reward (Eq. 4 input)", "reward"),
    ("mean_response_tokens", "response tokens", "tokens"),
    ("response_entropy", "policy entropy", "nats/token"),
    ("p2t_varying_bonus_over_advantage", "|varying token bonus| / |A^hat|", "ratio"),
    ("credit_ess_ratio", "attribution softmax ESS/T", "1 = flat (inert)"),
]


def load_metrics(run_dir: Path):
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no metrics at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True,
                        help="run directory, e.g. reports/smoke10 or runs/smoke10/report")
    parser.add_argument("--out", default=None, help="output PNG (default: <run>/reward.png)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.run)
    rows = load_metrics(run_dir)
    steps = [row["rollout"] for row in rows]
    out = Path(args.out) if args.out else run_dir / "reward.png"

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (key, title, ylabel) in zip(axes.flat, PANELS):
        values = [row.get(key) for row in rows]
        if all(value is None for value in values):
            axis.set_visible(False)
            continue
        points = [(step, value) for step, value in zip(steps, values) if value is not None]
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.6)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("rollout")
        axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(alpha=.3)

    title = args.title or f"P2T baseline — {run_dir.name}"
    first = rows[0]
    subtitle = (f"omega={first.get('p2t_omega')}, alpha={first.get('p2t_alpha')}, "
                f"method={first.get('p2t_protocol')}, sigma0={first.get('sigma0')}")
    figure.suptitle(f"{title}\n{subtitle}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=150)
    print(f"wrote {out}")

    summary = {
        "run": run_dir.name,
        "rollouts": len(rows),
        "raw_reward_first": rows[0].get("raw_reward_mean"),
        "raw_reward_last": rows[-1].get("raw_reward_mean"),
        "raw_reward_mean": _mean([row.get("raw_reward_mean") for row in rows]),
        "sigma0": rows[0].get("sigma0"),
        "alpha": rows[0].get("p2t_alpha"),
        "omega": rows[0].get("p2t_omega"),
        "credit_ess_ratio": _mean([row.get("credit_ess_ratio") for row in rows]),
        "p2t_flat_response_fraction": _mean([row.get("p2t_flat_response_fraction") for row in rows]),
        "p2t_onehot_response_fraction": _mean([row.get("p2t_onehot_response_fraction") for row in rows]),
        "p2t_bonus_over_advantage": _mean([row.get("p2t_bonus_over_advantage") for row in rows]),
        "p2t_varying_bonus_over_advantage": _mean(
            [row.get("p2t_varying_bonus_over_advantage") for row in rows]),
        "p2t_sign_flip_fraction": _mean([row.get("p2t_sign_flip_fraction") for row in rows]),
        "p2t_zero_attribution_share_mass": _mean(
            [row.get("p2t_zero_attribution_share_mass") for row in rows]),
        "p2t_unmapped_share_mean": _mean([row.get("p2t_unmapped_share_mean") for row in rows]),
        "mean_response_tokens": _mean([row.get("mean_response_tokens") for row in rows]),
        "elapsed_sec_total": sum(row.get("elapsed_sec") or 0 for row in rows),
        "actor_peak_gb": max((row.get("actor_peak_gb") or 0) for row in rows),
        "reward_peak_gb": max((row.get("reward_peak_gb") or 0) for row in rows),
        "figure": str(out.relative_to(run_dir.parent)) if run_dir.parent in out.parents else str(out),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


def _mean(values):
    numbers = [value for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


if __name__ == "__main__":
    main()
