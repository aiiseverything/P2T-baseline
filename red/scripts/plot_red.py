"""Plot a RED run's curves from ``metrics.jsonl``.

    python red/scripts/plot_red.py --run reports/red250 [--out reward.png]

A RED-owned plotter rather than the sibling arm's, for one substantive reason: the
sibling's panels key on ``p2t_*`` metrics (the attribution share, the varying-bonus
ratio) that a RED run does not emit, so pointed at a RED report it silently hides
those panels and titles the figure with the wrong arm.  The shared metric names are
reused verbatim so the two arms' reward, length, entropy and KL panels look the
same.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (key, panel title, y-label)
PANELS = [
    ("raw_reward_mean", "reward model score", "raw reward"),
    ("reward_mean", "shaped reward (length window applied)", "shaped reward"),
    ("mean_response_tokens", "response length", "tokens"),
    ("response_entropy", "response entropy", "nats"),
    ("red_advantage_flip_fraction",
     "RED vs plain-RLOO direction flips (RED_REPRO_NOTES 2.6)", "fraction"),
    ("credit_ess_ratio", "redistributed-credit concentration", "ESS ratio"),
]


def load(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "rollout" in row:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RED run metrics")
    parser.add_argument("--run", required=True, help="the report directory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    report = Path(args.run)
    rows = load(report / "metrics.jsonl")
    if not rows:
        print(f"no completed rollouts in {report / 'metrics.jsonl'}")
        return 1

    figure, axes = plt.subplots(2, 3, figsize=(15, 7))
    steps = [row["rollout"] for row in rows]
    for axis, (key, title, label) in zip(axes.ravel(), PANELS):
        values = [row.get(key) for row in rows]
        if all(value is None or (isinstance(value, float) and not math.isfinite(value))
               for value in values):
            axis.set_visible(False)
            continue
        axis.plot(steps, values, linewidth=1.2)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("rollout")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    last = rows[-1]
    title = args.title or f"RED (RLOO) — {report.name}"
    subtitle = (f"{last['rollout']} rollouts | beta_c {last.get('red_beta_c')} | "
                f"KL beta {last.get('red_beta')} | "
                f"advantage rule {last.get('red_advantage_rule')} | "
                f"protocol {last.get('red_protocol')} | sigma0 {last.get('sigma0')}")
    figure.suptitle(f"{title}\n{subtitle}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out) if args.out else report / "reward.png"
    figure.savefig(out, dpi=130)
    summary = {
        "run": report.name,
        "rollouts": last["rollout"],
        "raw_reward_first": rows[0].get("raw_reward_mean"),
        "raw_reward_last": last.get("raw_reward_mean"),
        "raw_reward_mean": sum(r.get("raw_reward_mean", 0.0) for r in rows) / len(rows),
        "beta_c": last.get("red_beta_c"),
        "advantage_rule": last.get("red_advantage_rule"),
        "redistribution_protocol": last.get("red_protocol"),
        "credit_ess_ratio": last.get("credit_ess_ratio"),
        "advantage_flip_fraction": last.get("red_advantage_flip_fraction"),
        "mean_response_tokens": sum(r.get("mean_response_tokens", 0.0) for r in rows) / len(rows),
        "kl_to_init_last": last.get("kl_to_init"),
        "sigma0": last.get("sigma0"),
        "actor_peak_gb": last.get("actor_peak_gb"),
        "reward_peak_gb": last.get("reward_peak_gb"),
        "figure": str(out.relative_to(report.parent) if out.is_relative_to(report.parent) else out),
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out} and {report / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
