"""Plot an he20 run's curves from ``metrics.jsonl``.

    python he20/scripts/plot_he20.py --run reports/he20250 [--out reward.png]

An he20-owned plotter rather than a sibling arm's, for one substantive reason: the
interesting panels differ per arm.  RED's figure is about a credit redistribution
and P2T's about an attribution share, and an he20 run emits neither; what it emits
instead is Eq. (6)'s mask -- the realised kept fraction over time against the
configured rho, and the kept tokens' mean entropy against the whole population's
(``HE20_REPRO_NOTES`` 4).  Those two panels are the arm's own diagnostic, and they
are why pointing a sibling's plotter at this arm's report would leave the figure
silent about the one thing the run changes.  The shared reward, length, entropy and
KL panels are drawn from the same metric names the other arms use, so the arms'
figures stay comparable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (panel title, y-label, [(metric key, legend label), ...]).  A panel with no
# drawable series hides itself, which is what keeps an unmasked run (rho = 1) from
# drawing panels that say nothing, and a series whose key the run does not emit is
# skipped rather than drawn as a gap.
PANELS = [
    ("reward model score", "raw reward", [("raw_reward_mean", None)]),
    ("shaped reward (length window applied)", "shaped reward", [("reward_mean", None)]),
    ("response length", "tokens", [("mean_response_tokens", None)]),
    ("response entropy", "nats", [("response_entropy", None)]),
    ("Eq. (6) kept fraction vs rho (HE20_REPRO_NOTES 4)", "fraction",
     [("entropy_top_kept_fraction", "realised kept fraction")]),
    ("kept vs population entropy", "nats",
     [("entropy_top_mean_kept_entropy", "kept tokens (top rho)"),
      ("entropy_top_mean_all_entropy", "all valid tokens")]),
    ("KL to init", "nats", [("kl_to_init", None)]),
    ("advantage |mean|", "|advantage|", [("advantage_abs_mean", None)]),
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


def mean_of(rows, key):
    """Mean of a metric over the run, or None if no row carries a number for it."""
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))
              and not isinstance(row.get(key), bool)]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot he20 run metrics")
    parser.add_argument("--run", required=True, help="the report directory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    report = Path(args.run)
    metrics = report / "metrics.jsonl"
    if not metrics.is_file():
        print(f"no metrics at {metrics}")
        return 1
    rows = load(metrics)
    if not rows:
        print(f"no completed rollouts in {metrics}")
        return 1

    figure, axes = plt.subplots(2, 4, figsize=(19, 8))
    steps = [row["rollout"] for row in rows]
    last = rows[-1]
    mask_axis = None
    for axis, (title, label, series) in zip(axes.ravel(), PANELS):
        drawn = False
        for key, series_label in series:
            values = [row.get(key) for row in rows]
            if not any(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in values):
                continue
            axis.plot(steps, values, linewidth=1.2, label=series_label)
            drawn = True
        if not drawn:
            axis.set_visible(False)
            continue
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("rollout")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        if len(series) > 1:
            axis.legend(fontsize=8)
        if any(key == "entropy_top_kept_fraction" for key, _ in series):
            mask_axis = axis

    # The configured ratio, on the panel that judges the mask against it: the kept
    # fraction is a statistic of the run, rho is what the config asked for, and the
    # paper's claim is about the gap between them.  Only drawn while the mask is on
    # (rho = null/1.0 is the unmasked baseline and has no line to draw).
    if mask_axis is not None:
        configured = last.get("entropy_top_ratio")
        if isinstance(configured, (int, float)) and not isinstance(configured, bool):
            mask_axis.axhline(configured, color="crimson", linestyle="--", linewidth=1.0,
                              alpha=0.8, label=f"entropy_top_ratio = {configured}")
            mask_axis.legend(fontsize=8)

    title = args.title or f"he20 (entropy-top mask, Eq. 6) — {report.name}"
    subtitle = (f"{last['rollout']} rollouts | entropy_top_ratio {last.get('entropy_top_ratio')} "
                f"| rule {last.get('entropy_top_rule')} "
                f"| mask population {last.get('he20_mask_population')} "
                f"| protocol {last.get('he20_protocol')} "
                f"| kept {last.get('entropy_top_kept_fraction')} "
                f"| sigma0 {last.get('sigma0')}")
    figure.suptitle(f"{title}\n{subtitle}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out) if args.out else report / "reward.png"
    figure.savefig(out, dpi=130)
    summary = {
        "run": report.name,
        "rollouts": last["rollout"],
        "raw_reward_first": rows[0].get("raw_reward_mean"),
        "raw_reward_last": last.get("raw_reward_mean"),
        "raw_reward_mean": mean_of(rows, "raw_reward_mean"),
        "entropy_top_ratio": last.get("entropy_top_ratio"),
        "entropy_top_rule": last.get("entropy_top_rule"),
        "mask_protocol": last.get("he20_protocol"),
        "mask_population": last.get("he20_mask_population"),
        "kept_fraction_first": rows[0].get("entropy_top_kept_fraction"),
        "kept_fraction_last": last.get("entropy_top_kept_fraction"),
        "kept_fraction_mean": mean_of(rows, "entropy_top_kept_fraction"),
        "entropy_threshold_last": last.get("entropy_top_threshold"),
        "mean_kept_entropy_last": last.get("entropy_top_mean_kept_entropy"),
        "mean_all_entropy_last": last.get("entropy_top_mean_all_entropy"),
        "mean_response_tokens": mean_of(rows, "mean_response_tokens"),
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
