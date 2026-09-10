#!/usr/bin/env python3
"""Plot offline RM eval curves (step vs fixed-prompt-set score) from eval.jsonl.

Reads the output of scripts/eval_checkpoints.py and renders one panel per
generation temperature: x = checkpoint step, y = mean RM score on the frozen
256-prompt validation set, band = bootstrap 95% CI.  One line per run.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="Offline RM eval — frozen 256 validation prompts")
    args = p.parse_args()

    # rows[run][temp][step] -> list of scores; lens likewise
    rows, lens = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), \
                 defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(args.eval_jsonl) as f:
        for line in f:
            r = json.loads(line)
            rows[r["run"]][r["temp"]][r["step"]].append(r["score"])
            lens[r["run"]][r["temp"]][r["step"]].append(r["response_tokens"])
    temps = sorted({t for run in rows.values() for t in run})
    runs = sorted(rows)

    fig, axes = plt.subplots(1, len(temps), figsize=(6.2 * len(temps), 4.8),
                             dpi=160, squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for col, t in enumerate(temps):
        ax = axes[0][col]
        ax.set_facecolor(SURFACE)
        for i, run in enumerate(runs):
            steps = sorted(rows[run][t])
            means = [statistics.mean(rows[run][t][s]) for s in steps]
            # 95% CI of the MEAN (normal approx matches the bootstrap CI in
            # summary.json closely at n=256), not the score distribution.
            ses = [1.96 * (statistics.stdev(rows[run][t][s]) / len(rows[run][t][s]) ** 0.5)
                   for s in steps]
            xs = steps
            los = [m - e for m, e in zip(means, ses)]
            his = [m + e for m, e in zip(means, ses)]
            color = SERIES[i % len(SERIES)]
            ax.fill_between(xs, los, his, color=color, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=4,
                    zorder=3, label=run)
            mean_len = statistics.mean(lens[run][t][steps[-1]])
            ax.annotate(f"{run} (end {means[-1]:.2f}, {mean_len:.0f} tok)",
                        xy=(xs[-1], means[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK_2, fontsize=8.5, zorder=4)
        ax.set_title(f"temp {t}" + (" (primary)" if t == temps[0] else " (greedy column)"),
                     color=INK_2, fontsize=10, loc="left")
        ax.set_xlabel("checkpoint step", color=MUTED, fontsize=10)
        if col == 0:
            ax.set_ylabel("mean RM score (256 prompts)", color=MUTED, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        # Room for the end-of-line annotations (they render outside the last x).
        ax.margins(x=0.12)
        leg = ax.legend(loc="lower right", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
        for txt in leg.get_texts():
            txt.set_color(INK_2)
    fig.suptitle(args.title, x=0.05, ha="left", color=INK, fontsize=13, fontweight="bold")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
