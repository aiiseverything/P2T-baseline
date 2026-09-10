#!/usr/bin/env python3
"""Plot training reward curves for one or more runs from their metrics.jsonl.

Usage:
    python3 scripts/plot_reward_curves.py \
        --run grpo=runs/formal-...-grpo-.../metrics.jsonl \
        --run vpo_rm=runs/formal-...-vpo_rm-.../metrics.jsonl \
        --output runs/reward_curve_grpo_vs_vpo.png

Each point is the mean RM score of one rollout's responses; the bold line is a
trailing mean (default window 25) that suppresses the prompt-sampling noise.
Palette and chrome follow the dataviz reference (light mode, 2 categorical slots).
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # slots 1-4


def load_series(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = sorted((r for r in rows if "reward_mean" in r), key=lambda r: r["rollout"])
    return [r["rollout"] for r in rows], [r["reward_mean"] for r in rows]


def trailing_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(st.mean(values[lo:i + 1]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="LABEL=PATH", help="label and metrics.jsonl path; repeatable")
    p.add_argument("--output", default="runs/reward_curve.png")
    p.add_argument("--window", type=int, default=25, help="trailing-mean window")
    p.add_argument("--title", default="Training reward — per rollout")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    subtitle_parts = []
    for i, spec in enumerate(args.run):
        label, path = spec.split("=", 1)
        x, y = load_series(Path(path))
        color = SERIES[i % len(SERIES)]
        ax.scatter(x, y, s=11, color=color, alpha=0.30, linewidths=0, zorder=2)
        avg = trailing_mean(y, args.window)
        ax.plot(x, avg, color=color, linewidth=2, zorder=3,
                label=f"{label} (trailing mean, w={args.window})")
        # direct label at the line's end
        ax.annotate(label, xy=(x[-1], avg[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    color=INK_2, fontsize=9, fontweight="bold", zorder=4)
        mx = f"{st.mean(y):.2f}" if y else "-"
        subtitle_parts.append(f"{label}: {x[-1]} rollouts · mean {mx}")

    # base-model reference: rollout 1 of the first run
    x0, y0 = load_series(Path(args.run[0].split("=", 1)[1]))
    if x0 and x0[0] == 1:
        ax.scatter([1], [y0[0]], s=90, facecolors="none", edgecolors=INK_2,
                   linewidths=1.4, zorder=5)
        ax.annotate("rollout 1 = base model", xy=(1, y0[0]), xytext=(8, -14),
                    textcoords="offset points", color=INK_2, fontsize=8.5, zorder=5)

    fig.suptitle(args.title, x=0.055, ha="left", color=INK,
                 fontsize=13, fontweight="bold")
    ax.set_title(" · ".join(subtitle_parts), loc="left", color=MUTED,
                 fontsize=9, pad=10)

    ax.set_xlabel("rollout", color=MUTED, fontsize=10)
    ax.set_ylabel("mean RM reward per rollout", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    leg = ax.legend(loc="lower right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts():
        t.set_color(INK_2)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
