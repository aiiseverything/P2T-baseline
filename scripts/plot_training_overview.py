#!/usr/bin/env python3
"""Multi-panel training overview from metrics.jsonl: reward, response length,
and optional method-specific gauges.

Panel 1: per-rollout mean reward (raw + trailing mean) — the "optimization
target" view.  Panel 2: mean response tokens per response — the reward-hacking
detector (a collapse to ~0 means the policy learned to emit a stop token
immediately).  Panel 3 (vpo runs only): credit ESS ratio.

Usage:
  python3 scripts/plot_training_overview.py \
    --run grpo=runs/.../metrics.jsonl --run vpo_rm=runs/.../metrics.jsonl \
    --output runs/overview.png
"""
import argparse
import json
import statistics as st
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return sorted((r for r in rows if "reward_mean" in r), key=lambda r: r["rollout"])


def trailing(values, window):
    return [st.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--title", default="Training overview")
    p.add_argument("--responses-per-rollout", type=int, default=64,
                   help="Divide response_tokens by this for per-response length")
    args = p.parse_args()

    runs = [(spec.split("=", 1)[0], load(spec.split("=", 1)[1])) for spec in args.run]
    show_ess = any("credit_ess_ratio" in r for _, rows in runs for r in rows[:1])
    panels = [("reward", "mean RM reward per rollout"),
              ("length", "response tokens per response")]
    if show_ess:
        panels.append(("ess", "credit ESS ratio"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 3.4 * len(panels)),
                             dpi=160, sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (key, ylabel) in enumerate(panels):
        ax = axes[row][0]
        ax.set_facecolor(SURFACE)
        for i, (label, rows) in enumerate(runs):
            x = [r["rollout"] for r in rows]
            color = SERIES[i % len(SERIES)]
            if key == "reward":
                y = [r["reward_mean"] for r in rows]
                ax.scatter(x, y, s=9, color=color, alpha=0.30, linewidths=0, zorder=2)
                ax.plot(x, trailing(y, args.window), color=color, linewidth=2, zorder=3)
            elif key == "length":
                y = [r["response_tokens"] / args.responses_per_rollout for r in rows]
                ax.plot(x, y, color=color, linewidth=1.6, alpha=0.9, zorder=3)
            else:
                y = [r["credit_ess_ratio"] for r in rows if "credit_ess_ratio" in r]
                xx = [r["rollout"] for r in rows if "credit_ess_ratio" in r]
                ax.plot(xx, y, color=color, linewidth=1.6, alpha=0.9, zorder=3)
            if row == 0:
                ax.annotate(label, xy=(x[-1], trailing([r['reward_mean'] for r in rows],
                                args.window)[-1]), xytext=(5, 0), textcoords="offset points",
                            va="center", color=INK_2, fontsize=9, fontweight="bold", zorder=4)
        if key == "ess":
            ax.axhline(0.99, color=MUTED, linewidth=1, linestyle="--", zorder=1)
            ax.text(0.995, 0.985, "ESS>0.99 ≈ GRPO (mechanism inert)", transform=ax.get_yaxis_transform(),
                    ha="right", va="top", fontsize=8, color=MUTED)
        if key == "length":
            ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
            ax.text(0.005, 0.05, "1 token/response = empty-answer reward hack", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=8, color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1][0].set_xlabel("rollout", color=MUTED, fontsize=10)
    if len(runs) > 1:
        handles = [plt.Line2D([], [], color=SERIES[i % len(SERIES)], linewidth=2,
                              label=label) for i, (label, _) in enumerate(runs)]
        leg = axes[0][0].legend(handles=handles, loc="lower right", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
        for t in leg.get_texts():
            t.set_color(INK_2)
    n = max(rows[-1]["rollout"] for _, rows in runs)
    fig.suptitle(args.title, x=0.055, ha="left", color=INK, fontsize=13, fontweight="bold")
    axes[0][0].set_title(f"{n} rollouts · trailing mean w={args.window} (top panel)",
                         loc="left", color=MUTED, fontsize=9, pad=8)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
