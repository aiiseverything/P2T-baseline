#!/usr/bin/env python3
"""SFT loss curves: v1 (buggy data) vs v2 2x2 arms."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("v1 (echo bug, 10k x 2ep)", "models/sft-init-qwen3-14b-base/sft_metrics.jsonl", "#888888"),
    ("v2 raw-10k", "models/sftv2-raw-10k/sft_metrics.jsonl", "#5B8FF9"),
    ("v2 clean-10k", "models/sftv2-clean-10k/sft_metrics.jsonl", "#5AD8A6"),
    ("v2 raw-2k5", "models/sftv2-raw-2k5/sft_metrics.jsonl", "#F6BD16"),
    ("v2 clean-2k5", "models/sftv2-clean-2k5/sft_metrics.jsonl", "#E8684A"),
]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
for name, path, color in RUNS:
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        continue
    loss = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    steps = [s for s, _ in loss]
    vals = [v for _, v in loss]
    ax.plot(steps, vals, label=name, color=color, lw=1.4)

ax.set_xlabel("optimizer step")
ax.set_ylabel("train loss (CE on response tokens)")
ax.set_title("SFT loss: v1 vs v2 2x2 (all runs 625 steps, batch 32)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("plots/sft_v2_loss_curves.png")
print("saved plots/sft_v2_loss_curves.png")

# compact table
print(f"\n{'run':28s} " + " ".join(f"{s:>6d}" for s in [20, 100, 200, 300, 400, 500, 600]))
for name, path, _ in RUNS:
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        continue
    d = {r["step"]: r["loss"] for r in rows if "loss" in r}
    cells = []
    for s in [20, 100, 200, 300, 400, 500, 600]:
        v = min(d, key=lambda k: abs(k - s)) if d else None
        cells.append(f"{d.get(v, float('nan')):6.2f}" if v else "   -  ")
    print(f"{name:28s} " + " ".join(cells))
