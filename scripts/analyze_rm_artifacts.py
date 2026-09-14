#!/usr/bin/env python3
"""Measure RM input-gradient artifacts (EOS / structural-token inflation).

Reads the credit dumps + token rows of a short vpo_rm run, categorizes every
response position into stop / structural / body, and reports median |d_t|
ratios. Reference point: Skywork-V2-Qwen3-8B showed ~100x at EOS and ~4-9x at
structural tokens before freezing was added (工程实现.md); a new RM should be
checked before its VPO arms are trained.

Files pair as rollout-N-tokens.json <-> rollout-N-credit.pt (both written with
the post-increment index). A length-match check guards the pairing: |d| must
be zero-padded beyond each response's true length.

Usage:
  python3 scripts/analyze_rm_artifacts.py --run runs/formal-... --out runs/rm4b-calib
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

# Must mirror the freeze set in vpo_rm/core.py::allocate().
STRUCTURAL_IDS = {198, 271, 143973, 6762,   # \n, \n\n, .\n\n
                  5687,                     # .\n
                  147950, 53990, 141437}    # \n\n\n, ", " variants
STOP_IDS = {151643, 151645}                # <|endoftext|>, <|im_end|>


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run dir with rollout dumps")
    p.add_argument("--out", required=True, help="output dir for summary.json")
    p.add_argument("--rollouts", type=int, nargs="+", default=[1, 2])
    args = p.parse_args()

    run = Path(args.run)
    cats = {"stop": [], "structural": [], "body": []}
    pairing_warnings = 0
    used = []

    for n in args.rollouts:
        tok_file = run / f"rollout-{n}-tokens.json"
        cr_file = run / f"rollout-{n}-credit.pt"
        if not tok_file.exists() or not cr_file.exists():
            print(f"skip rollout {n}: missing dumps")
            continue
        rows = json.loads(tok_file.read_text())
        dump = torch.load(cr_file, map_location="cpu", weights_only=True)
        d = dump["d"].float().abs()
        if d.shape[0] != len(rows):
            raise SystemExit(f"rollout {n}: batch mismatch d={tuple(d.shape)} vs {len(rows)} rows")
        for i, row in enumerate(rows):
            L = len(row)
            tail = d[i, L:]
            if tail.numel() and float(tail.max()) > 0:
                pairing_warnings += 1
            for t, tok in enumerate(row):
                if tok in STOP_IDS:
                    cats["stop"].append(float(d[i, t]))
                elif tok in STRUCTURAL_IDS:
                    cats["structural"].append(float(d[i, t]))
                else:
                    cats["body"].append(float(d[i, t]))
        used.append(n)

    if not used:
        raise SystemExit("no usable rollouts found")
    med = {k: (statistics.median(v) if v else float("nan")) for k, v in cats.items()}
    body = med["body"]
    summary = {
        "run": str(run),
        "rollouts": used,
        "pairing_warnings": pairing_warnings,
        "counts": {k: len(v) for k, v in cats.items()},
        "median_abs_d": med,
        "ratio_stop_vs_body": (med["stop"] / body if body else float("nan")),
        "ratio_structural_vs_body": (med["structural"] / body if body else float("nan")),
        "reference_rm8b": {"ratio_stop_vs_body": "~100x", "ratio_structural_vs_body": "~4-9x"},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
