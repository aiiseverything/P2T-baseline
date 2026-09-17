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
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vpo_rm.token_policy import get_stop_token_ids, get_structural_token_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training run dir with rollout dumps")
    p.add_argument("--out", required=True, help="output dir for summary.json")
    p.add_argument("--rollouts", type=int, nargs="+", default=[1, 2])
    p.add_argument("--tokenizer", help="local/cached tokenizer; defaults to the run's recorded actor model")
    args = p.parse_args()

    run = Path(args.run)
    tokenizer_source = args.tokenizer
    if tokenizer_source is None:
        manifest_path = run / "profile_manifest.json"
        if manifest_path.exists():
            tokenizer_source = json.loads(manifest_path.read_text()).get("config", {}).get("model_name")
    if not tokenizer_source:
        p.error("Provide --tokenizer or a profile_manifest.json containing config.model_name")
    if not Path(tokenizer_source).exists() and (ROOT / tokenizer_source).exists():
        tokenizer_source = str(ROOT / tokenizer_source)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    stop_ids = set(get_stop_token_ids(tokenizer))
    structural_ids = set(get_structural_token_ids(tokenizer))
    cats = {"stop": [], "structural": [], "body": []}
    observed_ids = {category: set() for category in cats}
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
                category = "stop" if tok in stop_ids else "structural" if tok in structural_ids else "body"
                cats[category].append(float(d[i, t]))
                observed_ids[category].add(tok)
        used.append(n)

    if not used:
        raise SystemExit("no usable rollouts found")
    med = {k: (statistics.median(v) if v else float("nan")) for k, v in cats.items()}
    body = med["body"]
    summary = {
        "run": str(run),
        "rollouts": used,
        "pairing_warnings": pairing_warnings,
        "tokenizer_source": str(tokenizer_source),
        "structural_rule": "nonempty whitespace, or newline-containing token that strips to '.'; exclude special tokens",
        "stop_token_ids": sorted(stop_ids),
        "structural_token_ids": sorted(structural_ids),
        "decoded_categories": {
            category: {str(token_id): tokenizer.decode([token_id], skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)
                       for token_id in sorted(token_ids)}
            for category, token_ids in observed_ids.items()},
        "counts": {k: len(v) for k, v in cats.items()},
        "median_abs_d": med,
        "ratio_stop_vs_body": (med["stop"] / body if body else float("nan")),
        "ratio_structural_vs_body": (med["structural"] / body if body else float("nan")),
        "historical_reference_note": "Prior structural-token ratios used incorrect token IDs and need recomputation.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
