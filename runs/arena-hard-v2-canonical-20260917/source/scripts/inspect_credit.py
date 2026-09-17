#!/usr/bin/env python3
"""Per-token credit inspection: how does VPO allocate w_t, especially on EOS?

Loads the rollout-N-credit.pt dumps (w, d, tau) next to the token-id files,
decodes the text, and reports for a few rollouts:
  - aggregate: mean w on the final (stop) token vs all other tokens
  - per response: top-5 / bottom-5 tokens by w with text, d, and the stop
    token's own (d, w, rank)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vpo_rm.token_policy import get_stop_token_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--rollouts", type=int, nargs="+", required=True)
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--num-responses", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from tokenizers import Tokenizer
    model_path = Path(args.model)
    if model_path.is_dir():
        tok = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    elif model_path.is_file():
        tok = Tokenizer.from_file(str(model_path))
    else:
        tok = Tokenizer.from_pretrained(args.model)
    vocab = tok.get_vocab()
    config_path = (model_path if model_path.is_dir() else model_path.parent) / "tokenizer_config.json"
    eos_token = json.loads(config_path.read_text()).get("eos_token") if config_path.is_file() else None
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    stop_ids = set(get_stop_token_ids(SimpleNamespace(
        get_vocab=tok.get_vocab, eos_token_id=vocab.get(eos_token))))
    run = Path(args.run)
    manifest = run / "profile_manifest.json"
    credit_lambda = json.loads(manifest.read_text()).get("config", {}).get("credit_lambda") if manifest.is_file() else None
    band_top = .9 * credit_lambda if credit_lambda is not None else None
    out = {"stop_token_ids": sorted(stop_ids), "credit_lambda": credit_lambda, "per_rollout": []}

    for n in args.rollouts:
        toks = json.loads((run / f"rollout-{n}-tokens.json").read_text())
        credit = torch.load(run / f"rollout-{n}-credit.pt", map_location="cpu",
                            weights_only=True)
        w, d = credit["w"].float(), credit["d"].float()
        B, T = w.shape
        stops = []
        rows = []
        for i in range(B):
            length = len(toks[i])
            if length == 0 or length > T:
                continue
            wi = w[i, :length]
            di = d[i, :length]
            ids = toks[i]
            last = length - 1
            last_is_stop = ids[last] in stop_ids
            rank_of_last = int((wi > wi[last]).sum().item())
            stops.append({"is_stop": last_is_stop,
                          "w": float(wi[last]), "d": float(di[last]),
                          "rank": rank_of_last, "length": length})
            if i < args.num_responses:
                top = torch.topk(wi, min(5, length))
                bot = torch.topk(-wi, min(5, length))
                rows.append({
                    "length": length,
                    "A_mean_w": float(wi.mean()),
                    "tau": round(float(credit["tau"][i]), 2),
                    "stop_token": {"id": ids[last], "text": tok.decode([ids[last]], skip_special_tokens=False),
                                   "is_stop": last_is_stop,
                                   "w": float(wi[last]), "d": float(di[last]),
                                   "rank": rank_of_last, "of": length},
                    "top5": [{"text": tok.decode([ids[j]], skip_special_tokens=False), "w": float(wi[j]),
                              "d": float(di[j]), "pos": int(j)}
                             for j in top.indices.tolist()],
                    "bottom5": [{"text": tok.decode([ids[j]], skip_special_tokens=False), "w": float(wi[j]),
                                "d": float(di[j]), "pos": int(j)}
                                for j in bot.indices.tolist()]})
        ws_last = [s["w"] for s in stops]
        actual_stops = [s for s in stops if s["is_stop"]]
        ws_stop = [s["w"] for s in actual_stops]
        w_all = w[w != 0]
        rec = {"rollout": n,
               "n_responses": len(stops),
               "stop_token_count": len(actual_stops),
               "last_token_w": {"mean": sum(ws_last) / len(ws_last), "max": max(ws_last)},
               "stop_token_w": {"mean": sum(ws_stop) / len(ws_stop) if ws_stop else None,
                                "max": max(ws_stop) if ws_stop else None,
                                "frac_at_band_top": (sum(x > band_top for x in ws_stop) / len(ws_stop))
                                if ws_stop and band_top is not None else None},
               "all_token_w_mean": float(w_all.mean()),
               "stop_rank_percentile": (sum(s["rank"] / max(1, s["length"] - 1) for s in actual_stops)
                                        / len(actual_stops)) if actual_stops else None,
               "examples": rows}
        out["per_rollout"].append(rec)
        print(json.dumps({k: rec[k] for k in ("rollout", "stop_token_w", "all_token_w_mean",
                                              "stop_rank_percentile")}, default=float))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
