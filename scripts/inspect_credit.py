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

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--rollouts", type=int, nargs="+", required=True)
    p.add_argument("--model", default="models/Qwen3-14B-Base")
    p.add_argument("--num-responses", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(args.model)
    run = Path(args.run)
    out = {"per_rollout": []}

    for n in args.rollouts:
        toks = json.loads((run / f"rollout-{n}-tokens.json").read_text())
        credit = torch.load(run / f"rollout-{n}-credit.pt", map_location="cpu",
                            weights_only=False)
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
            last_is_stop = ids[last] in (151643, 151645)
            rank_of_last = int((wi > wi[last]).sum().item())
            stops.append({"is_stop": last_is_stop,
                          "w": float(wi[last]), "d": float(di[last]),
                          "rank": rank_of_last, "length": length})
            if i < args.num_responses:
                top = torch.topk(wi, 5)
                bot = torch.topk(-wi, 5)
                rows.append({
                    "length": length,
                    "A_mean_w": float(wi.mean()),
                    "tau": round(float(credit["tau"][i]), 2),
                    "stop_token": {"id": ids[last], "text": tok.decode([ids[last]]),
                                   "w": float(wi[last]), "d": float(di[last]),
                                   "rank": rank_of_last, "of": length},
                    "top5": [{"text": tok.decode([ids[j]]), "w": float(wi[j]),
                              "d": float(di[j]), "pos": int(j)}
                             for j in top.indices.tolist()],
                    "bottom5": [{"text": tok.decode([ids[j]]), "w": float(wi[j]),
                                "d": float(di[j]), "pos": int(j)}
                                for j in bot.indices.tolist()]})
        ws_last = [s["w"] for s in stops]
        w_all = w[w != 0]
        rec = {"rollout": n,
               "n_responses": len(stops),
               "stop_token_w": {"mean": sum(ws_last) / len(ws_last),
                                "max": max(ws_last),
                                "frac_at_band_top": sum(1 for x in ws_last if x > 1.8) / len(ws_last)},
               "all_token_w_mean": float(w_all.mean()),
               "stop_rank_percentile": sum(s["rank"] / max(1, s["length"] - 1) for s in stops) / len(stops),
               "examples": rows}
        out["per_rollout"].append(rec)
        print(json.dumps({k: rec[k] for k in ("rollout", "stop_token_w", "all_token_w_mean",
                                              "stop_rank_percentile")}, default=float))

    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
