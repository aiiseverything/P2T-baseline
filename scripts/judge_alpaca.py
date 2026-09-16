#!/usr/bin/env python3
"""AlpacaEval 2.0 pairwise judging via linkapi relay (gpt-4.1 judge).

Faithful reimplementation of the official `weighted_alpaca_eval_gpt4_turbo`
annotator: same prompt template, same single-token (m/M) protocol, same
logprob-weighted preference, same seeded output-order randomization
(position-bias control). Only the judge model differs (gpt-4.1 via relay),
so absolute win rates are NOT comparable to the official leaderboard — this
is for internal comparison across our arms under one fixed judge.

Runs off-cluster (networked machine):
  /root/.venvs/alpacaeval/bin/python scripts/judge_alpaca.py            # all tags
  .../judge_alpaca.py --tags sft-init --limit 30                        # pilot
  .../judge_alpaca.py --selftest-judge                                  # ref-vs-ref sanity (≈50%)

Env:
  LINKAPI_KEY    API key (or --key-file, default /root/.linkapi_key)
Cost guard: reads the relay's usage endpoint; aborts before exceeding
--budget-cny (default 90; usage endpoint unit = 0.01 RMB, verified empirically).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE_PATH = Path("/root/.venvs/alpacaeval/lib/python3.13/site-packages/"
                     "alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/"
                     "alpaca_eval_clf.txt")
BASE_URL = "https://api.linkapi.ai/v1"
JUDGE_MODEL = "gpt-4.1"


def load_template() -> str:
    return TEMPLATE_PATH.read_text()


def to_chat_messages(filled: str):
    """Split an im_start-marked prompt into chat messages (package behavior)."""
    msgs = []
    for chunk in filled.split("<|im_start|>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        role, _, content = chunk.partition("\n")
        content = content.replace("<|im_end|>", "").strip()
        if role in ("system", "user", "assistant"):
            msgs.append({"role": role, "content": content})
    return msgs or [{"role": "user", "content": filled}]


def order_switch(instruction: str) -> bool:
    """Deterministic per-instruction swap: candidate is output_2 ('M') iff True.
    Same instruction -> same order for every tag (comparable across arms)."""
    return int(hashlib.md5(instruction.encode()).hexdigest(), 16) % 2 == 0


def preference_from_logprobs(top_logprobs) -> float | None:
    """P('m') / (P('m') + P('M')) from the first generated token's top logprobs."""
    if not top_logprobs:
        return None
    lp = {}
    for t in top_logprobs:
        tok = t["token"].strip()
        if tok in ("m", "M") and tok not in lp:
            lp[tok] = t["logprob"]
    if not lp:
        return None
    import math
    lm = lp.get("m", -100.0)
    lM = lp.get("M", -100.0)
    return math.exp(lm) / (math.exp(lm) + math.exp(lM))


class Relay:
    def __init__(self, key: str, budget_cny: float):
        import httpx
        self.client = httpx.Client(timeout=90,
                                   headers={"Authorization": f"Bearer {key}"},
                                   limits=httpx.Limits(max_connections=16))
        self.budget_cny = budget_cny
        self.usage0 = self.usage()

    def usage(self) -> float:
        """Total spent on this key, in RMB (endpoint unit = 0.01 RMB)."""
        for _ in range(3):
            try:
                r = self.client.get(
                    f"{BASE_URL}/dashboard/billing/usage",
                    params={"start_date": "2026-09-01", "end_date": "2026-09-16"}).json()
                return float(r.get("total_usage", 0.0)) / 100.0
            except Exception:
                time.sleep(2)
        return -1.0

    def spent_since_start(self) -> float:
        u = self.usage()
        return -1.0 if u < 0 else u - self.usage0

    def judge_call(self, messages) -> tuple[float | None, dict]:
        body = {"model": JUDGE_MODEL, "messages": messages, "max_tokens": 1,
                "temperature": 1, "logprobs": True, "top_logprobs": 5}
        last = None
        for attempt in range(5):
            try:
                r = self.client.post(f"{BASE_URL}/chat/completions", json=body).json()
                if "error" in r:
                    raise RuntimeError(r["error"])
                ch = r["choices"][0]
                tops = ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs")
                usage = r.get("usage", {})
                return preference_from_logprobs(tops), usage
            except Exception as e:  # noqa: BLE001 — retry any transport/API error
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"judge call failed after retries: {last}")


def judge_tag(tag: str, gens_path: Path, refs: list[dict], template: str,
              relay: Relay, limit: int | None, workers: int) -> dict:
    rows = [json.loads(l) for l in open(gens_path)]
    if limit:
        rows = rows[:limit]
    by_instr = {r["instruction"]: r for r in refs}

    def one(row):
        ref = by_instr[row["instruction"]]
        switch = order_switch(row["instruction"])
        out1, out2 = (ref["reference_output"], row["response"]) if switch \
            else (row["response"], ref["reference_output"])
        # the template contains literal JSON braces, so use plain replacement
        # instead of str.format (which would choke on them)
        filled = (template
                  .replace("{instruction}", row["instruction"])
                  .replace("{output_1}", out1)
                  .replace("{output_2}", out2))
        pref_first = relay.judge_call(to_chat_messages(filled))[0]
        if pref_first is None:
            return None, None
        # pref_first = P('m') = P(output_1). candidate is output_1 iff not switch
        return (pref_first if not switch else 1.0 - pref_first), len(row["response"])

    prefs, lens, failed = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for i, f in enumerate(as_completed(futs)):
            p, l = f.result()
            if p is None:
                failed += 1
            else:
                prefs.append(p)
                lens.append(l)
            if (i + 1) % 200 == 0:
                spent = relay.spent_since_start()
                print(f"  [{tag}] {i+1}/{len(rows)} judged, spent so far ¥{spent:.2f}", flush=True)
                if 0 <= spent >= relay.budget_cny:
                    raise RuntimeError(f"BUDGET GUARD: ¥{spent:.2f} >= ¥{relay.budget_cny}")

    n = len(prefs)
    return {
        "tag": tag, "n_judged": n, "n_failed_parse": failed,
        "weighted_win_rate": sum(prefs) / n if n else None,
        "win_rate": sum(p > 0.5 for p in prefs) / n if n else None,
        "mean_candidate_chars": sum(lens) / len(lens) if lens else None,
        "judge_model": JUDGE_MODEL, "spent_cny": round(relay.spent_since_start(), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens-root", default="runs/alpacaeval-evals")
    ap.add_argument("--refs", default="datasets/alpacaeval/eval_gpt4turbo_reference.jsonl")
    ap.add_argument("--tags", nargs="+", default=[],
                    help="subset of tags (default: every dir with generations)")
    ap.add_argument("--limit", type=int, default=0, help="judge only first N prompts (pilot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-cny", type=float, default=90.0)
    ap.add_argument("--key-file", default="/root/.linkapi_key")
    ap.add_argument("--selftest-judge", action="store_true",
                    help="judge reference-vs-reference on 30 prompts; expect WR≈50%")
    args = ap.parse_args()

    key = Path(args.key_file).read_text().strip()
    template = load_template()
    refs = [json.loads(l) for l in open(args.refs)]
    assert len(refs) == 805
    relay = Relay(key, args.budget_cny)
    print(f"relay balance check: ¥{relay.spent_since_start():.2f} spent this session "
          f"(guard ¥{args.budget_cny})", flush=True)

    if args.selftest_judge:
        # candidate = reference itself -> should be ~50/50
        tmp = Path("/tmp/alpacaeval-selftest")
        tmp.mkdir(exist_ok=True)
        rows = [{"idx": i, "instruction": r["instruction"],
                 "response": r["reference_output"], "response_tokens": 0}
                for i, r in enumerate(refs[:30])]
        gens = tmp / "generations_t1.0_n1.jsonl"
        gens.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        res = judge_tag("selftest-ref-vs-ref", gens, refs, template, relay, 30, args.workers)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        assert 0.3 <= res["weighted_win_rate"] <= 0.7, "ref-vs-ref should be ~50%!"
        print("SELFTEST OK — judge loop validated")
        return

    gens_root = Path(args.gens_root)
    tags = args.tags or sorted(d.name for d in gens_root.iterdir()
                               if (d / "generations_t1.0_n1.jsonl").exists())
    print(f"judging {len(tags)} tags: {tags}", flush=True)

    summary = {}
    suffix = "_pilot" if args.limit else ""
    for tag in tags:
        res_path = gens_root / tag / f"results_judged{suffix}.json"
        if res_path.exists():
            print(f"[{tag}] already judged, skipping", flush=True)
            summary[tag] = json.loads(res_path.read_text())
            continue
        print(f"[{tag}] judging ...", flush=True)
        t0 = time.time()
        res = judge_tag(tag, gens_root / tag / "generations_t1.0_n1.jsonl",
                        refs, template, relay, args.limit or None, args.workers)
        res["wall_min"] = round((time.time() - t0) / 60, 1)
        res["limit"] = args.limit or None
        res_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        summary[tag] = res
        print(f"[{tag}] WWR={res['weighted_win_rate']:.3f} WR={res['win_rate']:.3f} "
              f"({res['wall_min']}min, ¥{res['spent_cny']})", flush=True)

    out = gens_root / "judged_summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n{'tag':32s} {'WWR':>6s} {'WR':>6s} {'len':>7s}")
    for tag, r in sorted(summary.items()):
        if r.get("weighted_win_rate") is not None:
            print(f"{tag:32s} {r['weighted_win_rate']:6.3f} {r['win_rate']:6.3f} "
                  f"{r['mean_candidate_chars'] or 0:7.0f}")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
