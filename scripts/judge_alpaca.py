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
Cost guard: checks relay usage before each bounded batch and stops dispatch at
--budget-cny (default 90; unit = 0.01 RMB). In-flight requests and delayed billing
can exceed the threshold; this is a dispatch guard, not a hard prepaid spending cap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from datetime import datetime, timezone
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_artifacts import atomic_text, cache_matches, commit_cache, digest, file_hash

TEMPLATE_PATH = Path("/root/.venvs/alpacaeval/lib/python3.13/site-packages/"
                     "alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/"
                     "alpaca_eval_clf.txt")
BASE_URL = "https://api.linkapi.ai/v1"
JUDGE_MODEL = "gpt-4.1"


def load_template(template_path: Path | str | None = None) -> str:
    """Read an explicit template, the historical path, or installed package data."""
    if template_path is not None:
        # A missing explicit choice must not silently select another template.
        return Path(template_path).read_text(encoding="utf-8")
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    try:
        spec = importlib.util.find_spec("alpaca_eval")
    except (ImportError, ValueError):
        spec = None
    relative = Path("evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt")
    for root in (spec.submodule_search_locations or ()) if spec is not None else ():
        candidate = Path(root) / relative
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "Cannot locate the AlpacaEval judge template. Pass --template PATH, or install "
        f"alpaca_eval containing {relative}. Historical path: {TEMPLATE_PATH}")


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


def fill_template(template: str, instruction: str, output_1: str, output_2: str) -> str:
    """Substitute each template slot once, leaving inserted text byte-for-byte intact."""
    values = {'{instruction}': instruction, '{output_1}': output_1, '{output_2}': output_2}
    return re.sub(r'\{(?:instruction|output_[12])\}', lambda match: values[match.group()], template)


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
    scale = max(lm, lM)
    return math.exp(lm - scale) / (math.exp(lm - scale) + math.exp(lM - scale))


def validate_references(refs):
    """Validate official or overlap-filtered reference subsets."""
    if not isinstance(refs, list) or not refs:
        raise ValueError('reference dataset must be nonempty')
    by_instruction = {}
    for row in refs:
        if (not isinstance(row, dict) or not isinstance(row.get('instruction'), str)
                or not row['instruction'] or not isinstance(row.get('reference_output'), str)
                or not row['reference_output']):
            raise ValueError('each reference needs nonempty instruction and reference_output fields')
        if row['instruction'] in by_instruction:
            raise ValueError('duplicate reference instructions')
        by_instruction[row['instruction']] = row
    return by_instruction


class Relay:
    def __init__(self, key: str, budget_cny: float):
        import httpx
        self.client = httpx.Client(timeout=90,
                                   headers={"Authorization": f"Bearer {key}"},
                                   limits=httpx.Limits(max_connections=16))
        if not math.isfinite(budget_cny) or budget_cny <= 0:
            raise ValueError('budget_cny must be finite and positive')
        self.budget_cny = budget_cny
        self.usage_start = datetime.now(timezone.utc).date().replace(day=1).isoformat()
        self.usage0 = self.usage()

    def usage(self) -> float:
        """Total spent on this key, in RMB (endpoint unit = 0.01 RMB)."""
        last = None
        for attempt in range(3):
            try:
                response = self.client.get(
                    f"{BASE_URL}/dashboard/billing/usage",
                    params={"start_date": self.usage_start,
                            "end_date": datetime.now(timezone.utc).date().isoformat()})
                response.raise_for_status()
                payload = response.json()
                amount = float(payload["total_usage"])
                if "error" in payload or not math.isfinite(amount) or amount < 0:
                    raise ValueError('invalid usage response')
                return amount / 100.0
            except Exception as error:
                last = error
                if attempt < 2:
                    time.sleep(2)
        raise RuntimeError(f'Cannot verify usage; judge dispatch stopped: {last}')

    def spent_since_start(self) -> float:
        spent = self.usage() - self.usage0
        if spent < -1e-6:
            raise RuntimeError('Usage counter decreased; cannot verify budget')
        return max(0.0, spent)

    def judge_call(self, messages) -> tuple[float | None, dict]:
        body = {"model": JUDGE_MODEL, "messages": messages, "max_tokens": 1,
                "temperature": 1, "logprobs": True, "top_logprobs": 5}
        last = None
        for attempt in range(5):
            try:
                response = self.client.post(f"{BASE_URL}/chat/completions", json=body)
                response.raise_for_status()
                r = response.json()
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


def candidate_rows(gens_path, refs, limit=None):
    rows = [json.loads(line) for line in Path(gens_path).read_text().splitlines() if line.strip()]
    by_instruction = validate_references(refs)
    if limit:
        wanted = {r['instruction'] for r in refs[:limit]}
        rows = [r for r in rows if r['instruction'] in wanted]
    else:
        wanted = set(by_instruction)
    keys = [(r['instruction'], r.get('sample_idx', 0)) for r in rows]
    sample_sets = {instruction: set() for instruction in wanted}
    for instruction, sample in keys:
        if instruction not in wanted or not isinstance(sample, int) or sample < 0:
            raise ValueError('invalid candidate coverage/sample index')
        sample_sets[instruction].add(sample)
    n_samples = max((s for _, s in keys), default=-1) + 1
    if (not rows or len(set(keys)) != len(keys) or
            any(indices != set(range(n_samples)) for indices in sample_sets.values())):
        raise ValueError('candidate coverage must include each reference and every sample exactly once')
    return rows, by_instruction


def validated_checkpoint_records(saved, protocol, rows):
    """Validate resumable annotations before trusting them as completed work."""
    if not isinstance(saved, dict) or saved.get('protocol') != protocol:
        raise ValueError('judge checkpoint protocol mismatch')
    records = saved.get('rows')
    if not isinstance(records, dict):
        raise ValueError('judge checkpoint rows are invalid')
    expected = {digest(row): row for row in rows}
    if any(key not in expected for key in records):
        raise ValueError('judge checkpoint contains unknown rows')
    for key, record in records.items():
        row = expected[key]
        preference = record.get('preference') if isinstance(record, dict) else None
        valid_preference = preference is None or (
            isinstance(preference, (int, float)) and not isinstance(preference, bool)
            and math.isfinite(preference) and 0 <= preference <= 1
        )
        if (not isinstance(record, dict) or not valid_preference
                or record.get('instruction') != row['instruction']
                or record.get('sample_idx') != row.get('sample_idx', 0)
                or not isinstance(record.get('chars'), int)
                or isinstance(record.get('chars'), bool) or record['chars'] < 0
                or not isinstance(record.get('usage'), dict)):
            raise ValueError('judge checkpoint contains an invalid annotation')
    return records


def judge_result_config(gen_path, refs, template, limit, generation_file):
    """Identity of one complete judging result and its selected generation recipe."""
    return {'generation_sha256': file_hash(gen_path), 'generation_file': generation_file,
            'refs': digest(refs), 'template': digest(template),
            'judge': JUDGE_MODEL, 'limit': limit,
            'source_sha256': file_hash(__file__)}


def judge_tag(tag: str, gens_path: Path, refs: list[dict], template: str,
              relay: Relay, limit: int | None, workers: int) -> dict:
    if workers < 1:
        raise ValueError('workers must be positive')
    rows, by_instr = candidate_rows(gens_path, refs, limit)
    protocol = digest({'rows': rows, 'refs': refs, 'template': template,
                       'judge': JUDGE_MODEL, 'protocol': 'md5-order-logprob-v2', 'source': file_hash(__file__)})
    checkpoint = Path(gens_path).with_name(f'annotations_{protocol[:16]}.json')
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'protocol': protocol, 'rows': {}}
    records = validated_checkpoint_records(saved, protocol, rows)

    def one(row):
        ref = by_instr[row['instruction']]
        switch = order_switch(row['instruction'])
        out1, out2 = (ref['reference_output'], row['response']) if switch else (row['response'], ref['reference_output'])
        filled = fill_template(template, row['instruction'], out1, out2)
        pref_first, usage = relay.judge_call(to_chat_messages(filled))
        if pref_first is not None and (not math.isfinite(pref_first) or not 0 <= pref_first <= 1):
            raise ValueError('invalid judge preference')
        return {'preference': None if pref_first is None else (1 - pref_first if switch else pref_first),
                'chars': len(row['response']), 'usage': usage,
                'instruction': row['instruction'], 'sample_idx': row.get('sample_idx', 0)}

    pending = [(digest(row), row) for row in rows if digest(row) not in records]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(pending), workers):
            spent = relay.spent_since_start()
            if not math.isfinite(spent) or spent < 0:
                raise RuntimeError('Cannot verify usage; judge dispatch stopped')
            if spent >= relay.budget_cny:
                raise RuntimeError(f'BUDGET GUARD: ¥{spent:.2f} >= ¥{relay.budget_cny}; progress saved to {checkpoint}')
            futures = {executor.submit(one, row): key for key, row in pending[start:start + workers]}
            errors = []
            for future in as_completed(futures):
                try:
                    records[futures[future]] = future.result()
                    atomic_text(checkpoint, json.dumps(saved, ensure_ascii=False, allow_nan=False))
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            if start % (workers * 25) == 0:
                print(f'  [{tag}] {len(records)}/{len(rows)} judged, spent before batch ¥{spent:.2f}', flush=True)

    ordered = [records[digest(row)] for row in rows]
    valid = [record for record in ordered if record['preference'] is not None]
    prefs = [record['preference'] for record in valid]
    n = len(valid)
    return {'tag': tag, 'n_judged': n, 'n_failed_parse': len(rows) - n,
            'weighted_win_rate': sum(prefs) / n if n else None,
            'win_rate': sum(p > 0.5 for p in prefs) / n if n else None,
            'mean_candidate_chars': sum(r['chars'] for r in valid) / n if n else None,
            'judge_model': JUDGE_MODEL, 'spent_cny': round(relay.spent_since_start(), 3),
            'annotations': str(checkpoint), 'judge_protocol': protocol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens-root", default="runs/alpacaeval-evals")
    ap.add_argument("--generation-file", default="generations_t1.0_n1.jsonl",
                    help="Generation filename within each tag (select an n>1 recipe here)")
    ap.add_argument("--refs", default="datasets/alpacaeval/eval_gpt4turbo_reference.jsonl")
    ap.add_argument("--template", type=Path,
                    help="Judge template file; otherwise use the historical or installed AlpacaEval template")
    ap.add_argument("--tags", nargs="+", default=[],
                    help="subset of tags (default: every dir with generations)")
    ap.add_argument("--limit", type=int, default=0, help="judge only first N prompts (pilot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-cny", type=float, default=90.0)
    ap.add_argument("--key-file", default="/root/.linkapi_key")
    ap.add_argument("--selftest-judge", action="store_true",
                    help="judge reference-vs-reference on 30 prompts; expect WR≈50%%")
    args = ap.parse_args()

    if args.limit < 0 or args.workers < 1:
        ap.error('limit must be nonnegative and workers positive')
    template = load_template(args.template) if args.template is not None else load_template()
    refs = [json.loads(l) for l in open(args.refs)]
    validate_references(refs)
    relay = None

    def get_relay():
        nonlocal relay
        if relay is None:
            key = os.environ.get('LINKAPI_KEY') or Path(args.key_file).read_text().strip()
            relay = Relay(key, args.budget_cny)
            print(f"relay balance check: ¥{relay.spent_since_start():.2f} spent this session "
                  f"(guard ¥{args.budget_cny})", flush=True)
        return relay

    if args.selftest_judge:
        # candidate = reference itself -> should be ~50/50
        tmp = Path("/tmp/alpacaeval-selftest")
        tmp.mkdir(exist_ok=True)
        rows = [{"idx": i, "instruction": r["instruction"],
                 "response": r["reference_output"], "response_tokens": 0}
                for i, r in enumerate(refs[:30])]
        gens = tmp / "generations_t1.0_n1.jsonl"
        gens.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        res = judge_tag("selftest-ref-vs-ref", gens, refs, template, get_relay(), 30, args.workers)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        assert 0.3 <= res["weighted_win_rate"] <= 0.7, "ref-vs-ref should be ~50%!"
        print("SELFTEST OK — judge loop validated")
        return

    gens_root = Path(args.gens_root)
    tags = args.tags or sorted(d.name for d in gens_root.iterdir()
                               if (d / args.generation_file).exists())
    print(f"judging {len(tags)} tags: {tags}", flush=True)

    out = gens_root / ('judged_summary_pilot.json' if args.limit else 'judged_summary.json')
    summary = {}
    suffix = "_pilot" if args.limit else ""
    if out.exists():
        prior = json.loads(out.read_text())
        if not isinstance(prior, dict):
            raise ValueError(f'Existing judge summary is invalid: {out}')
        for tag in prior:
            gen_path = gens_root / tag / args.generation_file
            result_path = gens_root / tag / f"results_judged{suffix}.json"
            candidate_rows(gen_path, refs, args.limit or None)
            manifest = result_path.with_suffix('.manifest.json')
            config = judge_result_config(
                gen_path, refs, template, args.limit, args.generation_file)
            if not cache_matches(manifest, config, [result_path]):
                raise ValueError(
                    f'Existing summary tag {tag!r} has no verified result; use a new output directory')
            summary[tag] = json.loads(result_path.read_text())
    for tag in tags:
        res_path = gens_root / tag / f"results_judged{suffix}.json"
        gen_path = gens_root / tag / args.generation_file
        candidate_rows(gen_path, refs, args.limit or None)
        manifest = res_path.with_suffix('.manifest.json')
        config = judge_result_config(
            gen_path, refs, template, args.limit, args.generation_file)
        if cache_matches(manifest, config, [res_path]):
            print(f"[{tag}] already judged, skipping", flush=True)
            summary[tag] = json.loads(res_path.read_text())
            continue
        print(f"[{tag}] judging ...", flush=True)
        t0 = time.time()
        res = judge_tag(tag, gen_path,
                        refs, template, get_relay(), args.limit or None, args.workers)
        res["wall_min"] = round((time.time() - t0) / 60, 1)
        res["limit"] = args.limit or None
        atomic_text(res_path, json.dumps(res, indent=1, ensure_ascii=False))
        commit_cache(manifest, config, [res_path])
        summary[tag] = res
        print(f"[{tag}] WWR={res['weighted_win_rate']} WR={res['win_rate']} "
              f"({res['wall_min']}min, ¥{res['spent_cny']})", flush=True)

    atomic_text(out, json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n{'tag':32s} {'WWR':>6s} {'WR':>6s} {'len':>7s}")
    for tag, r in sorted(summary.items()):
        if r.get("weighted_win_rate") is not None:
            print(f"{tag:32s} {r['weighted_win_rate']:6.3f} {r['win_rate']:6.3f} "
                  f"{r['mean_candidate_chars'] or 0:7.0f}")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
