#!/usr/bin/env python3
"""Dissect long AlpacaEval responses: what actually fills the tail?"""
import json, re
from collections import Counter

PATH = "runs/alpacaeval-evals/sftv2-clean-10k/generations_t1.0_n1.jsonl"

rows = [json.loads(l) for l in open(PATH)]
rows.sort(key=lambda r: -len(r["response"]))

def shingle_frac_overlap(tail, head, k=40):
    """fraction of tail k-shingles already present in head"""
    sh = {tail[i:i+k] for i in range(0, len(tail) - k, k)}
    if not sh:
        return 0.0
    hits = sum(1 for s in sh if s in head)
    return hits / len(sh)

CLOSERS = ["Please let me know", "let me know if", "feel free to", "don't hesitate",
           "I hope this helps", "Happy to help", "Thank you", "Confidence"]

def classify(resp):
    L = len(resp)
    head, tail = resp[:int(L*0.6)], resp[int(L*0.6):]
    overlap = shingle_frac_overlap(tail, head)
    closers = sum(resp.count(c) for c in CLOSERS)
    restarts = len(re.findall(r'(?m)^\s*1[\.\)]\s', resp))  # "1." list starts
    return overlap, closers, restarts

print("=" * 70)
print("TOP-3 LONGEST RESPONSES — structure windows")
for r in rows[:3]:
    resp, inst = r["response"], r["instruction"]
    L = len(resp)
    print(f"\n--- len {L} | instruction: {inst[:80]!r}")
    for frac, label in [(0.0, "head"), (0.25, ""), (0.5, "mid"), (0.75, ""), (0.97, "tail")]:
        i = int(L * frac)
        print(f"  [{label or f'{int(frac*100)}%'}] ...{resp[i:i+180]!r}")

print("\n" + "=" * 70)
print("AGGREGATE over all 805 (sorted by length)")
buckets = {"<3k": [], "3-6k": [], "6-9k": [], ">9k": []}
for r in rows:
    L = len(r["response"])
    ov, cl, rs = classify(r["response"])
    b = "<3k" if L < 3000 else "3-6k" if L < 6000 else "6-9k" if L < 9000 else ">9k"
    buckets[b].append((ov, cl, rs, L))
print(f"{'bucket':6s} {'n':>4s} {'tail重复率':>9s} {'结尾客套/条':>10s} {'列表重启/条':>10s}")
for b, v in buckets.items():
    if not v:
        continue
    n = len(v)
    print(f"{b:6s} {n:4d} {sum(x[0] for x in v)/n:9.1%} {sum(x[1] for x in v)/n:10.1f} {sum(x[2] for x in v)/n:10.1f}")

# what share of responses are loop-dominated (tail >60% verbatim repeat)?
loop_dom = sum(1 for r in rows if classify(r["response"])[0] > 0.6)
semi = sum(1 for r in rows if 0.2 < classify(r["response"])[0] <= 0.6)
novel = sum(1 for r in rows if classify(r["response"])[0] <= 0.2)
print(f"\n尾部逐字循环为主(>60% 重复): {loop_dom}  半重复(20-60%): {semi}  新内容为主(<20%): {novel}")
