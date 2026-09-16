#!/usr/bin/env python3
"""Salad scanner built from chr() codepoints + control asserts (mangle-proof)."""
import json, re

RANGES = [
    (0x0E00, 0x0E7F),   # Thai
    (0x0900, 0x097F),   # Devanagari
    (0x0C80, 0x0CFF),   # Kannada
    (0x0600, 0x06FF),   # Arabic
    (0x0400, 0x04FF),   # Cyrillic
    (0x3040, 0x30FF),   # kana
    (0x4E00, 0x9FFF),   # CJK unified
    (0x3000, 0x303F),   # CJK punctuation
    (0xFF00, 0xFFEF),   # fullwidth forms
]
RX = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in RANGES) + "]")

assert RX.search(chr(0x0E01) * 3), "Thai control failed"
assert RX.search("你好"), "CJK control failed"
assert RX.search(chr(0xFF0C)), "fullwidth control failed"
assert not RX.search("plain English 123 cafe pi"), "false-positive control failed"


def analyze(path, name, show=False):
    hits, n, salad = [], 0, 0
    for line in open(path):
        r = json.loads(line)
        resp = r["response"]
        n += 1
        m = RX.search(resp)
        if m:
            salad += 1
            if len(hits) < 8:
                hits.append((m.start(), len(resp),
                             resp[max(0, m.start() - 45):m.start() + 15]))
    print(f"{name}: {100 * salad / n:.1f}% salad (n={n})")
    if show:
        for pos, L, ctx in hits:
            print(f"  @{pos}/{L}: ...{ctx!r}")


if __name__ == "__main__":
    analyze("runs/alpacaeval-evals/base/generations_t1.0_n1.jsonl", "base")
    analyze("runs/alpacaeval-evals/sftv2-clean-10k/generations_t1.0_n1.jsonl",
            "v2-clean-10k", show=True)
    analyze("runs/alpacaeval-evals/sftv2-raw-2k5/generations_t1.0_n1.jsonl", "v2-raw-2k5")
    analyze("runs/alpacaeval-evals/sft-init/generations_t1.0_n1.jsonl", "v1-sft-init")
