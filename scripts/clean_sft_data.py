#!/usr/bin/env python3
"""Build SFT v2 datasets (raw & clean) from the UltraFeedback binarized table.

Both variants use FIXED extraction: only the assistant turn of `chosen`
(the v1 bug joined the user turn too, teaching 94.7% question echo).

  raw   = prompt + assistant response, untouched otherwise (echo fix only)
  clean = raw, minus trailing "Confidence: NN[%]" suffixes (UltraFeedback's
          verbalized_calibration principle leaked as invisible system prompts),
          minus rows whose response contains non-Latin scripts (whitelist).

Also writes 16 held-out validation prompts for the free-running SFT monitor.

Output: datasets/sft_v2/{sft_raw.parquet, sft_clean.parquet, monitor_prompts.json}
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet"
OUT = Path("datasets/sft_v2")

# --- the fixed extraction (v1 joined [user, assistant] -> taught echo) -------
def assistant_text(chosen) -> str:
    if isinstance(chosen, str):
        return chosen
    msgs = list(chosen)
    texts = [str(m.get("content", "")) for m in msgs
             if isinstance(m, dict) and m.get("role") == "assistant"]
    if not texts:
        # never silently fall back to join-all-messages — that is the v1
        # echo bug. A row without an assistant turn is malformed; drop loudly.
        raise ValueError(f"no assistant turn in message list of length {len(msgs)}")
    return texts[-1]

# --- Confidence stripping (verbalized_calibration principle signature) -----
# covers: Confidence: 95% / Confidence: 0.95 / Confidence level: 80% /
# [Confidence: 90%] / *Confidence: 90%* — number required so prose like
# "confidence and self-esteem" is never touched
_NUM = r"\d{0,3}(?:\.\d+)?\s*%?"
CONF_TAIL = re.compile(
    rf"(?:\s*\n)*\s*\**\[?[Cc]onfidence(?:\s+[Ll]evel)?\s*[:：]\s*{_NUM}\]?\**\s*\.?\s*$")
# standalone variants anywhere (FLAN multi-part answers put them mid-text)
CONF_LINE = re.compile(
    rf"[ \t]*\**\[?[Cc]onfidence(?:\s+[Ll]evel)?\s*[:：]\s*{_NUM}\]?\**[ \t]*\n?")

def strip_confidence(text: str) -> str:
    prev = None
    while prev != text:               # some responses stack several suffixes
        prev = text
        text = CONF_TAIL.sub("", text)
    return CONF_LINE.sub("", text)

# --- script whitelist (Latin + Common; CJK punct/fullwidth excluded) --------
BANNED_BLOCKS = [
    (0x0E00, 0x0E7F),   # Thai
    (0x0900, 0x097F),   # Devanagari
    (0x0C80, 0x0CFF),   # Kannada
    (0x0600, 0x06FF),   # Arabic
    (0x0400, 0x04FF),   # Cyrillic
    (0x3040, 0x30FF),   # Hiragana/Katakana
    (0x4E00, 0x9FFF),   # CJK unified
    (0x3000, 0x303F),   # CJK punctuation
    (0xFF00, 0xFFEF),   # fullwidth forms
    (0xAC00, 0xD7AF),   # Hangul
]

def has_banned_script(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        for lo, hi in BANNED_BLOCKS:
            if lo <= o <= hi:
                return True
    return False

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(SRC).to_pylist()

    raw, clean = [], []
    stats = dict(rows=len(table), conf_stripped=0, script_dropped=0,
                 empty_after_strip=0)
    for r in table:
        prompt = str(r["prompt"])
        resp = assistant_text(r["chosen"])
        if not prompt.strip() or not resp.strip():
            continue
        raw.append((prompt, resp))

        s = strip_confidence(resp)
        if s != resp:
            stats["conf_stripped"] += 1
        if not s.strip():
            stats["empty_after_strip"] += 1
            continue
        if has_banned_script(s):
            stats["script_dropped"] += 1
            continue
        clean.append((prompt, s))

    schema = pa.schema([("prompt", pa.string()), ("chosen", pa.string())])
    for name, rows in [("sft_raw", raw), ("sft_clean", clean)]:
        pq.write_table(pa.Table.from_arrays(
            [pa.array([p for p, _ in rows]), pa.array([c for _, c in rows])],
            schema=schema), OUT / f"{name}.parquet")

    # validation prompts (SHA-ordered split, same rule as the trainer) for the
    # free-running monitor; take 16 evenly spaced from the valid head
    import hashlib
    def norm(t): return " ".join(unicodedata.normalize("NFC", t).split())
    by_key = {}
    for r in table:
        k = norm(str(r["prompt"]))
        if k and k not in by_key:
            by_key[k] = str(r["prompt"])
    keys = sorted(by_key, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    valid = [by_key[k] for k in keys[:2000]]
    mon = [valid[int(i * len(valid) / 16)] for i in range(16)]
    (OUT / "monitor_prompts.json").write_text(json.dumps(mon, ensure_ascii=False, indent=1))

    # report
    conf_after = sum(has_conf := ("Confidence" in c) for _, c in clean)
    print(json.dumps(stats, indent=1))
    print(f"raw rows:   {len(raw)}  (assistant-only extraction; echo bug fixed)")
    print(f"clean rows: {len(clean)}  ('Confidence' remaining: {conf_after})")
    ex = next(((p, c) for p, c in clean if "Confidence" in c), None)
    if ex:
        i = ex[1].find("Confidence")
        print("residual Confidence example:", repr(ex[1][max(0, i-60):i+60]))

if __name__ == "__main__":
    main()
