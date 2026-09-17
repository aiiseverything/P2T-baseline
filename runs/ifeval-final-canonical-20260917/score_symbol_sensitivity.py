#!/usr/bin/env python3
"""Supplemental literal-symbol sensitivity analysis; NOT official IFEval scores.

Read all six saved n=1 generations and official detail flags. Only replace the
letter_frequency flags for keys 1122 (# >= 4) and 1129 (! >= 6). The official
scorer's alphabet-only fallback is intentionally left untouched in main results.
This script uses only Python's standard library and never runs generation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ARMS = ("base", "sft-init", "grpo", "lam2", "lam4", "lam8")
METRICS = ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")
TARGETS = {1122: ("#", 4), 1129: ("!", 6)}
DATASET_SHA256 = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"


def loose_candidates(response: str) -> list[str]:
    """The exact eight transformations in official evaluation_lib's loose path."""
    lines = response.split("\n")
    first = "\n".join(lines[1:]).strip()
    last = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    return [response, response.replace("*", ""), first, last, both,
            first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]


def literal_flags(response: str, symbol: str, frequency: int) -> tuple[bool, bool]:
    def follows(text):
        return bool(text.strip()) and text.lower().count(symbol) >= frequency
    return follows(response), any(follows(text) for text in loose_candidates(response))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_rows(rows, label):
    require(len(rows) == 541, f"{label}: expected exactly 541 rows")
    indexed = {}
    for row in rows:
        key = row["key"]
        require(key not in indexed, f"{label}: duplicate key {key}")
        indexed[key] = row
    return indexed


def metrics_from_details(details):
    return {
        "prompt_strict": sum(all(r["strict_list"]) for r in details) / 541,
        "prompt_loose": sum(all(r["loose_list"]) for r in details) / 541,
        "inst_strict": sum(sum(r["strict_list"]) for r in details) / 834,
        "inst_loose": sum(sum(r["loose_list"]) for r in details) / 834,
    }


def audit_model(data, generations, official):
    dataset = unique_rows(data, "dataset")
    generated = unique_rows(generations, "generations")
    original = unique_rows(official["details"], "official details")
    require(set(dataset) == set(generated) == set(original), "key sets differ")
    require(len({r["prompt"] for r in data}) == 541, "dataset prompts must be unique")
    require(sum(len(r["instruction_id_list"]) for r in data) == 834, "expected 834 instructions")
    require(official["recipe"]["n"] == 1, "only n=1 saved results are supported")
    for key, row in dataset.items():
        gen, detail = generated[key], original[key]
        count = len(row["instruction_id_list"])
        require(count == len(row["kwargs"]), f"key {key}: mismatched kwargs")
        require(gen["prompt"] == row["prompt"], f"key {key}: prompt mismatch")
        require(len(gen["responses"]) == 1 and isinstance(gen["responses"][0], str),
                f"key {key}: expected exactly one string response")
        require(detail["sample"] == 0, f"key {key}: expected sample 0")
        for mode in ("strict", "loose"):
            flags = detail[mode + "_list"]
            require(len(flags) == count and all(type(v) is bool for v in flags),
                    f"key {key}: invalid {mode} flags")
            require(type(detail[mode + "_all"]) is bool and detail[mode + "_all"] == all(flags),
                    f"key {key}: inconsistent {mode}_all")
    baseline = metrics_from_details(list(original.values()))
    for name, actual in baseline.items():
        value = official[name]
        require(isinstance(value, (int, float)) and math.isfinite(value)
                and math.isclose(value, actual, rel_tol=0, abs_tol=1e-12),
                f"official {name} does not agree with detail flags")

    corrected = copy.deepcopy(original)
    targets = []
    for key, (symbol, frequency) in TARGETS.items():
        row = dataset[key]
        matches = [i for i, iid in enumerate(row["instruction_id_list"])
                   if iid == "keywords:letter_frequency"]
        require(len(matches) == 1, f"key {key}: expected one letter_frequency constraint")
        index = matches[0]
        kwargs = row["kwargs"][index]
        require(kwargs == {"letter": symbol, "let_frequency": frequency, "let_relation": "at least"},
                f"key {key}: unexpected target kwargs")
        response = generated[key]["responses"][0]
        strict, loose = literal_flags(response, symbol, frequency)
        for mode, flag in (("strict", strict), ("loose", loose)):
            corrected[key][mode + "_list"][index] = flag
            corrected[key][mode + "_all"] = all(corrected[key][mode + "_list"])
        targets.append({
            "key": key, "instruction_id": "keywords:letter_frequency", "instruction_index": index,
            "symbol": symbol, "relation": "at least", "frequency": frequency,
            "literal_count": response.lower().count(symbol),
            "loose_candidate_literal_counts": [x.lower().count(symbol) for x in loose_candidates(response)],
            "official_strict_list": original[key]["strict_list"],
            "official_loose_list": original[key]["loose_list"],
            "literal_strict_list": corrected[key]["strict_list"],
            "literal_loose_list": corrected[key]["loose_list"],
            "official_prompt_strict": original[key]["strict_all"],
            "official_prompt_loose": original[key]["loose_all"],
            "literal_prompt_strict": corrected[key]["strict_all"],
            "literal_prompt_loose": corrected[key]["loose_all"],
        })
    revised = metrics_from_details(list(corrected.values()))
    return {
        "official_metrics": baseline,
        "literal_symbol_metrics": revised,
        "delta": {name: revised[name] - baseline[name] for name in METRICS},
        "delta_percentage_points": {name: 100 * (revised[name] - baseline[name]) for name in METRICS},
        "targets": targets,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    suite = args.suite.resolve()
    dataset_path = suite / "input_data.jsonl"
    require(sha256(dataset_path) == DATASET_SHA256, "unexpected suite dataset SHA256")
    data = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    report = {
        "analysis": "supplemental_literal_symbol_sensitivity_v1",
        "official_standard_score": False,
        "scope": "Only key1122 #>=4 and key1129 !>=6; all other saved instruction flags unchanged.",
        "num_prompts": 541, "num_instructions": 834, "num_replaced_constraints": 2,
        "dataset_sha256": DATASET_SHA256,
        "script_sha256": sha256(Path(__file__).resolve()),
        "models": {},
    }
    for arm in ARMS:
        directory = suite / "results" / arm
        result_path = directory / "results_t1.0_n1.json"
        generation_path = directory / "generations_t1.0_n1.jsonl"
        official = json.loads(result_path.read_text())
        require(official["tag"] == arm, f"{arm}: result tag mismatch")
        generated = [json.loads(line) for line in generation_path.read_text().splitlines()]
        item = audit_model(data, generated, official)
        item["inputs"] = {
            "results": {"path": str(result_path), "sha256": sha256(result_path)},
            "generations": {"path": str(generation_path), "sha256": sha256(generation_path)},
        }
        report["models"][arm] = item
    output = suite / "symbol_sensitivity.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"Supplemental analysis (not official standard scores): {output}")
    for arm, item in report["models"].items():
        print(arm, "delta percentage points:", json.dumps(item["delta_percentage_points"]))


if __name__ == "__main__":
    main()
