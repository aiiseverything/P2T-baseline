"""Dataset isolation helpers shared by formal SFT and RL training."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATHS = {
    "alpacaeval": ROOT / "datasets/alpacaeval/eval_gpt4turbo_reference.jsonl",
    "ifeval": ROOT / "datasets/ifeval/ifeval_input_data.jsonl",
    "gsm8k": ROOT / "datasets/gsm8k/test.jsonl",
}
_PROMPT_FIELDS = ("instruction", "prompt", "question")


def normalize_prompt(text: str) -> str:
    """Canonical prompt key: Unicode NFC with whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def _key_hash(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _resolve_benchmark_paths(benchmark_paths) -> dict[str, Path]:
    if benchmark_paths is None:
        return dict(DEFAULT_BENCHMARK_PATHS)
    if isinstance(benchmark_paths, Mapping):
        return {str(name): Path(path) for name, path in benchmark_paths.items()}
    return {Path(path).stem: Path(path) for path in benchmark_paths}


def _load_benchmark_keys(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark dataset is missing: {path}")
    keys = set()
    with path.open() as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            field = next((name for name in _PROMPT_FIELDS if name in row), None)
            if field is None:
                raise ValueError(
                    f"benchmark row {path}:{line_number} has no supported prompt field"
                )
            key = normalize_prompt(row[field])
            if key:
                keys.add(key)
    return keys


def exclude_benchmark_prompts(prompts, benchmark_paths=None):
    """Stable-deduplicate prompts and remove exact normalized benchmark matches.

    Formal training callers use the default local benchmark files. Missing files
    fail closed so a run cannot silently proceed without benchmark isolation.
    Original prompt spelling and input order are preserved for retained rows.
    """
    paths = _resolve_benchmark_paths(benchmark_paths)
    benchmark_keys = {name: _load_benchmark_keys(path) for name, path in paths.items()}
    excluded_union = set().union(*benchmark_keys.values()) if benchmark_keys else set()

    first_by_key = {}
    input_count = 0
    for prompt in prompts:
        input_count += 1
        original = str(prompt)
        key = normalize_prompt(original)
        if key and key not in first_by_key:
            first_by_key[key] = original

    excluded_keys = set(first_by_key).intersection(excluded_union)
    output_keys = [key for key in first_by_key if key not in excluded_union]
    filtered = [first_by_key[key] for key in output_keys]
    metadata = {
        "normalization": "unicode_nfc_whitespace_collapse",
        "input_count": input_count,
        "unique_input_count": len(first_by_key),
        "duplicate_count": input_count - len(first_by_key),
        "excluded_count": len(excluded_keys),
        "output_count": len(filtered),
        "benchmark_paths": {name: str(path) for name, path in paths.items()},
        "benchmark_prompt_counts": {
            name: len(keys) for name, keys in benchmark_keys.items()
        },
        "excluded_by_benchmark": {
            name: len(set(first_by_key).intersection(keys))
            for name, keys in benchmark_keys.items()
        },
        "excluded_keys": sorted(excluded_keys),
        "excluded_keys_sha256": _key_hash(excluded_keys),
        "output_keys_sha256": _key_hash(output_keys),
    }
    return filtered, metadata
