"""UltraFeedback prompts: dedup, benchmark isolation and the fixed split.

Port of ``vpo_rm/data.py`` plus the split in ``vpo_rm/trainer.py``.  The split
must be reproducible because the P2T arm has to see the same prompts, in the
same order, as the GRPO and VPO-RM arms it will be compared against.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import unicodedata

DEFAULT_BENCHMARK_PATHS = {
    "alpacaeval": Path("datasets/alpacaeval/eval_gpt4turbo_reference.jsonl"),
    "ifeval": Path("datasets/ifeval/ifeval_input_data.jsonl"),
    "gsm8k": Path("datasets/gsm8k/test.jsonl"),
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
    resolved = {}
    for path in benchmark_paths:
        path = Path(path)
        if path.stem in resolved:
            raise ValueError(
                f"duplicate benchmark name {path.stem!r}; use an explicit name-to-path mapping")
        resolved[path.stem] = path
    return resolved


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
                raise ValueError(f"benchmark row {path}:{line_number} has no supported prompt field")
            key = normalize_prompt(row[field])
            if key:
                keys.add(key)
    return keys


def exclude_benchmark_prompts(prompts, benchmark_paths=None):
    """Stable-deduplicate prompts and drop exact normalized benchmark matches.

    Missing benchmark files fail closed: a run must not silently proceed without
    benchmark isolation.  Original spelling and input order are preserved.
    """
    paths = _resolve_benchmark_paths(benchmark_paths)
    benchmark_keys = {name: _load_benchmark_keys(path) for name, path in paths.items()}
    excluded_union = set().union(*benchmark_keys.values()) if benchmark_keys else set()

    first_by_key = {}
    input_count = 0
    empty_prompt_count = 0
    for prompt in prompts:
        input_count += 1
        original = str(prompt)
        key = normalize_prompt(original)
        if not key:
            empty_prompt_count += 1
        elif key not in first_by_key:
            first_by_key[key] = original

    excluded_keys = set(first_by_key).intersection(excluded_union)
    output_keys = [key for key in first_by_key if key not in excluded_union]
    filtered = [first_by_key[key] for key in output_keys]
    metadata = {
        "normalization": "unicode_nfc_whitespace_collapse",
        "input_count": input_count,
        "unique_input_count": len(first_by_key),
        "duplicate_count": input_count - empty_prompt_count - len(first_by_key),
        "empty_prompt_count": empty_prompt_count,
        "excluded_count": len(excluded_keys),
        "output_count": len(filtered),
        "benchmark_paths": {name: str(path) for name, path in paths.items()},
        "benchmark_prompt_counts": {name: len(keys) for name, keys in benchmark_keys.items()},
        "excluded_by_benchmark": {name: len(set(first_by_key).intersection(keys))
                                  for name, keys in benchmark_keys.items()},
        "excluded_keys_sha256": _key_hash(excluded_keys),
        "output_keys_sha256": _key_hash(output_keys),
    }
    return filtered, metadata


def split_prompts(prompts: Iterable[str], validation_size: int = 2000):
    """Sort by the SHA256 of the normalized key; the first N are validation.

    Sorting rather than shuffling keeps the split independent of the input
    ordering, so the same corpus always yields the same train set.
    """
    by_key = {}
    for prompt in prompts:
        original, key = str(prompt), normalize_prompt(prompt)
        if key and key not in by_key:
            by_key[key] = original
    keys = sorted(by_key, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    rows = [by_key[key] for key in keys]
    n = min(max(0, int(validation_size)), len(rows))
    valid, train = rows[:n], rows[n:]

    def payload(values):
        return hashlib.sha256("\n".join(
            hashlib.sha256(normalize_prompt(x).encode()).hexdigest() for x in values).encode()).hexdigest()

    return train, valid, {"train_hash": payload(train), "validation_hash": payload(valid),
                          "num_unique": len(rows), "validation_size": len(valid)}


def load_prompt_dataset(name: str = "HuggingFaceH4/ultrafeedback_binarized",
                        split: str = "train_prefs", field: str = "prompt",
                        dataset_path: str | None = None,
                        benchmark_paths=None):
    """Load prompts from a local parquet (preferred) or the Hub, then split."""
    if dataset_path:
        try:
            from datasets import load_dataset
        except (ImportError, AttributeError):
            load_dataset = None
        if load_dataset is not None:
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
            values = (row[field] for row in dataset)
        else:
            import pyarrow.parquet as pq
            values = (x for x in pq.read_table(dataset_path, columns=[field]).column(field).to_pylist())
    else:
        from datasets import load_dataset
        dataset = load_dataset(name, split=split)
        values = (row[field] for row in dataset)
    values, exclusion = exclude_benchmark_prompts(values, benchmark_paths=benchmark_paths)
    train, valid, hashes = split_prompts(values)
    hashes["benchmark_exclusion"] = exclusion
    return train, valid, hashes
