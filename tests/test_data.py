import json

import pytest

from vpo_rm.data import exclude_benchmark_prompts


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_exclude_benchmark_prompts_deduplicates_before_filtering(tmp_path):
    alpaca = _write_jsonl(
        tmp_path / "alpaca.jsonl",
        [{"instruction": "Benchmark   prompt"}],
    )
    ifeval = _write_jsonl(
        tmp_path / "ifeval.jsonl",
        [{"prompt": "unrelated instruction"}],
    )
    gsm = _write_jsonl(
        tmp_path / "gsm.jsonl",
        [{"question": "How many apples?"}],
    )

    prompts = [
        "kept prompt",
        "kept   prompt",
        "Benchmark prompt",
        "How many apples?",
        "Cafe\u0301 question",
        "Caf\u00e9 question",
    ]
    filtered, metadata = exclude_benchmark_prompts(
        prompts,
        {"alpacaeval": alpaca, "ifeval": ifeval, "gsm8k": gsm},
    )

    assert filtered == ["kept prompt", "Cafe\u0301 question"]
    assert metadata["input_count"] == 6
    assert metadata["unique_input_count"] == 4
    assert metadata["duplicate_count"] == 2
    assert metadata["excluded_count"] == 2
    assert metadata["output_count"] == 2
    assert metadata["excluded_by_benchmark"] == {
        "alpacaeval": 1,
        "ifeval": 0,
        "gsm8k": 1,
    }
    assert len(metadata["excluded_keys_sha256"]) == 64
    assert len(metadata["output_keys_sha256"]) == 64
    json.dumps(metadata)


def test_exclude_benchmark_prompts_fails_if_a_required_file_is_missing(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError, match="benchmark dataset"):
        exclude_benchmark_prompts(["prompt"], {"alpacaeval": missing})


def test_exclude_benchmark_prompts_rejects_unknown_row_schema(tmp_path):
    bad = _write_jsonl(tmp_path / "bad.jsonl", [{"text": "prompt"}])
    with pytest.raises(ValueError, match="prompt field"):
        exclude_benchmark_prompts(["prompt"], {"bad": bad})
