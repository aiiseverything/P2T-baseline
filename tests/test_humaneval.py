import gzip
import json
from pathlib import Path

import pytest


def test_humaneval_loader_preserves_completion_prompt_and_rejects_duplicate_ids(tmp_path):
    from scripts.eval_humaneval import load_problems
    row = dict(task_id="HumanEval/0", prompt="def add(a, b):\n    \"\"\"Add.\"\"\"\n", test="def check(f):\n    assert f(1, 2) == 3", entry_point="add")
    path = tmp_path / "problems.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(row) + "\n")
    assert load_problems(path, expected_count=1) == [row]
    with gzip.open(path, "at") as f:
        f.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_problems(path, expected_count=2)


def test_humaneval_samples_preserve_raw_indentation_and_require_full_coverage():
    from scripts.score_humaneval import validate_samples
    problem = {"task_id": "HumanEval/0"}
    sample = {"task_id": "HumanEval/0", "completion": "    return a + b\n"}
    assert validate_samples([problem], [sample]) == [sample]
    with pytest.raises(ValueError, match="exactly once"):
        validate_samples([problem], [])
    with pytest.raises(ValueError, match="exactly once"):
        validate_samples([problem], [sample, sample])


def test_final_checkpoint_gate_never_accepts_partially_written_checkpoint(tmp_path):
    from scripts.watch_final_humaneval import final_checkpoint_ready
    arm = tmp_path / "grpo"
    checkpoint = arm / "train/checkpoint-250"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    assert not final_checkpoint_ready(arm)
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "run_manifest.json").write_text(json.dumps({"step":250}))
    (arm / "completion.json").write_text(json.dumps({"rollouts":250}))
    (arm / "stage").write_text("complete\n")
    (arm / "exit_code").write_text("0\n")
    assert not final_checkpoint_ready(arm)
    (arm / "train/profile_summary.json").write_text(json.dumps({"rollouts":[{"rollout":i} for i in range(1,251)]}))
    assert final_checkpoint_ready(arm) == checkpoint
    (arm / "exit_code").write_text("1\n")
    assert not final_checkpoint_ready(arm)
