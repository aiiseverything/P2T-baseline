import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def plot_with_titles(tmp_path, rows):
    python = os.environ.get("PLOT_TEST_PYTHON", sys.executable)
    available = subprocess.run([python, "-c", "import matplotlib"], capture_output=True)
    if available.returncode:
        pytest.skip("matplotlib missing; set PLOT_TEST_PYTHON to an existing plotting environment")
    source, output = tmp_path / "eval.jsonl", tmp_path / "curve.png"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # Use real matplotlib and read the rendered axes; no plot mock or source inspection.
    script = ("import runpy,sys,json; import matplotlib.pyplot as plt; "
              "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__'); "
              "print(json.dumps([ax.get_title(loc='left') for ax in plt.gcf().axes]))")
    result = subprocess.run([python, "-c", script, str(ROOT / "scripts/plot_eval_curves.py"),
                             "--eval-jsonl", str(source), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0
    return json.loads(result.stdout.splitlines()[-1])


def test_eval_plot_labels_greedy_by_actual_temperature(tmp_path):
    rows = [{"run": "sample", "temp": temp, "step": 0, "score": score, "response_tokens": 10}
            for temp in [0., .7, 1.] for score in [1., 2.]]
    assert plot_with_titles(tmp_path, rows) == ["temp 0.0 (greedy)", "temp 0.7 (sampling)", "temp 1.0 (sampling)"]


def test_eval_plot_handles_missing_temperature_and_single_prompt(tmp_path):
    rows = [{"run": "a", "temp": 0., "step": 0, "score": 1., "response_tokens": 10},
            {"run": "b", "temp": .7, "step": 0, "score": 2., "response_tokens": 20}]
    assert len(plot_with_titles(tmp_path, rows)) == 2


@pytest.mark.parametrize('credit_lambda', [None, 4.])
def test_credit_inspection_loads_local_tokenizer_and_short_response(tmp_path, credit_lambda):
    import torch
    from tokenizers import Tokenizer, models

    model = tmp_path / "local-model"
    model.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "body": 1, "<|im_end|>": 2}, unk_token="<unk>"))
    tokenizer.add_special_tokens(["<|im_end|>"])
    tokenizer.save(str(model / "tokenizer.json"))
    run = tmp_path / "run"
    run.mkdir()
    if credit_lambda is not None:
        (run / "profile_manifest.json").write_text(json.dumps({"config": {"credit_lambda": credit_lambda}}))
    (run / "rollout-1-tokens.json").write_text(json.dumps([[1, 2], [1, 1]]))
    torch.save({"w": torch.tensor([[.1, 1.9], [.8, 1.2]]), "d": torch.tensor([[.1, .2], [.1, .3]]),
                "tau": torch.ones(2)}, run / "rollout-1-credit.pt")
    output = tmp_path / "inspection.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/inspect_credit.py"),
                             "--run", str(run), "--rollouts", "1", "--model", str(model),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    example = summary["per_rollout"][0]["examples"][0]
    assert len(example["top5"]) == 2
    assert {row["text"] for row in example["top5"]} == {"body", "<|im_end|>"}
    rollout = summary["per_rollout"][0]
    assert rollout["stop_token_count"] == 1
    assert rollout["stop_token_w"]["mean"] == pytest.approx(1.9)
    assert rollout["last_token_w"]["mean"] == pytest.approx(1.55)
    assert rollout["stop_token_w"]["frac_at_band_top"] == (None if credit_lambda is None else 0.)
