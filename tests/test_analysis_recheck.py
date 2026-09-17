import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_overview_uses_recorded_response_counts(tmp_path):
    python = os.environ.get("PLOT_TEST_PYTHON", sys.executable)
    if subprocess.run([python, "-c", "import matplotlib"], capture_output=True).returncode:
        pytest.skip("matplotlib missing; set PLOT_TEST_PYTHON")
    rows = [
        {"rollout": 1, "reward_mean": 1., "response_tokens": 120,
         "mean_response_tokens": 30., "reward_count": 4},
        {"rollout": 2, "reward_mean": 2., "response_tokens": 200, "reward_count": 5},
        {"rollout": 3, "reward_mean": 3., "response_tokens": 128},
    ]
    source = tmp_path / "metrics.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    script = (
        "import runpy,sys,json; import matplotlib.pyplot as plt; "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__'); "
        "print(json.dumps(list(plt.gcf().axes[1].lines[0].get_ydata())))"
    )
    result = subprocess.run(
        [python, "-c", script, str(ROOT / "scripts/plot_training_overview.py"),
         "--run", f"small={source}", "--output", str(tmp_path / "overview.png")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [30., 40., 2.]


@pytest.mark.parametrize(("tail", "head", "expected"), [
    ("abcd", "abcd", 1.),
    ("abcdabcdabcdWXYZ", "abcd", .75),
])
def test_long_response_overlap_counts_every_complete_shingle(tmp_path, monkeypatch, tail, head, expected):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "runs/alpacaeval-evals/sftv2-clean-10k/generations_t1.0_n1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"instruction": "q", "response": "answer"}) + "\n")
    script = runpy.run_path(str(ROOT / "scripts/dissect_long.py"))
    assert script["shingle_frac_overlap"](tail, head, k=4) == expected


@pytest.mark.parametrize("nested_output", [False, True])
def test_inspection_uses_registered_stops_and_creates_output_parent(tmp_path, monkeypatch, nested_output):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from scripts import inspect_eval_gen

    model = tmp_path / "model"
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "body": 1, "<|im_end|>": 2, "<eos>": 3}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(model)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "vocab_size": 4}))
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["body"]))
    captured = []

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params, **kwargs):
            captured.append(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="body", token_ids=[1, 3])])]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *args: args))
    output = (tmp_path / "nested" if nested_output else tmp_path) / "inspection.json"
    monkeypatch.setattr(sys, "argv", ["inspect_eval_gen.py", "--run", str(tmp_path),
                                      "--model", str(model), "--eval-prompts", str(prompts),
                                      "--pps", "0", "--temps", "1", "--output", str(output)])
    inspect_eval_gen.main()
    assert captured[0]["stop_token_ids"] == [2, 3]
    assert json.loads(output.read_text())["generations"] == {"pp0.0_t1.0": ["body"]}


def test_stop_probe_reports_named_tokens_when_native_eos_is_im_end(tmp_path, monkeypatch):
    if not (ROOT / "models/Qwen3-14B-Base/tokenizer_config.json").exists():
        pytest.skip("Local Qwen3-14B-Base tokenizer assets are not installed")
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts import probe_stop_probs

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-14B-Base", local_files_only=True)
    tokenizer.eos_token = "<|im_end|>"
    im_end = tokenizer.get_vocab()["<|im_end|>"]

    class Model(torch.nn.Module):
        def cuda(self):
            return self

        def forward(self, ids):
            logits = torch.full((1, 1, len(tokenizer)), -torch.inf)
            logits[0, 0, im_end] = 0.
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: Model())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=None))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"instruction": "q"}) + "\n")
    gold = tmp_path / "gold.parquet"
    pq.write_table(pa.table({"prompt": ["q"], "chosen": ["answer"]}), gold)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["probe_stop_probs.py", "--adapters", "base=none",
                                      "--train", str(prompts), "--test", str(prompts),
                                      "--gold-map", str(gold), "--out", str(output)])
    probe_stop_probs.main()
    result = json.loads(output.read_text())["base"]
    assert result["P_endoftext_mean"] == 0.
    assert result["P_im_end_mean"] == 1.
    assert result["P_either_mean"] == 1.
