import importlib
import importlib.util
from pathlib import Path

import pytest


def policy():
    assert importlib.util.find_spec("vpo_rm.token_policy") is not None, "Shared token policy is missing"
    return importlib.import_module("vpo_rm.token_policy")


class ExampleTokenizer:
    eos_token_id = 3
    all_special_ids = [0, 3, 4, 11]
    pieces = {0: "", 1: "\n", 2: " \t", 3: "<eos>", 4: "<|im_end|>",
              5: "ailed", 6: ".\n\n", 7: "...\n", 8: " tunes", 9: ".",
              10: " verfügbar", 11: "\n", 12: "🤾", 13: "książka"}

    def get_vocab(self):
        return {f"token-{i}": i for i in self.pieces} | {"<|im_end|>": 4}

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[i] for i in ids)


def test_structural_policy_excludes_words_and_special_tokens():
    assert policy().get_structural_token_ids(ExampleTokenizer()) == (1, 2, 6)


def test_stop_policy_uses_registered_tokens_without_unknown_fallback():
    assert policy().get_stop_token_ids(ExampleTokenizer()) == (3, 4)
    tok = ExampleTokenizer()
    tok.eos_token_id = [3, 4, 999]
    assert policy().get_stop_token_ids(tok) == (3, 4)


def test_local_qwen_structural_ids_are_decoded_not_assumed():
    source = Path(__file__).resolve().parents[1] / "models/Qwen3-8B-Base"
    if not (source / "tokenizer.json").exists():
        pytest.skip("Local Qwen3 tokenizer is not available")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    structural = set(policy().get_structural_token_ids(tokenizer))
    assert {198, 271, 6762} <= structural
    assert structural.isdisjoint({143973, 5687, 147950, 53990, 141437})
    assert set(policy().get_stop_token_ids(tokenizer)) == {151643, 151645}


def test_rm_artifact_analysis_uses_the_recorded_tokenizer(tmp_path):
    import json
    import subprocess
    import sys
    import torch
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    source = tmp_path / "tokenizer"
    vocab = {"<unk>": 0, "<eos>": 1, "\n": 2, "ailed": 3, ".\n\n": 4}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(source)
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile_manifest.json").write_text(json.dumps({"config": {"model_name": str(source)}}))
    (run / "rollout-1-tokens.json").write_text(json.dumps([[3, 2, 4, 1]]))
    torch.save({"d": torch.tensor([[1., 4., 6., 20.]])}, run / "rollout-1-credit.pt")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/analyze_rm_artifacts.py"),
                             "--run", str(run), "--out", str(tmp_path / "analysis"),
                             "--rollouts", "1"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "analysis/summary.json").read_text())
    assert summary["counts"] == {"stop": 1, "structural": 2, "body": 1}
    assert summary["median_abs_d"] == {"stop": 20., "structural": 5., "body": 1.}
    assert summary["decoded_categories"]["body"] == {"3": "ailed"}


def test_backend_special_tokens_are_excluded_from_structural_and_content_categories():
    from types import SimpleNamespace
    tokenizer = ExampleTokenizer()
    tokenizer.added_tokens_decoder = {
        1: SimpleNamespace(special=True), 2: SimpleNamespace(special=False)}
    classify = getattr(policy(), 'get_special_token_ids', None)
    assert callable(classify), 'Backend-special classification helper is missing'
    assert set(classify(tokenizer)) == {0, 1, 3, 4, 11}
    assert policy().get_structural_token_ids(tokenizer) == (2, 6)


def test_saved_tokenizer_export_from_newer_transformers_loads_through_fast_loader(monkeypatch):
    """A TF5 export names TokenizersBackend; TF4 must load the same saved backend, not the base."""
    import os
    from pathlib import Path
    import pytest
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from vpo_rm.token_policy import load_actor_tokenizer
    adapter = Path(os.environ.get("LLAMA_BASE_SFT_ADAPTER", "/data/VPO-RM/models/sft-llama31-8b-base-clean2k5e2-20260919"))
    base = Path("/data/VPO-RM/models/Llama-3.1-8B")
    if not (adapter / "tokenizer_config.json").is_file() or not (base / "tokenizer_config.json").is_file():
        pytest.skip("Local Llama base SFT export is required")
    seen = []

    def unresolved(source, *args, **kwargs):
        seen.append(str(source))
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unresolved)
    tokenizer = load_actor_tokenizer(str(base), str(adapter))
    assert seen == [str(adapter)] and isinstance(tokenizer, PreTrainedTokenizerFast)
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (128000, 128001, 128004)
    assert tokenizer.chat_template and tokenizer.padding_side == "left"

    def other(source, *args, **kwargs):
        raise ValueError("unrelated failure")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", other)
    with pytest.raises(ValueError, match="unrelated"):
        load_actor_tokenizer(str(base), str(adapter))
