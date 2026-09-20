import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch


def verifier():
    assert importlib.util.find_spec("scripts.verify_sft_adapter") is not None, (
        "Saved SFT adapter verifier is missing")
    return importlib.import_module("scripts.verify_sft_adapter")


@pytest.fixture(scope="module")
def saved_llama_adapter(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from scripts.sft_init import SFTDataset, collate
    from scripts.sft_response_tokens import build_response_eos_spec, configure_sft_pad_token

    tokenizer_source = Path(os.environ.get(
        "LLAMA_SFT_TOKENIZER_PATH", "/data/VPO-RM/models/Llama-3.1-8B-Instruct"))
    if not (tokenizer_source / "tokenizer_config.json").exists():
        pytest.skip("Local Llama tokenizer assets are not available")
    directory = tmp_path_factory.mktemp("saved-llama-sft")
    base_path, adapter_path = directory / "base", directory / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    tokenizer.save_pretrained(base_path)
    config = LlamaConfig(
        vocab_size=128256, hidden_size=8, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=False, bos_token_id=128000,
        eos_token_id=[128001, 128008, 128009])
    torch.manual_seed(42)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(base_path)
    base = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    configure_sft_pad_token(tokenizer)
    dataset = SFTDataset([("Say hello.", "Hello."), ("What is two plus two?", "Four.")],
                         tokenizer, 4096, build_response_eos_spec(tokenizer, "native"))
    ids, labels, attention = collate([dataset[0], dataset[1]], tokenizer.pad_token_id)
    model.train()
    loss = model(ids, labels=labels, attention_mask=attention, use_cache=False).loss
    loss.backward()
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4).step()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    manifest = {
        "config": {"model": str(base_path), "epochs": 1., "micro_batch": 1,
                   "grad_accum": 2, "max_examples": 2, "limit": 0,
                   "response_eos": "native", "selection_file": None},
        "dataset_stats": dataset.stats,
        "training_schedule": {"microbatches": 2, "optimizer_steps": 1,
                              "final_update_microbatches": 2,
                              "loss_normalization": "global_weighted_supervised_token_mean_per_update"},
        "token_protocol": {"bos_token_id": 128000, "pad_token_id": 128004,
                           "stop_token_ids": [128001, 128008, 128009],
                           "template_response_eos_id": 128009, "response_eos_id": 128009,
                           "trailing_ids": [], "chat_template_sha256": hashlib.sha256(
                               tokenizer.chat_template.encode()).hexdigest()},
        "data_sha256": "a" * 64,
    }
    (adapter_path / "sft_manifest.json").write_text(json.dumps(manifest))
    (adapter_path / "sft_metrics.jsonl").write_text(json.dumps(
        {"step": 1, "of": 1, "loss": float(loss.detach()), "microbatches": 2}) + "\n")
    return base_path, adapter_path


def test_saved_sft_adapter_reload_has_finite_native_forward(saved_llama_adapter):
    base, adapter = saved_llama_adapter
    report = verifier().verify_sft_adapter(base, adapter, device="cpu")
    assert report["status"] == "passed"
    assert report["weights"]["all_finite"] is True
    assert report["weights"]["updated_lora_b_tensors"] == 7
    assert report["training"]["optimizer_steps"] == 1
    assert report["tokenizer"]["pad_token_id"] == 128004
    assert report["forward"]["examples"] == 2
    assert report["forward"]["supervised_tokens"] > 2
    assert report["forward"]["head_dtype"] == "torch.bfloat16"
    assert report["forward"]["logits_dtype"] == "torch.bfloat16"
    assert report["forward"]["lora_dtypes"] == ["torch.float32"]
    assert report["forward"]["all_logits_finite"] is True
    assert len(report["hashes"]["adapter_model.safetensors"]) == 64


@pytest.mark.parametrize(("corruption", "match"), [
    ("base", "base model"), ("rank", "rank"), ("targets", "target modules"),
    ("steps", "optimizer_steps"), ("stats", "dataset counts"),
    ("metrics", "final training metric"), ("nan", "non-finite"),
    ("zero_b", "LoRA B"), ("missing_weight", "LoRA weight"),
    ("template", "chat template"), ("protocol", "token protocol"),
])
def test_corrupt_saved_adapter_fails_before_acceptance(saved_llama_adapter, tmp_path,
                                                      corruption, match):
    from safetensors.torch import load_file, save_file

    base, source = saved_llama_adapter
    adapter = tmp_path / "corrupted"
    shutil.copytree(source, adapter)
    if corruption in {"base", "rank", "targets"}:
        path = adapter / "adapter_config.json"
        config = json.loads(path.read_text())
        if corruption == "base":
            config["base_model_name_or_path"] = str(tmp_path / "different-base")
        elif corruption == "rank":
            config["r"] = 32
        else:
            config["target_modules"] = ["q_proj"]
        path.write_text(json.dumps(config))
    elif corruption in {"steps", "stats", "protocol"}:
        path = adapter / "sft_manifest.json"
        manifest = json.loads(path.read_text())
        if corruption == "steps":
            manifest["training_schedule"]["optimizer_steps"] = 2
        elif corruption == "stats":
            manifest["dataset_stats"]["kept"] = 1
        else:
            manifest["token_protocol"]["pad_token_id"] = 128009
        path.write_text(json.dumps(manifest))
    elif corruption == "metrics":
        (adapter / "sft_metrics.jsonl").write_text('{"step": 0, "of": 1, "loss": 1}\n')
    elif corruption == "template":
        (adapter / "chat_template.jinja").write_text("changed native template")
    else:
        path = adapter / "adapter_model.safetensors"
        weights = load_file(path)
        if corruption == "nan":
            next(iter(weights.values())).view(-1)[0] = float("nan")
        elif corruption == "zero_b":
            for name, tensor in weights.items():
                if ".lora_B." in name:
                    tensor.zero_()
        else:
            weights.pop(next(iter(weights)))
        save_file(weights, path)
    with pytest.raises(ValueError, match=match):
        verifier().verify_sft_adapter(base, adapter, device="cpu")


def test_verifier_cli_writes_failed_report_and_exits_nonzero(tmp_path):
    verifier()
    output = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, "-m", "scripts.verify_sft_adapter", "--model", str(tmp_path / "missing"),
        "--adapter", str(tmp_path / "missing-adapter"), "--device", "cpu", "--output", str(output),
    ], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["error"]


@pytest.fixture(scope="module")
def saved_llama_base_adapter(tmp_path_factory):
    """Base-checkpoint variant: no shipped chat template, native end-of-text terminator."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from scripts.sft_init import SFTDataset, collate
    from scripts.sft_response_tokens import (build_response_eos_spec, configure_sft_pad_token,
                                             install_chat_template)

    base_source = Path(os.environ.get("LLAMA_BASE_TOKENIZER_PATH", "/data/VPO-RM/models/Llama-3.1-8B"))
    template_source = Path(os.environ.get(
        "LLAMA_SFT_TOKENIZER_PATH", "/data/VPO-RM/models/Llama-3.1-8B-Instruct"))
    if not all((source / "tokenizer_config.json").exists() for source in (base_source, template_source)):
        pytest.skip("Local Llama base and Instruct tokenizer assets are not available")
    directory = tmp_path_factory.mktemp("saved-llama-base-sft")
    base_path, adapter_path = directory / "base", directory / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(base_source, local_files_only=True)
    assert not tokenizer.chat_template and tokenizer.eos_token_id == 128001
    tokenizer.save_pretrained(base_path)
    template = AutoTokenizer.from_pretrained(template_source, local_files_only=True).chat_template
    template_file = directory / "llama31_chat_template.jinja"
    template_file.write_text(template, encoding="utf-8")
    install_chat_template(tokenizer, template, hashlib.sha256(template.encode()).hexdigest())
    config = LlamaConfig(
        vocab_size=128256, hidden_size=8, intermediate_size=12,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=False, bos_token_id=128000, eos_token_id=128001)
    torch.manual_seed(42)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(base_path)
    base = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    configure_sft_pad_token(tokenizer)
    dataset = SFTDataset([("Say hello.", "Hello."), ("What is two plus two?", "Four.")],
                         tokenizer, 4096, build_response_eos_spec(tokenizer, "native"))
    assert dataset.stats["response_eos_id"] == 128001 and dataset.stats["response_eos_replaced"] == 2
    ids, labels, attention = collate([dataset[0], dataset[1]], tokenizer.pad_token_id)
    model.train()
    loss = model(ids, labels=labels, attention_mask=attention, use_cache=False).loss
    loss.backward()
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4).step()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    manifest = {
        "config": {"model": str(base_path), "epochs": 1., "micro_batch": 1,
                   "grad_accum": 2, "max_examples": 2, "limit": 0,
                   "response_eos": "native", "selection_file": None,
                   "chat_template_file": str(template_file),
                   "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()},
        "dataset_stats": dataset.stats,
        "training_schedule": {"microbatches": 2, "optimizer_steps": 1,
                              "final_update_microbatches": 2,
                              "loss_normalization": "global_weighted_supervised_token_mean_per_update"},
        "token_protocol": {"bos_token_id": 128000, "pad_token_id": 128004,
                           "stop_token_ids": [128001, 128008, 128009],
                           "template_response_eos_id": 128009, "response_eos_id": 128001,
                           "trailing_ids": [], "chat_template_sha256": hashlib.sha256(
                               template.encode()).hexdigest()},
        "data_sha256": "a" * 64,
    }
    (adapter_path / "sft_manifest.json").write_text(json.dumps(manifest))
    (adapter_path / "sft_metrics.jsonl").write_text(json.dumps(
        {"step": 1, "of": 1, "loss": float(loss.detach()), "microbatches": 2}) + "\n")
    return base_path, adapter_path, template_file


def test_saved_base_sft_adapter_verifies_with_pinned_template_and_end_of_text(saved_llama_base_adapter):
    base, adapter, template_file = saved_llama_base_adapter
    report = verifier().verify_sft_adapter(base, adapter, device="cpu",
                                           chat_template_file=template_file, response_eos_id=128001)
    assert report["status"] == "passed"
    assert report["response_eos_id"] == 128001
    assert report["tokenizer"]["response_eos_id"] == 128001
    assert report["tokenizer"]["stop_token_ids"] == [128001, 128008, 128009]
    assert report["training"]["dataset_stats"]["response_eos_replaced"] == 2
    assert report["weights"]["updated_lora_b_tensors"] == 7
    assert report["hashes"]["chat_template.jinja"] == hashlib.sha256(template_file.read_bytes()).hexdigest()


@pytest.mark.parametrize(("template", "response_eos_id", "match"), [
    (None, 128001, "chat template"),
    ("other", 128001, "chat template"),
    ("pinned", 128009, "response EOS"),
])
def test_base_sft_adapter_requires_pinned_template_and_matching_terminator(
        saved_llama_base_adapter, tmp_path, template, response_eos_id, match):
    base, adapter, template_file = saved_llama_base_adapter
    if template == "other":
        template_file = tmp_path / "other.jinja"
        template_file.write_text("{{ bos_token }}changed native template", encoding="utf-8")
    elif template is None:
        template_file = None
    with pytest.raises(ValueError, match=match):
        verifier().verify_sft_adapter(base, adapter, device="cpu",
                                      chat_template_file=template_file, response_eos_id=response_eos_id)


def test_instruct_adapter_rejects_base_terminator_expectation(saved_llama_adapter):
    base, adapter = saved_llama_adapter
    with pytest.raises(ValueError, match="response EOS"):
        verifier().verify_sft_adapter(base, adapter, device="cpu", response_eos_id=128001)
