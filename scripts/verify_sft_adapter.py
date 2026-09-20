#!/usr/bin/env python3
"""Verify one saved Llama SFT adapter, including an actual short reload forward."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sft_init import SFTDataset, collate
from scripts.sft_response_tokens import build_response_eos_spec
from vpo_rm.model_identity import validate_adapter_base
from vpo_rm.token_policy import get_stop_token_ids


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
PROBES = [("Say hello.", "Hello."), ("What is two plus two?", "Four.")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_manifest(adapter, model, response_eos_id=128009):
    manifest = json.loads((adapter / "sft_manifest.json").read_text())
    config, stats = manifest["config"], manifest["dataset_stats"]
    require(Path(config["model"]).resolve() == model.resolve(),
            "SFT manifest base model does not match the requested model")
    counts = [stats[key] for key in ("kept", "skipped_boundary", "skipped_length", "total")]
    require(all(isinstance(value, int) and value >= 0 for value in counts)
            and counts[0] > 0 and sum(counts[:3]) == counts[3], "Inconsistent SFT dataset counts")
    for key in ("max_examples", "limit"):
        bound = config.get(key, 0)
        require(not bound or stats["total"] <= bound, f"SFT dataset counts exceed {key}")
    # Llama-Instruct keeps the template EOT (128009, nothing replaced); a base
    # checkpoint supervises its native end-of-text (128001) at every example.
    expected_replaced = 0 if response_eos_id == 128009 else stats["kept"]
    require(config["response_eos"] == stats["response_eos_mode"] == "native"
            and stats["response_eos_id"] == response_eos_id
            and stats["response_eos_replaced"] == expected_replaced,
            "SFT dataset does not preserve the expected Llama native response EOS")
    micro_batch, grad_accum, epochs = config["micro_batch"], config["grad_accum"], config["epochs"]
    require(isinstance(micro_batch, int) and micro_batch > 0
            and isinstance(grad_accum, int) and grad_accum > 0
            and math.isfinite(epochs) and epochs > 0, "Invalid SFT training schedule configuration")
    microbatches = math.ceil(math.ceil(stats["kept"] / micro_batch) * epochs)
    steps = math.ceil(microbatches / grad_accum)
    expected = {"microbatches": microbatches, "optimizer_steps": steps,
                "final_update_microbatches": microbatches - (steps - 1) * grad_accum,
                "loss_normalization": "global_weighted_supervised_token_mean_per_update"}
    for key, value in expected.items():
        require(manifest["training_schedule"].get(key) == value,
                f"SFT training schedule {key} disagrees with dataset/configuration")
    metrics = [json.loads(line) for line in (adapter / "sft_metrics.jsonl").read_text().splitlines()
               if line.strip()]
    training = [row for row in metrics if "loss" in row and "event" not in row]
    require(training and training[-1].get("step") == steps and training[-1].get("of") == steps
            and training[-1].get("microbatches") == expected["final_update_microbatches"],
            "Missing or inconsistent final training metric")
    require(all(math.isfinite(row["loss"]) for row in training), "Non-finite training loss")
    return manifest, dict(expected, dataset_stats=stats, final_metric=training[-1])


def verify_lora_weights(adapter, base_config):
    from safetensors import safe_open

    hidden = base_config.hidden_size
    head_dim = getattr(base_config, "head_dim", hidden // base_config.num_attention_heads)
    query = base_config.num_attention_heads * head_dim
    kv = base_config.num_key_value_heads * head_dim
    intermediate = base_config.intermediate_size
    dimensions = {"q_proj": (hidden, query), "k_proj": (hidden, kv), "v_proj": (hidden, kv),
                  "o_proj": (query, hidden), "gate_proj": (hidden, intermediate),
                  "up_proj": (hidden, intermediate), "down_proj": (intermediate, hidden)}
    expected = {}
    for layer in range(base_config.num_hidden_layers):
        for target, (inputs, outputs) in dimensions.items():
            block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{block}.{target}"
            expected[f"{prefix}.lora_A.weight"] = (64, inputs)
            expected[f"{prefix}.lora_B.weight"] = (outputs, 64)
    summary = {"tensors": len(expected), "parameters": 0, "all_finite": True,
               "lora_b_tensors": 0, "updated_lora_b_tensors": 0, "max_abs": 0.0}
    dtypes = set()
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
        require(set(saved.keys()) == set(expected), "Saved LoRA weight keys are incomplete or unexpected")
        for name, shape in expected.items():
            weight = saved.get_tensor(name)
            require(tuple(weight.shape) == shape, f"Saved LoRA weight shape is invalid: {name}")
            require(bool(torch.isfinite(weight).all()), f"Saved LoRA weight is non-finite: {name}")
            require(weight.dtype == torch.float32, f"Saved LoRA weight is not float32: {name}")
            summary["parameters"] += weight.numel()
            summary["max_abs"] = max(summary["max_abs"], float(weight.abs().max()))
            dtypes.add(str(weight.dtype))
            if ".lora_B." in name:
                summary["lora_b_tensors"] += 1
                summary["updated_lora_b_tensors"] += int(bool(torch.count_nonzero(weight)))
    require(summary["updated_lora_b_tensors"] > 0, "Saved LoRA B matrices have no training updates")
    summary["dtypes"] = sorted(dtypes)
    return summary


def verify_sft_adapter(model, adapter, device="cuda:0", *, chat_template_file=None,
                       response_eos_id=128009):
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

    model, adapter = Path(model), Path(adapter)
    validate_adapter_base(adapter, model)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    require(adapter_config.get("peft_type") == "LORA"
            and adapter_config.get("task_type") == "CAUSAL_LM", "Adapter must be a causal LM LoRA")
    require(adapter_config.get("r") == 64, "Adapter LoRA rank must be 64")
    require(adapter_config.get("lora_alpha") == 128 and adapter_config.get("lora_dropout") == 0
            and adapter_config.get("bias") == "none", "Adapter LoRA alpha/dropout/bias differs from SFT recipe")
    require(set(adapter_config.get("target_modules", [])) == TARGETS,
            "Adapter target modules differ from the seven SFT projections")
    require(not any(adapter_config.get(key) for key in
                    ("rank_pattern", "alpha_pattern", "use_dora", "use_rslora", "modules_to_save")),
            "Adapter contains unsupported overrides to the SFT LoRA recipe")
    base_config = AutoConfig.from_pretrained(model, local_files_only=True)
    require(base_config.model_type == "llama" and base_config.architectures == ["LlamaForCausalLM"],
            "Expected a LlamaForCausalLM base model, not a reward model")
    manifest, training = verify_training_manifest(adapter, model, response_eos_id)
    weights = verify_lora_weights(adapter, base_config)

    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    expected_template = base_tokenizer.chat_template
    if chat_template_file is not None:
        pinned_template = Path(chat_template_file).read_text(encoding="utf-8")
        require(not expected_template or expected_template == pinned_template,
                "Pinned chat template differs from the native base chat template")
        expected_template = pinned_template
    require(isinstance(expected_template, str) and bool(expected_template),
            "Base model has no chat template; pass the pinned chat template file")
    require(tokenizer.chat_template == expected_template,
            "Saved tokenizer chat template differs from the native base template")
    require(tokenizer.get_vocab() == base_tokenizer.get_vocab(), "Saved tokenizer vocabulary differs from base")
    protocol = {
        "bos_token_id": tokenizer.bos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "stop_token_ids": list(get_stop_token_ids(tokenizer)),
        "template_response_eos_id": 128009, "response_eos_id": tokenizer.eos_token_id,
        "trailing_ids": [],
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    require((tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (
        128000, response_eos_id, 128004) and protocol["stop_token_ids"] == [128001, 128008, 128009],
        "Saved tokenizer does not use the expected Llama BOS/EOS/padding/stops")
    require(manifest.get("token_protocol") == protocol, "Manifest token protocol differs from saved tokenizer")
    dataset = SFTDataset(PROBES, tokenizer, 4096, build_response_eos_spec(tokenizer, "native"))
    require(len(dataset) == 2, "Saved tokenizer failed the two assistant-boundary probes")
    for ids, labels in dataset.examples:
        # The Instruct template also emits EOT inside the prompt; count the
        # terminator among supervised labels only.
        require(ids.count(128000) == 1 and labels[-1] == response_eos_id
                and labels.count(response_eos_id) == 1
                and labels[0] == -100 and sum(label != -100 for label in labels) > 1,
                "Saved tokenizer probe has invalid BOS/completion mask/EOS")
    ids, labels, attention = collate([dataset[0], dataset[1]], tokenizer.pad_token_id)
    require(bool((labels[attention == 0] == -100).all()), "Padding leaked into the completion labels")

    base = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    require(isinstance(base, LlamaForCausalLM), "Reloaded base model is not LlamaForCausalLM")
    base.config.pad_token_id = tokenizer.pad_token_id
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    actor = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    head_dtype = actor.get_output_embeddings().weight.dtype
    require(head_dtype == torch.bfloat16, "Reloaded output head must retain native bfloat16 precision")
    with torch.inference_mode():
        result = actor(input_ids=ids.to(device), attention_mask=attention.to(device),
                       labels=labels.to(device), use_cache=False)
    require(bool(torch.isfinite(result.loss)), "Reloaded SFT forward loss is non-finite")
    require(bool(torch.isfinite(result.logits).all()), "Reloaded SFT forward logits are non-finite")
    require(result.logits.dtype == torch.bfloat16, "Reloaded SFT logits are not native bfloat16")
    forward = {
        "model_class": type(base).__name__, "examples": len(dataset),
        "supervised_tokens": int((labels[:, 1:] != -100).sum()),
        "input_shape": list(ids.shape), "loss": float(result.loss),
        "all_logits_finite": True, "logits_dtype": str(result.logits.dtype),
        "loss_dtype": str(result.loss.dtype), "head_dtype": str(head_dtype),
        "embedding_dtype": str(actor.get_input_embeddings().weight.dtype),
        "lora_dtypes": sorted({str(parameter.dtype) for name, parameter in actor.named_parameters()
                               if ".lora_" in name}),
    }
    hashes = {name: file_sha256(adapter / name) for name in (
        "adapter_config.json", "adapter_model.safetensors", "sft_manifest.json",
        "sft_metrics.jsonl", "tokenizer.json", "tokenizer_config.json")}
    hashes["base_config.json"] = file_sha256(model / "config.json")
    if (adapter / "chat_template.jinja").exists():
        hashes["chat_template.jinja"] = file_sha256(adapter / "chat_template.jinja")
    return {"status": "passed", "model": str(model.resolve()), "adapter": str(adapter.resolve()),
            "device": str(device), "response_eos_id": response_eos_id,
            "chat_template_file": (str(Path(chat_template_file).resolve())
                                   if chat_template_file is not None else None),
            "weights": weights, "training": training,
            "tokenizer": protocol, "forward": forward, "hashes": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chat-template-file", type=Path, default=None,
                        help="pinned chat template expected on a base checkpoint without one")
    parser.add_argument("--response-eos-id", type=int, default=128009,
                        help="supervised terminator: 128009 (Instruct EOT) or 128001 (base end-of-text)")
    args = parser.parse_args()
    try:
        report = verify_sft_adapter(args.model, args.adapter, args.device,
                                    chat_template_file=args.chat_template_file,
                                    response_eos_id=args.response_eos_id)
    except Exception as error:
        report = {"status": "failed", "model": str(args.model), "adapter": str(args.adapter),
                  "device": args.device, "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
