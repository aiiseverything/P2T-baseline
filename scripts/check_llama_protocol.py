#!/usr/bin/env python3
"""Fail-closed Llama actor/RM protocol audit, with an optional real RM GPU gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from vpo_rm.alignment import check_tokenizers
from vpo_rm.reward_inputs import (REWARD_INPUT_PROTOCOL, _byte_decoder,
                                  build_reward_input, canonical_reward_input)
from vpo_rm.token_policy import get_special_token_ids, get_stop_token_ids


SOURCES = ("scripts/check_llama_protocol.py", "vpo_rm/reward_inputs.py",
           "vpo_rm/alignment.py", "vpo_rm/token_policy.py", "vpo_rm/trainer.py",
           "vpo_rm/reward.py", "vpo_rm/integration.py")
PAD, BOS, EOT, VOCAB, RM_BUDGET = 128004, 128000, 128009, 128256, 4096
STOPS = [128001, 128008, 128009]
MODEL_CARD_PROMPT = ("Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, "
                     "and finally splits all her apples equally among herself and her 2 siblings. "
                     "How many apples does each person get?")
MODEL_CARD_RESPONSES = (
    "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n"
    "2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n"
    "3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). "
    "9 ÷ 3 = 3 apples each. Each person gets 3 apples.",
    "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n"
    "2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n"
    "3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). "
    "9 ÷ 2 = 4.5 apples each. Each person gets 4 apples.",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(directory, required, optional=()):
    directory = Path(directory)
    for name in required:
        require((directory / name).is_file(), f"Required artifact missing: {directory / name}")
    return {name: file_sha256(directory / name) for name in (*required, *optional)
            if (directory / name).is_file()}


def checkpoint_headers(directory, architecture):
    """Read safetensors headers only; report prior shard hashes as prior evidence."""
    from safetensors import safe_open

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    require(config.get("architectures") == [architecture] and config.get("model_type") == "llama",
            f"Unexpected Llama checkpoint architecture: {directory}")
    require(config.get("vocab_size") == VOCAB and config.get("hidden_size") == 4096
            and config.get("num_hidden_layers") == 32, "Unexpected Llama 8B dimensions")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    manifest_name = ("DOWNLOAD_VERIFIED.json" if architecture == "LlamaForSequenceClassification"
                     else "DOWNLOAD_MANIFEST.json")
    verified = json.loads((directory / manifest_name).read_text())
    require(verified.get("status") == "complete_verified" or verified.get("verified") is True,
            "Missing completed download verification")
    records = {entry["path"]: entry for entry in verified["files"]}
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        record = records.get(filename, {})
        declared = record.get("sha256", record.get("local_sha256"))
        require(declared == file_sha256(directory / filename), f"Changed downloaded model metadata: {filename}")
    files, shapes, tensor_count = {}, {}, 0
    for filename in sorted(set(index["weight_map"].values())):
        path = directory / filename
        require(path.parent.resolve() == directory.resolve(), "Invalid checkpoint shard path")
        record = records.get(filename, {})
        declared = record.get("sha256", record.get("local_sha256", ""))
        require((record.get("verified") is True or record.get("verified_against_upstream") is True) and path.is_file()
                and path.stat().st_size == record.get("size")
                and len(declared) == 64, f"Unverified or changed checkpoint shard: {path}")
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            require(0 < header_size < 100 * 1024 * 1024, "Invalid safetensors header size")
            header_hash = hashlib.sha256(stream.read(header_size)).hexdigest()
        with safe_open(path, framework="pt", device="cpu") as saved:
            expected_keys = {name for name, shard in index["weight_map"].items() if shard == filename}
            require(set(saved.keys()) == expected_keys, f"Checkpoint tensor index mismatch: {filename}")
            tensor_count += len(expected_keys)
            for name in expected_keys:
                tensor = saved.get_slice(name)
                require(tensor.get_dtype() == "BF16", f"Unexpected checkpoint dtype: {name}")
                if name in {"score.weight", "model.embed_tokens.weight", "lm_head.weight"}:
                    shapes[name] = tensor.get_shape()
                if name == "score.weight":
                    require(saved.get_tensor(name).isfinite().all().item(), "Nonfinite RM score head")
        files[filename] = {"bytes": path.stat().st_size,
                           "declared_sha256": declared,
                           "sha256_evidence": f"prior_{manifest_name}_not_rehashed",
                           "header_sha256": header_hash}
    require(shapes.get("model.embed_tokens.weight") == [VOCAB, 4096], "Unexpected embedding shape")
    expected_head = "score.weight" if architecture == "LlamaForSequenceClassification" else "lm_head.weight"
    require(shapes.get(expected_head) == ([1, 4096] if expected_head == "score.weight" else [VOCAB, 4096]),
            "Unexpected model head shape")
    require(tensor_count == verified.get("tensor_count", verified.get("index_tensor_count")),
            "Downloaded tensor count changed")
    return {"files": files, "head_and_embedding_shapes": shapes,
            "tensor_count": tensor_count, "tensor_bytes": index["metadata"]["total_size"]}


def validate_reward_rows(rows):
    lengths = []
    for row in rows:
        require(bool(row) and all(type(token) is int and 0 <= token < VOCAB for token in row),
                "Invalid reward token IDs")
        require(row[0] == BOS and row.count(BOS) == 1, "Reward chat must contain exactly one initial BOS")
        require(row[-1] == EOT, "Reward chat must end with native assistant EOT")
        require(len(row) <= RM_BUDGET, f"Canonical RM input exceeds {RM_BUDGET} token budget; no truncation allowed")
        lengths.append(len(row))
    return lengths


def audit_tokenizers(actor, reward, actor_eos=EOT):
    """Audit the actor/RM token protocol; ``actor_eos`` is the SFT-supervised terminator.

    Llama-Instruct SFT ends answers with EOT 128009; the pretrained base SFT ends
    with its native <|end_of_text|> 128001. The RM always receives its own EOT.
    """
    require(actor_eos in STOPS, "actor response EOS must be a registered native stop")
    for name, tok, eos in (("actor", actor, actor_eos), ("reward", reward, EOT)):
        require(tok.pad_token_id == PAD, f"{name} must use native Llama pad 128004")
        require(tok.bos_token_id == BOS and tok.eos_token_id == eos, f"Unexpected {name} BOS/EOT")
        require(sorted(get_stop_token_ids(tok)) == STOPS, f"Unexpected {name} stop IDs")
        require(_byte_decoder(tok) is not None, f"Unsupported {name} byte decoder")
    check_tokenizers(actor, reward, VOCAB, VOCAB)
    encode = lambda text: actor.encode(text, add_special_tokens=False)
    cases = {
        "ascii": (encode("Hello world!") + [EOT], 3),
        "unicode": (encode("café 中文🙂") + [EOT], 6),
        "trim": (encode("\t\n") + encode("Hello world!") + encode(" \n") + [EOT], 3),
        "repeated": (encode("repeat repeat repeat") + [EOT], 3),
        "empty": ([EOT], 0),
        "whitespace": (encode("  \n") + [EOT], 0),
        "specials": (encode("Hello") + STOPS, 1),
        "marker": (encode("__VPO_RM_RESPONSE_BOUNDARY_9f174__") + [EOT], None),
        "actor_stop": (encode("Hello world!") + [actor_eos], 3),
        "actor_stop_empty": ([actor_eos], 0),
    }
    results = {}
    for name, (response, expected_count) in cases.items():
        encoded = build_reward_input(actor, reward, "Say hello", response)
        text = actor.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(encoded.response_text == text, "Reward decoding differs from actor text")
        official = reward.apply_chat_template([
            {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": text}],
            tokenize=True, add_generation_prompt=False, return_dict=False)
        require(encoded.input_ids == list(official), f"Canonical RM serialization differs from model card: {name}")
        validate_reward_rows([encoded.input_ids])
        positions = encoded.response_positions
        mapped = [position for position in positions if position >= 0]
        require(all(a < b for a, b in zip(mapped, mapped[1:])), "Nonmonotonic reward positions")
        require(all(encoded.input_ids[pos] == token for token, pos in zip(response, positions) if pos >= 0),
                "Actor/RM mapped token IDs differ")
        require(all(pos == -1 for token, pos in zip(response, positions) if token in get_special_token_ids(actor)),
                "Removed special token received a reward gradient")
        if expected_count is not None:
            require(len(mapped) == expected_count, f"Unexpected mapped content coverage: {name}")
        else:
            require(bool(mapped), f"No mapped content: {name}")
        results[name] = {"response_ids": response, "response_positions": positions,
                         "rm_tokens": len(encoded.input_ids), "mapped_content_tokens": len(mapped)}
    rendered = reward.apply_chat_template([
        {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello world!"}],
        tokenize=False)
    require("Cutting Knowledge Date: December 2023" in rendered
            and "Today Date: 26 Jul 2024" in rendered, "Unexpected automatic Llama system text")
    # With this native template, 2012 user tokens reserve exactly 2048 tokens
    # including the empty assistant turn. Exercise the actual serialization
    # immediately below/above the experiment budget, without a model forward.
    boundary_prompt = " word" * 2012
    boundary_response = " word" * 2048
    boundary_rows = [canonical_reward_input(reward, boundary_prompt, boundary_response + " word" * extra)
                     for extra in (0, 1)]
    for extra, row in enumerate(boundary_rows):
        official = reward.apply_chat_template([
            {"role": "user", "content": boundary_prompt},
            {"role": "assistant", "content": boundary_response + " word" * extra}],
            tokenize=True, return_dict=False)
        require(row == list(official) and len(row) == RM_BUDGET + extra,
                "Canonical RM boundary serialization changed or truncated")
    validate_reward_rows(boundary_rows[:1])
    try:
        validate_reward_rows(boundary_rows[1:])
    except ValueError:
        pass
    else:
        raise ValueError("RM budget validation accepted an overflowing conversation")
    return {"pad_token_id": PAD, "bos_token_id": BOS, "response_eot_id": EOT,
            "actor_response_eos_id": actor_eos,
            "stop_token_ids": STOPS, "reward_input_protocol": REWARD_INPUT_PROTOCOL,
            "reward_tokenizer_max_length": reward.model_max_length,
            "rm_budget": RM_BUDGET, "cases": results,
            "length_boundary": {"accepted_tokens": len(boundary_rows[0]),
                                "rejected_tokens": len(boundary_rows[1]),
                                "serialization_preserved": True},
            "actor_chat_template_sha256": hashlib.sha256(actor.chat_template.encode()).hexdigest(),
            "reward_chat_template_sha256": hashlib.sha256(reward.chat_template.encode()).hexdigest()}


def audit_cpu(actor_path, reward_path, adapter_path, actor_eos=EOT):
    from transformers import PreTrainedTokenizerFast
    from vpo_rm.model_identity import validate_adapter_base

    paths = {"actor": Path(actor_path), "reward": Path(reward_path), "init_adapter": Path(adapter_path)}
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    optional = ("chat_template.jinja", "special_tokens_map.json", "generation_config.json")
    identity = {role: artifact_identity(paths[role], required + (manifest,), optional)
                for role, manifest in (("actor", "DOWNLOAD_MANIFEST.json"),
                                       ("reward", "DOWNLOAD_VERIFIED.json"))}
    identity["init_adapter"] = artifact_identity(paths["init_adapter"],
        ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "sft_manifest.json", "sft_metrics.jsonl"))
    validate_adapter_base(adapter_path, actor_path)
    actor = PreTrainedTokenizerFast.from_pretrained(adapter_path, local_files_only=True)
    reward = PreTrainedTokenizerFast.from_pretrained(reward_path, local_files_only=True)
    base = PreTrainedTokenizerFast.from_pretrained(actor_path, local_files_only=True)
    base.pad_token = "<|finetune_right_pad_id|>"
    check_tokenizers(base, actor, VOCAB, VOCAB)
    protocol = audit_tokenizers(actor, reward, actor_eos)
    sft = json.loads((paths["init_adapter"] / "sft_manifest.json").read_text())
    require(sft["token_protocol"]["chat_template_sha256"] == protocol["actor_chat_template_sha256"],
            "Saved SFT chat template differs from its manifest")
    # An EOT-terminated SFT replaces nothing; a native-EOS SFT replaced every template EOT.
    expected_replaced = 0 if actor_eos == EOT else sft["dataset_stats"]["kept"]
    require(sft["token_protocol"]["pad_token_id"] == PAD
            and sft["token_protocol"]["response_eos_id"] == actor_eos
            and sft["token_protocol"]["template_response_eos_id"] == EOT
            and sft["dataset_stats"]["response_eos_replaced"] == expected_replaced,
            "Unexpected SFT token protocol")
    config = json.loads((paths["reward"] / "config.json").read_text())
    require(config.get("pad_token_id") == PAD and sorted(config.get("eos_token_id", [])) == STOPS,
            "RM configuration changed native pad/stop IDs")
    checkpoints = {"actor": checkpoint_headers(actor_path, "LlamaForCausalLM"),
                   "reward": checkpoint_headers(reward_path, "LlamaForSequenceClassification")}
    return {"identity": identity, "checkpoint_files": checkpoints, "protocol": protocol}


def audit_reward_model(model, tokenizer, device, *, atol=0.125, diagnostics=None,
                       batch_invariance_required=True):
    """Check identical-input paths; optionally enforce cross-layout score equality."""
    import torch
    from vpo_rm.reward import LastTokenReward, position_ids_from_mask, reward_input_gradients

    diagnostics = {} if diagnostics is None else diagnostics
    parameter_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    require(len(parameter_dtypes) == 1, "Reward audit requires one floating parameter dtype")
    diagnostics.update(status="failed", score_atol=atol, score_rtol=0, score_checks=[],
        score_protocol="raw_scalar_logit_no_sigmoid", parameter_dtype=str(next(iter(parameter_dtypes))),
        attention_implementation=model.config._attn_implementation,
        same_input_paths=["native", "wrapper_no_grad", "wrapper_input_gradients"],
        batch_invariance_required=batch_invariance_required)
    required_differences, batch_differences = [], []

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    def record_scores(native, wrapped, reference, ids, mask, case_indices, side):
        scores = {name: values.detach().cpu().tolist() for name, values in
                  (("native", native), ("wrapped", wrapped), ("singleton", reference))}
        indexes = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        native_pool = (indexes * ids.ne(PAD)).argmax(-1)
        wrapped_pool = indexes.masked_fill(~mask.bool(), -1).amax(-1)
        records = []
        for index, case_index in enumerate(case_indices):
            records.append({"case_index": case_index, "case": case_names[case_index],
                "padding_side": side, "batch_width": ids.shape[1],
                "native_pool_position": int(native_pool[index]),
                "wrapped_pool_position": int(wrapped_pool[index]),
                "terminal_token_id": int(ids[index, wrapped_pool[index]]),
                **{name + "_score": number(values[index]) for name, values in scores.items()},
                "wrapper_no_grad_score": number(scores["wrapped"][index]),
                "finite": {name: math.isfinite(values[index]) for name, values in scores.items()},
                "comparisons": {}})
        diagnostics["score_checks"].extend(records)
        return records

    def comparison(checks, name, left, right, *, required=True, batch=False):
        deltas = (left - right).detach().cpu().tolist()
        passed = torch.isclose(left, right, atol=atol, rtol=0).cpu().tolist()
        for index, check in enumerate(checks):
            check["comparisons"][name] = {"delta": number(deltas[index]),
                "abs_delta": number(abs(deltas[index])), "passed": passed[index], "required": required}
        difference = float((left - right).abs().max())
        if required:
            required_differences.append(difference)
        if batch:
            batch_differences.append(difference)
        return (bool(torch.allclose(left, right, atol=atol, rtol=0)),
                f"{checks[0]['padding_side']} raw reward differs ({name}); "
                f"abs_deltas={[number(abs(delta)) for delta in deltas]}, atol={atol}, rtol=0")

    require(model.config.pad_token_id == PAD and tokenizer.pad_token_id == PAD,
            "Direct HF scoring requires original native RM pad 128004")
    model.eval().requires_grad_(False)
    pairs = [("Say hello", "Hello world!"), ("Repeat", "café 中文🙂"),
             ("Repeat", ""), *[(MODEL_CARD_PROMPT, text) for text in MODEL_CARD_RESPONSES]]
    case_names = ("ascii", "unicode", "empty", "model_card_correct", "model_card_incorrect")
    rows = [canonical_reward_input(tokenizer, prompt, text) for prompt, text in pairs]
    lengths = validate_reward_rows(rows)
    diagnostics["row_lengths"] = lengths
    wrapper = LastTokenReward(model.base_model, model.score).eval()
    parameter_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    singletons = {path: [] for path in diagnostics["same_input_paths"]}
    padding_grad_max, mapped_grad_min = 0., float("inf")
    layouts = [("none", [index]) for index in range(len(rows))]
    layouts.extend((side, list(range(len(rows)))) for side in ("left", "right"))
    for side, indices in layouts:
        width = lengths[indices[0]] if side == "none" else max(lengths) + 3
        ids = torch.full((len(indices), width), PAD, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, case_index in enumerate(indices):
            row = rows[case_index]
            start = width - len(row) if side == "left" else 0
            ids[index, start:start + len(row)] = torch.tensor(row, device=device)
            mask[index, start:start + len(row)] = 1
        last = torch.arange(width, device=device).expand_as(ids).masked_fill(~mask.bool(), -1).amax(-1)
        require(ids[torch.arange(len(indices), device=device), last].eq(EOT).all().item(),
                "Pooling does not select terminal assistant EOT")
        with torch.no_grad():
            native = model(input_ids=ids, attention_mask=mask,
                           position_ids=position_ids_from_mask(mask)).logits[:, 0].float()
            wrapped = wrapper(inputs_embeds=wrapper.get_input_embeddings()(ids), attention_mask=mask)
        reference = native if side == "none" else torch.tensor(singletons["native"], device=device)
        checks = record_scores(native, wrapped, reference, ids, mask, indices, side)
        agreement = comparison(checks, "native_vs_wrapper", native, wrapped)
        require(torch.isfinite(native).all().item() and torch.isfinite(wrapped).all().item(),
                "Nonfinite singleton reward" if side == "none" else "Nonfinite padded reward")
        require(*agreement)
        with torch.inference_mode():
            gradient_scores, gradients = reward_input_gradients(wrapper, ids, mask)
        for index, check in enumerate(checks):
            check["wrapper_input_gradients_score"] = number(gradient_scores[index])
            check["finite"]["wrapper_input_gradients"] = bool(torch.isfinite(gradient_scores[index]))
        require(*comparison(checks, "native_vs_wrapper_input_gradients", native, gradient_scores))
        require(*comparison(checks, "gradient_mode_vs_no_grad", wrapped, gradient_scores))
        if side == "none":
            for path, values in (("native", native), ("wrapper_no_grad", wrapped),
                                 ("wrapper_input_gradients", gradient_scores)):
                singletons[path].append(float(values.item()))
        else:
            # Record every cross-layout difference before enforcing the FP32 phase.
            agreements = []
            for name, path, values in (("wrapped_vs_singleton", "wrapper_no_grad", wrapped),
                                      ("native_vs_singleton", "native", native),
                                      ("input_gradients_vs_singleton", "wrapper_input_gradients", gradient_scores)):
                reference = torch.tensor(singletons[path], device=device)
                agreements.append(comparison(checks, name, values, reference,
                    required=batch_invariance_required, batch=True))
            if batch_invariance_required:
                for agreement in agreements:
                    require(*agreement)
        require(torch.isfinite(gradients).all().item(), "Nonfinite RM input gradient")
        require(gradients.abs().sum().item() > 0, "Zero RM input gradient")
        if (~mask.bool()).any():
            padding_grad_max = max(padding_grad_max, float(gradients[~mask.bool()].abs().max()))
        require(padding_grad_max == 0, "Padding received an RM input gradient")
        for index, case_index in enumerate(indices):
            prompt, text = pairs[case_index]
            if not text:
                continue
            response = tokenizer.encode(text, add_special_tokens=False) + [EOT]
            positions = build_reward_input(tokenizer, tokenizer, prompt, response).response_positions
            shift = width - lengths[case_index] if side == "left" else 0
            positions = [position + shift for position in positions if position >= 0]
            require(bool(positions), "No mapped response gradients in GPU audit")
            norm = float(gradients[index, positions].float().norm())
            require(norm > 0, "Zero mapped response input gradient")
            mapped_grad_min = min(mapped_grad_min, norm)
    require(all(not parameter.requires_grad and parameter.grad is None
                and parameter._version == parameter_versions[name]
                for name, parameter in model.named_parameters()), "RM parameters changed or accumulated gradients")
    diagnostics.update(status="passed", singleton_scores=singletons["native"],
        max_score_difference=max(required_differences), max_batch_score_difference=max(batch_differences),
        padding_sides=["left", "right"], terminal_eot_pooling=True, nonzero_input_gradient=True,
        mapped_gradient_norm_min=mapped_grad_min, padding_gradient_max_abs=padding_grad_max,
        frozen_parameters=True)
    return diagnostics


def audit_gpu_reward_model(model, tokenizer, device, *, diagnostics=None):
    """Separate production BF16 identity from an FP32 padding-invariance control.

    This model is audit-only: its full parameter set is cast to FP32 in place.
    """
    import torch
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(status="failed", audit_protocol="llama_reward_precision_audit_v2",
        production_microbatch_responses=1, production_bf16={"status": "failed"},
        padding_fp32={"status": "not_run"})
    require(model.config._attn_implementation == "sdpa", "GPU reward audit requires SDPA")
    require(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()
                if parameter.is_floating_point()), "Production audit requires full BF16 parameters")
    audit_reward_model(model, tokenizer, device, atol=0.125,
        diagnostics=diagnostics["production_bf16"], batch_invariance_required=False)
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.float()
        require(all(parameter.dtype == torch.float32 for parameter in model.parameters()
                    if parameter.is_floating_point()), "Padding control requires full FP32 parameters")
        require(model.config._attn_implementation == "sdpa", "FP32 control changed SDPA")
        fp32 = diagnostics["padding_fp32"]
        fp32.update(status="failed", tf32_disabled=not (torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32), float32_matmul_precision=torch.get_float32_matmul_precision())
        require(fp32["tf32_disabled"] and fp32["float32_matmul_precision"] == "highest",
                "FP32 padding control requires TF32 disabled")
        audit_reward_model(model, tokenizer, device, atol=0.001, diagnostics=fp32,
                           batch_invariance_required=True)
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
    diagnostics["status"] = "passed"
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("actor", "reward", "init-adapter", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--actor-eos", type=int, choices=(128001, 128009), default=EOT,
                        help="SFT-supervised actor terminator: 128009 (Instruct EOT) or 128001 (base end-of-text)")
    args = parser.parse_args(argv)
    report = {"schema": "llama_protocol_v1", "status": "failed", "mode": "gpu" if args.gpu else "cpu",
              "actor_response_eos_id": args.actor_eos,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "paths": {"actor": str(args.actor.resolve()), "reward": str(args.reward.resolve()),
                        "init_adapter": str(args.init_adapter.resolve())},
              "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCES}}
    try:
        import torch
        import transformers
        report["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        report.update(audit_cpu(args.actor, args.reward, args.init_adapter, args.actor_eos))
        if args.gpu:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            require(torch.cuda.is_available(), "GPU protocol check requested but CUDA is unavailable")
            report["runtime"]["gpu_name"] = torch.cuda.get_device_name(0)
            tokenizer = AutoTokenizer.from_pretrained(args.reward, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.reward, torch_dtype=torch.bfloat16, local_files_only=True,
                attn_implementation="sdpa").to("cuda:0").eval()
            report["gpu"] = {"status": "failed"}
            audit_gpu_reward_model(model, tokenizer, "cuda:0", diagnostics=report["gpu"])
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "mode": report["mode"],
                      "output": str(args.output), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
