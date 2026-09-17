#!/usr/bin/env python3
"""Pre-flight checks for the Skywork UltraFeedback experiment.

The checker is intentionally dependency-light so it can run on a login node.  It
verifies the resolved experiment configuration, local model artifacts,
tokenizers, dataset files/schema, and the configured GPU resource budget. Checks that
need optional packages (``torch``, ``pyarrow`` or ``datasets``) are reported as
warnings unless the corresponding ``--require-*`` switch is supplied.

Examples::

    python scripts/check_readiness.py --setting ultrafeedback_skywork_8b
    python scripts/check_readiness.py --setting ultrafeedback_skywork_8b \
        --require-data-reader --require-gpu
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
import unicodedata


ROOT = Path(__file__).resolve().parents[1]


def _ok(name: str, detail: str, out: list[dict[str, Any]]) -> None:
    out.append({"name": name, "status": "ok", "detail": detail})


def _warn(name: str, detail: str, out: list[dict[str, Any]]) -> None:
    out.append({"name": name, "status": "warning", "detail": detail})


def _fail(name: str, detail: str, out: list[dict[str, Any]]) -> None:
    out.append({"name": name, "status": "error", "detail": detail})


def resolve_model(ref: str) -> Path | None:
    """Resolve a HF model id to a local model directory when available."""
    p = Path(ref).expanduser()
    if p.is_dir():
        return p.resolve()
    candidate = ROOT / "models" / ref.split("/")[-1]
    return candidate.resolve() if candidate.is_dir() else None


def normalized_prompt(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    # Keep this exactly in line with the experiment specification: NFC followed
    # by whitespace normalization (including line breaks and repeated spaces).
    return " ".join(unicodedata.normalize("NFC", text).split())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_dependencies(results: list[dict[str, Any]], require: bool) -> None:
    required = ["torch", "transformers", "peft", "accelerate"]
    data_readers = ["pyarrow", "datasets"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        ( _fail if require else _warn)("python-dependencies", "missing: " + ", ".join(missing), results)
    else:
        _ok("python-dependencies", "torch, transformers, peft and accelerate are importable", results)
    readers = [m for m in data_readers if importlib.util.find_spec(m) is not None]
    if not readers:
        (_fail if require else _warn)("dataset-reader", "install pyarrow or datasets to inspect parquet", results)
    else:
        _ok("dataset-reader", "available: " + ", ".join(readers), results)


def check_model(ref: str, role: str, expected_pad: str, results: list[dict[str, Any]],
                tokenizer_audit: dict[str, Any] | None) -> Path | None:
    model_dir = resolve_model(ref)
    if model_dir is None:
        _fail(f"{role}-files", f"model {ref!r} is not present locally (looked under {ROOT / 'models'})", results)
        return None
    config_path = model_dir / "config.json"
    tok_path = model_dir / "tokenizer.json"
    index_path = model_dir / "model.safetensors.index.json"
    missing = [str(p.name) for p in (config_path, tok_path) if not p.exists()]
    if missing:
        _fail(f"{role}-files", f"{model_dir}: missing {', '.join(missing)}", results)
        return model_dir
    cfg = _read_json(config_path)
    if cfg.get("vocab_size") != 151936:
        _fail(f"{role}-config", f"unexpected vocab_size={cfg.get('vocab_size')}", results)
    elif cfg.get("model_type") != "qwen3":
        _fail(f"{role}-config", f"unexpected model_type={cfg.get('model_type')}", results)
    else:
        _ok(f"{role}-config", f"qwen3 vocab_size={cfg['vocab_size']}", results)
    architectures = cfg.get("architectures") or []
    expected_arch = "Qwen3ForSequenceClassification" if role == "rm" else "Qwen3ForCausalLM"
    if expected_arch not in architectures:
        _fail(f"{role}-architecture", f"expected {expected_arch}, found {architectures}", results)
    else:
        _ok(f"{role}-architecture", expected_arch, results)

    if index_path.exists():
        index = _read_json(index_path)
        shards = set(index.get("weight_map", {}).values())
        absent = sorted(name for name in shards
                        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0)
        if not shards:
            _fail(f"{role}-weights", "safetensors index contains no weights", results)
        elif absent:
            _fail(f"{role}-weights", "missing safetensors shard(s): " + ", ".join(absent), results)
        else:
            size = sum((model_dir / n).stat().st_size for n in shards)
            _ok(f"{role}-weights", f"{len(shards)} shards, {size / 2**30:.2f} GiB", results)
    elif (model_dir / "model.safetensors").is_file() and (model_dir / "model.safetensors").stat().st_size:
        _ok(f"{role}-weights", "single model.safetensors present", results)
    else:
        _fail(f"{role}-weights", "missing nonempty model.safetensors or shard index", results)

    tok_cfg = _read_json(model_dir / "tokenizer_config.json") if (model_dir / "tokenizer_config.json").exists() else {}
    configured_pad = tok_cfg.get("pad_token")
    if configured_pad != expected_pad:
        _warn(f"{role}-padding", f"tokenizer_config pad_token={configured_pad!r}; loader must override to {expected_pad!r}", results)
    else:
        _ok(f"{role}-padding", f"pad_token={expected_pad!r}", results)

    # Compare recorded artifact hashes when the audit has an entry for this ref.
    if tokenizer_audit:
        for item in tokenizer_audit.get("artifacts", []):
            if item.get("model") in {ref, model_dir.name}:
                bad = []
                for fn, want in item.get("sha256", {}).items():
                    fp = model_dir / fn
                    if fp.exists():
                        got = hashlib.sha256(fp.read_bytes()).hexdigest()
                        if got != want:
                            bad.append(fn)
                if bad:
                    _fail(f"{role}-hashes", "hash mismatch: " + ", ".join(bad), results)
                else:
                    _ok(f"{role}-hashes", "recorded tokenizer/config hashes match", results)
                break
    return model_dir


def check_tokenizers(actor: Path | None, rm: Path | None, pad_token: str,
                     results: list[dict[str, Any]]) -> None:
    if actor is None or rm is None or importlib.util.find_spec("transformers") is None:
        _warn("tokenizer-compatibility", "deferred (transformers or model directory unavailable)", results)
        return
    try:
        from transformers import AutoTokenizer
        a, b = AutoTokenizer.from_pretrained(str(actor)), AutoTokenizer.from_pretrained(str(rm))
        for tok in (a, b):
            tok.pad_token = pad_token
        if a.get_vocab() != b.get_vocab():
            _fail("tokenizer-compatibility", "actor and Skywork vocabularies differ", results)
        elif a.pad_token_id != b.pad_token_id:
            _fail("tokenizer-compatibility", f"padding IDs differ: {a.pad_token_id} vs {b.pad_token_id}", results)
        else:
            _ok("tokenizer-compatibility", f"shared vocabulary and pad_token_id={a.pad_token_id}", results)
    except Exception as exc:
        _fail("tokenizer-compatibility", f"could not load tokenizer ({type(exc).__name__}: {exc})", results)


def check_dataset(cfg: dict[str, Any], results: list[dict[str, Any]], require_reader: bool) -> None:
    data = cfg["data"]
    root = ROOT / "datasets" / "ultrafeedback_binarized"
    files = sorted(root.glob("data/*prefs*.parquet"))
    train = [p for p in files if p.name.startswith("train_prefs")]
    if not train:
        _fail("dataset-files", f"missing train_prefs parquet under {root / 'data'}", results)
        return
    _ok("dataset-files", f"found {train[0].name} ({train[0].stat().st_size / 2**20:.1f} MiB)", results)
    reader = None
    try:
        import pyarrow.parquet as pq
        # Return the selected column directly; this keeps the pyarrow and
        # datasets code paths identical below.
        reader = lambda p: pq.read_table(p, columns=[data["prompt_field"]])[data["prompt_field"]]
    except Exception:
        try:
            from datasets import load_dataset
            reader = lambda p: load_dataset("parquet", data_files=str(p), split="train")[data["prompt_field"]]
        except Exception:
            pass
    if reader is None:
        (_fail if require_reader else _warn)("dataset-schema", "cannot inspect parquet schema without pyarrow/datasets", results)
        return
    try:
        prompts = reader(train[0])
        # pyarrow yields Scalar objects while datasets yields plain Python
        # values.  Convert both to the same representation before hashing.
        prompts = [x.as_py() if hasattr(x, "as_py") else x for x in prompts]
        if not prompts or not isinstance(prompts[0], str):
            raise ValueError(f"field {data['prompt_field']!r} is not a non-empty string column")
        count = len(prompts)
        keys = [hashlib.sha256(normalized_prompt(x).encode()).hexdigest() for x in prompts]
        dup = count - len(set(keys))
        if count <= data["validation_prompts"]:
            _fail("dataset-split", f"{count} rows <= validation target {data['validation_prompts']}", results)
        elif dup:
            _warn("dataset-split", f"{count} rows, {dup} duplicate normalized prompts; pipeline must deduplicate", results)
        else:
            _ok("dataset-split", f"{count} rows; normalized prompt keys are unique", results)
        split_hash = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
        _ok("dataset-split-hash", f"full normalized-key SHA256={split_hash}", results)
    except Exception as exc:
        (_fail if require_reader else _warn)("dataset-schema", f"read failed: {type(exc).__name__}: {exc}", results)


def check_gpu(cfg: dict[str, Any], results: list[dict[str, Any]], max_gpus: int, require: bool) -> None:
    requested = int(cfg["hardware"].get("gpu_count", 0))
    if requested > max_gpus:
        _fail("gpu-budget", f"config requests {requested} GPUs; hard limit is {max_gpus}", results)
    else:
        _ok("gpu-budget", f"config requests {requested} GPU(s), within limit {max_gpus}", results)
    try:
        import torch
        available = torch.cuda.device_count()
        if available < requested:
            (_fail if require else _warn)("gpu-availability", f"CUDA reports {available} GPU(s), need {requested}", results)
        else:
            names = [torch.cuda.get_device_name(i) for i in range(available)]
            _ok("gpu-availability", f"CUDA reports {available}: {', '.join(names[:requested])}", results)
    except Exception as exc:
            (_fail if require else _warn)("gpu-availability", f"CUDA unavailable ({type(exc).__name__}: {exc})", results)


def check_experiment_contract(cfg: dict[str, Any], setting: dict[str, Any] | None,
                              results: list[dict[str, Any]]) -> None:
    """Check values that define the Skywork main-run contract, not just artifacts."""
    if setting is None:
        return
    uses_vllm_device = cfg.get("hardware", {}).get("vllm_device") is not None
    checks = {
        "hardware.gpu_count": (cfg.get("hardware", {}).get("gpu_count"), 3 if uses_vllm_device else 2),
        "hardware.actor_device": (cfg.get("hardware", {}).get("actor_device"), "cuda:0"),
        "hardware.rm_device": (cfg.get("hardware", {}).get("rm_device"), "cuda:1"),
        "data.max_prompt_tokens": (cfg.get("data", {}).get("max_prompt_tokens"), 2048),
        "data.max_response_tokens": (cfg.get("data", {}).get("max_response_tokens"), 2048),
        "data.validation_prompts": (cfg.get("data", {}).get("validation_prompts"), 2000),
        "shared.prompts_per_rollout": (cfg.get("shared", {}).get("prompts_per_rollout"), 8),
        "shared.group_size": (cfg.get("shared", {}).get("group_size"), 8),
        "shared.rollout_iterations": (cfg.get("shared", {}).get("rollout_iterations"), 500),
        "shared.policy_epochs_per_rollout": (cfg.get("shared", {}).get("policy_epochs_per_rollout"), 1),
        "shared.optimizer_minibatch_responses": (cfg.get("shared", {}).get("optimizer_minibatch_responses"), 64),
        "shared.top_k": (cfg.get("shared", {}).get("top_k"), 0),
        "data.enable_thinking": (cfg.get("data", {}).get("enable_thinking"), False),
        "shared.microbatch_responses": (cfg.get("hardware", {}).get("microbatch_responses"), 1),
        "shared.generation_microbatch_responses": (cfg.get("hardware", {}).get("generation_microbatch_responses"), 1),
    }
    if uses_vllm_device:
        checks["hardware.vllm_device"] = (cfg["hardware"]["vllm_device"], "cuda:2")
    bad = [f"{key}={got!r} (expected {want!r})" for key, (got, want) in checks.items()
           if got != want]
    if bad:
        _fail("experiment-contract", "; ".join(bad), results)
    elif "vpo_rm" not in cfg.get("methods", []):
        _fail("experiment-contract", "methods does not include vpo_rm", results)
    elif "skywork" not in str(setting.get("rm", "")).lower():
        _fail("experiment-contract", f"setting RM is not Skywork: {setting.get('rm')!r}", results)
    else:
        _ok("experiment-contract", "Skywork main-run parameters match experiments.json", results)


def check_trainer(results: list[dict[str, Any]]) -> None:
    """Perform a static gate before importing torch-heavy training code."""
    candidates = [ROOT / "trainer.py", ROOT / "vpo_rm" / "trainer.py"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        _fail("trainer-entrypoint", f"missing trainer.py (looked in {', '.join(map(str, candidates))})", results)
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        required = {"TrainerConfig", "VPOTrainer", "load_prompt_dataset", "main"}
        missing = sorted(required - names)
        if missing:
            _fail("trainer-entrypoint", "missing definitions: " + ", ".join(missing), results)
        else:
            _ok("trainer-entrypoint", "trainer.py exposes config, trainer, dataset loader and CLI", results)
    except SyntaxError as exc:
        _fail("trainer-entrypoint", f"syntax error at line {exc.lineno}: {exc.msg}", results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "experiments.json")
    ap.add_argument("--setting", default="ultrafeedback_skywork_8b")
    ap.add_argument("--max-gpus", type=int, default=3)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--require-deps", action="store_true")
    ap.add_argument("--require-data-reader", action="store_true")
    ap.add_argument("--json", type=Path, help="also write machine-readable report")
    args = ap.parse_args()
    cfg = _read_json(args.config)
    results: list[dict[str, Any]] = []
    check_dependencies(results, args.require_deps)
    audit_path = args.config.parent / "tokenizer_audit.json"
    audit = _read_json(audit_path) if audit_path.exists() else None
    actor = check_model(cfg["actor"], "actor", cfg["shared"]["pad_token"], results, audit)
    setting = next((x for x in cfg.get("settings", []) if x.get("name") == args.setting), None)
    if setting is None:
        _fail("setting", f"unknown setting {args.setting!r}", results)
        rm = None
    else:
        rm = check_model(setting["rm"], "rm", cfg["shared"]["pad_token"], results, audit)
    check_tokenizers(actor, rm, cfg["shared"]["pad_token"], results)
    check_experiment_contract(cfg, setting, results)
    check_dataset(cfg, results, args.require_data_reader)
    check_gpu(cfg, results, args.max_gpus, args.require_gpu)
    check_trainer(results)
    report = {"config": str(args.config), "setting": args.setting, "results": results,
              "ready": not any(x["status"] == "error" for x in results)}
    for x in results:
        print(f"[{x['status'].upper():7}] {x['name']}: {x['detail']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
