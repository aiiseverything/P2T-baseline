#!/usr/bin/env python3
"""Check exact SSH handoff assets offline; download public files only on request."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/ssh-a6000-assets.json"
MODEL_ALLOW_PATTERNS = (
    "*.safetensors", "*.json", "*.jinja", "*.txt", "*.md", "LICENSE*",
)
MODEL_IGNORE_PATTERNS = (
    "*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.h5", "*.msgpack",
    "*.tflite", "*.ckpt", "assets/*",
)


def _path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ValueError(f"Asset path must be relative and inside the repository: {relative}")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Asset path resolves outside the repository: {relative}")
    return resolved


def load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
        raise ValueError("Expected SSH asset manifest schema_version 1 and an assets list")
    names = [asset.get("name") for asset in manifest["assets"]]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Asset names must be unique nonempty strings")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_file(path: Path, asset: dict, issues: list[str]) -> None:
    if not path.is_file():
        issues.append("file missing")
        return
    if "bytes" in asset and path.stat().st_size != asset["bytes"]:
        issues.append("byte size mismatch")
    if "sha256" in asset and _sha256(path) != asset["sha256"]:
        issues.append("sha256 mismatch")


def _check_key_files(path: Path, asset: dict, issues: list[str]) -> None:
    for filename, expected in asset.get("key_file_sha256", {}).items():
        candidate = path / filename
        if not candidate.is_file():
            issues.append(f"key file missing: {filename}")
        elif _sha256(candidate) != expected:
            issues.append(f"key file sha256 mismatch: {filename}")


def _check_model(path: Path, asset: dict, issues: list[str]) -> None:
    for filename in asset.get("required_files", []):
        if not (path / filename).is_file():
            issues.append(f"required model file missing: {filename}")
    _check_key_files(path, asset, issues)
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = set(weight_map.values())
        if not shards or any(not isinstance(name, str) or Path(name).name != name
                             or not name.endswith(".safetensors") for name in shards):
            raise ValueError("invalid shard names")
        expected = asset.get("expected_shards")
        if expected is not None and len(shards) != expected:
            issues.append(f"expected {expected} model shards; index lists {len(shards)}")
        for shard in sorted(shards):
            if not (path / shard).is_file() or (path / shard).stat().st_size == 0:
                issues.append(f"model shard missing or empty: {shard}")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        issues.append(f"invalid model index: {type(exc).__name__}")


def _check_adapter(path: Path, asset: dict, issues: list[str]) -> None:
    for filename in asset.get("required_files", []):
        if not (path / filename).is_file():
            issues.append(f"required adapter file missing: {filename}")
    _check_key_files(path, asset, issues)
    weights = asset["weights"]
    _check_file(path / weights["file"], weights, issues)
    config = path / "adapter_config.json"
    if config.is_file() and asset.get("base_model"):
        try:
            actual = json.loads(config.read_text()).get("base_model_name_or_path")
            if actual != asset["base_model"]:
                issues.append("adapter base_model_name_or_path mismatch")
        except (ValueError, TypeError):
            issues.append("invalid adapter_config.json")
    expected = asset.get("identity")
    manifest = path / "sft_manifest.json"
    if expected and manifest.is_file():
        try:
            saved = json.loads(manifest.read_text())
            actual = {
                "response_eos": saved["config"]["response_eos"],
                "response_eos_id": saved["dataset_stats"]["response_eos_id"],
                "sample_count": saved["dataset_stats"]["kept"],
                "epochs": saved["config"]["epochs"],
                "seed": saved["config"]["seed"],
                "sample_data_sha256": saved["data_sha256"],
                "target_token_sha256": saved["dataset_stats"]["target_token_sha256"],
            }
            if actual != expected:
                issues.append("SFT adapter identity mismatch")
        except (ValueError, TypeError, KeyError):
            issues.append("invalid sft_manifest.json")


def _check_jsonl(path: Path, asset: dict, issues: list[str]) -> None:
    count = 0
    try:
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                if any(field not in row for field in asset.get("fields", [])):
                    issues.append(f"JSONL required field missing at row {count}")
                    break
                expected_generator = asset.get("reference_generator")
                if expected_generator and row.get("reference_generator") != expected_generator:
                    issues.append(f"reference_generator mismatch at row {count}")
                    break
        if "rows" in asset and count != asset["rows"]:
            issues.append(f"row count mismatch: {count}")
    except (UnicodeError, ValueError, TypeError) as exc:
        issues.append(f"invalid JSONL: {type(exc).__name__}")


def _check_parquet(path: Path, asset: dict, issues: list[str]) -> None:
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(path)
        names = set(pq.read_schema(path).names)
        if "rows" in asset and metadata.num_rows != asset["rows"]:
            issues.append(f"row count mismatch: {metadata.num_rows}")
        missing = set(asset.get("fields", [])) - names
        if missing:
            issues.append("Parquet required fields missing: " + ", ".join(sorted(missing)))
    except ImportError:
        issues.append("pyarrow is required to validate Parquet schema")
    except Exception as exc:
        issues.append(f"invalid Parquet: {type(exc).__name__}")


def check_assets(root: Path, manifest: dict) -> dict:
    """Read only: validate required paths, file hashes and lightweight schemas."""
    root = Path(root)
    rows = []
    for asset in manifest["assets"]:
        path = _path(root, asset["path"])
        required = asset.get("required", True)
        issues = []
        if not path.exists():
            status = "missing" if required else "optional_missing"
            if required:
                issues.append("asset missing")
        else:
            kind = asset["kind"]
            if kind == "model":
                _check_model(path, asset, issues)
            elif kind == "adapter":
                _check_adapter(path, asset, issues)
            elif kind in {"jsonl", "parquet"}:
                _check_file(path, asset, issues)
                if path.is_file():
                    if kind == "jsonl":
                        _check_jsonl(path, asset, issues)
                    else:
                        _check_parquet(path, asset, issues)
            else:
                raise ValueError(f"Unknown asset kind: {kind}")
            status = "invalid" if issues else "ok"
        row = {"name": asset["name"], "path": asset["path"],
               "required": required, "status": status, "issues": issues}
        if asset["kind"] == "model":
            row["weight_verification"] = "index_shards_present_and_nonempty_only"
        elif asset["kind"] == "adapter":
            row["weight_verification"] = "sha256"
        rows.append(row)
    return {"ok": all(row["status"] == "ok" for row in rows if row["required"]),
            "assets": rows}


def _download_hf_model(source: dict, destination: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=source["repo_id"], revision=source.get("revision"),
                      local_dir=str(destination), allow_patterns=list(MODEL_ALLOW_PATTERNS),
                      ignore_patterns=list(MODEL_IGNORE_PATTERNS))


def _download_hf_file(source: dict, destination: Path) -> None:
    from huggingface_hub import hf_hub_download
    filename = Path(source["filename"])
    if filename.is_absolute() or ".." in filename.parts:
        raise ValueError("Hugging Face filename must be repository-relative")
    local_dir = destination
    for _ in filename.parts:
        local_dir = local_dir.parent
    result = hf_hub_download(repo_id=source["repo_id"], filename=source["filename"],
                             repo_type=source.get("repo_type", "dataset"),
                             revision=source.get("revision"), local_dir=str(local_dir))
    if Path(result).resolve() != destination.resolve():
        raise RuntimeError("Hugging Face download returned an unexpected path")


def _download_url(source: dict, destination: Path, asset: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".asset-", delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            with urlopen(source["url"], timeout=120) as response:
                shutil.copyfileobj(response, tmp)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    issues = []
    _check_file(temporary, asset, issues)
    if issues:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {asset['name']} failed exact byte/hash validation: {issues}")
    os.replace(temporary, destination)


def download_public_assets(root: Path, manifest: dict) -> dict:
    """Explicit network path: fetch only missing public files; never fetch LoRA."""
    root = Path(root)
    report = check_assets(root, manifest)
    by_name = {row["name"]: row for row in report["assets"]}
    result = {"downloaded": [], "requires_transfer": [], "already_present": []}
    for asset in manifest["assets"]:
        if not asset.get("required", True):
            continue
        name = asset["name"]
        current = by_name[name]
        if current["status"] == "ok":
            result["already_present"].append(name)
            continue
        source = asset["source"]
        if source["type"] == "transfer":
            result["requires_transfer"].append(name)
            continue
        destination = _path(root, asset["path"])
        if current["status"] == "invalid" and destination.is_file():
            raise ValueError(f"Existing {name} is invalid; refusing to overwrite it automatically")
        if (asset["kind"] == "model" and destination.exists()
                and any("sha256 mismatch" in issue or "invalid model index" in issue
                        or "expected " in issue and "model shards" in issue
                        for issue in current["issues"])):
            raise ValueError(f"Existing {name} model is invalid; refusing to mix public shards into it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source["type"] == "hf_model":
            _download_hf_model(source, destination)
        elif source["type"] == "hf_file":
            _download_hf_file(source, destination)
        elif source["type"] == "url":
            _download_url(source, destination, asset)
        else:
            raise ValueError(f"Unknown public source type: {source['type']}")
        result["downloaded"].append(name)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="VPO-RM checkout root")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="offline validation (default)")
    mode.add_argument("--download", action="store_true", help="fetch missing public files, then validate")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    actions = download_public_assets(args.root, manifest) if args.download else None
    report = check_assets(args.root, manifest)
    if actions is not None:
        report["download_actions"] = actions
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
