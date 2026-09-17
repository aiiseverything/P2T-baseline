"""CPU-only contracts for transferring the exact SSH training assets."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_ssh_assets as assets


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_manifest(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    model = root / "models/actor"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"model")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    adapter = root / "models/init-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "models/actor"}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    bench = root / "datasets/benchmark.jsonl"
    bench.parent.mkdir()
    bench.write_text('{"prompt":"one","key":1}\n{"prompt":"two","key":2}\n')
    parquet = root / "datasets/train.parquet"
    pq.write_table(pa.table({"prompt": ["one", "two"], "chosen": ["a", "b"],
                             "rejected": ["c", "d"]}), parquet)
    manifest = {"schema_version": 1, "assets": [
        {"name": "actor", "kind": "model", "path": "models/actor", "required": True,
         "source": {"type": "hf_model", "repo_id": "example/actor", "revision": "abc123"},
         "required_files": ["config.json", "tokenizer.json", "model.safetensors.index.json"],
         "key_file_sha256": {"config.json": digest(b"{}"),
                             "tokenizer.json": digest(b"{}")}},
        {"name": "init_adapter", "kind": "adapter", "path": "models/init-adapter",
         "required": True, "source": {"type": "transfer"},
         "base_model": "models/actor",
         "required_files": ["adapter_config.json"],
         "weights": {"file": "adapter_model.safetensors", "sha256": digest(b"adapter"),
                     "bytes": len(b"adapter")}},
        {"name": "benchmark", "kind": "jsonl", "path": "datasets/benchmark.jsonl",
         "required": True, "source": {"type": "transfer"},
         "sha256": digest(bench.read_bytes()), "bytes": bench.stat().st_size,
         "rows": 2, "fields": ["prompt", "key"]},
        {"name": "train", "kind": "parquet", "path": "datasets/train.parquet",
         "required": True,
         "source": {"type": "hf_file", "repo_id": "example/train", "repo_type": "dataset",
                    "revision": "def456", "filename": "train.parquet"},
         "sha256": digest(parquet.read_bytes()), "bytes": parquet.stat().st_size,
         "rows": 2, "fields": ["prompt", "chosen", "rejected"]},
    ]}
    return root, manifest


def test_offline_check_validates_adapter_hash_and_dataset_schema(tmp_path):
    root, manifest = fixture_manifest(tmp_path)
    report = assets.check_assets(root, manifest)
    assert report["ok"] is True
    assert {item["name"]: item["status"] for item in report["assets"]} == {
        "actor": "ok", "init_adapter": "ok", "benchmark": "ok", "train": "ok"}

    (root / "models/init-adapter/adapter_model.safetensors").write_bytes(b"changed")
    report = assets.check_assets(root, manifest)
    assert report["ok"] is False
    adapter = next(item for item in report["assets"] if item["name"] == "init_adapter")
    assert adapter["status"] == "invalid"
    assert any("sha256" in issue for issue in adapter["issues"])


def test_present_but_different_actor_tokenizer_is_rejected(tmp_path):
    root, manifest = fixture_manifest(tmp_path)
    (root / "models/actor/tokenizer.json").write_text('{"different":true}')
    report = assets.check_assets(root, manifest)
    actor = next(item for item in report["assets"] if item["name"] == "actor")
    assert actor["status"] == "invalid"
    assert any("tokenizer.json" in issue and "sha256" in issue for issue in actor["issues"])


def test_wrong_actor_config_hash_is_rejected_even_if_file_exists(tmp_path):
    root, manifest = fixture_manifest(tmp_path)
    (root / "models/actor/config.json").write_text('{"architectures":["wrong"]}')
    report = assets.check_assets(root, manifest)
    actor = next(item for item in report["assets"] if item["name"] == "actor")
    assert actor["status"] == "invalid"
    assert any("config.json" in issue and "sha256" in issue for issue in actor["issues"])


def test_download_refuses_to_mix_wrong_model_with_public_shards(tmp_path, monkeypatch):
    root, manifest = fixture_manifest(tmp_path)
    (root / "models/actor/config.json").write_text('{"architectures":["wrong"]}')
    monkeypatch.setattr(assets, "_download_hf_model",
                        lambda *a: pytest.fail("must not merge into a wrong model folder"))
    with pytest.raises(ValueError, match="invalid.*actor|actor.*invalid"):
        assets.download_public_assets(root, manifest)


def test_offline_check_reports_missing_transfer_asset_without_downloading(tmp_path, monkeypatch):
    root, manifest = fixture_manifest(tmp_path)
    (root / "datasets/benchmark.jsonl").unlink()
    monkeypatch.setattr(assets, "download_public_assets", lambda *a, **kw: pytest.fail("network path used"))
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(manifest))
    assert assets.main(["--root", str(root), "--manifest", str(path)]) == 1


def test_explicit_download_only_fetches_public_assets_and_never_adapter(tmp_path, monkeypatch):
    root, manifest = fixture_manifest(tmp_path)
    (root / "datasets/train.parquet").unlink()
    (root / "models/init-adapter/adapter_model.safetensors").unlink()
    calls = []

    def fake_hf_file(source, destination):
        calls.append((source["repo_id"], source["revision"], source["filename"]))
        destination.write_bytes(b"downloaded")

    monkeypatch.setattr(assets, "_download_hf_file", fake_hf_file)
    result = assets.download_public_assets(root, manifest)
    assert calls == [("example/train", "def456", "train.parquet")]
    assert result["downloaded"] == ["train"]
    assert result["requires_transfer"] == ["init_adapter"]


def test_model_download_allowlist_excludes_other_weight_formats(tmp_path, monkeypatch):
    root, manifest = fixture_manifest(tmp_path)
    actor = root / "models/actor"
    (actor / "model-00001-of-00001.safetensors").unlink()
    calls = []

    def fake_snapshot(source, destination):
        calls.append((source, destination))

    monkeypatch.setattr(assets, "_download_hf_model", fake_snapshot)
    result = assets.download_public_assets(root, manifest)
    assert result["downloaded"] == ["actor"]
    assert calls[0][0]["revision"] == "abc123"
    assert all(".bin" not in pattern for pattern in assets.MODEL_ALLOW_PATTERNS)
    assert any("*.safetensors" == pattern for pattern in assets.MODEL_ALLOW_PATTERNS)


def test_manifest_rejects_paths_outside_repository(tmp_path):
    root, manifest = fixture_manifest(tmp_path)
    manifest["assets"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="relative|outside"):
        assets.check_assets(root, manifest)


@pytest.mark.parametrize('filename,content,asset_name', [
    ('models/actor/model.safetensors.index.json', '{"weight_map":[]}', 'actor'),
    ('models/init-adapter/adapter_config.json', '[]', 'init_adapter'),
])
def test_malformed_asset_json_is_reported_invalid_without_crashing(tmp_path, filename, content, asset_name):
    root, manifest = fixture_manifest(tmp_path)
    (root / filename).write_text(content)
    report = assets.check_assets(root, manifest)
    assert report['ok'] is False
    row = next(item for item in report['assets'] if item['name'] == asset_name)
    assert row['status'] == 'invalid'
    assert any('invalid' in issue for issue in row['issues'])
