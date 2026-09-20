#!/usr/bin/env python3
"""Fetch the models, dataset and benchmark prompts this baseline needs.

Uses only the standard library so it can run before the pinned environment
exists, and resumes partial files rather than restarting them -- the actor is
27.5 GiB.

    python scripts/prepare_assets.py --download        # fetch what is missing
    python scripts/prepare_assets.py --check           # report what is present

Benchmark prompt files are required even for RL-only runs: the data pipeline
removes their prompts from the training corpus before splitting, so a run that
silently skipped the exclusion would train on evaluation data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# This host has no IPv6 route, and urllib would otherwise prefer a AAAA record
# and fail with "Network is unreachable" while curl over IPv4 succeeds.
_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    results = _getaddrinfo(*args, **kwargs)
    return [item for item in results if item[0] == socket.AF_INET] or results


socket.getaddrinfo = _ipv4_first

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"
HF_API = "https://huggingface.co/api"
ROWS_API = "https://datasets-server.huggingface.co/rows"

MODELS = {
    "models/Qwen3-14B-Base": "Qwen/Qwen3-14B-Base",
    "models/Skywork-Reward-V2-Qwen3-8B": "Skywork/Skywork-Reward-V2-Qwen3-8B",
}
DATASET_FILES = {
    "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet":
        ("HuggingFaceH4/ultrafeedback_binarized", "data/train_prefs-00000-of-00001.parquet"),
    "datasets/ifeval/ifeval_input_data.jsonl": ("google/IFEval", "ifeval_input_data.jsonl"),
    "datasets/alpacaeval/alpaca_eval.json": ("tatsu-lab/alpaca_eval", "alpaca_eval.json"),
}
SKIP_SUFFIXES = (".md", ".gitattributes", ".pth", ".msgpack", ".h5", ".ot", ".tflite")


def fetch(url: str, destination: Path, *, attempts: int = 6) -> None:
    """Download with resume; Hugging Face supports Range requests on resolve URLs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        offset = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "p2t-baseline/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = response.status == 206
                if offset and not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                with destination.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1 << 20)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1}/{attempts} after {type(error).__name__}: {error}",
                  file=sys.stderr)
            time.sleep(2 ** attempt)


def repo_files(repo: str, *, attempts: int = 6) -> dict[str, int]:
    """Map every repository file to its byte size, with retry.

    Uses ``?blobs=true`` so the declared size is available: a download that was
    interrupted leaves a short file that looks present, and size is the only
    cheap way to tell a complete shard from a truncated one.  The metadata call
    has no resumable body, so it needs its own retry.
    """
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{HF_API}/models/{repo}?blobs=true", timeout=60) as response:
                payload = json.load(response)
            return {item["rfilename"]: int(item.get("size") or 0)
                    for item in payload.get("siblings", [])}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == attempts - 1:
                raise
            print(f"  metadata retry {attempt + 1}/{attempts} after "
                  f"{type(error).__name__}: {error}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return {}


def model_problems(repo: str, target: Path) -> list[tuple[str, int, int]]:
    """Files whose on-disk size disagrees with the repository's declared size."""
    problems = []
    for name, expected in repo_files(repo).items():
        if name.endswith(SKIP_SUFFIXES):
            continue
        path = target / name
        actual = path.stat().st_size if path.is_file() else -1
        if expected and actual != expected:
            problems.append((name, actual, expected))
    return problems


def download_model(repo: str, target: Path) -> None:
    sizes = {name: size for name, size in repo_files(repo).items()
             if not name.endswith(SKIP_SUFFIXES)}
    print(f"{repo}: {len(sizes)} files -> {target}")
    # Size, not existence: an interrupted transfer leaves a short file behind,
    # and `fetch` resumes from whatever is on disk rather than restarting.
    incomplete = [name for name, expected in sizes.items()
                  if not (target / name).is_file()
                  or (target / name).stat().st_size != expected]
    if not incomplete:
        print("  already complete")
        return
    total = sum(sizes[name] for name in incomplete)
    print(f"  {len(incomplete)} file(s) to fetch or repair, {total / 2**30:.1f} GiB")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: _one(repo, target, name), incomplete))
    remaining = [(name, (target / name).stat().st_size if (target / name).is_file() else -1,
                  sizes[name]) for name in sizes
                 if not (target / name).is_file() or (target / name).stat().st_size != sizes[name]]
    if remaining:
        raise RuntimeError(f"{len(remaining)} file(s) still incomplete after download: "
                           f"{[name for name, _, _ in remaining]}")


def _one(repo: str, target: Path, name: str) -> None:
    destination = target / name
    before = destination.stat().st_size if destination.exists() else 0
    if not hub_download(repo, name, destination, repo_type="model"):
        fetch(f"{HF}/{repo}/resolve/main/{name}", destination)
    after = destination.stat().st_size
    print(f"  {name}: {before / 2**20:.0f} -> {after / 2**20:.0f} MiB")


def hub_download(repo: str, remote: str, target: Path, *, repo_type: str) -> bool:
    """Fetch through ``huggingface_hub`` when it is importable.

    Hand-rolled urllib gets HTTP 401 on the Hub's signed CDN redirects, and it
    cannot verify an etag.  The library handles auth, redirects, chunked resume
    and integrity, so it is used whenever the pinned environment is present;
    the stdlib path stays for bootstrapping before that environment exists.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = hf_hub_download(repo_id=repo, filename=remote, repo_type=repo_type,
                             local_dir=str(target.parent.parent))
    staged_path = Path(staged)
    if staged_path.resolve() != target.resolve() and staged_path.is_file():
        shutil.copyfile(staged_path, target)
    return True


def download_dataset_file(repo: str, remote: str, target: Path,
                          *, dataset: bool = True) -> None:
    print(f"{repo}/{remote} -> {target}")
    if target.is_file() and target.stat().st_size:
        return
    if hub_download(repo, remote, target,
                    repo_type="dataset" if dataset else "model"):
        return
    fetch(f"{HF}/{repo}/resolve/main/{remote}", target)


def build_jsonl_from_rows(dataset: str, config: str, split: str, field: str,
                          target: Path) -> None:
    """Pull a small split through the datasets-server rows API and write JSONL."""
    print(f"{dataset}/{config}/{split} -> {target} (rows api)")
    rows, offset, length = [], 0, 100
    while True:
        url = f"{ROWS_API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        if offset >= payload.get("num_rows_total", offset):
            break
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({field: row[field]}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows")


def alpaca_reference(source: Path, target: Path) -> None:
    """The project's AlpacaEval reference file is this set, one row per line."""
    if target.exists() and target.stat().st_size:
        return
    rows = json.loads(source.read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  converted {len(rows)} alpaca rows -> {target}")


def gsm8k_jsonl(target: Path) -> None:
    if target.exists() and target.stat().st_size:
        return
    local = target.parent / "test.parquet"
    if hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", local,
                    repo_type="dataset"):
        import pyarrow.parquet as pq
        table = pq.read_table(local, columns=["question"])
        with target.open("w") as handle:
            for question in table.column("question").to_pylist():
                handle.write(json.dumps({"question": question}, ensure_ascii=False) + "\n")
        print(f"  wrote {len(table)} gsm8k rows -> {target}")
        return
    build_jsonl_from_rows("openai/gsm8k", "main", "test", "question", target)


def check() -> int:
    required = [
        ROOT / "models/Qwen3-14B-Base/config.json",
        ROOT / "models/Qwen3-14B-Base/tokenizer.json",
        ROOT / "models/Skywork-Reward-V2-Qwen3-8B/config.json",
        ROOT / "datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet",
        ROOT / "datasets/ifeval/ifeval_input_data.jsonl",
        ROOT / "datasets/gsm8k/test.jsonl",
        ROOT / "datasets/alpacaeval/eval_gpt4turbo_reference.jsonl",
    ]
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    for path in required:
        state = "ok" if path.is_file() and path.stat().st_size else "MISSING"
        size = f"{path.stat().st_size / 2**20:.1f} MiB" if path.is_file() else "-"
        print(f"{state:8} {size:>12}  {path.relative_to(ROOT)}")
    # Declared size, not presence: a truncated shard is a file that exists, and
    # loading it fails only once the run has already started.
    for name, repo in ((relative.split("/")[1], repo) for relative, repo in MODELS.items()):
        problems = model_problems(repo, ROOT / "models" / name)
        if problems:
            missing.append(ROOT / "models" / name)
            for filename, actual, expected in problems:
                print(f"INCOMPLETE  {actual / 2**20:>8.1f}/{expected / 2**20:.1f} MiB  "
                      f"models/{name}/{filename}", file=sys.stderr)
        else:
            total = sum((ROOT / "models" / name / f).stat().st_size
                        for f in repo_files(repo) if f.endswith(".safetensors"))
            print(f"         {total / 2**30:>9.2f} GiB  {name} (verified against the Hub)")
    if missing:
        print(f"\n{len(missing)} required asset(s) missing or incomplete", file=sys.stderr)
        return 1
    print("\nall assets present and complete")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--models-only", action="store_true",
                        help="skip datasets (useful to start the long download first)")
    args = parser.parse_args(argv)

    if args.check:
        return check()

    for relative, repo in MODELS.items():
        download_model(repo, ROOT / relative)
    if args.models_only:
        return 0
    for relative, (repo, remote) in DATASET_FILES.items():
        download_dataset_file(repo, remote, ROOT / relative)
    alpaca_reference(ROOT / "datasets/alpacaeval/alpaca_eval.json",
                     ROOT / "datasets/alpacaeval/eval_gpt4turbo_reference.jsonl")
    gsm8k_jsonl(ROOT / "datasets/gsm8k/test.jsonl")
    return check()


if __name__ == "__main__":
    sys.exit(main())
