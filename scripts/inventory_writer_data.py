#!/usr/bin/env python3
"""Read-only, metadata-level inventory of both VPO-RM storage roots."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paper-writer-handoff-20260919/audit"
ROOTS = {"shared": ROOT, "data": Path("/data/VPO-RM")}


def category(path):
    parts = path.parts
    name = path.name.lower()
    if any(x in parts for x in [".git", ".venv", ".vllm-extra", ".hf_cache", ".cache", "cache", "caches", ".download-cache", ".download-tmp", ".tmp", "__pycache__", ".pytest_cache", ".mplconfig", ".matplotlib-cache", "site-packages", "dist-packages", "sandbox", "sandbox-canonical", "sandbox-selftest-v1"]):
        return "runtime_cache_and_dependencies"
    if name.endswith(".safetensors") or name.startswith("pytorch_model") and name.endswith(".bin"):
        return "adapter_weights" if "adapter" in name or any("lora" in x for x in parts) else "base_and_reward_model_weights"
    if name in ["trainer_state.pt", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        return "optimizer_training_state"
    if name.endswith("-credit.pt"):
        return "token_credit_tensors"
    if name.endswith("-probabilities.pt"):
        return "token_probability_tensors"
    if name.endswith((".pt", ".pth", ".npy", ".npz")):
        return "other_numeric_tensors"
    if name.startswith("rollout-") and name.endswith("-tokens.json"):
        return "rollout_response_token_ids"
    if name.startswith("rollout-") and name.endswith("-rewards.json"):
        return "rollout_per_response_rewards"
    if name.startswith("rollout-") and name.endswith("-prompts.json"):
        return "rollout_prompt_text"
    if name in ["metrics.jsonl", "credit_stats.jsonl", "profile_steps.jsonl", "train_log.jsonl", "train_metrics.jsonl"]:
        return "training_metric_series"
    if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        return "figures_and_documents"
    if path.suffix.lower() in [".log", ".out", ".err"]:
        return "execution_logs"
    if "datasets" in parts or name.endswith((".parquet", ".arrow")):
        return "datasets_and_prompt_sets"
    if path.suffix.lower() in [".py", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".ipynb"]:
        return "code_docs_and_text_metadata"
    if name.endswith((".tar.gz", ".tar", ".tgz", ".zip", ".gz", ".zst")):
        return "archives"
    if path.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv"]:
        if any("arena" in x for x in parts):
            return "arena_raw_and_summaries"
        if any(any(key in x for key in ["eval", "reward256", "reward-arena"]) for x in parts):
            return "benchmark_raw_and_summaries"
        return "configuration_audits_and_other_records"
    return "other_files"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen, totals, suites, links, errors = set(), defaultdict(lambda: [0, 0, 0, 0]), defaultdict(lambda: [0, 0, 0]), [], []
    count = 0
    started = datetime.now(timezone.utc).isoformat()
    with (OUT / "all_files.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root", "relative_path", "category", "suite", "bytes", "allocated_bytes", "mtime_ns", "device", "inode", "first_inode_occurrence"])
        for label, root in ROOTS.items():
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
                base = Path(directory)
                dirs[:] = sorted(d for d in dirs if base / d != OUT.parent)
                for name in list(dirs):
                    p = base / name
                    if p.is_symlink():
                        links.append(dict(root=label, path=str(p.relative_to(root)), target=os.readlink(p)))
                for name in sorted(files):
                    p = base / name
                    try:
                        st = p.lstat()
                    except OSError as e:
                        errors.append(str(e))
                        continue
                    rel = p.relative_to(root)
                    if stat.S_ISLNK(st.st_mode):
                        links.append(dict(root=label, path=str(rel), target=os.readlink(p)))
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    cat = category(rel)
                    suite = "/".join(rel.parts[:2]) if len(rel.parts) > 1 else rel.parts[0]
                    key = (st.st_dev, st.st_ino)
                    unique = key not in seen
                    seen.add(key)
                    allocated = st.st_blocks * 512
                    writer.writerow([label, str(rel), cat, suite, st.st_size, allocated, st.st_mtime_ns, st.st_dev, st.st_ino, int(unique)])
                    v = totals[(label, cat)]
                    v[0] += 1; v[1] += st.st_size; v[2] += allocated; v[3] += allocated if unique else 0
                    s = suites[(label, suite)]
                    s[0] += 1; s[1] += st.st_size; s[2] += allocated
                    count += 1
                if count and count % 10000 == 0:
                    handle.flush()
    for filename, grouping, columns in [
        ("storage_by_category.csv", totals, ["root", "category", "files", "logical_bytes", "allocated_bytes", "unique_inode_allocated_bytes"]),
        ("storage_by_suite.csv", suites, ["root", "suite", "files", "logical_bytes", "allocated_bytes"]),
    ]:
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(columns)
            writer.writerows([list(key) + value for key, value in sorted(grouping.items())])
    (OUT / "scan.json").write_text(json.dumps(dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        roots={k: str(v) for k, v in ROOTS.items()}, files=count, symlinks=links, errors=errors,
        excluded_directories=[str(OUT.parent)], policy="No directory symlinks followed; logical byte totals count each path; unique-inode allocation is separately recorded. No file contents inspected."), indent=2) + "\n")
    print(json.dumps(dict(files=count, symlinks=len(links), errors=len(errors), output=str(OUT))))
    for cat in sorted({key[1] for key in totals}):
        n = sum(value[0] for key, value in totals.items() if key[1] == cat)
        size = sum(value[1] for key, value in totals.items() if key[1] == cat)
        print(f"{cat:42s} {n:8d} {size / 1e9:10.3f} GB")


if __name__ == "__main__":
    main()
