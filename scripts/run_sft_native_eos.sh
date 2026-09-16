#!/usr/bin/env bash
# One-variable counterpart of sftv2-clean-2k5e2: replace the assistant EOS only.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/ma4agi-gpu/suminle/interests/VPO-RM}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?required}"
SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:?required}"
SFT_OUTPUT="${SFT_OUTPUT:?required}"
export PROJECT_ROOT EXPERIMENT_DIR SOURCE_SNAPSHOT SFT_OUTPUT
cd "$SOURCE_SNAPSHOT"
export PYTHONPATH="$SOURCE_SNAPSHOT:$PROJECT_ROOT/.vllm-extra${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$EXPERIMENT_DIR"
exec > >(tee -a "$EXPERIMENT_DIR/job.log") 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$EXPERIMENT_DIR/exit_code"' EXIT

if [[ -e "$SFT_OUTPUT" ]]; then
  echo "Refusing to overwrite an existing adapter: $SFT_OUTPUT" >&2
  exit 1
fi
python3 - <<'PY'
import os, shutil
root = os.environ["PROJECT_ROOT"]
free = shutil.disk_usage(root).free
print(f"Shared storage available: {free / 2**30:.2f} GiB", flush=True)
if free < 8 * 2**30:
    raise SystemExit("Need at least 8 GiB free before this single-adapter run")
PY

if ! python3 -c 'import peft, pyarrow' 2>/dev/null; then
  python3 -m pip install --no-cache-dir \
    --index-url http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    'peft==0.20.0' 'pyarrow>=15,<22'
fi

printf '%s\n' preflight > "$EXPERIMENT_DIR/stage"
python3 -m unittest discover -s "$SOURCE_SNAPSHOT/tests" -p 'test_sft_response_tokens.py' -v
python3 - <<'PY'
import hashlib, importlib.metadata, json, os, random, shutil
from pathlib import Path
import torch
from scripts.sft_init import load_chosen_pairs

root = Path(os.environ["PROJECT_ROOT"])
baseline = json.loads((root / "models/sftv2-clean-2k5e2/sft_manifest.json").read_text())
pairs, split = load_chosen_pairs(str(root / "datasets/sft_v2/sft_clean.parquet"), 2000)
pairs = random.Random(42).sample(pairs, 2500)
digest = hashlib.sha256("\n".join(f"{p}|{c[:64]}" for p, c in pairs).encode()).hexdigest()
assert digest == baseline["data_sha256"], "Training examples differ from the control"
assert split == baseline["split"], "Data split differs from the control"
assert torch.cuda.device_count() == 1, "Expected exactly one visible GPU"
runtime = {
    "baseline": "models/sftv2-clean-2k5e2", "data_sha256": digest,
    "split": split, "gpu": torch.cuda.get_device_name(0),
    "software": {name: importlib.metadata.version(name)
                 for name in ["torch", "transformers", "peft", "vllm", "pyarrow"]},
    "free_bytes_at_start": shutil.disk_usage(root).free,
    "source_snapshot": os.environ["SOURCE_SNAPSHOT"],
    "only_training_change": "Final assistant im_end ID 151645 -> native EOS ID 151643; trailing newline retained",
}
(Path(os.environ["EXPERIMENT_DIR"]) / "runtime.json").write_text(json.dumps(runtime, indent=2))
print(json.dumps(runtime, indent=2), flush=True)
PY

printf '%s\n' training > "$EXPERIMENT_DIR/stage"
python3 "$SOURCE_SNAPSHOT/scripts/sft_init.py" \
  --model models/Qwen3-14B-Base \
  --dataset-path "$PROJECT_ROOT/datasets/sft_v2/sft_clean.parquet" \
  --output "$SFT_OUTPUT" \
  --max-examples 2500 --epochs 2 --learning-rate 1e-4 \
  --micro-batch 4 --grad-accum 8 --max-len 4096 \
  --warmup-ratio 0.03 --seed 42 --monitor-every 100 \
  --monitor-prompts "$PROJECT_ROOT/datasets/sft_v2/monitor_prompts.json" \
  --eos-weight 1.0 --response-eos native

printf '%s\n' evaluating > "$EXPERIMENT_DIR/stage"
for split in train test; do
  python3 "$SOURCE_SNAPSHOT/scripts/eval_alpaca.py" \
    --model models/Qwen3-14B-Base \
    --output "$EXPERIMENT_DIR/tvt" --max-tokens 2048 \
    --recipes 1.0:1:1.0:-1 --seed 42 \
    --dataset "$PROJECT_ROOT/datasets/sft_v2/${split}25.jsonl" \
    --adapters "control-${split}=$PROJECT_ROOT/models/sftv2-clean-2k5e2" \
               "native-${split}=$SFT_OUTPUT"
done

python3 - <<'PY'
import collections, json, os, statistics
from pathlib import Path
root, out = Path(os.environ["PROJECT_ROOT"]), Path(os.environ["EXPERIMENT_DIR"])
summary = {}
for split in ["train", "test"]:
    paths = {"native": out / f"tvt/native-{split}/generations_t1.0_n1.jsonl",
             "control": out / f"tvt/control-{split}/generations_t1.0_n1.jsonl"}
    rows = {tag: [json.loads(line) for line in path.open() if line.strip()]
            for tag, path in paths.items()}
    assert len(rows["native"]) == len(rows["control"]) == 25
    assert [r["instruction"] for r in rows["native"]] == [r["instruction"] for r in rows["control"]]
    summary[split] = {}
    for tag, data in rows.items():
        lengths = [r["response_tokens"] for r in data]
        summary[split][tag] = {
            "n": len(data), "mean_tokens": statistics.mean(lengths),
            "median_tokens": statistics.median(lengths),
            "at_cap": sum(n >= 2048 for n in lengths),
            "at_cap_fraction": sum(n >= 2048 for n in lengths) / len(data),
            "finish_reasons": dict(collections.Counter(r.get("finish_reason", "unrecorded") for r in data)),
            "last_token_ids": dict(collections.Counter(str(r.get("last_token_id", "unrecorded")) for r in data)),
            "source": str(paths[tag]),
        }
(out / "comparison.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
PY
printf '%s\n' complete > "$EXPERIMENT_DIR/stage"
