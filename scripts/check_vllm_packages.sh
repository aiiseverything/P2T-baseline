#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
mods = ["torch", "vllm", "transformers", "peft", "accelerate", "datasets",
        "pyarrow", "jinja2", "PIL", "safetensors", "sentencepiece"]
for name in mods:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        print(f"{name}: OK {version}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
PY
