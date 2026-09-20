#!/usr/bin/env bash
# Build the pinned P2T runtime: Python 3.12 + CUDA 12.9 wheels + the project lock.
#
# The host python (/opt/venv/main, torch 2.4.1+cu121) is incompatible with the
# project's recorded runtime, so every dependency comes from the lock file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UV_BIN="${HOME}/.local/bin/uv"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${ROOT}/.cache/uv"
# Several CUDA wheels are hundreds of MB and the default timeout drops them.
export UV_HTTP_TIMEOUT=600

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "[setup] uv $(uv --version)"
uv python install 3.12
uv venv --python 3.12 --allow-existing "${ROOT}/.venv"

# The internal pip mirror (172.22.1.36) is unreachable from this host; the
# public indexes are reachable and are what the lock was resolved against.
# uv caches completed downloads, so a retry resumes rather than restarts.
for attempt in 1 2 3 4; do
  if uv pip sync --python "${ROOT}/.venv/bin/python" \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      --index-strategy unsafe-best-match \
      "${ROOT}/requirements/ssh-a6000-cu129.txt"; then
    break
  fi
  echo "[setup] sync attempt ${attempt} failed; retrying"
  sleep 5
  if [ "${attempt}" = 4 ]; then
    echo "[setup] sync failed after 4 attempts" >&2
    exit 1
  fi
done

# The lock pins vllm and torch; add the few runtime extras P2T needs that the
# H200 lock did not carry (matplotlib for the reward curves).
uv pip install --python "${ROOT}/.venv/bin/python" \
  --index-url https://pypi.org/simple \
  "matplotlib>=3.8" "pytest>=8"

echo "[setup] done: ${ROOT}/.venv"
