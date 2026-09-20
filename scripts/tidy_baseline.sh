#!/usr/bin/env bash
# Tidy the baseline working copy.  Moves only, never deletes, so every step is
# reversible, and touches nothing git tracks.
#
#   bash scripts/tidy_baseline.sh --dry-run   # show the plan (default)
#   bash scripts/tidy_baseline.sh --apply     # do it
#
# What it does NOT touch, deliberately:
#   * the parent project's tracked root files -- the Chinese-named design notes
#     (数学原理.md, 工程实现.md, tau失配与修复方案.md, ...), AGENTS.md, README.md,
#     check_models.py, setup_env.sh, pyproject.toml.  This directory is a working
#     copy of VPO-RM on the p2t-baseline branch; relocating tracked files would
#     produce a large spurious diff against upstream, which is not what tidying
#     means here.
#   * models/, datasets/, .venv/, .wheels/ -- 43+ GiB of prefetched assets.
#   * paper/ -- the PDF and its extracted text, gitignored and needed for review.
#   * the parent project's own evaluation runs under runs/.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:---dry-run}"

say() { printf '%s\n' "$*"; }
move() {  # move() <src> <dest-dir>
  local src="$1" dest="$2"
  [ -e "$src" ] || return 0
  if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    say "  SKIP (git-tracked) $src"
    return 0
  fi
  if [ "$MODE" = "--apply" ]; then
    mkdir -p "$dest"
    mv "$src" "$dest"/
    say "  moved   $src -> $dest/"
  else
    say "  would move $src -> $dest/"
  fi
}

say "mode: $MODE  (use --apply to make changes)"
say
say "1. Setup and build logs from the repository root -> logs/"
for f in .adapter_check.log .setup_env.log .setup_assets.log .setup_imports.log \
         .setup_wheels.log .pytest_p2t.log; do
  move "$f" logs
done

say
say "2. Finished and failed run directories -> runs/_archive/"
# The two standalone vLLM probes and the pilot that died at engine startup.
# Kept rather than deleted: their logs are the evidence for the all-reduce fix.
for d in runs/vllm-probe runs/vllm-probe2 runs/p2t-pilot; do
  move "$d" runs/_archive
done
move reports/p2t-pilot reports/_archive

say
say "3. Python caches -> left alone (regenerated, already gitignored)"

say
say "Done. Nothing was deleted; everything moved is under logs/, runs/_archive/"
say "or reports/_archive/. Reverse any step with a plain mv."
