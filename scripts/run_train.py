#!/usr/bin/env python3
"""CLI entry for P2T baseline training.

    python scripts/run_train.py --config configs/smoke10.json

Prefer ``scripts/start_detached.sh`` on the server so the run survives the
terminal that started it.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p2t.trainer import main  # noqa: E402

if __name__ == "__main__":
    main()
