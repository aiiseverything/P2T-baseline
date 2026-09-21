"""Entry point for a RED run: forwards to ``red.trainer.main``.

Mirrors ``scripts/run_train.py`` for the sibling arm, which is a thin shim so the
package stays importable as a library.  Kept inside ``red/`` rather than in the
shared ``scripts/`` directory so a RED run never depends on a file the other arm
owns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from red.trainer import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
