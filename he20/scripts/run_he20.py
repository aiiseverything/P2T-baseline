"""Entry point for a he20 run: forwards to ``he20.trainer.main``.

Mirrors ``scripts/run_train.py`` for the sibling arm, which is a thin shim so the
package stays importable as a library.  Kept inside ``he20/`` rather than in the
shared ``scripts/`` directory so a he20 run never depends on a file the other arm
owns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from he20.trainer import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
