"""
Convenience entry point at repo root — delegates to `src.train`.

Usage (from project root):
    python train.py
    python train.py --epochs 8 --batch-size 8
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
