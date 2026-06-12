"""Ensure the repo root is importable so `scripts._compare.*` resolves
during tests (this repo's tests/ dir is intentionally not a package)."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
