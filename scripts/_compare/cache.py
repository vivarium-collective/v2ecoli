"""Content-addressed caching for expensive engine runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(config: dict[str, Any], *, commit: str, mode: str) -> str:
    """Stable 16-hex-char digest of (config, engine commit, mode)."""
    payload = json.dumps(
        {"config": config, "commit": commit, "mode": mode},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def is_stale(run_dir: Path) -> bool:
    """A run dir is fresh only if it exists and holds a ``.done`` marker."""
    return not (Path(run_dir) / ".done").exists()


def mark_done(run_dir: Path) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / ".done").write_text("ok")
