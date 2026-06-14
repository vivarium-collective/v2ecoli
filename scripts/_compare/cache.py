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


def is_stale(run_dir: Path, token: str | None = None) -> bool:
    """A run dir is fresh only if it exists and holds a ``.done`` marker.

    If ``token`` is provided, the run is also considered stale when the stored
    token does not match — so mode/config changes invalidate the cache.
    """
    marker = Path(run_dir) / ".done"
    if not marker.exists():
        return True
    if token is not None and marker.read_text().strip() != token:
        return True
    return False


def mark_done(run_dir: Path, token: str = "ok") -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / ".done").write_text(token)
