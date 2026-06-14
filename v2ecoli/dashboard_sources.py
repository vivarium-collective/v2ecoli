"""Dashboard data-source provider: the ecoli-sources bundle, with links.

Surfaces every canonical "ecoli-source" v2ecoli draws on (the ParCa/reconstruction
input bundle) on the dashboard's **Sources** page, each with a hyperlink:

* inherited files  -> the pinned ``ecoli-sources`` GitHub commit
  (``raw.githubusercontent.com/...``)
* v2ecoli overrides (diverged locally under ``flat_overrides/``) -> a GitHub
  blob link into this repo.

Wired via ``workspace.yaml``::

    dashboard:
      data_sources:
        provider: "v2ecoli.dashboard_sources:enumerate_sources"
        label: "ecoli-sources"

The dashboard server calls ``enumerate_sources()`` (no args) and renders each
entry's ``url`` as a link — the only access path that works in the published
read-only snapshot. Link logic mirrors ``reports/ecoli_sources_report.py``.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_ECOLI_SOURCES_RAW_BASE = (
    "https://raw.githubusercontent.com/vivarium-collective/ecoli-sources"
)
_V2ECOLI_BLOB_BASE = "https://github.com/vivarium-collective/v2ecoli/blob/main"


def _ecoli_sources_rev() -> str | None:
    """Pinned ``rev`` SHA for ecoli-sources from ``[tool.uv.sources]``.

    Returns None if pinned to a branch (no ``rev``) — then inherited files get
    no link (raw URLs only resolve against an immutable commit).
    """
    try:
        try:
            import tomllib  # py3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        with open(_PYPROJECT, "rb") as f:
            data = tomllib.load(f)
        src = (data.get("tool", {}).get("uv", {})
               .get("sources", {}).get("ecoli-sources", {}))
        rev = src.get("rev") if isinstance(src, dict) else None
        return str(rev) if rev else None
    except Exception:
        return None


def _category(key: str) -> str:
    """Group key: the prefix before ``__`` (matches the standalone report)."""
    parts = key.split("__")
    return parts[0] if len(parts) > 1 else "root"


def enumerate_sources() -> list[dict]:
    """Return the ecoli-sources bundle as dashboard data-source entries.

    Each entry: ``{key, path, category, kind, size_bytes, url}``. Never raises
    (the dashboard wraps providers defensively, but we stay quiet regardless).
    """
    try:
        from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
        from ecoli_sources import DATA_DIR
    except Exception:
        return []

    rev = _ecoli_sources_rev()
    data_dir = Path(DATA_DIR).resolve()

    def _raw_url(p: Path) -> str:
        """Pinned GitHub raw URL for an inherited file (empty if unmappable)."""
        if not rev:
            return ""
        try:
            rel = p.resolve().relative_to(data_dir).as_posix()
        except Exception:
            return ""
        return f"{_ECOLI_SOURCES_RAW_BASE}/{rev}/ecoli_sources/data/{rel}"

    def _override_url(p: Path) -> str:
        """GitHub blob link into this repo for a local override (empty if unmappable)."""
        try:
            rel = p.resolve().relative_to(_REPO_ROOT).as_posix()
        except Exception:
            return ""
        return f"{_V2ECOLI_BLOB_BASE}/{rel}"

    try:
        index = SourceBundle()._index  # {canonical_key: Path}
    except Exception:
        return []

    entries: list[dict] = []
    for key, p in index.items():
        is_override = "flat_overrides" in str(p)
        try:
            size = p.stat().st_size if p.exists() else 0
        except Exception:
            size = 0
        entries.append({
            "key": key,
            "path": str(p),
            "category": _category(key),
            "kind": "override" if is_override else "inherited",
            "size_bytes": size,
            "url": _override_url(p) if is_override else _raw_url(p),
        })
    entries.sort(key=lambda e: (e["kind"] != "override", e["category"], e["key"]))
    return entries
