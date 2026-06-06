"""
Resolver for the ecoli-sources data bundle.

A *bundle* maps each ``canonical_key`` (an addressable data role in ParCa) to a
source file. The default reference bundle ships with ``ecoli-sources``
(``ecoli_sources.BUNDLE_PATH``). v2ecoli layers a small override manifest on
top so its locally-diverged flat files (equilibrium / metabolism biology) win
over the upstream defaults without copying the whole 135-key manifest.

Ported and adapted from CovertLab/vEcoli's ``wholecell/io/sources.py``
(PR #426); the override-merge is a v2ecoli addition.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, os.PathLike]

# Default location of the v2ecoli override spec (sibling of this module).
_DEFAULT_OVERRIDES = Path(__file__).resolve().parent / "parca_overrides.tsv"


def relpath_to_key(rel_path: str) -> str:
    """Map a flat-relative file path to its bundle canonical key.

    ``condition/media/MIX0-55.tsv`` -> ``condition__media__MIX0-55``. Strips a
    single trailing extension and replaces path separators with ``__``.
    """
    norm = rel_path.replace(os.sep, "/")
    stem, _, _ext = norm.rpartition(".")
    norm = stem if stem else norm  # no extension -> keep as-is
    return norm.replace("/", "__")


class SourceBundle:
    """Resolve canonical keys / flat-relpaths to absolute source paths."""

    def __init__(
        self,
        base_manifest: Optional[PathLike] = None,
        overrides: Optional[PathLike] = None,
        validate: bool = True,
    ):
        if base_manifest is None:
            from ecoli_sources import BUNDLE_PATH
            base_manifest = BUNDLE_PATH
        base_manifest = Path(base_manifest).resolve()
        if not base_manifest.is_file():
            raise FileNotFoundError(f"Bundle manifest not found: {base_manifest}")

        index: dict[str, Path] = {}
        base_root = base_manifest.parent
        index.update(self._read_manifest(base_manifest, base_root))

        if overrides is None and _DEFAULT_OVERRIDES.is_file():
            overrides = _DEFAULT_OVERRIDES
        if overrides is not None:
            overrides = Path(overrides).resolve()
            index.update(self._read_manifest(overrides, overrides.parent))

        self._index = index
        if validate:
            self._validate(base_manifest, overrides)

    @staticmethod
    def _read_manifest(manifest: Path, root: Path) -> dict[str, Path]:
        df = pd.read_csv(manifest, sep="\t", comment="#")
        out: dict[str, Path] = {}
        for _, row in df.iterrows():
            out[str(row["canonical_key"])] = (root / str(row["source_path"])).resolve()
        return out

    def _validate(self, base_manifest: Path, overrides: Optional[Path]) -> None:
        # Best-effort: reuse ecoli-sources' Pandera schema on the merged set;
        # always verify every resolved path exists.
        try:
            from schemas import ReferenceBundleSchema  # ecoli-sources package
            rows = [
                {"canonical_key": k, "source_path": str(p), "description": "", "schema_name": ""}
                for k, p in self._index.items()
            ]
            ReferenceBundleSchema.validate(pd.DataFrame(rows), lazy=True)
        except ImportError:
            pass
        missing = {k: p for k, p in self._index.items() if not p.is_file()}
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} bundle key(s) resolve to missing files: "
                f"{sorted(missing)[:5]}..."
            )

    def path(self, canonical_key: str) -> Path:
        try:
            return self._index[canonical_key]
        except KeyError:
            raise KeyError(f"canonical_key not in bundle: {canonical_key}")

    def resolve_relpath(self, rel_path: str) -> Path:
        return self.path(relpath_to_key(rel_path))

    def has_key(self, canonical_key: str) -> bool:
        return canonical_key in self._index

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [k for k in self._index if k.startswith(prefix)]
