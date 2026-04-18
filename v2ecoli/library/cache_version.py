"""Cache version fingerprinting.

The cache at ``out/cache/`` (``sim_data_cache.dill`` + ``initial_state.json``)
is derived from (a) the shipped ParCa fixture at ``models/parca/parca_state.pkl.gz``
and (b) the code in ``v2ecoli.library.sim_data`` and its pint-boundary
helpers.  When either side changes incompatibly — e.g. the unum→pint migration
in #18 — a cache built from the previous code drops through sim simulation
steps with obscure ``AttributeError: 'Unum' object has no attribute 'to'``
tracebacks several frames deep.

This module computes a content hash over the inputs that determine cache
compatibility, writes it into ``cache_version.json`` at build time, and
verifies it at load time.  On mismatch, ``verify_cache_version`` raises
``StaleCacheError`` with a one-line rebuild instruction — no detective work
required.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable


SCHEMA_VERSION = "1"
CACHE_VERSION_FILENAME = "cache_version.json"

# Files whose content determines whether an existing cache is compatible with
# the current code.  Hash is computed over the *sorted* concatenation of
# ``path\n<sha256 of file>`` lines so reordering or renaming any file is
# detected.
#
# The ParCa fixture is hashed because it *is* the cache's biological content.
# The sim_data / unit-bridge modules are hashed because they shape how that
# content is projected into configs (the unum→pint migration boundary).
INPUT_FILES: tuple[str, ...] = (
    # Biological content.
    "models/parca/parca_state.pkl.gz",
    # LoadSimData: turns sim_data into process configs.
    "v2ecoli/library/sim_data.py",
    # Unum↔pint migration boundary — regressions here are the whole reason
    # this module exists.
    "v2ecoli/library/unit_bridge.py",
    # Custom pint UnitRegistry with nucleotide/amino_acid/count; also
    # defines the Quantity schema type. A registry change can silently
    # change how Quantity fields round-trip through dill.
    "v2ecoli/types/quantity.py",
    # Seeds bulk/unique molecules into initial_state.json.
    "v2ecoli/library/initial_conditions.py",
    # save_cache writes both files.
    "v2ecoli/composite.py",
)


class StaleCacheError(RuntimeError):
    """Raised when cache_version.json does not match the current code/fixture.

    The message includes the rebuild command so humans and CI logs both get
    an actionable next step without reading this module.
    """


@dataclass(frozen=True)
class CacheVersion:
    schema_version: str
    inputs_hash: str
    per_file_hashes: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "inputs_hash": self.inputs_hash,
            "per_file_hashes": dict(self.per_file_hashes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheVersion":
        return cls(
            schema_version=d.get("schema_version", ""),
            inputs_hash=d.get("inputs_hash", ""),
            per_file_hashes=dict(d.get("per_file_hashes", {})),
        )


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cache_version(repo_root: str = ".",
                          files: Iterable[str] = INPUT_FILES) -> CacheVersion:
    """Compute the fingerprint over all INPUT_FILES under ``repo_root``."""
    per_file: dict[str, str] = {}
    for rel in sorted(files):
        path = os.path.join(repo_root, rel)
        if not os.path.exists(path):
            # Missing input is itself part of the fingerprint — encode as
            # a sentinel so a file appearing/disappearing changes the hash.
            per_file[rel] = "MISSING"
            continue
        per_file[rel] = _hash_file(path)

    agg = hashlib.sha256()
    for rel in sorted(per_file):
        agg.update(f"{rel}\n{per_file[rel]}\n".encode())
    return CacheVersion(
        schema_version=SCHEMA_VERSION,
        inputs_hash=agg.hexdigest(),
        per_file_hashes=per_file,
    )


def write_cache_version(cache_dir: str, version: CacheVersion | None = None,
                        repo_root: str = ".") -> CacheVersion:
    """Write cache_version.json inside ``cache_dir``.  Called by save_cache."""
    if version is None:
        version = compute_cache_version(repo_root=repo_root)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CACHE_VERSION_FILENAME)
    with open(path, "w") as f:
        json.dump(version.to_dict(), f, indent=2, sort_keys=True)
    return version


def read_cache_version(cache_dir: str) -> CacheVersion | None:
    """Return the cached version, or ``None`` if not present."""
    path = os.path.join(cache_dir, CACHE_VERSION_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return CacheVersion.from_dict(json.load(f))


def verify_cache_version(cache_dir: str, repo_root: str = ".") -> None:
    """Raise StaleCacheError if the cache on disk doesn't match current inputs.

    Called from the cache load path.  A missing ``cache_version.json`` is a
    hard error too — we can't prove a pre-versioning cache is safe, so treat
    it the same as a mismatch.
    """
    current = compute_cache_version(repo_root=repo_root)
    stored = read_cache_version(cache_dir)

    if stored is None:
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"{cache_dir}/{CACHE_VERSION_FILENAME} missing "
                   f"(cache was built before versioning was introduced, "
                   f"or was partially written)",
            expected=current,
            actual=None,
        ))

    if stored.schema_version != current.schema_version:
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"schema_version mismatch "
                   f"(stored={stored.schema_version!r}, "
                   f"current={current.schema_version!r})",
            expected=current,
            actual=stored,
        ))

    if stored.inputs_hash != current.inputs_hash:
        changed = [
            rel for rel in current.per_file_hashes
            if current.per_file_hashes.get(rel)
               != stored.per_file_hashes.get(rel)
        ]
        raise StaleCacheError(_rebuild_message(
            cache_dir,
            reason=f"inputs_hash mismatch; files differ: {changed}",
            expected=current,
            actual=stored,
        ))


def _rebuild_message(cache_dir: str, reason: str,
                     expected: CacheVersion,
                     actual: CacheVersion | None) -> str:
    lines = [
        f"Cache at {cache_dir!r} is stale or unversioned: {reason}.",
        "",
        "Rebuild it:",
        "    python scripts/build_cache.py",
        "",
        f"Expected inputs_hash: {expected.inputs_hash[:16]}...",
    ]
    if actual is not None:
        lines.append(f"Actual   inputs_hash: {actual.inputs_hash[:16]}...")
    return "\n".join(lines)
