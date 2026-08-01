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
import sys
from dataclasses import dataclass, field
from typing import Iterable


SCHEMA_VERSION = "2"
CACHE_VERSION_FILENAME = "cache_version.json"

# Packages on the ParCa fit path whose version genuinely changes fit output
# (or how a previously-fit cache unpickles) — see the module docstring at
# ":238-266" / PARCA_REVIEW.md A9. Recorded into ``CacheVersion.context`` and
# folded into ``inputs_hash`` so a cache built under one scipy/numba/etc and
# loaded under another is detectable instead of silently mis-unpickling deep
# in a simulation step.
CONTEXT_PACKAGES: tuple[str, ...] = (
    "scipy", "numpy", "numba", "dill", "cvxpy", "ecos", "stochastic-arrow",
)

#: build_params keys that describe *which artifact* a bundle is (not the code
#: that produced it) — condition/media/seed/n_seeds/patch identity. Folding
#: these into inputs_hash is what makes ``out/cache`` (basal) and
#: ``out/cache-stage1-heuristic`` (dnaA-patched) produce distinguishable
#: ``cache_version.json`` files instead of byte-identical ones (PARCA_REVIEW
#: A7). ``None`` for every key is the basal/default build.
DEFAULT_BUILD_PARAMS: dict = {
    "condition": None,
    "fixed_media": None,
    "seed": None,
    "n_seeds": None,
    "condition_manifest_hash": None,
}


def _module_version(name: str) -> str:
    """Installed version of ``name``, or the sentinel ``"absent"``.

    Never raises: an uninstalled/unimportable optional dep (or a package
    metadata lookup failing for any other reason) must not crash cache-version
    computation — it should just show up plainly in the recorded context.
    """
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        return "absent"


def probe_context() -> dict:
    """Snapshot the runtime-environment versions that can silently change a fit.

    Mirrors the pattern in ``v2ecoli/comparison/vecoli_parca.py``'s
    ``VEcoliParcaBuild._ref`` (see its module docstring at ``:12-19`` for the
    failure this guards against — a cache built under one scipy silently
    mis-unpickling under another). Called fresh on every
    ``compute_cache_version()`` so build-time and verify-time context reflect
    whatever environment is *actually running*, not an echoed value — that is
    what lets a scipy/numpy/etc upgrade between build and load move
    ``inputs_hash``.
    """
    ctx = {
        "python": "%d.%d.%d" % sys.version_info[:3],
    }
    for pkg in CONTEXT_PACKAGES:
        ctx[pkg] = _module_version(pkg)
    return ctx

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
    # This module itself. Self-referential on purpose: editing INPUT_FILES
    # (or the hashing/raise logic below) changes this file's own bytes, so
    # future additions here automatically bust every existing cache without
    # requiring a separate SCHEMA_VERSION bump.
    "v2ecoli/library/cache_version.py",
    # save_cache + shared composite infrastructure.
    "v2ecoli/core.py",
    # Per-architecture document builders. A change here can shift the
    # document shape and silently invalidate a cache built against the
    # old architecture. Renamed to the ecoli_* scheme in 645fe178; keep
    # this list in sync per AGENTS.md's "Adding a new composite
    # architecture" step 3.
    "v2ecoli/composites/ecoli_baseline.py",
    "v2ecoli/composites/ecoli_population.py",
    "v2ecoli/composites/ecoli_time_varying_env.py",
    "v2ecoli/composites/ecoli_colony.py",
    "v2ecoli/composites/ecoli_millard.py",
    # Shared builders imported by the composites above (make_edge,
    # _make_instance, _get_special_step, per-step config dispatch, ...).
    # A change here shifts document shape for every composite that imports
    # it, exactly like a change to the composite file itself.
    "v2ecoli/composites/_helpers.py",
    "v2ecoli/composites/_millard_helpers.py",
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
    # Runtime-environment package versions (A9) and per-build parameters
    # (A7) — both folded into inputs_hash. default_factory=dict keeps old
    # callers (e.g. tests constructing CacheVersion directly) working without
    # passing these.
    context: dict = field(default_factory=dict)
    build_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "inputs_hash": self.inputs_hash,
            "per_file_hashes": dict(self.per_file_hashes),
            "context": dict(self.context),
            "build_params": dict(self.build_params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheVersion":
        return cls(
            schema_version=d.get("schema_version", ""),
            inputs_hash=d.get("inputs_hash", ""),
            per_file_hashes=dict(d.get("per_file_hashes", {})),
            context=dict(d.get("context", {})),
            build_params=dict(d.get("build_params", {})),
        )


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_repo_root() -> str:
    """Repo root resolved from THIS file's location, not the cwd.

    Bundles are generated inside a chdir'd isolation dir (run_comparison_ensemble
    os.chdir's into ``.regen_*`` so the default emitter's relative side-writes
    don't collide across parallel seeds). With ``repo_root="."`` every INPUT_FILE
    then resolves under that throwaway dir, hashes as MISSING, and the fingerprint
    collapses to a constant that never changes when the source changes — so the
    whole staleness check silently no-ops. Anchor to the package instead:
    this file is ``<repo>/v2ecoli/library/cache_version.py``.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_cache_version(repo_root: str | None = None,
                          files: Iterable[str] = INPUT_FILES,
                          build_params: dict | None = None,
                          context: dict | None = None) -> CacheVersion:
    """Compute the fingerprint over INPUT_FILES + context + build_params.

    ``context`` defaults to a fresh live probe (see ``probe_context``) so two
    calls in different environments naturally disagree — that is what makes a
    scipy/numpy/etc upgrade between build and load move ``inputs_hash``
    (A9). ``build_params`` defaults to :data:`DEFAULT_BUILD_PARAMS` (a plain
    basal build with no condition/seed/patch) and is otherwise supplied by
    the caller that actually knows what it built (``core.py``'s
    ``save_sim_input`` / ``save_cache``) — unlike ``context`` there is no
    environment to "probe" for build params, they are inherent to the
    artifact (A7).
    """
    if repo_root is None:
        repo_root = _default_repo_root()
    per_file: dict[str, str] = {}
    for rel in sorted(files):
        path = os.path.join(repo_root, rel)
        if not os.path.exists(path):
            # A vanished fingerprint input is a bug, not a state: hashing it
            # to a stable "MISSING" sentinel silently drops the file from
            # the fingerprint forever (its edits stop moving inputs_hash).
            # That is exactly how 5/11 INPUT_FILES went dead unnoticed after
            # the ecoli_* composite rename in 645fe178. Fail loudly instead.
            raise FileNotFoundError(
                f"cache_version INPUT_FILES entry does not exist: {path!r} "
                f"(from repo_root={repo_root!r}, rel={rel!r}). This file was "
                f"renamed or deleted without updating "
                f"v2ecoli/library/cache_version.py:INPUT_FILES — see "
                f"AGENTS.md 'Adding a new composite architecture' step 3."
            )
        per_file[rel] = _hash_file(path)

    if context is None:
        context = probe_context()
    resolved_build_params = dict(DEFAULT_BUILD_PARAMS)
    if build_params:
        resolved_build_params.update(
            {k: v for k, v in build_params.items() if k in resolved_build_params})

    agg = hashlib.sha256()
    for rel in sorted(per_file):
        agg.update(f"{rel}\n{per_file[rel]}\n".encode())
    agg.update(b"\ncontext\n")
    agg.update(json.dumps(context, sort_keys=True).encode())
    agg.update(b"\nbuild_params\n")
    agg.update(json.dumps(resolved_build_params, sort_keys=True).encode())
    return CacheVersion(
        schema_version=SCHEMA_VERSION,
        inputs_hash=agg.hexdigest(),
        per_file_hashes=per_file,
        context=dict(context),
        build_params=resolved_build_params,
    )


def write_cache_version(cache_dir: str, version: CacheVersion | None = None,
                        repo_root: str | None = None,
                        build_params: dict | None = None) -> CacheVersion:
    """Write cache_version.json inside ``cache_dir``.  Called by save_cache."""
    if version is None:
        version = compute_cache_version(repo_root=repo_root, build_params=build_params)
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


def verify_cache_version(cache_dir: str, repo_root: str | None = None) -> None:
    """Raise StaleCacheError if the cache on disk doesn't match current inputs.

    Called from the cache load path.  A missing ``cache_version.json`` is a
    hard error too — we can't prove a pre-versioning cache is safe, so treat
    it the same as a mismatch.

    ``build_params`` (A7) describes *which artifact* the cache is (condition,
    seed, n_seeds, ...) — it is a property of the bundle, not something
    "current code" can independently re-derive, so recomputing "current"
    echoes ``stored.build_params`` back rather than defaulting them away.
    That keeps a real non-basal bundle (e.g. built with a non-default seed)
    from failing verification against itself; the value of folding
    build_params into inputs_hash is that two *different* bundles now hash
    differently (inspectable via a plain diff of their cache_version.json),
    not that this function detects a mismatched --cache-dir on its own.
    ``context`` (A9) is the opposite: it is re-probed fresh here so an
    environment change between build and load is exactly what this catches.
    """
    stored = read_cache_version(cache_dir)
    current = compute_cache_version(
        repo_root=repo_root,
        build_params=(stored.build_params if stored is not None else None),
    )

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
