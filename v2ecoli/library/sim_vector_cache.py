"""Run-keyed cache for the extracted sim vectors.

``card_vectors.extract_vectors`` scans every history parquet in a sweep
(~8 min over the 1052 files of an 8x16 ensemble), so it cannot run per render.
This caches its result **keyed by the run that produced it**, which is what
makes the artifact self-describing: a new sweep yields a new key automatically,
and a cached vector always states which run, which burn-in, and which extractor
version it came from.

Key — ``(experiment_id, generation_lower_bound, extractor_version)``:

* ``experiment_id`` is the canonical run id (``docs/conventions/run-provenance.md``:
  experiment_id == run_id == the top parquet partition key).
* ``generation_lower_bound`` is part of the key because it changes the ensemble,
  not just a filter — the same sweep at gen_lb 3 vs 0 is a different set of
  cells and therefore a different vector. A key without it silently serves a
  stale vector after a burn-in change.
* ``extractor_version`` is part of the key because the vector is a function of
  the extraction code as much as of the run. Bump
  ``card_vectors.EXTRACTOR_VERSION`` whenever the aggregation semantics change.

Location — follows the convention PR #418 established for S3-resident sweeps:
results go to ``out_dir``, which defaults to the sweep dir when the sweep is
local and is **required** when it is an ``s3://`` URI (those are read-only).
The cache is therefore machine-local and invisible to CI **by design**; the
committed ``tests/fixtures/`` copies remain the run-free path that CI and the
report cards grade against.

Provenance — two distinct commits, never one ambiguous ``sim_commit``:

* ``run_commit`` — the code that produced the *sweep*, read from the sweep's
  ``run_identity.json`` sidecar (v2ecoli#472/#473,
  ``v2ecoli.library.run_provenance.write_run_identity``). Null for a sweep
  that predates that sidecar or genuinely never recorded one — honestly null
  rather than backfilled with something that looks authoritative.
* ``extracted_at_commit`` — the code that ran the *extraction*, which for a
  re-extracted older sweep is a different thing entirely.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from v2ecoli.library.card_vectors import EXTRACTOR_VERSION, extract_vectors
from v2ecoli.library.run_provenance import (
    RUN_IDENTITY_FILENAME,
    code_provenance,
    read_run_identity,
    run_id_from_run_dir,
)

CACHE_SCHEMA = "sim_vector_cache/v1"
_CACHE_SUBDIR = "sim_vectors"
_S3_PREFIX = "s3://"
_REPO = Path(__file__).resolve().parents[2]


def is_s3(sweep_dir: str) -> bool:
    return str(sweep_dir).startswith(_S3_PREFIX)


def cache_path(sweep_dir: str, generation_lower_bound: int = 0,
               out_dir: str | None = None) -> Path:
    """Where the cached vectors for this (run, gen_lb, extractor) live.

    ``out_dir`` defaults to the sweep dir for a local sweep; it is required for
    an ``s3://`` sweep, which is read-only (PR #418's convention)."""
    if out_dir is None:
        if is_s3(sweep_dir):
            raise ValueError(
                "out_dir is required for an s3:// sweep — the sweep is read-only, "
                "so the cache cannot be written beside it (PR #418 convention)")
        out_dir = sweep_dir
    run_id = run_id_from_run_dir(str(sweep_dir))
    name = f"{run_id}__gen{int(generation_lower_bound)}__v{EXTRACTOR_VERSION}.json"
    return Path(out_dir) / _CACHE_SUBDIR / name


def _read_run_identity_s3(sweep_dir: str) -> dict | None:
    """``run_identity.json`` from an ``s3://`` sweep, via DuckDB's ``read_text``
    over the same httpfs credential path :mod:`sweep_io` already establishes
    for parquet reads — no new dependency, no S3 SDK call duplicated here."""
    from v2ecoli.library.sweep_io import connect_for

    uri = sweep_dir.rstrip("/") + "/" + RUN_IDENTITY_FILENAME
    conn = connect_for(sweep_dir)
    try:
        rows = conn.sql(f"SELECT content FROM read_text('{uri}')").fetchall()
    except Exception:
        return None
    finally:
        conn.close()
    if not rows:
        return None
    try:
        blob = json.loads(rows[0][0])
    except Exception:
        return None
    return blob if isinstance(blob, dict) else None


def _run_commit(sweep_dir: str) -> str | None:
    """The commit that produced the sweep, if the run recorded one.

    Reads the canonical ``run_identity.json`` sidecar (v2ecoli#472/#473) —
    local or ``s3://``, an S3 sweep gets one automatically once it's produced
    by a runner on this side of #472, since the Ray/Batch teardown syncs the
    whole output dir (docker/ray-batch-entrypoint.sh). Falls back to the
    legacy flat-key scan of ``run_config.json`` / ``summary.json`` for
    sweeps produced before this brief, none of which ever populated those
    keys in practice but the shape was speculatively read for — kept in
    case an external tool already emits it. Returns None, never the
    extracting tree's HEAD, when nothing was recorded — substituting would
    assert something false.
    """
    if is_s3(sweep_dir):
        blob = _read_run_identity_s3(sweep_dir)
    else:
        blob = read_run_identity(sweep_dir)
    if isinstance(blob, dict):
        commit = blob.get("code", {}).get("commit")
        if commit:
            return str(commit)
    if is_s3(sweep_dir):
        return None
    for name in ("run_config.json", "summary.json"):
        p = Path(sweep_dir) / name
        if not p.is_file():
            continue
        try:
            legacy = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(legacy, dict):
            for key in ("commit", "git_commit", "code_commit"):
                if legacy.get(key):
                    return str(legacy[key])
    return None


def load_or_extract(sweep_dir: str, generation_lower_bound: int = 0,
                    out_dir: str | None = None, refresh: bool = False) -> dict:
    """Return the cache envelope for this run, extracting only on a miss.

    The envelope is ``{schema, run, extractor, provenance, vectors}``; the
    vectors are ``extract_vectors``' output verbatim, so a caller can swap a
    direct call for a cached one without reshaping anything.
    """
    path = cache_path(sweep_dir, generation_lower_bound, out_dir)
    if path.is_file() and not refresh:
        cached = json.load(open(path, encoding="utf-8"))
        if cached.get("schema") == CACHE_SCHEMA:
            return cached

    t0 = time.time()
    vectors = extract_vectors(str(sweep_dir), generation_lower_bound)
    elapsed = time.time() - t0

    n_cells = {f"{g}.{n}": node.get("n_cells")
               for g, group in vectors.items() for n, node in group.items()}
    envelope = {
        "schema": CACHE_SCHEMA,
        "run": {
            "experiment_id": run_id_from_run_dir(str(sweep_dir)),
            "sweep_dir": str(sweep_dir),
            "generation_lower_bound": int(generation_lower_bound),
            "n_cells": n_cells,
        },
        "extractor": {
            "module": "v2ecoli.library.card_vectors",
            "function": "extract_vectors",
            "version": EXTRACTOR_VERSION,
            "extract_seconds": round(elapsed, 3),
        },
        "provenance": {
            # The sweep's own commit, when it recorded one — see _run_commit.
            "run_commit": _run_commit(str(sweep_dir)),
            # The tree that ran the extraction. A different thing entirely for a
            # re-extracted older sweep, which is why these are two fields.
            "extracted_at_commit": code_provenance(_REPO),
            "extracted": time.strftime("%Y-%m-%d"),
        },
        "vectors": vectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh)
    os.replace(tmp, path)          # atomic: a killed run leaves no half-file
    return envelope


def cached_vectors(sweep_dir: str, generation_lower_bound: int = 0,
                   out_dir: str | None = None, refresh: bool = False) -> dict:
    """``extract_vectors``' output, from cache when available.

    Drop-in for ``extract_vectors(sweep_dir, gen_lb)``."""
    return load_or_extract(sweep_dir, generation_lower_bound, out_dir,
                           refresh)["vectors"]
