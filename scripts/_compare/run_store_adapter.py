"""Translate a `vivarium-workbench` run reference into the observables shape
the existing comparison report cards consume — a THIN adapter, no new science.

Background (Phase 1 Task 3, see
docs/superpowers/plans/2026-08-01-comparison-convergence-phase-1.md): the
comparison harness (``scripts/_compare/report_cards``) grew up reading two
kinds of engine output directly — S3 vEcoli parquet
(``vecoli_parquet_reader.read_vecoli_finals``/``read_vecoli_trajectory``) and
local "v2ecoli-format" zarr (``scripts.compare_matched_trajectories.
read_pbg_local``, used by the trajectory/distribution/metabolism/composition
cards via ``state["v2_dir"]``/``state["ve_dir"]``). The general
``vivarium-workbench`` runner stores a composite run's emitted output under
``<study_dir>/<emitter_subdir>/<run_id>/`` and records that path (relative to
the study dir) as ``emitter_path`` in the study's ``runs.db``
(``runs_meta`` table — see vendored ``vivarium_workbench.lib.run_registry.
RUNS_META_DDL`` / ``vivarium_workbench.lib.backfill_runs.backfill_study_runs``
in the ``vivarium-workbench`` package).

When a run uses the ``xarray`` emitter (the workspace default as of
pbg-emitters Task 6 — see ``vivarium_workbench.lib.emitters.DEFAULT_EMITTER``),
that stored output IS a zarr store in the same "v2ecoli-format" layout
``read_pbg_local`` already reads: a ``lineage_seed=<N>`` DataTree group whose
direct children are the observable leaves (``<leaf>/generation=<G>`` data
vars), verified structurally by
``vivarium_workbench.lib.comparative_viz._extract_trace_from_zarr`` and by
this repo's own ``tests/test_xarray_colony_step.py`` /
``tests/fixtures/redux_cards``. So a workbench run's zarr needs no new
reader — only *locating* it (``<study_dir>/<emitter_path>``, or a bare
``.zarr`` path/dir) and handing it to the existing reader.

ASSUMPTION (documented per Task 3's brief): the concrete on-disk store form
targeted here is "a directory that is itself a ``.zarr`` store, or a run
directory containing one" — either directly (``<emitter_path>`` ends in
``.zarr``, the v2ecoli/vEcoli naming convention, e.g.
``v2ecoli_seed00.zarr``) or one level down (``store.zarr``, the vwb
XArrayEmitter default name — see ``vivarium_workbench.lib.explorer_data.
_resolve_run_source``, whose candidate list this mirrors). If a future
workbench run-store layout diverges, only ``_resolve_zarr_store`` here needs
updating — ``load_run_observables``'s public contract is unaffected.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping


class RunStoreError(LookupError):
    """A workbench run reference could not be resolved to a stored zarr path."""


def _resolve_zarr_store(path: "str | Path") -> Path:
    """Resolve a run-store path to the concrete ``.zarr`` directory within it.

    Accepts (in order):
      1. a path that already IS a ``.zarr`` store (v2ecoli/vEcoli naming,
         e.g. ``.../v2ecoli_seed00.zarr``);
      2. a run directory containing ``store.zarr`` directly or one/two
         levels down (the vwb XArrayEmitter default name — mirrors
         ``vivarium_workbench.lib.explorer_data._resolve_run_source``'s
         candidate list);
      3. a run directory containing any other ``*.zarr`` child (the
         v2ecoli/vEcoli naming convention landed inside a run dir).
    """
    p = Path(path)
    if str(p).endswith(".zarr"):
        if p.exists():
            return p
        raise RunStoreError(f"zarr store does not exist: {p}")
    if not p.is_dir():
        raise RunStoreError(f"run store path not found: {p}")
    for cand in (p / "store.zarr", *sorted(p.glob("*/store.zarr")),
                 *sorted(p.glob("*/*/store.zarr"))):
        if cand.exists():
            return cand
    zarrs = sorted(p.glob("*.zarr"))
    if zarrs:
        return zarrs[0]
    raise RunStoreError(f"no .zarr store found under {p}")


def _lookup_emitter_path(runs_db: "str | Path", run_id: str) -> str:
    """Read ``runs_meta.emitter_path`` for ``run_id`` out of a study's runs.db."""
    runs_db = Path(runs_db)
    if not runs_db.is_file():
        raise RunStoreError(f"runs.db not found: {runs_db}")
    conn = sqlite3.connect(f"file:{runs_db}?mode=ro", uri=True, timeout=1.0)
    try:
        row = conn.execute(
            "SELECT emitter_path FROM runs_meta WHERE run_id=?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise RunStoreError(
            f"no emitter_path recorded for run_id={run_id!r} in {runs_db}")
    return row[0]


def resolve_run_store(run_ref, *, study_dir=None, runs_db=None) -> Path:
    """Resolve a workbench run reference to its stored ``.zarr`` path.

    ``run_ref`` may be:
      - a path (str/Path) directly to a ``.zarr`` store or a run directory
        containing one (see ``_resolve_zarr_store``);
      - a bare run id (str), resolved via ``runs_db`` (or
        ``<study_dir>/runs.db``) — the study's ``runs_meta.emitter_path``,
        joined onto ``study_dir`` if relative;
      - a mapping with any of ``{"path", "run_id", "study_dir", "runs_db"}``
        (keys mirror the keyword args; ``study_dir``/``runs_db`` keyword
        args are used as fallbacks when absent from the mapping).
    """
    if isinstance(run_ref, Mapping):
        path = run_ref.get("path")
        if path is not None:
            return _resolve_zarr_store(path)
        run_id = run_ref.get("run_id")
        study_dir = run_ref.get("study_dir", study_dir)
        runs_db = run_ref.get("runs_db", runs_db)
    else:
        candidate = Path(run_ref)
        if candidate.exists():
            return _resolve_zarr_store(candidate)
        run_id = str(run_ref)

    if not run_id:
        raise RunStoreError(f"unresolvable run reference: {run_ref!r}")
    if runs_db is None:
        if study_dir is None:
            raise RunStoreError(
                f"run_id={run_id!r} given without study_dir/runs_db to look it up in")
        runs_db = Path(study_dir) / "runs.db"
    emitter_path = _lookup_emitter_path(runs_db, run_id)
    p = Path(emitter_path)
    if not p.is_absolute() and study_dir is not None:
        p = Path(study_dir) / p
    return _resolve_zarr_store(p)


def load_run_observables(run_ref, observables=None, *, study_dir=None,
                         runs_db=None) -> dict:
    """One workbench run's observables, in the shape the trajectory/
    distribution/metabolism/composition report cards consume off
    ``state["v2_dir"]``/``state["ve_dir"]`` — ``{obs: (times, values), ...,
    "_generation": (times, gen_numbers)}`` — via
    ``scripts.compare_matched_trajectories.read_pbg_local`` (the reader those
    cards already use for local "v2ecoli-format" zarr; both v2ecoli and
    vEcoli-pbg emit this same format, so this reader serves either side of a
    comparison unmodified).

    ``observables`` defaults to
    ``scripts.compare_matched_trajectories.OBSERVABLES`` (the standard
    comparison set) when not given.
    """
    from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

    store = resolve_run_store(run_ref, study_dir=study_dir, runs_db=runs_db)
    obs = list(observables) if observables is not None else list(OBSERVABLES)
    return read_pbg_local(str(store), obs)
