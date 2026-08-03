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
    """Read ``runs_meta.emitter_path`` for ``run_id`` (tried first) or,
    failing that, ``sim_name`` (tried second) out of a study's runs.db.

    A bare-string ``run_ref`` reaching this function may be a real run_id OR
    a **sim_name** — Phase B's native ``comparison_cards`` passes
    ``candidate_run``/``reference_run`` as sim_names (``<config>`` for the
    baseline/candidate, ``"reference"`` for the variant; the substrate
    writes ``runs_meta.sim_name`` = the baseline entry's name for a baseline
    run, and the variant's name for a variant run — see
    ``vivarium_workbench.lib.run_registry``). run_id is tried first because
    it's the primary key (unambiguous); sim_name is a non-unique label, so
    if MULTIPLE rows share it, the most recently started row wins (``ORDER
    BY started_at DESC, rowid DESC`` — ``started_at`` is the semantic
    ordering, ``rowid`` breaks exact ties deterministically rather than
    picking arbitrarily).
    """
    runs_db = Path(runs_db)
    if not runs_db.is_file():
        raise RunStoreError(f"runs.db not found: {runs_db}")
    conn = sqlite3.connect(f"file:{runs_db}?mode=ro", uri=True, timeout=1.0)
    try:
        row = conn.execute(
            "SELECT emitter_path FROM runs_meta WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row or not row[0]:
            row = conn.execute(
                "SELECT emitter_path FROM runs_meta WHERE sim_name=?"
                " ORDER BY started_at DESC, rowid DESC LIMIT 1", (run_id,)
            ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise RunStoreError(
            f"no emitter_path recorded for run_id or sim_name={run_id!r} "
            f"in {runs_db}")
    return row[0]


def resolve_run_store(run_ref, *, study_dir=None, runs_db=None) -> Path:
    """Resolve a workbench run reference to its stored ``.zarr`` path.

    ``run_ref`` may be:
      - a path (str/Path) directly to a ``.zarr`` store or a run directory
        containing one (see ``_resolve_zarr_store``);
      - a bare run id OR sim_name (str), resolved via ``runs_db`` (or
        ``<study_dir>/runs.db``) — the study's ``runs_meta.emitter_path``,
        joined onto ``study_dir`` if relative. run_id is tried first (the
        primary key); sim_name is tried second (see
        ``_lookup_emitter_path``) — Phase B's native ``comparison_cards``
        passes ``candidate_run``/``reference_run`` as sim_names, not run_ids;
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


def _run_dir_candidate(run_ref, *, study_dir=None) -> "Path | None":
    """Best-effort filesystem directory for ``run_ref``, for the PARQUET
    fallback below — used only when zarr resolution fails, so a missing/odd
    ``run_ref`` shape here just means "no parquet fallback available" rather
    than a hard error (the original zarr ``RunStoreError`` still surfaces).
    """
    if isinstance(run_ref, Mapping):
        p = run_ref.get("path")
        if p is not None:
            return Path(p)
        return None
    candidate = Path(run_ref)
    if candidate.exists():
        return candidate
    if study_dir is not None:
        joined = Path(study_dir) / str(run_ref)
        if joined.exists():
            return joined
    return None


def _resolve_parquet_history(path: "str | Path") -> "list[Path] | None":
    """Hive-partitioned ``*.pq`` history files under a run directory, or
    ``None`` if none are found.

    Matches the local layout ``vivarium_workbench.lib.emitters``' parquet
    branch actually writes for a ``run composite``/``run study`` whose
    composite DECLARES a default emitter (e.g. ``ecoli_baseline``'s
    ``@composite_generator(..., emitters=[...])`` — see
    ``vivarium_workbench.lib.run_runner._select_emitter_name``, which forces
    ``"parquet"`` for any composite with a declared default, REGARDLESS of
    the workspace's own ``runtime.default_emitter`` (``xarray`` here) — so a
    candidate ``ecoli_baseline`` run through the general runner lands as
    parquet, not zarr, even though the reference ``vecoli`` composite (no
    declared default) lands as zarr under the same workspace. Confirmed
    empirically (Phase-2 Task 5 gate): ``<run_dir>/parquet/<run_id>/history/
    experiment_id=<run_id>/variant=*/lineage_seed=*/generation=*/agent_id=*/
    <tick>.pq``.
    """
    p = Path(path)
    if not p.is_dir():
        return None
    files = sorted(p.glob("parquet/**/history/**/*.pq"))
    if not files:
        files = sorted(p.glob("**/history/**/*.pq"))
    return files or None


def _load_local_parquet_observables(files: "list[Path]", observables) -> dict:
    """``{obs: (times, values), ..., "_generation": (times, gens)}`` off a
    LOCAL hive-partitioned parquet history — the same shape
    ``read_pbg_local`` returns for zarr, so callers don't need to branch.

    Reuses ``scripts._compare.vecoli_parquet_reader``'s dotted/leaf ->
    flattened-column mapping (``_observable_to_column``, e.g. ``"cell_mass"``
    -> ``"listeners__mass__cell_mass"``) READ-ONLY — the same convention this
    repo's own local ``ParquetEmitter`` output uses (confirmed empirically:
    both flatten dotted paths to ``__``). DuckDB reads local files directly
    (no S3/AWS creds needed, unlike ``vecoli_parquet_reader``'s S3 callers);
    the local emitter's time column is ``global_time`` (S3 vEcoli's is
    ``time``), so this queries ``global_time`` explicitly rather than
    reusing ``read_vecoli_trajectory``'s SQL verbatim.
    """
    import duckdb
    import numpy as np

    from scripts._compare.vecoli_parquet_reader import _observable_to_column

    obs = list(observables)
    uris = [str(f) for f in files]
    con = duckdb.connect()
    present_cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{uris[0]}')").fetchall()}
    obs_to_col = {}
    for o in obs:
        col = _observable_to_column(o)
        if col in present_cols:
            obs_to_col[o] = col
    if not obs_to_col:
        raise RunStoreError(
            f"none of {obs!r} found as columns in local parquet history "
            f"({len(present_cols)} columns present, e.g. {sorted(present_cols)[:5]})")

    file_list = ", ".join(f"'{u}'" for u in uris)
    sel = ", ".join(f'"{c}" AS "{o}"' for o, c in obs_to_col.items())
    query = (f"SELECT global_time, generation, {sel} "
             f"FROM read_parquet([{file_list}], hive_partitioning=true) "
             f"ORDER BY global_time")
    cur = con.execute(query)
    col_names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    arr = {c: np.array([r[i] for r in rows], dtype=float)
           for i, c in enumerate(col_names)}
    out: dict = {}
    t = arr["global_time"]
    for o in obs_to_col:
        out[o] = (t, arr[o])
    out["_generation"] = (t, arr["generation"])
    return out


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

    **Parquet fallback (Phase-2 Task 5 gate finding):** a composite whose
    generator DECLARES a default emitter (``ecoli_baseline``) lands as a
    local hive-partitioned parquet history under the general runner, not
    zarr — confirmed against a real ``vivarium-workbench run composite``
    run, not assumed. When zarr resolution fails, this falls back to
    ``_resolve_parquet_history``/``_load_local_parquet_observables`` before
    raising; a run_ref that resolves to neither zarr nor parquet still
    raises the original ``RunStoreError``.
    """
    from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

    obs = list(observables) if observables is not None else list(OBSERVABLES)
    try:
        store = resolve_run_store(run_ref, study_dir=study_dir, runs_db=runs_db)
    except RunStoreError as zarr_err:
        run_dir = _run_dir_candidate(run_ref, study_dir=study_dir)
        files = _resolve_parquet_history(run_dir) if run_dir is not None else None
        if not files:
            raise
        try:
            return _load_local_parquet_observables(files, obs)
        except RunStoreError:
            raise
        except Exception as parquet_err:  # noqa: BLE001 — report both attempts
            raise RunStoreError(
                f"neither zarr ({zarr_err}) nor local parquet ({parquet_err}) "
                f"could be read for run_ref={run_ref!r}") from parquet_err
    return read_pbg_local(str(store), obs)
