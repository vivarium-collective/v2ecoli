"""Shared helpers for the v2ecoli composite generators.

These were previously defined in ``v2ecoli/generate.py`` and re-imported by
``generate_departitioned.py`` and ``generate_reconciled.py``.  Task 14 moves
them here so the legacy generate*.py files can be deleted.

Exported names (all are considered semi-private implementation details):
  - make_edge
  - inject_flow_dependencies
  - _seed_state_from_defaults
  - seed_mass_listener
  - _normalize_boundary_units
  - _make_instance
  - _get_special_step
  - _expand_flushes
  - FLUSH
  - PARTITIONED_PROCESSES
  - ALL_PARTITIONED
  - ALLOCATOR_LAYERS
"""

from __future__ import annotations

import copy
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Process imports (needed for PARTITIONED_PROCESSES and _instantiate_step)
# ---------------------------------------------------------------------------
from v2ecoli.processes.rna_degradation import RnaDegradation
from v2ecoli.processes.transcript_elongation import TranscriptElongation
from v2ecoli.processes.polypeptide_elongation import PolypeptideElongation


# ---------------------------------------------------------------------------
# FLUSH sentinel and helper
# ---------------------------------------------------------------------------

# FLUSH marks a position where the UniqueNumpyUpdater buffer should be
# drained before the next layer runs.  Expanded to a real step by
# _expand_flushes() at build time.
FLUSH = '__unique_flush__'


# --- Cache-driven config loader -------------------------------------------
# Composite generators build from a cache bundle (pre-resolved configs) rather
# than a live ParCa sim_data object. CachedConfigLoader is the small stand-in
# they pass to step builders: it serves configs by name and exposes the few
# sim_data attributes those builders read (unique-molecule definitions +
# expected dry-mass increase). Previously each generator (baseline,
# departitioned, reconciled, millard_pdmp_baseline) defined its own nested
# _CachedLoader with a 4-level inner-class mock; this is the single flattened,
# module-level version.

class _MockUniqueMolecule:
    def __init__(self, names):
        self.unique_molecule_definitions = {n: {} for n in names}


class _MockInternalState:
    def __init__(self, names):
        self.unique_molecule = _MockUniqueMolecule(names)


class _MockSimData:
    def __init__(self, unique_names, dry_mass_inc_dict):
        self.internal_state = _MockInternalState(unique_names)
        self.expectedDryMassIncreaseDict = dry_mass_inc_dict or {}


class CachedConfigLoader:
    """Serve pre-resolved configs from the cache bundle + a minimal sim_data
    stand-in. Replaces the per-generator nested ``_CachedLoader``."""

    def __init__(self, configs, unique_names, dry_mass_inc_dict,
                 cache_dir='out/cache'):
        self._configs = configs
        self.unique_names = unique_names
        self.cache_dir = cache_dir
        self.sim_data = _MockSimData(unique_names, dry_mass_inc_dict)

    def get_config_by_name(self, name):
        try:
            return self._configs[name]
        except KeyError:
            raise KeyError(f'Unknown: {name}')


def _expand_flushes(layers):
    """Replace each FLUSH sentinel with a real [unique_update_N] sub-layer.

    N is assigned in declaration order so state keys stay stable across
    architecture variants (baseline, departitioned, reconciled).
    """
    out, n = [], 0
    for layer in layers:
        if layer == FLUSH:
            n += 1
            out.append([f'unique_update_{n}'])
        else:
            out.append(layer)
    return out


# ---------------------------------------------------------------------------
# Partitioned process registry
# ---------------------------------------------------------------------------

PARTITIONED_PROCESSES = {
    'ecoli-rna-degradation': RnaDegradation,
    'ecoli-transcript-elongation': TranscriptElongation,
    'ecoli-polypeptide-elongation': PolypeptideElongation,
}

ALL_PARTITIONED = list(PARTITIONED_PROCESSES.keys())


# ---------------------------------------------------------------------------
# Canonical visualization set for single-cell architectures.
# ---------------------------------------------------------------------------
# Shared by baseline / departitioned / reconciled — all three resolve to the
# same observables.mass / unique-molecule layout so the same viz tiles apply.
# Opt-in: every study built on one of the v2ecoli single-cell composites
# (baseline / reconciled / departitioned) that wants the legacy
# WorkflowVisualization + NetworkVisualization panels must declare them
# explicitly in its study.yaml `visualizations:` block. Auto-attaching them
# here used to be the default, but the panels rendered confusingly empty for
# planning-not-yet-run studies and for any study whose runs.db wasn't
# populated yet — see docs/superpowers/notes/2026-05-19-dashboard-runner-friction.md
# item #17. Re-enable per study by copying the two-entry list below into
# the study's spec, OR opt back into project-wide auto-attach by setting
# ``visualizations=v2ecoli_default_single_cell_visualizations()`` on the
# specific @composite_generator call.
#
# Note: the cell-mass / cell-volume / growth-rate / absolute-mass-components /
# mass-fold-change TimeSeriesPlots that previously lived here used a
# `config.observable: '<short-name>'` convention that the dashboard's
# build_viz_composite (vivarium-dashboard, lib/investigations.py) does not yet
# understand — it only honors `inputs_map` for port→observable wiring, so
# those plots came out with empty y-data even though the emitter recorded
# them. They will land back here once the dashboard grows a short-name /
# leaf-name resolver for canonical viz wiring.
DEFAULT_SINGLE_CELL_VISUALIZATIONS: list[dict] = []


def v2ecoli_default_single_cell_visualizations() -> list[dict]:
    """The legacy workflow + topology viz spec, available for explicit opt-in.

    Returns a fresh list each call so the consumer can mutate without
    polluting other studies' specs.
    """
    return [
        {
            'name': 'workflow',
            'address': 'local:WorkflowVisualization',
            'config': {'title': 'v2ecoli — single-cell lifecycle'},
        },
        {
            'name': 'topology',
            'address': 'local:NetworkVisualization',
            'config': {'title': 'Process topology'},
        },
    ]

ALLOCATOR_LAYERS = {
    # RNA degradation shares water with polymerizations
    'allocator_2': ['ecoli-rna-degradation'],
    # Elongation processes compete for NTPs / charged tRNAs
    'allocator_3': ['ecoli-polypeptide-elongation',
                    'ecoli-transcript-elongation'],
}


# ---------------------------------------------------------------------------
# Emitter override (module-level)
# ---------------------------------------------------------------------------
# When a Study runner needs persistent time-series history, it can set
# ``_EMITTER_OVERRIDE`` to a dict of SQLiteEmitter-config kwargs (e.g.
# ``{'file_path': '.../studies/<name>', 'db_file': 'runs.db',
#   'name': 'baseline-steady-state'}``) BEFORE building the composite.
# When set, the special 'emitter' step in baseline / departitioned /
# reconciled swaps RAMEmitter for SQLiteEmitter and expands the emit
# schema to cover the per-listener fields the dnaa investigation reads.
#
# Use the ``sqlite_emitter()`` context manager below for safety —
# ensures the override is cleared even on exceptions.
_EMITTER_OVERRIDE: dict | None = None

# Parallel override for the ParquetEmitter path. When set, the 'emitter' step
# materialises as a v2ecoli ParquetEmitter writing to a hive-partitioned dir,
# independent of the SQLite override above. Both may be set simultaneously
# (parquet wins — sqlite stays available for dashboard's Simulations-DB tab).
_PARQUET_EMITTER_OVERRIDE: dict | None = None

# When True (and no parquet/sqlite override is set), the 'emitter' step
# materialises as a minimal RAMEmitter capturing ONLY global_time — instead of
# the default full-state RAMEmitter (bulk + chromosome + listeners, multi-MB
# per row). Used by callers that emit out-of-band (e.g. the workflow's external
# XArrayEmitter) and don't want the internal emitter wasting memory.
_NULL_EMITTER_OVERRIDE: bool = False

# The generator-declared default emitter (see pbg_superpowers
# composite_generator's ``emitters=`` convention + ``emitter_defaults``). A
# ``{address, config, paths?}`` dict that the 'emitter' step materialises when
# NO external override (parquet / sqlite / null) is set — i.e. the composite's
# own default sink travels with the generator instead of being toggled by a
# caller. This is the lowest-priority *declared* source; the RAMEmitter remains
# the final fallback when even this is None. The baseline generator sets it
# from its own ``@composite_generator(emitters=[...])`` entry around the build.
_DEFAULT_EMITTER_DECL: dict | None = None


def set_emitter_override(config: dict | None) -> None:
    """Set the module-level SQLite emitter override. Pass None to clear."""
    global _EMITTER_OVERRIDE
    _EMITTER_OVERRIDE = config


def set_parquet_emitter_override(config: dict | None) -> None:
    """Set the module-level Parquet emitter override. Pass None to clear."""
    global _PARQUET_EMITTER_OVERRIDE
    _PARQUET_EMITTER_OVERRIDE = config


def set_null_emitter_override(flag: bool) -> None:
    """Minimise the internal 'emitter' step to global_time only (see
    ``_NULL_EMITTER_OVERRIDE``). Pass False to restore the full-state default."""
    global _NULL_EMITTER_OVERRIDE
    _NULL_EMITTER_OVERRIDE = bool(flag)


def set_default_emitter_decl(decl: dict | None) -> None:
    """Set the generator-declared default emitter (see ``_DEFAULT_EMITTER_DECL``).

    ``decl`` is a ``{address, config, paths?}`` dict from a generator's
    ``@composite_generator(emitters=[...])`` declaration (read via
    ``pbg_superpowers.composite_generator.emitter_defaults``). Pass None to
    clear. The 'emitter' step uses it only when no external parquet / sqlite /
    null override is active. A generator should set this around its build and
    clear it in a ``finally`` so it never leaks into a later composite built in
    the same process.
    """
    global _DEFAULT_EMITTER_DECL
    _DEFAULT_EMITTER_DECL = decl


def _build_declared_emitter(decl: dict, listeners_schema: dict, core):
    """Materialise the generator-declared default emitter step.

    Maps ``decl['address']`` (the registered emitter link, with or without a
    ``local:`` prefix) to the right emitter class and an emit-schema/topology
    appropriate for v2ecoli. ``decl['config']`` is merged in as the base
    config. Returns ``(instance, topo)``.

    Only the emitters v2ecoli actually ships are recognised; an unknown
    address raises rather than silently falling back, so a typo in a
    generator's ``emitters=`` declaration surfaces at build time.
    """
    from process_bigraph.emitter import RAMEmitter, SQLiteEmitter

    address = (decl.get("address") or "").split(":", 1)[-1]
    cfg_in = dict(decl.get("config") or {})

    if address == "ParquetEmitter":
        try:
            from pbg_emitters import ParquetEmitter
        except ImportError:
            # A generator-declared *default* must not hard-fail the build when
            # the optional [parquet] extra is absent (e.g. CI behavior-tests
            # install only the dev extra). Degrade to the historical full-capture
            # in-memory RAMEmitter — the pre-declaration default — and warn.
            warnings.warn(
                "generator declared a ParquetEmitter default but the [parquet] "
                "extra is not installed; falling back to in-memory RAMEmitter. "
                "Install with: pip install 'v2ecoli[parquet]' to persist parquet.")
            emit_schema = {
                "global_time": "float", "bulk": "array",
                "listeners": listeners_schema,
                "full_chromosome": "node", "active_replisome": "node",
                "active_RNAP": "node", "chromosome_domain": "node",
            }
            topo = {
                "global_time": ("global_time",), "bulk": ("bulk",),
                "listeners": ("listeners",),
                "full_chromosome": ("unique", "full_chromosome"),
                "active_replisome": ("unique", "active_replisome"),
                "active_RNAP": ("unique", "active_RNAP"),
                "chromosome_domain": ("unique", "chromosome_domain"),
            }
            return RAMEmitter({"emit": emit_schema}, core), topo
        # Reuse the vEcoli-shaped preset so the hive layout + dtype overrides
        # match upstream, seeded by whatever the declaration provided.
        from v2ecoli.library.emitter_presets import parquet_vecoli
        out_dir = cfg_in.pop("out_dir", None)
        if out_dir is None:
            ws_root = _find_workspace_root()
            out_dir = (str(ws_root / ".pbg" / "parquet-runs")
                       if ws_root is not None else "out/parquet")
        preset = parquet_vecoli(out_dir=out_dir,
                                experiment_id=cfg_in.pop("experiment_id", "default"))
        emit_schema = {
            "global_time": "float",
            "bulk": "array[integer]",
            "listeners": listeners_schema,
        }
        topo = {
            "global_time": ("global_time",),
            "bulk": ("bulk",),
            "listeners": ("listeners",),
        }
        cfg = {"emit": emit_schema, **preset, **cfg_in}
        return ParquetEmitter(cfg, core), topo

    if address == "SQLiteEmitter":
        emit_schema = {"global_time": "float", "listeners": listeners_schema}
        topo = {"global_time": ("global_time",), "listeners": ("listeners",)}
        cfg = {"emit": emit_schema, **cfg_in}
        return SQLiteEmitter(cfg, core), topo

    if address == "RAMEmitter":
        emit_schema = {"global_time": "float", "listeners": listeners_schema}
        topo = {"global_time": ("global_time",), "listeners": ("listeners",)}
        cfg = {"emit": emit_schema, **cfg_in}
        return RAMEmitter(cfg, core), topo

    raise ValueError(
        f"declared default emitter address {decl.get('address')!r} is not "
        "recognised (expected one of ParquetEmitter, SQLiteEmitter, RAMEmitter)"
    )


import contextlib  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402


def _find_workspace_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) looking for ``workspace.yaml``.

    Returns the directory that contains ``workspace.yaml``, or ``None`` if
    none is found before hitting the filesystem root. Cheap, no imports.
    """
    cur = Path(start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / 'workspace.yaml').is_file():
            return d
    return None


def _ensure_study_columns(db_path: str) -> None:
    """Idempotently add study_slug / investigation_slug columns to ``simulations``.

    Safe to call on a missing DB (creates the parent dir + an empty DB with
    just the columns we add — the SQLiteEmitter's _init_history_db will
    create the rest of the schema on its own init).
    """
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        # Mirror the SQLiteEmitter's table layout — without recreating the
        # full schema, we just need the `simulations` table to exist before
        # we ALTER. If SQLiteEmitter has already initialized it, the CREATE
        # is a no-op.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id TEXT PRIMARY KEY,
                name TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                elapsed_seconds REAL,
                composite_config TEXT,
                metadata TEXT,
                emit_schema TEXT
            )
        ''')
        existing = {row[1] for row in conn.execute(
            'PRAGMA table_info(simulations)')}
        if 'study_slug' not in existing:
            conn.execute('ALTER TABLE simulations ADD COLUMN study_slug TEXT')
        if 'investigation_slug' not in existing:
            conn.execute(
                'ALTER TABLE simulations ADD COLUMN investigation_slug TEXT')
    finally:
        conn.close()


def _stamp_study_metadata(
    db_path: str,
    simulation_id: str,
    study_slug: str | None,
    investigation_slug: str | None,
) -> None:
    """Write study_slug + investigation_slug for ``simulation_id``.

    Upserts a row keyed on ``simulation_id`` and leaves any existing
    SQLiteEmitter-managed fields (name, emit_schema, ...) untouched. Use
    after the SQLiteEmitter step has been constructed (which inserts the
    initial row) — or before, in which case ``started_at`` defaults to
    now-UTC and the emitter will overwrite it on its own insert.
    """
    if study_slug is None and investigation_slug is None:
        return
    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ')
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            'INSERT INTO simulations '
            '(simulation_id, started_at, study_slug, investigation_slug) '
            'VALUES (?, ?, ?, ?) '
            'ON CONFLICT(simulation_id) DO UPDATE SET '
            '  study_slug = COALESCE(excluded.study_slug, simulations.study_slug), '
            '  investigation_slug = COALESCE(excluded.investigation_slug, simulations.investigation_slug)',
            (simulation_id, now_iso, study_slug, investigation_slug),
        )
    finally:
        conn.close()


@contextlib.contextmanager
def sqlite_emitter(*, file_path: str | None = None,
                   db_file: str | None = None,
                   name: str | None = None,
                   simulation_id: str | None = None,
                   subsample: int = 1,
                   study_slug: str | None = None,
                   investigation_slug: str | None = None):
    """Context manager: build composite with a SQLiteEmitter step.

    By default writes to ``<workspace_root>/.pbg/composite-runs.db`` — the
    same workspace-shared DB the vivarium-dashboard's Simulations DB tab
    aggregates from. Pass ``study_slug`` / ``investigation_slug`` to tag the
    run; the slugs are stored as columns on the ``simulations`` row so the
    dashboard can group / filter by them.

    To keep using a per-study DB, pass ``file_path`` (and optionally
    ``db_file``) explicitly — the old behavior is preserved.

    Usage::

        with sqlite_emitter(name='baseline-seed0',
                            study_slug='dnaa-01-expression-dynamics',
                            investigation_slug='dnaa-replication'):
            doc = baseline(cache_dir='out/cache')
            comp = Composite(doc, core=core)
            comp.update({}, 60.0 * 60)
    """
    # Resolve workspace-default DB path when caller didn't pin one.
    if file_path is None:
        ws_root = _find_workspace_root()
        if ws_root is None:
            raise RuntimeError(
                'sqlite_emitter(): no file_path given and no workspace.yaml '
                'found walking up from cwd; either run from inside a '
                'workspace or pass file_path explicitly.')
        file_path = str(ws_root / '.pbg')
        if db_file is None:
            db_file = 'composite-runs.db'
    else:
        if db_file is None:
            db_file = 'runs.db'

    # Pin the simulation_id upfront so we can stamp metadata around the
    # emitter's own init insert.
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())

    cfg = {
        'file_path': file_path,
        'db_file': db_file,
        'simulation_id': simulation_id,
    }
    if name is not None:
        cfg['name'] = name
    if subsample != 1:
        cfg['subsample'] = subsample

    db_path = os.path.join(file_path, db_file)
    # Ensure our extra columns exist on the simulations table BEFORE the
    # SQLiteEmitter step runs its own _init_history_db (which is a no-op for
    # already-existing tables). Then stamp the slugs. Safe to run unconditionally
    # — the helpers are idempotent and tolerate a missing file.
    _ensure_study_columns(db_path)
    _stamp_study_metadata(db_path, simulation_id, study_slug, investigation_slug)

    set_emitter_override(cfg)
    try:
        yield cfg
    finally:
        # Re-stamp in case the emitter's own insert ran after ours and we
        # need to ensure the slugs survived. ON CONFLICT DO UPDATE preserves
        # any non-NULL value, so this is a belt-and-suspenders no-op when
        # things already stuck.
        try:
            _stamp_study_metadata(
                db_path, simulation_id, study_slug, investigation_slug)
        except sqlite3.OperationalError:
            pass
        set_emitter_override(None)


class ParquetEmitterContext:
    """Yielded from ``parquet_emitter()``. Bind the composite via
    :meth:`bind` so its accumulated rows flush on context exit; this
    closes the friction-#3 hole where the context manager couldn't
    enforce its own lifecycle correctness (only ``configuration/`` landed,
    no ``history/`` or ``success/``).

    .cfg holds the emitter config dict (back-compat with the dict that
    pre-2026-05-28 ``parquet_emitter()`` yielded directly).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._composite = None
        self._flushed = False

    def bind(self, composite) -> None:
        """Register the composite for auto-flush on context exit."""
        self._composite = composite

    def flush(self, composite=None, *, success: bool = True) -> int:
        """Flush now. Uses the bound composite if ``composite`` is None.

        Returns the number of ParquetEmitter instances flushed. Safe to
        call multiple times — ``flush_all_in_composite`` skips already-
        closed instances."""
        target = composite if composite is not None else self._composite
        if target is None:
            raise ValueError(
                "ParquetEmitterContext.flush(): no composite given and "
                "none bound; call .bind(composite) first.")
        n = flush_parquet(target, success=success)
        self._flushed = True
        return n


@contextlib.contextmanager
def parquet_emitter(*, out_dir: str | None = None,
                    experiment_id: str | None = None,
                    variant: int = 0,
                    lineage_seed: int = 0,
                    generation: int | None = None,
                    agent_id: str = "1",
                    batch_size: int = 400,
                    threaded: bool = True,
                    study_slug: str | None = None,
                    investigation_slug: str | None = None,
                    extra_metadata: dict | None = None):
    """Context manager: build composite with a ParquetEmitter step.

    Workspace-default ``out_dir`` is ``<workspace_root>/.pbg/parquet-runs/``.
    Hive partitioning matches vEcoli's layout
    (``experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=...``),
    so anyone with a vEcoli analysis pipeline can point at the output dir
    directly.

    Usage (auto-flush — recommended, since 2026-05-28)::

        with parquet_emitter(experiment_id='baseline-seed0',
                             study_slug='dnaa-01-expression-dynamics') as emit:
            doc = baseline(cache_dir='out/cache')
            comp = Composite(doc, core=core)
            emit.bind(comp)             # <- enables auto-flush on exit
            comp.update({}, 60.0 * 60)
        # ParquetEmitter.close(success=True) called automatically; if an
        # exception fired in the with-block, close(success=False) instead.

    Legacy usage (still works for callers that already manage the flush)::

        with parquet_emitter(...):
            comp = ...
            comp.update(...)
            flush_parquet(comp, success=True)   # explicit, no .bind()

    If you neither call .bind() nor flush_parquet(), the context manager
    silently degrades to the pre-2026-05-28 behaviour: the override
    clears but the trailing partial batch + success sentinel are lost.
    Friction note 2026-05-27 #3 for the original incident.

    Storage trade-off vs ``sqlite_emitter()``: Parquet is column-oriented and
    typically 3-5x smaller on disk for v2ecoli-shaped runs (sparse arrays,
    listener-heavy schema). The dashboard's Simulations-DB tab does not yet
    read parquet — for now use ``sqlite_emitter()`` if dashboard inspection
    is required.
    """
    if out_dir is None:
        ws_root = _find_workspace_root()
        if ws_root is None:
            raise RuntimeError(
                "parquet_emitter(): no out_dir given and no workspace.yaml "
                "found walking up from cwd; either run from inside a "
                "workspace or pass out_dir explicitly.")
        out_dir = str(ws_root / ".pbg" / "parquet-runs")

    if experiment_id is None:
        experiment_id = "default-" + uuid.uuid4().hex[:8]

    # Build the emitter config dict via the vEcoli-shaped preset so the
    # hive layout + dtype overrides match upstream conventions exactly.
    from v2ecoli.library.emitter_presets import parquet_vecoli
    cfg = parquet_vecoli(
        out_dir=out_dir,
        experiment_id=experiment_id,
        variant=variant,
        lineage_seed=lineage_seed,
        agent_id=agent_id,
        generation=generation,
        batch_size=batch_size,
        threaded=threaded,
        extra_metadata=extra_metadata,
    )
    # Carry the study/investigation slugs in metadata so downstream readers
    # can group hive-partitioned runs by study without crossing into the
    # SQLite simulations-table convention.
    if study_slug or investigation_slug:
        meta = dict(cfg.get("metadata") or {})
        if study_slug:
            meta["study_slug"] = study_slug
        if investigation_slug:
            meta["investigation_slug"] = investigation_slug
        cfg["metadata"] = meta

    ctx = ParquetEmitterContext(cfg)
    set_parquet_emitter_override(cfg)
    exc_fired = False
    try:
        yield ctx
    except BaseException:
        exc_fired = True
        raise
    finally:
        # Auto-flush if bound but not explicitly flushed. On exception,
        # close with success=False so the sentinel honestly reflects the
        # failed run. Swallow auto-flush errors so they can't mask the
        # caller's original exception.
        if ctx._composite is not None and not ctx._flushed:
            try:
                ctx.flush(success=not exc_fired)
            except Exception:
                pass
        set_parquet_emitter_override(None)


def flush_parquet(composite, *, success: bool = True) -> int:
    """Flush all ParquetEmitter steps inside ``composite`` to disk.

    The ``parquet_emitter()`` context manager only sets / clears the global
    override — it never sees the composite, so it cannot trigger ``close()``
    on the emitter step. Without an explicit flush the trailing partial
    batch (rows after the last batch_size flush) stays in memory; only the
    ``configuration/`` parquet lands, no ``history/`` or ``success/``.

    Call this right before the runner exits its ``with parquet_emitter(...)``
    block, after the simulation loop completes::

        with parquet_emitter(out_dir=..., experiment_id=...):
            composite = build_composite(...)
            ... sim loop ...
            flush_parquet(composite, success=True)

    Returns the number of ParquetEmitter instances flushed (0 when the
    [parquet] extra is not installed or the composite has no parquet
    emitter step).
    """
    try:
        from pbg_emitters import ParquetEmitter
    except ImportError:
        return 0
    return ParquetEmitter.flush_all_in_composite(composite, success=success)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _seed_state_from_defaults(cell_state):
    """Walk each edge's port_defaults and inject values into cell_state.

    Each step instance provides port_defaults() which returns a nested dict
    of default values for ports that need pre-population.
    """
    for edge in list(cell_state.values()):
        if not (isinstance(edge, dict) and 'instance' in edge):
            continue
        instance = edge['instance']
        try:
            defaults = instance.port_defaults()
        except (AttributeError, Exception):
            continue
        if not defaults:
            continue
        for port_name, wire_path in edge.get('inputs', {}).items():
            port_default = defaults.get(port_name)
            if port_default is None or not isinstance(wire_path, list):
                continue
            if isinstance(port_default, dict):
                _inject_nested_defaults(cell_state, wire_path, port_default)
            else:
                _inject_port_default(cell_state, wire_path, {'_default': port_default})


def _inject_nested_defaults(state, wire_path, defaults_dict):
    """Recursively inject nested default values into state."""
    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, val in defaults_dict.items():
        if isinstance(val, dict):
            sub = target.setdefault(key, {})
            if isinstance(sub, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict):
                        sub2 = sub.setdefault(k2, {})
                        if isinstance(sub2, dict):
                            for k3, v3 in v2.items():
                                sub2.setdefault(k3, v3)
                    else:
                        sub.setdefault(k2, v2)
        else:
            target.setdefault(key, val)


def _inject_port_default(state, wire_path, port_schema):
    """Inject _default values along wire_path into state."""
    if '_default' in port_schema:
        default = port_schema['_default']
        target = state
        for segment in wire_path[:-1]:
            if not isinstance(target, dict):
                return
            target = target.setdefault(segment, {})
        if isinstance(target, dict) and wire_path:
            key = wire_path[-1]
            current = target.get(key)
            if current is None or (
                    isinstance(current, (list, dict, tuple))
                    and len(current) == 0):
                target[key] = default
        return

    target = state
    for segment in wire_path:
        if not isinstance(target, dict):
            return
        target = target.setdefault(segment, {})
    if not isinstance(target, dict):
        return
    for key, subport in port_schema.items():
        if key.startswith('_') or key == '*' or not isinstance(subport, dict):
            continue
        _inject_port_default(target, [key], subport)


def seed_mass_listener(cell_state, core):
    """Run mass listener once to populate initial mass values."""
    for name in ['post-division-mass-listener', 'ecoli-mass-listener']:
        edge = cell_state.get(name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue
        instance = edge['instance']
        if not hasattr(instance, 'next_update'):
            continue

        view = {}
        wires = edge.get('inputs', {})
        for port, wire in wires.items():
            if isinstance(wire, list) and wire:
                target = cell_state
                for seg in wire:
                    if isinstance(target, dict):
                        target = target.get(seg)
                    else:
                        target = None
                        break
                if target is not None:
                    view[port] = target

        for key in ['bulk']:
            arr = view.get(key)
            if arr is not None and hasattr(arr, 'flags'):
                try:
                    arr.flags.writeable = True
                except ValueError:
                    view[key] = arr.copy()
                    view[key].flags.writeable = True
        for uname, uarr in view.get('unique', {}).items():
            if hasattr(uarr, 'flags'):
                try:
                    uarr.flags.writeable = True
                except ValueError:
                    pass

        try:
            delta = instance.next_update(1.0, view)
            if delta and 'listeners' in delta:
                mass = delta['listeners'].get('mass', {})
                cell_state['listeners']['mass'].update(mass)
        except Exception as exc:
            # Don't crash composite construction if the mass listener fails
            # on partial state — but surface the failure. A silent miss here
            # leaves cell_mass=0, which surfaces several layers downstream
            # as a non-finite y_init in Equilibrium's ODE solver.
            warnings.warn(
                f"seed_mass_listener({name}) failed: "
                f"{type(exc).__name__}: {exc}. "
                f"cell_mass / dry_mass left unset.",
                RuntimeWarning,
                stacklevel=2,
            )
        break


def list_paths(path):
    """Convert tuple paths to list paths. Flatten _path dicts."""
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        if '_path' in path:
            result = {}
            for key, subpath in path.items():
                if key == '_path':
                    continue
                result[key] = list_paths(subpath)
            return result
        return {key: list_paths(subpath) for key, subpath in path.items()}
    return path


def inject_flow_dependencies(cell_state, flow_order, layers=None):
    """Add flow token wiring and priorities to enforce execution order.

    When layers is provided, uses layer-based tokens: all steps in a layer
    depend on the previous layer's token and produce the current layer's
    token.
    """
    if layers is None:
        n = len(flow_order)
        for i, step_name in enumerate(flow_order):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                continue
            edge['priority'] = float(n - i)
            if i == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']
            if i > 0:
                edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
            if i < n - 1:
                edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']
        return

    n_layers = len(layers)
    step_idx = 0
    total_steps = sum(len(layer) for layer in layers)

    for layer_idx, layer in enumerate(layers):
        for j, step_name in enumerate(layer):
            edge = cell_state.get(step_name)
            if not isinstance(edge, dict):
                step_idx += 1
                continue

            edge['priority'] = float(total_steps - step_idx)

            if layer_idx == 0:
                edge.setdefault('inputs', {})['global_time'] = ['global_time']

            if layer_idx > 0:
                edge.setdefault('inputs', {})[f'_layer_in_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx - 1}']

            if layer_idx < n_layers - 1:
                edge.setdefault('outputs', {})[f'_layer_out_{layer_idx}'] = \
                    [f'_layer_token_{layer_idx}']

            step_idx += 1


def make_edge(instance, topology, input_topology=None, output_topology=None,
              edge_type='step', config=None):
    """Create an edge dict for a process/step instance."""
    wires = list_paths(topology)
    in_wires = list_paths(input_topology) if input_topology is not None else wires
    out_wires = list_paths(output_topology) if output_topology is not None else wires
    state = {'priority': 1.0} if edge_type == 'step' else {'interval': 1.0}

    inputs_schema = {}
    outputs_schema = {}
    if hasattr(instance, 'inputs'):
        try:
            inputs_schema = instance.inputs()
        except Exception:
            pass
    if hasattr(instance, 'outputs'):
        try:
            outputs_schema = instance.outputs()
        except Exception:
            pass

    cls = type(instance)
    address = f'local:{cls.__module__}.{cls.__qualname__}'
    raw_config = config or getattr(instance, '_raw_config', {})

    state.update({
        '_type': edge_type,
        'address': address,
        'config': raw_config,
        '_inputs': inputs_schema,
        '_outputs': outputs_schema,
        'instance': instance,
        'inputs': copy.deepcopy(in_wires),
        'outputs': copy.deepcopy(out_wires),
    })
    return state


def _normalize_boundary_units(cell_state):
    """Re-create pint Quantities in boundary.external with the current registry."""
    boundary = cell_state.get('boundary', {})
    external = boundary.get('external', {})
    if not isinstance(external, dict):
        return
    from v2ecoli.library.unit_bridge import unum_to_pint
    for key, val in external.items():
        q = unum_to_pint(val)
        if hasattr(q, 'magnitude') and hasattr(q, 'units'):
            external[key] = float(q.magnitude)


# ---------------------------------------------------------------------------
# Step instantiation helpers
# ---------------------------------------------------------------------------

def _make_instance(cls, config, core):
    """Instantiate a Step/Process class, trying multiple signatures."""
    from v2ecoli.library.ecoli_step import set_current_core
    set_current_core(core)
    try:
        return cls(parameters=config)
    except TypeError:
        try:
            return cls(config=config, core=core)
        except TypeError:
            return cls(config)
    finally:
        set_current_core(None)


def _get_special_step(loader, step_name, core):
    """Handle steps that aren't in LoadSimData's config map."""
    from v2ecoli.steps.unique_update import UniqueUpdate

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule
        .unique_molecule_definitions.keys())

    if step_name.startswith('unique_update'):
        UNIQUE_PLURAL = {
            'full_chromosome': 'full_chromosomes',
            'chromosome_domain': 'chromosome_domains',
            'active_replisome': 'active_replisomes',
            'oriC': 'oriCs',
            'promoter': 'promoters',
            'chromosomal_segment': 'chromosomal_segments',
            'DnaA_box': 'DnaA_boxes',
            'active_RNAP': 'active_RNAPs',
            'RNA': 'RNAs',
            'gene': 'genes',
            'active_ribosome': 'active_ribosome',
        }
        unique_topo_v1 = {}
        unique_names_v1 = []
        for name in unique_names:
            plural = UNIQUE_PLURAL.get(name, name)
            unique_topo_v1[plural] = ('unique', name)
            unique_names_v1.append(plural)
        config = {'unique_names': unique_names_v1, 'unique_topo': unique_topo_v1}
        instance = _make_instance(UniqueUpdate, config, core)
        return instance, unique_topo_v1, 'step'

    if step_name == 'ecoli-mass-conservation':
        from v2ecoli.steps.listeners.mass_conservation import MassConservationListener
        from v2ecoli.types.quantity import ureg as units
        # Per-molecule masses (fg/count) of the metabolic exchange molecules,
        # keyed by their environment name (compartment-stripped), so the
        # listener can convert per-tick exchange counts into a mass.
        exchange_masses = {}
        try:
            from v2ecoli.library.unit_bridge import unum_to_pint
            from v2ecoli.library.config_resolver import resolve_config
            # resolve_config realizes the method specs (get_masses) into callables.
            met_cfg = resolve_config(loader.get_config_by_name('ecoli-metabolism'))
            exchange_molecules = list(met_cfg['exchange_molecules'])
            mws = unum_to_pint(met_cfg['get_masses'](exchange_molecules))
            n_avogadro = 6.02214076e23  # molecules / mol
            for mol, mw in zip(exchange_molecules, mws):
                g_per_molecule = mw.to(units.g / units.mol).magnitude / n_avogadro
                exchange_masses[mol[:-3]] = (g_per_molecule * units.g).to(units.fg)
        except Exception:
            # Listener still runs (residual == Δdry_mass) if masses are
            # unavailable; better an observable signal than a build failure.
            exchange_masses = {}
        # tolerance + warmup_ticks fall back to the Step's schema defaults
        # (5% cumulative drift, 10-tick warmup).
        config = {'exchange_masses': exchange_masses}
        instance = _make_instance(MassConservationListener, config, core)
        return instance, instance.topology, 'step'

    if step_name == 'global_clock':
        from v2ecoli.steps.global_clock import GlobalClock
        instance = GlobalClock(config={}, core=core)
        topo = {
            'global_time': ('global_time',),
            'next_update_time': ('next_update_time',),
        }
        return instance, topo, 'process'

    if step_name == 'emitter':
        from process_bigraph.emitter import RAMEmitter, SQLiteEmitter
        # ParquetEmitter is optional — the import lives behind an extra so
        # workspaces without the [parquet] extra still build composites.
        # Imported directly from pbg-emitters (the upstream library);
        # ``v2ecoli.library.parquet_emitter`` is just a re-export shim.
        try:
            from pbg_emitters import ParquetEmitter
        except ImportError:
            ParquetEmitter = None  # type: ignore[assignment]
        # Mass listener fields — always emitted, used by the workflow report.
        mass_schema = {
            'cell_mass': 'float',
            'water_mass': 'float',
            'dry_mass': 'float',
            'protein_mass': 'float',
            'rna_mass': 'float',
            'rRna_mass': 'float',
            'tRna_mass': 'float',
            'mRna_mass': 'float',
            'dna_mass': 'float',
            'smallMolecule_mass': 'float',
            'instantaneous_growth_rate': 'float',
            'volume': 'float',
        }
        listeners_schema = {'mass': mass_schema}

        # When a caller-set override is active (typical for study runs that
        # need persistent SQLite history), expand the emit schema to capture
        # the dnaa-investigation readouts: monomer_counts / rna_counts /
        # rna_synth_prob / rnap_data. Cheap to add — they're already in the
        # per-step state tree.
        # Parquet override wins over SQLite — see set_parquet_emitter_override.
        parquet_override = _PARQUET_EMITTER_OVERRIDE
        override = _EMITTER_OVERRIDE
        default_decl = _DEFAULT_EMITTER_DECL
        if (parquet_override is not None or override is not None
                or default_decl is not None):
            # Only the leaves the dnaa-investigation readouts actually consume.
            # NB: ``monomer_counts`` is a flat array store (not nested under
            # a ``monomerCounts`` key) — confirmed at runtime via the
            # composite-emitted state shape.
            listeners_schema.update({
                'monomer_counts': 'array[integer]',
                'rna_counts': {
                    'mRNA_counts': 'array[integer]',
                },
                'rna_synth_prob': {
                    'n_bound_TF_per_TU': 'array[float]',
                    'n_actual_bound': 'array[integer]',
                    'total_rna_init': 'integer',
                },
                'rnap_data': {
                    'rna_init_event': 'array[integer]',
                    'did_initialize': 'integer',
                    # Phase-3 sprint-1 per-tick log-likelihood; needed
                    # alongside rna_init_event so post-hoc inference
                    # can compare observation to model.
                    'log_likelihood': 'float',
                },
                'ribosome_data': {
                    # Phase-3 sprint-2 per-tick log-likelihood from
                    # PolypeptideInitiation.
                    'log_likelihood': 'float',
                },
                'replication_data': {
                    'number_of_oric': 'integer',
                    'free_DnaA_boxes': 'integer',
                    'total_DnaA_boxes': 'integer',
                },
                # Phase-3 sprint-3: aggregate likelihood listener
                # written by LikelihoodCollector. Single-scalar per-tick
                # observable for downstream inference tools (ABC-SMC,
                # SBC) — read directly from the emitter store without
                # re-aggregating per-process fields.
                'likelihood': {
                    'transcript_init': 'float',
                    'polypeptide_init': 'float',
                    'total': 'float',
                },
            })

        if parquet_override is not None:
            # Parquet path — column-oriented hive-partitioned output. Unlike
            # the SQLite path, this captures the raw ``bulk`` count array
            # (~25k molecules) in addition to global_time + listeners. Parquet
            # is column-oriented and compresses the wide bulk vector far better
            # than SQLite's row store, so the disk-cost concern documented in
            # the SQLite branch below does not apply here — and downstream
            # analyses (and vEcoli parity) need per-molecule counts.
            if ParquetEmitter is None:
                raise ImportError(
                    "ParquetEmitter override set but [parquet] extra not "
                    "installed. Run: pip install 'v2ecoli[parquet]'"
                )
            emit_schema = {
                'global_time': 'float',
                'bulk': 'array[integer]',
                'listeners': listeners_schema,
            }
            topo = {
                'global_time': ('global_time',),
                'bulk': ('bulk',),
                'listeners': ('listeners',),
            }
            cfg = {'emit': emit_schema, **parquet_override}
            instance = ParquetEmitter(cfg, core)
        elif override is not None:
            # Persistent SQLite path (default-baseline + per-study run
            # scripts). Capture ONLY global_time + listeners. The raw
            # ``bulk`` store (~25k molecules) and the unique-node structures
            # (full_chromosome / active_replisome / active_RNAP /
            # chromosome_domain) are MULTI-MB PER ROW — emitting them every
            # tick produced a 5.5 GB default-baseline db (1.75 MB × 2461
            # rows) that timed out the dashboard's post-hoc chart extraction
            # (blank study pages). No viz/test reads them: study yamls
            # declare listeners.* paths, and raw bulk-id paths aren't even
            # json_extract-able. The bulk-derived observable studies DO use
            # (monomer_counts) lives under listeners and is kept.
            emit_schema = {
                'global_time': 'float',
                'listeners': listeners_schema,
            }
            topo = {
                'global_time': ('global_time',),
                'listeners': ('listeners',),
            }
            cfg = {'emit': emit_schema, **override}
            instance = SQLiteEmitter(cfg, core)
        elif _NULL_EMITTER_OVERRIDE:
            # Minimal RAMEmitter (global_time only): the caller emits out of
            # band (e.g. the workflow's external XArrayEmitter), so the internal
            # emitter must not waste memory capturing full state every tick.
            emit_schema = {'global_time': 'float'}
            topo = {'global_time': ('global_time',)}
            instance = RAMEmitter({'emit': emit_schema}, core)
        elif default_decl is not None:
            # Generator-declared default sink (e.g. baseline ships a
            # ParquetEmitter via @composite_generator(emitters=[...])). Lower
            # priority than every external override above; higher than the bare
            # RAMEmitter fallback below. The composite's own default emitter
            # thus travels with the generator for standalone runs.
            instance, topo = _build_declared_emitter(
                default_decl, listeners_schema, core)
        else:
            # In-memory RAMEmitter: no disk cost, so keep the full capture
            # (bulk + chromosome/replisome structures) for callers that
            # want the complete state tree in RAM.
            emit_schema = {
                'global_time': 'float',
                'bulk': 'array',
                'listeners': listeners_schema,
                'full_chromosome': 'node',
                'active_replisome': 'node',
                'active_RNAP': 'node',
                'chromosome_domain': 'node',
            }
            topo = {
                'global_time': ('global_time',),
                'bulk': ('bulk',),
                'listeners': ('listeners',),
                'full_chromosome': ('unique', 'full_chromosome'),
                'active_replisome': ('unique', 'active_replisome'),
                'active_RNAP': ('unique', 'active_RNAP'),
                'chromosome_domain': ('unique', 'chromosome_domain'),
            }
            instance = RAMEmitter({'emit': emit_schema}, core)

        return instance, topo, 'step'

    if step_name == 'ppgpp-initiation':
        from v2ecoli.steps.ppgpp_initiation import PpgppInitiation
        try:
            ti_config = loader.get_config_by_name('ecoli-transcript-initiation')
        except (KeyError, AttributeError):
            ti_config = {}
        ppgpp_config = {
            'ppgpp': ti_config.get('ppgpp', ''),
            'synth_prob': ti_config.get('synth_prob'),
            'copy_number': ti_config.get('copy_number', 1),
            'n_avogadro': ti_config.get('n_avogadro', 0),
            'cell_density': ti_config.get('cell_density', 0),
            'get_rnap_active_fraction_from_ppGpp': ti_config.get(
                'get_rnap_active_fraction_from_ppGpp'),
            'trna_attenuation': ti_config.get('trna_attenuation', False),
            'attenuated_rna_indices': ti_config.get('attenuated_rna_indices', []),
            'attenuation_adjustments': ti_config.get('attenuation_adjustments', []),
        }
        from v2ecoli.library.config_resolver import resolve_config
        ppgpp_config = resolve_config(ppgpp_config)
        instance = _make_instance(PpgppInitiation, ppgpp_config, core)
        topo = {
            'bulk': ('bulk',),
            'listeners': ('listeners',),
            'ppgpp_state': ('ppgpp_state',),
        }
        return instance, topo, 'step'

    if step_name == 'trna-attenuation-config':
        from v2ecoli.steps.trna_attenuation import TrnaAttenuationConfig
        try:
            te_config = loader.get_config_by_name('ecoli-transcript-elongation')
        except (KeyError, AttributeError):
            te_config = {}
        att_config = {
            'get_attenuation_stop_probabilities': te_config.get(
                'get_attenuation_stop_probabilities'),
            'attenuated_rna_indices': te_config.get(
                'attenuated_rna_indices', []),
            'location_lookup': te_config.get('location_lookup', {}),
            'cell_density': te_config.get('cell_density', 0),
            'n_avogadro': te_config.get('n_avogadro', 0),
            'charged_trnas': te_config.get('charged_trnas', []),
        }
        instance = _make_instance(TrnaAttenuationConfig, att_config, core)
        topo = {
            'attenuation_config': ('attenuation_config',),
        }
        return instance, topo, 'step'

    if step_name == 'replication_data_listener':
        from v2ecoli.steps.listeners.replication_data import ReplicationData
        config = {'time_step': 1}
        instance = ReplicationData(config=config, core=core)
        topology = getattr(instance, 'topology', {})
        return instance, topology, 'step'

    if step_name == 'mark_d_period':
        from v2ecoli.steps.division import MarkDPeriod
        instance = MarkDPeriod(config={}, core=core)
        topo = {
            'full_chromosome': ('unique', 'full_chromosome'),
            'global_time': ('global_time',),
            'divide': ('divide',),
        }
        return instance, topo, 'step'

    if step_name == 'division':
        from v2ecoli.steps.division import Division
        try:
            div_config = loader.get_config_by_name('division')
        except Exception:
            div_config = {}
        div_config.setdefault('agent_id', '0')
        div_config.setdefault('division_threshold', 'mass_distribution')
        dry_mass_inc = getattr(getattr(loader, 'sim_data', None),
                               'expectedDryMassIncreaseDict', {})
        from v2ecoli.library.unit_bridge import unum_to_pint
        dry_mass_inc = {k: unum_to_pint(v) for k, v in dry_mass_inc.items()}
        div_config.setdefault('dry_mass_inc_dict', dry_mass_inc)
        if hasattr(loader, '_configs'):
            div_config['configs'] = loader._configs
        div_config.setdefault('unique_names', getattr(loader, 'unique_names', []))
        div_config.setdefault('cache_dir', getattr(loader, 'cache_dir', 'out/cache'))
        instance = _make_instance(Division, div_config, core)
        topo = {
            'bulk': ('bulk',),
            'unique': ('unique',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'boundary': ('boundary',),
            'global_time': ('global_time',),
            'division_threshold': ('division_threshold',),
            'media_id': ('environment', 'media_id'),
            'agents': ('..',),
        }
        return instance, topo, 'step'

    return None
