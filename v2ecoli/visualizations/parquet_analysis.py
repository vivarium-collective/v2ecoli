"""Visualization Step that renders a native v2ecoli analysis from a run's
hive-partitioned parquet history.

The Composite Explorer's single-cell run leaves a per-agent, agent-rooted
parquet sweep (installed by vivarium_workbench's
``inject_analysis_parquet_emitters``) whose ``__``-flattened columns and
``variant/lineage_seed/generation/agent_id`` hive partitions match what the
v2ecoli native analyses (``v2ecoli/workflow/analyses/``) consume.  This adapter
runs ONE named analysis over that sweep and returns its Altair / matplotlib
HTML, so the real figures (mass fractions, cell mass, replication, ribosome
usage, ...) render in the Visualizations tab instead of the generic observable
line plots.

Config (the dashboard's ``_render_canonical_viz`` injects ``sweep_dir`` and
``sim_data_path`` per run, mirroring how it injects ``runs_db_path``):

  analysis       registry name, e.g. ``"mass_fraction_summary_view"``
  sweep_dir      ``<run_dir>/parquet/<run_id>`` (the dir holding ``history/``)
  sim_data_path  path to a ParCa ``parca_state.pkl.gz`` (gzipped state)
  title          panel title (display only)

The class is intentionally tolerant: any missing prerequisite (no sweep yet, no
sim_data, an analysis that needs more cells than a one-off run produced) returns
an explanatory ``{"html": ...}`` panel rather than raising — a broken figure
must not sink the whole Visualizations render.
"""

from __future__ import annotations

import html as _html

from viva_superpowers.visualization import Visualization

# Module-level caches so rendering N panels for one run loads the (large) ParCa
# sim_data pickle + validation data ONCE, and imports the analysis registry once.
_SIM_DATA_CACHE: dict = {}
_REGISTRY_LOADED = False


def _load_registry() -> None:
    """Import every native-analysis module so ``ANALYSIS_REGISTRY`` is populated
    (each ``Analysis`` subclass registers on import). Idempotent."""
    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    import importlib
    import pkgutil
    import v2ecoli.workflow.analyses as anpkg
    for mod in pkgutil.iter_modules(anpkg.__path__):
        if not mod.name.startswith("_"):
            try:
                importlib.import_module(f"v2ecoli.workflow.analyses.{mod.name}")
            except Exception:
                # An analysis with an unmet optional dep just isn't available;
                # the others still register.
                pass
    _REGISTRY_LOADED = True


def _sim_data_for(sim_data_path: str):
    """Hydrate + cache (sim_data, validation_data) for a gzipped ParCa state.

    ``run_analyses``'s own loader (``LoadSimData``) does a raw ``pickle.load``
    and chokes on the gzip magic byte, so we use the state loader
    (``load_parca_state`` + ``hydrate_sim_data_from_state``) that the showcase
    render driver uses.
    """
    if sim_data_path in _SIM_DATA_CACHE:
        return _SIM_DATA_CACHE[sim_data_path]
    from v2ecoli.processes.parca.data_loader import (
        load_parca_state, hydrate_sim_data_from_state,
    )
    from v2ecoli.workflow.analysis_runner import resolve_validation_data
    sim_data = hydrate_sim_data_from_state(load_parca_state(sim_data_path))
    try:
        validation_data = resolve_validation_data(sim_data)
    except Exception:
        validation_data = None
    _SIM_DATA_CACHE[sim_data_path] = (sim_data, validation_data)
    return sim_data, validation_data


def _panel(title: str, body_html: str) -> str:
    """Wrap a message (or a rendered view) with the panel title."""
    return f"<div class='pqav-panel'>{body_html}</div>"


def _note_panel(title: str, message: str) -> str:
    """A panel explaining why an analysis rendered no figure. Same shape as
    :func:`_note` but returned as HTML, for the render path that must hand back
    a view string rather than a step-output dict."""
    esc = _html.escape(f"{title}: {message}")
    return _panel(
        title,
        f"<p style='color:#6b7280;font-size:0.9em;padding:8px 4px'>{esc}</p>",
    )


def _note(title: str, message: str) -> dict:
    esc = _html.escape(message)
    return {"html": _panel(
        title,
        f"<p style='color:#6b7280;font-size:0.9em;padding:8px 4px'>{esc}</p>",
    )}


class ParquetAnalysisView(Visualization):
    """Render one native v2ecoli analysis from a run's parquet sweep."""

    config_schema = {
        **Visualization.config_schema,
        "analysis":      {"_type": "string", "_default": ""},
        "sweep_dir":     {"_type": "string", "_default": ""},
        "sim_data_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        # Self-contained: the class reads the parquet sweep itself.
        return {}

    def outputs(self) -> dict:
        return {}

    # Rendered in one shot (legacy Visualization contract): override update()
    # directly so the base orchestrator stays out of the way.
    def update(self, state: dict) -> dict:
        cfg = dict(self.config)
        name = cfg.get("analysis") or ""
        title = cfg.get("title") or name or "Analysis"
        sweep_dir = cfg.get("sweep_dir") or ""
        sim_data_path = cfg.get("sim_data_path") or ""

        if not name:
            return _note(title, "No analysis name configured.")
        if not sweep_dir:
            return _note(title, "Run this composite to generate the parquet "
                                "history this figure reads.")

        try:
            return {"html": _panel(title, self._render(name, sweep_dir,
                                                        sim_data_path))}
        except Exception as exc:  # never sink the whole Visualizations render
            return _note(title, f"Could not render {name}: "
                                f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    def _render(self, name: str, sweep_dir: str, sim_data_path: str) -> str:
        import glob
        import os

        # Single source of truth for "which history parquet" — hive-scoped, so a
        # stray non-hive default/history/*.pq neither passes this gate on its own
        # nor reaches the duckdb hive read below (via _history_from_clause).
        from v2ecoli.library.sweep_io import history_files

        hist = history_files(sweep_dir)
        if not hist:
            raise FileNotFoundError(
                "no parquet history under the run's sweep dir yet")

        _load_registry()
        from bigraph_schema import allocate_core
        from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
        from v2ecoli.workflow.analysis_runner import (
            _history_from_clause, group_for_scale, scale_history_sql,
            build_cell_records,
        )
        import duckdb

        cls = ANALYSIS_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"analysis {name!r} not in registry")
        scale = getattr(cls, "scale", "single")

        records = list(build_cell_records(sweep_dir).values())
        if not records:
            raise ValueError("no cells found in the run's parquet history")
        groups = group_for_scale(scale, records)
        if not groups:
            raise ValueError(f"no {scale} groups for this run")
        gkey = next(iter(groups))

        sim_data = validation_data = None
        if not sim_data_path:
            # The caller (the workbench run renderer) looks for a ParCa build in
            # a few fixed workspace locations and passes "" when it finds none.
            # A sweep that paired itself with its own sim_data — every batch run
            # does — carries the exact pickle right here, so prefer it over
            # rendering the sim_data-dependent analyses empty.
            local = sorted(glob.glob(os.path.join(sweep_dir, "simData*.cPickle"))
                           + glob.glob(os.path.join(sweep_dir, "sim_data*.cPickle")))
            sim_data_path = local[0] if local else ""
        if sim_data_path:
            sim_data, validation_data = _sim_data_for(sim_data_path)

        core = getattr(self, "core", None) or allocate_core()
        conn = duckdb.connect()
        from_clause = _history_from_clause(sweep_dir)
        history_sql = scale_history_sql(scale, from_clause, gkey)

        step = cls({}, core=core)
        out = step.update({
            "conn": conn, "history_sql": history_sql,
            "config_sql": "", "success_sql": "",
            "sim_data": sim_data, "validation_data": validation_data,
            "variant_metadata": {},
        })
        view = out.get("view") if isinstance(out, dict) else None
        if view:
            return view
        # No figure. Three different situations, and reporting them all as
        # "analysis produced no view" made a run's Visualizations tab look
        # broken when nothing was wrong: the analysis said WHY it declined (an
        # `error` in its data), or it is a data-only analysis (a table export
        # with no plot), or the run's shape simply doesn't cover it (asking for
        # growth ACROSS generations of a single-generation batch). Render the
        # reason as a panel instead — the same analysis renders normally on a
        # run that does cover it.
        data = out.get("data") if isinstance(out, dict) else None
        reason = ""
        if isinstance(data, dict):
            reason = data.get("error") or ""
        if reason:
            return _note_panel(name, reason)
        if isinstance(data, dict) and data:
            return _note_panel(
                name,
                f"produced data ({', '.join(sorted(data)[:6])}) but no figure — "
                f"a data-only analysis at the {scale} scale.")
        return _note_panel(
            name,
            f"produced nothing for this run: it reads {scale}-scale groups, "
            f"which this run may not have enough cells or generations to fill.")
