# v2ecoli/workflow/flush.py
"""The post-simulation analysis flush: extract a finished run once, then dispatch
to the unified POST_SIM_REGISTRY and place each output where the report renders
it. Plan 1 wires the report_card + visualization kinds; analyses keep their
existing run_analyses path (folded in by Plan 2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from v2ecoli.workflow.report_cards import StudyContext

_STUDIES_RE = re.compile(r"(?:^|/)studies/([A-Za-z0-9_.\-]+)(?:/|$)")


def resolve_owning_study(out_dir: str, config: dict, ws_root) -> "str | None":
    """The study slug this run belongs to: config['study'] if set, else a
    studies/<slug>/ segment in out_dir, else None. Only returns a slug whose
    studies/<slug>/study.yaml exists under ws_root."""
    ws_root = Path(ws_root)
    slug = (config or {}).get("study")
    if not slug:
        m = _STUDIES_RE.search(str(out_dir).replace("\\", "/"))
        slug = m.group(1) if m else None
    if not slug:
        return None
    if (ws_root / "workspace" / "studies" / slug / "study.yaml").is_file():
        return slug
    return None


class RunExtract:
    """Lazy extraction context for a finished run. Heavy bits (DuckDB conn +
    sim_data) are provisioned only when conn_ctx()/records() are called."""

    def __init__(self, out_dir: str, config: dict, ws_root):
        self.out_dir = str(out_dir)
        self.config = config or {}
        self.ws_root = Path(ws_root)
        self.study_slug = resolve_owning_study(out_dir, config, ws_root)
        self._ctx: dict[str, Any] = {}
        self._records = None

    def study_ctx(self) -> "StudyContext | None":
        if not self.study_slug:
            return None
        return StudyContext.load(self.ws_root, self.study_slug)

    def study_viz_dir(self) -> "Path | None":
        if not self.study_slug:
            return None
        return self.ws_root / "workspace" / "studies" / self.study_slug / "viz"

    def records(self) -> list:
        if self._records is None:
            from v2ecoli.workflow.analysis_runner import build_cell_records
            self._records = list(build_cell_records(self.out_dir).values())
        return self._records

    def conn_ctx(self) -> tuple:
        if not self._ctx:
            import duckdb
            from v2ecoli.workflow.analysis_runner import (
                _history_from_clause, resolve_sim_data, resolve_validation_data)
            self._ctx["conn"] = duckdb.connect()
            self._ctx["from_clause"] = _history_from_clause(self.out_dir)
            self._ctx["sim_data"] = resolve_sim_data(self.out_dir)
            self._ctx["validation_data"] = resolve_validation_data(self._ctx["sim_data"])
        return (self._ctx["conn"], self._ctx["from_clause"],
                self._ctx["sim_data"], self._ctx["validation_data"])

    def context_bag(self) -> dict:
        """The full provisioning bag the flush filters by each step's inputs().
        `study` is eager (cheap); `conn`/`sim_data`/`history_sql`/`records` are
        lazy CALLABLES so a step that doesn't declare them never triggers the
        heavy extraction."""
        return {
            "study": self.study_ctx(),
            "out_dir": self.out_dir,
            "conn": None,
            "history_sql": "",
            "config_sql": "",
            "success_sql": "",
            "sim_data": None,
            "validation_data": None,
            "_conn_ctx": self.conn_ctx,   # callable: () -> (conn, from_clause, sim_data, validation_data)
            "_records": self.records,     # callable: () -> records
        }

    def close(self) -> None:
        conn = self._ctx.get("conn")
        if conn is not None:
            conn.close()
        self._ctx = {}


def _write_html(path, html: str):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


def place_output(kind: str, name: str, view: str, data: dict,
                 extract: "RunExtract") -> "str | None":
    """Route one step's output to the owning study's report location by kind.
    Returns the written html path (str) or None if nothing was written."""
    if not view:
        return None
    if kind == "report_card":
        from v2ecoli.workflow.report_cards import write_card
        ctx = extract.study_ctx()
        if ctx is not None:
            return str(write_card(ctx, name, data or {}, view))
        # no study: drop the card next to the run so it is not lost
        return str(_write_html(Path(extract.out_dir) / "viz" / "report_card" / f"{name}.html", view))
    # visualization / analysis view
    viz = extract.study_viz_dir() or (Path(extract.out_dir) / "viz")
    return str(_write_html(viz / f"{name}.html", view))


def _run_one_step(cls, kind, extract, core):
    """Instantiate + run one post-sim step; return (view, data). For report
    cards we call build()/applies() directly (their native API); visualizations
    and analyses go through update() with an inputs()-filtered bag."""
    step = cls({}, core=core)
    bag = extract.context_bag()
    inputs = {}
    try:
        inputs = step.inputs() or {}
    except Exception:  # noqa: BLE001
        inputs = {}
    # report cards: skip when applies() is False; build() returns (verdict, html)
    if kind == "report_card":
        ctx = bag.get("study")
        if ctx is None or not step.applies(ctx):
            return "", {}
        res = step.build(ctx)
        if not res:
            return "", {}
        verdict, html = res
        return html, verdict
    # analyses/visualizations declaring DuckDB inputs get the lazy conn ctx
    state = {}
    for key in inputs:
        if key in ("conn", "history_sql", "sim_data", "validation_data") and bag.get("conn") is None:
            conn, from_clause, sim_data, validation_data = bag["_conn_ctx"]()
            state.update({"conn": conn, "history_sql": from_clause,
                          "sim_data": sim_data, "validation_data": validation_data})
        elif key in bag:
            state[key] = bag[key]
    out = step.update(state) or {}
    return out.get("view", ""), out.get("data", {}) or {}


def _flush_analyses(extract: "RunExtract", config: dict) -> tuple:
    """Run the configured analyses over the run, then copy their outputs into the
    owning study's report dir. Returns (placed, skipped). Empty analysis_options
    -> ([], []). Any failure -> ([], [{"name":"analyses","error":...}])."""
    analysis_options = (config or {}).get("analysis_options") or {}
    if not any(analysis_options.values()):
        return [], []
    try:
        from v2ecoli.workflow.analysis_runner import run_analyses
        run_analyses(extract.out_dir, analysis_options)
        return place_analysis_outputs(extract), []
    except Exception as e:  # noqa: BLE001 — never abort the flush
        return [], [{"name": "analyses", "error": f"{type(e).__name__}: {e}"}]


def run_flush(out_dir, config, ws_root, *, core=None,
              kinds=("analysis", "report_card", "visualization")) -> dict:
    """Dispatch the registered post-sim steps of the given kinds over a finished
    run and place each output where the study report renders it. Graceful
    per-step skip. The 'analysis' kind calls _flush_analyses (not the per-step
    iter_post_sim loop, which is for report_card/visualization)."""
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.post_sim import iter_post_sim
    if core is None:
        core = allocate_core()
    extract = RunExtract(out_dir, config, ws_root)
    placed, skipped = [], []
    try:
        for kind in kinds:
            if kind == "analysis":
                a_placed, a_skipped = _flush_analyses(extract, config)
                placed.extend(a_placed)
                skipped.extend(a_skipped)
                continue
            for name, cls in iter_post_sim(kind):
                try:
                    view, data = _run_one_step(cls, kind, extract, core)
                    path = place_output(kind, name, view, data, extract)
                except Exception as e:  # noqa: BLE001 — one step never aborts the flush
                    skipped.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                    continue
                if path:
                    placed.append({"kind": kind, "name": name, "path": path})
    finally:
        extract.close()
    return {"placed": placed, "skipped": skipped, "study": extract.study_slug}


def place_analysis_outputs(extract: "RunExtract") -> list:
    """Copy a finished run's analysis artifacts into the owning study's report
    dir so the study report surfaces them: <out_dir>/viz/*.html ->
    <study viz>/<stem>.html and <out_dir>/ptools/*.tsv -> <study>/ptools/<stem>.tsv.
    Returns [{"kind":"analysis","name":<stem>,"path":<dest>}] per copied html.
    No owning study -> returns [] (the run's out_dir stays the home)."""
    import shutil

    study_viz = extract.study_viz_dir()
    if study_viz is None:
        return []
    placed = []
    src_viz = Path(extract.out_dir) / "viz"
    if src_viz.is_dir():
        study_viz.mkdir(parents=True, exist_ok=True)
        for html in sorted(src_viz.glob("*.html")):
            dest = study_viz / html.name
            shutil.copyfile(html, dest)
            placed.append({"kind": "analysis", "name": html.stem, "path": str(dest)})
    src_ptools = Path(extract.out_dir) / "ptools"
    if src_ptools.is_dir():
        study_ptools = study_viz.parent / "ptools"
        study_ptools.mkdir(parents=True, exist_ok=True)
        for tsv in sorted(src_ptools.glob("*.tsv")):
            shutil.copyfile(tsv, study_ptools / tsv.name)
    return placed
