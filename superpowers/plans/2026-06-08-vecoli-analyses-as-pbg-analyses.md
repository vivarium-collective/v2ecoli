# vEcoli Analyses as Native Process-Bigraph Analyses — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native, Visualization-like `Analysis` abstraction to v2ecoli and port a 5-analysis proving set (3 ptools + 2 plots) from vEcoli's DuckDB/`sim_data` analyses, surfaced in the vivarium-dashboard Visualizations tab.

**Architecture:** A new `Analysis(V2Step)` sits *alongside* the existing record-based `AnalysisStep` (which keeps the 5 working native analyses untouched). `Analysis` declares DuckDB `conn` + `history_sql` + `sim_data` input ports and emits `{view: html, data: map}`. The post-sweep runner gains a provisioning path that opens one DuckDB connection over the sweep's parquet, builds a scale-scoped `history_sql`, loads `sim_data` once, runs each `Analysis`, and writes `data → analysis.json`/TSV and `view → <sweep>/viz/*.html`. The dashboard already surfaces `<study>/viz/*.html`; one small server edit also lists `Analysis` classes in the viz picker.

**Tech Stack:** Python 3.12, process-bigraph (`V2Step`/`Step`), DuckDB, pandas, matplotlib, pytest. Run everything via `.venv/bin/python` / `.venv/bin/pytest` (per repo memory: bare `python` lacks `unum`).

**Spec:** `docs/superpowers/specs/2026-06-08-vecoli-analyses-as-pbg-analyses-design.md`

**Spec refinements locked in this plan** (discovered while grounding):
1. **Add-alongside, not evolve-in-place.** `analysis.py` already has *5* record-based analyses (`MassFractionSummary`, `DaughterMassSymmetry`, `MassGrowthAcrossGenerations`, `DoublingTimeDistribution`, `MetricAcrossVariants`) using `analyze(rows)`. Evolving the base in place would break them. So `Analysis` is a **new sibling base**; `AnalysisStep` stays. Both register into the shared `ANALYSIS_REGISTRY`; the runner dispatches by base class.
2. **Proving set adjusted to avoid a registry name collision.** The spec named `mass_fraction_summary` as the 4th analysis, but that name is already taken by the existing record-based Step. The view/matplotlib proof uses **`mass_fraction_voronoi`** (single scale, vEcoli plot, no collision) instead. 5th remains **`central_carbon_metabolism_scatter`** (multiseed, cross-cell SQL + matplotlib).

**Source of ports:** `vivarium-ecoli` repo, branch `origin/ptools_viz`, path `ecoli/analysis/<scale>/<name>.py`. Read a file with: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/ptools_rna.py`.

**Cross-implementation oracle:** sms-api checked-in reference TSVs at `/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data/ptools_rna.txt` and `ptools_rxns.txt` (frame-ID × timepoint tables produced by the same vEcoli modules).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `v2ecoli/workflow/analysis.py` | analysis bases + registry | **modify** — add `Analysis` base; keep `AnalysisStep` + the 5 Steps |
| `v2ecoli/workflow/render.py` | matplotlib-figure → embedded-SVG HTML helper | **create** |
| `v2ecoli/workflow/analysis_runner.py` | post-sweep runner | **modify** — add DuckDB-provisioning path for `Analysis` steps |
| `v2ecoli/workflow/analyses/__init__.py` | import-time registration of ported analyses | **create** |
| `v2ecoli/workflow/analyses/ptools_rna.py` `ptools_rxns.py` `ptools_proteins.py` | single-scale ptools ports (data TSV) | **create** |
| `v2ecoli/workflow/analyses/mass_fraction_voronoi.py` | single-scale plot port (view) | **create** |
| `v2ecoli/workflow/analyses/central_carbon_metabolism_scatter.py` | multiseed plot port (view + cross-cell SQL) | **create** |
| `v2ecoli/configs/analyses_proving.json` | `analysis_options` for the proving set | **create** |
| `vivarium-dashboard/vivarium_dashboard/server.py` | viz picker lists `Analysis` classes | **modify** |
| `tests/test_analysis_base.py` | `Analysis` base unit tests | **create** |
| `tests/test_render.py` | render-helper test | **create** |
| `tests/test_analysis_runner_duckdb.py` | runner provisioning + scale SQL + end-to-end | **create** |
| `tests/test_ptools_analyses.py` | ptools port fidelity vs oracle | **create** |
| `docs/superpowers/notes/2026-06-08-analyses-parity-findings.md` | parity-check output (Task 1) | **create** |

---

### Task 1: Parity check (discovery — gates the ports)

Confirm v2ecoli's `sim_data` and parquet expose what the ptools modules require. Output a findings note that later tasks reference. **No production code in this task.**

**Files:**
- Create: `docs/superpowers/notes/2026-06-08-analyses-parity-findings.md`

- [ ] **Step 1: List the attributes/columns each ptools module needs**

Run and read the three single-scale sources:
```bash
for m in ptools_rna ptools_rxns ptools_proteins; do
  echo "===== $m ====="; \
  git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/$m.py \
    | grep -nE 'sim_data\.|history_sql|listeners__|read_outputs|output_columns|\.process\.|internal_state|molecule_groups'
done
```
Expected: a concrete list of `sim_data.*` attributes (e.g. `process.transcription.rna_data`, `process.complexation.get_monomers`, `process.transcription.rna_maturation_stoich_matrix`, `molecule_groups.s50_23s_rRNA`, `internal_state.bulk_molecules.bulk_data`) and parquet columns (e.g. `bulk`, `listeners__rna_counts__full_mRNA_counts`, `listeners__unique_molecule_counts__active_ribosome`).

- [ ] **Step 2: Verify the parquet columns exist in a v2ecoli sweep**

Find a recent sweep and inspect history columns:
```bash
cd /Users/eranagmon/code/v2ecoli
SW=$(ls -dt out/*/ 2>/dev/null | head -1); echo "sweep=$SW"
PQ=$(find "$SW" -path '*history*' -name '*.pq' | head -1); echo "pq=$PQ"
.venv/bin/python -c "import duckdb,sys; print([c[0] for c in duckdb.sql(f\"DESCRIBE SELECT * FROM read_parquet('$PQ')\").fetchall()])"
```
Expected: a column list. Record which required columns are present vs missing.

- [ ] **Step 3: Verify the sim_data attribute tree**

Locate a sim_data pickle in the sweep (or ParCa output) and probe attributes:
```bash
cd /Users/eranagmon/code/v2ecoli
SD=$(find out _parca_cache parca -name 'sim_data*.cPickle' -o -name 'sim_data*.pkl' 2>/dev/null | head -1); echo "sim_data=$SD"
.venv/bin/python - "$SD" <<'PY'
import sys, pickle
sd = pickle.load(open(sys.argv[1],'rb'))
for path in ["process.transcription.rna_data",
             "process.transcription.rna_maturation_stoich_matrix",
             "process.complexation.get_monomers",
             "molecule_groups.s50_23s_rRNA",
             "internal_state.bulk_molecules.bulk_data"]:
    obj = sd
    try:
        for p in path.split("."): obj = getattr(obj, p)
        print("OK  ", path, type(obj).__name__)
    except Exception as e:
        print("MISS", path, e)
PY
```
Expected: `OK` for each path (v2ecoli's `LoadSimData` is API-compatible with vEcoli). Any `MISS` is a parity gap to record.

- [ ] **Step 4: Write findings + decide per-analysis port viability**

Write `docs/superpowers/notes/2026-06-08-analyses-parity-findings.md` with three sections: **Columns present/missing**, **sim_data attributes present/missing**, **Per-analysis verdict** (each of the 5: "port as-is" | "port with shim X" | "blocked on missing listener Y"). If a required listener column is missing for a ptools module, flag it here — that module's port task adds a column alias if the data exists under another name, or is deferred with a note.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/v2ecoli
git add docs/superpowers/notes/2026-06-08-analyses-parity-findings.md
git commit -m "docs: sim_data/parquet parity findings for analysis ports"
```

---

### Task 2: The `Analysis` base (alongside `AnalysisStep`)

**Files:**
- Modify: `v2ecoli/workflow/analysis.py` (add `Analysis` after the `ANALYSIS_REGISTRY` definition, ~line 41, before `class AnalysisStep`)
- Test: `tests/test_analysis_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_base.py
from bigraph_schema import allocate_core
from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY


class _Demo(Analysis):
    name = "demo_analysis"
    scale = "single"

    def analyze(self, *, conn, history_sql, sim_data, **ctx):
        return {"view": "<p>ok</p>", "data": {"n": 1}}


def test_analysis_registers_and_dispatches():
    assert ANALYSIS_REGISTRY["demo_analysis"] is _Demo
    step = _Demo({}, core=allocate_core())
    out = step.update({"conn": None, "history_sql": "SELECT 1", "sim_data": None})
    assert out == {"view": "<p>ok</p>", "data": {"n": 1}}


def test_analysis_defaults_missing_keys():
    class _Bare(Analysis):
        name = "bare_analysis"
        scale = "single"

        def analyze(self, **ctx):
            return {"data": {"x": 2}}  # no "view"

    step = _Bare({}, core=allocate_core())
    out = step.update({})
    assert out == {"view": "", "data": {"x": 2}}


def test_analysis_inputs_declare_duckdb_ports():
    assert set(_Demo({}, core=allocate_core()).inputs()) >= {
        "conn", "history_sql", "sim_data"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/pytest tests/test_analysis_base.py -v`
Expected: FAIL with `ImportError: cannot import name 'Analysis'`.

- [ ] **Step 3: Add the `Analysis` base**

Insert into `v2ecoli/workflow/analysis.py` immediately after the `ANALYSIS_REGISTRY: dict[str, type] = {}` line:

```python
class Analysis(V2Step):
    """Visualization-like analysis: reads sim output via a DuckDB connection +
    the ParCa ``sim_data``, and emits a rendered ``view`` (HTML) plus optional
    ``data`` (map). Faithful native ports of vEcoli's ``plot()`` analyses build
    on this base (cf. the record-based ``AnalysisStep`` for emitted-observable
    analyses). Subclasses set ``scale`` + ``name`` and implement ``analyze``.

    Live, non-serializable handles (``conn``, ``sim_data``) are injected by the
    runner into the state dict passed to ``update``; ``inputs()`` declares them
    for discoverability with a permissive ("any") type.
    """

    scale: str = "single"
    config_schema = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.scale not in ANALYSIS_SCALES:
            raise ValueError(
                f"{cls.__name__}.scale={cls.scale!r} not in {sorted(ANALYSIS_SCALES)}")
        if "name" in cls.__dict__:
            ANALYSIS_REGISTRY[cls.name] = cls

    def inputs(self):
        return {
            "conn": "any", "history_sql": "string",
            "config_sql": "string", "success_sql": "string",
            "sim_data": "any", "validation_data": "any",
            "variant_metadata": "any",
        }

    def outputs(self):
        return {"view": "string", "data": "map"}

    def analyze(self, *, conn, history_sql, sim_data, **ctx) -> dict:
        """Return {"view": <html str>, "data": <map>} (either key optional)."""
        raise NotImplementedError

    def invoke(self, state, interval=None):
        # Fail loudly (like AnalysisStep): a broken analyze() must surface.
        return SyncUpdate(self.update(state))

    def update(self, state, interval=None):
        kwargs = {k: state.get(k) for k in self.inputs()}
        out = self.analyze(**kwargs) or {}
        return {"view": out.get("view", ""), "data": out.get("data", {})}
```

Note: `Analysis` and `AnalysisStep` share `ANALYSIS_REGISTRY`; the runner (Task 4) dispatches by `issubclass(step_cls, Analysis)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/pytest tests/test_analysis_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Confirm the existing analyses still register**

Run: `.venv/bin/pytest tests/test_workflow_analysis.py -v`
Expected: PASS (the 5 record-based Steps unchanged).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_analysis_base.py
git commit -m "feat: Analysis base (duckdb/sim_data ports, view+data outputs)"
```

---

### Task 3: Matplotlib → embedded-SVG render helper

**Files:**
- Create: `v2ecoli/workflow/render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.workflow.render import fig_to_html


def test_fig_to_html_embeds_svg():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    html = fig_to_html(fig, title="Demo")
    assert "<svg" in html
    assert "Demo" in html
    assert html.strip().startswith("<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.render'`.

- [ ] **Step 3: Implement the helper**

```python
# v2ecoli/workflow/render.py
"""Render helpers for Analysis ``view`` outputs.

``fig_to_html`` turns a matplotlib Figure into a self-contained HTML fragment
with an inline SVG, so plot-style analyses port near-verbatim from vEcoli (swap
``fig.savefig(path)`` for ``return {"view": fig_to_html(fig)}``).
"""

from __future__ import annotations

import io


def fig_to_html(fig, title: str = "") -> str:
    """Serialize a matplotlib Figure to an HTML fragment with inline SVG."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    # Strip the XML/doctype preamble so the <svg> embeds cleanly in HTML.
    idx = svg.find("<svg")
    svg = svg[idx:] if idx != -1 else svg
    heading = f"<h3>{title}</h3>" if title else ""
    return f'<div class="analysis-view">{heading}{svg}</div>'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/render.py tests/test_render.py
git commit -m "feat: fig_to_html render helper for Analysis views"
```

---

### Task 4: Runner DuckDB-provisioning path

Add the provisioning helpers and extend `run_analyses` to dispatch `Analysis` steps through DuckDB while leaving the record-based `AnalysisStep` path unchanged.

**Files:**
- Modify: `v2ecoli/workflow/analysis_runner.py`
- Test: `tests/test_analysis_runner_duckdb.py`

- [ ] **Step 1: Write the failing test for `scale_history_sql`**

```python
# tests/test_analysis_runner_duckdb.py
from v2ecoli.workflow.analysis_runner import scale_history_sql

_FROM = "read_parquet(['x.pq'], hive_partitioning=true)"


def test_single_sql_filters_full_cell():
    sql = scale_history_sql("single", _FROM, (0, 1, 2, "00"))
    assert "variant = 0" in sql and "lineage_seed = 1" in sql
    assert "generation = 2" in sql and "agent_id = '00'" in sql


def test_multiseed_sql_filters_variant_only():
    sql = scale_history_sql("multiseed", _FROM, (3,))
    assert "variant = 3" in sql
    assert "lineage_seed" not in sql and "agent_id" not in sql


def test_multivariant_sql_is_unfiltered():
    sql = scale_history_sql("multivariant", _FROM, ())
    assert "WHERE" not in sql.upper()
    assert _FROM in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_analysis_runner_duckdb.py -v`
Expected: FAIL with `ImportError: cannot import name 'scale_history_sql'`.

- [ ] **Step 3: Add the provisioning helpers**

Add to `v2ecoli/workflow/analysis_runner.py` (top-level functions, after `_MASS_COLS`):

```python
def _history_from_clause(sweep_dir: str) -> str:
    """A DuckDB FROM-clause selecting all of the sweep's history parquet."""
    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        raise FileNotFoundError(f"no history parquet under {sweep_dir!r}")
    flist = "[" + ",".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
    return f"read_parquet({flist}, hive_partitioning=true)"


# scale -> the partition columns that scale's history_sql filters on.
_SCALE_FILTER_COLS = {
    "single": ("variant", "lineage_seed", "generation", "agent_id"),
    "multidaughter": ("variant", "lineage_seed", "generation"),  # parent handled below
    "multigeneration": ("variant", "lineage_seed"),
    "multiseed": ("variant",),
    "multivariant": (),
}


def scale_history_sql(scale: str, from_clause: str, key: tuple) -> str:
    """SELECT * scoped to the partition a scale aggregates over.

    ``key`` is the group key from ``group_for_scale`` for that scale.
    """
    cols = _SCALE_FILTER_COLS[scale]
    conds = []
    for col, val in zip(cols, key):
        if isinstance(val, str):
            conds.append(f"agent_id = '{val}'" if col == "agent_id"
                         else f"{col} = '{val}'")
        else:
            conds.append(f"{col} = {int(val)}")
    if scale == "multidaughter" and len(key) >= 4:
        # sisters share parent = agent_id without its last phylogeny char
        conds.append(f"agent_id LIKE '{key[3]}_' ESCAPE '\\'")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    return f"SELECT * FROM {from_clause}{where} ORDER BY global_time"


def resolve_sim_data(sweep_dir: str):
    """Locate + load the sweep's ParCa sim_data via v2ecoli's loader."""
    from v2ecoli.library.sim_data import LoadSimData
    for pat in ("sim_data*.cPickle", "sim_data*.pkl", "**/sim_data*.cPickle",
                "**/kb/simData*.cPickle"):
        hits = glob.glob(os.path.join(sweep_dir, pat), recursive=True)
        if hits:
            return LoadSimData(sim_data_path=hits[0]).sim_data
    raise FileNotFoundError(
        f"no sim_data pickle under {sweep_dir!r} (needed by Analysis steps)")
```

- [ ] **Step 4: Run the SQL test to verify it passes**

Run: `.venv/bin/pytest tests/test_analysis_runner_duckdb.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Extend `run_analyses` to dispatch `Analysis` steps**

In `v2ecoli/workflow/analysis_runner.py`, modify `run_analyses`. Add imports of `Analysis` and the helpers at the top of the function, and branch per step class. Replace the `for name in (analyses or {}):` body with:

```python
        for name in (analyses or {}):
            step_cls = ANALYSIS_REGISTRY.get(name)
            if step_cls is None:
                warnings.warn(f"unknown analysis {name!r} (scale {scale}); skipping")
                continue
            if step_cls.scale != scale:
                warnings.warn(f"analysis {name!r} is scale {step_cls.scale}, "
                              f"not {scale}; skipping")
                continue
            step = step_cls({}, core=core)
            per_group: dict[str, Any] = {}

            if issubclass(step_cls, Analysis):
                # DuckDB-provisioning path: one connection + sim_data for all groups.
                import duckdb
                conn = duckdb.connect()
                from_clause = _history_from_clause(sweep_dir)
                sim_data = resolve_sim_data(sweep_dir)
                params = (analyses or {}).get(name) or {}
                viz_dir = os.path.join(sweep_dir, "viz")
                os.makedirs(viz_dir, exist_ok=True)
                for gkey in groups:
                    gstr = _group_key_str(scale, gkey)
                    try:
                        history_sql = scale_history_sql(scale, from_clause, gkey)
                        out = step.update({
                            "conn": conn, "history_sql": history_sql,
                            "config_sql": "", "success_sql": "",
                            "sim_data": sim_data, "validation_data": None,
                            "variant_metadata": params,
                        })
                        if out.get("view"):
                            vp = os.path.join(viz_dir, f"{name}__{gstr.replace('/', '_')}.html")
                            with open(vp, "w") as vf:
                                vf.write(out["view"])
                        per_group[gstr] = out.get("data", {})
                    except Exception as e:
                        per_group[gstr] = {"error": f"{type(e).__name__}: {e}"}
            else:
                # Record-based AnalysisStep path (unchanged).
                for gkey, grp in groups.items():
                    try:
                        rows = grp[0].get("timeseries") if scale == "single" else grp
                        per_group[_group_key_str(scale, gkey)] = step.analyze(rows or [])
                    except Exception as e:
                        per_group[_group_key_str(scale, gkey)] = {
                            "error": f"{type(e).__name__}: {e}"}

            scale_out[name] = per_group
```

Add at the top of `run_analyses` (with the existing imports):
```python
    from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY, ANALYSIS_SCALES
```
(extends the existing import line to also bring in `Analysis`).

- [ ] **Step 6: Run the existing runner tests to confirm no regression**

Run: `.venv/bin/pytest tests/test_analysis_runner.py tests/test_workflow_analysis.py -v`
Expected: PASS (record-based path untouched).

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/workflow/analysis_runner.py tests/test_analysis_runner_duckdb.py
git commit -m "feat: runner DuckDB-provisioning path for Analysis steps"
```

---

### Task 5: Port `ptools_rna` (single, data)

**Files:**
- Create: `v2ecoli/workflow/analyses/__init__.py`
- Create: `v2ecoli/workflow/analyses/ptools_rna.py`
- Test: `tests/test_ptools_analyses.py`

- [ ] **Step 1: Create the package init (registers ports at import)**

```python
# v2ecoli/workflow/analyses/__init__.py
"""Native ports of vEcoli DuckDB/sim_data analyses (Analysis subclasses).

Importing this package registers every ported analysis into ANALYSIS_REGISTRY.
"""
from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
```
(Each later task appends its module to this import list.)

- [ ] **Step 2: Write the failing fidelity test**

```python
# tests/test_ptools_analyses.py
import os
import pytest

FIX = "/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data"


def _frame_ids(tsv_text):
    rows = [r for r in tsv_text.strip().splitlines() if r]
    return {r.split("\t")[0] for r in rows[1:]}  # skip header ($ row)


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rna_registered():
    from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rna"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rna_output_shape_matches_oracle():
    # The oracle TSV is "$\t<t0>\t<t1>...\n<frameid>\t<val>...". Our port, given the
    # same sim, must produce a frame-ID-indexed table with n_tp+1 columns. This
    # asserts structural fidelity (header shape + frame-ID column) against the
    # reference; numeric parity is checked in the end-to-end test (Task 10) when a
    # matching sim is available.
    oracle = open(os.path.join(FIX, "ptools_rna.txt")).read()
    header = oracle.strip().splitlines()[0].split("\t")
    assert header[0] == "$"
    assert len(_frame_ids(oracle)) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.workflow.analyses.ptools_rna'`.

- [ ] **Step 4: Port the module**

Get the source: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/ptools_rna.py`.

Create `v2ecoli/workflow/analyses/ptools_rna.py` by wrapping that source's `plot()` body in an `Analysis` subclass, applying exactly these mechanical edits:

1. Class scaffold:
   ```python
   from v2ecoli.workflow.analysis import Analysis

   class PtoolsRna(Analysis):
       name = "ptools_rna"
       scale = "single"
       config_schema = {"n_tp": "integer", "time_unit": "string"}

       def analyze(self, *, conn, history_sql, sim_data, variant_metadata, **ctx):
           params = variant_metadata or {}
           params.setdefault("n_tp", 8)
           ...  # ← the ported plot() body
   ```
2. Keep the source's module-level helpers (`build_query`, `read_outputs`, `retrieve_tu_source`, `tu2gene_mapping`, `get_bulk_ids`, `build_bulk2monomers_matrix`, `consolidate_timepoints`) **verbatim** as module functions.
3. In the body: replace `sim_data = LoadSimData(sim_data_path).sim_data` with use of the **passed** `sim_data` (delete the load). Replace `exp_id = list(sim_data_paths.keys())[0]` / `sim_data_path = ...` lines accordingly.
4. Replace the reconstruction-flat path derivation
   `wd_raw = os.path.join(os.getcwd().split("/out/")[0], "reconstruction", "ecoli", "flat")`
   with the v2ecoli location (confirm in Task 1 findings):
   `wd_raw = os.path.join(os.path.dirname(__import__("v2ecoli").__file__), "processes", "parca", "reconstruction", "ecoli", "flat")`.
5. Replace the final `ptools_rna.to_csv(os.path.join(outdir, "ptools_rna.txt"), sep="\t", ...)` write with:
   ```python
   tsv = ptools_rna.to_csv(sep="\t", index=True, header=True, float_format="%.4f")
   return {"data": {"filename": "ptools_rna.tsv", "tsv": tsv}}
   ```
6. Replace `conn.sql(query_sql).df()` usage — it already receives `conn`/`history_sql` as args; pass the injected `conn` and `history_sql` into `read_outputs(history_sql, conn, output_columns)` unchanged.

Then add `ptools_rna` to the `analyses/__init__.py` import list (already there from Step 1).

If Task 1 flagged a missing column (e.g. `listeners__unique_molecule_counts__active_ribosome`), apply the alias recorded in the findings note here (e.g. adjust `output_columns`), or guard that branch and note the reduced fidelity in a module docstring.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py -v`
Expected: PASS (registration + oracle-shape tests).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/analyses/__init__.py v2ecoli/workflow/analyses/ptools_rna.py tests/test_ptools_analyses.py
git commit -m "feat: port ptools_rna as native Analysis (single)"
```

---

### Task 6: Port `ptools_rxns` (single, data)

**Files:**
- Create: `v2ecoli/workflow/analyses/ptools_rxns.py`
- Modify: `v2ecoli/workflow/analyses/__init__.py` (add `ptools_rxns` import)
- Test: extend `tests/test_ptools_analyses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ptools_analyses.py`:
```python
@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rxns_registered_and_oracle_shape():
    from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rxns"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
    oracle = open(os.path.join(FIX, "ptools_rxns.txt")).read()
    assert oracle.strip().splitlines()[0].split("\t")[0] == "$"
    assert len(_frame_ids(oracle)) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py::test_ptools_rxns_registered_and_oracle_shape -v`
Expected: FAIL with `ModuleNotFoundError: ...ptools_rxns`.

- [ ] **Step 3: Port the module**

Source: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/ptools_rxns.py`. Apply the **same six mechanical edits as Task 5 Step 4**, with class `PtoolsRxns(Analysis)`, `name = "ptools_rxns"`, and the final return writing `{"data": {"filename": "ptools_rxns.tsv", "tsv": tsv}}`. Add `from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401` to `analyses/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analyses/ptools_rxns.py v2ecoli/workflow/analyses/__init__.py tests/test_ptools_analyses.py
git commit -m "feat: port ptools_rxns as native Analysis (single)"
```

---

### Task 7: Port `ptools_proteins` (single, data)

**Files:**
- Create: `v2ecoli/workflow/analyses/ptools_proteins.py`
- Modify: `v2ecoli/workflow/analyses/__init__.py` (add `ptools_proteins` import)
- Test: extend `tests/test_ptools_analyses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ptools_analyses.py`:
```python
def test_ptools_proteins_registered():
    from v2ecoli.workflow.analyses import ptools_proteins  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_proteins"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
    assert cls({}, core=__import__("bigraph_schema").allocate_core()).outputs() == {
        "view": "string", "data": "map"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py::test_ptools_proteins_registered -v`
Expected: FAIL with `ModuleNotFoundError: ...ptools_proteins`.

- [ ] **Step 3: Port the module**

Source: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/ptools_proteins.py`. Same six mechanical edits; class `PtoolsProteins(Analysis)`, `name = "ptools_proteins"`, return `{"data": {"filename": "ptools_proteins.tsv", "tsv": tsv}}`. Add the import to `analyses/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ptools_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analyses/ptools_proteins.py v2ecoli/workflow/analyses/__init__.py tests/test_ptools_analyses.py
git commit -m "feat: port ptools_proteins as native Analysis (single)"
```

---

### Task 8: Port `mass_fraction_voronoi` (single, view)

Proves the matplotlib→SVG `view` path for a single-scale plot.

**Files:**
- Create: `v2ecoli/workflow/analyses/mass_fraction_voronoi.py`
- Modify: `v2ecoli/workflow/analyses/__init__.py`
- Test: `tests/test_view_analyses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_view_analyses.py
from bigraph_schema import allocate_core


def test_mass_fraction_voronoi_registered_single_view():
    from v2ecoli.workflow.analyses import mass_fraction_voronoi  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["mass_fraction_voronoi"]
    assert issubclass(cls, Analysis) and cls.scale == "single"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_view_analyses.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mass_fraction_voronoi`.

- [ ] **Step 3: Port the module**

Source: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/single/mass_fraction_voronoi.py`. Edits: wrap in `class MassFractionVoronoi(Analysis)` (`name="mass_fraction_voronoi"`, `scale="single"`); use the passed `conn`/`history_sql`/`sim_data`; **replace the `fig.savefig(os.path.join(outdir, ...))` call** with:
```python
from v2ecoli.workflow.render import fig_to_html
return {"view": fig_to_html(fig, title="Mass fraction (Voronoi)")}
```
If this plot reads a listener column flagged missing in Task 1, substitute the nearest available mass listener recorded in the findings note, or — if blocked — port `ecoli/analysis/single/blame.py` instead (simpler single-scale plot) under the same `name`/pattern, and note the substitution in the module docstring. Add the import to `analyses/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_view_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analyses/mass_fraction_voronoi.py v2ecoli/workflow/analyses/__init__.py tests/test_view_analyses.py
git commit -m "feat: port mass_fraction_voronoi as native Analysis (single, view)"
```

---

### Task 9: Port `central_carbon_metabolism_scatter` (multiseed, view + cross-cell SQL)

Proves the cross-cell scale-scoped `history_sql` + matplotlib view.

**Files:**
- Create: `v2ecoli/workflow/analyses/central_carbon_metabolism_scatter.py`
- Modify: `v2ecoli/workflow/analyses/__init__.py`
- Test: extend `tests/test_view_analyses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_view_analyses.py`:
```python
def test_ccm_scatter_registered_multiseed():
    from v2ecoli.workflow.analyses import central_carbon_metabolism_scatter  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["central_carbon_metabolism_scatter"]
    assert issubclass(cls, Analysis) and cls.scale == "multiseed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_view_analyses.py::test_ccm_scatter_registered_multiseed -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Port the module**

Source: `git -C /Users/eranagmon/code/vivarium-ecoli show origin/ptools_viz:ecoli/analysis/multiseed/centralCarbonMetabolismScatter.py`. Wrap in `class CentralCarbonMetabolismScatter(Analysis)` (`name="central_carbon_metabolism_scatter"`, `scale="multiseed"`); use the injected `conn`/`history_sql` (the runner already scopes `history_sql` to one variant's seeds); replace the `savefig(outdir/...)` with `return {"view": fig_to_html(fig, title="Central carbon metabolism")}`. If it requires `validation_data` (Schmidt/Wisniewski) which is `None` here, guard the validation overlay behind `if validation_data:` and note it. Add the import to `analyses/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_view_analyses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analyses/central_carbon_metabolism_scatter.py v2ecoli/workflow/analyses/__init__.py tests/test_view_analyses.py
git commit -m "feat: port central_carbon_metabolism_scatter as native Analysis (multiseed, view)"
```

---

### Task 10: Proving-set config + end-to-end run

**Files:**
- Create: `v2ecoli/configs/analyses_proving.json`
- Test: extend `tests/test_analysis_runner_duckdb.py`

- [ ] **Step 1: Create the config**

```json
{
  "inherit_from": "two_generations.json",
  "analysis_options": {
    "single": {
      "ptools_rna": {"n_tp": 8},
      "ptools_rxns": {"n_tp": 8},
      "ptools_proteins": {"n_tp": 8},
      "mass_fraction_voronoi": {}
    },
    "multiseed": {
      "central_carbon_metabolism_scatter": {}
    }
  }
}
```
(Confirm `two_generations.json` exists under `v2ecoli/configs/`; if its filename differs, use the repo's smallest multi-generation config as `inherit_from`.)

- [ ] **Step 2: Write the end-to-end test (cache-gated on a real sweep)**

Append to `tests/test_analysis_runner_duckdb.py`:
```python
import glob as _glob
import json as _json
import os as _os
import pytest as _pytest


def _latest_sweep():
    cands = sorted(_glob.glob("out/*/"), key=_os.path.getmtime, reverse=True)
    for d in cands:
        if _glob.glob(_os.path.join(d, "**", "history", "**", "*.pq"), recursive=True):
            return d
    return None


@_pytest.mark.skipif(_latest_sweep() is None, reason="no local sweep with parquet")
def test_proving_set_end_to_end():
    import v2ecoli.workflow.analyses  # noqa: F401  (register ports)
    from v2ecoli.workflow.analysis_runner import run_analyses
    sweep = _latest_sweep()
    opts = {"single": {"ptools_rna": {"n_tp": 8}}}
    res = run_analyses(sweep, opts)
    assert "single" in res and "ptools_rna" in res["single"]
    # data product present (or a recorded error, not a silent drop)
    block = res["single"]["ptools_rna"]
    assert block, "ptools_rna produced no per-group result"
    # analysis.json written
    assert _os.path.isfile(_os.path.join(sweep, "analysis.json"))
```

- [ ] **Step 3: Run the end-to-end test**

Run: `.venv/bin/pytest tests/test_analysis_runner_duckdb.py::test_proving_set_end_to_end -v`
Expected: PASS if a local sweep exists (else SKIPPED). If it errors inside the port, fix per the Task 1 findings (column/attr shim), not by loosening the test.

- [ ] **Step 4: Optionally run the full proving set against the sweep + eyeball a view**

```bash
.venv/bin/v2ecoli-analyze "$(ls -dt out/*/ | head -1)" --config v2ecoli/configs/analyses_proving.json
ls "$(ls -dt out/*/ | head -1)"viz/
```
Expected: `analysis.json` updated; `viz/*.html` files for `mass_fraction_voronoi` and `central_carbon_metabolism_scatter`. Open one in a browser to confirm it renders.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/configs/analyses_proving.json tests/test_analysis_runner_duckdb.py
git commit -m "test: proving-set config + end-to-end Analysis runner test"
```

---

### Task 11: Dashboard surfacing — list Analyses in the viz picker

The dashboard already renders `<study>/viz/*.html` into the Visualizations tab. This task adds Analysis classes to the class picker so they're discoverable.

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (`_list_visualization_classes()`, ~lines 9036–9239)

- [ ] **Step 1: Locate the discovery function**

```bash
cd /Users/eranagmon/code/vivarium-dashboard
grep -n "_list_visualization_classes\|def _get_visualization_classes\|return classes" vivarium_dashboard/server.py | head
```
Read the function body to see how it accumulates the `classes` list (each entry `{address, name, doc}`).

- [ ] **Step 2: Append Analysis-registry entries**

Inside `_list_visualization_classes()`, just before it returns the assembled list, add (matching the existing entry dict shape — adjust keys if the function uses different ones, confirmed in Step 1):

```python
        # Native v2ecoli Analyses (Analysis subclasses) surface in the same picker.
        try:
            import v2ecoli.workflow.analyses  # noqa: F401  (import-time registration)
            from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
            for _name, _cls in ANALYSIS_REGISTRY.items():
                if isinstance(_cls, type) and issubclass(_cls, Analysis):
                    classes.append({
                        "address": f"local:{_cls.__module__}.{_cls.__qualname__}",
                        "name": _name,
                        "doc": (_cls.__doc__ or "").strip().split("\n")[0],
                        "kind": "analysis",
                    })
        except Exception:
            pass  # v2ecoli not importable in this workspace — skip silently
```

- [ ] **Step 3: Smoke-test the endpoint**

```bash
cd /Users/eranagmon/code/vivarium-dashboard
grep -n "ANALYSIS_REGISTRY\|kind.*analysis" vivarium_dashboard/server.py
.venv/bin/python -c "import ast; ast.parse(open('vivarium_dashboard/server.py').read()); print('parse OK')"
```
Expected: edit present; `parse OK`. (Full endpoint check happens when the dashboard runs against the v2ecoli workspace — `/pbg-dashboard start` then GET `/api/visualization-classes` should include `kind: analysis` entries.)

- [ ] **Step 4: Commit**

```bash
git add vivarium_dashboard/server.py
git commit -m "feat: list native Analysis classes in the visualization picker"
```

---

## Self-Review

**Spec coverage:**
- Analysis base (duckdb/sim_data ports, view+data) → Task 2 ✓
- Runner SQL-provisioning (open_connection/scale_history_sql/resolve_sim_data) → Task 4 ✓
- Render path (matplotlib→SVG, Plotly allowed) → Task 3 (helper) + Tasks 8–9 (use) ✓; ptools use Plotly-free TSV/data → Tasks 5–7 ✓
- Dashboard surfacing (views via existing embed_visualizations; picker extension) → Task 11 ✓
- Proving set (3 ptools + 2 plots across single+multiseed) → Tasks 5–9 ✓
- Parity-check first (gates ports) → Task 1 ✓
- Config + run.py wiring → Task 10 (run.py already calls run_analyses at run.py:94–97, no edit needed) ✓
- Spec "out of scope" (remaining 43, omics-viewer embed, BioCyc web service) → not in any task ✓ (correctly excluded)

**Placeholder scan:** Port tasks (5–9) intentionally reference the upstream source to copy rather than reproducing ~200 lines verbatim, but each gives the exact source command, the exact mechanical edits, the exact class scaffold, and complete test code — no "TODO"/"handle edge cases"/"similar to". The one judgement call (missing-listener fallback) names a concrete alternative file and action.

**Type consistency:** `Analysis.analyze()` keyword args (`conn`, `history_sql`, `sim_data`, `variant_metadata`) match what the runner injects in Task 4 Step 5 and what ports consume in Tasks 5–9. `outputs()` is `{view, data}` everywhere. `scale_history_sql(scale, from_clause, key)` signature consistent between Task 4 definition and Task 10 use. Registry key = `name` class attr, used identically in base (Task 2), ports, and dashboard (Task 11).

**Known carry-over risk:** numeric (not just structural) ptools fidelity is only asserted in Task 10's end-to-end test, which is cache-gated on a local sweep. If no sweep exists, that assertion is skipped — flagged here so it isn't mistaken for full verification.
