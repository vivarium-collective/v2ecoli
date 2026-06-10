# v2ecoli Workflow Analyses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port one native `AnalysisStep` per scale (computed from emitted observables) and wire a post-sweep analysis runner into `run_workflow` + a `v2ecoli-analyze` CLI.

**Architecture:** Four new `AnalysisStep`s (multidaughter/multigeneration/multiseed/multivariant) join the existing `MassFractionSummary` and auto-register into `ANALYSIS_REGISTRY` by name. A new `analysis_runner` module reads a finished sweep's parquet (per-cell timeseries) + a `summary.json` (division metadata `run_workflow` now writes), builds per-cell records, groups them per scale, runs the analyses named in `analysis_options`, and writes `analysis.json`. `run_workflow` calls it post-sweep; the same runner is a CLI.

**Tech Stack:** Python 3.12, `process-bigraph`/`bigraph-schema`, `duckdb` (parquet readback), `pytest`. Run tests with `.venv/bin/python -m pytest` (bare `python` lacks `unum`). Cache-gated tests use `out/cache`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `v2ecoli/workflow/analysis.py` | Extend: `ANALYSIS_REGISTRY` + auto-register; 4 new `AnalysisStep`s. |
| `v2ecoli/workflow/analysis_runner.py` | New: `build_cell_records`, `group_for_scale`, `run_analyses`, `main` (CLI). |
| `v2ecoli/workflow/run.py` | Write `summary.json`; call `run_analyses` post-sweep; drop the "not wired" warning. |
| `pyproject.toml` | `v2ecoli-analyze` console script. |
| `v2ecoli/configs/two_generations.json` | Add multigeneration + multiseed `analysis_options`. |
| `tests/test_workflow_analysis.py` | Extend: 4 new Steps + registry. |
| `tests/test_analysis_runner.py` | New: grouping, run_analyses, CLI, cache-gated end-to-end. |

**Per-cell summary record** (the unit cross-scale Steps consume), produced by `build_cell_records`:
```python
{"variant": int, "lineage_seed": int, "generation": int, "agent_id": str,
 "divided": bool | None, "division_time": float,
 "newborn_dry_mass": float, "final_dry_mass": float,
 "protein_fraction_mean": float, "rRna_fraction_mean": float, "dna_fraction_mean": float,
 "timeseries": [ {"listeners": {"mass": {"dry_mass","protein_mass","rRna_mass","dna_mass"}}} , ... ]}
```
`single`-scale Steps receive a cell's `timeseries`; cross-scale Steps receive a list of these records (scalar fields only).

---

## Task 1: ANALYSIS_REGISTRY + auto-registration

**Files:**
- Modify: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_workflow_analysis.py`:

```python
def test_analysis_registry_maps_names_to_steps():
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, MassFractionSummary
    assert ANALYSIS_REGISTRY["mass_fraction_summary"] is MassFractionSummary
    # every registered class is an AnalysisStep with a name
    from v2ecoli.workflow.analysis import AnalysisStep
    for name, cls in ANALYSIS_REGISTRY.items():
        assert issubclass(cls, AnalysisStep)
        assert cls.name == name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_analysis.py::test_analysis_registry_maps_names_to_steps -v`
Expected: FAIL with `ImportError: cannot import name 'ANALYSIS_REGISTRY'`

- [ ] **Step 3: Implement** — in `v2ecoli/workflow/analysis.py`, add the registry after `ANALYSIS_SCALES` (after line 32):

```python
# analysis name -> AnalysisStep subclass. Populated by __init_subclass__ for any
# subclass that defines its own ``name``; analysis_options entries resolve here.
ANALYSIS_REGISTRY: dict[str, type] = {}
```

Then extend `AnalysisStep.__init_subclass__` (replace the existing method) to also register:

```python
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.scale not in ANALYSIS_SCALES:
            raise ValueError(
                f"{cls.__name__}.scale={cls.scale!r} not in {sorted(ANALYSIS_SCALES)}")
        # Register concrete analyses (those declaring their own ``name``).
        if "name" in cls.__dict__:
            ANALYSIS_REGISTRY[cls.name] = cls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_analysis.py::test_analysis_registry_maps_names_to_steps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(analysis): ANALYSIS_REGISTRY auto-registration by name"
```

---

## Task 2: `DaughterMassSymmetry` (multidaughter)

**Files:**
- Modify: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_daughter_mass_symmetry():
    from v2ecoli.workflow.analysis import DaughterMassSymmetry
    from bigraph_schema import allocate_core
    step = DaughterMassSymmetry({}, core=allocate_core())
    out = step.analyze([{"newborn_dry_mass": 300.0}, {"newborn_dry_mass": 360.0}])
    assert out["n_sisters"] == 2
    assert abs(out["mass_asymmetry"] - (60.0 / 660.0)) < 1e-9
    # fewer than two daughters -> skipped
    one = step.analyze([{"newborn_dry_mass": 300.0}])
    assert one["n_sisters"] == 1 and "skipped" in one
```

- [ ] **Step 2: Run test** — Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_analysis.py::test_daughter_mass_symmetry -v` — Expected: FAIL (`cannot import name 'DaughterMassSymmetry'`).

- [ ] **Step 3: Implement** — append to `v2ecoli/workflow/analysis.py`:

```python
class DaughterMassSymmetry(AnalysisStep):
    """Multidaughter: birth-mass asymmetry |m1-m0|/(m1+m0) of sister cells.

    Dormant for single-lineage sweeps (only one daughter is carried); produces
    a value once binary-tree lineages (single_daughters=false) land.
    """

    name = "daughter_mass_symmetry"
    scale = "multidaughter"

    def analyze(self, rows):
        masses = [float(r.get("newborn_dry_mass", 0.0)) for r in rows]
        if len(masses) < 2:
            return {"n_sisters": len(masses),
                    "skipped": "needs >=2 daughters (single_daughters=false)"}
        m0, m1 = masses[0], masses[1]
        total = m0 + m1
        return {"n_sisters": len(masses),
                "mass_asymmetry": (abs(m1 - m0) / total) if total > 0 else 0.0}
```

- [ ] **Step 4: Run test** — same command — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(analysis): DaughterMassSymmetry (multidaughter)"
```

---

## Task 3: `MassGrowthAcrossGenerations` (multigeneration)

**Files:**
- Modify: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_mass_growth_across_generations():
    from v2ecoli.workflow.analysis import MassGrowthAcrossGenerations
    from bigraph_schema import allocate_core
    step = MassGrowthAcrossGenerations({}, core=allocate_core())
    rows = [
        {"generation": 1, "newborn_dry_mass": 350.0, "final_dry_mass": 700.0, "division_time": 2500.0},
        {"generation": 0, "newborn_dry_mass": 380.0, "final_dry_mass": 702.0, "division_time": 2400.0},
    ]
    out = step.analyze(rows)
    assert out["n_generations"] == 2
    # sorted by generation
    assert [g["generation"] for g in out["per_generation"]] == [0, 1]
    assert abs(out["per_generation"][0]["fold_change"] - (702.0 / 380.0)) < 1e-9
    assert abs(out["mean_division_time"] - 2450.0) < 1e-9
```

- [ ] **Step 2: Run test** — `...::test_mass_growth_across_generations -v` — Expected: FAIL.

- [ ] **Step 3: Implement** — append:

```python
class MassGrowthAcrossGenerations(AnalysisStep):
    """Multigeneration: per-generation newborn/final mass, cycle time, fold change
    across one lineage."""

    name = "mass_growth_across_generations"
    scale = "multigeneration"

    def analyze(self, rows):
        cells = sorted(rows, key=lambda r: int(r.get("generation", 0)))
        per_gen = []
        for c in cells:
            nb = float(c.get("newborn_dry_mass", 0.0))
            fn = float(c.get("final_dry_mass", 0.0))
            per_gen.append({
                "generation": int(c.get("generation", 0)),
                "newborn_dry_mass": nb, "final_dry_mass": fn,
                "division_time": float(c.get("division_time", 0.0)),
                "fold_change": (fn / nb) if nb > 0 else 0.0,
            })
        dts = [g["division_time"] for g in per_gen if g["division_time"] > 0]
        return {"n_generations": len(per_gen), "per_generation": per_gen,
                "mean_division_time": (sum(dts) / len(dts)) if dts else 0.0}
```

- [ ] **Step 4: Run test** — same — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(analysis): MassGrowthAcrossGenerations (multigeneration)"
```

---

## Task 4: `DoublingTimeDistribution` (multiseed)

**Files:**
- Modify: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_doubling_time_distribution():
    from v2ecoli.workflow.analysis import DoublingTimeDistribution
    from bigraph_schema import allocate_core
    step = DoublingTimeDistribution({}, core=allocate_core())
    rows = [
        {"divided": True, "division_time": 2400.0, "final_dry_mass": 700.0},
        {"divided": True, "division_time": 2600.0, "final_dry_mass": 720.0},
        {"divided": False, "division_time": 4000.0, "final_dry_mass": 500.0},  # excluded from doubling
    ]
    out = step.analyze(rows)
    assert out["n_cells"] == 3 and out["n_divided"] == 2
    assert abs(out["doubling_time_mean"] - 2500.0) < 1e-9
    assert out["doubling_time_std"] > 0
    assert abs(out["final_dry_mass_mean"] - (700.0 + 720.0 + 500.0) / 3) < 1e-6
```

- [ ] **Step 2: Run test** — `...::test_doubling_time_distribution -v` — Expected: FAIL.

- [ ] **Step 3: Implement** — append (note `import statistics` at top of the file or inside; put it at module top with the other imports):

```python
class DoublingTimeDistribution(AnalysisStep):
    """Multiseed: division-time mean/std over divided cells across seeds, plus
    mean final dry mass over all cells."""

    name = "doubling_time_distribution"
    scale = "multiseed"

    def analyze(self, rows):
        import statistics
        times = [float(r.get("division_time", 0.0)) for r in rows
                 if r.get("divided") is not False and float(r.get("division_time", 0.0)) > 0]
        finals = [float(r.get("final_dry_mass", 0.0)) for r in rows
                  if float(r.get("final_dry_mass", 0.0)) > 0]
        return {
            "n_cells": len(rows),
            "n_divided": len(times),
            "doubling_time_mean": statistics.mean(times) if times else 0.0,
            "doubling_time_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "final_dry_mass_mean": statistics.mean(finals) if finals else 0.0,
        }
```

- [ ] **Step 4: Run test** — same — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(analysis): DoublingTimeDistribution (multiseed)"
```

---

## Task 5: `MetricAcrossVariants` (multivariant)

**Files:**
- Modify: `v2ecoli/workflow/analysis.py`
- Test: `tests/test_workflow_analysis.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_metric_across_variants():
    from v2ecoli.workflow.analysis import MetricAcrossVariants
    from bigraph_schema import allocate_core
    step = MetricAcrossVariants({}, core=allocate_core())
    rows = [
        {"variant": 0, "divided": True, "division_time": 2400.0, "final_dry_mass": 700.0},
        {"variant": 0, "divided": True, "division_time": 2600.0, "final_dry_mass": 720.0},
        {"variant": 1, "divided": True, "division_time": 3000.0, "final_dry_mass": 650.0},
    ]
    out = step.analyze(rows)
    pv = out["per_variant"]
    assert pv[0]["n_cells"] == 2 and abs(pv[0]["mean_division_time"] - 2500.0) < 1e-9
    assert pv[1]["n_cells"] == 1 and abs(pv[1]["mean_final_dry_mass"] - 650.0) < 1e-9
```

- [ ] **Step 2: Run test** — `...::test_metric_across_variants -v` — Expected: FAIL.

- [ ] **Step 3: Implement** — append:

```python
class MetricAcrossVariants(AnalysisStep):
    """Multivariant: mean division time + final dry mass per variant."""

    name = "metric_across_variants"
    scale = "multivariant"

    def analyze(self, rows):
        import statistics
        by_variant: dict[int, list] = {}
        for r in rows:
            by_variant.setdefault(int(r.get("variant", 0)), []).append(r)
        per_variant = {}
        for v, cells in sorted(by_variant.items()):
            dts = [float(c.get("division_time", 0.0)) for c in cells
                   if c.get("divided") is not False and float(c.get("division_time", 0.0)) > 0]
            fms = [float(c.get("final_dry_mass", 0.0)) for c in cells
                   if float(c.get("final_dry_mass", 0.0)) > 0]
            per_variant[v] = {
                "n_cells": len(cells),
                "mean_division_time": statistics.mean(dts) if dts else 0.0,
                "mean_final_dry_mass": statistics.mean(fms) if fms else 0.0,
            }
        return {"per_variant": per_variant}
```

- [ ] **Step 4: Run test** — same — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis.py tests/test_workflow_analysis.py
git commit -m "feat(analysis): MetricAcrossVariants (multivariant)"
```

---

## Task 6: `group_for_scale` (runner grouping)

**Files:**
- Create: `v2ecoli/workflow/analysis_runner.py`
- Test: `tests/test_analysis_runner.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_analysis_runner.py`:

```python
from v2ecoli.workflow.analysis_runner import group_for_scale


def _recs():
    # two seeds, two generations each, single daughter per division
    return [
        {"variant": 0, "lineage_seed": 0, "generation": 0, "agent_id": "0"},
        {"variant": 0, "lineage_seed": 0, "generation": 1, "agent_id": "00"},
        {"variant": 0, "lineage_seed": 1, "generation": 0, "agent_id": "0"},
        {"variant": 1, "lineage_seed": 0, "generation": 0, "agent_id": "0"},
    ]


def test_group_single_is_per_cell():
    assert len(group_for_scale("single", _recs())) == 4


def test_group_multigeneration_by_lineage():
    g = group_for_scale("multigeneration", _recs())
    assert (0, 0) in g and len(g[(0, 0)]) == 2     # both gens of variant0/seed0
    assert (0, 1) in g and (1, 0) in g


def test_group_multiseed_by_variant():
    g = group_for_scale("multiseed", _recs())
    assert set(g) == {(0,), (1,)}
    assert len(g[(0,)]) == 3                         # all variant-0 cells


def test_group_multivariant_is_all():
    g = group_for_scale("multivariant", _recs())
    assert set(g) == {()} and len(g[()]) == 4


def test_group_multidaughter_by_parent():
    g = group_for_scale("multidaughter", _recs())
    # agent "00" groups under parent "0"; the gen-0 "0" cells group under "0" too
    assert any(k[3] == "0" for k in g)
```

- [ ] **Step 2: Run test** — Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_analysis_runner.py -v` — Expected: FAIL (`No module named 'v2ecoli.workflow.analysis_runner'`).

- [ ] **Step 3: Implement** — create `v2ecoli/workflow/analysis_runner.py`:

```python
"""Post-sweep analysis runner.

Reads a finished sweep's emitted parquet (per-cell timeseries) + summary.json
(division metadata written by run_workflow), builds per-cell records, groups
them per scale, runs the AnalysisSteps named in analysis_options, and writes
analysis.json. Also runnable standalone:

    v2ecoli-analyze <sweep_dir> [--config cfg.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from typing import Any


def group_for_scale(scale: str, records: list[dict]) -> dict[tuple, list[dict]]:
    """Group per-cell records by the key the scale aggregates over."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        v, s = int(r["variant"]), int(r["lineage_seed"])
        g, a = int(r["generation"]), str(r["agent_id"])
        if scale == "single":
            key = (v, s, g, a)
        elif scale == "multidaughter":
            parent = a[:-1] if len(a) > 1 else a   # sisters share the parent id
            key = (v, s, g, parent)
        elif scale == "multigeneration":
            key = (v, s)
        elif scale == "multiseed":
            key = (v,)
        elif scale == "multivariant":
            key = ()
        else:
            continue
        groups.setdefault(key, []).append(r)
    return groups
```

- [ ] **Step 4: Run test** — same command — Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis_runner.py tests/test_analysis_runner.py
git commit -m "feat(analysis): group_for_scale grouping in the runner"
```

---

## Task 7: `build_cell_records` + `run_analyses`

**Files:**
- Modify: `v2ecoli/workflow/analysis_runner.py`
- Test: `tests/test_analysis_runner.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_analysis_runner.py`:

```python
def test_run_analyses_over_synthetic_records(monkeypatch):
    """run_analyses dispatches each scale's Steps over the right groups and
    writes analysis.json. Stub build_cell_records to avoid IO."""
    import v2ecoli.workflow.analysis_runner as ar

    recs = {
        (0, 0, 0, "0"): {"variant": 0, "lineage_seed": 0, "generation": 0, "agent_id": "0",
                         "divided": True, "division_time": 2400.0,
                         "newborn_dry_mass": 380.0, "final_dry_mass": 700.0,
                         "timeseries": [{"listeners": {"mass": {"dry_mass": 380.0,
                            "protein_mass": 180.0, "rRna_mass": 38.0, "dna_mass": 7.0}}}]},
        (0, 1, 0, "0"): {"variant": 0, "lineage_seed": 1, "generation": 0, "agent_id": "0",
                         "divided": True, "division_time": 2600.0,
                         "newborn_dry_mass": 382.0, "final_dry_mass": 710.0,
                         "timeseries": [{"listeners": {"mass": {"dry_mass": 382.0,
                            "protein_mass": 190.0, "rRna_mass": 40.0, "dna_mass": 7.0}}}]},
    }
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: recs)

    import tempfile
    d = tempfile.mkdtemp()
    options = {"single": {"mass_fraction_summary": {}},
               "multiseed": {"doubling_time_distribution": {}}}
    results = ar.run_analyses(d, options)
    # single: one result block per cell
    assert len(results["single"]["mass_fraction_summary"]) == 2
    # multiseed: one variant group, mean of 2400/2600
    ms = list(results["multiseed"]["doubling_time_distribution"].values())[0]
    assert ms["n_cells"] == 2 and abs(ms["doubling_time_mean"] - 2500.0) < 1e-9
    # analysis.json written
    assert os.path.isfile(os.path.join(d, "analysis.json"))


def test_run_analyses_unknown_name_skips(monkeypatch):
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: {})
    import tempfile
    out = ar.run_analyses(tempfile.mkdtemp(), {"single": {"nope_not_real": {}}})
    assert out["single"] == {}    # unknown analysis dropped
```

- [ ] **Step 2: Run test** — Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_analysis_runner.py::test_run_analyses_over_synthetic_records tests/test_analysis_runner.py::test_run_analyses_unknown_name_skips -v` — Expected: FAIL (`run_analyses`/`build_cell_records` not defined).

- [ ] **Step 3: Implement** — append to `v2ecoli/workflow/analysis_runner.py`:

```python
_MASS_COLS = ("listeners__mass__dry_mass", "listeners__mass__protein_mass",
              "listeners__mass__rRna_mass", "listeners__mass__dna_mass")


def build_cell_records(sweep_dir: str) -> dict[tuple, dict]:
    """Build per-cell summary records from the sweep's parquet + summary.json."""
    import duckdb

    # division metadata (divided + division_time) from summary.json
    div_by_cell: dict[tuple, dict] = {}
    spath = os.path.join(sweep_dir, "summary.json")
    if os.path.isfile(spath):
        with open(spath) as f:
            summary = json.load(f)
        for bkey, bs in summary.items():
            m = re.search(r"variant=(\d+)/seed=(\d+)", bkey)
            if not m:
                continue
            v, s = int(m.group(1)), int(m.group(2))
            for gen in bs.get("generations", []):
                ck = (v, s, int(gen["generation"]), str(gen["agent_id"]))
                div_by_cell[ck] = {"divided": bool(gen.get("divided", False)),
                                   "division_time": float(gen.get("duration", 0.0))}

    files = glob.glob(os.path.join(sweep_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        return {}
    flist = "[" + ",".join("'" + f + "'" for f in files) + "]"
    sel = ("variant, lineage_seed, generation, agent_id, global_time, "
           + ", ".join(_MASS_COLS))
    rows = duckdb.sql(
        f"SELECT {sel} FROM read_parquet({flist}, hive_partitioning=true) "
        f"ORDER BY variant, lineage_seed, generation, agent_id, global_time"
    ).fetchall()

    by_cell: dict[tuple, list] = {}
    for row in rows:
        v, ls, g, a, t, dry, prot, rrna, dna = row
        ck = (int(v), int(ls), int(g), str(a))
        by_cell.setdefault(ck, []).append(
            (float(t), float(dry), float(prot), float(rrna), float(dna)))

    records: dict[tuple, dict] = {}
    for ck, rs in by_cell.items():
        fr = {"protein": [], "rRna": [], "dna": []}
        ts = []
        for (t, dry, prot, rrna, dna) in rs:
            ts.append({"listeners": {"mass": {"dry_mass": dry, "protein_mass": prot,
                                              "rRna_mass": rrna, "dna_mass": dna}}})
            if dry > 0:
                fr["protein"].append(prot / dry)
                fr["rRna"].append(rrna / dry)
                fr["dna"].append(dna / dry)
        div = div_by_cell.get(ck, {})
        records[ck] = {
            "variant": ck[0], "lineage_seed": ck[1], "generation": ck[2], "agent_id": ck[3],
            "divided": div.get("divided"),
            "division_time": div.get("division_time", float(rs[-1][0])),
            "newborn_dry_mass": rs[0][1], "final_dry_mass": rs[-1][1],
            "protein_fraction_mean": (sum(fr["protein"]) / len(fr["protein"])) if fr["protein"] else 0.0,
            "rRna_fraction_mean": (sum(fr["rRna"]) / len(fr["rRna"])) if fr["rRna"] else 0.0,
            "dna_fraction_mean": (sum(fr["dna"]) / len(fr["dna"])) if fr["dna"] else 0.0,
            "timeseries": ts,
        }
    return records


def _group_key_str(scale: str, key: tuple) -> str:
    if scale == "single":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/agent={key[3]}"
    if scale == "multidaughter":
        return f"variant={key[0]}/seed={key[1]}/gen={key[2]}/parent={key[3]}"
    if scale == "multigeneration":
        return f"variant={key[0]}/seed={key[1]}"
    if scale == "multiseed":
        return f"variant={key[0]}"
    return "all"


def run_analyses(sweep_dir: str, analysis_options: dict) -> dict:
    """Run the analyses named in ``analysis_options`` over the sweep's cells,
    write ``analysis.json``, and return the nested results."""
    from bigraph_schema import allocate_core
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, ANALYSIS_SCALES

    records = list(build_cell_records(sweep_dir).values())
    core = allocate_core()
    results: dict[str, dict] = {}
    for scale, analyses in (analysis_options or {}).items():
        if scale not in ANALYSIS_SCALES:
            warnings.warn(f"unknown analysis scale {scale!r}; skipping")
            continue
        groups = group_for_scale(scale, records)
        scale_out: dict[str, dict] = {}
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
            for gkey, grp in groups.items():
                try:
                    rows = grp[0].get("timeseries") if scale == "single" else grp
                    per_group[_group_key_str(scale, gkey)] = step.analyze(rows or [])
                except Exception as e:  # one bad analysis must not sink the rest
                    per_group[_group_key_str(scale, gkey)] = {
                        "error": f"{type(e).__name__}: {e}"}
            scale_out[name] = per_group
        results[scale] = scale_out

    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results
```

- [ ] **Step 4: Run test** — same command — Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/workflow/analysis_runner.py tests/test_analysis_runner.py
git commit -m "feat(analysis): build_cell_records + run_analyses"
```

---

## Task 8: CLI + console script

**Files:**
- Modify: `v2ecoli/workflow/analysis_runner.py`
- Modify: `pyproject.toml`
- Test: `tests/test_analysis_runner.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_cli_main_runs(monkeypatch, tmp_path, capsys):
    import v2ecoli.workflow.analysis_runner as ar
    monkeypatch.setattr(ar, "build_cell_records", lambda sweep_dir: {})
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"analysis_options": {"single": {"mass_fraction_summary": {}}}}')
    monkeypatch.setattr("sys.argv", ["v2ecoli-analyze", str(tmp_path), "--config", str(cfg)])
    ar.main()
    assert os.path.isfile(str(tmp_path / "analysis.json"))
    assert "analysis.json" in capsys.readouterr().out
```

- [ ] **Step 2: Run test** — `...::test_cli_main_runs -v` — Expected: FAIL (`main` not defined).

- [ ] **Step 3: Implement** — append `main()` to `v2ecoli/workflow/analysis_runner.py`:

```python
def main() -> None:
    p = argparse.ArgumentParser(description="Run configured analyses over a sweep.")
    p.add_argument("sweep_dir", help="sweep output dir (parquet + summary.json)")
    p.add_argument("--config", default=None,
                   help="config JSON with analysis_options (with inherit_from)")
    args = p.parse_args()

    analysis_options: dict = {}
    if args.config:
        from v2ecoli.workflow.config import load_config_with_inheritance
        analysis_options = load_config_with_inheritance(args.config).get(
            "analysis_options") or {}
    if not analysis_options:
        print("no analysis_options found; nothing to run")
        return
    run_analyses(args.sweep_dir, analysis_options)
    print(f"Wrote {os.path.join(args.sweep_dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the console script** — in `pyproject.toml` under `[project.scripts]` (next to `v2ecoli-workflow`), add:

```toml
v2ecoli-analyze = "v2ecoli.workflow.analysis_runner:main"
```

Then: `cd /Users/eranagmon/code/v2ecoli && uv pip install --python .venv/bin/python -e . --no-deps`

- [ ] **Step 5: Run test** — `...::test_cli_main_runs -v` — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/analysis_runner.py pyproject.toml tests/test_analysis_runner.py
git commit -m "feat(analysis): v2ecoli-analyze CLI"
```

---

## Task 9: Wire into run_workflow (summary.json + post-sweep analyses)

**Files:**
- Modify: `v2ecoli/workflow/run.py`
- Test: `tests/test_analysis_runner.py`

- [ ] **Step 1: Write the failing test** (cache-gated end-to-end) — append:

```python
import pytest

_CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")


@pytest.mark.skipif(not os.path.isdir(_CACHE), reason="ParCa cache not present")
def test_run_workflow_runs_analyses_end_to_end(tmp_path):
    from v2ecoli.workflow.run import run_workflow
    out = str(tmp_path / "parquet")
    config = {
        "experiment_id": "anlz", "n_init_sims": 2, "generations": 1,
        "single_daughters": True, "cache_dir": _CACHE, "out_dir": out,
        "variants": {}, "max_duration_per_gen": 5.0, "time_step": 1.0,
        "analysis_options": {
            "single": {"mass_fraction_summary": {}},
            "multiseed": {"doubling_time_distribution": {}},
        },
    }
    result = run_workflow(config, max_sim_time=30.0)
    assert result["complete"] is True
    assert os.path.isfile(os.path.join(out, "summary.json"))
    assert os.path.isfile(os.path.join(out, "analysis.json"))
    with open(os.path.join(out, "analysis.json")) as f:
        analysis = json.load(f)
    # single: one block per cell (2 cells, gen 0)
    assert len(analysis["single"]["mass_fraction_summary"]) == 2
    # multiseed: a doubling-time block exists for the variant
    assert analysis["multiseed"]["doubling_time_distribution"]
```

(add `import json` at the top of the test file if not present.)

- [ ] **Step 2: Run test** — Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_analysis_runner.py::test_run_workflow_runs_analyses_end_to_end -v` — Expected: FAIL (no summary.json / analysis.json written) — or SKIP without cache.

- [ ] **Step 3: Implement** — in `v2ecoli/workflow/run.py`:

(a) Delete the warning block at the top of `run_workflow` (lines 41-47, the `analysis_options = ...` + `if any(...)` warning).

(b) Replace the `return { ... }` at the end of `run_workflow` with:

```python
    branch_result = {
        k: {
            "complete": v.get("complete"),
            "summary": proc_summaries.get(k, v.get("summary") or {}),
        }
        for k, v in branches.items()
    }

    out_dir = config.get("out_dir") or "out/workflow"
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({k: rv["summary"] for k, rv in branch_result.items()},
                  f, indent=2, default=str)

    result = {
        "complete": complete,
        "elapsed": elapsed,
        "timed_out": not complete,
        "branches": branch_result,
    }

    analysis_options = config.get("analysis_options") or {}
    if any((analysis_options or {}).values()):
        from v2ecoli.workflow.analysis_runner import run_analyses
        run_analyses(out_dir, analysis_options)
        result["analysis"] = os.path.join(out_dir, "analysis.json")
    return result
```

- [ ] **Step 4: Run test** — same command — Expected: PASS (or SKIP without cache).

- [ ] **Step 5: Run the parquet smoke + meta tests for regression**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_smoke.py tests/test_meta_composite_build.py -v`
Expected: PASS (run_workflow still returns the same keys plus the new summary.json side-effect).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/workflow/run.py tests/test_analysis_runner.py
git commit -m "feat(workflow): write summary.json + run analyses post-sweep"
```

---

## Task 10: Ported config + full suite

**Files:**
- Modify: `v2ecoli/configs/two_generations.json`
- Test: `tests/test_workflow_config.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `tests/test_workflow_config.py`:

```python
def test_two_generations_config_has_multiscale_analyses():
    import os
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "v2ecoli", "configs")
    cfg = load_config_with_inheritance(os.path.join(cfg_dir, "two_generations.json"))
    opts = cfg["analysis_options"]
    assert "mass_fraction_summary" in opts["single"]
    assert "mass_growth_across_generations" in opts["multigeneration"]
    assert "doubling_time_distribution" in opts["multiseed"]
```

- [ ] **Step 2: Run test** — `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_config.py::test_two_generations_config_has_multiscale_analyses -v` — Expected: FAIL (keys absent).

- [ ] **Step 3: Implement** — edit `v2ecoli/configs/two_generations.json`, replacing its `analysis_options` block with:

```json
    "analysis_options": {
        "single": {"mass_fraction_summary": {}},
        "multigeneration": {"mass_growth_across_generations": {}},
        "multiseed": {"doubling_time_distribution": {}}
    }
```

- [ ] **Step 4: Run test** — same — Expected: PASS.

- [ ] **Step 5: Run the full analysis + workflow suite**

Run: `cd /Users/eranagmon/code/v2ecoli && .venv/bin/python -m pytest tests/test_workflow_analysis.py tests/test_analysis_runner.py tests/test_workflow_config.py tests/test_workflow_smoke.py -v`
Expected: all PASS (cache-gated end-to-end runs if `out/cache` present, else SKIP).

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/configs/two_generations.json tests/test_workflow_config.py
git commit -m "feat(workflow): two_generations config runs multiscale analyses"
```

---

## Self-Review Notes (addressed during planning)

- **Spec coverage:** registry (T1) ✓, the 4 native Steps single/multidaughter/multigeneration/multiseed/multivariant (T2-T5; single exists) ✓, grouping per scale (T6) ✓, per-cell records from parquet + summary.json incl. the source split (T7) ✓, run_analyses + error isolation + unknown-name skip (T7) ✓, CLI (T8) ✓, run_workflow summary.json + post-sweep call + drop warning (T9) ✓, config wiring (T10) ✓.
- **Type consistency:** per-cell record keys (`newborn_dry_mass`, `final_dry_mass`, `division_time`, `divided`, `variant`, `lineage_seed`, `generation`, `agent_id`, `timeseries`) are identical across `build_cell_records`, the cross-scale Steps, and the synthetic-record tests. `analyze(rows)` returns dicts; `ANALYSIS_REGISTRY[name]=cls`, `cls.scale`/`cls.name` used consistently.
- **Deferred (per spec §Out of scope):** faithful vEcoli ports, report-HTML integration, xarray-as-analysis-source, multidaughter activation (needs binary-tree), >1 analysis per scale. Each is loud (a `skipped` result or simply absent), not a silent gap.
- **Cache dependency:** the end-to-end (T9) + the config-run are cache-gated and SKIP without `out/cache`; all Step unit tests + grouping + run_analyses (stubbed) + CLI run without the cache.
