# Phase 0 — Local Basal Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove locally and cheaply that the upstream-wrapper and v2ecoli engines each produce a *physically valid* basal trajectory (`cell_mass` ~doubles and divides, requested generations reached) — supplying the dynamics validation PR #289 is missing, and gating the merge of #289 + sms-api #147.

**Architecture:** A small, fully unit-tested `physical_validity` module (pure numpy core + a thin zarr loader) decides PASS/FAIL from a lineage `cell_mass` series. A `phase0_validate_basal.py` orchestrator runs both engines basal locally at shallow depth via the existing `run_comparison_ensemble.py`, applies the checker to each engine's zarr, and emits a PASS/FAIL gate verdict. No simulator or wrapper code is edited — Phase 0 only *validates*.

**Tech Stack:** Python 3.13, numpy, xarray/zarr, pytest. Repo venv at `.venv/bin/python`. Existing runner `scripts/run_comparison_ensemble.py`.

## Global Constraints

- Run everything with the repo venv: `.venv/bin/python` (bare `python` lacks `unum`/`pint`).
- Set `PYTHONHASHSEED=0` and `PYTHONPATH=$PWD` for every sim invocation (matches `run_upstream_multigen.py`).
- **No edits to upstream vEcoli or the wrapper** — Phase 0 only validates. Upstream stays git-clean.
- **"It ran" ≠ "it's right":** the gate asserts physical validity on raw `cell_mass` values, never on exit code or a rendered report alone.
- Branch context: work in the existing worktree `/Users/eranagmon/code/v2e-compare-harness` on `feat/upstream-vecoli-pbg` (HEAD `9be885a3`). Do **not** create a new branch — Phase 0 lands on the same PR #289 branch it validates.
- v2ecoli ParCa cache dir: `out/cache`. Upstream-built ParCa cache dir: `out/compare_harness/vecoli_parca/` (the upstream-master `simData.cPickle`, kept separate so the v2ecoli TCS skew can't crash upstream's two-component ODE — see `run_comparison_ensemble.py:402-405`).

---

### Task 1: Physical-validity core (segmentation + assessment)

Pure numpy. No I/O, no sim. Fully TDD with synthetic trajectories — this is the gate's logic and must be airtight before any slow sim runs.

**Files:**
- Create: `scripts/_compare/physical_validity.py`
- Test: `tests/test_physical_validity.py`

**Interfaces:**
- Consumes: nothing (pure functions over numpy arrays).
- Produces:
  - `segment_generations(cell_mass: np.ndarray, *, drop_frac: float = 0.6) -> list[tuple[int, int]]`
    — returns `[(start, end), ...]` half-open index ranges, one per generation, split at a division (a step where `cell_mass[i+1] < drop_frac * cell_mass[i]`).
  - `@dataclass Verdict` with fields: `physical: bool`, `generations_reached: int`, `divisions_detected: int`, `per_gen_ratios: list[float]`, `reasons: list[str]`.
  - `assess_physical(cell_mass: np.ndarray, *, min_generations: int = 2, doubling_band: tuple[float, float] = (1.5, 3.5)) -> Verdict`
    — segments, computes each *complete* generation's growth ratio `cell_mass[end-1] / cell_mass[start]`, and sets `physical=True` iff every complete-generation ratio is within `doubling_band` AND `divisions_detected >= min_generations - 1`. Each failing condition appends a human-readable string to `reasons`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_physical_validity.py
import numpy as np
from scripts._compare.physical_validity import segment_generations, assess_physical


def _doubling_trajectory(n_gens=3, steps_per_gen=50, m0=5000.0):
    """Physical: linear growth m -> ~2m within a gen, then halve at division."""
    segs = []
    m = m0
    for _ in range(n_gens):
        seg = np.linspace(m, 2 * m, steps_per_gen, endpoint=False)
        segs.append(seg)
        m = seg[-1] / 2.0  # division halves the mother into the next founder
    return np.concatenate(segs)


def test_segment_splits_at_each_division():
    cm = _doubling_trajectory(n_gens=3, steps_per_gen=50)
    segs = segment_generations(cm)
    assert len(segs) == 3
    # half-open, contiguous, covering
    assert segs[0][0] == 0
    assert segs[-1][1] == len(cm)


def test_physical_doubling_passes():
    cm = _doubling_trajectory(n_gens=3)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is True
    assert v.divisions_detected == 2
    assert all(1.5 <= r <= 3.5 for r in v.per_gen_ratios)


def test_mass_explosion_fails():
    # 5k -> ~90k in one generation (the pre-fix bug), no division
    cm = np.linspace(5000.0, 90000.0, 80)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is False
    assert any("ratio" in r.lower() or "division" in r.lower() for r in v.reasons)


def test_truncated_run_fails_on_generation_count():
    # one clean generation then nothing — fewer divisions than required
    cm = _doubling_trajectory(n_gens=1)
    v = assess_physical(cm, min_generations=2)
    assert v.physical is False
    assert any("division" in r.lower() or "generation" in r.lower() for r in v.reasons)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_physical_validity.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.physical_validity` / function not defined.

- [ ] **Step 3: Write the implementation**

```python
# scripts/_compare/physical_validity.py
"""Decide whether a lineage cell_mass trajectory is physically valid.

A valid whole-cell basal run grows cell_mass ~2x within a generation and then
halves at division. The pre-fix wrapper bug made cell_mass explode ~18x in one
generation (fail-open partition gate -> evolvers re-applied bulk deltas every
tick). This module turns that distinction into a hard PASS/FAIL gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def segment_generations(cell_mass: np.ndarray, *, drop_frac: float = 0.6) -> list[tuple[int, int]]:
    cm = np.asarray(cell_mass, dtype=float)
    if cm.size == 0:
        return []
    bounds = [0]
    for i in range(cm.size - 1):
        if cm[i] > 0 and cm[i + 1] < drop_frac * cm[i]:
            bounds.append(i + 1)
    bounds.append(cm.size)
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


@dataclass
class Verdict:
    physical: bool
    generations_reached: int
    divisions_detected: int
    per_gen_ratios: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def assess_physical(
    cell_mass: np.ndarray,
    *,
    min_generations: int = 2,
    doubling_band: tuple[float, float] = (1.5, 3.5),
) -> Verdict:
    cm = np.asarray(cell_mass, dtype=float)
    segs = segment_generations(cm)
    divisions = max(len(segs) - 1, 0)
    # complete generations are all but the last (the last may be mid-cycle, no division yet)
    complete = segs[:-1] if len(segs) >= 1 else []
    ratios: list[float] = []
    reasons: list[str] = []
    lo, hi = doubling_band
    for (s, e) in complete:
        if e - s < 2 or cm[s] <= 0:
            continue
        r = float(cm[e - 1] / cm[s])
        ratios.append(r)
        if not (lo <= r <= hi):
            reasons.append(f"generation [{s}:{e}] growth ratio {r:.2f} outside physical band {doubling_band}")
    if divisions < min_generations - 1:
        reasons.append(
            f"only {divisions} division(s) detected; require >= {min_generations - 1} "
            f"for {min_generations} generations"
        )
    physical = len(reasons) == 0 and len(ratios) >= max(min_generations - 1, 1)
    if len(ratios) < max(min_generations - 1, 1) and not reasons:
        reasons.append("no complete generation with a measurable growth ratio")
        physical = False
    return Verdict(
        physical=physical,
        generations_reached=len(segs),
        divisions_detected=divisions,
        per_gen_ratios=ratios,
        reasons=reasons,
    )
```

(`scripts/_compare/` is already a package with an `__init__.py`, and tests already import via `from scripts._compare.X import` under `PYTHONPATH=$PWD` — no new `__init__.py` needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_physical_validity.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/physical_validity.py tests/test_physical_validity.py
git commit -m "feat(compare): physical-validity gate core (mass doubling + division)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Lineage zarr loader for cell_mass

Thin adapter from a `run_multigen_xarray` store to the 1-D `cell_mass` series the core consumes. The exact nesting depth of the store is confirmed against a real zarr in Task 4; here we lock the contract against a synthetic store and search for the variable by name.

**Files:**
- Modify: `scripts/_compare/physical_validity.py`
- Test: `tests/test_physical_validity.py`

**Interfaces:**
- Consumes: a zarr store path produced by `run_multigen_xarray` (variable named `cell_mass` somewhere in the tree).
- Produces: `load_cell_mass(store_path: str) -> np.ndarray` — returns the time-ordered `cell_mass` series concatenated across the store, as a 1-D float array. Raises `ValueError` if no `cell_mass` variable is found.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_physical_validity.py
import xarray as xr
from scripts._compare.physical_validity import load_cell_mass


def test_load_cell_mass_from_zarr(tmp_path):
    store = str(tmp_path / "lineage.zarr")
    cm = np.linspace(5000.0, 10000.0, 30)
    ds = xr.Dataset({"cell_mass": ("time", cm)}, coords={"time": np.arange(30)})
    ds.to_zarr(store, mode="w")
    out = load_cell_mass(store)
    assert out.shape == (30,)
    assert np.allclose(out, cm)


def test_load_cell_mass_missing_raises(tmp_path):
    store = str(tmp_path / "empty.zarr")
    xr.Dataset({"dry_mass": ("time", np.ones(5))}, coords={"time": np.arange(5)}).to_zarr(store, mode="w")
    import pytest
    with pytest.raises(ValueError):
        load_cell_mass(store)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_physical_validity.py -k load_cell_mass -v`
Expected: FAIL — `load_cell_mass` not defined.

- [ ] **Step 3: Write the implementation**

```python
# add to scripts/_compare/physical_validity.py
import xarray as xr


def load_cell_mass(store_path: str) -> np.ndarray:
    """Return the time-ordered cell_mass series from a run_multigen_xarray store.

    Searches the dataset's data variables for one named 'cell_mass' (the view in
    run_upstream_multigen.py emits listeners/mass/cell_mass, which xarray flattens
    to a 'cell_mass' variable). Concatenates along the leading (time) axis.
    """
    ds = xr.open_zarr(store_path)
    name = next((v for v in ds.data_vars if str(v).split("/")[-1] == "cell_mass"), None)
    if name is None:
        raise ValueError(f"no 'cell_mass' variable in {store_path}; vars={list(ds.data_vars)}")
    arr = np.asarray(ds[name].values, dtype=float).reshape(-1)
    return arr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_physical_validity.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/physical_validity.py tests/test_physical_validity.py
git commit -m "feat(compare): zarr cell_mass loader for the physical-validity gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Confirm local ParCa caches exist (build if missing)

Both engines need a ParCa cache on disk before a local basal run. This task verifies presence and builds whatever is missing. No new code — uses existing build paths.

**Files:**
- Modify: none (verification + cache build only).

**Interfaces:**
- Consumes: `scripts/build_upstream_parca.py` (upstream cache), the v2ecoli ParCa build path.
- Produces: `out/cache/` (v2ecoli `simData.cPickle`) and `out/compare_harness/vecoli_parca/` (upstream `simData.cPickle`) present on disk.

- [ ] **Step 1: Check what already exists**

Run:
```bash
ls -la out/cache/ 2>&1 | head
ls -la out/compare_harness/vecoli_parca/ 2>&1 | head
```
Expected: either a populated cache dir, or "No such file or directory".
Note: in a worktree the v2 `out/cache` is commonly a symlink to the main checkout's cache (see memory `reference_dashboard`/worktree-cache-symlink). If `out/cache` is absent, create the symlink first:
```bash
ls -la /Users/eranagmon/code/v2ecoli/out/cache/simData.cPickle 2>&1
# if the main cache exists and out/cache is absent here:
[ -e out/cache ] || ln -s /Users/eranagmon/code/v2ecoli/out/cache out/cache
```

- [ ] **Step 2: Build the upstream cache if missing (serial, ~14 min)**

Only if `out/compare_harness/vecoli_parca/simData.cPickle` is absent. Serial `--cpus 1` is **required** (parallel workers re-import uncompiled Cython and respawn-loop into a >1 h hang — see sms-api #147).
```bash
PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/build_upstream_parca.py --cpus 1 2>&1 | tail -20
```
Expected: completes; `out/compare_harness/vecoli_parca/simData.cPickle` exists afterward.
(If `build_upstream_parca.py` does not accept `--cpus`, run `--help` to read its flags and use the serial option it exposes — do not run parallel.)

- [ ] **Step 3: Build the v2ecoli cache if missing**

Only if `out/cache/simData.cPickle` is still absent after the symlink check. v2ecoli ParCa builds fast (~2.5 min, full mode). Use the repo's standard cache build (confirm the exact entry point):
```bash
grep -rnE "build_cache|fit_sim_data|save_sim_input" scripts/*.py | head
```
Then run the identified builder with `--mode full` (NOT fast — fast mis-calibrates regulation; see memory `reference_v2ecoli_parca_fast_not_for_sim`). Verify `out/cache/simData.cPickle` exists.

- [ ] **Step 4: Record cache provenance**

Run:
```bash
ls -la out/cache/simData.cPickle out/compare_harness/vecoli_parca/simData.cPickle
```
Expected: both files present. No commit (caches are gitignored build artifacts).

---

### Task 4: Phase-0 orchestrator + run-command smoke

Build the orchestrator that runs both engines basal at shallow depth and applies the checker. First confirm the run commands actually produce a zarr and that `load_cell_mass` reads it (locks the loader contract against real data).

**Files:**
- Create: `scripts/phase0_validate_basal.py`
- Create: `docs/phase0_validation.md` (verdict record, filled in Task 5)

**Interfaces:**
- Consumes: `scripts/run_comparison_ensemble.py` (CLI), `scripts/_compare/physical_validity.py` (`load_cell_mass`, `assess_physical`).
- Produces: `out/phase0/<engine>/<engine>_seed00.zarr` per engine; `out/phase0/verdict.json`; process exit code 0 iff both engines PASS.

- [ ] **Step 1: Smoke each engine at 1 generation and confirm a zarr + cell_mass**

v2ecoli:
```bash
PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/run_comparison_ensemble.py \
  --composite v2ecoli --condition basal --cache-dir out/cache \
  --n-seeds 1 --max-generations 1 --max-steps 6000 --mode serial \
  --out-root out/phase0_smoke/v2ecoli 2>&1 | tail -15
```
Upstream:
```bash
PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/run_comparison_ensemble.py \
  --composite vecoli --condition basal --cache-dir out/compare_harness/vecoli_parca \
  --n-seeds 1 --max-generations 1 --max-steps 6000 --mode serial \
  --out-root out/phase0_smoke/vecoli 2>&1 | tail -15
```
Then confirm the loader reads each store:
```bash
PYTHONPATH=$PWD .venv/bin/python -c "
from scripts._compare.physical_validity import load_cell_mass
for p in ['out/phase0_smoke/v2ecoli/v2ecoli_seed00.zarr','out/phase0_smoke/vecoli/vecoli_seed00.zarr']:
    cm = load_cell_mass(p); print(p, cm.shape, float(cm.min()), float(cm.max()))
"
```
Expected: each prints a non-empty series. If a store path differs from `{out_root}/{kind}_seed00.zarr`, read `run_comparison_ensemble.py:570` and use the actual pattern. If `cell_mass` is nested under a group xarray does not flatten, extend `load_cell_mass` to walk groups (open with `xr.open_datatree`) and re-run Task 2's tests.

- [ ] **Step 2: Write the orchestrator**

```python
# scripts/phase0_validate_basal.py
"""Phase 0 gate: run both engines basal locally (shallow) and assert physical mass.

Exit 0 iff BOTH the upstream-wrapper and v2ecoli basal runs grow cell_mass ~2x
per generation and divide for the requested number of generations. This is the
dynamics validation PR #289 is missing; a green run here gates merging #289 + #147.

    PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/phase0_validate_basal.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts._compare.physical_validity import assess_physical, load_cell_mass

GENS = int(os.environ.get("PHASE0_GENS", "2"))
MAX_STEPS = int(os.environ.get("PHASE0_MAX_STEPS", "9000"))
ENGINES = {
    "v2ecoli": "out/cache",
    "vecoli": "out/compare_harness/vecoli_parca",
}


def _run(engine: str, cache_dir: str) -> str:
    out_root = f"out/phase0/{engine}"
    cmd = [
        ".venv/bin/python", "scripts/run_comparison_ensemble.py",
        "--composite", engine, "--condition", "basal", "--cache-dir", cache_dir,
        "--n-seeds", "1", "--max-generations", str(GENS),
        "--max-steps", str(MAX_STEPS), "--mode", "serial", "--out-root", out_root,
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": os.getcwd()}
    print(f"[phase0] running {engine}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    return f"{out_root}/{engine}_seed00.zarr"


def main() -> int:
    results = {}
    ok = True
    for engine, cache_dir in ENGINES.items():
        store = _run(engine, cache_dir)
        cm = load_cell_mass(store)
        v = assess_physical(cm, min_generations=GENS)
        results[engine] = {
            "store": store, "physical": v.physical,
            "generations_reached": v.generations_reached,
            "divisions_detected": v.divisions_detected,
            "per_gen_ratios": v.per_gen_ratios, "reasons": v.reasons,
        }
        ok = ok and v.physical
        print(f"[phase0] {engine}: physical={v.physical} ratios={v.per_gen_ratios} "
              f"reasons={v.reasons}", flush=True)
    Path("out/phase0").mkdir(parents=True, exist_ok=True)
    Path("out/phase0/verdict.json").write_text(json.dumps(results, indent=2))
    print(f"[phase0] GATE {'PASS' if ok else 'FAIL'} -> out/phase0/verdict.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Commit the orchestrator**

```bash
git add scripts/phase0_validate_basal.py
git commit -m "feat(compare): phase-0 local basal physical-validity gate orchestrator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run the gate, capture the verdict

The actual Phase-0 deliverable: run the gate at 2 generations per engine and record the honest outcome.

**Files:**
- Modify: `docs/phase0_validation.md`

- [ ] **Step 1: Run the gate**

Run:
```bash
PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/phase0_validate_basal.py 2>&1 | tee out/phase0/run.log
```
Expected: `[phase0] GATE PASS` with both engines' `per_gen_ratios` inside the physical band. If FAIL, the verdict's `reasons` localize the issue — do **not** proceed to later phases; instead open systematic-debugging on the failing engine (the fail-closed gate or emitter close is the prime suspect if mass still explodes).

- [ ] **Step 2: Record the verdict honestly**

Write `docs/phase0_validation.md` with: the exact commands run, `out/phase0/verdict.json` contents, per-engine `cell_mass` min/max and per-generation ratios, and a one-line PASS/FAIL conclusion. If PASS, state explicitly that this validates PR #289's dynamics locally and is the evidence to merge #289 + sms-api #147. If FAIL, state which engine and the reason, and that the merge is blocked pending a fix.

- [ ] **Step 3: Commit**

```bash
git add docs/phase0_validation.md out/phase0/verdict.json
git commit -m "docs(compare): phase-0 local basal validation verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(`out/phase0/verdict.json` may be gitignored under `out/`; if `git add` skips it, use `git add -f out/phase0/verdict.json` so the verdict is recorded with the doc.)

---

## Phase-0 exit criteria

- `tests/test_physical_validity.py` green (6 tests).
- `scripts/phase0_validate_basal.py` exits 0: both engines grow `cell_mass` within the physical doubling band across 2 generations and divide.
- `docs/phase0_validation.md` records the honest verdict.

**On PASS:** the gate is the evidence to merge v2ecoli #289 + sms-api #147 (user approves merges — do not auto-merge). Then write the **Phase 1** plan (per-process delta-capture tooling: Tier A co-execution harness + Tier B emit + correspondence map + noise-band baseline).

**On FAIL:** stop. Open systematic-debugging on the failing engine before anything else; the merge stays blocked.

## Follow-on plans (not written yet — gated)

- **Phase 1** — per-process delta-capture tooling (Tier A + Tier B).
- **Phase 2** — local 5-condition divergence pass (1–2 gens/condition), fix structural bugs.
- **Phase 3** — GovCloud 16×16×5 (merge #147, rebuild image, smoke on exact job IDs, full run).
- **Phase 4** — honest comparison report (bands + per-process attribution + residual-divergence section).

Each is written only after the prior gate clears, because each depends on the prior's empirical outcome.
