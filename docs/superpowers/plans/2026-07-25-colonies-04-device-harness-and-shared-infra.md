# colonies-04: Device Harness + Shared Phenotype Infra — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared colony-phenotyping infrastructure (cell-tier factory, geometry builders, phenotype extractor, run harness) and land study `colonies-04` running the mother machine, daughter machine, and free colony with *simple* and *WCM* agents, producing a common phenotype panel.

**Architecture:** Factor geometry from cell model. A `cell_factory(tier)` returns a cell-body dict with a uniform port contract (`mass`/`length`/`volume`/`agents`); geometry builders (`free_colony`, `mother_machine`, `daughter_machine`) place N such cells via an injected factory; a pure `phenotype_extractor` reads a sampled trajectory (decoupled from the emitter, avoiding the RAM leak) and computes growth / size-at-division / added-length / inter-division-time, plus optional exchange. A thin run harness wires core + geometry + tier, samples per-tick state, and calls the extractor.

**Tech Stack:** Python, `process_bigraph` (Composite), `viva_munk` (PymunkProcess, build_microbe, grow_divide), `v2ecoli` (EcoliWCM bridge, ECOLI_TYPES), `numpy`, `pytest`.

## Global Constraints

- Run v2ecoli via `.venv/bin/python` (bare `python` lacks `unum`). Tests: `.venv/bin/python -m pytest`.
- WCM tier requires a ParCa cache at `out/cache`; in a worktree symlink it to main's (`ln -s ../v2ecoli/out/cache out/cache`) — never rebuild in the worktree.
- The uniform cell-body port contract is load-bearing: every tier exposes `mass` (fg), `length` (µm), `location`, `angle`, `id`, and an `agents` division wire `['..','..','cells']`. `volume` (fL) and `exchange` are WCM-primary and OPTIONAL for other tiers; the extractor treats them as absent when missing.
- Do NOT emit the full cells-map via the outer emitter for long runs (`emit_cells=False` equivalent) — the harness samples `composite.state['cells']` directly. This is the established colonies-02 pattern that dodges the ~1 MB/tick/cell outer-emitter leak.
- New shared code lives in a focused subpackage `v2ecoli/colony_bench/`; leave the existing `v2ecoli/colony.py` untouched.
- Study lives at `workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/`. `study.yaml` uses `schema_version: 4`.
- WCM runs are excluded from CI (heavy) behind `@pytest.mark.wcm`; simple-agent smoke tests run in CI.

---

## File Structure

- `v2ecoli/colony_bench/__init__.py` — public exports.
- `v2ecoli/colony_bench/tiers.py` — `cell_factory`, tier builders (`simple`, `wcm`), port contract.
- `v2ecoli/colony_bench/geometries.py` — `free_colony`, `mother_machine`, `daughter_machine` document builders taking a factory.
- `v2ecoli/colony_bench/phenotypes.py` — `Trajectory` type, `phenotype_extractor`, lineage/division helpers.
- `v2ecoli/colony_bench/harness.py` — `build_bench_core`, `run_bench` (build → run → sample → extract).
- `tests/colony_bench/test_tiers.py`, `test_geometries.py`, `test_phenotypes.py`, `test_harness.py`.
- `workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/{study.yaml,README.md,sims/run.py}`.
- `workspace/investigations/colonies/investigation.yaml` — pivot (retitle + Part B studies).

---

### Task 1: Cell-tier factory (`simple` + `wcm`)

**Files:**
- Create: `v2ecoli/colony_bench/__init__.py`
- Create: `v2ecoli/colony_bench/tiers.py`
- Test: `tests/colony_bench/test_tiers.py`

**Interfaces:**
- Produces: `cell_factory(tier: str, *, rng, env_size: float, seed: int, cache_dir: str = "out/cache", agents_key: str = "cells", ecoli_interval: float = 1.0, init_mass: float | None = None, x: float, y: float, angle: float, length: float = 2.0, radius: float = 0.5, density: float = 0.02) -> tuple[str, dict]` returning `(agent_id, cell_body)`. Supported `tier` values here: `"simple"`, `"wcm"`. Every returned `cell_body` has keys `id`, `location`, `angle`, `mass`, `length`, and either a `grow_divide` process (simple) or an `ecoli` process (wcm), plus `local`/`volume`/`exchange` stores for wcm.

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_tiers.py
import pytest
from viva_munk.processes.multibody import make_rng

def test_simple_tier_has_port_contract():
    from v2ecoli.colony_bench.tiers import cell_factory
    rng = make_rng(0)
    aid, body = cell_factory("simple", rng=rng, env_size=30, seed=0,
                             x=15, y=15, angle=0.0)
    assert isinstance(aid, str)
    assert body["id"] == aid
    assert "location" in body and "mass" in body and "length" in body
    # simple tier drives division via an embedded grow_divide process
    assert body["grow_divide"]["_type"] == "process"
    assert body["grow_divide"]["outputs"]["agents"] == ["..", "..", "cells"]

def test_unknown_tier_raises():
    from v2ecoli.colony_bench.tiers import cell_factory
    rng = make_rng(0)
    with pytest.raises(ValueError, match="unknown tier"):
        cell_factory("bogus", rng=rng, env_size=30, seed=0, x=1, y=1, angle=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_tiers.py -v`
Expected: FAIL — `ModuleNotFoundError: v2ecoli.colony_bench`.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/colony_bench/__init__.py
from v2ecoli.colony_bench.tiers import cell_factory  # noqa: F401
```

```python
# v2ecoli/colony_bench/tiers.py
"""Cell-tier factory — one cell body per model tier, uniform port contract.

Every tier returns (agent_id, cell_body) placeable by any geometry builder.
Port contract: id, location, angle, mass, length (+ volume/exchange for wcm),
and an `agents` division wire ['..','..',agents_key] matching what
GrowDivide / EcoliWCM._handle_division write {_remove,_add} to.
"""
from __future__ import annotations
from typing import Any

from viva_munk.processes.multibody import build_microbe
from viva_munk.processes.grow_divide import make_adder_grow_divide_process


def _simple(rng, env_size, seed, x, y, angle, length, radius, density,
            agents_key, **_):
    agent_id, body = build_microbe(
        rng, env_size=env_size, x=x, y=y, angle=angle,
        length=length, radius=radius, density=density,
        velocity=(0, 0), speed_range=(0, 0),
    )
    body["grow_divide"] = make_adder_grow_divide_process(
        agents_key=agents_key,
    )
    return agent_id, body


def _wcm(rng, env_size, seed, x, y, angle, length, radius, density,
         agents_key, cache_dir, ecoli_interval, init_mass, **_):
    agent_id, body = build_microbe(
        rng, env_size=env_size, x=x, y=y, angle=angle,
        length=length, radius=radius, density=density,
    )
    if init_mass is not None:
        body["mass"] = float(init_mass)
    body["ecoli"] = {
        "_type": "process",
        "address": "local:EcoliWCM",
        "config": {
            "cache_dir": cache_dir, "seed": seed,
            "transport": "local", "init_mass": init_mass, "env_size": env_size,
        },
        "interval": ecoli_interval,
        "inputs": {
            "local": ["local"], "agent_id": ["id"],
            "location": ["location"], "angle": ["angle"],
        },
        "outputs": {
            "mass": ["mass"], "length": ["length"], "volume": ["volume"],
            "exchange": ["exchange"], "agents": ["..", "..", agents_key],
        },
    }
    body.setdefault("local", {})
    body.setdefault("volume", 0.0)
    body.setdefault("exchange", {})
    return agent_id, body


_TIERS = {"simple": _simple, "wcm": _wcm}


def cell_factory(tier: str, *, rng, env_size: float, seed: int,
                 cache_dir: str = "out/cache", agents_key: str = "cells",
                 ecoli_interval: float = 1.0, init_mass: float | None = None,
                 x: float, y: float, angle: float,
                 length: float = 2.0, radius: float = 0.5,
                 density: float = 0.02) -> tuple[str, dict[str, Any]]:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier: {tier!r} (have {sorted(_TIERS)})")
    return _TIERS[tier](
        rng, env_size, seed, x, y, angle, length, radius, density,
        agents_key, cache_dir=cache_dir, ecoli_interval=ecoli_interval,
        init_mass=init_mass,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_tiers.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/colony_bench/__init__.py v2ecoli/colony_bench/tiers.py tests/colony_bench/test_tiers.py
git commit -m "feat(colony_bench): cell-tier factory (simple + wcm) with uniform port contract"
```

---

### Task 2: Geometry builders (free colony, mother machine, daughter machine)

**Files:**
- Create: `v2ecoli/colony_bench/geometries.py`
- Test: `tests/colony_bench/test_geometries.py`

**Interfaces:**
- Consumes: `cell_factory` (Task 1).
- Produces: three builders, each returning a process-bigraph document dict with `cells`, a `multibody` PymunkProcess, an `emitter`, and (machines only) a `remove_crossing`:
  - `free_colony(tier, *, n_cells=2, env_size=30.0, seed=0, cache_dir="out/cache", physics_interval=1.0, ecoli_interval=1.0, init_mass=None, jitter_per_second=1e-4, damping_per_second=0.5, emit_cells=False) -> dict`
  - `mother_machine(tier, *, n_channels=6, env_size=None, seed=0, cache_dir="out/cache", channel_width=1.5, spacer_thickness=0.3, channel_height=20.0, physics_interval=30.0, ecoli_interval=1.0, init_mass=None, emit_cells=False) -> dict`
  - `daughter_machine(tier, *, env_size=30.0, seed=0, cache_dir="out/cache", flow_x=None, physics_interval=30.0, ecoli_interval=1.0, init_mass=None, emit_cells=False) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_geometries.py
import pytest

@pytest.mark.parametrize("builder,kwargs", [
    ("free_colony", dict(n_cells=2)),
    ("mother_machine", dict(n_channels=3)),
    ("daughter_machine", dict()),
])
def test_geometry_builds_simple_document(builder, kwargs):
    from v2ecoli.colony_bench import geometries
    doc = getattr(geometries, builder)("simple", seed=0, **kwargs)
    assert "cells" in doc and len(doc["cells"]) >= 1
    assert doc["multibody"]["address"] == "local:PymunkProcess"
    # every cell carries the simple-tier division process
    for cell in doc["cells"].values():
        assert "grow_divide" in cell

def test_mother_machine_has_barriers_and_removal():
    from v2ecoli.colony_bench import geometries
    doc = geometries.mother_machine("simple", n_channels=4, seed=0)
    assert doc["multibody"]["config"]["barriers"]
    assert "remove_crossing" in doc
    assert len(doc["cells"]) == 4  # one cell per channel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_geometries.py -v`
Expected: FAIL — `ModuleNotFoundError: v2ecoli.colony_bench.geometries`.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/colony_bench/geometries.py
"""Geometry builders — place cells of any tier into a physical device.

Generalized from viva_munk's mother_machine/daughter_machine documents and
v2ecoli.colony.make_colony_document: the cell body comes from cell_factory,
so any tier runs in any geometry.
"""
from __future__ import annotations
import math

import numpy as np
from process_bigraph.emitter import emitter_from_wires
from viva_munk.processes.multibody import make_rng
from viva_munk.processes.remove_crossing import make_remove_crossing_process

from v2ecoli.colony_bench.tiers import cell_factory


def _emitter(emit_cells):
    wires = ({"agents": ["cells"], "time": ["global_time"]}
             if emit_cells else {"time": ["global_time"]})
    return emitter_from_wires(wires)


def _multibody(env_size, physics_interval, extra_config=None):
    config = {"env_size": env_size}
    if extra_config:
        config.update(extra_config)
    return {
        "_type": "process", "address": "local:PymunkProcess",
        "config": config, "interval": physics_interval,
        "inputs": {"segment_cells": ["cells"]},
        "outputs": {"segment_cells": ["cells"]},
    }


def free_colony(tier, *, n_cells=2, env_size=30.0, seed=0,
                cache_dir="out/cache", physics_interval=1.0,
                ecoli_interval=1.0, init_mass=None, jitter_per_second=1e-4,
                damping_per_second=0.5, emit_cells=False):
    rng = make_rng(seed)
    cells = {}
    for i in range(n_cells):
        x = env_size / 2 + rng.uniform(-5, 5)
        y = env_size / 2 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)
        aid, body = cell_factory(
            tier, rng=rng, env_size=env_size, seed=seed + i, x=x, y=y,
            angle=angle, cache_dir=cache_dir, ecoli_interval=ecoli_interval,
            init_mass=init_mass,
        )
        cells[aid] = body
    return {
        "cells": cells,
        "multibody": _multibody(env_size, physics_interval, {
            "jitter_per_second": jitter_per_second,
            "damping_per_second": damping_per_second,
        }),
        "emitter": _emitter(emit_cells),
    }


def mother_machine(tier, *, n_channels=6, env_size=None, seed=0,
                   cache_dir="out/cache", channel_width=1.5,
                   spacer_thickness=0.3, channel_height=20.0,
                   physics_interval=30.0, ecoli_interval=1.0, init_mass=None,
                   emit_cells=False):
    cell_radius, cell_length = 0.5, 2.0
    if env_size is None:
        width = n_channels * (channel_width + spacer_thickness) + spacer_thickness + 2.0
        env_size = float(max(width, channel_height + 5.0))
    barriers, x = [], spacer_thickness
    for _ in range(n_channels + 1):
        barriers.append({"start": (x, 0), "end": (x, channel_height),
                         "thickness": spacer_thickness})
        x += channel_width + spacer_thickness
    rng = make_rng(seed)
    cells, x = {}, spacer_thickness + spacer_thickness / 2
    for i in range(n_channels):
        cx = x + channel_width / 2
        cy = cell_length / 2 + cell_radius + 0.5
        aid, body = cell_factory(
            tier, rng=rng, env_size=env_size, seed=seed + i, x=cx, y=cy,
            angle=math.pi / 2, length=cell_length, radius=cell_radius,
            cache_dir=cache_dir, ecoli_interval=ecoli_interval, init_mass=init_mass,
        )
        cells[aid] = body
        x += channel_width + spacer_thickness
    return {
        "cells": cells,
        "multibody": _multibody(env_size, physics_interval, {
            "elasticity": 0.1, "barriers": barriers,
            "wall_thickness": spacer_thickness,
        }),
        "remove_crossing": make_remove_crossing_process(
            crossing_y=channel_height, agents_key="cells"),
        "emitter": _emitter(emit_cells),
    }


def daughter_machine(tier, *, env_size=30.0, seed=0, cache_dir="out/cache",
                     flow_x=None, physics_interval=30.0, ecoli_interval=1.0,
                     init_mass=None, emit_cells=False):
    if flow_x is None:
        flow_x = env_size * 0.85
    rng = make_rng(seed)
    aid, body = cell_factory(
        tier, rng=rng, env_size=env_size, seed=seed,
        x=env_size / 2, y=env_size / 2, angle=0.0,
        cache_dir=cache_dir, ecoli_interval=ecoli_interval, init_mass=init_mass,
    )
    return {
        "cells": {aid: body},
        "multibody": _multibody(env_size, physics_interval, {"elasticity": 0.1}),
        "remove_crossing": make_remove_crossing_process(
            x_max=flow_x, agents_key="cells"),
        "emitter": _emitter(emit_cells),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_geometries.py -v`
Expected: PASS (4 passed — 3 parametrized + 1).

Note: if `make_remove_crossing_process` rejects the `crossing_y=`/`x_max=` kwargs, open `viva_munk/processes/remove_crossing.py`, read its actual signature, and match it (the mother/daughter machine documents call it the same way — mirror those call sites verbatim).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/colony_bench/geometries.py tests/colony_bench/test_geometries.py
git commit -m "feat(colony_bench): geometry builders (free colony, mother/daughter machine) taking a tier factory"
```

---

### Task 3: Phenotype extractor (lineage, division, panel)

**Files:**
- Create: `v2ecoli/colony_bench/phenotypes.py`
- Test: `tests/colony_bench/test_phenotypes.py`

**Interfaces:**
- Produces:
  - `Trajectory = list[dict]` — each frame `{"time": float, "cells": {cell_id: {"mass": float, "length": float, "volume"?: float, "exchange"?: dict}}}`.
  - `phenotype_extractor(trajectory: Trajectory) -> dict` returning:
    `{"n_division_events": int, "size_at_division": {"length": [float], "mass": [float]}, "added_length": [{"birth_length": float, "delta_length": float}], "interdivision_time": [float], "growth_rate": [float], "exchange": {molecule: mean_flux} | None, "lineage": {daughter_id: mother_id}}`.
- Division detection rule: at frame *t*, if a cell id present at *t-1* disappears and one or more **new** ids appear, that is a division; the disappeared cell is the mother, the new ids are daughters. Size-at-division = mother's last observed `length`/`mass`. `interdivision_time` = time between a cell's birth (first frame it appears) and its own division. `added_length` = mother's division length − mother's birth length, keyed by birth length. `growth_rate` = slope of `log(length)` vs time over a cell's lifetime (≥3 frames), else skipped.

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_phenotypes.py
def _frame(t, cells):
    return {"time": float(t), "cells": cells}

def test_single_division_stats():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    # mother "m" grows 2->4 um over t=0..2, divides at t=3 into d1,d2
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m": {"mass": 150.0, "length": 3.0}}),
        _frame(2, {"m": {"mass": 200.0, "length": 4.0}}),
        _frame(3, {"d1": {"mass": 100.0, "length": 2.0},
                   "d2": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 1
    assert out["size_at_division"]["length"] == [4.0]
    assert out["lineage"] == {"d1": "m", "d2": "m"}
    added = out["added_length"][0]
    assert added["birth_length"] == 2.0 and added["delta_length"] == 2.0

def test_interdivision_time_between_generations():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(10, {"a": {"mass": 100.0, "length": 2.0},
                    "b": {"mass": 100.0, "length": 2.0}}),
        _frame(25, {"a": {"mass": 100.0, "length": 2.0},
                    "c": {"mass": 100.0, "length": 2.0},
                    "d": {"mass": 100.0, "length": 2.0}}),
    ]
    out = phenotype_extractor(traj)
    # "b" was born at t=10, divided at t=25 -> interdivision 15
    assert 15.0 in out["interdivision_time"]

def test_no_divisions_is_empty_panel():
    from v2ecoli.colony_bench.phenotypes import phenotype_extractor
    traj = [
        _frame(0, {"m": {"mass": 100.0, "length": 2.0}}),
        _frame(1, {"m": {"mass": 110.0, "length": 2.2}}),
    ]
    out = phenotype_extractor(traj)
    assert out["n_division_events"] == 0
    assert out["size_at_division"]["length"] == []
    assert out["exchange"] is None  # no exchange present in frames
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_phenotypes.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/colony_bench/phenotypes.py
"""Tier-agnostic phenotype extraction from a sampled colony trajectory.

Reconstructs lineage + division events from cell-id appearance/disappearance
and computes a common phenotype panel. Pure over its input trajectory so it is
unit-testable with synthetic data and reused by every study/tier.
"""
from __future__ import annotations
from typing import Any
import math

Trajectory = list[dict[str, Any]]


def phenotype_extractor(trajectory: Trajectory) -> dict[str, Any]:
    frames = sorted(trajectory, key=lambda f: f["time"])
    birth = {}          # cell_id -> (time, length)
    last_seen = {}      # cell_id -> (time, length, mass)
    length_track = {}   # cell_id -> list[(time, length)] for growth-rate fit
    lineage = {}        # daughter_id -> mother_id
    size_len, size_mass = [], []
    added_length, interdiv = [], []
    exch_sums, exch_n = {}, 0

    prev_ids = set()
    prev_state = {}
    for frame in frames:
        t = frame["time"]
        cells = frame["cells"]
        ids = set(cells)
        for cid, cell in cells.items():
            if cid not in birth:
                birth[cid] = (t, float(cell["length"]))
            last_seen[cid] = (t, float(cell["length"]), float(cell["mass"]))
            length_track.setdefault(cid, []).append((t, float(cell["length"])))
            if "exchange" in cell and isinstance(cell["exchange"], dict):
                for mol, val in cell["exchange"].items():
                    exch_sums[mol] = exch_sums.get(mol, 0.0) + float(val)
                exch_n += 1
        gone = prev_ids - ids
        new = ids - prev_ids
        if gone and new:
            for mother in gone:
                mt, mlen, mmass = last_seen[mother]
                size_len.append(mlen)
                size_mass.append(mmass)
                b_t, b_len = birth[mother]
                added_length.append({"birth_length": b_len,
                                      "delta_length": mlen - b_len})
                interdiv.append(mt - b_t)
                for daughter in new:
                    lineage[daughter] = mother
        prev_ids, prev_state = ids, cells

    growth_rate = []
    for cid, series in length_track.items():
        if len(series) >= 3:
            ts = [p[0] for p in series]
            ys = [math.log(p[1]) for p in series if p[1] > 0]
            if len(ys) == len(ts) and len(ts) >= 3:
                growth_rate.append(_slope(ts, ys))

    exchange = ({m: s / exch_n for m, s in exch_sums.items()}
                if exch_n else None)

    return {
        "n_division_events": len(size_len),
        "size_at_division": {"length": size_len, "mass": size_mass},
        "added_length": added_length,
        "interdivision_time": interdiv,
        "growth_rate": growth_rate,
        "exchange": exchange,
        "lineage": lineage,
    }


def _slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_phenotypes.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/colony_bench/phenotypes.py tests/colony_bench/test_phenotypes.py
git commit -m "feat(colony_bench): phenotype extractor (lineage, division stats, growth, exchange)"
```

---

### Task 4: Run harness (core + build + sample + extract)

**Files:**
- Create: `v2ecoli/colony_bench/harness.py`
- Modify: `v2ecoli/colony_bench/__init__.py` (export `run_bench`, `build_bench_core`)
- Test: `tests/colony_bench/test_harness.py`

**Interfaces:**
- Consumes: geometry builders (Task 2), `phenotype_extractor` (Task 3).
- Produces:
  - `build_bench_core(tier: str)` — returns a bigraph core with viva_munk processes registered, plus (tier=="wcm") `ECOLI_TYPES` + `EcoliWCM` link.
  - `run_bench(geometry: str, tier: str, *, n_ticks: int, dt: float = 1.0, sample_every: int = 1, seed: int = 0, builder_kwargs: dict | None = None) -> dict` — builds the Composite, runs `n_ticks` steps of `dt`, samples `composite.state['cells']` every `sample_every` ticks into a `Trajectory`, and returns `{"trajectory": Trajectory, "phenotypes": <extractor output>, "n_final": int}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_harness.py
import pytest

def test_run_bench_simple_free_colony_smoke():
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench("free_colony", "simple", n_ticks=5, dt=1.0, seed=0,
                    builder_kwargs={"n_cells": 2, "env_size": 30})
    assert out["n_final"] >= 2
    assert len(out["trajectory"]) == 5
    assert "phenotypes" in out and "n_division_events" in out["phenotypes"]

@pytest.mark.wcm
def test_run_bench_wcm_daughter_machine_smoke():
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench("daughter_machine", "wcm", n_ticks=2, dt=1.0, seed=0)
    assert out["n_final"] >= 1
    assert len(out["trajectory"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_harness.py -v -m "not wcm"`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/colony_bench/harness.py
"""Run harness — build a geometry+tier Composite, run, sample, extract.

Samples composite.state['cells'] directly each tick (emit_cells=False) to
avoid the outer-emitter RAM leak, mirroring the colonies-02 perf pattern.
"""
from __future__ import annotations
from typing import Any

from process_bigraph import Composite
from viva_munk import core_import

from v2ecoli.colony_bench import geometries
from v2ecoli.colony_bench.phenotypes import phenotype_extractor

_GEOMETRIES = {
    "free_colony": geometries.free_colony,
    "mother_machine": geometries.mother_machine,
    "daughter_machine": geometries.daughter_machine,
}


def build_bench_core(tier: str):
    core = core_import()  # registers PymunkProcess, GrowDivide, AdderGrowDivide
    if tier == "wcm":
        from v2ecoli.bridge import EcoliWCM
        from v2ecoli.types import ECOLI_TYPES
        core.register_types(ECOLI_TYPES)
        core.register_link("EcoliWCM", EcoliWCM)
    return core


def _sample(state) -> dict[str, dict[str, Any]]:
    out = {}
    for cid, cell in state.get("cells", {}).items():
        rec = {"mass": float(cell.get("mass", 0.0)),
               "length": float(cell.get("length", 0.0))}
        if "volume" in cell:
            rec["volume"] = float(cell.get("volume", 0.0))
        if isinstance(cell.get("exchange"), dict) and cell["exchange"]:
            rec["exchange"] = dict(cell["exchange"])
        out[cid] = rec
    return out


def run_bench(geometry: str, tier: str, *, n_ticks: int, dt: float = 1.0,
              sample_every: int = 1, seed: int = 0,
              builder_kwargs: dict | None = None) -> dict[str, Any]:
    if geometry not in _GEOMETRIES:
        raise ValueError(f"unknown geometry: {geometry!r}")
    core = build_bench_core(tier)
    doc = _GEOMETRIES[geometry](tier, seed=seed, **(builder_kwargs or {}))
    comp = Composite({"state": doc}, core=core)

    trajectory = []
    for tick in range(n_ticks):
        comp.run(dt)
        if tick % sample_every == 0:
            trajectory.append({
                "time": float(comp.state.get("global_time", (tick + 1) * dt)),
                "cells": _sample(comp.state),
            })
    return {
        "trajectory": trajectory,
        "phenotypes": phenotype_extractor(trajectory),
        "n_final": len(comp.state.get("cells", {})),
    }
```

Then add to `v2ecoli/colony_bench/__init__.py`:

```python
from v2ecoli.colony_bench.harness import run_bench, build_bench_core  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_harness.py -v -m "not wcm"`
Expected: PASS (1 passed, 1 deselected).

Then verify the WCM path on the mini or locally if `out/cache` exists:
Run: `.venv/bin/python -m pytest tests/colony_bench/test_harness.py -v -m wcm`
Expected: PASS (heavy; skip in CI).

- [ ] **Step 5: Register the `wcm` marker and commit**

Add to `pyproject.toml` / `pytest.ini` markers (if a markers block exists, append; else create):

```ini
[tool.pytest.ini_options]
markers = ["wcm: heavy full-WCM colony runs; excluded from CI"]
```

```bash
git add v2ecoli/colony_bench/harness.py v2ecoli/colony_bench/__init__.py tests/colony_bench/test_harness.py pyproject.toml
git commit -m "feat(colony_bench): run harness (build/sample/extract) + wcm pytest marker"
```

---

### Task 5: Study scaffold `colonies-04` + `sims/run.py`

**Files:**
- Create: `workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/study.yaml`
- Create: `workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/README.md`
- Create: `workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/sims/run.py`
- Test: `tests/colony_bench/test_study_04_smoke.py`

**Interfaces:**
- Consumes: `run_bench` (Task 4).
- Produces: `sims/run.py` exposing `main(geometry, tier, n_ticks, out_dir)` that runs the harness and writes `phenotypes.json` + a `summary.json` under the study's `runs/` dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_study_04_smoke.py
import json, importlib.util, pathlib

STUDY = pathlib.Path("workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness")

def test_study_run_writes_phenotypes(tmp_path):
    spec = importlib.util.spec_from_file_location("c04run", STUDY / "sims" / "run.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = mod.main(geometry="free_colony", tier="simple", n_ticks=5, out_dir=tmp_path)
    pheno = json.loads((tmp_path / "phenotypes.json").read_text())
    assert "n_division_events" in pheno
    assert out["n_final"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_study_04_smoke.py -v`
Expected: FAIL — run.py does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness/sims/run.py
"""colonies-04 driver — run a geometry+tier, extract the phenotype panel.

Usage:
    .venv/bin/python .../sims/run.py free_colony simple --ticks 60
"""
from __future__ import annotations
import argparse, json, pathlib, sys


def main(*, geometry, tier, n_ticks, out_dir, seed=0, builder_kwargs=None):
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench(geometry, tier, n_ticks=n_ticks, seed=seed,
                    builder_kwargs=builder_kwargs)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phenotypes.json").write_text(json.dumps(out["phenotypes"], indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "geometry": geometry, "tier": tier, "n_ticks": n_ticks,
        "n_final": out["n_final"],
        "n_division_events": out["phenotypes"]["n_division_events"],
    }, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("geometry"); p.add_argument("tier")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    here = pathlib.Path(__file__).resolve().parent.parent
    out_dir = a.out or (here / "runs" / f"{a.geometry}__{a.tier}__seed{a.seed}")
    r = main(geometry=a.geometry, tier=a.tier, n_ticks=a.ticks,
             out_dir=out_dir, seed=a.seed)
    print(f"n_final={r['n_final']} divisions={r['phenotypes']['n_division_events']}")
    sys.exit(0)
```

```yaml
# .../colonies-04-device-phenotype-harness/study.yaml
schema_version: 4
name: colonies-04-device-phenotype-harness
investigation: colonies
title: Device harness + simple-agent phenotype baseline
description: |
  Stand up the shared colony-phenotyping infrastructure — cell-tier factory,
  geometry builders (mother machine, daughter machine, free colony), and the
  tier-agnostic phenotype extractor — and validate the measurement pipeline
  with cheap simple agents before spending WCM compute. Produces the common
  phenotype panel: growth rate, size-at-division, added length, inter-division
  time.
topic: phenotype-quantification
tags: [colony, mother-machine, daughter-machine, phenotype, adder, requires-viva-munk]
created: '2026-07-25'

status: designed
phase: Build
design_status: approved
implementation_status: in_progress
simulation_status: not_run
evaluation_status: not_evaluated
gate_status: pending
expert_review_status: not_requested

acceptance_criteria:
  - behavior: factory-yields-runnable-cell-per-tier
  - behavior: geometry-builders-run-with-simple-agents
  - behavior: extractor-recovers-known-division-stats
  - behavior: harness-produces-phenotype-panel
```

```markdown
<!-- .../colonies-04-device-phenotype-harness/README.md -->
# colonies-04 — Device harness + simple-agent phenotype baseline

Shared infra for the phenotype-quantification pivot. Builds `cell_factory`,
the three geometry builders, and the `phenotype_extractor`, then runs the
mother machine, daughter machine, and free colony with **simple agents** to
validate the pipeline cheaply.

Run: `.venv/bin/python sims/run.py free_colony simple --ticks 60`

Tiers: `simple` (this study), `wcm` (colonies-06), `surrogate` (colonies-05).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_study_04_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workspace/investigations/colonies/studies/colonies-04-device-phenotype-harness tests/colony_bench/test_study_04_smoke.py
git commit -m "feat(colonies-04): device-phenotype-harness study scaffold + simple-agent driver"
```

---

### Task 6: Pivot the investigation manifest to Part B

**Files:**
- Modify: `workspace/investigations/colonies/investigation.yaml`
- Test: `tests/colony_bench/test_investigation_manifest.py`

**Interfaces:**
- Consumes: nothing (doc/config task).
- Produces: `investigation.yaml` retitled, `studies` list extended with `colonies-04..07`, Part B described. Studies 01–03 retained.

- [ ] **Step 1: Write the failing test**

```python
# tests/colony_bench/test_investigation_manifest.py
import yaml, pathlib

INV = pathlib.Path("workspace/investigations/colonies/investigation.yaml")

def test_manifest_lists_part_b_studies():
    data = yaml.safe_load(INV.read_text())
    studies = data["studies"]
    for s in ["colonies-01-hpc-readiness", "colonies-02-parallel-multigen-perf",
              "colonies-03-wcm-rss-leak", "colonies-04-device-phenotype-harness"]:
        assert s in studies, f"missing {s}"
    assert "phenotype" in data["title"].lower() or "phenotype" in data["question"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_investigation_manifest.py -v`
Expected: FAIL — `colonies-04-device-phenotype-harness` not yet in `studies`.

- [ ] **Step 3: Edit the manifest**

In `workspace/investigations/colonies/investigation.yaml`:
1. Change `title:` to `Colony & Microfluidic Phenotype Quantification`.
2. Add a `## Part B` paragraph to `description:` summarizing the pivot: same geometries × three cell-model tiers (simple, surrogate, WCM) × a common phenotype panel; studies 01–03 are retained as Part A (compute foundation), the native RSS leak is now a stated run-length bound not a blocker.
3. Extend the `studies:` list, appending:
   ```yaml
     - colonies-04-device-phenotype-harness
     # - colonies-05-surrogate-agent-tier   # planned; buildable, own plan
     # - colonies-06-wcm-media-geometry      # planned; compute-bounded, own plan
     # - colonies-07-mother-machine-data     # planned; pending real dataset
   ```
4. Leave `acceptance_criteria` for 01–03 intact; append the four colonies-04 behaviors from its `study.yaml`.

Preserve existing keys and ordering; only retitle + append (do not delete Part A content). If a round-trip YAML style matters for the dashboard, edit in place with minimal diff rather than re-serializing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/colony_bench/test_investigation_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest tests/colony_bench -v -m "not wcm"`
Expected: all PASS.

```bash
git add workspace/investigations/colonies/investigation.yaml tests/colony_bench/test_investigation_manifest.py
git commit -m "docs(colonies): pivot investigation to phenotype quantification; add colonies-04 (Part B)"
```

---

## Self-Review

**Spec coverage:**
- Geometry × tier factorization (Approach A) → Tasks 1–2. ✓
- Uniform port contract → Task 1 (contract) + Global Constraints. ✓
- `SurrogateProcess` / surrogate tier → **deferred to the colonies-05 plan** (spec scopes it as its own study; this plan builds `simple`+`wcm` only). Stated in goal + Task 1 tier set.
- `phenotype_extractor` (lineage, size-at-division, added length, inter-division time, exchange) → Task 3. ✓
- Studies: colonies-04 → Task 5; 05/06/07 → planned entries in Task 6 (own plans). ✓
- Testing (factory/extractor units, geometry smoke in CI, WCM behind marker) → Tasks 1–5 + marker in Task 4. ✓
- Investigation pivot → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step has real code. The one deliberate deferral (surrogate tier, media-matrix, real-data) is scoped out with named follow-up plans, not left as an in-plan placeholder.

**Type consistency:** `cell_factory(tier, ...)` signature identical across Tasks 1/2/4. `run_bench(...)` return dict keys (`trajectory`, `phenotypes`, `n_final`) consistent Tasks 4/5. `phenotype_extractor` output keys consistent Tasks 3/4/5/6. `Trajectory` frame shape (`time`, `cells`) identical across Tasks 3/4.

## Follow-up plans (not this plan)
- **colonies-05:** `SurrogateProcess` wrapping the linear growth emulator + surrogate tier in the factory; comparison vs simple.
- **colonies-06:** WCM across media {minimal, +AA, rich} × geometry; exchange/media panel; short-window (RSS-bounded) runs.
- **colonies-07:** real mother-machine data loader + phenotype comparison (pending dataset).
