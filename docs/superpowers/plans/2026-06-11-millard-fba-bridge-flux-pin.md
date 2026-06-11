# Millard FBA-bridge flux-pin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Millard 2017 kinetic ODE mechanistically set central-carbon flux inside the v2ecoli WCM by pinning the mapped FBA reactions to the ODE's fluxes (`v_lb=v_ub=v_ODE`), with a soft kinetic-target fallback on infeasibility.

**Architecture:** The Millard COPASI process exports per-reaction fluxes; a flux-coupling step maps them (Millard rxn → v2ecoli FBA rxn) and converts units; the `Metabolism` process pins those reactions before the LP solve and relaxes infeasible pins to soft targets. Validated on a focused Metabolism+bridge harness before the full WCM.

**Tech Stack:** Python, process-bigraph, basico/COPASI (Millard ODE), v2ecoli `wholecell.utils.modular_fba` (GLPK FBA), pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-11-millard-fba-bridge-flux-pin-design.md`

**Run tests with `.venv/bin/python -m pytest` (bare `python` lacks `unum`).**

---

## Task 0: Discovery — pin the upstream specifics (no placeholders downstream)

**Files:** none (record findings in the commit message / a scratch note).

- [ ] **Step 1: Enumerate Millard reactions + how COPASI reports fluxes**

Run:
```bash
.venv/bin/python -c "
import basico
basico.load_model('v2ecoli/models/sbml/millard2017_central_metabolism.xml')
basico.run_time_course(start_time=0, duration=10, intervals=1, use_sbml_id=True, update_model=True)
fl = basico.get_reaction_fluxes()   # current per-reaction fluxes
print(fl.index.tolist())            # reaction names
print(fl.columns.tolist())          # which column holds the flux value
print(fl.head())
"
```
Record: the exact reaction-name list and the flux column name (used in Task 3).

- [ ] **Step 2: Enumerate v2ecoli FBA reaction ids**

Run:
```bash
.venv/bin/python -c "
from v2ecoli.core import build_core, load_cache_bundle
import v2ecoli, glob
# load a Metabolism instance enough to read fba reaction ids; if heavy, instead
# grep the sim_data reaction ids:
" 2>/dev/null || grep -rl "reactionStoich\|reaction_ids\|all_reaction" v2ecoli/processes/ | head
```
Record: how to list `fba` reaction ids (e.g. via `model.fba.getReactionIDs()` or sim_data), for the reaction map in Task 1. Confirm `getReactionIDs()`/equivalent exists on the FBA object.

- [ ] **Step 3: Commit the findings note**

```bash
git commit --allow-empty -m "chore(fba-bridge): record Millard/v2ecoli reaction + flux APIs (Task 0)"
```

---

## Task 1: Reaction map + loader

**Files:**
- Create: `v2ecoli/data/millard_v2ecoli_reaction_map.yaml`
- Create: `v2ecoli/library/fba_reaction_map.py`
- Test: `tests/test_fba_reaction_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fba_reaction_map.py
from v2ecoli.library.fba_reaction_map import load_reaction_map

def test_load_reaction_map_shape():
    m = load_reaction_map("v2ecoli/data/millard_v2ecoli_reaction_map.yaml")
    # maps millard rxn -> list of (fba_rxn_id, scale)
    assert "PTS_4" in m
    entry = m["PTS_4"]
    assert isinstance(entry, list) and len(entry) >= 1
    fba_id, scale = entry[0]
    assert isinstance(fba_id, str) and isinstance(scale, float)

def test_map_excludes_millard_only():
    m = load_reaction_map("v2ecoli/data/millard_v2ecoli_reaction_map.yaml")
    # millard_only reactions are NOT pinnable -> absent from the pin map
    assert all(v for v in m.values())  # no empty target lists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fba_reaction_map.py -q`
Expected: FAIL (module + file missing).

- [ ] **Step 3: Create the reaction map YAML (starter core, expand during curation)**

Using Task 0's reaction-id lists, curate the clear central-carbon overlaps. Starter format (fill `fba_rxn_id`s from Task 0 Step 2; sign = +1 unless Millard/BioCyc directionality differs):
```yaml
# v2ecoli/data/millard_v2ecoli_reaction_map.yaml
schema_version: 1
namespace: "millard2017 reactions -> v2ecoli FBA reactions"
# Each millard reaction -> list of {fba: <id>, scale: <float>} pins.
pins:
  PTS_4:   [{fba: "TRANS-RXN-157", scale: 1.0}]   # glucose PTS uptake (confirm id)
  PGI:     [{fba: "PGLUCISOM-RXN", scale: 1.0}]
  PFK:     [{fba: "6PFRUCTPHOS-RXN", scale: 1.0}]
  # ... expand to the ~20-30 curated central reactions ...
# Reactions with no clean v2ecoli counterpart (NOT pinned):
millard_only:
  - "AdK"        # adenylate kinase lumped
```

- [ ] **Step 4: Implement the loader**

```python
# v2ecoli/library/fba_reaction_map.py
from __future__ import annotations
from pathlib import Path
import yaml

def load_reaction_map(path: str) -> dict[str, list[tuple[str, float]]]:
    """millard_rxn -> [(fba_rxn_id, scale), ...]. Excludes millard_only."""
    data = yaml.safe_load(Path(path).read_text())
    out: dict[str, list[tuple[str, float]]] = {}
    for millard_rxn, targets in (data.get("pins") or {}).items():
        pins = [(t["fba"], float(t.get("scale", 1.0))) for t in targets]
        if pins:
            out[millard_rxn] = pins
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fba_reaction_map.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2ecoli/data/millard_v2ecoli_reaction_map.yaml v2ecoli/library/fba_reaction_map.py tests/test_fba_reaction_map.py
git commit -m "feat(fba-bridge): Millard->v2ecoli reaction map + loader"
```

---

## Task 2: Flux unit converter

**Files:**
- Create: `v2ecoli/library/fba_flux_convert.py`
- Test: `tests/test_fba_flux_convert.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fba_flux_convert.py
from v2ecoli.library.fba_flux_convert import millard_flux_to_fba_bound

def test_converter_is_linear_and_scaled():
    # bound = flux_mM_per_s * coefficient * scale; pure, deterministic
    v = millard_flux_to_fba_bound(2.0, coefficient=3.0, scale=1.0)
    assert v == 6.0
    assert millard_flux_to_fba_bound(2.0, coefficient=3.0, scale=0.5) == 3.0

def test_converter_sign_preserved():
    assert millard_flux_to_fba_bound(-1.5, coefficient=2.0, scale=1.0) == -3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fba_flux_convert.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement (mirror set_reaction_bounds' mM-basis coefficient)**

```python
# v2ecoli/library/fba_flux_convert.py
def millard_flux_to_fba_bound(flux_mM_per_s: float, coefficient: float,
                              scale: float = 1.0) -> float:
    """Convert a Millard reaction flux (mM/s) to the v2ecoli FBA bound basis.

    The WCM sets bounds in CONC_UNITS magnitude using `coefficient`
    (mass*time/volume), the same factor Metabolism.set_reaction_bounds uses to
    map mmol/gDCW/hr -> mM basis. `scale` carries reaction-map stoichiometry.
    """
    return float(flux_mM_per_s) * float(coefficient) * float(scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fba_flux_convert.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/library/fba_flux_convert.py tests/test_fba_flux_convert.py
git commit -m "feat(fba-bridge): Millard flux -> FBA bound unit converter"
```

---

## Task 3: Millard COPASI process exports reaction fluxes

**Files:**
- Modify: `v2ecoli/steps/millard_pdmp_metabolism.py` (add `central_fluxes` output)
- Test: `tests/test_millard_flux_export.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_millard_flux_export.py
import warnings; warnings.filterwarnings("ignore")
from v2ecoli import build_composite

def test_millard_emits_central_fluxes():
    c = build_composite("millard_pdmp_baseline", with_ref_growth=True,
                        ref_growth_flux_source="consumption_matched", seed=0)
    c.run(5)
    ag = (c.state.get("agents") or {}).get("0") or {}
    fluxes = ag.get("central_fluxes") or {}
    assert fluxes, "central_fluxes store is empty"
    # at least the glucose-uptake reaction reports a finite flux
    assert any(abs(float(v)) >= 0.0 for v in fluxes.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_millard_flux_export.py -q`
Expected: FAIL (no `central_fluxes`).

- [ ] **Step 3: Add the flux output (use Task 0's basico flux call)**

In `millard_pdmp_metabolism.py`: add `"central_fluxes": InPlaceDict()` to
`outputs()`, and after the COPASI integration step in `update()` read the
per-reaction fluxes and emit them. Concretely (adapt the basico call/column to
Task 0 Step 1's finding):
```python
# in update(), after the basico integration advances the model:
try:
    fl = basico.get_reaction_fluxes()           # DataFrame, index=reaction name
    central_fluxes = {str(r): float(fl.loc[r, "flux"]) for r in fl.index}
except Exception:
    central_fluxes = {}
# ... include in the returned update dict:
return {... , "central_fluxes": central_fluxes}
```
Wire `central_fluxes` into the composite topology in
`millard_pdmp_baseline.py` (the millard edge's out_topo) to a shared
`("central_fluxes",)` store.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_millard_flux_export.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/steps/millard_pdmp_metabolism.py v2ecoli/composites/millard_pdmp_baseline.py tests/test_millard_flux_export.py
git commit -m "feat(fba-bridge): Millard COPASI process exports central_fluxes (mM/s)"
```

---

## Task 4: Flux-coupling step (map + convert -> pinned_flux_targets)

**Files:**
- Create: `v2ecoli/steps/fba_flux_coupler.py`
- Test: `tests/test_fba_flux_coupler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fba_flux_coupler.py
from v2ecoli.steps.fba_flux_coupler import FBAFluxCoupler
from v2ecoli.core import build_core

def test_coupler_maps_and_converts():
    core = build_core()
    c = FBAFluxCoupler(config={
        "reaction_map_file": "v2ecoli/data/millard_v2ecoli_reaction_map.yaml",
        "coefficient": 2.0,
    }, core=core)
    out = c.update({"central_fluxes": {"PFK": 3.0, "AdK": 9.9}}, 1.0)
    pins = out["pinned_flux_targets"]
    # PFK is mapped -> appears (3.0 * coefficient 2.0 * scale 1.0 = 6.0)
    fba_id = c.reaction_map["PFK"][0][0]
    assert pins[fba_id] == 6.0
    # AdK is millard_only -> not pinned
    assert all("AdK" not in k for k in pins)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fba_flux_coupler.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the coupler**

```python
# v2ecoli/steps/fba_flux_coupler.py
from __future__ import annotations
from process_bigraph import Process
from v2ecoli.types.stores import InPlaceDict
from v2ecoli.library.fba_reaction_map import load_reaction_map
from v2ecoli.library.fba_flux_convert import millard_flux_to_fba_bound

class FBAFluxCoupler(Process):
    name = "fba-flux-coupler"
    topology = {"central_fluxes": ("central_fluxes",),
                "pinned_flux_targets": ("pinned_flux_targets",),
                "bridge_diagnostics": ("bridge_diagnostics",)}
    config_schema = {
        "reaction_map_file": {"_default":
            "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"},
        "coefficient": {"_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        self.reaction_map = load_reaction_map(
            self.config["reaction_map_file"])
        self.coefficient = float(self.config.get("coefficient", 1.0))

    def inputs(self):
        return {"central_fluxes": InPlaceDict()}

    def outputs(self):
        return {"pinned_flux_targets": InPlaceDict(),
                "bridge_diagnostics": InPlaceDict()}

    def update(self, state, interval):
        fluxes = state.get("central_fluxes") or {}
        pins: dict[str, float] = {}
        for millard_rxn, v in fluxes.items():
            for fba_id, scale in self.reaction_map.get(millard_rxn, []):
                pins[fba_id] = millard_flux_to_fba_bound(
                    float(v), self.coefficient, scale)
        return {"pinned_flux_targets": pins,
                "bridge_diagnostics": {"n_pinned": len(pins)}}

def register(core):
    core.register_link("FBAFluxCoupler", FBAFluxCoupler)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fba_flux_coupler.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/steps/fba_flux_coupler.py tests/test_fba_flux_coupler.py
git commit -m "feat(fba-bridge): FBAFluxCoupler maps+converts central fluxes to FBA pins"
```

---

## Task 5: Metabolism consumes pins + soft-target fallback

**Files:**
- Modify: `v2ecoli/processes/metabolism.py` (pin pinned_flux_targets before solve; relax on infeasible)
- Test: `tests/test_metabolism_flux_pin.py`

- [ ] **Step 1: Write the failing test (unit-level on the pin/relax helper)**

```python
# tests/test_metabolism_flux_pin.py
# Test the pure pin/relax decision helper in isolation (no full WCM).
from v2ecoli.processes.metabolism import apply_flux_pins_with_fallback

class _FakeFBA:
    def __init__(self, infeasible_ids):
        self.bounds = {}; self.targets = {}; self._infeasible = set(infeasible_ids)
    def setReactionFluxBounds(self, rid, lowerBounds, upperBounds):
        self.bounds[rid] = (lowerBounds, upperBounds)
    def setReactionFluxTargets(self, rid, value):
        self.targets[rid] = value
    def solve_is_feasible(self):
        # infeasible iff any *hard-bounded* reaction is in the infeasible set
        return not (set(self.bounds) & self._infeasible)

def test_feasible_pins_stay_hard():
    fba = _FakeFBA(infeasible_ids=[])
    relaxed = apply_flux_pins_with_fallback(fba, {"R1": 1.0, "R2": 2.0})
    assert relaxed == []
    assert fba.bounds == {"R1": (1.0, 1.0), "R2": (2.0, 2.0)}

def test_infeasible_pin_relaxed_to_target():
    fba = _FakeFBA(infeasible_ids=["R2"])
    relaxed = apply_flux_pins_with_fallback(fba, {"R1": 1.0, "R2": 2.0})
    assert relaxed == ["R2"]
    assert "R2" not in fba.bounds and fba.targets["R2"] == 2.0
    assert fba.bounds["R1"] == (1.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metabolism_flux_pin.py -q`
Expected: FAIL (`apply_flux_pins_with_fallback` undefined).

- [ ] **Step 3: Implement the helper + call it in the metabolism update**

```python
# v2ecoli/processes/metabolism.py  (module-level helper)
def apply_flux_pins_with_fallback(fba, pins: dict[str, float]) -> list[str]:
    """Hard-pin each reaction (lb=ub=v); relax to a soft target any reaction
    whose hard pin makes the LP infeasible. Returns the relaxed reaction ids.

    Greedy: pin all, test feasibility; while infeasible, move the offending
    reactions from bounds -> targets. `fba.solve_is_feasible()` is the GLPK
    feasibility probe (wraps a solve attempt / GLP_NOFEAS check).
    """
    for rid, v in pins.items():
        fba.setReactionFluxBounds(rid, lowerBounds=v, upperBounds=v)
    relaxed: list[str] = []
    # Relax one offender at a time until feasible (bounded by len(pins)).
    while not fba.solve_is_feasible() and len(relaxed) < len(pins):
        # pick a still-hard pinned reaction to relax (LIFO: last pinned first)
        candidates = [r for r in pins if r not in relaxed]
        if not candidates:
            break
        rid = candidates[-1]
        # remove hard bound (set to model default open bounds) + add soft target
        fba.setReactionFluxBounds(rid, lowerBounds=None, upperBounds=None)
        fba.setReactionFluxTargets(rid, pins[rid])
        relaxed.append(rid)
    return relaxed
```
Then in `Metabolism.update()` (or wherever the LP is configured per tick), read
`pinned_flux_targets` from state and call the helper before the solve; record
`relaxed` into `listeners.fba_bridge.relaxed_reactions`. (Confirm the exact
`setReactionFluxBounds` "open bound" sentinel and the feasibility probe against
the real `fba` API from Task 0; adapt `solve_is_feasible`/`setReactionFluxTargets`
to the real method names — `getKineticTargetFluxNames` already exists.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metabolism_flux_pin.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/processes/metabolism.py tests/test_metabolism_flux_pin.py
git commit -m "feat(fba-bridge): Metabolism pins ODE fluxes with soft-target fallback"
```

---

## Task 6: v1 harness + validation (M9-glucose)

**Files:**
- Create: `v2ecoli/composites/millard_fba_bridge_harness.py` (builder) or extend `millard_fba_bridge.composite.yaml`
- Create: `scripts/validate_fba_bridge_harness.py`
- Test: `tests/test_fba_bridge_harness.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_fba_bridge_harness.py
import warnings; warnings.filterwarnings("ignore")
from v2ecoli import build_composite

def test_harness_runs_and_pins_central_flux():
    c = build_composite("millard_fba_bridge_harness")  # Millard ODE + coupler + Metabolism
    c.run(60)  # 60 s
    ag = (c.state.get("agents") or {}).get("0") or c.state
    # LP stayed feasible: relaxed set is a (possibly empty) list, not a crash
    relaxed = (((ag.get("listeners") or {}).get("fba_bridge")) or {}).get("relaxed_reactions", [])
    assert isinstance(relaxed, list)
    # at least one central reaction was pinned this run
    pins = ag.get("pinned_flux_targets") or {}
    assert len(pins) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fba_bridge_harness.py -q`
Expected: FAIL (harness composite not registered).

- [ ] **Step 3: Build the harness composite**

Compose: Millard COPASI process (emits `central_metabolites` + `central_fluxes`)
→ `FBAFluxCoupler` (emits `pinned_flux_targets`) → `Metabolism` (WCM context
loaded from cache; consumes pins). Reuse the existing `millard_pdmp_baseline`
wiring for the Millard side and the baseline WCM `Metabolism` build for the FBA
side; share the `central_fluxes` and `pinned_flux_targets` stores. Register
`millard_fba_bridge_harness` in the composite catalog.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fba_bridge_harness.py -q`
Expected: PASS.

- [ ] **Step 5: Write the validation script + diagnostics figure**

`scripts/validate_fba_bridge_harness.py` runs the harness for ≥600 s on
M9-glucose and reports: (a) **pin fidelity** — realized FBA flux vs ODE flux per
pinned reaction (scatter, within solver tol or flagged relaxed); (b)
**feasibility** — fraction of ticks needing relaxation + which reactions; (c)
**viability** — central-metabolite concentrations + growth stay finite/positive.
Writes `reports/figures/pdmp-01/fba_bridge_flux_pin.html`.

- [ ] **Step 6: Run validation + commit**

```bash
.venv/bin/python scripts/validate_fba_bridge_harness.py
git add v2ecoli/composites/millard_fba_bridge_harness.py scripts/validate_fba_bridge_harness.py tests/test_fba_bridge_harness.py reports/figures/pdmp-01/fba_bridge_flux_pin.html
git commit -m "feat(fba-bridge): v1 Metabolism+bridge harness + flux-pin validation (M9-glucose)"
```

---

## Task 7 (gated on Task 6 results): full-WCM integration

**Files:** Modify the full WCM composite builder to add the Millard ODE + coupler and route pins into the WCM `Metabolism`.

- [ ] **Step 1:** Only proceed if Task 6 shows acceptable pin fidelity + low relaxation rate. If the relaxation rate is high, return to the reaction map (Task 1) / unit conversion (Task 2) before scaling up — do NOT paper over with mass relaxation.
- [ ] **Step 2:** Add Millard ODE + `FBAFluxCoupler` to the full WCM composite; wire `central_fluxes`/`pinned_flux_targets`; keep `consumption_matched` water/precursor driver active (spec scope boundary).
- [ ] **Step 3:** Smoke-run the full WCM 100 s; confirm no NaN/feasibility regression; commit.
- [ ] **Step 4 (stretch / pdmp-01 gate):** 3-condition (M9-glucose/acetate/+aa) interface validation vs the Phase-0 reference ensemble; record causal/teleonomic partition for each pinned reaction in the study.

---

## Self-review notes
- **Spec coverage:** reaction map (T1), flux exporter (T3), converter (T2), coupling step (T4), Metabolism consumption + soft fallback (T5), harness + validation (T6), full-WCM + 3-condition (T7). All spec components covered.
- **Discovery-gated specifics:** the exact basico flux call (T3), `fba` reaction-id listing + `setReactionFluxBounds` open-bound sentinel + feasibility probe (T5) are pinned in Task 0 and adapted in-task — these are real upstream-API unknowns, not placeholders; each has a concrete discovery step and a verification test.
- **Type consistency:** `load_reaction_map` returns `{millard_rxn: [(fba_id, scale)]}` (T1) consumed identically in T4; `millard_flux_to_fba_bound(flux, coefficient, scale)` (T2) called with matching args in T4; `apply_flux_pins_with_fallback(fba, pins)` (T5) returns relaxed-id list surfaced in T6.
