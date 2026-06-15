# Millard Kinetic Metabolism in the Bioreactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Millard 2017 kinetic ODE as the WCM's central-carbon metabolism (growth still from the WCM), make it responsive to the bioreactor environment, validate it with a reusable WCM mass-conservation gate, and (gated on mbp-03) compare it to the plain WCM against Beulig batch.

**Architecture:** Build on the existing `v2ecoli/composites/millard_pdmp_baseline.py`, which already swaps `ecoli-metabolism` for `MillardPDMPMetabolism` and writes real structured `bulk` deltas inline (`delta_mode`). Two additions make it a bioreactor cell: drop the pdmp LQR controller, and add an external-concentration input so the Millard ODE sees the reactor's glucose/O₂ each tick (via basico `setInitialConcentration`, the supported state-overwrite path). Mass conservation is a **native `derived` behavior test** (no custom evaluator) backed by a small mass-balance listener.

**Tech Stack:** Python, process-bigraph, basico/COPASI (`pbg_copasi`), polars, pytest. Run tests with `./.venv/bin/python -m pytest`. Composites built via `v2ecoli.build_composite`; runs via `v2ecoli.library.sqlite_run.run_multigen_sqlite`.

**Pre-flight:** `out/cache` must exist (ParCa cache). If missing: `./.venv/bin/python scripts/build_cache.py --mode full`. Sim tests are marked `@pytest.mark.sim`.

---

## File structure

| File | Responsibility | New/Edit |
|---|---|---|
| `v2ecoli/steps/millard_pdmp_metabolism.py` | add external-concentration input + boundary-set before integrate | Edit |
| `v2ecoli/composites/baseline_millard.py` | LQR-free, env-responsive Millard cell composite generator | New |
| `v2ecoli/steps/derivers/mass_balance_listener.py` | emit cumulative cell-mass-delta + net-exchange-mass | New |
| `tests/test_baseline_millard.py` | composite build + env-responsiveness + growth tests | New |
| `tests/test_mass_balance_listener.py` | listener emits the two cumulative series | New |
| `workspace/studies/mbp-07-millard-kinetic-metabolism-swap/study.yaml` | Build study | New |
| `workspace/studies/mbp-08-millard-swap-validation/study.yaml` | Validation study + mass gate | New |
| `workspace/studies/mbp-01-time-varying-environment/study.yaml` | fix `derived_ratio` → native `derived` | Edit |
| `workspace/studies/mbp-04-multigeneration-runs/study.yaml` | add mass-conservation gate | Edit |
| `investigations/multiscale-bioprocess/investigation.yaml` | register mbp-07/08, at_a_glance + acceptance_criteria | Edit |

---

## Task 1: Scaffold the mbp-07 build study

**Files:**
- Create: `workspace/studies/mbp-07-millard-kinetic-metabolism-swap/study.yaml`

- [ ] **Step 1: Write the study.yaml**

Model it on `workspace/studies/mbp-01-time-varying-environment/study.yaml` (same v3 shape). Minimum fields:

```yaml
schema_version: 3
name: mbp-07-millard-kinetic-metabolism-swap
created: '2026-06-14'
phase: Build
study_kind: construction

design_status: drafted
implementation_status: not_started
simulation_status: not_run
evaluation_status: not_evaluated
expert_review_status: not_requested

baseline:
- name: millard-cell
  composite: v2ecoli.composites.baseline_millard
  params: {seed: 0, cache_dir: out/cache}

purpose:
  question: |
    Can v2ecoli's central-carbon metabolism be replaced by the Millard 2017
    kinetic ODE (growth still from the WCM) and made responsive to an external
    environment, as a drop-in cell engine under the cell-side interface contract?

pipeline_gate:
  prerequisites: []

behavior_tests:
- name: composite-builds-and-runs
  measure: {kind: derived, formula: "listeners.mass.cell_mass", window: full_lineage_from_gen_0}
  pass_if: {op: ">", value: 0.0}
  acceptance_form: qualitative_direction
- name: central-fluxes-nontrivial
  measure: {kind: derived, formula: "central_fluxes.PGI", window: full_lineage_from_gen_0}
  pass_if: {op: "!=", value: 0.0}
  acceptance_form: qualitative_direction

tests:
  pytest_args: [tests/test_baseline_millard.py]
```

- [ ] **Step 2: Validate it parses**

Run: `./.venv/bin/python -c "import yaml; yaml.safe_load(open('workspace/studies/mbp-07-millard-kinetic-metabolism-swap/study.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add workspace/studies/mbp-07-millard-kinetic-metabolism-swap/study.yaml
git commit -m "feat(mbp-07): scaffold Millard kinetic-metabolism-swap build study"
```

---

## Task 2: Add an external-concentration input to the Millard step (env responsiveness)

**Files:**
- Modify: `v2ecoli/steps/millard_pdmp_metabolism.py` (class `MillardPDMPMetabolism`, `inputs()` ~line 90, `update()` ~line 203)
- Test: `tests/test_baseline_millard.py`

Context: `MillardPDMPMetabolism.update()` integrates the Millard model each tick. Today `inputs()` declares `lqr_control`, `bulk`, `listeners_mass` — no environment. We add an optional `external_concentrations` input (SBML-ID → mM) and, when present, overwrite those species in the COPASI model before integrating, using the same basico path `CopasiUTCProcess` uses (`setInitialConcentration` via `_set_initial_concentrations`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_millard.py
import pytest

@pytest.mark.sim
def test_millard_step_accepts_external_concentrations():
    """The Millard metabolism step exposes an external_concentrations input port."""
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    step = MillardPDMPMetabolism(config={}, core=None)
    assert "external_concentrations" in step.inputs()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_millard_step_accepts_external_concentrations -v`
Expected: FAIL — `KeyError`/assert (port absent).

- [ ] **Step 3: Add the input port**

In `inputs()` add:

```python
"external_concentrations": {
    "_type": "node",
    "_default": {},   # SBML species id -> mM; empty = no override this tick
},
```

- [ ] **Step 4: Apply the override before integrating**

In `update()`, before the model is advanced, add (use the model handle the class already holds — see how it loads `model_source`; basico exposes `set_species(name, initial_concentration=...)` or `setInitialConcentration`):

```python
ext = inputs.get("external_concentrations") or {}
if ext:
    # Overwrite ONLY the named boundary species; internal metabolites carry over.
    for sbml_id, conc_mM in ext.items():
        name = self._sbml_to_name.get(sbml_id)  # build this map at setup like CopasiUTCProcess.sbml_to_name
        if name is not None:
            basico.set_species(name, initial_concentration=float(conc_mM), model=self._dm)
```

If the class does not already keep `_sbml_to_name`/`_dm`, build them in setup mirroring `pbg_copasi/processes.py` `_set_initial_concentrations` (lines ~29-49) and `sbml_to_name`.

- [ ] **Step 5: Make the port test pass**

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_millard_step_accepts_external_concentrations -v`
Expected: PASS.

- [ ] **Step 6: Add a responsiveness behavior test**

```python
@pytest.mark.sim
def test_millard_uptake_responds_to_external_glucose():
    """Lower external glucose -> lower glucose-uptake flux (kinetic responsiveness)."""
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    glc_id = "GLCx"  # DISCOVERY: confirm the external-glucose SBML id from
                     # v2ecoli/data/millard_v2ecoli_species_map.yaml (grep glucose)
    def run_with(conc):
        s = MillardPDMPMetabolism(config={}, core=None)
        out = s.update({"external_concentrations": {glc_id: conc},
                        "bulk": None, "lqr_control": {},
                        "listeners_mass": {"cell_mass": 1000.0, "dry_mass": 300.0}}, 1.0)
        return out["central_fluxes"]
    hi = run_with(20.0); lo = run_with(0.1)
    uptake = "PTS_4"  # glucose PTS uptake reaction
    assert abs(lo[uptake]) < abs(hi[uptake])
```

- [ ] **Step 7: Run + verify** (adjust `glc_id`/`uptake` to the real ids if the assert errors on KeyError)

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_millard_uptake_responds_to_external_glucose -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add v2ecoli/steps/millard_pdmp_metabolism.py tests/test_baseline_millard.py
git commit -m "feat(millard): env-responsive external_concentrations input on metabolism step"
```

---

## Task 3: `baseline_millard` composite — LQR-free, env-responsive

**Files:**
- Create: `v2ecoli/composites/baseline_millard.py`
- Test: `tests/test_baseline_millard.py`

Context: copy the structure of `millard_pdmp_baseline.py` (the `@composite_generator` function at ~line 828, `build_execution_layers`, `_build_millard_pdmp_edge`). Two deltas: (1) omit the `lqr-controller` edge and the `lqr_control` wiring (use the JAX backend, which has no LQR — `v2ecoli/steps/millard_pdmp_metabolism_jax.py` — or pass an LQR-disabled flag); (2) wire the new `external_concentrations` input of the metabolism edge to the cell's environment store (`("environment", "external_concentrations")` or the boundary path baseline.py uses — confirm against `baseline.py` env wiring).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.sim
def test_baseline_millard_builds():
    from v2ecoli import build_composite
    from process_bigraph.composite import Composite
    comp = build_composite("baseline_millard", seed=0, cache_dir="out/cache")
    assert isinstance(comp, Composite)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_baseline_millard_builds -v`
Expected: FAIL — composite `baseline_millard` not registered.

- [ ] **Step 3: Write the composite generator**

Create `v2ecoli/composites/baseline_millard.py`. Start from `millard_pdmp_baseline.py`; in the edge builder set `in_topo` without `lqr_control` and add `"external_concentrations": ("environment", "external_concentrations")`; in `out_topo` drop `control_applied`; remove `lqr-controller` from `MILLARD_EDGES` and the execution layers. Register via `@composite_generator(name="baseline_millard", ...)`.

- [ ] **Step 4: Verify build passes**

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_baseline_millard_builds -v`
Expected: PASS.

- [ ] **Step 5: Smoke-run test (growth + fluxes)**

```python
@pytest.mark.sim
def test_baseline_millard_runs_and_grows():
    from v2ecoli import build_composite
    c = build_composite("baseline_millard", seed=0, cache_dir="out/cache")
    c.run(5)
    ag = (c.state.get("agents") or {}).get("0") or {}
    assert (ag.get("listeners", {}).get("mass", {}).get("cell_mass", 0.0)) > 0.0
    assert ag.get("central_fluxes")
```

- [ ] **Step 6: Run + verify**

Run: `./.venv/bin/python -m pytest tests/test_baseline_millard.py::test_baseline_millard_runs_and_grows -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/composites/baseline_millard.py tests/test_baseline_millard.py
git commit -m "feat(baseline_millard): LQR-free env-responsive Millard cell composite"
```

---

## Task 4: Mass-balance listener (emit cumulative series for the native gate)

**Files:**
- Create: `v2ecoli/steps/derivers/mass_balance_listener.py`
- Test: `tests/test_mass_balance_listener.py`

Context: mirror `v2ecoli/steps/derivers/mass_deriver.py` (a `Step` emitting `listeners.<group>.<name>` with `overwrite[...]` typed ports). Emit two cumulative scalars under `listeners.mass_balance`: `cumulative_cell_mass_delta_fg` (running Σ of per-tick Δ`cell_mass`) and `cumulative_net_exchange_mass_fg` (running Σ of net import−export mass over the tick, derived from `listeners.fba_results.external_exchange_fluxes` × molar masses × dry_mass × dt). These become the `formula` tokens for the native mass-conservation test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mass_balance_listener.py
import pytest

@pytest.mark.sim
def test_mass_balance_listener_emits_cumulative_series():
    from v2ecoli import build_composite
    c = build_composite("baseline", seed=0, cache_dir="out/cache",
                        features=["mass_balance_listener"])
    c.run(5)
    ml = (c.state["agents"]["0"]["listeners"].get("mass_balance") or {})
    assert "cumulative_cell_mass_delta_fg" in ml
    assert "cumulative_net_exchange_mass_fg" in ml
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_mass_balance_listener.py -v`
Expected: FAIL — feature/listener absent.

- [ ] **Step 3: Implement the listener Step**

```python
# v2ecoli/steps/derivers/mass_balance_listener.py
from process_bigraph import Step

NAME = "ecoli-mass-balance-listener"

class MassBalanceListener(Step):
    """Emit cumulative cell-mass delta and cumulative net exchange mass (fg),
    so a native `derived` behavior test can check closed-balance conservation."""
    name = NAME

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._prev_cell_mass = None
        self._cum_delta = 0.0
        self._cum_exch = 0.0

    def inputs(self):
        return {
            "listeners": {
                "mass": {"cell_mass": "float", "dry_mass": "float"},
                "fba_results": {"external_exchange_fluxes": "list[float]"},
            },
        }

    def outputs(self):
        return {"listeners": {"mass_balance": {
            "cumulative_cell_mass_delta_fg": {"_type": "overwrite[float]", "_default": 0.0},
            "cumulative_net_exchange_mass_fg": {"_type": "overwrite[float]", "_default": 0.0},
        }}}

    def update(self, state, interval):
        m = state["listeners"]["mass"]
        cm = float(m["cell_mass"])
        if self._prev_cell_mass is not None:
            self._cum_delta += (cm - self._prev_cell_mass)
        self._prev_cell_mass = cm
        # net exchange mass over the tick: Σ flux_i [mmol/gDW/h] * MW_i [g/mmol]
        #   * dry_mass [fg→gDW] * dt[h]. DISCOVERY: reuse the molar-mass +
        #   externalMoleculeIDs ordering from v2ecoli/processes/metabolism.py
        #   (self.externalMoleculeIDs) so flux index ↔ species MW line up.
        self._cum_exch += self._net_exchange_mass_fg(state, float(m["dry_mass"]), interval)
        return {"listeners": {"mass_balance": {
            "cumulative_cell_mass_delta_fg": self._cum_delta,
            "cumulative_net_exchange_mass_fg": self._cum_exch,
        }}}

    def _net_exchange_mass_fg(self, state, dry_mass_fg, interval):
        fluxes = state["listeners"]["fba_results"].get("external_exchange_fluxes") or []
        # implement with the MW vector + unit conversion described above
        ...  # returns float
```

Wire it as a feature in `baseline.py` and `baseline_millard.py` execution layers (append to the `ecoli-mass-listener` layer, mirroring the `feat.get('listeners', [])` injection at `millard_pdmp_baseline.py:212-217`). Register a `mass_balance_listener` feature entry.

- [ ] **Step 4: Run + verify**

Run: `./.venv/bin/python -m pytest tests/test_mass_balance_listener.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/steps/derivers/mass_balance_listener.py tests/test_mass_balance_listener.py v2ecoli/composites/baseline.py v2ecoli/composites/baseline_millard.py
git commit -m "feat(listener): mass-balance listener emitting cumulative delta + net-exchange"
```

---

## Task 5: Native mass-conservation behavior test + fix mbp-01's degraded test

**Files:**
- Modify: `workspace/studies/mbp-01-time-varying-environment/study.yaml` (test `cumulative-mass-balance-closes`)
- Modify: `workspace/studies/mbp-04-multigeneration-runs/study.yaml` (add the gate)

- [ ] **Step 1: Confirm the current test degrades to the agent bucket**

Run:
```bash
./.venv/bin/python -c "
from pbg_superpowers.study_evaluator import evaluate_test
import yaml
d=yaml.safe_load(open('workspace/studies/mbp-01-time-varying-environment/study.yaml'))
t=[t for s in ('behavior_tests','tests') for t in (d.get(s) or []) if isinstance(t,dict) and t.get('name')=='cumulative-mass-balance-closes'][0]
print(t['measure'])  # kind: derived_ratio -> not native"
```
Expected: prints `{'kind': 'derived_ratio', ...}` (confirms it's the non-native kind that falls to agent).

- [ ] **Step 2: Re-author mbp-01's test to native `derived`**

Using ruamel (preserve comments) set the test's `measure`/`pass_if`:

```yaml
measure:
  kind: derived
  formula: "cumulative_cell_mass_delta_fg / cumulative_net_exchange_mass_fg"
  window: full_lineage_from_gen_0
pass_if: {op: in_range, low: 0.9, high: 1.1}
```

Apply via a ruamel round-trip script (mirror the cites-fill script used earlier):
```bash
./.venv/bin/python - <<'PY'
from ruamel.yaml import YAML
y=YAML(); y.preserve_quotes=True; y.width=4096
p='workspace/studies/mbp-01-time-varying-environment/study.yaml'
d=y.load(open(p))
for sec in ('behavior_tests','tests'):
    for t in (d.get(sec) or []):
        if isinstance(t,dict) and t.get('name')=='cumulative-mass-balance-closes':
            t['measure']={'kind':'derived','formula':'cumulative_cell_mass_delta_fg / cumulative_net_exchange_mass_fg','window':'full_lineage_from_gen_0'}
            t['pass_if']={'op':'in_range','low':0.9,'high':1.1}
y.dump(d, open(p,'w'))
print('rewrote mbp-01 mass-balance test')
PY
```

- [ ] **Step 3: Add the gate to mbp-04**

Append a `wcm-mass-conservation-closes` behavior_test to `workspace/studies/mbp-04-multigeneration-runs/study.yaml` (ruamel round-trip), identical `measure`/`pass_if` shape but tighter band `{op: in_range, low: 0.99, high: 1.01}` and `cites: [agmon2022]`.

- [ ] **Step 4: Lint stays clean**

Run: `./.venv/bin/python -c "from pbg_superpowers.report_linter import lint_workspace_report; print(sum(1 for f in lint_workspace_report('.') if getattr(f,'level',None)=='error'))"`
Expected: `0`

- [ ] **Step 5: Commit**

```bash
git add workspace/studies/mbp-01-time-varying-environment/study.yaml workspace/studies/mbp-04-multigeneration-runs/study.yaml
git commit -m "fix(mbp-01)+feat(mbp-04): native derived mass-conservation gate (was agent-bucket)"
```

---

## Task 6: Scaffold mbp-08 validation study (homes the mass gate + responsiveness)

**Files:**
- Create: `workspace/studies/mbp-08-millard-swap-validation/study.yaml`

- [ ] **Step 1: Write the study.yaml**

Same v3 shape as mbp-07. Key block:

```yaml
schema_version: 3
name: mbp-08-millard-swap-validation
created: '2026-06-14'
phase: Build
study_kind: construction
baseline:
- name: millard-cell
  composite: v2ecoli.composites.baseline_millard
  params: {seed: 0, cache_dir: out/cache, features: [mass_balance_listener]}
pipeline_gate:
  prerequisites: [mbp-07-millard-kinetic-metabolism-swap]
behavior_tests:
- name: wcm-mass-conservation-closes
  measure: {kind: derived, formula: "cumulative_cell_mass_delta_fg / cumulative_net_exchange_mass_fg", window: full_lineage_from_gen_0}
  pass_if: {op: in_range, low: 0.99, high: 1.01}
  cites: [agmon2022]
  acceptance_form: quantitative_range
- name: growth-recovered-vs-baseline
  measure: {kind: derived, formula: "listeners.mass.instantaneous_growth_rate", window: full_lineage_from_gen_0}
  pass_if: {op: in_range, low: 0.0, high: 0.04}
  cites: [Monod1949AnnRevMicrobiol]
  acceptance_form: quantitative_range
- name: uptake-falls-as-glucose-depletes
  measure: {kind: derived, formula: "central_fluxes.PTS_4", window: full_lineage_from_gen_0}
  pass_if: {op: ">", value: 0.0}
  acceptance_form: qualitative_direction
tests:
  pytest_args: [tests/test_baseline_millard.py, tests/test_mass_balance_listener.py]
```

- [ ] **Step 2: Validate parse + lint**

Run: `./.venv/bin/python -c "import yaml; yaml.safe_load(open('workspace/studies/mbp-08-millard-swap-validation/study.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add workspace/studies/mbp-08-millard-swap-validation/study.yaml
git commit -m "feat(mbp-08): scaffold Millard-swap validation study with mass-conservation gate"
```

---

## Task 7: Register mbp-07/08 in the investigation spine

**Files:**
- Modify: `investigations/multiscale-bioprocess/investigation.yaml`

- [ ] **Step 1: Add to `studies:`, `at_a_glance.studies`, and `acceptance_criteria`**

Via ruamel round-trip: append `mbp-07-...` and `mbp-08-...` to `studies:`; add `at_a_glance.studies` role lines; add per_study acceptance_criteria entries (e.g. `{study: mbp-08-millard-swap-validation, behavior: wcm-mass-conservation-closes, gating: per_study}`).

- [ ] **Step 2: Validate parse**

Run: `./.venv/bin/python -c "import yaml; yaml.safe_load(open('investigations/multiscale-bioprocess/investigation.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run the full suite + lint gate**

Run: `./.venv/bin/python -m pytest -q -m sim tests/test_baseline_millard.py tests/test_mass_balance_listener.py`
Expected: all PASS.
Run: `./.venv/bin/python -c "from pbg_superpowers.report_linter import lint_workspace_report; print(sum(1 for f in lint_workspace_report('.') if getattr(f,'level',None)=='error'))"`
Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add investigations/multiscale-bioprocess/investigation.yaml
git commit -m "feat(investigation): register mbp-07/08 in spine + acceptance criteria"
```

---

## Task 8 (scoped, GATED on mbp-03): mbp-09 Millard↔WCM↔Beulig comparison

**Blocked on:** `v2ecoli.composites.reactor_bird_coupled` (mbp-03's deliverable — does not exist yet). Do NOT start until mbp-03 lands.

When unblocked: create `workspace/studies/mbp-09-millard-reactor-comparison/study.yaml` (study_kind: evaluation; baseline = the Millard cell coupled to `reactor_bird_coupled`, plus the plain-WCM coupled cell as the contrast arm). Reuse mbp-05's report-card pattern: a `report_card_axis` behavior_test per Beulig observable group (glucose, biomass, acetate, dissolved O₂), divergences categorized, execute-and-report (no tuning). Surface the pbg-bioreactordesign#2 O₂ temp-sign bias as a declared divergence. prerequisites: [mbp-03-bird-reactor-coupling, mbp-08-millard-swap-validation].

---

## Self-review

- **Spec coverage:** engine swap (Tasks 2-3), real bulk writeback (pre-existing inline path, confirmed in Task 3 growth test), env responsiveness (Task 2), mass-conservation native gate (Tasks 4-6), mbp-01 degraded-test fix (Task 5), 3-study decomposition + prerequisite ordering (Tasks 1,6,7,8), mbp-09 scoped+gated (Task 8). Covered.
- **Placeholders:** two deliberate DISCOVERY steps remain (exact glucose SBML id from the species map; the MW vector reuse from `metabolism.py`) — these are explicit lookups against named files, not vague "handle X". The `_net_exchange_mass_fg` body is specified by formula + the exact source to reuse.
- **Type consistency:** observable tokens `cumulative_cell_mass_delta_fg` / `cumulative_net_exchange_mass_fg` are defined in Task 4 and consumed identically in Tasks 5-6; composite name `baseline_millard` consistent across Tasks 3,6,8; feature name `mass_balance_listener` consistent across Tasks 4,6.
