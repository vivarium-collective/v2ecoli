# mbp-03 BiRD Reactor Coupling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Couple v2ecoli (with the mbp-01 env hook + mbp-02 population aggregator) to pbg-bioreactordesign's `BiRDTransportProcess` via a `ReactorCellCoupler` Step, in a new `reactor_bird_coupled` composite, producing a physically consistent 0D well-mixed coupled trajectory.

**Architecture:** Option-B coupling — BiRD owns transport physics (kLa, Henry's-law saturation, gas holdup) and reads biomass as input; v2ecoli owns biomass (population aggregator) and metabolic O2/CO2 exchange demand. A `ReactorCellCoupler` Step bridges them each emit cycle, with the unit conversions below. Dissolved O2/CO2 net out as additive bare-float deltas (cell consumption − , reactor transport +), per the BiRD coupled-factory convention.

**Tech stack:** process-bigraph, pbg-bioreactordesign (`BiRDTransportProcess`), v2ecoli (`baseline_population`, `environment_driver` external_store mode, `EnvironmentMirror`), pytest. `./.venv/bin/python`; `out/cache` present.

## Verified interface (from the seam map)

**BiRDTransportProcess** (`pbg_bioreactordesign/processes.py:381`): inputs `{dissolved_o2 (mg/L), dissolved_co2 (mg/L), biomass (g/L), glucose (g/L), gas_flow_rate_Lpm}`; outputs additive deltas `dissolved_o2`/`dissolved_co2` (mg/L) + diagnostics `kla_o2`, `o2_saturation`, `gas_holdup`, … Time base **hours**. Registered via `core.register_link("BiRDTransportProcess", BiRDTransportProcess)` (precedent: `scripts/runnable_sims/bird_reactor_run.py:29`).

**v2ecoli stores:** `population.biomass_concentration_gL` (g/L; `baseline_population.py:41`); per-agent `agents.*.metabolism.external_exchange_fluxes` (mmol/gDW·h; molecule ids `OXYGEN-MOLECULE[p]`, `CARBON-DIOXIDE[p]`); `environment.external_concentrations` (mM dict; written by the coupler in `external_store` mode — `environment_driver.py:106` is a no-op then); `EnvironmentMirror` propagates env→`agents.*.boundary.external`.

**Unit conversions the coupler owns:** mg/L ↔ mM = ÷(MW)÷1000 (O2 MW 31.999, CO2 44.010); per-agent biomass gDW from `cell_mass` fg × `cells_per_agent` × 1e-15; reactor rates per hour vs v2ecoli timestep seconds (÷3600). O2 uptake → reactor: `−Σ_agents flux_O2[mmol/gDW·h] × agent_biomass_gDW` ⇒ mmol/h ÷ volume_L ⇒ then ×MW to mg/L·h for the BiRD additive store.

## File structure

| File | Responsibility | New/Edit |
|---|---|---|
| `pyproject.toml` / `uv.lock` | declare pbg-bioreactordesign as a workspace dep (req-1) | Edit |
| `v2ecoli/core.py` | register `BiRDTransportProcess` link in `build_core()` | Edit |
| `v2ecoli/steps/reactor_cell_coupler.py` | the `ReactorCellCoupler` Step (req-2) | New |
| `v2ecoli/composites/reactor_bird_coupled.py` | composite: baseline_population + BiRDTransportProcess + coupler + env_driver(external_store) | New |
| `v2ecoli/composites/__init__.py` | register composite | Edit |
| `tests/test_reactor_cell_coupler.py` | coupler unit tests (conversions, aggregation) | New |
| `tests/test_reactor_bird_coupled.py` | composite build + the 5 mbp-03 behavior tests | New |

## Task 1 — req-1: formal dependency on pbg-bioreactordesign
(Decision pending — see the controller note; this task is filled once the pinning form is chosen: git `vivarium-collective/pbg-bioreactordesign@main` vs local path vs version. The dep is already editable-installed in the venv for development.)
- [ ] Add the dependency to `pyproject.toml` (chosen form), `uv lock`, verify `./.venv/bin/python -c "import pbg_bioreactordesign"`, run an existing sim test to confirm the env still resolves. Commit.

## Task 2 — register BiRDTransportProcess in build_core
- [ ] Test: `tests/test_reactor_cell_coupler.py::test_bird_transport_registered` asserts `build_core()` resolves `BiRDTransportProcess` (e.g. `core.find("BiRDTransportProcess")` or composite address `local:BiRDTransportProcess` builds). Run → FAIL.
- [ ] In `v2ecoli/core.py` `build_core()`, add a try/except import + `core.register_link("BiRDTransportProcess", BiRDTransportProcess)` mirroring the KetchupEstimator block.
- [ ] Run → PASS. Commit.

## Task 3 — ReactorCellCoupler Step (req-2)
TDD, one behavior per test. The Step (Step subclass, runs each emit cycle):
- inputs: `population.biomass_concentration_gL`, `reactor.volume_L`, `agents.*.metabolism.external_exchange_fluxes` + `agents.*.listeners.mass.cell_mass`, `reactor.dissolved_o2`, `reactor.dissolved_co2` (mg/L).
- outputs: `reactor.biomass` (g/L, overwrite), additive deltas to `reactor.dissolved_o2`/`dissolved_co2` (mg/L) from aggregated metabolic O2 uptake / CO2 evolution, and `environment.external_concentrations.{OXYGEN-MOLECULE[p],CARBON-DIOXIDE[p]}` (mM, overwrite) from the reactor dissolved gases.

- [ ] Test `test_biomass_passthrough`: coupler writes `population.biomass_concentration_gL` → `reactor.biomass` unchanged (g/L). FAIL → implement → PASS.
- [ ] Test `test_mgL_to_mM_conversion`: given `reactor.dissolved_o2 = 8.0` mg/L, coupler writes `environment.external_concentrations['OXYGEN-MOLECULE[p]'] ≈ 8.0/31.999/1000` (mM). FAIL → implement → PASS.
- [ ] Test `test_o2_uptake_aggregation`: two agents with known flux + cell_mass → coupler emits the correct aggregated negative O2 delta (mg/L) into `reactor.dissolved_o2` (verify sign + magnitude via the conversion chain). FAIL → implement → PASS.
- [ ] Test `test_co2_evolution_sign`: positive CO2 flux → positive `reactor.dissolved_co2` delta. PASS.
- [ ] Commit.

## Task 4 — reactor_bird_coupled composite
- [ ] Test `test_reactor_bird_coupled_builds`: `build_composite("reactor_bird_coupled", seed=0, cache_dir="out/cache")` returns a Composite. FAIL.
- [ ] Implement composite: base = `baseline_population`; add `BiRDTransportProcess` (address `local:BiRDTransportProcess`, config from `bird_reactor_config` param, default `{reactor_type: bubble_column, volume_L: 1.0, gas_flow_rate_Lpm: 2.0, temperature_K: 310.15}`); add `ReactorCellCoupler`; set env_driver to `external_store` mode; wire shared `reactor.dissolved_o2`/`dissolved_co2` so coupler+transport deltas aggregate additively; ensure `EnvironmentMirror` runs so agents see updated boundary. Register in `__init__.py`.
- [ ] Run → PASS. Commit.

## Task 5 — the 5 mbp-03 behavior tests (in tests/test_reactor_bird_coupled.py)
Author tests matching mbp-03 study.yaml behavior_tests:
- [ ] `one-generation-completes-without-divergence`: run ~60 sim-min; assert no NaN, dissolved_o2 ∈ [0, o2_saturation], cell_count ≥ 1.
- [ ] `cells-drop-do-below-saturation`: with active biomass, steady-state dissolved_o2 < o2_saturation.
- [ ] `higher-kla-raises-steady-state-do`: two configs (higher kLa via gas_flow/impeller) → higher steady-state dissolved_o2.
- [ ] `cells-raise-dissolved-co2-above-saturation`: CO2 evolution pushes dissolved_co2 > co2_saturation.
- [ ] `o2-mass-balance-closes`: cumulative O2 consumed (Σ uptake×biomass×dt) vs cumulative transferred (Σ kLa×(C*−C)×V×dt) balance within tolerance (reactor.diagnostics integrals).
- [ ] Wire these into mbp-03 study.yaml (replace the placeholder composite TODO; set implementation_status; add pytest_args). Commit.

⚠️ Known caveat to surface as a divergence (not fix): pbg-bioreactordesign#2 — O2 saturation wrong temperature sign; at 310 K c*(O2) ~25% high. Document in mbp-03 findings/open_questions.

## Self-review
- Spec coverage: req-1 (Task 1), req-2 coupler (Task 3), BiRDTransportProcess dep+register (Tasks 1-2), composite (Task 4), all 5 behavior_tests (Task 5). Covered.
- Unit conversions specified with exact factors. Molecule ids verified (`OXYGEN-MOLECULE[p]`, `CARBON-DIOXIDE[p]`).
- Open: the o2-mass-balance test needs the reactor diagnostics integrals — confirm BiRDTransportProcess exposes kLa + saturation per tick (it does: `kla_o2`, `o2_saturation` outputs) so the coupler/a small accumulator can integrate them.
