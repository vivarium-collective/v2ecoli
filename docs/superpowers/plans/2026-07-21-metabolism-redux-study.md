# metabolism_redux Comparison Study — Design + Plan

**Date:** 2026-07-21
**Branch:** `feat/investigation-report-card-summary`
**Decisions:** variant `ecoli-metabolism-redux`; condition `basal`; **both engines** run redux; wire now, run in background.

## Goal
Add a `metabolism_redux` study to the `v2ecoli-vecoli-comparison` investigation where BOTH engines swap FBA `Metabolism` → `MetabolismRedux`, testing whether v2ecoli reproduces vEcoli under the redux metabolism at the basal condition.

## Why both engines
The investigation's premise is "does v2ecoli reproduce vEcoli." Comparing v2ecoli-redux vs vEcoli-FBA would confound engine differences with metabolism-model differences. So the vEcoli reference must also run redux.

## Mechanism (mostly existing)
- vEcoli ships `configs/metabolism_redux.json`: `swap_processes: {"ecoli-metabolism":"ecoli-metabolism-redux"}` + a `flow` reorder + `exclude_processes:["exchange_data"]`.
- `run_comparison_ensemble.py --from-vecoli-config <cfg>` already: (v2 side) resolves the config → `injected_processes` so v2 convert+injects redux (the `metabolism_redux` unit handling in `vivarium_bridge.py` exists for this).
- **Gaps:** (a) the vEcoli-side wrapper `run_vivarium_ecoli_pbg_multigen` ignores `swap_processes`; (b) the study→runner path (`runner._run_engines`) never passes a config.

## Work items

### 1. vEcoli config for the study
Create `<vEcoli fork>/configs/metabolism_redux_basal.json` (path passed as fork-relative `configs/metabolism_redux_basal.json`; `resolve_vecoli_config_local` also accepts absolute):
```json
{
  "experiment_id": "metabolism_redux_basal",
  "condition": "basal",
  "swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
  "exclude_processes": ["exchange_data"],
  "flow": { ...copied from configs/metabolism_redux.json... }
}
```
Run shape (seeds/gens/steps) is driven by the runner CLI, not this config. Keep a tracked copy at `workspace/investigations/v2ecoli-vecoli-comparison/studies/metabolism_redux/metabolism_redux_basal.json` for provenance.

### 2. Extend the vEcoli engine wrapper to honor the swap
`v2ecoli/library/vivarium_ecoli_engine.py` — `run_vivarium_ecoli_pbg_multigen(...)`:
- Add params `swap_processes: dict | None = None`, `flow: dict | None = None`.
- After the existing `sim.config[...]` assignments, if `swap_processes`: `sim.config["swap_processes"] = dict(swap_processes)`; if `flow`: merge into `sim.config["flow"]`. EcoliSim natively applies these (same path metabolism_redux.json uses upstream).
- Thread the same two params through the pbg `Process` wrapper (`config_schema` + `__init__` call site around lines 226–250) so both entrypoints support it.

### 3. Pass the swap on the vEcoli side in the ensemble runner
`scripts/run_comparison_ensemble.py` — `make_run_one`:
- For `composite_kind == "vecoli"`, when `from_vecoli_config` is set, resolve it (reuse `resolve_vecoli_config_local`) and pass `swap_processes=resolved.get("swap_processes")`, `flow=resolved.get("flow")` into `run_vivarium_ecoli_pbg_multigen(...)`.
- (v2 side already builds `injected_processes` from the same resolved config — unchanged.)

### 4. Thread a config through the study runner
- `scripts/_compare/study_spec.py`: add `from_vecoli_config: str = ""` to `StudySpec`; in `_spec_from_study` read `data.get("from_vecoli_config") or (comp.get("from_vecoli_config"))`.
- `scripts/_compare/runner.py` `_run_engines`: when `spec.from_vecoli_config`, append `--from-vecoli-config <cfg>` to BOTH the v2ecoli and vecoli `run_comparison_ensemble.py` invocations.

### 5. Author the study + register it
- `workspace/investigations/v2ecoli-vecoli-comparison/studies/metabolism_redux/study.yaml`:
  - `schema_version: 4`, `name: metabolism_redux`, `condition: basal`, `comparison: {seeds: 1, generations: 4}`, `from_vecoli_config: configs/metabolism_redux_basal.json`.
  - `report_cards: [viz/report_card/config.html, viz/report_card/standard.html]`.
  - `conditions.baseline`: `{composite: v2ecoli.composites.ecoli_baseline.ecoli_baseline, params: {condition: basal, swap: ecoli-metabolism-redux}}` (drives the summary's config-JSON display).
  - `pipeline_gate.prerequisites: [parca]`, `depends_on: [parca]`, `status: build`, a hand-authored `question` + one placeholder `finding` (status: investigating) + `tests` (config + standard report_card, hand-written per the no-machine-projected-tests rule).
- Add `metabolism_redux` to `investigation.yaml` `studies:` (after `basal`, before `statistical` reads fine; order is display/DAG only).

### 6. Run (background) + report
Env: `JAVA_HOME=/opt/homebrew/opt/openjdk@21/...`, vEcoli venv on PATH, `V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli`. Caches are MISSING → the run first builds v2 ParCa (`out/cache_full`) + vEcoli ParCa (`out/compare_harness/vecoli_parca`) — hours, fragile. Run via the study runner (`scripts/compare_run.py` or `run_study`) for just `metabolism_redux`. Monitor; report real verdict/cards or the actual failure.

### 7. Regenerate the summary
`reports/investigation_summary.py --investigation v2ecoli-vecoli-comparison` — the new study appears with its config JSON + graded card + matrix row + nav link.

## Risks
- ParCa builds (both engines) are the long pole and can fail; surfaced honestly.
- EcoliSim swap under the pbg wrapper's config overrides (`generations=None`, `divide=False`) may interact with the redux `flow` reorder — validate the vEcoli side actually loads redux (log `n_processes` / process list) before trusting the comparison.
- The redux bridge into v2 (`injected_processes`) has never run in a study before; the `vivarium_bridge.py` redux unit handling suggests it was prepared, but this is first real use.

## Out of scope
No changes to the summary generator (already done). No other conditions (basal only). No `redux-classic`.
