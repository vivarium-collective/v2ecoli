# trna_charging_final port audit

**Branch:** `trna_charging_final` (local in `v2ecoli`)
**Upstream:** `CovertLab/vEcoli@trna_charging_final` (HEAD `330ee3f4`, +20 commits ahead of `origin/master`)
**Generated:** 2026-06-08
**Source repo on disk:** `/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli` (already checked out on `trna_charging_final`)

## Headline

Upstream `trna_charging_final` adds a **`KineticTrnaChargingModel`** (`polypeptide_elongation.py:2198`) as an additional `BaseElongationModel` subclass alongside the existing `SteadyStateElongationModel`. It tracks per-ribosome codon-by-codon state via a 638-line Cython kernel (`wholecell/utils/_trna_charging.pyx`), with new ParCa-fitted parameters in TSVs of up to 5688 rows, plus three validation datasets.

v2ecoli's existing port has the **steady-state ppGpp/aminoacyl-synthetase charging model** (the older `calculate_trna_charging` in `polypeptide/kinetics.py`). The new kinetic model is **not yet present** in v2ecoli.

## File-by-file audit

Status legend: ✅ already covered · ⚠️ partial · ❌ missing · 🚫 not applicable (infra/CI/docs/wheels).

### Core process code

| Upstream file | Δ lines | v2ecoli counterpart | Status | Notes |
|---|---:|---|:-:|---|
| `ecoli/processes/polypeptide_elongation.py` | +1715 | `v2ecoli/processes/polypeptide_elongation.py` (1811) + `polypeptide/kinetics.py` (627) + `polypeptide/common.py` (10) | ⚠️ | Existing port = steady-state model only. Missing `KineticTrnaChargingModel` (line 2198 upstream) and supporting plumbing. v2ecoli has already split this into a `polypeptide/` subpackage (PRs #110, #117) — new model must land alongside that structure, not replace it. |
| `ecoli/processes/polypeptide_initiation.py` | +60 | `v2ecoli/processes/polypeptide_initiation.py` | ⚠️ | Diff pending |
| `ecoli/processes/protein_degradation.py` | +19 | `v2ecoli/processes/protein_degradation.py` | ⚠️ | Small diff |
| `ecoli/processes/transcript_elongation.py` | +30 | `v2ecoli/processes/transcript_elongation.py` | ⚠️ | Small diff |
| `ecoli/processes/tf_binding.py` | +5 | `v2ecoli/processes/tf_binding.py` | ⚠️ | Trivial |
| `ecoli/processes/chromosome_structure.py` | +58 | `v2ecoli/processes/chromosome_structure.py` | ⚠️ | Diff pending |
| `ecoli/processes/cell_division.py` | +22 | `v2ecoli/processes/cell_division.py` (search) | ⚠️ | v2ecoli has cell-division as a step; verify destination |
| `ecoli/processes/metabolism.py` | +8 | `v2ecoli/processes/metabolism.py` | ⚠️ | Trivial |
| `ecoli/processes/metabolism_redux.py` | +4 | (search; v2ecoli baseline uses metabolism, not metabolism_redux) | ❌ | v2ecoli currently has no `metabolism_redux` process. Decide: port the redux variant as a new architecture, or skip. |
| `ecoli/processes/metabolism_redux_classic.py` | +130 | none | ❌ | Same as above |
| `ecoli/processes/listeners/monomer_counts.py` | +69 | (v2ecoli emits via parquet emitters, not Listener processes) | ❌→⚠️ | Logic needs to land as parquet emitter additions or be folded into existing emitters. |
| `ecoli/processes/listeners/ribosome_data.py` | +2 | same as above | ⚠️ | Trivial |

### Cython / kinetics kernel

| Upstream file | Δ lines | v2ecoli counterpart | Status | Notes |
|---|---:|---|:-:|---|
| `wholecell/utils/_trna_charging.pyx` | +638 | none | ❌ | **Largest single port.** v2ecoli has no Cython build step. Translate to NumPy (vectorized where possible) + `@numba.njit` for hot loops (precedent: `polypeptide/kinetics.py` already uses `@njit`). Likely lands as `v2ecoli/library/trna_charging_kernel.py` or `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py`. |
| `wholecell/tests/utils/test_trna_charging.py` | +580 | none | ❌ | Companion unit tests. Port alongside the kernel into `tests/test_trna_charging_kernel.py`. |

### Library code

| Upstream file | Δ lines | v2ecoli counterpart | Status | Notes |
|---|---:|---|:-:|---|
| `ecoli/library/sim_data.py` | +212 | `v2ecoli/library/sim_data.py` | ⚠️ | Likely adds kinetic-charging param-loading paths. v2ecoli's `sim_data.py` is part of the StaleCacheError fingerprint — touching it forces a `build_cache.py` re-run. |
| `ecoli/library/schema.py` | +65 | `v2ecoli/library/` (search; v2ecoli uses bigraph-schema types) | ⚠️ | Many schema concepts are typed differently in v2ecoli (`v2ecoli/types/`). Selectively port the parts that affect the elongation contract. |
| `ecoli/library/initial_conditions.py` | +61 | `v2ecoli/library/initial_conditions.py` | ⚠️ | Adds kinetic-charging state initialization. |
| `ecoli/library/json_state.py` | +5 | (v2ecoli uses bigraph-schema serializers, not json_state) | 🚫→⚠️ | Trivial; equivalent logic may need to land in the serializer dispatchers. |
| `ecoli/library/logging_tools.py` | +9 | n/a | 🚫 | Logging plumbing |
| `ecoli/library/parquet_emitter.py` | +139 | `v2ecoli/library/` (search) | ⚠️ | New listener emit paths. v2ecoli has parquet support per recent feat/default-baseline-parquet PR — verify columns line up. |
| `ecoli/library/test_parquet_emitter.py` | ±400 | n/a | 🚫 | Tests for the emitter — port only if v2ecoli's emitter tests are affected. |

### ParCa (reconstruction)

| Upstream file | Δ lines | v2ecoli counterpart | Status | Notes |
|---|---:|---|:-:|---|
| `reconstruction/ecoli/fit_sim_data_1.py` | +268 | `v2ecoli/processes/parca/.../fit_sim_data_1.py` (verify) | ⚠️ | Likely the tRNA-charging re-optimization entry point. Multiprocessing path. |
| `reconstruction/ecoli/dataclasses/relation.py` | +1276 | none | ❌ | **Brand-new file** — likely the `Relation` dataclass that owns the tRNA↔codon↔synthetase relations. Whole file must be ported. |
| `reconstruction/ecoli/dataclasses/process/transcription.py` | +70 | `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/transcription.py` | ⚠️ | tRNA boundary adjustments |
| `reconstruction/ecoli/dataclasses/process/two_component_system.py` | +103 | `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/two_component_system.py` | ⚠️ | Diff pending |
| `reconstruction/ecoli/dataclasses/process/translation.py` | +9 | (verify path) | ⚠️ | Trivial |
| `reconstruction/ecoli/dataclasses/getter_functions.py` | (small) | `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/getter_functions.py` | ⚠️ | Small diff |
| `reconstruction/ecoli/dataclasses/molecule_groups.py` | +21 | (verify path) | ⚠️ | New molecule group entries |
| `reconstruction/ecoli/dataclasses/process/metabolism.py` | (small) | `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/metabolism.py` | ⚠️ | Already in v2ecoli per earlier grep |
| `reconstruction/ecoli/scripts/growth_rate_dependent_parameters.py` | +169 | `v2ecoli/processes/parca/reconstruction/ecoli/scripts/...` | ⚠️ | growth-rate refit |
| `reconstruction/ecoli/scripts/update_biocyc_files.py` | (small) | `v2ecoli/processes/parca/reconstruction/ecoli/scripts/update_biocyc_files.py` | ⚠️ | Already touched per earlier grep |
| `reconstruction/ecoli/knowledge_base_raw.py` | +12 | `v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py` | ⚠️ | Trivial loader changes |
| `reconstruction/ecoli/simulation_data.py` | +27 | `v2ecoli/processes/parca/reconstruction/ecoli/simulation_data.py` (verify) | ⚠️ | Small diff |
| `reconstruction/ecoli/scripts/nca/run_all.py` | +12 | (verify path) | ⚠️ | NCA pipeline |

### ParCa flat data

| Upstream flat file | Rows | v2ecoli counterpart | Status |
|---|---:|---|:-:|
| `reconstruction/ecoli/flat/trnas.tsv` | +47 | none | ❌ |
| `reconstruction/ecoli/flat/trna_charging_kinetics.tsv` | +22 | none | ❌ |
| `reconstruction/ecoli/flat/trna_charging_kinetics_curated.tsv` | +22 | none | ❌ |
| `reconstruction/ecoli/flat/trna_charging_kinetics_constants.tsv` | +42 | none | ❌ |
| `reconstruction/ecoli/flat/trna_charging_kinetics_solutions.tsv` | +5688 | none | ❌ |
| `reconstruction/ecoli/flat/trna_charging_reactions{,_removed,_added}.tsv` | small | `v2ecoli/processes/parca/reconstruction/ecoli/flat/trna_charging_reactions*.tsv` | ✅ already present (verify content matches) |
| `reconstruction/ecoli/flat/optimization/trna_synthetase_dynamic_range.tsv` | +49 | none | ❌ |
| `reconstruction/ecoli/flat/parameters.tsv` adjustments | small | `v2ecoli/processes/parca/reconstruction/ecoli/flat/parameters.tsv` | ⚠️ already partly touched |
| `reconstruction/ecoli/flat/adjustments/rna_deg_rates_adjustments.tsv` | small | already in v2ecoli per earlier grep | ⚠️ |

### Validation data (new top-level directory in v2ecoli)

| Upstream file | Δ lines | v2ecoli counterpart | Status |
|---|---:|---|:-:|
| `validation/ecoli/flat/dong1996_table_5.tsv` | +47 | none — no `validation/` tree in v2ecoli | ❌ |
| `validation/ecoli/flat/jakubowski1984_table_3.tsv` | +21 | none | ❌ |
| `validation/ecoli/flat/trna_synthetase_kinetics.tsv` | +86 | none | ❌ |
| `validation/ecoli/validation_data.py` | +158 | none | ❌ |
| `validation/ecoli/validation_data_raw.py` | +3 | none | ❌ |

v2ecoli has no `validation/` directory at all. Needs a destination decision: either mirror as `v2ecoli/processes/parca/validation/` (under ParCa) or add `v2ecoli/validation/` as a sibling.

### Analyses

| Upstream file | Δ lines | v2ecoli counterpart | Status |
|---|---:|---|:-:|
| `ecoli/analysis/multivariant/trna_synthetase_concs.py` | +241 | none | ❌ |
| `ecoli/analysis/multivariant/dummy.py` | +10 | none | ❌ |
| `ecoli/analysis/single/*.py` (selected_fluxes, centralCarbonMetabolismScatter) | various | (search) | ⚠️ |
| `ecoli/analysis/antibiotics_colony/plot.py` | +9 | (verify) | ⚠️ |

Many of these correspond to reports/visualizations in v2ecoli; port selectively to `reports/`.

### Composites / experiments

| Upstream file | Δ lines | v2ecoli counterpart | Status |
|---|---:|---|:-:|
| `ecoli/composites/ecoli_master.py` | +54 | `v2ecoli/composites/baseline.py` (or relevant arch) | ⚠️ | Composite wiring for the new kinetic model. Probably needs a new architecture or a flag on `baseline`. |
| `ecoli/composites/ecoli_master_tests.py` | ±228 | n/a | 🚫 | Upstream test infra |
| `ecoli/experiments/ecoli_master_sim.py` | +65 | n/a | 🚫 | Upstream sim runner |
| `ecoli/experiments/ecoli_master_sim_tests.py` | -74 | n/a | 🚫 | Upstream test runner |

### Configs

`configs/no_grc.json` (+33) and `configs/ecoli-no-growth-rate-control.json` (+17) — new no-growth-rate-control profile that gets exercised in the kinetic-charging workflow. Maps to a v2ecoli composite parameter set.

### Out-of-scope (not biology / not v2ecoli surface)

CI workflows (`.github/workflows/*`), Jenkins, Docker, Nextflow, runscripts/, doc/*.rst, cloud_pricing/, `wholecell/io/` ingestion, RNAseq experimental_data deletions, dashboard wheels — all upstream-only infrastructure or files v2ecoli doesn't carry. Skip.

## Sizing the work

| Phase | Realistic effort |
|---|---|
| Cython kernel → NumPy/numba (638 lines + 580-line test) | 1–2 days |
| `KineticTrnaChargingModel` class + composite wiring inside v2ecoli's `polypeptide/` subpackage | 2–3 days |
| `Relation` dataclass (1276 lines, brand-new) | 1 day |
| Library/sim_data/initial_conditions/schema port | 1 day |
| ParCa scripts + dataclass deltas + flat files | 1 day |
| Validation tree + data + loader | 0.5 day |
| Other process deltas (initiation, transcript_elongation, monomer_counts, etc.) | 1 day |
| Run full ParCa pipeline (compute) | hours–day of wall clock |
| Behavior tests + parity gate + reports | 0.5–1 day |
| **Total** | **8–12 person-days** of careful work |

## Recommended phasing

1. **Phase A — Refactor groundwork (no behavior change)**
   - Create `v2ecoli/validation/` destination + port the three validation TSVs and `validation_data*.py` skeleton (so kinetic-charging code has somewhere to land later).
   - Port `Relation` dataclass into `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/relation.py`.
   - Port the four flat data files into `v2ecoli/processes/parca/reconstruction/ecoli/flat/`.
   - Bring the trivial ParCa dataclass + script deltas (transcription, two_component_system, knowledge_base_raw, simulation_data, growth_rate_dependent_parameters).

2. **Phase B — Cython kernel translation**
   - Port `_trna_charging.pyx` → pure-NumPy + numba module under `v2ecoli/processes/polypeptide/`.
   - Port the 580-line companion test into `tests/test_trna_charging_kernel.py`. Gate on bit-for-bit parity vs upstream output where feasible (run upstream Cython kernel, dump golden outputs, assert in v2ecoli test).

3. **Phase C — KineticTrnaChargingModel class**
   - Implement `KineticTrnaChargingModel` inside `v2ecoli/processes/polypeptide/` (likely a new file `kinetic_charging.py` alongside `kinetics.py`), wired into `SteadyStatePolypeptideElongation`'s model selector.
   - Composite wiring on `v2ecoli/composites/baseline.py` (or a new architecture `kinetic_charging_baseline`).
   - Behavior test under `tests/test_behavior_kinetic_charging.py`.

4. **Phase D — ParCa pipeline + cache**
   - Wire the new dataclass loaders + kinetic param solving into the ParCa pipeline.
   - Run `scripts/build_cache.py` first to validate the cache-fingerprint path; then the full ParCa pipeline.

5. **Phase E — Validation runs**
   - Fast tests → behavior tests → parity gate → workflow_report / multigeneration_report / dedicated tRNA-charging HTML with provenance banner.

## Open decisions

1. **Where do the new top-level dirs live?** Two reasonable answers for `validation/`:
   - `v2ecoli/processes/parca/validation/` (treat as ParCa's downstream consumer — matches the way `reconstruction/` is nested under ParCa).
   - `v2ecoli/validation/` (top-level sibling — mirrors upstream layout exactly; easier to track refactors).
2. **Architecture vs. flag?** The kinetic model can either be (a) a new composite architecture (`kinetic_charging_baseline`), or (b) a flag on `baseline` (`charging_model="kinetic"`). The latter matches the upstream `KineticTrnaChargingModel` design, but (a) is more in line with v2ecoli's "architectures as separate composite functions" pattern from AGENTS.md.
3. **Cython vs numba.** numba `@njit` is precedent (used in `polypeptide/kinetics.py`) and avoids a build step. Risk: a few of the kernel routines use C `rand()` for tRNA stochasticity — must seed deterministically through numpy's RNG to keep `seed_rng` semantics.
4. **Test golden capture.** Best parity guarantee is to run the upstream Cython kernel on a fixed seed + state, dump inputs/outputs to JSON, and assert against them inside v2ecoli's port. Adds a one-time `vEcoli_trna` build step but pins behavior.

---

## Phase A progress log (this session)

**2026-06-08 — Phase A.1 + A.2 landed (uncommitted).**

### Files added
- `v2ecoli/validation/__init__.py`, `v2ecoli/validation/ecoli/__init__.py`
- `v2ecoli/validation/ecoli/validation_data.py` (570 lines, import paths rewritten from `reconstruction.*` / `wholecell.*` → `v2ecoli.processes.parca.*`)
- `v2ecoli/validation/ecoli/validation_data_raw.py` (43 lines, `from reconstruction.spreadsheets` → `from v2ecoli.processes.parca.reconstruction.spreadsheets`)
- `v2ecoli/validation/ecoli/flat/` — 24 files (Toya2009_CCMfluxes, Dong 1996, Jakubowski 1984, trna_synthetase_kinetics, etc.) copied verbatim from upstream `trna_charging_final`. Note: most of these are upstream-master content too — needed because `ValidationDataRawEcoli` enumerates them all in `LIST_OF_DICT_FILENAMES`.
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/trnas.tsv` (47 entries — supersedes legacy trna_data/ split).
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/trna_charging_kinetics.tsv` (22 rows).
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/trna_charging_kinetics_curated.tsv` (22 rows).
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/optimization/trna_charging_kinetics_constants.tsv` (11.8 KB).
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/optimization/trna_charging_kinetics_solutions.tsv` (5.1 MB — large ParCa-fit table).
- `v2ecoli/processes/parca/reconstruction/ecoli/flat/optimization/trna_synthetase_dynamic_range.tsv`.

### Files modified
- `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/relation.py`: replaced 145-line stub with full 1421-line port from upstream `trna_charging_final`. Imports rewritten:
  - `from .process.replication import MAX_TIMESTEP` → absolute `v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.process.replication`
  - `from wholecell.utils import units` → `from v2ecoli.processes.parca.wholecell.utils import units`
  - `from wholecell.utils.polymerize import polymerize` → `from v2ecoli.processes.parca.wholecell.utils.polymerize import polymerize`
  - `Relation.__init__` now calls 4 new builders: `_build_codon_sequences`, `_build_codon_based_translation`, `_build_codon_dependent_trna_charging`, `_build_trna_charging_kinetics`.
- `v2ecoli/processes/parca/reconstruction/ecoli/knowledge_base_raw.py`: added 6 new TSV loader entries to `LIST_OF_DICT_FILENAMES` (`trnas`, `trna_charging_kinetics`, `trna_charging_kinetics_curated`, `optimization/{kinetics_constants, kinetics_solutions, dynamic_range}`). Legacy `trna_data/trna_ratio_to_16SrRNA_*.tsv` paths **kept** (conservative — Phase D will remove them once all downstream references are migrated).

### Open issues discovered
1. **Venv missing `pbg_superpowers`** — `v2ecoli/__init__.py` line 8 imports it, but `.venv/bin/python` reports `ModuleNotFoundError`. Blocks runtime smoke-tests. `pyproject.toml` declares it as a dep but the directory isn't on disk anywhere under `/Users/arnabmutsuddy/projects/`. Either the venv is stale (re-`uv sync` needed) or the upstream package URL is broken. Out of scope for this port, but blocks Task #10/#11/#13 and the ParCa run.
2. **Syntax-check passes for all 4 modified/new Python files** (`py_compile`). Runtime import test not possible until the venv is fixed.

### Files NOT touched yet (still on backlog)
- Cython kernel `_trna_charging.pyx` → numba port. (Task #2.)
- `polypeptide_elongation.py` refresh: bring `KineticTrnaChargingModel` class in alongside the existing `SteadyStateElongationModel`. (Task #3.)
- ParCa dataclass small deltas: transcription.py (+70), two_component_system.py (+103), molecule_groups.py (+21), translation.py (+9), getter_functions.py, simulation_data.py (+27), growth_rate_dependent_parameters.py (+169), scripts/nca/run_all.py (+12). (Task #6.)
- Library deltas: sim_data.py (+212), schema.py (+65), initial_conditions.py (+61), parquet_emitter.py (+139). (Task #5.)
- Other process deltas (Task #4).
- Composite wiring (new `kinetic_charging_baseline` architecture) and behavior test.
- `scripts/build_cache.py` rebuild + full ParCa run + behavior tests + reports.
