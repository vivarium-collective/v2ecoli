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

---

## Task #6 progress log

**2026-06-08 — ParCa dataclass deltas (uncommitted at write time).**

### Applied (real tRNA-charging deltas)
- `dataclasses/process/translation.py`: added `cleavage_of_initial_methionine` column to `monomer_data` struct (bool, no unit). Used downstream by KineticTrnaChargingModel to know which proteins start at M+1.
- `dataclasses/molecule_groups.py`: added `codons` (generated 4³ minus UAA/UAG stop codons), `initiator_trnas` (4 fMet-tRNAs), `elongator_trnas` (2 elongator Met-tRNAs).
- `simulation_data.py`: added `self.codon_read_rate = {}` initialization (parallel to `translation_supply_rate`). Used by `Relation._build_trna_charging_kinetics` and the optimization loop.
- `dataclasses/process/transcription.py`:
  - Added `anticodon` (`U3`) column to `cistron_data`, populated from `raw_data.rnas[*]["anticodon"]` for tRNAs (empty string for non-tRNAs).
  - Changed `aa_from_trna` dtype from float to `int` (matches new integer-mapping convention from upstream commit `bf0c2c3e Integer mapping matrix for tRNAs to amino acids`).
- `dataclasses/growth_rate_dependent_parameters.py`: rewrote `_build_trna_data` from legacy multi-file (`trna_data/trna_ratio_to_16SrRNA_*.tsv`) to anticodon-based mapping from single `trnas.tsv`. Adds `from Bio.Seq import reverse_complement_rna` for sequence-disambiguation of ambiguous Kurland-tRNA → WCM-tRNA matches. Output matrix is now `(n_trnas, n_growth_rates)` — `get_trna_distribution` adjusted accordingly.

### Skipped (upstream-master infra reversion, not tRNA-charging)
- `dataclasses/process/two_component_system.py` (+103 lines of diff): trna_charging_final **removes** `modified_molecules` attribute + `make_modified_molecule_list`, drops `_buildComplexToMonomer`'s `sim_data` parameter, removes compartment-tag validation. All of that was added on upstream master after trna_charging_final was branched. Reverting it on v2ecoli would be a regression unrelated to tRNA.
- `scripts/nca/run_all.py` (+45 lines of diff): drops `r"..."` raw-string prefixes on regex literals. Pure upstream-branch noise (master added them; trna_charging_final pre-dates that). Reverting would re-introduce `DeprecationWarning`s under Python 3.12+.

### Verified
- All 5 modified files pass `py_compile`.
- `KnowledgeBaseEcoli` instantiates in 1.5 s (44 trnas, 5680 optimization solutions accessible) with the new `anticodon` field present on every `rnas` row.

---

## Task #2a progress log

**2026-06-09 — Cython parity-test scaffold landed.**

### Built
- Compiled upstream `_trna_charging.pyx` in `vEcoli_trna/` via its own `.venv` (Cython 3.1.2):
  - `cd /Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli && .venv/bin/python setup.py build_ext --inplace`
  - Produced `wholecell/utils/_trna_charging.cpython-312-darwin.so`.
- Verified upstream test suite passes: `pytest wholecell/tests/utils/test_trna_charging.py` → 15 passed in 0.44s on macOS arm64 Python 3.12.9.

### New files in v2ecoli
- `workspace/investigations/trna-charging-final/capture_kernel_golden.py` — captures golden inputs+outputs for every kernel function by running the upstream Cython kernel with the test-case inputs from upstream `test_trna_charging.py` plus larger seeded random inputs for `reconcile_via_*`. Output: gz-compressed JSON, no Cython needed for read-back.
- `tests/fixtures/trna_charging_kernel_golden.json.gz` (1.7 KB gz, 12 KB raw) — 25 cases across 9 functions:
  - `get_initiations`: 1 case
  - `get_codon_at`: 5 cases (current, +1, −1, beyond C-term, beyond N-term)
  - `get_candidates_to_C`: 3 cases
  - `get_candidates_to_N`: 2 cases
  - `select_candidate`: 2 cases (r=0, r=1)
  - `get_elongation_rate`: 1 case
  - `get_codons_read`: 1 case
  - `reconcile_via_ribosome_positions`: 7 cases (equal, forward, backward, backward_beyond, attempts_threshold, use_free_trna_prelim, 2 big-seeded with seed=12345 and 54321)
  - `reconcile_via_trna_pools`: 2 cases (use_free_trna, forward_undo_charging)
- `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` — module skeleton with:
  - `seed(int)`/`randint_below(n)` RNG wrapper backed by `numpy.random.RandomState` (independent of `numpy.random` global state).
  - Stubs for all 10 kernel functions (8 from 2b + reconcile_via_ribosome_positions, reconcile_via_trna_pools, get_elongation_rate) raising `NotImplementedError("Task 2b/2c/2d/2e")` until filled in.
  - Module docstring spells out the RNG-equivalence policy: deterministic functions parity exactly; stochastic functions parity statistically.
- `tests/test_kinetic_charging_kernel_scaffold.py` — 10 tests, all green:
  - Golden round-trips: fixture exists, metadata present (`captured_at`, `upstream_sha`, `platform`, `rng`, the libc-rand note), all 9 functions covered, array serialization spot-check passes.
  - RNG: deterministic given seed, divergent across seeds, independent of `numpy.random.seed`, raises when unseeded.
  - Stubs: every function exists with the documented signature (parameter-name parity) and raises `NotImplementedError`.

### Why the goldens are libc-rand stamped, not numpy-RandomState
The upstream kernel calls `rand() % n`. On macOS, libc `rand()` is *not* the same RNG as glibc — so even the upstream's *own* golden outputs are platform-dependent. Our numpy `RandomState.randint(0, n)` is yet a third RNG. There are three reasonable choices:

1. Bit-identical port via `ctypes` to libc's `rand`/`srand` — brittle, doesn't help if upstream ever rebuilds on a different libc.
2. Statistical equivalence — port uses `numpy.random.RandomState` and we test that means/variances/distributions match across many seeds.
3. Per-RNG goldens — capture one set of expected outputs per RNG implementation, asserted exactly within that RNG.

The scaffold supports option 3 cleanly: the existing fixture is the `libc-rand-macos-arm64` golden, and 2c/2d will write a sibling `numpy-randomstate.json.gz` golden by running the ported kernel once at port time. Deterministic functions (8 of 11) use the libc-rand golden directly because they don't touch the RNG; stochastic functions (3) use both — libc-rand for "shape" sanity (sums of mutated arrays match) and numpy-randomstate for byte-identity.

### Verified
- All 10 scaffold tests pass: `pytest tests/test_kinetic_charging_kernel_scaffold.py` → 10 passed in 5.0s.
- Golden file's metadata includes `upstream_sha=330ee3f4...` so re-captures can be diffed against the source SHA.

### What 2b–2e inherit
- `kernel.seed` / `kernel.randint_below` for stochastic functions.
- `_load_golden()` pattern from the scaffold test for parity assertions.
- For stochastic cases, the convention is to compute the ported function's output once at port time, pickle into the test (or a sibling `numpy-randomstate.json.gz` if we want a separate file), and assert bit-identity for that RNG. The libc-rand golden then exists only as documentation + sanity-shape check.

---

## Task #2b progress log

**2026-06-09 — 7 deterministic kernel functions ported.**

### Ported (all `@njit(error_model="numpy")`)
- `get_initiations(elongations, lengths, indexes) -> int` — counts ribosomes with `elongations > 0 and lengths == 0`. `indexes` is unused; kept for API parity with upstream.
- `get_codon_at(sequences, elongations, ith_ribosome, relative_position, absolute_position=0) -> int` — bounded lookup; returns -1 outside range.
- `get_candidates_to_C(sequences, elongations, codon_id) -> (candidates, relative_position)` — C-ward scan. Cleaner signature than upstream (dropped 4 scratch parameters that Cython used to recycle locals).
- `get_candidates_to_N(sequences, elongations, codon_id) -> (candidates, relative_position)` — N-ward mirror.
- `select_candidate(sequences, elongations, relative_position, codon_id, r) -> int` — returns index of `r`-th match. **Discovered during port:** doesn't actually call `rand()` despite upstream test setting `seed_rng(0)` in `setUp` — the RNG draw happens at the caller's `r = rand() % candidates` before `select_candidate` is invoked. So this function is purely deterministic and the RNG seam is exercised first in 2c/2d, not here.
- `is_initial_state(initial_state, state) -> bool` — element-wise int32 equality. Upstream Cython emits "unused function" warning at build (confirmed in the .o object's compiler warnings). Ported for API symmetry.
- `get_codons_read(sequences, elongations, size) -> int64[size]` — codon-usage histogram.

### Parity verification
- `tests/test_kinetic_charging_kernel.py` — 7 parity tests, each iterates every relevant case from `tests/fixtures/trna_charging_kernel_golden.json.gz` and asserts bit-identical output vs the upstream Cython captures.
- Plus `test_is_initial_state_local_cases` (no golden coverage since upstream doesn't call it).
- Plus `test_2b_covers_every_relevant_golden_case` belt-and-suspenders gate.
- Stub-still-raises tests for 2c/2d/2e updated to assert `Task 2c`/`2d`/`2e` markers in the NotImplementedError messages.

### Results
- `pytest tests/test_kinetic_charging_kernel.py tests/test_kinetic_charging_kernel_scaffold.py` → 18 passed, 3 skipped, 3.93s including numba JIT warm-up.

### Notes for 2c/2d
- `select_candidate` being deterministic means the RNG-seam introduction in `reconcile_via_*` is the first place the seeded-RandomState policy gets exercised. The pattern will be:
  1. `kernel.seed(seed)` once at the top of the function.
  2. `r = kernel.randint_below(candidates)` where upstream did `r = rand() % candidates`.
  3. Parity tests assert against a per-RNG golden captured at port time (the existing `libc-rand-macos-arm64` golden is for sanity-shape only on stochastic cases).
- `get_codon_at` is called from `get_candidates_to_C/N`, `select_candidate`, and (in upstream) from `reconcile_via_ribosome_positions`. Numba can nest @njit calls — confirmed working in this commit. 2c can call our ported `get_codon_at`, `get_candidates_to_C/N`, `select_candidate` directly.

---

## Task #2c progress log

**2026-06-09 — `reconcile_via_ribosome_positions` ported.**

### Implementation
- ~140 LOC of pure-Python orchestration in `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py`. Calls the 2b `@njit`'d helpers (`get_candidates_to_C/N`, `select_candidate`, `get_codon_at`) for the array-walk hot paths. RNG draws go through the module-level `randint_below`.

### Two non-obvious upstream behaviors preserved
1. **`disagreements_remaining` leaks across attempts.** Initialized to `True` at function entry; phase 1 sets it `False` on exit and never resets it; phase 2 explicitly resets it to `True`. Consequence: on attempt 2+, the forward phase is **skipped** — only the backward phase runs. `test_reconcile_attempts_threshold` depends on this for its `[10, 15]` expected output (without the leak, the algorithm would oscillate forward/backward indefinitely on this non-convergent input). Documented in the module docstring + worth re-reading if 2d turns up a similar puzzle.
2. **Phase 2 has no `exhausted` array** because `sequence_codons[c] > 0` implies the codon must be reachable somewhere in the consumed-so-far range, so `get_candidates_to_N` is guaranteed to find ≥1 candidate. Upstream relies on this — no check before `r = rand() % candidates`.

### Parity strategy
Two complementary tests for `reconcile_via_ribosome_positions` (and the pattern carries to 2d and 2e):

1. **Byte-identity vs `numpy-RandomState` golden** — `tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz` (1.1 KB gz, 5.9 KB raw). Regenerated by `workspace/investigations/trna-charging-final/capture_numpy_randomstate_golden.py` (runs the v2ecoli port against each `(input, seed)` triple from the libc-rand golden and saves the result). Detects regressions: any change to the port that alters output for these seeds must be intentional + bundled with a re-captured golden.
2. **Algorithmic invariants vs libc-rand golden** — bytes will differ because libc `rand()` and `RandomState.randint` pick different ribosomes, but these properties are RNG-independent and must hold:
   - `kinetics_codons` never mutated.
   - `sequence_codons` and `elongations` stay non-negative.
   - Conservation: `delta(elongations.sum()) == delta(sequence_codons.sum())` (each step changes both by 1).
   - Convergence parity: when upstream reached `compromise == 0`, v2ecoli must too.

### Sanity check across goldens
For `reconcile_via_ribosome_positions/equal` (no RNG calls): byte-identical libc ↔ numpy. For `reconcile_via_ribosome_positions/attempts_threshold`: both reach `[10, 15]` final `sequence_codons` and `elongations.sum() == 25` despite different per-ribosome elongations. Confirms the algorithm's convergence is RNG-invariant.

### Files added/changed
- `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` — body for `reconcile_via_ribosome_positions`, docstring polish.
- `workspace/investigations/trna-charging-final/capture_numpy_randomstate_golden.py` — capture script (skips stochastic functions that are still stubs, so 2d can re-run it later).
- `tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz` — committed per-RNG golden.
- `tests/test_kinetic_charging_kernel.py` — 2 new tests (byte-identity, invariants).
- `tests/test_kinetic_charging_kernel_scaffold.py` — dropped the now-stale `reconcile_via_ribosome_positions` stub assertion.

### Results
`pytest tests/test_kinetic_charging_kernel{,_scaffold}.py` → **20 passed, 2 skipped (for 2d, 2e)** in 3.78 s.

---

## Task #2d progress log

**2026-06-09 — `reconcile_via_trna_pools` ported.**

### Implementation
- ~95 LOC pure-Python orchestration. Same shape as 2c: one `while True` loop, RNG draws via `randint_below`. Three RNG calls per inner iteration (codon pick + tRNA pick; branch determines whether the tRNA pick weights against free or charged pools).

### Two-branch structure
- **Free-tRNA branch** (free tRNAs reading this codon ≥ 1): pick one free tRNA weighted by free count → free → charged. Net: `free_trnas[i] -= 1`, `charged_trnas[i] += 1`, `codons_to_trnas_counter[i, codon] -= 1`, `kinetics_codons[codon] -= 1`.
- **Charged-tRNA branch** (no free): pick one charged tRNA weighted by charged count → undo both the most recent charging *and* the codon read. Net: `chargings[i] -= 1`, `amino_acids_used[aa] -= 1`, `codons_to_trnas_counter[i, codon] -= 1`, `kinetics_codons[codon] -= 1`. Free/charged abundances unchanged (the tRNA went free → charged → free, ending where it started).

### Key difference from 2c
This function **mutates `kinetics_codons`** (decrementing one entry per pick). 2c left it as a read-only input. Loop exits when `kinetics_codons[c] <= sequence_codons[c]` for all `c`, i.e., when no codon is in surplus.

### Parity strategy (same pattern as 2c)
1. **Byte-identity vs `numpy-RandomState` golden** — `tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz` re-captured to include 2 new `reconcile_via_trna_pools` cases (was 8 → now 10 cases, 1.1 KB → 1.3 KB gz).
2. **Invariants vs libc-rand golden**:
   - `sequence_codons` is never mutated (read-only input).
   - Per-tRNA total conservation: `free_trnas[i] + charged_trnas[i]` unchanged.
   - Post-loop: `kinetics_codons[c] <= sequence_codons[c]` for all `c`.
   - Non-negativity: `chargings`, `amino_acids_used`, `codons_to_trnas_counter` ≥ 0.
   - **`kinetics_codons` final state is RNG-invariant** — loop runs until disagreements=0 and each iteration decrements exactly one entry, so the total decrement is fully determined by initial disagreements. Asserts byte-identity for `kinetics_codons_out` vs upstream.

### Sanity check
Both libc-rand test cases (`use_free_trna`, `forward_undo_charging`) yield byte-identical libc ↔ numpy outputs. This is because the test inputs only allow one viable tRNA pick per iteration (only one tRNA reads each codon, etc.) — the RNG sequence has nothing to distinguish.

### Files added/changed
- `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` — body for `reconcile_via_trna_pools`, docstring polish.
- `tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz` — refreshed (10 cases now, was 8).
- `tests/test_kinetic_charging_kernel.py` — 2 new tests for trna_pools (byte-identity, invariants).
- `tests/test_kinetic_charging_kernel_scaffold.py` — dropped now-stale `reconcile_via_trna_pools` stub assertion.

### Results
`pytest tests/test_kinetic_charging_kernel{,_scaffold}.py` → **22 passed, 1 skipped (only 2e left)** in 3.83 s.

---

## Task #2e progress log

**2026-06-09 — `get_elongation_rate` ported. Task #2 (the full Cython kernel port) is complete.**

### Implementation
- ~110 LOC of `@njit(error_model="numpy")` in `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py`. Pure deterministic — no RNG, no mutating writes to inputs.

### Algorithm
Binary search over column count to find the number of timesteps whose measured elongation rate is closest to `target`. The rate at a given `col` is `(# of non-(-1) entries in sequences[:, :col]) / n_ribosomes / time`. Each search step counts that subarray, compares to target, and bisects `[lower, upper]`. Termination: the search proposes a `col` already visited.

After the binary search, the "snap to time-step boundary" logic returns the column count divided by `time_int` (when `best_col % 2 == 0`) or the floor/ceil rounded step whose rate is closer to target.

### Quirks preserved verbatim from upstream
- The `best_col % 2 == 0` check is only strictly correct for `time_int <= 2`. For higher integer time values it can return the wrong rounded result, but we match upstream for byte-identity. Documented in the function docstring.
- When the binary search hits `rate == target` and breaks without incrementing `index`, the post-loop "find best" sweep `for i in range(index)` skips the just-written entry. Upstream's Cython relies on `cdef int i = 0` initialization to make this case work via the degenerate `range(0)`; my port initializes `best_i = 0` to match.

### Parity verification
- `test_get_elongation_rate_parity` — uses the 1 captured case (basic 2×5 setup, target=4) and asserts bit-identical output (4) vs upstream Cython. Passes.
- `test_reconcile_seed_propagates_to_kernel_output` — replaces upstream's `test_reconcile_different_seeds_different_results`. **Surprising finding while writing this test:** with our `numpy.random.RandomState`, the upstream big_seed_12345 vs big_seed_54321 inputs converge to **identical** final state. (`libc rand` DID produce different elongations on the same inputs — seed1: `[8,5,0,10,3]`, seed2: `[8,5,0,8,2]`.) This is a real RNG × inputs coincidence, not a port bug. Adapted the test to use `attempts_threshold` inputs (10 ribosomes, 5 viable forward picks) where divergence is reliable.

### Stub gate replaced
- `tests/test_kinetic_charging_kernel_scaffold.py::test_stubs_raise_not_implemented_until_filled_in` → renamed to `test_no_stub_functions_remain`. Iterates every documented kernel function and calls it with shape-minimal inputs; fails fast if any function still raises `NotImplementedError`. Catches drift in case a future refactor reintroduces a stub.

### Results
- `pytest tests/test_kinetic_charging_kernel{,_scaffold}.py` → **24 passed, 0 skipped** in 2.89 s.
- Runs cleanly under `-m 'not sim'` (no markers needed — they're already fast unit tests).

## Task #2 final summary

The full upstream `_trna_charging.pyx` kernel (638 LOC Cython, 11 functions) is now ported to `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` (~700 LOC Python + numba + docstrings). Parity is enforced by two committed goldens:

- `tests/fixtures/trna_charging_kernel_golden.json.gz` (25 cases, libc-rand-macos-arm64 capture from upstream Cython at SHA `330ee3f4`)
- `tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz` (10 stochastic cases, regenerated by `capture_numpy_randomstate_golden.py`)

8 deterministic functions assert byte-identity vs upstream. 3 stochastic functions assert byte-identity vs the per-RNG numpy golden, plus algorithmic invariants (conservation, non-negativity, convergence) vs the upstream libc golden.

The kernel is ready to be wired into `KineticTrnaChargingModel` (Task #3).

---

## Task #3a progress log

**2026-06-09 — `KineticTrnaChargingPolypeptideElongation` scaffold landed.**

### Architecture decision
v2ecoli's polypeptide architecture is different from upstream:

- **Upstream:** `PolypeptideElongation` is one Process; each elongation MODEL (`BaseElongationModel`, `TranslationSupplyElongationModel`, `SteadyStateElongationModel`, `KineticTrnaChargingModel`) is a strategy class on it. The model is selected via the `trna_charging_model` config flag.
- **v2ecoli:** each model is its own Process subclass of `BasePolypeptideElongation` (which subclasses `PartitionedProcess`). Model is selected per-composite by importing the right class.

So `KineticTrnaChargingPolypeptideElongation` extends `BasePolypeptideElongation` as a peer of `SteadyStatePolypeptideElongation`. Composite arch `kinetic_charging_baseline` (3f) picks it instead of `SteadyStatePolypeptideElongation`.

### File location & circular-import management
- Class lives at `v2ecoli/processes/polypeptide/kinetic_charging.py` (per HANDOFF spec).
- The new file imports `BasePolypeptideElongation` from `v2ecoli/processes/polypeptide_elongation.py`. **No circular import** because `polypeptide_elongation.py` doesn't import the new file — the composite arch (3f) is the only thing that imports `kinetic_charging.py`.

### Class scaffold
- `KineticTrnaChargingPolypeptideElongation(BasePolypeptideElongation)`
- Inherits `name`, `topology` from base (so partitioner treats it as the same process slot — only one elongation model active at a time).
- `description` field summarizes the model in one block (per v2ecoli convention).
- `config_schema = {**Base.config_schema, ...12 new kinetic keys...}` — dict-merge inheritance preserves every base entry. New keys cover codon-sequence tables, tRNA↔codon mapping, kinetic constants, and reconciliation buffer. Defaults are empty / zero-shaped; Task #5 will populate from `sim_data.relation`.

### Method stubs
14 methods stubbed, each raising `NotImplementedError` with an explicit task marker (`"Task 3b"`, `"Task 3c"`, etc.) in the message. Catches accidental composite builds against the partial port — the runtime error tells you which session to pick up next.

| Method | Owner | Notes |
|---|---|---|
| `initialize` | 3b | Calls `super().initialize(config)` then raises. |
| `get_kinetic_constants` | 3b | Mass-density-dependent kinetic constants. |
| `elongation_rate` | 3c | Wraps `kernel.get_elongation_rate`. |
| `request` | 3c | Runs `run_model` for resource sizing. |
| `run_model` | 3c | Predict deltas before allocation. |
| `codon_sequences_width` | 3c | Read-ahead width for next tick. |
| `sequences` | 3c | Ribosome-position → codon array. |
| `max_charging_rate` | 3c | Kinetic ceiling for the allocation. |
| `final_amino_acids` | 3d | AA pool + charged-tRNA contribution. |
| `evolve` | 3d | Apply elongation + reconciliation. |
| `reconcile` | 3d | Calls kernel reconcile_via_* pair. |
| `protein_maturation` | 3d | N-terminal Met cleavage by MAP. |
| `monomer_to_aa` | 3e | Codon ID → amino acid ID. |
| `monomer_limit` | 3e | Per-codon usage cap. |

### Smoke test
`tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` — 20 tests, all green:
- 6 structural checks (module imports, subclass relationship, name/topology inheritance, all kinetic keys present, base keys preserved, schema entries well-formed).
- 14 parametric checks (one per stubbed method) verifying the task-marker string appears in each stub's source. If 3b lands `initialize` but forgets to drop the test marker, this catches it.

`pytest tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` → 20 passed, 1 warning, 3.80 s.

### What's gated for 3b
- 3b can't drop any of the 14 stubs without re-running the scaffold test — the marker check fails immediately.
- The smoke test would have caught a typo in any config_schema key name (since each is listed explicitly).
- Inheritance pattern verified, so 3b doesn't have to re-derive how to extend the base.

---

## Task #3b progress log

**2026-06-09 — `__init__` parameter unpacking + `get_kinetic_constants` ported.**

### Implementation
Replaced 2 stubs in `v2ecoli/processes/polypeptide/kinetic_charging.py`:

**`initialize(config)`** (~90 LOC) — calls `super().initialize(config)` then unpacks all the kinetic-charging-specific params into instance attrs:
- Constants: `cell_density` (from base's `cellDensity` schema key).
- Codon sequences: `protein_sequences`, `monomer_weights_incorporated`, `n_monomers`, `i_start_codon`, `is_map_substrate`.
- Tools: `n_trnas`, `n_codons`, plus the 6-segment slice index layout for the molecules buffer (`slice_free_trnas`, `slice_charged_trnas`, `slice_amino_acids`, `slice_charging_counter`, `slice_reading_counter`, `slice_codons_to_trnas_counter`).
- Mapping arrays: `trnas_to_amino_acids`, `amino_acids_to_trnas`, `trnas_to_codons`, `codons_to_trnas`, `codons_to_amino_acids`, and the derived `trnas_to_amino_acid_indexes`.
- `max_attempts = np.byte(4)` for the kernel reconcile loop.
- Kinetic params: `k_cat__per_s`, `K_M_amino_acid__per_L`, `K_M_trna__per_L`.
- `buffer = reconciliation_buffer`.
- `previous_rate` warm-start for the next tick's binary search.

**`get_kinetic_constants(cell_mass)`** (~5 LOC) — converts the per-litre Michaelis constants back to per-cell quantities using current cell volume (`cell_mass × fg / cell_density`).

### Differences from upstream documented in the docstrings
- `self.process.X` → `self.X` (v2ecoli's class IS the process; no parent reference).
- `cellDensity` matches the base schema key (upstream calls it `cell_density`).
- `n_avogadro` already set by base; not re-fetched.
- pint Quantities throughout (`.to(units.L).magnitude`) instead of Unum's `.asNumber()`.

### Tests
`tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` extended with 5 new tests (23 total):
1. **`test_initialize_is_no_longer_a_stub`** — fails loud if `initialize` or `get_kinetic_constants` regress to NotImplementedError or carry a `Task 3b` marker.
2. **`test_initialize_sets_documented_attrs_via_source_scan`** — pure source scan, no cache needed. 27 expected `self.X = ...` assignments must appear in `initialize`'s body. Catches accidental drops in a future refactor.
3. **`test_get_kinetic_constants_uses_cell_volume_conversion`** — source scan that `cell_volume`, `cell_density`, and both K_M outputs are referenced (guards against regression to the pre-port "return the inputs verbatim" form).
4. **`test_initialize_runs_end_to_end_against_cache`** (`@pytest.mark.sim @_needs_cache`) — loads the real cache, augments with synthetic kinetic extensions via a `_make_kinetic_extensions` helper, instantiates the class, verifies all the kinetic attrs landed with the right shapes/values.
5. **`test_get_kinetic_constants_returns_volume_scaled_arrays`** — instantiates + asserts doubling `cell_mass` doubles the returned K_M arrays.

Also removed `initialize` and `get_kinetic_constants` from the parametric `test_method_stub_carries_task_marker` list (they're no longer stubs).

### Results
- `pytest tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` → **23 passed, 1 warning** in 3.74 s.
- `pytest tests/test_kinetic_charging_*.py` → **47 passed, 1 warning** in 3.30 s. No regressions in the kernel tests.

### Synthetic extensions helper for 3c–3e
`_make_kinetic_extensions(n_aas, n_trnas, n_codons, n_proteins, n_synthetases)` in the test file builds a dict that mirrors the shape contract Task #5 will populate from `sim_data.relation`. 3c–3e can reuse it via test fixtures without re-writing the boilerplate.

---

## Task #3c progress log

**2026-06-09 — Request-side methods ported.**

### Implementation
Six stubs replaced in `v2ecoli/processes/polypeptide/kinetic_charging.py` plus one base method override:

- **`_init_bulk_indices(bulk_ids)`** (override, ~10 LOC) — extends base layout with `atp_idx`, `amp_idx`, `ppi_idx`, `met_idx`, `map_idx`. Mirrors the upstream `PolypeptideElongation.calculate_request` block at lines 534–538.
- **`elongation_rate(states)`** (~45 LOC) — re-derives `protein_indexes/peptide_lengths` from `states["active_ribosome"]` (v2ecoli's contract doesn't pass them); builds `self.longer_sequences` (codon-based) via `buildSequences`; calls `kernel.get_elongation_rate`; updates `self.previous_rate` warm-start.
- **`request(states, aasInSequences)`** (~95 LOC) — IGNORES `aasInSequences` (it's amino-acid-domain; we work in codons); recomputes `monomers_in_sequences` from `self.longer_sequences`; runs `run_model` against `"bulk_total"`; builds bulk requests for AAs (+1% buffer), ATP, both tRNA pools, synthetases, MAP, and water (incl. termination); returns `(fraction_charged, amino_acids_used, requests)`.
- **`run_model(codons, attr, states)`** (~270 LOC) — drives a `scipy.integrate.solve_ivp` RK45 ODE over the 6-segment molecules buffer. Inner `ode_model` closure computes charging + reading rates with sin/sin² roll-offs for low charged-fraction and low AA-pool corner cases. On `attr="bulk_total"` precomputes `K_M_amino_acids`, `K_M_trnas`, `cell_amino_acid_saturation` for the subsequent `"bulk"` call to reuse. Discretizes outputs (ceil for sizing, `stochasticRound` for evolve), caps at AA availability, reconciles tRNA-pool under/overflow. Emits `trna_charging.{saturation_trna, turnover}` listener fields on `"bulk"`.
- **`max_charging_rate(states, attr)`** (~5 LOC) — `v_max = k_cat__per_s * n_synthetases`. Uses `self.synthetase_idx` (v2ecoli) for upstream's `self.process.trna_synthetases_for_aas_idx`.
- **`codon_sequences_width(elongation_rates)`** (~1 LOC) — returns the per-tick width cached in `elongation_rate`. The `elongation_rates` arg is unused (kinetic model fixes width at basal + buffer).
- **`sequences(sequences)`** (~1 LOC) — returns `self.longer_sequences`. The `sequences` arg is intentionally ignored (kept for API parity with upstream).

### Upstream-vs-v2ecoli diffs documented in docstrings
- `self.process.X_idx` → `self.X_idx` (v2ecoli's class IS the process).
- Unum's `.asNumber(...)` → pint's `.to(...).magnitude`.
- v2ecoli's `request(states, aasInSequences)` contract vs upstream's `request(states, monomers_in_sequences, protein_indexes, peptide_lengths)` — `aasInSequences` is ignored and the kinetic model recomputes its own codon-domain `monomers_in_sequences`.

### Tests
`tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` grows from 23 to 28 tests:
- Removes 6 entries from the `test_method_stub_carries_task_marker` parametric list (they're ported now).
- Adds `test_3c_method_no_longer_stub` (parametric, 7 methods incl. `_init_bulk_indices`) — fails loud if any reverts.
- Adds `test_elongation_rate_calls_kernel_and_sets_longer_sequences` — source-scan verifying `kernel.get_elongation_rate`, `buildSequences`, `self.longer_sequences`, `self.sequences_width`, `self.previous_rate` are referenced.
- Adds `test_request_requests_all_kinetic_bulk_keys` — source-scan verifying every required bulk index is referenced (amino_acid, atp, uncharged_trna, charged_trna, synthetase, map, water) plus the v2ecoli return tuple.
- Adds `test_run_model_uses_ode_and_kernel_constants` — source-scan verifying `solve_ivp` with RK45 + rtol=1e-4 + atol=1e-7, `K_M_amino_acids`, `K_M_trnas`, `stochasticRound`, and the 7-tuple return.
- Adds `test_init_bulk_indices_adds_kinetic_keys` — source-scan verifying `super()._init_bulk_indices(bulk_ids)` is called and the 5 new indices are referenced.

### Why source-scan instead of runtime
A full `run_model` end-to-end requires the cache to populate the kinetic config keys (currently only synthetic-shape-only via `_make_kinetic_extensions`) AND the active_ribosome / bulk / listeners state to be plausibly initialized. End-to-end runtime tests come together in Task 3f's behavior test after Task #5 wires up sim_data.

### Results
- `pytest tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` → **28 passed, 1 warning** in 4.74 s.
- `pytest tests/test_kinetic_charging_*.py -m 'not sim'` → **50 passed, 2 deselected, 1 warning** in 3.52 s. No regressions in kernel tests or the cache-gated 3b instantiation.

### Remaining work
- 3d: `evolve`, `reconcile`, `protein_maturation`, `final_amino_acids`.
- 3e: `monomer_to_aa`, `monomer_limit`, listener emission.
- 3f: composite arch + behavior test (depends on Task #5).

---

## Task #3d progress log

**2026-06-09 — `evolve_state` override + 7 evolve-side methods ported.**

### Scope expansion flagged at planning time
The user's 3d task description listed 4 methods (`evolve`, `reconcile`, `protein_maturation`, `final_amino_acids`). v2ecoli's architecture forced a wider scope:

- v2ecoli's `BasePolypeptideElongation.evolve_state` is amino-acid-centric (polymerizes against `aa_counts_for_translation`). The kinetic model needs codon-based polymerize → reconcile → protein_maturation → evolve, so we must override `evolve_state` *entirely* (~225 LOC).
- `evolve_state` consumes `monomer_to_aa`, `monomer_limit`, `next_amino_acids` — originally slated for 3e. Pulled forward into 3d so the override is functional.
- `final_amino_acids` is the AA-based hook the kinetic model *bypasses*; left as a NotImplementedError raise with an explanatory message ("kinetic model uses monomer_limit via evolve_state override") so accidental re-routing surfaces loud.

### Implementation
Replaced 5 stubs + added 2 helpers + 1 override in `v2ecoli/processes/polypeptide/kinetic_charging.py`:

**`evolve_state(timestep, states)`** (~225 LOC, override) — codon-based replacement for the base. Builds both AA sequences (for polymerize) and codon sequences (for reconcile), runs polymerize against the codon limit, calls `reconcile` → `protein_maturation` → `evolve`, emits the listener block. Drops upstream's non-kinetic branches (served by other v2ecoli classes).

**`reconcile(states, result)`** (~80 LOC) — runs `run_model` against `"bulk"`, compares predicted vs realized per-codon usage, seeds the kernel RNG, dispatches through `kernel.reconcile_via_ribosome_positions` (first pass) and `kernel.reconcile_via_trna_pools` (fallback). Emits `initial_disagreements`, `charging_events`, `reading_events`, `codons_to_trnas_counter` listener fields.

**`protein_maturation(states, did_terminate, terminated_proteins, protein_indexes)`** (~65 LOC) — MAP kinetic capacity (`k_cat = 6 / s`, per-cell concentration) caps cleavages; deferred terminations get rolled back via `multinomial(not_cleaved, candidates/candidates.sum())`. Unum → pint conversion `.asNumber()` → `.to(units.dimensionless).magnitude`.

**`evolve(...)`** (~55 LOC, 9-arg signature) — builds bulk deltas: initialization water, net tRNA charging, AA used, ATP/AMP/PPi per charging event, proton per charged-tRNA-mediated incorporation, water per direct elongation, water consumed + Met released per cleaved initial Met.

**`monomer_to_aa(monomer)`** (1 LOC) — `codons_to_amino_acids @ monomer`.

**`monomer_limit(states, _)`** (4 LOC) — returns `(codons_kinetics_model, codons_to_amino_acids @ codons_kinetics_model)`.

**`next_amino_acids(all_sequences, sequence_elongations)`** (1 LOC) — returns 0 (matches upstream Base default).

**`final_amino_acids(...)`** — kept as NotImplementedError with bypass-explanation message.

### Tests
`tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` grows from 28 to 31 tests:
- Drops the parametric `test_method_stub_carries_task_marker` (all entries ported).
- Adds `test_no_task_3_stub_markers_remain_in_any_method` (sentinel: scans every method's source for `raise NotImplementedError` paired with `Task 3X`; docstring cross-refs ignored).
- Adds 8 source-scan tests covering:
  - `evolve_state` override + all kinetic hook calls in order
  - `reconcile` calls both kernel helpers + seeds RNG + emits listener fields
  - `protein_maturation` uses MAP kinetics with stochasticRound
  - `evolve` emits all 9 bulk indices
  - `final_amino_acids` raises with the bypass-explanation message
  - `monomer_to_aa` uses the matmul
  - `monomer_limit` returns the prediction tuple
  - `next_amino_acids` returns 0

### Results
- `pytest tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` → **31 passed, 1 warning** in 2.85 s.
- `pytest tests/test_kinetic_charging_*.py -m 'not sim'` → **53 passed, 2 deselected, 1 warning** in 2.99 s. No regressions.

### Why source-scan still
`evolve_state` end-to-end needs both Task #5 (sim_data populates the kinetic config keys) AND the partitioned-step machinery (allocation, active_ribosome state). The cache-gated `test_initialize_runs_end_to_end_against_cache` already verifies instantiation; full evolve-tick verification is Task 3f's behavior test.

### Remaining work
- 3e collapsed: only listener emission paths remained, and they were folded into the `evolve_state` override. 3e is effectively a no-op now — fold into 3f's behavior test or mark complete.
- 3f: composite arch + behavior test (depends on Task #5).
