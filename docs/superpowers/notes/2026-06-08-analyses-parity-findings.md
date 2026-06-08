# Analyses Parity Findings — 2026-06-08

Discovery task: confirm what `sim_data` attributes and parquet columns the five
proving-set analyses require, and whether they exist in v2ecoli's outputs.

Reference parquet:
`out/compare_harness/v2_sim/parquet/two_generations/history/experiment_id=two_generations/variant=0/lineage_seed=0/generation=0/agent_id=0/800.pq`

Reference sim_data pickle: `out/kb/simData.cPickle`

---

## 1. Parquet columns — present / missing

### Present (all five analyses can use these directly)

| Column | Used by |
|---|---|
| `listeners__rna_counts__full_mRNA_counts` | ptools_rna, ptools_rxns (default), ptools_proteins (default) |
| `listeners__fba_results__base_reaction_fluxes` | ptools_rxns, central_carbon_metabolism_scatter |
| `listeners__mass__rna_mass` | mass_fraction_voronoi |
| `listeners__mass__protein_mass` | mass_fraction_voronoi |
| `listeners__mass__tRna_mass` | mass_fraction_voronoi |
| `listeners__mass__rRna_mass` | mass_fraction_voronoi |
| `listeners__mass__mRna_mass` | mass_fraction_voronoi |
| `listeners__mass__dna_mass` | mass_fraction_voronoi |
| `listeners__mass__smallMolecule_mass` | mass_fraction_voronoi |
| `listeners__mass__cell_mass` | central_carbon_metabolism_scatter |
| `listeners__mass__dry_mass` | central_carbon_metabolism_scatter |

### Missing — require shims

| vEcoli column | v2ecoli equivalent | Notes |
|---|---|---|
| `bulk` | `bulk__id` + `bulk__count` | v2ecoli splits the struct into two parallel arrays. The ptools analyses call `np.stack(output_df["bulk"].values)` to get an `(n_timepoints, n_bulk_molecules)` matrix. Shim: join on `sim_data.internal_state.bulk_molecules.bulk_data["id"]` to produce the positionally-ordered count array. **Key constraint**: `bulk__id` order in the parquet may not match `sim_data` bulk order, so the shim must reindex. |
| `listeners__unique_molecule_counts__active_ribosome` | `list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)` | v2ecoli has no `unique_molecule_counts` listener. `n_ribosomes_per_transcript` is a per-transcript count array; its sum per row = active ribosome count. Verified: sum ≈ 14 857 at one timestep — plausible. |
| `listeners__unique_molecule_counts__oriC` | `listeners__replication_data__number_of_oric` | Direct scalar replacement; same semantics. Verified present (value = 2 at one timestep). |
| `listeners__unique_molecule_counts__active_RNAP` | `len(listeners__rnap_data__active_rnap_unique_indexes)` | `active_rnap_unique_indexes` is an `INTEGER[]`; its length per row = number of active RNAPs. Verified: 870 at one timestep — plausible. |

---

## 2. sim_data attributes — present / missing

All probed attributes are **present**. v2ecoli's `SimulationDataEcoli` object is
API-compatible with vEcoli's (it is the same class, fork-evolved in sync).

| Attribute path | Status | Used by |
|---|---|---|
| `process.transcription.rna_data` | OK | ptools_rna |
| `process.transcription.rna_maturation_stoich_matrix` | OK | ptools_rna |
| `process.transcription.mature_rna_data` | OK | ptools_rna |
| `process.transcription.uncharged_trna_names` | OK | ptools_rna |
| `process.transcription.charged_trna_names` | OK | ptools_rna |
| `process.complexation.get_monomers` | OK | ptools_rna, ptools_proteins |
| `internal_state.bulk_molecules.bulk_data` | OK | ptools_rna, ptools_proteins |
| `molecule_groups.s50_23s_rRNA` | OK | ptools_rna |
| `molecule_groups.s30_16s_rRNA` | OK | ptools_rna |
| `molecule_groups.s50_5s_rRNA` | OK | ptools_rna |
| `process.metabolism.base_reaction_ids` | OK | ptools_rxns |
| `process.translation.monomer_data` | OK | ptools_proteins |
| `molecule_groups.replisome_monomer_subunits` | OK | ptools_proteins |
| `molecule_groups.replisome_trimer_subunits` | OK | ptools_proteins |
| `molecule_ids.s30_full_complex` | OK | ptools_proteins |
| `molecule_ids.s50_full_complex` | OK | ptools_proteins |
| `molecule_ids.full_RNAP` | OK | ptools_proteins |
| `constants.n_avogadro` | OK | mass_fraction_voronoi |
| `getter.get_masses` | OK | mass_fraction_voronoi |
| `getter.get_mass` | OK | mass_fraction_voronoi |
| `constants.cell_density` | OK | central_carbon_metabolism_scatter |

---

## 3. Per-analysis verdict

| Analysis | Scale | Verdict | Notes |
|---|---|---|---|
| `ptools_rna` | single | **port with shims A+B** | Needs `bulk` shim (A) and `unique_molecule_counts__active_ribosome` shim (B). All sim_data attributes present. |
| `ptools_rxns` | single | **port as-is** | Only needs `bulk` (for default `read_outputs` signature — actually only uses `listeners__fba_results__base_reaction_fluxes` in its main body) and `listeners__fba_results__base_reaction_fluxes`. The latter is present; `bulk` appears only in the shared `read_outputs` default but is not used in `ptools_rxns` main logic. However if `bulk` is passed as a literal column it will fail — shim A still needed if the `read_outputs` helper is called with default columns. **Recommendation**: port as-is, calling `read_outputs` with an explicit `columns=["listeners__fba_results__base_reaction_fluxes"]` rather than the default. |
| `ptools_proteins` | single | **port with shims A+C+D** | Needs `bulk` shim (A), `oriC` shim (C: `listeners__replication_data__number_of_oric`), and `active_RNAP` shim (D: `len(active_rnap_unique_indexes)`). Also needs `active_ribosome` shim (B). All sim_data attributes present. |
| `mass_fraction_voronoi` | single | **port as-is** | All seven `listeners__mass__*` columns present. Uses `sim_data.constants.n_avogadro`, `getter.get_masses/get_mass` — all present. No `bulk` or `unique_molecule_counts` needed. |
| `central_carbon_metabolism_scatter` | multiseed | **port as-is** | `listeners__fba_results__base_reaction_fluxes`, `listeners__mass__cell_mass`, `listeners__mass__dry_mass` all present. `sim_data.constants.cell_density` present. `sim_data.process.metabolism.base_reaction_ids` present. No missing columns. |

### Shim summary (for Tasks 5–7)

| Shim | Implementation |
|---|---|
| **A — `bulk` column** | After `read_outputs`, reconstruct positional count matrix: for each row, build a dict `{id: count}` from `bulk__id`/`bulk__count`, then reindex against `sim_data.internal_state.bulk_molecules.bulk_data["id"]`. Return as `ndarray` to match `np.stack(output_df["bulk"].values)` usage. |
| **B — `active_ribosome`** | `list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)` as an extra derived column in the SELECT. |
| **C — `oriC`** | Alias `listeners__replication_data__number_of_oric` → `listeners__unique_molecule_counts__oriC`. |
| **D — `active_RNAP`** | `len(listeners__rnap_data__active_rnap_unique_indexes)` as an extra derived column (DuckDB: `array_length(...)` or `len(...)`). |

Shims A–D can all be encapsulated in a thin `read_outputs_v2` adapter function in the ported modules (or in a shared helper imported by Tasks 5–7), keeping the main analysis logic unchanged.
