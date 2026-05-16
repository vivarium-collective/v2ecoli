# Investigation walkthrough — friction notes (live)

> The author has decided to try executing the DnaA / Replication Initiation
> investigation. This file accumulates friction points + streamlining
> proposals as we go. Companion to
> `2026-05-16-dnaa-studies-test-run.md` (the design-phase notes); this
> file is the **execution-phase** counterpart.

## Workspace state at start

- Branch: `dnaa-replication-studies` on v2ecoli.
- All 6 dnaa-* studies fully designed (Q + H + EB DSL + variants +
  interventions + gaps + cites + units).
- Only dnaa-01 has `status: implemented` behaviors that can run
  against existing v2ecoli code.
- No runs.db files yet anywhere.

## Goal of this walkthrough

Get the first real run on disk for dnaa-01's baseline. Surface the
gap between "study spec says X" and "v2ecoli actually emits Y".
Document every step so /pbg-study can eventually automate the
end-to-end loop.

---

## Phase 0 — Sanity-check identifiers in the existing baseline

### Friction #1 — `models/<arch>.pbg` is structural-only

I assumed `models/partitioned.pbg` carried real molecule counts so the
bigraph picker (#17 from the design-phase notes) would show real
initial-state values. It doesn't: it carries only the type schema +
default values (most counts = 0). The actual initial counts live at
`out/cache/initial_state.json`, populated by ParCa.

**Implication**: the dashboard's bigraph picker shows store *shapes*
but no live values. Should the API also serve initial_state.json so
the picker can preview a count next to each leaf?

### Friction #2 — bulk vs. unique storage mismatch in the EB DSL

**Symptom.** dnaa-01's `dnaA_mrna_count` observable is declared as
`store_path: agents.0.bulk` with `index_by: {type: bulk_id, value:
EG10235_RNA}`. Loaded `out/cache/initial_state.json`, searched `bulk`:
`EG10235_RNA` is NOT in the bulk array. dnaA mRNA is tracked as a
**unique molecule** in `agents.<id>.unique.RNA`, indexed by `TU_index`
(an integer mapping into `sim_data.process.transcription.rna_data.id`).

Three storage classes I had collapsed into one:
- **bulk**: molecules tracked by total count per id (most metabolites,
  some proteins like DnaA's `PD03831[c]`).
- **unique**: individual molecules with attributes (mRNAs, ribosomes,
  replisomes, genes, oriCs, DnaA_boxes, etc.).
- **listener**: derived aggregate (e.g.,
  `monomer_counts_listener` re-aggregates uniques + bulk via
  complexation/equilibrium stoichiometry).

**Implication for the EB DSL** (proposal to lift in #16-rev):
The `measure.kind` needs more granularity:
- `bulk_count` (id) — what I have
- `unique_count` (path, filter) — count unique molecules matching a
  predicate (e.g., `unique.RNA where TU_index == EG10235_idx`)
- `listener_indexed` (path, index_lookup) — read a specific index
  of a listener array via a sim_data lookup (e.g.,
  `monomer_counts_listener.monomerCounts[monomer_ids.index('PD03831[c]')]`)

### Friction #3 — DnaA's bulk ID is **`PD03831[c]`**, not `MONOMER0-160[c]`

**Symptom.** I scoped dnaa-01 on the assumption that DnaA's bulk id was
`MONOMER0-160[c]` (per the v2ecoli `molecule_ids.py` label
`"DnaA_ATP_complex": "MONOMER0-160[c]"`). But:

- `MONOMER0-160[c]` exists in the bulk at count = **0**. The
  `MONOMER0-160` prefix is a complex-form prefix; the various
  `MONOMER0-160{1,2,3,4,5}[i]/[p]` entries are membrane complexes,
  not the cytoplasmic DnaA monomer.
- DnaA in v2ecoli's `monomer_counts_listener.monomer_ids` resolves
  to `PD03831[c]` (index **3861**). PD03831 is the protein-product
  id linked to EG10235 (dnaA) via `rnas.tsv` column 7.
- Initial count of `PD03831[c]` is **124 molecules**.

**Three identifier ambiguities I trip over here:**

| Concept                  | Plan / spec label   | Actual v2ecoli id           | Initial count |
| ---                      | ---                 | ---                         | ---           |
| DnaA monomer (cytoplasm) | `MONOMER0-160[c]` (wrong) | `PD03831[c]` ✓        | 124           |
| DnaA-ATP complex (label) | `MONOMER0-160[c]`   | `MONOMER0-160[c]`           | 0             |
| dnaA gene                | `EG10235`           | `EG10235` ✓ in unique.gene  | 1             |
| dnaA mRNA (TU)           | `EG10235_RNA`       | `EG10235_RNA` in
                                                     `sim_data.…rna_data['id']`
                                                     → idx into unique.RNA  | (depends) |

**Implication.** The spec's `index_by` block needs to know *which
identifier system* to use:
- "bulk_id" — direct bulk-table lookup (works for `PD03831[c]`,
  not for `EG10235_RNA`).
- "rna_id" — for indexing into RNA_counts_listener (and unique.RNA via
  TU_index).
- "monomer_id" — for monomer_counts_listener.

I had `index_by.type` as one of `bulk_id | rna_id | tf_id |
literal_index` in my proposal. Now I see we also need `monomer_id`
(for monomer_counts_listener), and the value space is the per-listener
config table, not raw sim_data. Worth formalizing into
`vivarium_dashboard.lib.expected_behavior` (per #16-rev).

### Friction #4 — Initial count (124) is below the spec's 300-800 band

DnaA at cell birth = 124. The plan's prediction is 300-800/cell at
**steady state**. Two possibilities:
1. 124 is the daughter-cell pool right after division. Across one
   doubling, translation should ramp to 200-400 monomers.
2. v2ecoli's calibration runs at a lower DnaA baseline than the
   experimental measurements.

Either way: **the BT-01 test as written today (median over second-half
of run within [300, 800])** is on shaky ground for the existing
v2ecoli baseline. Possible adjustments:
- Loosen to [100, 800] for the dashboard run, document v2ecoli's
  calibration discrepancy.
- Run multi-generation to verify the ramp-up.
- Push ParCa to calibrate against Schmidt2016 / Mori2021 if we want
  the upper band.

This is exactly the value of running the investigation: design-phase
assumptions hit reality immediately.

### Streamlining proposal: `pbg-study verify-identifiers <slug>`

A new skill / dashboard action that, given a study slug:
1. Loads the workspace's `out/cache/initial_state.json` + `sim_data`.
2. For each observable in `study.yaml.observables`, resolves the
   `index_by` lookup against the real catalogs.
3. Reports: ✓ id resolves to value X · ✗ id `EG10235_RNA` not in bulk
   (try `rna_id` instead of `bulk_id`) · ⚠ id `MONOMER0-160[c]`
   resolves but count is 0 (probably wrong identifier).

Would have caught all 3 of the issues above in one shot. Add to the
P0 shortlist for the listening Claude.

## Phase 1 — Ran the workflow pipeline, hit calibration mismatch

### Friction #5 — `python reports/workflow_report.py` is the right CLI but pulls a cached run

Ran `python reports/workflow_report.py --duration 60 --no-daughters
--study studies/dnaa-01-expression-dynamics/study.yaml`. It hit cached
metadata at every step (parca, load_model, single_cell, division) and
finished in 11s — meaning a previous single-cell simulation already
exists at `out/workflow/single_cell.dill` (t=2350s, ~39 min,
matches a doubling time on minimal media). Convenient for poking at
real data, less convenient for testing a *fresh* run.

**Streamlining**: `--clean` flag exists but blows away everything
including the ParCa cache (10-min rebuild). Want a `--clean-sim`
flag that wipes only `out/workflow/single_cell.dill +
division_meta.json + daughters_meta.json` and keeps the ParCa.

### Friction #6 — three DIFFERENT counts of "DnaA" at the same timestep

At `out/workflow/single_cell.dill` (t=2350s):

| Source                              | Count    |
| ---                                 | ---      |
| `bulk['PD03831[c]']`                | **0**    | ← free DnaA monomer
| `bulk['MONOMER0-160[c]']`           | **100**  | ← DnaA-ATP complex form
| `listeners.monomer_counts[3861]`    | **256,299** | ← aggregate via complexation_stoich

Cross-check:
- `proteinLengths[3861] = 467 aa` matches DnaA's known length ✓
- `proteinIds[3861] = PD03831[c]` ✓
- ATP/ADP at same timestep: 12.3M / 635k = ratio 19.4 (well above 10, so
  BT-02's `atp-pool-much-greater-than-adp` would pass).

**Three different "DnaA counts" are three different concepts**:
1. **Free DnaA monomer in solution** — `bulk['PD03831[c]']`. Near
   zero because most DnaA is bound to DNA / in complexes.
2. **DnaA-ATP complex** — `bulk['MONOMER0-160[c]']`. The actual
   bookkeeping for the ATP-bound form in v2ecoli's existing model.
3. **Total DnaA across all forms** — `monomer_counts[3861]`,
   re-aggregated via `complexation_stoich` from every complex that
   contains a DnaA subunit.

**Which one does the plan's 300-800/cell figure refer to?** Plan cites
Sekimizu 1991 + Schmidt 2016 + Mori 2021. Those reports measure DnaA
by western blot or mass-spec on whole-cell extracts — so they measure
the TOTAL pool (concept 3). The simulated value 256,299 is ≈300×
above. Either:
- v2ecoli is calibrated way above mass-spec values for DnaA.
- The `monomer_counts` aggregation is over-counting because DnaA
  appears in many overlapping `complexation_stoich` rows.
- The published mass-spec values undercount complex-bound DnaA
  (immunoblot doesn't always extract from complexes).

This is exactly the kind of discrepancy the *behavioral test* should
flag for the expert. Our test threshold [300, 800] doesn't fit
v2ecoli's calibration; the test as written would say "FAIL"
unhelpfully without explaining whether the model is wrong or the
threshold is wrong.

### Friction #7 — observables today don't carry "which concept"

dnaa-01's `dnaA_protein_count` observable was specified as
`monomer_counts.monomerCounts[PD03831[c]]` (the aggregate concept 3).
That maps to 256k. The plan's 300-800 was clearly concept 1
(free / "active" DnaA — sometimes called "available initiator").

**The observable schema doesn't yet distinguish these.** Proposals:

1. Add a `concept:` field to observables: `free_pool | total_protein |
   bound | complex_form`. The visualization layer can group plots by
   concept; the expected_behavior evaluator can scale ranges
   appropriately (e.g., the textbook 300-800 is `concept: total_protein`).
2. Have the dashboard's verify-identifiers tool emit:
   `dnaA_protein_count → 256299 (concept inferred as total_protein
   via monomer_counts); your expect range [300, 800] is 320× off`.

### Streamlining proposal: BT-01 acceptance band needs a calibration step

Before the behavioral test can be a real gate, somebody (expert) must
either:
- Confirm that v2ecoli's 256k DnaA matches *complex-aware* mass-spec
  reanalysis (in which case the 300-800 number is the wrong concept
  and the test threshold should be ~10⁵).
- Or recalibrate v2ecoli's DnaA pool by adjusting translation
  efficiency / degradation / etc.

That's a real research decision, not a code fix. It's exactly the
kind of "validate before executing" question the report's "Expert
Questions" section is for. Should auto-surface in the report:
**"v2ecoli currently produces ~256k DnaA at t=2350s; the test
threshold [300, 800] would fail by 320×. Resolve before running."**

## Phase 1 takeaways

After ~30 minutes of digging:
- Validated the v2ecoli pipeline DOES run cleanly today (cached).
- Found 3 different DnaA counts (1 concept mismatch + 1 calibration
  question).
- Updated dnaa-01 study.yaml's `dnaA_protein_count` to use
  `monomer_counts.monomerCounts[PD03831[c]]` (the closest match to
  the textbook concept) but the value disagreement remains.
- Fixed 2 wrong identifiers in observables (`EG10235_RNA` bulk →
  `rna_counts.mRNA_counts[rna_id]`; DnaA was tracked under
  `PD03831[c]`, not `MONOMER0-160[c]`).

**Net**: a real investigation walkthrough surfaces three distinct
streams of work:
1. **Spec quality**: catch wrong identifiers BEFORE running. Tool:
   `pbg-study verify-identifiers`.
2. **Concept-level observability**: bulk count vs total-aggregated
   count vs complex-form are different *concepts*. Tool: `concept:`
   field on observables; report surfaces this.
3. **Calibration disagreements** between v2ecoli and literature
   thresholds. Tool: behavioral test result is annotated with
   "calibration question" not just pass/fail.

These are exactly the streamlining proposals to feed back into
pbg-superpowers.
