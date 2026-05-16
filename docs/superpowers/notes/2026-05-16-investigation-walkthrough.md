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

---

## Phase 2 — How to build studies + investigations MORE ROBUSTLY

User feedback after Phase 1: *"I want to make sure we build future
studies and investigations more robustly based on what we learned.
We should check that all the requested readouts are valid, and even
run some baseline visualizations to confirm things are looking fine
before we resume the study. Each study should have its planned
visualizations that an expert could look over and confirm."*

This is the validation-before-execution theme. Phase 1 showed that
design-phase looks complete but reality finds 3 wrong identifiers in
5 minutes. The fix is a structured **lifecycle** with gates between
phases, plus tooling at each gate.

### Proposed study lifecycle (state machine)

```
planned          (spec exists, no validation done yet)
  ↓ run /pbg-study verify <slug>
spec-validated   (all identifiers resolve; all variants' params have a hook
                  OR are marked aspirational; cross-study refs resolve)
  ↓ run /pbg-study preview-viz <slug>
viz-previewed    (each declared visualization renders against the cached
                  baseline; expert can sanity-check the shape of every plot)
  ↓ expert review (manual)
ready            (expert questions addressed; calibration disagreements
                  resolved; gate to actually execute)
  ↓ run /pbg-study run-baseline (existing)
running          (simulation in flight)
  ↓
ran              (at least one completed run on disk)
  ↓ run /pbg-study run-tests (existing)
tests-passed     (parametrized behaviors all green)
  ↓ expert review
complete         (study contributes its results to investigation acceptance)
```

The existing dashboard `_VALID_STATUSES = {planned, running, ran,
complete, failed, invalid}` covers the *runtime* states but not the
*pre-run* validation states. Three new states needed:
`spec-validated`, `viz-previewed`, `ready`. Each is gated by a
specific tool that produces an artifact + an explicit pass/fail.

### Tool proposals (P0 for the listening Claude)

#### Finding #25 — `/pbg-study verify <slug>`: spec-validated gate

What it does, per study.yaml:
- Loads `out/cache/initial_state.json` + sim_data + listener configs.
- **For each observable**: resolves `index_by` against the right
  catalog (bulk_id → bulk array; rna_id → RNA_counts_listener config;
  monomer_id → monomer_counts_listener config; tf_id → tf_binding
  config). Reports ✓ / ✗ / ⚠ (value found but zero) per entry.
- **For each variant**: checks that its `params:` map onto a real
  composite parameter, OR that the variant is `status: aspirational`
  with a matching `requires:` block.
- **For each behavioral test**: checks that the `measure.path` + any
  `index_by` resolve. Catches the dnaa-01-style "MONOMER0-160[c] is
  the wrong identifier" bug before pytest discovers it.
- **For each cross-study reference** (`requires.cross_study`,
  `compare_to: dnaa-02-baseline`): checks that the named study exists
  + has a corresponding baseline run on disk.
- **Cited bib keys**: every key in `cites:` must be in
  `references/papers.bib`.

Output: a structured JSON report + an HTML render that the
investigation report links to. A failing `verify` blocks the
spec-validated transition.

#### Finding #26 — `/pbg-study preview-viz <slug>`: viz-previewed gate

What it does:
- Reads each entry in `study.yaml.visualizations`.
- For each, renders a preview against the most-recent cached baseline
  run (or against `out/cache/initial_state.json` if no run exists).
- Saves the rendered HTML / SVG to `studies/<slug>/viz/<viz-name>.preview.html`.
- The Visualizations tab in the dashboard shows the rendered plot
  with a "Looks right?" thumbs-up/down chip the expert can click.

Critical missing piece: **a generic `TimeSeriesFromObservables`
Visualization Step** that consumes any list of observable specs +
a runs.db and produces a time-series plot. Without this, studies
that declare `address: local:TimeSeriesPlot` have no backing code
and can't be previewed.

Proposal: add `v2ecoli/visualizations/timeseries_from_observables.py`
that:
1. Reads `study.yaml.observables` for the named observables.
2. Resolves each via the same index_by mechanism `verify` uses.
3. Pulls the time-series from `runs.db` history rows (or constructs
   from `out/workflow/single_cell.dill` if no runs.db yet).
4. Renders a multi-line Plotly chart with the observable name + units
   from the spec.

This single Viz class lights up most studies' declared
visualizations without per-study code. Move it into the dashboard
package long-term so every workspace gets it.

#### Finding #27 — Investigation-level "all studies ready" gate

Each investigation's `acceptance_criteria:` maps to per-study
behavioral tests. The investigation should NOT start runs until:
- Every study in `studies:` is `status: ready` or higher.
- Every acceptance-criterion's (study, behavior) pair resolves
  cleanly via `/pbg-study verify`.
- At least one preview viz per study has been thumbs-up'd by the
  expert.

The Generate-report flow should highlight non-ready studies in red:
*"3 of 6 studies are not yet ready: dnaa-01 (1 viz failed preview),
dnaa-04 (3 variants without hooks)…"*. The expert reads the report,
clicks the red items, and the dashboard guides them to the fix.

### Finding #28 — Each study needs **calibration anchors** alongside expectations

Phase 1 surfaced a 256k vs 800 DnaA discrepancy that broke a behavior
test. The spec didn't say "this is the calibrated value in v2ecoli
today; the textbook value is 800; if they diverge by >Nx, surface
to the expert as an alarm". Propose:

```yaml
# In each behavior with quantitative thresholds:
expect:
  op: in_range
  low: 300
  high: 800
calibration_anchor:
  v2ecoli_observed: 256299     # populated by /pbg-study preview-viz
  literature_target: 550       # midpoint of [300, 800]
  divergence_factor: 466       # auto-computed; > 10 → alarm
  resolution: "expert"          # one of: model | thresholds | concept
```

The behavioral test reports calibration_anchor.divergence_factor
alongside pass/fail so an expert can immediately tell whether the
disagreement is a model-calibration problem or a threshold-vs-concept
mismatch.

### Finding #29 — Generic readout types observed in v2ecoli

For the listening Claude building `verify-identifiers`, here is the
catalog of how each storage concept is reached. This table belongs in
`docs/concepts/observable-storage-classes.md` (which doesn't exist
yet but should).

| Concept | Path pattern | Index by | sim_data catalog |
|---|---|---|---|
| Bulk count | `agents.<id>.bulk` (structured array `id`/`count`) | molecule id | n/a — the array itself carries id strings |
| Aggregated monomer count | `agents.<id>.listeners.monomer_counts.monomerCounts` (ndarray length 4309) | monomer id | `monomer_counts_listener.monomer_ids` |
| RNA TU count | `agents.<id>.listeners.rna_counts.mRNA_counts` (or `full_mRNA_counts`, `partial_mRNA_counts`) | rna id | `RNA_counts_listener.mRNA_TU_ids` |
| Transcription init events | `agents.<id>.listeners.rnap_data.rna_init_event` | TU index | `sim_data.process.transcription.rna_data.id` |
| TF bound per TU | `agents.<id>.listeners.rna_synth_prob.n_bound_TF_per_TU` (n_TU × n_TF) | (TU id, TF id) | TBD |
| Unique molecules | `agents.<id>.unique.<type>` (structured arrays with attributes) | filter by attribute (e.g. TU_index, DnaA_bound) | n/a |
| Mass / volume | `agents.<id>.listeners.mass.{cell_mass, dry_mass, cell_volume}` | scalar | n/a |

Each of those needs a `kind:` in the EB DSL's `measure:` block
(`kind: bulk_count | monomer_count | rna_count | listener_indexed |
unique_count | listener_scalar`). The current set (`bulk_count`,
`listener_path`, `listener_sum`, `xy_correlation`) is too narrow.

### Investigation-level checklist (the "before you run" doc)

The /pbg-study tool should publish a per-investigation checklist
template the expert + author work through together. Draft:

```
Investigation: <name>
For each study in dependency order:

  [ ] verify: all observables resolve, status: spec-validated
  [ ] preview-viz: at least one declared viz renders against baseline
  [ ] expert review: question + hypothesis correspond to the data
        actually rendered
  [ ] calibration check: behavior thresholds within an order of
        magnitude of the v2ecoli baseline; large divergences
        annotated with resolution: {model, thresholds, concept}
  [ ] all expert_questions addressed (or moved to acceptance_criteria)

When every study in the investigation passes all 5 checks, set the
investigation's status to 'ready' and proceed to run.
```

The dashboard renders this as a checklist at the top of each study's
Overview tab + the investigation's detail view. Each checkbox is
backed by tool output (verify result, preview render presence, expert
question count). Manual ticks for "expert review" + "expert questions
addressed" because those are human judgments.

### Summary for the listening Claude

The investigation walkthrough revealed that the **design phase
artifacts are necessary but not sufficient**. Three additional gates
are needed before execution can be trusted:

1. **Spec validation** (mechanical): catch wrong ids, broken links,
   unbacked variants. Tool: `/pbg-study verify`.
2. **Visual preview** (sanity check): render the declared plots against
   real (or initial-state) data so the expert sees shapes before
   waiting for hour-long simulations. Tool: `/pbg-study preview-viz`
   + a generic `TimeSeriesFromObservables` Viz.
3. **Calibration anchoring**: surface model-vs-literature divergences
   AS PART OF the behavioral test, not just pass/fail.

The Generate-report flow + sticky-nav + DAG + dependency framework
all already exist; they're the *display* layer. The *validation*
layer is what's missing. That's the next P0 batch of pbg-superpowers
work.
