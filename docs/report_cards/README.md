# Behavioral report cards

## What is a behavioral test?

A behavioral test drives the model with a known **stimulus**, records its
**behavior**, and compares that behavior to an **expectation**. It grades the
model at the level of its *observable behavior* — phenotypes, dynamics, emergent
population properties — rather than its implementation (unit tests on code
paths). Stimulus, recorded behavior, and expectation are the three moving parts;
different choices of each yield different tests.

Because the same stimulus should produce the same behavior, this one shape is a
**unifying abstraction across software and science**, at a shared behavioral
abstraction layer:

- as **software** — development and regression testing: the stimulus is a code
  change, the expectation is behavioral parity or a sanctioned delta;
- as **science** — meta-evaluation of the model's performance across a range of
  conditions and perturbations: the stimulus is an environmental or genetic
  perturbation, the expectation is experimental data, literature, or a prior
  model version.

Same shape, different stimulus / expectation / timescale. A **report card** is
the auditable, comparable, regression-tracked artifact a behavioral test
produces — the model's graded behavior under a defined stimulus.

> **Under development.** The ontology of behavioral tests and the patterns for
> applying them are still being worked out. This directory is *one working
> pattern* for implementing them in v2ecoli; we anticipate a growing **"zoo"** of
> behavioral tests, built and composed for different purposes, that will refine
> both the ontology and these conventions.

## This directory

The index + catalog for v2ecoli's behavioral report cards: what each card is and
**where across the repo the pieces of each card live** (config, analysis,
reference, test, output). See [`../meta_report_cards.md`](../meta_report_cards.md)
for more of the conceptual spec; this file is the map.

> Snapshot as of `f3867c2`. Git log is the source of truth for per-file changes;
> the "updated" notes below are a convenience and may lag.

## Key concepts (brief)

- **Three objects, often conflated.** (1) a *behavioral spec* — the
  implementation-agnostic "is this an E. coli" protocol; (2) a *report card* —
  one model's grades against the spec; (3) the *CI/operational substrate* that
  runs the suite and gates merges. This directory is mostly about (2), with
  hooks toward (3).

- **A test station = `{stimulus config × analysis × reference × test ×
  rendered output}`.** No single directory holds a whole station — each *kind*
  of artifact lives in its conventional home (configs in `configs/`, code in
  `library/`+`workflow/`, tests in `tests/`, output in `docs/report_cards/`).
  This README is the hub that ties the spokes together.

- **Two orthogonal axes: card kind × reference mode.** A rendered card is one
  point in a grid of *what is measured* × *what it is graded against*.
  - **Card kind** (what is measured):
    - *Single-cell invariants* — binary: does the machinery work (cell grows,
      divides, conserves mass, daughters viable)? Run from a single-cell
      checkpoint, in seconds.
    - *Population phenotypes* — quantitative: emergent behaviors measured across a
      large ensemble (seeds × generations), graded vs a reference within tolerance.
  - **Reference mode** (what it is graded against — the *same* card, a different
    reference source):
    - *self-pin (drift)* — graded against this model's own blessed ensemble;
      catches drift over time ("has v2 changed from its pinned self?").
    - *equivalence vs a reference model* — graded against another model's
      ensemble; e.g. **v1↔v2** (vs vEcoli): "is v2 still the same E. coli as v1?"
      (future: literature targets, vs-PDMP).
  - **Equivalence is a reference mode, not a card** — any card can be rendered in
    either mode. Rendered outputs live as subdirs of the card:
    `docs/report_cards/<card>/report_card.*` (self-pin) and
    `docs/report_cards/<card>/vs_vecoli/report_card.*` (v1↔v2 equivalence).

- **Composed for purpose.** These checks are building blocks; how they're
  combined depends on the use:
  - *Within a dev investigation / PR* — run the relevant checks to gate a change
    before merge. A common, practical composition runs the fast single-cell mechanics
    checks before committing compute to a full ensemble, but the ordering is a
    convenience, not a fixed hierarchy.
  - *Longitudinally across the project* — track drift over time and show
    behavioral equivalence across model versions, by re-grading the same cards
    as the model evolves.

- **Reference-driven, one grader, many references.** One grader + one renderer
  serve every card; cards differ only by their *reference source* (e.g. pinned
  current-model "is it E. coli" vs a v1↔v2 equivalence reference). Each axis
  carries a typed `criterion` (`rel_tol` / `ttest` / `r2` / `flux_scatter` /
  `boolean`) earning a 4-state verdict (`within_tol` / `drift` / `mismatch` /
  `ungraded`).

- **Cell-level aggregation.** Population stats aggregate per cell (time-average
  within a cell → one value/vector per cell) then across the N cells — never raw
  statistics over timepoints.

## Shared machinery (used by every card)

| Piece | Path |
|-------|------|
| Typed criteria (verdict logic) | `v2ecoli/library/card_criteria.py` |
| Inline-SVG plots (violin / scatter / flux / loglog) | `v2ecoli/library/card_plots.py` |
| Vector extraction (omics / flux, from parquet) | `v2ecoli/library/card_vectors.py` |
| Grader + Markdown/HTML renderer | `v2ecoli/library/report_card.py` |
| Per-cell record builder | `v2ecoli/workflow/analysis_runner.py` |

## Catalog of stations

Rows are **cards** (what is measured). Each can be rendered in multiple
**reference modes** (the second table).

| Station | Kind | Stimulus | Analysis | Reference | Test | Output | Updated |
|---------|------|----------|----------|-----------|------|--------|---------|
| Single-cell mechanics | Single-cell invariants | pre-division checkpoint (single cell) | `tests/test_model_behavior.py` + `reports/single_cell_mechanics_report.py` | `tests/fixtures/single_cell_mechanics_reference.json` | `tests/test_model_behavior.py` | `docs/report_cards/single_cell_mechanics/` | `0c1ee93` |
| Basal-condition phenotype | Population phenotypes | `configs/population_phenotype_basal.json` | `PopulationPhenotypeBasalCard` | `tests/fixtures/population_phenotype_basal_reference.json` | `tests/test_population_phenotype_basal.py` | `docs/report_cards/population_phenotype_basal/` | `f3867c2` |

**Reference modes** (the second axis — same cards, different reference source):

| Mode | Reference source | Rendered under | Status |
|------|------------------|----------------|--------|
| self-pin (drift) | the model's own blessed ensemble | `<card>/report_card.*` | live |
| v1↔v2 equivalence (vs vEcoli) | a vEcoli ("v1") ensemble | `<card>/vs_vecoli/` | **live** — full 21-axis basal phenotype graded at matched 8×16; single-cell mechanics planned |

### Single-cell mechanics

Binary invariants of a single cell — does the machinery work (grows, divides,
conserves mass, daughters viable)? Run from a saved pre-division checkpoint
(seconds, not minutes), suitable to block merges.

- **Stimulus**: pre-division checkpoint `tests/fixtures/pre_division_state.json.gz`
  + single-cell trajectory (`out/workflow/single_cell_meta.json`, via
  `reports/workflow_report.py`); base config `v2ecoli/configs/default.json`.
- **Fixtures**: `tests/conftest.py` (`predivision_state`, `single_cell_trajectory`,
  `sim_data_cache`).
- **Assertions** (`tests/test_model_behavior.py`, marker `behavior`): mass
  roughly doubles · replication completes in window · division splits +
  conserves bulk/chromosomes · daughters viable + grow.
- **Acceptance**: absolute biological windows baked in the test — these encode
  biology directly (not a pinned number), so they can compose with the
  population cards as a fast pre-merge check or a stable cross-version baseline.
- **Card** (`reports/single_cell_mechanics_report.py`): renders the checks as a
  **boolean** report card via the *shared* grader/renderer
  (`tests/fixtures/single_cell_mechanics_reference.json` declares one boolean axis per
  check). It's **pytest-as-evidence** — runs the `behavior` suite, maps each
  test's pass/fail/skip onto its axis; a skipped check (missing checkpoint /
  trajectory) renders `ungraded`. Output:
  `docs/report_cards/single_cell_mechanics/report_card.{html,md}`. The measurement stays
  procedural + single-cell (in pytest); only the *result* flows through the
  shared card — the abstraction shares the output layer, not the measurement.

### Basal-condition phenotype

**Population phenotypes** — emergent behaviors of the model measured across a
large ensemble of simulations (seeds × generations), not properties of any one
cell. Each axis is a cell-level statistic (time-averaged within a cell) then
aggregated across the whole population, and graded vs a pinned reference within
tolerance. 21 axes across 5 groups: Physiology · Composition · Ribosomes ·
Exchange fluxes · Gene expression.

- **Stimulus**: `v2ecoli/configs/population_phenotype_basal.json` (4 seeds × 8
  generations, burn-in 3) — the seeds × generations are the population the
  phenotypes are computed over.
- **Measurement**: `v2ecoli/workflow/analysis.py::PopulationPhenotypeBasalCard` +
  `analysis_runner.build_cell_records`. Regenerate over an existing sweep with
  `v2ecoli-analyze <sweep> --config configs/population_phenotype_basal.json`
  (no re-sim).
- **Reference**: `tests/fixtures/population_phenotype_basal_reference.json` (typed criteria
  + baked ref values/vectors). Re-pin with
  `scripts/pin_population_phenotype_basal_reference.py` from a blessed ensemble + its
  sim_data.
- **Grade test**: `tests/test_population_phenotype_basal.py` (measurement math +
  criterion dispatch + the meta-tier grade vs the pin).
- **Render**: `reports/population_phenotype_basal_report.py` →
  `docs/report_cards/population_phenotype_basal/report_card.{html,md}` (+
  `report_card_DRIFT_DEMO.html`, a committed "what a regression looks like").
- **Provenance**: reference pinned to a full-ParCa blessed ensemble (model ref
  in the reference's `stimulus.blessed_model_ref`); card infrastructure updated
  at `f3867c2`.

### v1↔v2 equivalence (reference mode — live, full 21 axes)

Not a separate card — a **reference mode**: the *same* card graded against a
vEcoli ("v1") ensemble instead of v2's self-pin. Same grader + renderer; only the
reference's `ref_values` change ("does v2 still behave like v1?").

- **v1 ensemble**: produced by vEcoli's own Nextflow workflow (it emits the same
  hive-partitioned `**/history/**/*.pq` schema this card reads). Generated in a
  dedicated clean checkout (`SMS/vecoli-benchmarking`, vEcoli `master`); the v1
  commit is stamped in the reference's `stimulus.blessed_model_ref`.
- **Reference pin**: `scripts/pin_vecoli_equivalence_reference.py` — reuses the
  self-pin reference as the presentation/criterion *template* and swaps in v1's
  per-cell distributions. It carries a **self-contained cross-implementation
  reader** for the two v1↔v2 schema differences (vEcoli emits `time`, cumulative
  across the lineage, not `global_time`; and positional bulk, not
  `bulk__id`/`bulk__count`) so the **shared `analysis_runner` stays untouched**.
- **Scope**: all 21 axes (Physiology · Composition · Ribosomes · Exchange fluxes ·
  Gene expression), graded at matched **8×16**. Ribosomes read v1's positional
  bulk via an index adapter (IDs identical, only the access pattern differs);
  fluxes/omics align positionally (shared reconstruction → same cistron/monomer/
  flux ordering). All adapters live in the pin script, so the shared
  `analysis_runner` stays untouched.
- **Headline result**: Physiology / Composition / Ribosomes equivalent within
  tolerance; **metabolism (exchange fluxes) is the divergence** (O₂/CO₂
  respiration deficit; near-floor exchanges that flip on/off are shown but held
  below a significance floor so FBA jitter doesn't drive the verdict); gene
  expression is highly correlated (R² ≈ 0.93–0.94) but below the strict self-pin
  R² band. See the rendered card for the full verdict table.
- **Outlier-gene tables**: each gene-expression axis carries a companion table
  of the genes that disagree most between the two models, by log2 fold-change of
  ensemble-mean counts (over-/under-expressed in the measured model), gated by a
  min-count floor and labelled with gene symbol + descriptive name. On the v1↔v2
  card these surface coherent, interpretable divergences (e.g. the threonine
  operon and cytochrome bd-I up in v2; several transcriptional regulators and
  two-component sensors down) — turning the single R² into an actionable list.
- **Render**: `reports/population_phenotype_basal_report.py --reference
  <card>/vs_vecoli/vecoli_reference.json --no-vectors` →
  `docs/report_cards/population_phenotype_basal/vs_vecoli/report_card.{html,md}`.
- **Legacy**: the single-trajectory visual comparison `reports/v1_v2_report.py` →
  `docs/v1_v2_comparison.html` predates this; the card lifts it to an
  ensemble/typed-criteria/graded instrument.

## Adding a station

1. **Stimulus** → a config in `v2ecoli/configs/`.
2. **Analysis** → an `AnalysisStep` in `v2ecoli/workflow/analysis.py` emitting
   grouped axis nodes (`group.axis = {values, mean, std, cv, n}` or `{vector}`).
3. **Reference** → a pinned fixture in `tests/fixtures/`, ideally regenerated by
   a committed pin script in `scripts/`.
4. **Test** → a grade test in `tests/` (marker `behavior` for the merge-gating
   suite).
5. **Render** → a CLI in `reports/` writing to `docs/report_cards/<station>/`.
6. **Register it here** (catalog row + a detail subsection).
