# Meta-level report cards

A **report card** is an auditable, versioned record of what a model can do,
graded against a behavioral specification of the organism. This document is the
design rationale for the basal-condition card; see
[`report_cards/README.md`](report_cards/README.md) for the cross-card index and
the definition of a behavioral test.

> **Status: pinned.** The basal-condition card grades **21 axes across 5 groups**
> (Physiology · Composition · Ribosomes · Exchange fluxes · Gene expression)
> against a blessed 4×8 ensemble (full-mode ParCa cache; model ref in the
> reference's `stimulus.blessed_model_ref`) — see
> [`report_cards/basal_phenotype/`](report_cards/basal_phenotype/report_card.md).
> All axes grade **PASS** against the pin. Two fixes were required to get here:
> sustained multi-generation division needs `#127` (MarkDPeriod divide-flag
> detection for gen ≥ 1), and a **full-mode** ParCa cache (the shipped fast-mode
> fixture leaves metabolite concentrations unpopulated and degrades FBA). The
> pin therefore requires a full ParCa run to reproduce — recorded in the
> reference's `cache_provenance`.

## Three objects, often conflated

1. **Behavioral spec** — an implementation-agnostic specification of *E. coli
   the organism*. Tests are written against the organism, not against any one
   model. (Analogy: an HTTP spec declares "if you pass these tests you are an
   HTTP server" without caring how you are implemented.)
2. **Report card** — a given model's grades against the spec, persisted next
   to the code that produced it. Each model extension must raise a new grade
   **and** not drop any prior grade — grades only move up.
3. **CI / operational substrate** — what runs the suite, persists the cards,
   and enforces the cumulative-only discipline.

This document is about object (2). Objects (1) and (3) are referenced where
they touch the v1 card, but their full design is deliberately left open.

## Two tiers within the report card

- **Meta** — the durable, cross-investigation transcript. A failure blocks any
  merge; grades persist across investigations. *Meta is a durability tier, not
  "fundamental physiology"* — basal growth + composition is the cleanest first
  meta test because it is universally applicable, not because basal == meta.
- **Phase** — within-investigation acceptance tests, wired to specific
  implementation work; may be retired when the investigation merges.

### How the existing test suite already sorts into tiers

Triaging the current behavior/regression/parity tests against this framing
produces **three** buckets, and only one is meta-report-card material — useful
evidence that the tiering is a real distinction, not just vocabulary:

| Bucket | What it grades | Examples | Tier |
|---|---|---|---|
| **A. Organism behavior** | "is it being E. coli?" — phenotype, survives a substrate reshape | `test_model_behavior.py` (growth doubles / replication / division / daughters), `test_sustained_growth.py` | **meta** |
| **B. Parity / regression gates** | "matches the reference implementation / code still works" — implementation-specific by design | `test_growth_parity.py` (submasses vs a golden run), `test_parca_alignment_vs_vecoli.py`, emitter parity | CI substrate (object 3), not a card grade |
| **C. Machinery invariants** | the stochastic simulator, not the organism | `test_seed_determinism.py`, `test_seed_diversity.py` | phase / infra |

The sharpest illustration: a parity gate asserts submasses match a *reference
implementation* (bucket B — it should, and would, break under a major
substrate reshape); the composition **meta** grade below asserts submass
*fractions* match a *pinned reference value* (bucket A — it must survive the
reshape). Same mass data, opposite tier.

## The card: basal-condition phenotype

The basal-condition card grades **population phenotypes** — emergent behaviors
measured across an **ensemble** of cells (N seeds × M generations of the
baseline, no variant). Each axis is a cell-level statistic (time-averaged within
a cell) aggregated across the population, graded against a pinned reference with
a typed criterion and a 4-state verdict (within_tol / drift / mismatch /
ungraded). 21 axes across 5 groups:

- **Physiology** (cell cycle) — doubling time, cell mass, cell volume, oriC,
  replication initiation + completion.
- **Composition** — protein / total-RNA / DNA mass fractions of dry weight.
- **Ribosomes** — total (active + free subunits), active fraction, elongation
  rate, production (rRNA-initiation proxy).
- **Exchange fluxes** — 87-flux fingerprint (scatter + appeared/disappeared
  flag) + glucose / O₂ / NH₄ / CO₂ / acetate KPIs.
- **Gene expression** — transcriptome + proteome log-log R² vs blessed vectors.

Early generations are dropped as burn-in (`generation_lower_bound`) so the
ensemble reflects steady balanced growth rather than the inoculation transient.
Criteria are typed per axis (`rel_tol` / `ttest` / `r2` / `flux_scatter` /
`boolean`); the shared grader + renderer are described in
[`report_cards/README.md`](report_cards/README.md).

### v1 reference = a pinned current-model ensemble (not a literature value)

The v1 card pins its reference to a **blessed ensemble run of current `main`**
— no biological judgement yet. This makes the card a *drift instrument*
immediately: a model passes if its basal ensemble reproduces the pinned values
within tolerance. The question "is the pinned number itself biologically
right?" is deferred to a later pass that reads drift against the pin. (The
literature-grounded version — growth-rate-dependent composition references — is
a natural follow-up once the instrument exists.)

This mirrors the reference-ensemble pattern used elsewhere in the workspace:
characterize a reference, pin it, and accept later work only if it stays
indistinguishable within tolerance.

## Artifacts and how to run

| Artifact | File |
|---|---|
| Measurement (multiseed analysis) | `v2ecoli/workflow/analysis.py` → `BasalPhenotypeCard` (+ `analysis_runner.build_cell_records`) |
| Grade + render machinery | `v2ecoli/library/{card_criteria,card_plots,card_vectors,report_card}.py` |
| Stimulus (the canonical ensemble) | `v2ecoli/configs/basal_phenotype_card.json` |
| Reference (typed criteria) | `tests/fixtures/basal_phenotype_reference.json` |
| Re-pin script | `scripts/pin_basal_phenotype_reference.py` |
| Grade test | `tests/test_basal_phenotype_card.py` |
| Render CLI | `reports/basal_phenotype_card_report.py` |

```bash
# 1. Produce a sweep (needs a ParCa cache), then run the analysis over it
#    (re-runnable with no re-sim):
v2ecoli-workflow --config v2ecoli/configs/basal_phenotype_card.json
v2ecoli-analyze out/basal_phenotype_card --config v2ecoli/configs/basal_phenotype_card.json
#    -> writes out/basal_phenotype_card/analysis.json (card under
#       results["multiseed"]["basal_phenotype_card"]).

# 2. (Re)pin the reference from a blessed ensemble + its sim_data (bakes the
#    typed-criteria ref values/vectors + the exchange-flux id order):
python scripts/pin_basal_phenotype_reference.py \
    --sweep-dir out/basal_phenotype_card \
    --sim-data out/sim_data_full/parca_state.pkl --gen-lb 3

# 3. Render the card (md + html; reads omics/flux vectors from the sweep):
python reports/basal_phenotype_card_report.py \
    --analysis out/basal_phenotype_card/analysis.json

# 4. Grade any model against the pin (grade_card vs the typed-criteria reference):
V2ECOLI_BASAL_ANALYSIS=out/basal_phenotype_card/analysis.json \
    pytest tests/test_basal_phenotype_card.py -k matches_pinned_reference
```

Until the reference is `status: populated`, the grade test **skips** (same
pattern as the other behavior tests skipping without a checkpoint). The
measurement-math + criterion unit tests run without a cache.

## Deliberately open (do not resolve in v1)

- **Phase→meta promotion mechanism** — reviewer-gated? structural (asserted
  across ≥N investigations)? population-vote? This is the highest-leverage open
  design question; v1 does not commit to an answer.
- **CI surface shape** — one tier-tagged suite, two surfaces, or one surface
  with a promotion hook.
- **Spec-vs-card separation** — is the config the "spec" and the reference the
  "card", or are they two readings of one versioned object?

## Known limitations

- **Execution layer.** Producing the reference and grading a candidate both
  require a ParCa cache and an ensemble run; how that runs in CI (nightly /
  cloud / against a cached summary) is intentionally out of scope here.
- **Vector extraction at render time.** The omics + exchange-flux axes read the
  sweep parquet directly (~a minute over a 4×8 ensemble); they render
  `ungraded` if skipped (`--no-vectors`).
- **Replication event semantics.** Under overlapping rounds (oriC > 1), the
  per-cell replication initiation and completion times track *different* rounds,
  so their population means do not pair as one round's start/end (flagged in the
  axis descriptions).
