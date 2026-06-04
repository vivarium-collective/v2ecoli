# Meta-level report cards

A **report card** is an auditable, versioned record of what a model can do,
graded against a behavioral specification of the organism. This document
describes the v1 card in v2ecoli and the design decisions behind it.

> **Status (not yet pinned).** The card pipeline is functional end-to-end
> (measurement → grade → rendered report), but the reference is intentionally
> left `pending_blessed_run`. The **growth axis** requires sustained
> multi-generation division, and on current `main` a lineage divides only in
> generation 0 — later generations grow to the duration cap without dividing
> (a multi-generation execution issue, under investigation; not the ParCa
> cache, which was ruled out with a full-mode rebuild). The composition axis
> is measurable but is not trustworthy until cells divide normally across
> generations. Pin the reference once sustained multi-gen division is restored.

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

## The v1 card: basal-condition phenotype

The first meta card grades two universally-applicable basal-condition axes
over an **ensemble** of cells (N seeds × M generations of the baseline, no
variant), each reported as a cell-to-cell mean / std / CV:

- **Growth** — doubling time over confirmed divisions.
- **Composition** — protein / RNA / DNA mass fractions of dry weight.

Early generations are dropped as burn-in (`generation_lower_bound`) so the
ensemble reflects steady balanced growth rather than the inoculation
transient.

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
| Measurement (multiseed analysis) | `v2ecoli/workflow/analysis.py` → `BasalPhenotypeCard` |
| Stimulus (the canonical ensemble) | `v2ecoli/configs/basal_phenotype_card.json` |
| Grade + reference | `tests/test_basal_phenotype_card.py`, `tests/fixtures/basal_phenotype_reference.json` |

```bash
# 1. Produce a card (needs a ParCa cache):
v2ecoli-workflow --config v2ecoli/configs/basal_phenotype_card.json
#    -> writes out/basal_phenotype_card/analysis.json with the card under
#       results["multiseed"]["basal_phenotype_card"].

# 2. Populate the pinned reference (once, on blessed main):
#    copy growth.doubling_time.mean + composition.*_fraction.mean from that
#    analysis.json into tests/fixtures/basal_phenotype_reference.json,
#    set blessed_model_ref to the git sha, set status: "populated".

# 3. Grade any model against the pin:
V2ECOLI_BASAL_ANALYSIS=out/basal_phenotype_card/analysis.json \
    pytest tests/test_basal_phenotype_card.py -k matches_pinned_reference
```

Until the reference is populated, the grade test **skips** (same pattern as the
other behavior tests skipping without a checkpoint). The measurement-math unit
tests run without a cache.

## Deliberately open (do not resolve in v1)

- **Phase→meta promotion mechanism** — reviewer-gated? structural (asserted
  across ≥N investigations)? population-vote? This is the highest-leverage open
  design question; v1 does not commit to an answer.
- **CI surface shape** — one tier-tagged suite, two surfaces, or one surface
  with a promotion hook.
- **Spec-vs-card separation** — is the config the "spec" and the reference the
  "card", or are they two readings of one versioned object?

## Known v1 limitations

- **RNA == rRna.** The per-cell mass columns pulled by the analysis runner
  include rRna but not tRna/mRna (rRna is ~80–85% of total RNA). Total-RNA/DW
  is a follow-up (extend `analysis_runner._MASS_COLS`).
- **Execution layer.** Producing the reference and grading a candidate both
  require a ParCa cache and an ensemble run; how that runs in CI (nightly /
  cloud / against a cached summary) is intentionally out of scope here.
- **Tolerances are placeholders** pending the blessed run's observed
  cell-to-cell CV.
