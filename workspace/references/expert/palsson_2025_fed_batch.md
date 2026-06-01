# Palsson 2025 — Beulig et al., high-cell-density fed-batch *E. coli* trade-off study

> *Internally we colloquially call this "Palsson 2025" (Palsson lab, senior
> author). The bib entry uses the canonical first-author key
> `Beulig2025mSystems`.*

## Citation

Beulig F, Bafna-Rührer J, Jensen PE, Kim SH, Patel A, Kandasamy V, Steffen C,
Decker K, Zielinski D, Yang L, Özdemir E, Sudarsan S, **Palsson B**.
**Trade-off between resistance and persistence in high cell density cultures.**
*mSystems* **10**(7), 2025.
DOI: [10.1128/msystems.00323-25](https://doi.org/10.1128/msystems.00323-25).
Open access (CC BY 4.0). Published 13 June 2025.

PDF: [`references/papers/Beulig_2025_mSystems_high_cell_density.pdf`](../papers/Beulig_2025_mSystems_high_cell_density.pdf)

## Why this is the mbp-05 benchmark

The paper is the first systems-level characterization of *E. coli*
high-cell-density physiology under industrially relevant fed-batch
conditions, with publicly accessible (open access) operating profiles,
biomass trajectories, AND transcriptomic state — making it an unusually
rich benchmark for the v2ecoli ↔ pbg-bioreactordesign coupled model: per-
study mbp-05 can compare not just OD / biomass / acetate trajectories, but
also (in a future extension) transcriptional states.

## Operating profile (extract for mbp-05 spec PR)

- **Strategy.** Two-stage batch → fed-batch fermentation in parallel
  well-controlled bioreactors. Exponential feeding strategy after the
  batch-to-fed-batch transition.
- **Density.** 50–80 g_cell_dry_weight / L — very high cell density.
- **Strains (5 groups).**
  - **WT** — *E. coli* BW25113 (wild type, control)
  - **SGKO** — 31 single-gene knockout strains
  - **TRP** — tryptophan production strain
  - **TRPp** — empty plasmid control for TRP
  - **MEL** — plasmid-carrying melatonin production strain
- **Phases (I–IV).**
  - I/II at 0 h: batch → fed-batch transition
  - II/III at 10 h: changes in melatonin and tryptophan formation
  - III/IV at 20 h: growth arrest of strain MEL
- **Observables.**
  - Biomass / OD600 over time
  - O2, pH buffering, exponential feed (continuous media + O2 + base supply)
  - >470 transcriptomic samples (per-strain, per-phase)
  - Stress-related gene expression patterns (resistance / persistence /
    maintenance stimulons)

## Key findings relevant to v2ecoli benchmarking

1. **Maintenance requirements distinguish growth into high cell density** —
   metabolic burden re-allocates resources from resistance functions toward
   increased maintenance. v2ecoli's metabolism Process should reproduce
   declining μ at high density.
2. **Trade-off between resistance and persistence** — at high density,
   metabolic burden modulates growth-versus-survival transitions. v2ecoli
   reproduces a metabolic state; transcriptional persistence states (bistable
   expression) are a stretch target.
3. **Engineered strains (TRP, MEL) show growth arrest** — strain-specific
   bioreactor performance. v2ecoli's plasmid / heterologous expression
   handling is the natural comparison point for these strains; out of scope
   for mbp-05's WT comparison but candidate-future-study.

## Scope for mbp-05

**In:** WT (BW25113) batch + early fed-batch trajectories — OD / biomass /
acetate / pH / DO — used as the quantitative benchmark for the v2ecoli ↔
pbg-bioreactordesign coupled composite.

**Deferred:** SGKO / TRP / MEL strain-specific comparisons; transcriptional
state matching (>470 transcriptomes); detailed maintenance-burden modeling.

**Blocker:** `pbg-bioreactordesign` has no fed-batch operations yet. Until
that lands upstream (or is added in v2ecoli), mbp-05 scopes to the BATCH
phase prefix (0–~6 h before exponential feed kicks in). The spec PR makes
the scope call explicit.

## Bib key

`Beulig2025mSystems` (canonical first-author convention).
Studies that cite this paper use this key in `behavior_tests[].cites` and
`bibliography.bib_keys`.

## Open questions to feed into the mbp-05 spec PR

- [ ] OD600 ↔ gDW conversion factor used by the paper (figure / methods detail)
- [ ] Initial conditions (medium composition, initial OD, initial volume)
- [ ] Exact exponential-feed profile (F0, μ_set, feed glucose concentration)
- [ ] Aeration / kLa / O2-supplementation profile
- [ ] Are the trajectory CSVs (or equivalent supplementary data) downloadable?
  If yes, into `references/papers/palsson-2025-supp/`
