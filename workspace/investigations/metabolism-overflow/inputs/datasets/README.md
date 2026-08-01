# Experimental claim datasets

Curated experimental response curves this investigation grades against. Each is
paired with its claim + provenance in `../../investigation.yaml` (`inputs.datasets`)
and cited via `../references/papers.bib`.

## `basan2015_acetate_vs_growth.csv` — POPULATED

Acetate excretion rate vs growth rate from Basan et al. 2015 (Nature 528:99–104),
Fig. 1 source data (supplement `MOESM62` = Extended Data Table 1). Columns:

```
series_type, series, condition, inducer_3MBA_uM, growth_rate_per_h,
acetate_excretion_mM_per_OD600_per_h, symbol_note
```

`series_type` partitions the data onto our condition ladder:
- **`uptake_titration`** — transporter expression titrated (Pu-ptsG glucose / LacY lactose)
  → the direct GUR-sweep analogue. **The overflow study grades against this** (`ptsG_glucose`
  primary). 
- **`uptake_mutant`** — glpK glycerol mutants (part of Basan's uptake-varied set).
- **`carbon_source`** / **`carbon_source_aa`** — WT across carbon sources (± 7 AAs);
  **earmarked for the future carbon-source-swap study**, not graded in the overflow study.

**Provenance + the unit caveat (OD600→gDCW is an external assumption):** see
`../notes/basan2015.md`. Acetate is stored **as reported** (mM/OD600/h) — not pre-converted.

## Future perturbations and paired data bundles

Discrete carbon-source swaps and metabolic gene knockouts will each need a paired
data bundle (a ParCa condition cache / knockout override) alongside their
experimental claim. The dataset entry and the bundle reference are two halves of
one perturbation unit and should be added together so they stay coupled.

## raw/

Local, gitignored staging for source files (PDFs, supplements, figure grabs). See
`raw/README.md`. Only the curated CSV(s) + provenance notes are committed.
