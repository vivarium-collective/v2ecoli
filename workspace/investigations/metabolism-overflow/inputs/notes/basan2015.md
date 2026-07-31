# Basan 2015 — acetate overflow vs growth rate (provenance + extraction notes)

**Source:** Basan, Hui, Okano, Zhang, Shen, Williamson, Hwa. *Overflow metabolism in
Escherichia coli results from efficient proteome allocation.* Nature 528:99–104 (2015).
DOI [10.1038/nature15765](https://doi.org/10.1038/nature15765). Bib key `basan2015`.

**Extracted data:** `../datasets/basan2015_acetate_vs_growth.csv` — the Fig. 1 source data
(supplement `MOESM62`, = "Extended Data Table 1"). The full PDF + supplement live (gitignored)
in `../datasets/raw/basan_2015/`; `basan2015_fulltext.txt` there is the pdftotext extract.

## What the data is

Acetate excretion rate `Jac` vs growth rate `λ`, for *E. coli* NCM3722, across four series
(`series_type` column). Acetate is reported **per biomass as `mM/OD600/h`** (i.e. mM acetate
accumulated per hour per unit OD600) — the verbatim header is *"Acetate excretion rate Jac
(mM/OD600/h)"*.

- **`uptake_titration`** — the direct GUR analogue. Carbon uptake is set by **titrating the
  transporter's expression**, not by changing the carbon source:
  - `ptsG_glucose` (strain **NQ1243**, **Pu-ptsG**): glucose uptake titrated by the inducer
    **3MBA (3-methylbenzyl alcohol)** at 0/20/300/800 µM → λ 0.58→0.95, Jac 0→2.06.
  - `lacY_lactose` (strain **NQ381**, titratable LacY): lactose uptake titrated by 3MBA → λ
    0.35→0.92, Jac ~0 until λ≈0.8 then →2.30.
  Single carbon source, uptake dialed up → growth rate emerges → acetate measured. **This is
  the series the overflow study grades against** (`ptsG_glucose` is the apples-to-apples match
  for a glucose-cache model GUR sweep).
- **`uptake_mutant`** — glpK mutants (NQ636/638/640) on glycerol; faster-growing glycerol
  variants that cross into overflow. Part of Basan's "titratable or mutant uptake" (purple) set.
- **`carbon_source`** — WT NCM3722 on 13 carbon sources at their natural growth rates. Earmarked
  for the **future carbon-source-swap study**, not graded in the overflow study.
- **`carbon_source_aa`** — WT on carbon sources + 7 non-degradable amino acids (richer media,
  higher λ/acetate). A future condition.

## The claim (verbatim, for the band citation)

> "For strains with titratable carbon uptake systems … the same linear dependence is seen for
> acetate excretion … These results suggest that **acetate overflow is an innate response that
> depends on the degree of carbon influx and not specifically on the nature of carbon sources.**"
> (Basan 2015, main text)

Operationally: **acetate is ~0 below a threshold growth rate `λac` (≈0.7–0.8 h⁻¹ here) and rises
~linearly with λ above it** (Basan eq. 1, the "acetate line"). The `curve_response` criterion
grades the model's acetate-vs-growth response on the **onset (λac)** + the **slope** above it.

## ⚠️ Unit reconciliation (build-time assumption — NOT from this paper)

`Jac` is reported in **mM/OD600/h**; the v2ecoli observable is the acetate exchange flux in
**mmol/gDCW/h**. The OD600→gDCW conversion (gDCW · L⁻¹ · OD600⁻¹) is **not stated in Basan
2015** (it reports everything per-OD). Two faithful options at grade time, to decide at build:
1. Convert the experimental `Jac` to mmol/gDCW/h with an **externally-sourced** NCM3722 factor
   (companion Hwa-lab papers, e.g. You 2013 / Hui 2015; typical ~0.4–0.5 gDCW·L⁻¹·OD600⁻¹) —
   flagged as an external assumption with its own citation.
2. Or convert the **model** side to per-OD and grade in the reported units (avoids importing a
   factor). Either way, **do not bake an unsourced factor into the CSV** — values here are
   as-reported.

## Caveats

- `acetate_excretion ≤ 0` = no detectable excretion (measurement noise around zero); below `λac`.
- The titration sets transporter *expression* → uptake *capacity*; growth rate is emergent
  (contrast: a chemostat clamps growth rate and lets uptake follow — the inverse causal
  direction). The model GUR sweep (clamp uptake bound → growth emerges) mirrors the titration.
