# Li2014Cell — Absolute protein synthesis rates in E. coli

**Li, Burkhardt, Gross, Weissman, 2014, Cell 157(3):624-635.**
[https://www.cell.com/fulltext/S0092-8674(14)00232-3](https://www.cell.com/fulltext/S0092-8674(14)00232-3)

## Why it matters here

Ribosome-profiling-derived absolute synthesis rates. For dnaA, gives
synthesis flux that pairs with the `dnaA_translation_rate` knob in
dnaa-01's `stop-dnaA-synthesis` variant.

## Key numbers (to extract on read)

- DnaA synthesis rate (molecules / cell / min)
- DnaA translation efficiency vs other proteins

## Behaviors / claims supported

- `stop-synthesis-drops-30pct` (dnaa-01) — synthesis rate sets the
  dilution+turnover balance the variant breaks.
- claim: `dnaa.expression.steady-300-800`
