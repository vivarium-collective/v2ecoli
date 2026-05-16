# Hansen1999JBacteriol — DnaA protein stability

**Hansen et al., 1999, J Bacteriol 181(18):5557-5562.**
[https://journals.asm.org/doi/full/10.1128/jb.181.18.5557-5562.1999](https://journals.asm.org/doi/full/10.1128/jb.181.18.5557-5562.1999)

## Why it matters here

DnaA turnover rate / half-life. Sets the dilution-only-vs-degradation
expectation for dnaa-01's `stop-synthesis-drops-30pct` test: if DnaA is
intrinsically stable, the drop is purely dilution-driven (doubling time
~40-60 min); if there's appreciable degradation, the drop is faster.

## Key numbers (to extract on read)

- DnaA half-life (in growing vs non-growing cells)
- Degradation rate constant

## Behaviors / claims supported

- claim: `dnaa.protein.stability`
- dnaa-01 `stop-synthesis-drops-30pct`, `stop-synthesis-monotonic-decrease`
