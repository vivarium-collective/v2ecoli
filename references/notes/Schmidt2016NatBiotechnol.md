# Schmidt2016NatBiotechnol — Quantitative absolute proteome of E. coli across conditions

**Schmidt et al., 2016, Nature Biotechnology 34(1):104-110.**
[https://www.nature.com/articles/nbt.3418](https://www.nature.com/articles/nbt.3418)

## Why it matters here

Provides absolute, mass-spec-derived per-cell copy numbers for ~2,000 E. coli
proteins across 22 different growth conditions. DnaA copy numbers in steady-
state growth are the experimental anchor for dnaa-01's
`dnaa-count-in-mass-spec-range` test (300-800 per cell).

## Key numbers (to extract on read)

- DnaA monomer count under each condition (M9, LB, ...)
- Comparison with Sekimizu1991JBacteriol's earlier western-blot estimate

## Behaviors / claims supported

- `dnaa-count-in-mass-spec-range` (dnaa-01)
- claim: `dnaa.expression.steady-300-800`
