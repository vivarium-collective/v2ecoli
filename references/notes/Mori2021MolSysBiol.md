# Mori2021MolSysBiol — Absolute E. coli proteome under diverse growth conditions

**Mori et al., 2021, Mol Syst Biol 17(5):e9536.**
[https://link.springer.com/article/10.15252/msb.20209536](https://link.springer.com/article/10.15252/msb.20209536)

## Why it matters here

Mass-spec proteome atlas across growth-rate / nutrient conditions. Provides
a check that DnaA scales (or doesn't) with growth rate — relevant for the
`fast-growth-earlier-replication` test in dnaa-04 and the growth-rate sweep
intervention.

## Key numbers (to extract on read)

- DnaA copies vs growth rate
- Other replication-initiation proteins (DiaA, SeqA, IHF, Hda) per cell

## Behaviors / claims supported

- claim: `dnaa.expression.steady-300-800`
- dnaa-04 `fast-growth-earlier-replication` (background)
