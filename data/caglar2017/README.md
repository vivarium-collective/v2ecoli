# Caglar et al. 2017 — E. coli molecular phenotype under different growth conditions

Citation: Caglar MU, Houser JR, Barnhart CS, et al. *The E. coli molecular phenotype
under different growth conditions.* Scientific Reports 7, 45303 (2017).
DOI: [10.1038/srep45303](https://doi.org/10.1038/srep45303)

Strain: *E. coli* REL606. Experiments vary carbon source (glucose, glycerol,
gluconate, lactate), Na⁺ (5–300 mM), and Mg²⁺ (0.005–400 mM) across exponential,
stationary, and late-stationary phases. Measurements: RNA-seq, mass-spec
proteomics, ¹³C central-carbon fluxes, and exponential-phase doubling times.

## Files in this directory (committed)

| File | Size | Contents |
|---|---|---|
| `MOESM51_ESM.pdf` | 1.5M | Supplementary methods |
| `MOESM52_ESM.csv` | 32K | **Sample metadata** (S1): one row per sample, with `carbonSource`, `Mg_mM`, `Na_mM`, `growthPhase`, `doublingTimeMinutes` + 95% CI |
| `MOESM55_ESM.csv` | 20K | **Flux ratios** (S?): mean ¹³C flux ratios by salt/conc/phase across 13 branch points |
| `MOESM56_ESM.csv` | 3.8K | **Doubling times** (S5): per-replicate exponential doubling time by condition |
| `MOESM57_ESM.csv` | 1.3K | mRNA cophenetic Z-scores (clustering) |
| `MOESM58_ESM.csv` | 1.2K | Protein cophenetic Z-scores (clustering) |
| `MOESM62_ESM.csv` | 238K | Genes changed across phase/condition (mRNA/protein) |
| `MOESM63_ESM.csv` | 166K | KEGG/GO enrichment for the gene lists in MOESM62 |
| `MOESM64_ESM.csv` | 2.4K | Regression of flux ratios against Na⁺/Mg²⁺ concentration |
| `MOESM65_ESM.csv` | 1.6K | Regression of flux ratios against mean doubling time |

## Files not committed (fetch via `raw_large/fetch.sh`)

Large RNA-seq and proteomics abundance matrices are gitignored — they total
~120 MB and are pulled on demand:

| File | Size | Contents |
|---|---|---|
| `MOESM53_ESM.csv` | 11M | Normalized mRNA abundances (long format) |
| `MOESM59_ESM.csv` | 87M | Raw RNA-seq counts (wide: 4196 genes × 152 samples) |
| `MOESM60_ESM.csv` | 5.7M | Normalized protein abundances |
| `MOESM61_ESM.csv` | 17M | Raw protein mass-spec counts |

`srep45303.pdf` (main paper, 4.8 MB) is also in `raw_large/`.

## How this data is used for v2ecoli tuning

- **Doubling times** (MOESM52, MOESM56) are the primary calibration target:
  simulate each condition and compare exponential-phase doubling time.
- **Flux ratios** (MOESM55) validate central-carbon metabolism output
  (`v2ecoli/processes/metabolism.py`) under each condition.
- **RNA/protein abundances** (MOESM53/59–61) back secondary checks on
  expression patterns once growth rate matches.
