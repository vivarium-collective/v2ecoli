# dnaa-3 — DnaA binding to DnaA boxes

Build a dynamic DnaA-box occupancy mechanism on top of the dnaa-2 V=1e-3 succinate baseline. We are not replacing the initiation mechanism yet.

## Reference documents

- **`Molecular information for simulating replication initiation.pdf`** — biology source. Covers the 11 oriC DnaA boxes (3 high-aff: R1, R2, R4 · 8 low-aff: R5M, τ2, I1, I2, C3, C2, I3, C1), affinity ranges (high ≈ 1 nM · low > 100 nM), nucleotide-form specificity (high-aff bind both DnaA-ATP and DnaA-ADP · low-aff prefer DnaA-ATP), and the dnaA promoter box layout. Key references: Schaper-Messer 1995, Roth-Messer 1998, Hansen 2006, Olivi 2025, Katayama 2017, Speck 1999, Saggioro 2013.
- **`oric_sequence.pdf`** — full 462-bp oriC sequence with all 11 boxes annotated. Use this to extract precise positions for the 8 low-affinity sites.
- **`Parameters for WCM (Stage 1_ heuristic values) - Stage 1.pdf`** — Stage 1 heuristic parameter brief; rows 13-17 cover the DnaA-box parameters.

## Required knowledge / mechanisms

All the DnaA boxes and their affinities.

## Assumptions at the first step

*The following first-step assumptions are provided by Haochen.*

- All **307 consensus DnaA boxes** (TTWTNCACA) are **high-affinity** (K_d ≈ 1 nM)
- oriC has **3 high-affinity + 8 low-affinity boxes**; low-affinity K_d ≈ 100 nM
- DnaA-ATP and DnaA-ADP binding to DnaA boxes is in **fast equilibrium** (only the K_d matters; absolute k_fwd / k_rev magnitudes are degenerate after the steady-state solve)
- Assume all DNA binding **does not alter the DnaA-ATP intrinsic hydrolysis rate** (bf8b82e's k ≈ 0.046 / min applies uniformly to bound and free DnaA-ATP)
- Cooperativity between low-affinity sites is deferred to a later study
- High-affinity pools track bound-ATP and bound-ADP separately so the correct nucleotide form can be released in later studies

## Inputs in place

- **307 high-affinity DnaA box catalog**, computed at ParCa-build time (Schaper-Messer consensus on the K-12 genome). Includes the 3 oriC high-aff sites (R1 at coord −55, R2 at +50, R4 at +124) and the 2 high-affinity dnaA-promoter sites (box 1 at −41917, box 2 at −42883).
- **Static dnaA autorepression** via `fold_changes_nca.tsv` (`dnaA → dnaA = −2.31` log2 FC). The 5 weaker dnaA-promoter boxes are intentionally **not** modeled here — autorepression is already covered.
- **bf8b82e DnaA-ATP intrinsic hydrolysis** (k ≈ 0.046 / min) active in the equilibrium process.

## Inputs to supply

- **8 low-affinity oriC box coordinates** (R5M, τ2, I1, I2, C3, C2, I3, C1). Pull positions from `oric_sequence.pdf` using R1 / R2 / R4 at the known signed coords as anchors.

## What the mechanism does

Add DnaA binding to the 315 boxes according to affinity. Four pools:

- **chromosomal high-affinity** (302 sites) — K_d ≈ 1 nM · binds DnaA-ATP **or** DnaA-ADP
- **oriC high-affinity** (3 sites: R1, R2, R4) — K_d ≈ 1 nM · binds DnaA-ATP **or** DnaA-ADP
- **oriC low-affinity** (8 sites) — K_d ≈ 100 nM · DnaA-ATP **only**
- **dnaA promoter high-affinity** (2 sites: box 1, box 2) — K_d ≈ 1 nM · binds DnaA-ATP **or** DnaA-ADP

Per Haochen's first-step assumptions above, **track bound-ATP and bound-ADP separately** on high-affinity pools so the correct nucleotide form can be released in later studies (RIDA, SeqA, fork-passage disruption). Each high-affinity pool has three state counts: free / bound-ATP / bound-ADP. The low-affinity pool has two: free / bound-ATP.

## Validation

Run the V=1e-3 succinate 6-gen seed=1 burned-in protocol from dnaa-2:

| metric | dnaa-2 baseline | dnaa-3 target |
|---|---|---|
| cycles divided | 6 / 6 | 6 / 6 |
| oriC pattern | 1 ↔ 2 | 1 ↔ 2 (unchanged) |
| re-init events | 0 | 0 |
| DnaA-ATP fraction (gen 3) | ~0.26 | ~0.2-0.5 (Boesen) |

## Visualizations

1. **DnaA-ATP partition across the cell** — concentration (count / cell_mass), three traces:
   - DnaA-ATP bound to high-affinity boxes (sum across chromosomal_high + oriC_high + promoter_high bound-ATP)
   - DnaA-ATP bound to low-affinity boxes (oriC_low bound)
   - DnaA-ATP in free cytoplasm (free bulk pool)

   Sum equals total DnaA-ATP concentration at every tick.

2. **DnaA-ADP partition across the cell** — concentration (count / cell_mass), two traces (no low-affinity bound trace, since low-affinity boxes are DnaA-ATP only):
   - DnaA-ADP bound to high-affinity boxes (sum across chromosomal_high + oriC_high + promoter_high bound-ADP)
   - DnaA-ADP in free cytoplasm (free bulk pool)

   Sum equals total DnaA-ADP concentration.

3. **Total DnaA boxes available over time** — raw count time-series. Total = free + bound across all four pools (315 at gen birth → 630 at replication termination → 315 in each daughter). Plot in raw count units (not concentration) so the replication-driven doubling is visible.

4. **Total DnaA bound concentration** — concentration (count / cell_mass), one trace:
   - sum of (chromosomal_high bound-ATP + chromosomal_high bound-ADP + oriC_high bound-ATP + oriC_high bound-ADP + oriC_low bound-ATP + promoter_high bound-ATP + promoter_high bound-ADP) / cell_mass

   Dominated by the chromosomal_high sink across the cycle.

## Gotchas

- **Promoter weak boxes deliberately excluded** — static autorepression handles dnaA self-regulation. Don't double-count by modelling them here.
- **Cooperativity deliberately deferred** (R4-anchored right-half, R1+IHF-anchored left-half).
- **Don't enable the dormant `dnaa_nucleotide` opt-in feature** — alternative partitioning implementation that conflicts with bf8b82e.
