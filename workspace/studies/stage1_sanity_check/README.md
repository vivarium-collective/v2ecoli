# Stage 1 sanity-check study

Goal: verify v2ecoli's existing chromosome-replication machinery behaves as
expected under the parameter inputs specified in the Stage 1 heuristic brief
(`Parameters for WCM (Stage 1: heuristic values) - Stage 1.pdf`). The
collaborator framing is "**run sanity check simulations so that we know
v2ecoli does what we think it does under the exact conditions we instruct
the simulation**".

This is **not** a test of the proposed DnaA-oriC mechanism — that machinery
(DnaA-ATP binding kinetics, RIDA, DDAH, DARS1/2 reactivation, etc.) doesn't
exist in v2ecoli yet. The Stage 1 mechanistic parameters with no current
consumer are listed in [`../../docs/stage1_parameter_comparison.html`](../../docs/stage1_parameter_comparison.html)
under "Not implemented".

## What this study does

Layers 6 in-memory overrides onto the canonical v2ecoli raw_data before any
ParCa step runs. The canonical flat TSVs are untouched. Then rebuilds the
ParCa cache and runs a 1-seed × 5-generation simulation at the `acetate`
growth condition (136 min doubling — closest to Stage 1's 150 min ABT
minimal glycerol target without plumbing in a new growth condition).

| # | Parameter | Canonical v2ecoli | Stage 1 override | Source | Applied? |
|---|-----------|-------------------|------------------|--------|----------|
| 1 | `d_period` | 20 min | 30 min | parameters.tsv | **NO** (see below) |
| 2 | `replisome_elongation_rate` | 967 nt/s (→ C ≈ 40 min) | **552.58 nt/s** (→ C ≈ 70 min) | parameters.tsv | yes |
| 3 | dnaA translation efficiency | 0.35 (Li 2014) | **1.0** (Hansen & Atlung 2018) | translation_efficiency.tsv | yes |
| 4 | DARS1 window | 813107–813141 (35 bp) | **813086–813186** (100 bp) | dna_sites.tsv | yes (dormant) |
| 5 | DARS2 window | 2969135–2969169 (35 bp) | **2969112–2969367** (255 bp) | dna_sites.tsv | yes (dormant) |
| 6 | `DATA` row | — | **4392732–4392914** | dna_sites.tsv | yes (dormant) |

**Why D=30 is dropped.** We initially applied all six overrides. ParCa Step 6 (the ECOS promoter-binding solver) failed with `Solver could not find optimal value`. Pinpoint diagnostic (`logs/diag1` through `diag7`) showed:

- Each of {d_period, replisome_rate, translation_eff} **alone** passes Step 6.
- The pair `d_period + replisome_rate` (C+D = 100 min globally) **fails** — even without the translation-efficiency bump.
- `d_period + translation_eff` and `replisome_rate + translation_eff` both pass.
- Bumping basal+with_aa to 110 min (to remove their overlap under C+D=100) did **not** rescue the fit. So the failure isn't simply about replication overlap; the global Cooper-Helmstetter DNA-mass shift at every condition makes the cross-condition promoter-binding fit infeasible.

We drop the D=30 override. D stays canonical at 20 min, giving effective C+D = 90 min. At acetate (136 min doubling) the cell cycle is still non-overlapping. The Stage 1 brief's slow-replication requirement (C=70) is preserved; the brief's D=30 requirement is not.

Overrides 4–6 are **dormant data** under the current code (no process reads
DARS or datA coordinates yet). They're applied here so the ParCa cache
already carries them when the DnaA-oriC mechanism is added later — and so
that the loader is exercised against the Stage 1 coordinate set.

## Files

```
studies/stage1_sanity_check/
├── README.md              this file
├── overrides.py           apply(raw) mutates raw_data in memory
└── run.sh                 orchestrator — ParCa rerun → cache → sim → report
```

## How to run

```bash
# 1. Full ParCa rerun (4-8 hours, dominated by Step 5)
v2ecoli-parca --mode full --cpus 8 \
    -o out/sim_data_stage1 \
    --overrides-module studies.stage1_sanity_check.overrides

# 2. Promote to fixture + rebuild cache at acetate condition (~2 min)
gzip -c out/sim_data_stage1/parca_state.pkl \
    > out/sim_data_stage1/parca_state.pkl.gz
python scripts/build_cache.py \
    --fixture out/sim_data_stage1/parca_state.pkl.gz \
    --cache   out/cache_stage1 \
    --condition acetate

# 3. Multi-generation sim (~12 hours at 136-min doubling × 5 gens)
python reports/multigeneration_report.py \
    --generations 5 \
    --seed 0 \
    --cache-dir out/cache_stage1 \
    --max-duration 12000 \
    --out docs/stage1_sanity_check.html
```

Or run all three with `bash studies/stage1_sanity_check/run.sh`.

## Notes

- The Stage 1 brief calls for C = 70 min. The `replisome_elongation_rate`
  value of 552.58 nt/s comes from `4_641_652 nt / (70 min × 60 s × 2 forks)`.
- The Stage 1 brief calls for 150 min doubling. We use **native acetate
  (136 min)** as a stand-in. Documenting this explicitly because it's the
  most-visible departure from the brief.
- Translation efficiency 1.0 is Hansen & Atlung 2018's "1 protein per
  mRNA" textbook simplification. v2ecoli's canonical 0.35 from Li 2014 is a
  ribosome-density-based relative weight (different units of measurement,
  same biological quantity).
- DARS1/2 widenings restore the EcoCyc annotation windows in the
  Stage 1 brief; v2ecoli's narrower defaults come from the
  `extragenic-site` minimal-functional-region annotation.
- DATA is a new row — `id="DATA"` is provisional; EcoCyc canonical
  identifier should be verified before any Aim 2 process consumer
  references it by id.

## What the diagnostic plots will show

The deliverable is `docs/stage1_sanity_check.html` — a parameter-by-parameter
expected-vs-observed report. Each panel: Stage 1 brief value (expected) vs
v2ecoli sim measurement (observed), with a match/differ marker. Groups:

- **Timing**: cell volume traces, C period histogram, D period, mass at
  initiation, # overlapping replication forks per generation.
- **DnaA**: total DnaA over cell cycle, DnaA per oriC, dnaA mRNA count,
  transcription events/gene/min (vs Stage 1's 1.5), translation
  events/mRNA (vs Stage 1's 1), DnaA stability (vs Stage 1's 0 turnover).
- **Replication**: fork count, observed elongation rate (kbp/min vs
  66.31), genome equivalents per cell, oriC count, replication completion
  time (vs C = 70).
- **Loci**: oriC / datA / DARS1 / DARS2 positions confirmed against
  ParCa-loaded coordinates.
