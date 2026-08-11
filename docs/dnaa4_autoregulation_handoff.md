# dnaA self-autoregulation via dnaA-promoter occupancy — handoff

**Branch**: `feat/aim2-dnaa-oric-box-binding`
**Commit**: `eb4ea39` (`feat(dnaa-4): dynamic dnaA autoregulation + K_d/k_h refinements`)
**Status**: committed locally on `feat/aim2-dnaa-oric`; pushed to remote
`feat/aim2-dnaa-oric-box-binding` (fast-forward on top of `c648d51`).

Builds on `4fe5cde` (Phase 2 box-binding + bound-pool hydrolysis) and
`c648d51` (Phase 2 plotting).

## What this adds

A live negative-feedback loop from DnaA-promoter occupancy back to the dnaA
TU's transcription rate. Each tick:

1. `dnaa_box_binding.py` computes `f = bound / total` over the dnaA-promoter
   sites (`pool_label == POOL_PROMOTER_HIGH`, dynamic — 2 sites pre-fork,
   4 sites post-fork).
2. `transcript_initiation.py` reads `f` from the `replication_data` listener
   and scales the dnaA TU init prob by `(1 − s · f)`.

```
prob_dnaA  ←  prob_dnaA × (1 − s · f)
```

With `s = 0.6`:
- f = 0 (promoter empty): factor = 1 → no repression
- f = 1 (fully bound): factor = 0.4 → 60% max repression

That maximum is roughly consistent with the −2.31 log2 FC encoded in
`fold_changes_nca.tsv` (which the L1-norm promoter fit drives to zero at
runtime — this step restores it as a dynamic, occupancy-driven feedback).

## Code changes (in commit eb4ea39)

| file | change |
|------|--------|
| `v2ecoli/processes/equilibrium.py` | `_HYDROLYSIS_RATE_PER_SEC = 0.025/60.0` (was 0.046/60.0) |
| `v2ecoli/steps/dnaa_box_binding.py` | `KD_HIGH_M = 3e-9` (was 1e-9); `HYDROLYSIS_RATE_PER_MIN = 0.025` (was 0.046); publishes `promoter_fraction` |
| `v2ecoli/processes/transcript_initiation.py` | New constants `DNAA_TU_IDX=2778`, `AUTOREG_STRENGTH=0.6`; reads promoter fraction from `listeners.replication_data`; scales dnaA init prob |
| `scripts/plot_dnaa_traj.py` | 8-panel layout + `--start-gen` flag |
| `scripts/plot_dnaa3_phase2_schematic.py` | Schematic refreshed for dynamic autoreg loop |

## Configuration

| knob | value | location |
|------|-------|----------|
| `AUTOREG_STRENGTH` (s) | 0.6 | `transcript_initiation.py` |
| `DNAA_TU_IDX` | 2778 (TU00259[c]) | `transcript_initiation.py` |
| `KD_HIGH_M` | 3 nM | `dnaa_box_binding.py` (chromosomal_high, oriC_high, promoter_high) |
| `KD_LOW_M` | 100 nM | `dnaa_box_binding.py` (oriC_low, ATP-only) |
| `HYDROLYSIS_RATE_PER_MIN` (k_h) | 0.025 | `dnaa_box_binding.py` + `equilibrium.py` |

## NOT in commit — applied at cache-build time

Two patches must be applied to the cache for the reference run:

1. **F-05 (apo+ATP kinetic)**: in
   `configs["ecoli-equilibrium"]["fluxesAndMoleculesToSS"]["_data"]["integrate_dt_mask"]`,
   set index 24 to `True`. Makes the apo + ATP ⇌ DnaA-ATP charging reaction
   integrate kinetically each tick rather than fast-equilibrating each step.
2. **Mechanism A V perturbation**: in
   `configs["ecoli-transcript-initiation"]["perturbations"]`,
   set `"TU00259[c]"` to the target V (we used `1.5e-3`). Pins the dnaA TU's
   per-promoter init prob target to V before autoregulation scales it down
   by `(1 − s · f)`.

Both are sim_data patches written into the cache pickle; no source code
change is required to apply them.

## Running

The reference configuration we explored:
- 12 generations on succinate
- Seed = 1
- Resume from the burned-in gen-3 dill at
  `out/steady_state_inputs/succinate_default_gen3_start_dnaa3.dill`
  to skip cold-start transients
- Multi-gen runner: `scripts/run_condition_multigen_parquet.py`

Approximately 70 min wall time for the full 12-gen run.

## Figures

The reference result PDF (`v1.5e-3_s06_F05_seed1_combined.pdf`) is
assembled from two plotting scripts applied to the parquet output of the
reference run:

1. **Multi-generation trajectory** (page 1) — produced by
   `scripts/plot_dnaa_traj.py` over the steady-state window (gens 4–11,
   excluding cold-start transients and the run-tail). Multi-panel layout
   with cell mass, oriC count, DnaA forms (ATP / ADP / apo / total),
   DnaA per cell mass, ATP-fraction band, bulk pools, and dnaA-promoter
   occupancy.
2. **Per-chromosome region snapshots** (page 2) — produced by
   `scripts/plot_dnaa3_region_panels.py` on a chosen steady-state
   generation (we used gen 7). Five time slices across the generation
   showing per-chromosome DnaA-box occupancy by region (chromosomal_high,
   oriC_high, oriC_low, promoter_high) for the parent chromosome and
   each daughter chromosome after fork passage, plus the oriC trajectory
   and bulk DnaA-ATP / DnaA-ADP across the generation.

The two PNGs are concatenated into a single PDF using
`matplotlib.backends.backend_pdf.PdfPages`.

Both scripts take the parquet experiment root and the lineage seed as
arguments and write a PNG. The combined PDF is one of several presentation
artifacts — the underlying data is the parquet output, which can be
re-plotted in any format the dashboard prefers.

## To disable as a control

Set `AUTOREG_STRENGTH = 0.0` at the top of `transcript_initiation.py` and
re-run with the same cache. Gives a no-autoregulation control with the same
V, K_d, and k_h for comparison.
