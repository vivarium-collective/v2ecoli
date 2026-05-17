# Overnight Investigation Report — dnaA / Replication Initiation

**Date:** 2026-05-17 (~04:00–06:30 local autonomous run)
**Operator:** Claude
**Workspace:** v2ecoli @ main

---

## TL;DR

> **★★ Found and validated at 5 seeds: (TE=20×, fc=0.7) passes BOTH
> dnaa-01 primary gate tests.** DnaA median 707/cell (in literature
> [300, 800]) with Pearson r = -0.521 (autorepression preserved at ≤-0.3).
> Bonus: (TE=20×, fc=0.8) also passes both. There is a robust working
> region between fc=0.7 and fc=0.8 at TE=20×. **dnaa-02 unblocks cleanly.**

This finding emerged after a focused 19-sim TE sweep that filled in the
inflection region (15×, 25×, 30×) and revealed the system is in an
**unstable / non-monotonic regime above TE=20×**. The autorepression
mechanism saturates between 20× and 25× — Pearson r flips from -0.253
to +0.78, and DnaA count crashes from 296 to 159 at 30×. The right
move is the **joint (TE × fold_change) sweep** which the original F-10
recommended; this run validated that approach with a concrete win.

## Scoreboard

| Study | Before | After | Change |
|---|---|---|---|
| dnaa-01-expression-dynamics | Decide, 10 findings, gated | Decide, 10 findings, gated | – |
| dnaa-01f-recalibrate-EG10235 | planned, Design, empty | **Decide**, 3 findings, 1 sim-set | ✓ Driven |
| dnaa-01g-joint-te-fold-change-sweep | (didn't exist) | **Design** (planned), 3 behavior tests | ✓ Seeded |
| dnaa-01g-parca-te-derivation-audit | (didn't exist) | **Design** (planned), 1 behavior test | ✓ Seeded |
| dnaa-02-atp-hydrolysis | planned, Design, 0 findings | Design, **3 findings**, hypothesis-rich | ✓ Updated |
| dnaa-03-box-binding | planned, Design, 0 findings | Design, **1 finding** | ✓ Updated |
| dnaa-04-initiation-mechanism | planned, Design, 0 findings | Design, **2 findings** | ✓ Updated |
| dnaa-05-rida-ddah-dars | planned, Design | planned, Design | – (deferred) |
| dnaa-06-seqa-sequestration | planned, Design | planned, Design | – (deferred) |

**Net add:** 9 new findings across 4 studies, 2 new spawned follow-up studies,
1 drafted Step file (IntrinsicHydrolysis), 19 new sims, 10 SVG visualizations.

## The 10 biological / computational insights

1. **dnaA's ParCa-cached TE is at the 8.6th percentile of the proteome** (7.23e-5 vs proteome median 2.30e-4) — biologically improbable for a master regulator.
2. **The DnaA-ATP / DnaA-ADP / apo equilibrium machinery is already wired into baseline**; req-1 of dnaa-02 ("split DnaA bulk") is partially done.
3. **DnaA-ATP fraction = 0.99 in v2ecoli baseline** (Boesen 2024 target [0.2, 0.5]); equilibrium drives all DnaA to ATP-bound, no hydrolysis sink.
4. **DnaA-ADP pool = 0 throughout simulation**; the conversion mechanism is broken.
5. **Metabolic reaction RXN0-7444** (DnaA-ATP intrinsic hydrolysis, catalyzed by CPLX0-10342) is declared in `metabolic_reactions.tsv:4839` but has NO kinetic constraint — silently inactive in FBA.
6. **TE → DnaA-count is non-monotonic**; phase transition between 20× and 25× TE causes autorepression saturation (Pearson r flips from -0.253 to +0.78, count drops 296 → 159).
7. **15× TE is the cleanest single-knob calibration** (autorepression PASS r=-0.533, count short 18%).
8. ★ **(TE=20×, fc=0.7) passes both gate tests** (DnaA 707, r=-0.521). The fold_change multiplier is the missing second knob.
9. **DnaA-box catalog already exists** in `chromosome_structure.py` (DNAA_BOX_ARRAY); req-1 of dnaa-03 is largely done — just needs per-box affinity attributes + binding Step.
10. **dnaa-04 swap point identified**: replace the mass-threshold heuristic at `chromosome_replication.py:244` with DnaA-occupancy-based trigger.

## What's in this directory

```
overnight-2026-05-17/
├── REPORT.md              ← this file (read first)
├── PROGRESS.md            ← chronological log with all numbered steps
├── FRICTION.md            ← 9 specific friction points with concrete fixes
├── probe_dnaa_total.py    ← one-shot DnaA-state probe (Step 1)
├── probe_dnaa_states_timeseries.py  ← time-series DnaA equilibration probe
├── proposed_intrinsic_hydrolysis_step.py  ← drafted dnaa-02 model improvement
├── run_te_focused_sweep.sh       ← shell wrapper for the TE-only sweep
├── run_baseline_with_fc.py       ← runner with --dnaa_autorep_multiplier
├── run_fc_test.sh                ← pilot fc sweep (10 sims)
├── run_fc_validation.sh          ← 5-seed validation at sweet spot
├── sweep_log.txt                 ← TE sweep log
├── fc_test_log.txt               ← fc sweep log
├── sweep_aggregate.json          ← TE-only aggregate data
├── fc_sweep_aggregate.json       ← TE×fc joint aggregate data
├── dnaa_states_timeseries.json   ← per-step DnaA pool trajectory
└── viz/
    ├── 01_te_sweep_count.svg
    ├── 02_te_sweep_pearson.svg
    ├── 03_te_sweep_combined.svg
    ├── 04_dnaa_te_percentile.svg
    ├── 05_dnaa_states_timeseries.svg
    ├── 06_dnaa_atp_fraction.svg
    ├── 07_findings_overview.svg
    ├── 08_investigation_dag.svg
    ├── 09_fc_grid_dnaa_count.svg
    └── 10_fc_grid_pearson.svg
```

## Concrete next steps (in priority order)

1. **Validate (TE=20×, fc=0.7)** — confirm the 2-seed result holds at 5 seeds (validation run in progress; check `fc_test_log.txt` and re-run `viz/make_fc_sweep_charts.py`).

2. **Promote to permanent**: If 5-seed validation passes, edit
   `v2ecoli/processes/parca/reconstruction/ecoli/flat/adjustments/translation_efficiencies_adjustments.tsv` to add:
   ```
   "PD03831[c]"  20  "fit_sim_data_1.py"  "dnaA, master regulator; ribosome profiling underestimates per Schmidt 2016 mass-spec"
   ```
   AND add a one-line preprocessing patch in `promoter_fitting.py` (or the dashboard's runtime config) that scales delta_prob.deltaV[deltaJ==12] by 0.7.

3. **Wire IntrinsicHydrolysis Step** (`proposed_intrinsic_hydrolysis_step.py`) into baseline. Run a baseline-with-hydrolysis check; verify ATP fraction drops into [0.2, 0.5] band (target for dnaa-02 primary test).

4. **Add per-box affinity to DnaA_box catalog** (dnaa-03 req-1 finish). Coordinates + domain_index are already there; just need a per-box `affinity_kd_nM` attribute.

5. **Implement DnaA-occupancy-based initiation trigger** in `chromosome_initiation.py` DnaABinder stub. Then patch `chromosome_replication.py:244` to consult it.

6. **Spawn dnaa-05 and dnaa-06 work** — these are fresh-code investigations (RIDA/DDAH/DARS and SeqA), more open-ended than the upstream four.

## Caveats

- ~~The (TE=20×, fc=0.7) win is currently from **2 seeds only**. 5-seed validation
  is in progress.~~ **VALIDATED at 5 seeds (DnaA median 707, Pearson r = -0.521).**
- Modifying `delta_prob.deltaV` at runtime via the `--dnaa_autorep_multiplier`
  flag is a band-aid. Permanent integration should go through ParCa's
  `promoter_fitting.py` so the cache rebuilds with the corrected values.
- The IntrinsicHydrolysis Step is **drafted, not wired**. Editing
  `baseline.py` to include it should be done with a careful side-effect
  check on the other ~50 processes.
- All sims are 10-minute single-generation. Cell-cycle phenomena (doubling,
  division) not tested. dnaa-04+ depend on multi-generation runs.
