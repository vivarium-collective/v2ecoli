# MetabolismRedux 5-condition comparison investigation — design

**Date:** 2026-07-26
**Investigation:** `v2ecoli-vecoli-comparison`
**Status:** design (awaiting review)

## Goal

Turn the `metabolism_redux` study (currently basal-only, just unblocked by PR #389)
into a **developed, evidence-rich investigation**: v2ecoli reproducing genuine
vEcoli under the **MetabolismRedux swap across all 5 nutrient conditions**
(basal, with_aa, succinate, no_oxygen, acetate), graded statistically, with
interactive Plotly report cards, run on the Mac mini as a remote compute backend
and served from the mini's read-only dashboard.

## Run shape

- **Swap:** both engines swap FBA `ecoli-metabolism` → `ecoli-metabolism-redux`.
- **Conditions:** basal, with_aa, succinate, no_oxygen, acetate.
- **Statistical shape:** **2 seeds × 3 generations** per condition (seeds × gens).
  = 5 conditions × 2 engines × 2 seeds = 20 single-lineage multigen runs, 3 gens each.
- **Grading gates:** `statistical` + `parca` (existing gate set). `standard` +
  new cards are illustrative/evidence.
- **Statistical-power note:** 2 seeds alone gives a weak Welch t-test (n=2). The
  `distribution`/`statistical` cards therefore **pool per-cell values across the
  3 generations** (n ≈ 6 lineage-cells/condition) to recover power, and ALSO
  report a per-generation breakdown. This is a deliberate, documented compromise
  to keep the mini run tractable while exercising multi-generation dynamics.

## Approach (sequencing)

**Harden + validate locally, then one mini run.** Build configs, studies, and all
new cards locally — validating cards against the existing basal redux data plus a
1-seed×2-gen smoke run — BEFORE shipping to the mini. Multi-gen redux is untested
and remote card-iteration is slow; we must not discover a broken card or a
division bug after a multi-hour mini run.

---

## Section A — Harness hardening (prerequisites)

1. **Merge PR #389** (`fix/inject-vivarium-step-as-step`) to `main` so main and the
   mini carry the as_step fix. *(Requires user merge.)* The build branch is based
   off the fix so work can proceed in parallel; it rebases onto main after merge.

2. **Per-condition redux config generator** — a committed script
   (`scripts/gen_redux_condition_configs.py`) that, for each of the 4 non-basal
   conditions, fuses `configs/cond_<X>.json` (media/condition/nutrients) with the
   redux swap/flow/strip_pint/attach_pint/output_ports block from
   `configs/metabolism_redux_basal.json`, writing `configs/metabolism_redux_<X>.json`
   into the vEcoli fork. Deterministic and re-runnable; the generator is committed
   to v2ecoli and the 4 generated configs are committed to the vEcoli fork (so the
   run is reproducible and the mini gets them via git). UTF-8 I/O (known CI
   ASCII-locale gotcha).

3. **Five redux studies** in the investigation:
   `workspace/investigations/v2ecoli-vecoli-comparison/studies/metabolism_redux_<cond>/study.yaml`
   (basal reuses/renames the existing study). Each: `comparison.seeds=2`,
   `generations=3`, `from_vecoli_config=configs/metabolism_redux_<cond>.json`,
   `cards=[config, parca, standard, statistical, trajectory, distribution,
   metabolism, composition]`. Registered as investigation members.

4. **Multi-gen redux smoke gate (KEY RISK).** Before the full sweep, run
   **1 seed × 2 gens** redux (basal) locally through both engines. Confirms
   division + redux survive multi-generation (the PR #389 proof was 1 gen). If it
   fails → systematic-debugging fix FIRST; do not start the mini run. Gate output:
   a short pass/fail note with the observed per-gen masses.

## Section B — New report cards

All new cards are `@as_step`-decorated process-bigraph Steps in
`scripts/_compare/report_cards/`, following the existing contract
(`inputs=CARD_INPUTS, outputs=CARD_OUTPUTS`, returning `{card_html, verdict, axes}`),
self-registering into `REPORT_CARD_STEPS` and imported in `report_cards/__init__.py`.
Interactive figures are emitted as Plotly `fig.to_html(include_plotlyjs="cdn",
full_html=False)` placed verbatim in a section's `html` field (which
`_sections_to_html` and `report._section_html` pass through untouched). CDN
Plotly is acceptable — the mini-served dashboard has network; note the offline
tradeoff in the card docstring.

1. **`trajectory`** — interactive per-observable (the 7 standard axes: cell/dry/
   protein/rna mass, growth_rate, active_RNAP, active_ribosome) v2 (amber) vs
   vEcoli (indigo) value-vs-time traces, with per-seed spread bands, generation
   boundary markers, hover + zoom. Reads both engines' zarr via
   `read_pbg_local`/`read_v2ecoli_trajectory`. Verdict: `ungraded` (evidence).

2. **`distribution`** — per-axis violin + strip of per-cell values, v2 vs vEcoli,
   pooled across seeds×gens, annotated with Welch Δ%, p, Cohen's d, and CI (reuses
   `v2ecoli/library/card_criteria.grade_axis` ttest). Hardens the statistical
   verdict visually. Verdict: `worst` of axes (graded, non-gate).

3. **`metabolism`** — metabolism-specific evidence: growth-rate-vs-nutrient "growth
   law" across all 5 conditions on one figure (v2 vs vEcoli), plus biomass/uptake
   or exchange fluxes WHERE cheaply emittable. Flux observables are NOT in the
   current 8-path compact view; the plan will (a) check whether redux emits a
   comparable flux/exchange leaf, and (b) either extend the emit view minimally or
   scope this card to growth-law + biomass and flag fluxes as follow-up. Verdict:
   `ungraded` (evidence) unless a graded growth-law axis is added.

4. **`composition`** (+ cell-cycle/perf panel) — proteome/RNA mass-fraction bars
   (v2 vs vEcoli), cell-cycle/division timing (doubling time, division tick per
   gen), and a small convergence/performance panel (steps/s, timestep — showcasing
   the PR #389 fix's effect). Verdict: `ungraded` (evidence).

**Harden existing cards:** enable violin rendering on the `statistical` card axes
(set `plot="violin"` on `CARD_AXES`/`EXTRA_AXES`); confirm `config`/`parca` render
cleanly for the redux swap.

## Section C — Run on mini + serve

1. **Sync** the fix + generated configs + studies to the mini
   (`~/code/sync-pbg-to-mini.sh` and/or git fetch of the build branch). Verify mini
   reachability, mirrored v2ecoli + vEcoli checkouts, `out/cache_full` +
   `out/compare_harness` caches, and the fix present.
2. **Run** the 5-condition redux ensemble on the mini via **Ray**
   (`run_comparison_ensemble.py --mode ray`, `V2E_RAY_THREADS` set to bound the
   ~16 GB/seed memory to ~3 concurrent on the 12-core/69 GB mini), one subprocess
   per seed for isolation. Driven headless via `mct` or a detached ssh run; verify
   progress via on-disk zarr stores, never the buffered log.
3. **Render** the investigation (`runner.run_investigation` →
   `comparison_report_card.py`) producing per-study cards + verdicts + Plotly
   embeds + the assembled `standardized_comparison_report.html`.
4. **Serve** the finished investigation read-only from the mini dashboard
   (`mdash` / `vdash-ro`) so report cards, Plotly figures, and result stores are
   accessible "remotely."

---

## Risks & mitigations

- **Multi-gen redux untested** → Section A.4 smoke gate before the mini run.
- **Weak stats at 2 seeds** → pool per-cell across generations (n≈6); report
  per-gen; document the compromise.
- **Metabolism flux observables absent from the compact view** → scope the
  `metabolism` card to available observables; flag flux extension as follow-up.
- **Mini connectivity/state** → verify-first prerequisite step; SSH was hanging at
  design time.
- **CDN Plotly offline** → acceptable for the networked mini dashboard; documented.
- **Runtime** → 3 gens × redux is multi-hour on the mini; run overnight/detached.

## Success criteria

- All 5 redux conditions run to completion (2 seeds × 3 gens) on the mini, both
  engines, no hang/crash.
- Each condition graded on `statistical` + `parca`; verdicts materialized.
- All four new cards render with real data, with working interactive Plotly.
- Investigation served read-only from the mini dashboard; report cards, figures,
  and result stores accessible.
- The investigation's executive summary + findings updated with the 5-condition
  redux reproduction result.

## Out of scope (YAGNI / follow-up)

- Stage-2 first-class sms-api `ComputeBackend.MINI` (the workbench "Run on remote"
  card driving the mini) — the mini is used as a compute backend + served results,
  not a first-class API peer.
- Standard FBA-metabolism re-run (this investigation is redux-only).
- Flux-level report card if it requires non-trivial new emit paths.
- Colony/multi-lineage runs (single-lineage multigen only).
