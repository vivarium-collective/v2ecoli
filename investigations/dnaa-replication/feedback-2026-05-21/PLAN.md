# dnaa-replication — Expert feedback response plan (round 1, 2026-05-21)

Reviewer: **Haochen**. Feedback captured via the inline-feedback widget on an
exported report (`investigation-dnaa-replication-2026-05-20-3.html`). Raw
feedback: `./feedback.yaml`.

## The headline problem (Eran)

> "Results are mixed — some generated 5/17, some 5/19 — hard to read. Make
> result-tracking robust so visualizations/results are guaranteed coordinated."

**Root cause.** There is no atomic *result generation*. `runs.db` accumulates
runs across dates; each viz HTML carries its own independent `repro-banner`
timestamp (e.g. dnaa-02 `dnaa_state.html` = "Generated 2026-05-19 13:26 · git
d146458"); the exported report bakes whatever is on disk at export time.
Nothing links all studies' displayed results to one code+param state, so an
exported report glues together panels from different days — and some are
literally pre-parameter-change (the reviewer: "this seems to be before
implementing new parameters").

## Critical realization: the params were saved but never *implemented*

The reviewer's "newly provided transcription and translation parameters" and
"parameters provided in the new parameter file" are **the same Stage-1 table we
already have** (`references/expert/wcm_stage1_parameters.pdf` +
`references/data/stage1_parameters.yaml`, content-identical to the PDF re-sent
2026-05-21). They were catalogued (expert doc + dataset + investigation
guideline) but never **wired into the model** — so every run used v2ecoli's
ParCa defaults (translation efficiency ~20×), which the reviewer immediately
recognized as unimplemented. **The gap is enforcement, not availability.**

---

## Part 1 — Guaranteed-coordinated results (provenance/generation model)

- Add a `generation_id` to `runs_meta` + a workspace-level current-generation
  pointer and manifest: `{git_sha, param_set_hash, composite_versions,
  created_at, runs: [{study, run_id, sim_name}]}`.
- `scripts/prepare_investigation.py` becomes the **generation driver**: one
  invocation runs every study's baseline + comparison variants and renders
  every comparative under ONE `generation_id`, stamping each run + viz.
- The report (and exported HTML) shows the generation id + timestamp once,
  prominently, and **flags any panel from an older generation** ("⚠ stale").
  Mixing becomes visible, not silent.

## Part 2 — Scientific feedback → per-study actions

| Study | Expert point | Action |
|---|---|---|
| dnaa-00 | runs too short | run **one+ cell cycle** (τ≈150 min ⇒ ~9000 s) |
| dnaa-01 | transcription (1.5 mRNA/min/gene) + TE (1) **not implemented** | **wire** these Stage-1 params (ParCa override hook) |
| dnaa-02 | DnaA-ATP too high (TE 20×) | same TE wiring — TE must be 1 |
| dnaa-02 | **RIDA premature** (datA/DARS absent) | dnaa-02 = **intrinsic-hydrolysis only**; RIDA → later study |
| dnaa-02 | verify hydrolysis math = `dX/dt=-kX` | audit `DnaaIntrinsicHydrolysis` |
| dnaa-02 | tests 3/4 fail (fraction flat) | re-run variants with captured `atp_fraction` (current = None/stale) |
| dnaa-03 | **456 boxes vs 307 consensus**; search criteria unclear | constrain to 307 (3 high-affinity oriC, no low-affinity); document criteria |
| dnaa-03 | Hill coeff can't track oriC occupancy | **differential affinity by occupancy** for low-affinity boxes (50→1 nM) |

## Part 3 — Friction points in our assumptions

1. **Test-passing drove mechanism choice.** dnaa-02f picked rida-v0 because it
   made the ATP-band test pass — but RIDA is premature. We fell into the exact
   "regression-compatibility vs biological-validation" trap the
   `dnaa_biology_feedback` expert doc warned about.
2. **Stage-1 params treated as a catalog, not values to enforce.** Registered
   everywhere, wired nowhere → every run used v2ecoli defaults.
3. **Accepted v2ecoli's box-finder (456) without reconciling to literature
   (307)** or surfacing the search criteria.
4. **Convenient abstractions (Hill) over the observable the expert needs
   (per-box occupancy).**
5. **Sim length defaulted to "fast" (120–3600 steps), not biological cell
   cycles.**
6. **Provenance was an afterthought** — the headline problem.

## Part 4 — Framework restructuring

1. Generation/run-set provenance (Part 1).
2. **Param-enforcement gate:** study declares enforced Stage-1 params; framework
   verifies they're actually applied (param-set hash vs declared), fails if not.
3. **Mechanism-prerequisite ordering in the DAG:** RIDA/DDAH/DARS can't be "on"
   before their loci/partners are modeled.
4. **Two-track verdicts:** explicit `regression_compatibility` vs
   `biological_validation` per test.
5. **Bio-fidelity assertions** (`box_count == 307`, documented search criteria).
6. **Sim length in cell cycles**, not step counts.

## Part 5 — Sequencing

1. Plan doc + feedback import + confirm param file saved. *(this commit)*
2. Provenance/generation model (headline infra).
3. Wire Stage-1 transcription/TE (unblocks dnaa-01 + dnaa-02 ATP level).
4. dnaa-02 → intrinsic-only + hydrolysis-math audit.
5. dnaa-03 → 307-box catalog + occupancy-affinity model.
6. Re-run whole investigation as ONE generation at cell-cycle length; export a
   coordinated report.

Tracked as tasks #17–#23 in the working session.
