# Honest & Robust pbg-vs-pbg Comparison Harness — Divergence Attribution

*Design — 2026-06-24*

## Objective

Build an honest, robust comparison harness between the `v2ecoli` port and the genuine,
**unmodified** upstream `CovertLab/vEcoli` model — both externally wrapped as process-bigraph
(pbg) composites, **both on the same ParCa**. The goal is not just to *quantify* agreement but to
**systematically attribute divergence to specific processes**, fix the real port bugs, and then run
the full **16 seeds × 16 generations × 5 media conditions** comparison on GovCloud, rendering a
matched-timepoint statistical report with per-process attribution.

Running both engines *as pbg composites* removes the Nextflow-vs-Ray execution-path confound, so any
residual divergence is a true port difference.

## Starting point (what already exists — reuse, don't rebuild)

Two CI-green, mergeable PRs from a prior session are the foundation. **We branch from #289's branch
and reuse everything; we do not fork the base.**

- **v2ecoli #289** (`feat/upstream-vecoli-pbg`, HEAD `9be885a3`) — the entire wrapper + tooling:
  - Wrapper: `v2ecoli/library/vecoli_pbg_upstream.py`, `v2ecoli/library/upstream_division.py`
  - Three fixes: fail-closed partition gate (`vivarium_bridge.py`), guarded per-generation emitter
    close + fail-loud truncation guard (`xarray_run.py`), `translate_ports` contract fix
  - Report tooling: `scripts/comparison_report_card.py --pbg-vs-pbg`,
    `scripts/run_comparison_ensemble.py`, `scripts/compare_matched_trajectories.py`
  - Run scripts: `scripts/run_upstream_multigen.py`, `scripts/build_upstream_parca.py`,
    `scripts/comparison_harness.sh`
  - Docs/specs: `docs/comparison_pipeline.md`, `comparison_spec_4x4x5.json`
- **sms-api #147** (`feat/comparison-upstream-parca-cache`, `0.9.16`) — GovCloud routing: separate
  pristine-upstream ParCa cache, serial `--cpus 1` (required, not a perf knob). **Only used at
  Phase 3.**

**Critical nuance:** #289 is green on *unit* CI, but its body states the wrapper **dynamics are not
yet validated** — basal's `cell_mass` exploded (5k→98k over one generation) before the fixes, and the
fixes are "validated by code analysis, pending image rebuild." Our Phase 0 supplies that missing
validation **locally and cheaply**, which is exactly what should gate the merge of #289 + #147.

Working tree: `/Users/eranagmon/code/v2e-compare-harness` (already at `9be885a3`, 0 behind origin).

## Design decisions (from brainstorming)

1. **Sequence:** local basal → local 5 conditions → GovCloud full run. Start small, prove the science
   locally and cheaply before paying for any model-image rebuild.
2. **Divergence depth:** systematic **per-process attribution** — a per-tick, per-process bulk/unique
   delta diff between matched processes, producing a ranked "which process diverges most, and when"
   table. Not just headline equivalence bands.
3. **Attribution method (two-tier):**
   - **Tier A** — shared-seed, tick-locked single-seed diff to catch *gross/structural* divergence
     deterministically (wrong wiring, unit error, off-by-one): the first tick a process delta diverges
     grossly localizes the bug.
   - **Tier B** — multi-seed per-process delta *distributions* to catch *systematic* biases buried
     under stochastic noise; flag processes whose distribution shifts beyond a noise band.
4. **Final scale:** 16 seeds × 16 generations × 5 conditions (= 1280 generation-sims/engine, 2560
   total) on GovCloud.
5. **Delta-capture architecture (Approach C / hybrid):** co-execution harness for Tier A; emit-and-diff
   into the existing XArray/zarr emitter for Tier B. Each tier uses the capture mechanism that fits it;
   no wasted tooling.

## Architecture

### Tier A — co-execution harness (local, single process)
A new harness instantiates **both** pbg composites (v2ecoli + upstream wrapper) in one process,
initializes them from a **shared initial state and shared RNG seed**, and steps them **tick-locked**.
Each tick it captures each process's output update (bulk counts + unique-molecule deltas) and diffs
matched processes. It records the **first tick** at which any process delta diverges grossly (beyond a
structural threshold), with the offending process, ports, and molecules. Output: a structural-onset
report.

> Determinism caveat: the port and upstream may not consume RNG in identical order. Tier A targets
> *gross* divergence (large, deterministic-enough to see through noise — wrong wiring, missing/extra
> process, unit/compartment error). Subtle stochastic-scale differences are Tier B's job. The harness
> first empirically checks how far shared-seed determinism holds (1-tick bulk-delta agreement on
> clearly-deterministic processes) and reports it, rather than assuming bit-identity.

### Tier B — per-process delta emit + statistical diff (local & GovCloud)
Instrument both composites to emit each process's per-tick output delta into the existing
XArray/zarr emitter (additive, non-invasive; reuses the emitter infra). Run each engine independently
across N seeds. Offline, compare per-process delta **distributions** between engines at matched
timepoints; flag processes whose mean/median shift exceeds a **noise band** established from a
v2-vs-v2 across-seed baseline. Output: a ranked per-process divergence table.

### Process-correspondence map
A name-based mapping (v2ecoli process ↔ upstream vivarium process) used by both tiers, with explicit
handling of processes that exist in one engine but not the other (reported, not silently dropped).

### Reporting (Phase 4)
Extend the existing `comparison_report_card.py --pbg-vs-pbg` output with: (a) matched-timepoint
equivalence bands per condition (existing), (b) the per-process attribution table, (c) a documented
residual-divergence section distinguishing fixed / characterized-as-expected / open.

## Phases & gates

**Phase 0 — Local basal validation** *(start small)*
Run upstream wrapper + v2ecoli basal locally on the current worktree. **Gate on physical validity,
not "it ran":** `cell_mass` ~doubles over a generation (not ~18×) and divides cleanly; per-generation
emitter close path runs without the `include_static` assert; fail-loud guard confirms no silent
truncation. → Green here = evidence to merge #289 + #147.

**Phase 1 — Per-process delta-capture tooling**
Build Tier A co-execution harness, Tier B per-process delta emit, the process-correspondence map, and
the v2-vs-v2 noise-band baseline. Unit-test the diff/attribution logic.

**Phase 2 — Local 5-condition divergence pass**
Run both engines locally across all 5 conditions at shallow depth (**1–2 generations per condition** —
scale depth is what GovCloud is for). Tier A onset + Tier B statistical → ranked divergence table →
fix structural bugs → re-run → confirm bands tighten. Iterate locally (cheap) until each divergence is
either fixed or characterized as expected.

**Phase 3 — GovCloud scale-up (16 × 16 × 5)**
Merge #147, rebuild the v2ecoli image, smoke run (1 seed) — **keyed on exact job IDs**, reading the
**correct** `smsvpctest-ray-batch-RayBatchLogs` log group — then the full two-engine run.

**Phase 4 — Honest comparison report**
Matched-timepoint statistical report card (equivalence bands per condition) + per-process attribution
table + documented residual-divergence section. Honest by construction: raw values inspected, not just
verdicts.

## Cross-cutting discipline (from the friction log §4b)

- **"It ran" ≠ "it's right."** Assert physical validity and inspect *raw values* before trusting any
  verdict or rendered report.
- **Aggregate Batch counts lie.** Key all monitors on *exact job IDs*, never name-substring status
  counts (terminated → FAILED false-trips).
- **Read the right log group** (`smsvpctest-ray-batch-RayBatchLogs…`), not `/aws/batch/job`.
- **Don't trust unverified assumptions** (e.g. fork-inherits-Cython-pin cost >1 h). Test the cheap
  thing locally before the expensive rebuild.
- **Slow iteration loops** (build→push→deploy ~8 min; image rebuild heavier) reward reading code/logs
  carefully before each cycle.

## Out of scope (YAGNI)

- No fork of, or PR into, upstream vEcoli; no source edits — upstream stays git-clean (external
  wrapper only).
- No restoration of adaptive `GlobalClock` timestepping *unless* Phase 0/2 shows mass still drifts
  after the fail-closed gate (noted as a contingency, not built up front).
- No new emitter backend — reuse the existing XArray/zarr emitter for Tier B.

## Open items to verify during implementation

- Whether the 5 media conditions need separate upstream ParCa caches or share one cache with condition
  as a runtime knob (affects Phase 2 local ParCa cost: upstream serial ParCa ~14 min/condition).
- How far shared-seed determinism actually holds between the two engines (empirical, Phase 1 Tier A).
