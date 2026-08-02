# Phase B — the comparison on the investigation-as-composite substrate

**Date:** 2026-08-02
**Status:** design (for approval before implementation)
**Repos:** `v2ecoli` (compare-generalize) primarily + one substrate refinement in `vivarium-workbench` (inv-composite / PR #715)
**Depends on:** the investigation-as-composite substrate (PR #715). Runs against it via `PYTHONPATH=<inv-composite worktree>`.

## Goal

Make the whole-cell-model comparison run **through the composite substrate**: the `comparison:` block materializes to native `investigation.yaml` + per-config `study.yaml` files that the substrate executes — ParCa first (prerequisite), then each config's candidate+reference, then the cross-config matrix — with dependency order and cross-study data-flow coming from the composite graph.

## Re-model: paired studies → one native study per config

Today `materialize_comparison` returns *in-memory* paired specs (`<config>-candidate` + `<config>-reference`, two studies per config; `comparison_cards` on the candidate only). Phase B changes this to the workbench-native model and **writes files**:

- **Per config → ONE `study.yaml`** (`workspace/studies/<config>/study.yaml`):
  - `baseline`: the **candidate** — `ecoli_baseline` with `match_simdata` = the reference simData (matched initial state). Its run registers with `sim_name = <config>` (the v4 slug convention).
  - `variants: [{name: reference, composite: vecoli, params: {...}}]` — the **reference** arm. Its run registers with `sim_name = reference`.
  - `comparative_visualizations`: key observables (cell/dry/protein/RNA mass, growth), each with `runs: [{sim_name: <config>, label: candidate}, {sim_name: reference, label: reference}]` → the substrate renders overlays from the study's single `runs.db`.
  - `analyses: [{name: comparison_cards, params: {candidate_run: <config>, reference_run: reference, cards: [...]}}]` — per-study. `comparison_cards`'s adapter already resolves two run_ids from one `runs.db` by `sim_name`, so a baseline + variant in one study works.
  - `pipeline_gate.prerequisites: [{study: parca, relation: leads-to}]`.
- **`investigation.yaml`**: members = `[parca, <config>...]`; `analyses: [{name: comparison_matrix, params: {...}}]` (investigation-level, see Gap 2); keeps the `comparison:` block as the declarative source.

**The writer** (new): serialize each `to_study_specs()` entry via `yaml.safe_dump` to `studies/<name>/study.yaml` + emit `investigation.yaml`, reusing `study_seed`'s `pipeline_gate` edge convention (`prerequisite_edge` already emits the exact `{study, relation}` shape).

## Gap 1 — ParCa as a normal composite-study (not a special `kind`)

The substrate has no `parca_prerequisite` handler and shouldn't need one. Resolution: the **ParCa study is an ordinary study** whose baseline composite is a thin **`parca_prep` `@composite_generator`** that wraps `resolve_or_build_parca` (pull-or-compute):

- `parca_prep(candidate_cache_dir, reference_cache_dir, reference_repo)`: calls `resolve_or_build_parca(engine="candidate", ...)` and `(engine="reference", ...)` with `build=True` when the check returns `stale`; reused when valid. Emits a minimal completion so the substrate's `run_study` harvest records a `runs.db` row (a status/marker emit, not a sim trajectory).
- Each config study's `pipeline_gate.prerequisites` names `parca` → the substrate's scheduler runs `parca` first; the caches exist/validated before any candidate/reference run resolves its `simData.cPickle`.

**Risk (flagged):** `run_study` runs a study via the sim-run machinery, which may expect emitter/trajectory output. If a prep composite that emits only a marker doesn't satisfy that harvest, the fallback is a small **substrate refinement**: `run_study` recognizes a study whose composite is a `prep` kind and skips the trajectory-harvest expectation. Prefer the pure-composite approach; escalate to the substrate refinement only if the harvest rejects a marker-only run. This is the one place Phase B may touch the substrate.

## Gap 2 — the matrix reads per-config verdicts from disk (no `run_study`-reply change)

The cross-config `comparison_matrix` needs each config's `comparison_cards` **verdict**. In the substrate, the analysis step is wired to study **result stores** (the `run_study` reply), but `comparison_cards` runs as a **per-study analysis** whose verdict lands in the study's on-disk analysis output — not in the result-store payload. Rather than change `run_study`'s reply shape (a substrate change), the **`comparison_matrix` step loads each config's verdict from disk by study slug**:

- The `InvestigationAnalysisStep` for `comparison_matrix` is still **wired to every config study's result store** — that gives it the correct ordering (runs after all config studies) for free.
- Its `params` carry the member config slugs; at run time `comparison_matrix` reads each `studies/<config>/`'s persisted `comparison_cards` verdict (the analysis output / conclusion artifact) and assembles `config_verdicts`. This **replaces the `<candidate_run>::comparison_cards` placeholder token** with a real on-disk resolution — no substrate `run_study` change, and the wiring still enforces order.

So both gaps resolve **without changing #715's `run_study` contract** (Gap 1 ideally; Gap 2 fully). The only possible substrate touch is the Gap-1 prep-harvest fallback.

## Testing (hermetic; real paired e2e deferred to the mini)

- **Materializer/writer:** given a small `comparison:` block, assert the written `study.yaml` (baseline candidate + variant reference + comparative_visualizations + analyses + prerequisites) and `investigation.yaml` (members incl. `parca`, matrix analysis) match expected shapes. Hermetic.
- **Substrate run (stubbed worker):** run the written investigation through `run_investigation_composite` with the worker stubbed → assert order (`parca` before configs; matrix after all configs) and that the matrix step is invoked with the member slugs.
- **Gap-2 resolver:** given fixture per-config verdict files on disk, `comparison_matrix` assembles `config_verdicts` correctly.
- **`parca_prep` unit:** with a valid cache → `reused` (no build); with a stale cache → would build (mock the build).
- **DEFERRED — real paired e2e (mini):** candidate runs locally, but the reference (`vecoli`) side needs `$V2E_VECOLI_DIR` + a vEcoli ParCa rebuilt against current HEAD (the Jul-21-cache-vs-Jul-27-HEAD skew = the known `monomer_counts` shape mismatch). A genuine `basal seeds:1 gens:1 cards:[parca]` (t=0, no dynamics) paired run is the smallest real proof, run on the mini after a one-time reference ParCa rebuild. Tracked as a follow-up, not gating this build.

## Non-goals
- Retiring `v2e-compare` (separate; keep it working in parallel).
- GovCloud `run-remote` for the comparison (later).
- Multi-seed statistical card wiring (n=1 defers `statistical`).
- Intra-layer parallelism (substrate v1 is serial).

## Risks
- **Gap-1 prep-harvest** (above) — the one possible substrate touch.
- **`match_simdata` for the candidate** needs the reference simData path resolvable at study-run time — depends on the ParCa study having produced/validated the reference cache first (which the prerequisite ordering guarantees).
- **sim_name consistency** — the `comparative_visualizations` runs + the `comparison_cards` `candidate_run`/`reference_run` must reference the exact sim_names the substrate assigns (baseline = `<config>` slug; variant = `reference`). A mismatch renders empty overlays / missing verdicts silently.
