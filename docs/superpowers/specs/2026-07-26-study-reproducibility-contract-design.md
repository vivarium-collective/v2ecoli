# Study Reproducibility Contract & Audit System — Design

- **Date:** 2026-07-26
- **Status:** Approved (brainstorming); ready to plan Phase 1
- **Repos:** `v2ecoli` (workspace + CI gate), `viva_superpowers` (migrators + audit module), `vivarium-workbench` (execution spine + Audit view)
- **Branch (this spec):** `spec/study-reproducibility-audit` (v2ecoli)

## 1. Motivation

Every v2ecoli study should be **reproducibly rerunnable end to end**: a definite
set of models (composite + config) listed in the Model tab, simulations that run
and save, an analysis flush that produces visualizations + downloadable analyses
+ report cards in the correct study tabs, report cards that yield verdicts which
feed the decision and surface as evidence — and, across an investigation, studies
that run **in dependency order** so one study's outputs feed the next. We want
"rerun study" and "rerun investigation" to trigger all of this deterministically,
and we want a **formalized audit system** that enforces these standards reliably
rather than checking them once by hand.

### 1.1 Ground-truth audit (2026-07-26, against `v2e-main-serve`/`vivarium-workbench--serve` at latest `origin/main`)

The pieces largely exist as scaffolding but are drifted, unwired, or stubbed:

**Data model has drifted (the biggest hazard):**
- **Dual study layout is back.** 48 canonical `studies/<slug>/study.yaml` **plus 9
  nested `investigations/<inv>/studies/<slug>/study.yaml`**, all git-tracked. A
  migration (commit `a06844c0`) removed the nested copies; later commits re-created
  9. The resolver `viva_superpowers/workspace_paths.iter_study_dirs()` yields
  **nested first and dedups by slug**, so the older, diverged nested copy *wins* on
  the dashboard while `lint-workspace.py` validates only the top-level copy — a
  split brain (e.g. nested `ko-and-media` lacks the `inputs: sim_data from parca`
  block the top-level copy has). Affected: `ketchup-exchange-comparison`,
  `pdmp-00-characterization`, `colonies-04/-05/-06/-08/-09`, `metabolism_redux`.
- **No `config:` field exists in any study (0/48).** "Config" is today inline
  `params` attached to each composite. Composite attachment has **two rival
  schemas**: Style A (`conditions.baseline` + `conditions.variants[]`), Style B
  (top-level `baseline:` list). Some studies (pdmp-*, mbp-01/02) carry *both* plus
  a `variants:` list — genuine schema drift.
- **Investigations split**: 6 use `members:`, 4 use `studies:`. Flat slug lists,
  no ordering at the investigation level.
- **Ordering lives in each study's `pipeline_gate.prerequisites`/`enables`** — the
  real DAG — but it's inconsistent: numbering inversions (`param-uq-00` requires
  `-01`) and dangling references (`colonies-01` enables a slug that doesn't exist).
- **Multiple composites per study** is real and works (ketchup lists 10, pdmp-01
  lists 9); the Model tab renders these.
- **No nested-studies guard is vendored** in v2ecoli; `lint-workspace.py` checks
  depth-1 only, so the 9 nested duplicates are invisible to it.

**Execution / evidence pipeline exists but is largely unwired:**
- The content-addressed, hash-gated, pull-or-compute pipeline
  (`vivarium_workbench/lib/artifacts/*`) is correct **but dead code** — only tests
  call it. Its `_default_compute` does reach the real engine
  (`run_core.invoke_run` → `run_runner.execute`) but is fed no real emit-paths and
  is invoked by no endpoint.
- **Reruns work at the single-run level** (`POST /api/run-rerun` replays a recorded
  manifest; `POST /api/investigation-rerun` re-runs each member baseline) **but in
  declared order, with no topological output-chaining and no "prereq passed" gate.**
  No `toposort` in any live path.
- **Report-card generator is a stub** — `lib/composite_flush.render_report_card`
  emits an HTML list of figure names, no verdict, no tolerance comparison, no
  `verdict.json`. Behavior-test outcomes *are* computed
  (`viva_superpowers.study_evaluator.compute_outcomes`).
- **Decision/Evidence is a read-time derivation over authored fields**
  (`lib/chain_derivation`, stamped `actor="derived"`), not an automatic
  sim→verdict→finding→decision causal pipeline. No dedicated Evidence tab in the
  study view.

## 2. Goals / Non-goals

**Goals**
- One canonical, drift-free study/investigation data model.
- A single source of truth for cross-study execution order (data-flow).
- A reproducibility **contract** expressed as runnable checks, enforced by a
  tiered CI gate and surfaced in the workbench.
- Wire the reproducible execution spine so "rerun study/investigation" is
  deterministic and dependency-ordered.

**Non-goals (now)**
- Rewriting the simulation engine or emitters.
- Adding AI to the dashboard (dashboard stays AI-free; all AI stays in
  `viva_superpowers` skills).
- Redesigning the study-detail tab layout beyond what the contract requires.

## 3. The Reproducibility Contract (north star)

A conformance ladder. Each check is `hard-fail` or `warn` per the tiered posture
(structure hard-fails; richness warns and ratchets to fail as coverage grows).
The Phase-3 audit module asserts these; Phases 1–2 build toward them.

| # | Check | Tier |
|---|---|---|
| **L0 Structure** | No nested `study.yaml` under `investigations/`; investigations use `members:` only; every study uses the canonical `baseline:`+`variants:` model schema; slug == dir name | hard-fail |
| **L1 Resolvability** | Every `composite:` resolves in the registry; every model `config` loads/canonicalizes; every `inputs.from` names a real study + declared output artifact; the inputs-derived DAG is acyclic with no dangling edges | hard-fail |
| **L2 Executability** | Study resolves→init→simulate→save (plan/smoke check); rerun is deterministic (same composite+config+inputs+commit → same `artifact_id`) | warn → hard-fail after Phase 2 |
| **L3 Outputs** | Flush lands viz in `studies/<slug>/viz/`, downloadable analyses in exports, report cards in `viz/report_card/` — in the tab the UI reads | warn (ratchet) |
| **L4 Evidence** | Each declared report card emits a computed `verdict.json`; verdicts + behavior-test outcomes feed the decision and surface as evidence; findings link to a test/run | warn (ratchet) |
| **L5 Ordering** | Investigation has a valid topological order over `inputs.from`; ordered execution runs producers before consumers, passing artifacts forward | graph-validity hard-fail; execution warn → hard-fail after Phase 2 |

## 4. Phased decomposition

- **Phase 1 — Data-model canonicalization** (this spec's implementation target):
  the clean baseline. Kill the dual layout, unify the composite/config schema,
  one investigation key, one ordering source, fix the resolver.
- **Phase 2 — Wire the execution spine** (own spec later): make `lib/artifacts`
  pull-or-compute live in the run/rerun endpoints; run investigations in
  topological order with output-chaining; make report cards compute real
  verdicts → outcomes.
- **Phase 3 — The audit system** (own spec later): the contract as runnable checks
  in `viva_superpowers`, tiered CI gate in v2ecoli, read-only workbench Audit view.

## 5. Phase 1 design (detailed)

### 5.1 Canonical study model schema

Collapse Style A/B into one. `config` is the formalized per-model params dict
(the "config" in the user's mental model: a study lists models, each is
`{composite, config}`).

```yaml
# studies/<slug>/study.yaml
name: <slug>
title: ...
baseline:                     # list (usually 1); the control model(s)
  - name: baseline
    composite: v2ecoli.composites.ecoli_baseline.ecoli_baseline
    config: {}                # was `params`; now the explicit, required key
variants:                     # alternate models; rendered after baseline in the Model tab
  - name: <variant>
    composite: <composite-id>
    config: {...}
inputs:  [{artifact: sim_data, from: parca}]
outputs: [{artifact: <name>, ...}]   # required if a downstream study consumes this
tests: [...]
report_cards: [...]
```

- `baseline` is a **list** to match the workbench Model panel, which already
  iterates `study.baseline[]`. **Open confirmation:** verify list-vs-single against
  the workbench study-spec normalizer while writing the plan; if the normalizer
  emits a single baseline object, adjust the canonical form to match so no UI
  change is required.
- `config` replaces `params`. Empty config is allowed but the **key must be
  present** (so L0 can assert `{name, composite, config}` on every entry).
- `conditions.model_settings[]` metadata folds into each model's entry (as
  `config` or a sibling `metadata`, decided in the plan against how the Model tab
  reads model_settings).
- The `conditions:` wrapper is removed after lifting.

### 5.2 Ordering = `inputs.from` (data-flow is canonical)

- Execution edge = data dependency: study B runs after A because B consumes A's
  output artifact.
- Derive the DAG + topological order from `inputs.from`. Migrate
  `pipeline_gate.prerequisites` into `inputs.from` edges. A prerequisite with **no
  artifact to consume** is flagged: either the real data dependency is made
  explicit (add the producer's `outputs` + the consumer's `inputs.from`), or the
  edge is dropped as spurious ordering. Decisions on flagged edges are surfaced,
  not guessed.
- Delete `prerequisites` / `enables` / `parent_studies` after migration.
- Fix known breakage: `colonies-01` enables a nonexistent slug; `param-uq`
  numbering inversion (`param-uq-00-screen` requires `param-uq-01-elongation`).

### 5.3 Kill the dual layout

- For each of the 9 nested `study.yaml` duplicates: ruamel round-trip **merge any
  hand-authored comments/prose** the nested copy has and the top-level lost into
  the top-level copy, then **delete the nested `study.yaml`**.
- Fix `viva_superpowers/workspace_paths.iter_study_dirs()` to yield **top-level
  `studies/*/` only**, so nesting can never shadow again. (Nested run-output sinks
  like `runs.db`/`charts/` under `investigations/*/studies/*` are a Phase-2 run-
  location concern; Phase 1 only removes the nested `study.yaml`.)

### 5.4 Unify investigation key

- `studies:` → `members:` for the 4 investigations still using `studies:`
  (`multiscale-bioprocess`, `parameter-uq`, `structural-ecoli`,
  `surrogate-modeling`). Members remain flat slug lists; ordering comes from
  `inputs.from`, not member order.

## 6. Implementation surface (Phase 1)

- **`viva_superpowers`** (new, ruamel comment-preserving, idempotent — reusing the
  existing round-trip discipline):
  - `study_schema_migrate.py` — Style A/B (+ both) → canonical `baseline`/`variants`
    with `config`.
  - `ordering_migrate.py` — `pipeline_gate.prerequisites`/`enables`/`parent_studies`
    → `inputs.from`; flag no-artifact prereqs.
  - `workspace_paths.iter_study_dirs()` — top-level-only resolver fix.
  - `investigation_key_migrate.py` (or fold into schema migrate) — `studies:` →
    `members:`.
- **`v2ecoli` workspace** — the 48 studies migrated in place; 9 nested
  `study.yaml` deleted (after comment salvage); 4 investigations key-renamed; DAG
  breakage fixed.
- **`v2ecoli/scripts/lint-workspace.py`** — extend as an **interim guard**
  (nested-study detection + schema-drift detection + members-only + acyclic DAG)
  until the Phase-3 audit module supersedes it.

## 7. Testing

- **Golden / idempotence:** run each migrator twice → second run is a no-op;
  byte-compare hand-authored comments on a comment-heavy study before/after
  (comment preservation is a hard requirement — programmatic study.yaml edits must
  use the ruamel round-trip, never `yaml.safe_dump`).
- **Guard test** (interim, in `lint-workspace.py` and/or a pytest): no nested
  `study.yaml`; investigations members-only; every study matches the canonical
  schema with `{name, composite, config}` per model; inputs-derived DAG acyclic
  with no dangling edges.
- **Resolver test:** `iter_study_dirs()` never yields a nested path; a slug that
  exists only nested is *not* resolved (proves the shadow can't recur).

## 8. Risks & mitigations

- **Comment loss** on programmatic study.yaml edits → ruamel round-trip only;
  golden byte-check gates the migrators.
- **Shared-checkout collisions** — do all edits/commits on a dedicated worktree
  (`~/code/v2ecoli--repro-audit`); verify branch/HEAD before every commit.
- **`config` folding ambiguity** (model_settings vs params vs metadata) → resolve
  against the actual Model-tab reader in the plan, not by guessing.
- **Flagged ordering edges** (prereq with no artifact) → surfaced for a human
  call; never silently dropped or fabricated.
- **Cross-repo coupling** — the resolver fix lands in `viva_superpowers`; v2ecoli
  must pin/consume the updated package. Sequence: `viva_superpowers` change +
  release/pin, then the v2ecoli data migration, so the guard runs against the
  fixed resolver.

## 9. Open items to resolve in the plan (not blockers)

- Confirm `baseline` list-vs-single against the workbench normalizer.
- Decide `model_settings` folding target (`config` vs sibling `metadata`).
- Decide whether the interim guard is `lint-workspace.py`-only or also a pytest in
  v2ecoli's CI (the tiered CI gate proper is Phase 3).
