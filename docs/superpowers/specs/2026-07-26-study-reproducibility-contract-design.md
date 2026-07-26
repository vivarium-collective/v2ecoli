# Study Reproducibility Contract & Audit System — Design

- **Date:** 2026-07-26
- **Status:** Approved (brainstorming); ready to plan Phase 1
- **Repos:** `v2ecoli` (workspace + CI gate), `viva_superpowers` (migrators + audit module), `vivarium-workbench` (workflow engine + Audit view)
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

A second, unifying observation drives the design: **the investigation knowledge
graph is a workflow.** The finding→evidence→decision→conclusion chain, the
cross-study `inputs.from` DAG, and the intra-study run→flush→outcome pipeline are
all the same shape — typed nodes producing/consuming typed artifacts,
deterministic and replayable. We formalize them as one workflow (§4), which makes
reproducibility *structural* rather than merely audited.

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
  is invoked by no endpoint. **This is the latent workflow engine we promote in §4.**
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
- Model the knowledge graph and study pipeline as **one typed-artifact workflow
  DAG** executed by a content-addressed engine, so reproducibility is structural.
- A reproducibility **contract** expressed as runnable checks, enforced by a
  tiered CI gate and surfaced in the workbench.

**Non-goals (now)**
- Rewriting the simulation engine or emitters.
- Modeling the knowledge graph as a *literal* process-bigraph composite — the
  workflow engine is a standalone runner aligned with the idiom, not a composite.
- Adding AI to the dashboard (dashboard stays AI-free; all AI stays in
  `viva_superpowers` skills).
- Redesigning the study-detail tab layout beyond what the contract requires.

## 3. The Reproducibility Contract (north star)

A conformance ladder. Each check is `hard-fail` or `warn` per the tiered posture
(structure hard-fails; richness warns and ratchets to fail as coverage grows).
The Phase-3 audit module asserts these; Phases 1–2 build toward them.

| # | Check | Tier |
|---|---|---|
| **L0 Structure** | No nested `study.yaml` under `investigations/`; investigations use `members:` only; every study uses the canonical single model schema (`conditions.baseline` + `conditions.variants[]`) with `{name, composite, params}` per model; slug == dir name | hard-fail |
| **L1 Resolvability** | Every `composite:` resolves in the registry; every model's `params` loads/canonicalizes; every `inputs.from` names a real study + declared output artifact; the inputs-derived DAG is acyclic with no dangling edges | hard-fail |
| **L2 Executability** | A study's subgraph resolves→run→save through the workflow engine; every node is content-addressed, so a deterministic node re-keys identically (structural reproducibility) | warn → hard-fail after Phase 2 |
| **L3 Outputs** | Flush lands viz in `studies/<slug>/viz/`, downloadable analyses in exports, report cards in `viz/report_card/` — in the tab the UI reads | warn (ratchet) |
| **L4 Evidence** | Each declared report card emits a computed `verdict`; evidence nodes (verdict, outcome, finding, decision, conclusion) are **computed workflow artifacts**, not read-time derivations; findings link to a test/run | warn (ratchet) |
| **L5 Ordering** | The investigation is an executable typed-artifact DAG over `inputs.from` (acyclic); execution is topological with artifact forwarding; "rerun" re-executes only the minimal invalidated subgraph | graph-validity hard-fail; execution warn → hard-fail after Phase 2 |

## 4. The knowledge graph as a workflow (unified execution model)

The investigation knowledge graph and the study execution pipeline are the same
object. We formalize **one typed-artifact DAG**, executed by **one content-
addressed pull-or-compute engine**, viewed at three zoom levels. This is the
generalization of the existing `vivarium_workbench/lib/artifacts` pipeline, whose
`resolve_study` recursion ("resolve each `inputs.from` producer first, then
hash-gate pull-or-compute this node") is already a topological pull-or-compute
runner. It is promoted from study-only + unwired to the workspace's **live
workflow engine**.

### 4.1 One graph, three zoom levels

- **Investigation zoom** — study nodes; edges = `inputs.from` artifacts.
- **Study zoom** — the intra-study stage nodes (run → flush → analyses/report-cards → outcomes).
- **Evidence zoom** — outcome/verdict → finding → decision → conclusion.

The dashboard `aig-graph` view and the audit both read this one node/artifact
store; there is no separate read-time derivation. `chain_derivation` becomes a
*reader* of computed evidence nodes rather than a deriver from authored fields —
authored content remains allowed as enrichment/override, keeping provenance.

### 4.2 Typed nodes and artifacts

Every node has a type, typed inputs, and typed outputs (artifacts):

| Node | Consumes | Produces |
|---|---|---|
| upstream (e.g. `parca`) | — | `sim_data` |
| study-model-run | `sim_data` + composite + config | run artifact (emitter store) |
| flush | run artifact | viz / analyses / report-card-input artifacts |
| report-card | run + flush artifacts | `verdict` |
| test | run artifact | `outcome` |
| finding | `outcome` / `verdict` | `finding` |
| decision | `finding`(s) | `decision` |
| conclusion | `decision`(s) | `conclusion` |

Report cards and analyses remain process-bigraph **Steps** *inside* the run/flush
nodes — the engine is aligned with the process-bigraph idiom (typed ports = typed
artifacts) but is a standalone runner, not itself a composite.

### 4.3 Content-addressing and execution

- Node identity: `id = H(node_type + canonical(config) + sorted(input_artifact_ids) + code_commit)`.
  Same inputs + config + code → same id → cache hit.
- Execution = topological order over the DAG; each node is pull-or-compute (reuse
  the stored artifact if present, else compute exactly once and store atomically).
- **"Rerun study"** = invalidate that study's subgraph and recompute the minimal
  set of downstream-affected nodes. **"Rerun investigation"** = resolve the whole
  DAG (unchanged nodes are cache hits; only stale/missing nodes recompute).
- Reproducibility is therefore structural: a deterministic node with the same key
  never yields a different result, and the audit's L2/L5 checks assert the keying
  rather than re-deriving correctness by hand.

### 4.4 Relationship to Phase 1

Phase 1 is unchanged and load-bearing: making `inputs.from`/`outputs` the single,
typed ordering source *is* defining the investigation-zoom edges the engine
executes on, and the canonical model schema (`{name, composite, config}`) *is* the
`study-model-run` node's typed inputs.

## 5. Phased decomposition

- **Phase 1 — Data-model canonicalization** (this spec's implementation target):
  the clean baseline. Kill the dual layout, unify the composite/config schema, one
  investigation key, one ordering source (`inputs.from`), fix the resolver.
- **Phase 2 — Build/wire the unified workflow engine** (own spec later):
  generalize `lib/artifacts` from `resolve_study` (study-only) to a typed-node DAG
  resolver (`resolve_node`) with per-node-type compute (run, flush,
  report-card→verdict, test→outcome, finding, decision, conclusion); wire it into
  the live run/rerun endpoints (`/api/study-run-baseline`, `/api/run-rerun`,
  `/api/investigation-rerun`) so execution goes through the engine (the current
  `study_runs`/`cli_runs` paths delegate to it); implement the real report-card
  verdict node (kills the HTML stub); make `chain_derivation` read computed
  evidence nodes; replace declared-order investigation iteration with topological
  execution + artifact forwarding. The workbench graph view + audit read the
  node/artifact store directly.
- **Phase 3 — The audit system** (own spec later): the contract as runnable checks
  in `viva_superpowers`, tiered CI gate in v2ecoli, read-only workbench Audit view.

## 6. Phase 1 design (detailed)

### 6.1 Canonical study model schema

Collapse Style A/B into one. **Resolved against the workbench normalizer** (open
items from an earlier draft): the canonical authoring form is the **`conditions:`
block (Style A)** — this is what `lib/investigations.py` + `study-detail.html`
read *natively* (the template renders variants from `study.conditions.variants`;
the v4 projection synthesizes the Model panel's `baseline[]` list from
`conditions.baseline`). Canonicalizing to it is therefore a **zero-UI-change data
migration**. The per-model config key stays **`params`** — the Model panel reads
`b.params` only, and a distinct top-level `config:` already exists as the
execution-interface config (feeds the workflow node hash), so renaming would break
the UI and collide names.

```yaml
# studies/<slug>/study.yaml
schema_version: 4
name: <slug>
title: ...
conditions:
  baseline:                   # single model (the control); Model panel wraps it into baseline[]
    name: baseline            # optional; defaults to the study name
    composite: v2ecoli.composites.ecoli_baseline.ecoli_baseline
    params: {}                # the per-model config dict (kept as `params`)
  variants:                   # alternate models; rendered after baseline in the Model tab
    - name: <variant>
      composite: <composite-id>     # explicit — inherited from baseline when Style A omits it
      params: {...}                 # from the old `parameter_overrides`/`params`
      # perturbation / expected_contrast / description preserved verbatim
  model_settings: []          # kept under conditions.model_settings (NOT folded into params)
inputs:  [{artifact: sim_data, from: parca}]
outputs: [{artifact: <name>, ...}]   # required if a downstream study consumes this
tests: [...]
report_cards: [...]
```

- **Style B** (top-level `baseline:` list) and **"both"** studies (top-level
  `baseline:` *and* `conditions:`) migrate INTO the conditions form: the single
  baseline model moves to `conditions.baseline`, any top-level `variants:` merge
  into `conditions.variants`, and the redundant top-level `baseline:`/`variants:`
  keys are removed. A top-level `baseline:` list with >1 entry is **flagged for a
  human call** (no study currently has one).
- **Variants inherit the baseline composite**: Style A variants omit `composite`
  and only override params (e.g. `knockouts`, `media`); the migrator sets each
  variant's explicit `composite` = the baseline's when absent, so every model
  entry is a complete `{name, composite, params}`.
- YAML anchors/aliases (e.g. pdmp-00's `params: &id001`) and any nested
  `pipeline_gate` blocks deeper in prose must survive the ruamel round-trip
  untouched; only the top-level `conditions`/`baseline`/`variants` keys move.

### 6.2 Ordering = `inputs.from` (data-flow is canonical)

- Execution edge = data dependency: study B runs after A because B consumes A's
  output artifact. These edges are exactly the investigation-zoom edges of §4.
- Derive the DAG + topological order from `inputs.from`. Migrate
  `pipeline_gate.prerequisites` into `inputs.from` edges. A prerequisite with **no
  artifact to consume** is flagged: either the real data dependency is made
  explicit (add the producer's `outputs` + the consumer's `inputs.from`), or the
  edge is dropped as spurious ordering. Decisions on flagged edges are surfaced,
  not guessed.
- Delete `prerequisites` / `enables` / `parent_studies` after migration.
- Fix known breakage: `colonies-01` enables a nonexistent slug; `param-uq`
  numbering inversion (`param-uq-00-screen` requires `param-uq-01-elongation`).

### 6.3 Kill the dual layout

- For each of the 9 nested `study.yaml` duplicates: ruamel round-trip **merge any
  hand-authored comments/prose** the nested copy has and the top-level lost into
  the top-level copy, then **delete the nested `study.yaml`**.
- Fix `viva_superpowers/workspace_paths.iter_study_dirs()` to yield **top-level
  `studies/*/` only**, so nesting can never shadow again. (Nested run-output sinks
  like `runs.db`/`charts/` under `investigations/*/studies/*` are a Phase-2 run-
  location concern; Phase 1 only removes the nested `study.yaml`.)

### 6.4 Unify investigation key

- `studies:` → `members:` for the 4 investigations still using `studies:`
  (`multiscale-bioprocess`, `parameter-uq`, `structural-ecoli`,
  `surrogate-modeling`). Members remain flat slug lists; ordering comes from
  `inputs.from`, not member order.

## 7. Implementation surface (Phase 1)

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

## 8. Testing

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

## 9. Risks & mitigations

- **Comment loss** on programmatic study.yaml edits → ruamel round-trip only;
  golden byte-check gates the migrators.
- **Shared-checkout collisions** — do all edits/commits on a dedicated worktree
  (`~/code/v2ecoli--repro-audit`); verify branch/HEAD before every commit.
- **Config-key ambiguity** → RESOLVED: per-model key stays `params` (Model panel
  reads `b.params`); `model_settings` stays under `conditions.model_settings`; the
  canonical form is the `conditions:` block. No workbench normalizer change.
- **Flagged ordering edges** (prereq with no artifact) → surfaced for a human
  call; never silently dropped or fabricated.
- **Cross-repo coupling** — the resolver fix lands in `viva_superpowers`; v2ecoli
  must pin/consume the updated package. Sequence: `viva_superpowers` change +
  release/pin, then the v2ecoli data migration, so the guard runs against the
  fixed resolver.

## 10. Open items (resolved / deferred)

- ~~Confirm `baseline` list-vs-single~~ — RESOLVED: normalized `baseline` is a
  list synthesized from `conditions.baseline`; canonical authoring form is the
  `conditions:` block. See §6.1.
- ~~`model_settings` folding target~~ — RESOLVED: stays under
  `conditions.model_settings`; per-model key stays `params`. See §6.1.
- Decide whether the interim guard is `lint-workspace.py`-only or also a pytest in
  v2ecoli's CI (the tiered CI gate proper is Phase 3) — decided in the plan.
- Phase 2 detail (typed-node resolver API, per-node compute contracts, endpoint
  delegation) is designed in its own spec; §4 is the model it must satisfy.
