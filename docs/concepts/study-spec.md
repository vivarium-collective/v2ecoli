# Study spec — the canonical template

Every `studies/<slug>/study.yaml` follows this template. The template
replaces an earlier narrative spec (Question / Hypothesis / Objective /
Background / Predicted behavior / Variants / Interventions / Gaps /
Expert Questions / References) with **eight explicit sections** that
mirror the lifecycle of a model-building study.

The driving change: each study is a **model-building gate**. It
answers *"Can we safely build the next layer?"* — not just "what do we
want to know about this system?"

## Section overview

| # | Section | Purpose |
|---|---|---|
| 1 | `purpose:` | One-paragraph triple: question · mechanism · expected outcome. |
| 2 | `pipeline_gate:` | Prerequisites · enables · proceed_condition. |
| 3 | `simulation_set:` | Concrete run plans (replaces variants + interventions). |
| 4 | `model_change:` | What CODE this study adds / modifies. |
|   | `key_assumptions:` | Short biological context. |
| 5 | `readouts:` + `behavior_tests:` | What we collect + how we judge it. |
| 6 | `conclusion_logic:` | If primary tests pass / fail, what follows? |
| 7 | `limitations:` | What this study explicitly does NOT validate. |
| 8 | `implementation_requirements:` | Actionable items that turn into tasks. |

Plus a `bibliography:` block for cited papers, and dashboard-back-compat
shims for `baseline:` (will be removed when the renderer follows
`simulation_set:`).

---

## 1. `purpose:` — Replaces Q + H + Objective

```yaml
purpose:
  question: |
    Biological / modeling question. ONE PARAGRAPH.
  mechanism: |
    What the model is doing in this study (NEW code, or existing
    chain being certified). ONE PARAGRAPH.
  expected_outcome: |
    What success looks like, with quantitative thresholds where
    possible. ONE PARAGRAPH.
```

The three sub-fields used to live as separate top-level `question:`,
`hypothesis:`, `objective:` — they overlapped. Now grouped under
`purpose:` so the reader sees them as ONE statement of intent.

## 2. `pipeline_gate:` — Each study is a gate, not just a question

```yaml
pipeline_gate:
  prerequisites:                # other studies that must reach `tests-passed`
    - dnaa-02-atp-hydrolysis    # before this one starts
  enables:                      # downstream studies this unblocks
    - dnaa-04-initiation-mechanism
  proceed_condition: |
    What pass condition lets the pipeline proceed past this study.
  blocks_until_resolved: |
    What happens if primary tests fail — which downstream studies
    must NOT start.
```

Supersedes `parent_studies:` (which only captured prerequisites).
The dashboard's DAG view consumes `pipeline_gate.prerequisites` +
`enables` instead.

## 3. `simulation_set:` — One unified run-plan list (replaces variants + interventions)

```yaml
simulation_set:
- name: <slug>
  base_model: <composite ref>
  perturbation: null | {<param>: <value>, ...} | [<sweep values>, ...]
  condition: M9-minimal-glucose | LB | custom
  duration_min: 60
  seeds: [0, 1, 2, 3, 4]
  readouts: [<readout-name>, ...]
  applies_tests: [<behavior-test-name>, ...]
  status: ready | gated
  blocked_by_requirements: [req-1, ...]
```

The old `variants:` block described *what to perturb*; the old
`interventions:` block described *how to run it*. They overlapped 1:1
in practice (every variant had a matching intervention). Now ONE
section captures both. A perturbation is a dict (single config) or a
list of dicts (a sweep).

`applies_tests:` is the explicit mapping from this run to the behavior
tests it exercises — replaces the `triggers_tests:` field on
interventions.

## 4. `model_change:` — Implementation, separated from biology

```yaml
model_change:
  base_model: <composite ref>
  new_processes:        []      # Process or Step classes added
  new_state_variables:  []      # bulk ids, unique stores, etc.
  new_parameters:       []      # composite-level knobs
  modified_processes:   []      # processes whose config / wiring changes
  new_listeners:                # listeners + emit paths
    - name: <name>
      emits: [<store path>, ...]
      status: optional | required
      requirement_id: <req id>
  notes: |
    Short comment about the scope of changes.
```

The old `Background` section conflated biology and implementation. The
new `model_change:` strictly lists what code lands. Long biological
explanation lives separately as a Markdown `references/notes/<bib>.md`
file or in a brief `key_assumptions:` list.

## 4b. `key_assumptions:` — Short biological context

```yaml
key_assumptions:
  - "Steady-state slow growth (M9-glucose; ~60 min doubling)."
  - "Total DnaA pool concept; nucleotide-state split is dnaa-02."
  - "..."
```

5-10 bullets max. If you need paragraphs, write them in a study-local
README; the YAML's job is to be machine-readable.

## 5. `readouts:` + `behavior_tests:`

`readouts:` declares the *quantities collected* from a simulation.
`behavior_tests:` declares the *pass/fail rules* applied to readouts.

```yaml
readouts:
- name: <name>
  description: |
    What the readout measures + how it resolves against the store.
  store_path: agents.0.listeners.monomer_counts.monomerCounts
  index_by: {type: monomer_id, value: "PD03831[c]"}
  units: molecules/cell
  status: available | derived-needed | aspirational
  blocked_by_requirements: [<req id>, ...]

behavior_tests:
- name: <slug>
  classification: primary | supporting | diagnostic | regression
  description: |
    What this test checks, in english.
  measure:
    kind: monomer_count | rna_count | bulk_count | listener_indexed |
          listener_sum | listener_path | xy_correlation | event_count_per_window
    # ...kind-specific args
    reduce: median | mean | series | first_and_last | rolling_cv | ...
    window: full | second_half | post_initiation_10min | ...
  pass_if:
    op: in_range | at_most | at_least | ratio_at_most | ratio_at_least |
        rolling_cv_at_most | pearson_at_most | monotonic_decreasing | ...
    # ...op-specific args
  units: <units of the measure>
  requires_simulation: <simulation_set name>
  blocked_by_requirements: [<req id>, ...]
  calibration_anchor:                # ← optional, for thresholds vs literature
    v2ecoli_observed: <auto-populated by preview-viz>
    literature_target: <value>
    divergence_alarm_factor: 10
    notes: |
      Free-text explaining why the threshold and the simulator might
      disagree (concept mismatch, calibration issue, ...).
  cites: [<bib-key>, ...]
```

**Classification rubric**:
- `primary` — must pass for the study to be considered successful.
  Maps to the investigation's acceptance criteria.
- `supporting` — corroborates the primary results. Failure flags the
  primary results as suspect but doesn't block on its own.
- `diagnostic` — for a specific variant or perturbation. Failure does
  not block the primary result.
- `regression` — guards against future changes. Should never fail
  unless something upstream broke.

The dashboard's "Tests" tab groups by classification and shows
primary results at the top.

## 6. `conclusion_logic:` — Make the if/then explicit

```yaml
conclusion_logic:
  if_primary_tests_pass:
    implementation_status: |
      What we now know about the IMPLEMENTATION (the code is correct).
    biological_validation: |
      What we now know about the BIOLOGY (the model matches reality).
    pipeline_unblocks:
      - <downstream study>
  if_primary_tests_fail:
    diagnose:
      - <heuristic for which test → likely cause>
    block_downstream: |
      Which studies must NOT start.
```

The key distinction: **implementation success ≠ biological
validation**. Passing tests means the code does what the spec says.
Whether the code's predictions match physical reality is a separate
question — captured in `biological_validation:`.

## 7. `limitations:` — Explicit "this study does not validate X"

```yaml
limitations:
  - "Single doubling time; multi-generation drift not validated here."
  - "Total DnaA pool only; no nucleotide-state split (dnaa-02)."
  - "..."
```

Prevents overclaim from a passing test. The Generate-report flow
should display this prominently next to the test results so a domain
expert reading the report doesn't extrapolate beyond what was
actually tested.

## 8. `implementation_requirements:` — Concrete task generators

```yaml
implementation_requirements:
- id: req-1-<slug>
  kind: process | step | listener | parameter_hook | validation_data | reference_data
  title: <human readable>
  effort: XS | S | M | L | XL
  description: |
    What needs to exist for this requirement to be satisfied.
  steps:
    - "First touch step."
    - "Second touch step."
    - "..."
  defer_until: <downstream study where this naturally lands>     # optional
  unblocks:
    - <simulation_set name or behavior_test name>
```

Each entry is a unit of work the dev team can claim. The dashboard
should be able to convert these into pbg-superpowers tasks
automatically (`/pbg-study generate-tasks <slug>`).

---

## Validation lifecycle

```
planned                              (spec exists)
  ↓ /pbg-study verify <slug>
spec-validated                       (all readouts resolve, all
                                      simulation_set hooks exist or
                                      are gated)
  ↓ /pbg-study preview-viz <slug>
viz-previewed                        (each readout produces a sane
                                      sample plot)
  ↓ expert review
ready                                (calibration anchors annotated;
                                      limitations acknowledged)
  ↓ /pbg-study run-baseline <slug>
running → ran → tests-passed → complete
```

Validation tooling (proposed for the listening Claude in
`docs/superpowers/notes/2026-05-16-investigation-walkthrough.md`):

- `/pbg-study verify` — mechanical resolution of every identifier
  and reference. Catches wrong-id bugs before runtime.
- `/pbg-study preview-viz` — renders each readout + behavior test
  against the cached baseline so the expert sanity-checks shapes.
- `calibration_anchor` field above — surfaces v2ecoli-vs-literature
  divergences as questions, not failures.

## Migration from the old narrative template

| Old field | New field |
|---|---|
| `question` (top-level) | `purpose.question` |
| `hypothesis` (top-level) | `purpose.expected_outcome` (or split into mechanism) |
| `objective` (top-level) | `purpose.mechanism` (where it spoke to implementation) |
| `description` (long-form, with biology + assumptions mixed) | Split into `model_change.notes` + `key_assumptions` |
| `variants:` | Folded into `simulation_set[i].perturbation` |
| `interventions:` | Folded into `simulation_set[i]` directly |
| `observables:` | `readouts:` (renamed for clarity) |
| `expected_behavior:` | `behavior_tests:` (renamed; same DSL grammar) |
| `gaps:` | `implementation_requirements:` (renamed, with stricter shape) |
| `parent_studies:` | `pipeline_gate.prerequisites` (and `enables` is new) |
| `expert_questions:` | Folded into `limitations:` (when the question is about scope) or `behavior_tests[].calibration_anchor.notes` (when it's about thresholds) |
| `references` (list of bib keys) | `bibliography.bib_keys` (renamed; `references:` is v4-reserved by the dashboard) |

The dashboard's v4 reserved field names (`tests:` dict shape,
`references:` list of file refs, `implementation_tasks:` string) are
still respected as compatibility shims — the per-study `tests:` block
at the bottom of the YAML carries the auto_discover config.

## Example

See `studies/dnaa-01-expression-dynamics/study.yaml` for a complete
example. Notes on its structure:

- `purpose.question` is one sentence; the verbose biological context
  moved to `references/notes/Schmidt2016NatBiotechnol.md` etc.
- `simulation_set` has 4 entries: one `ready` (the baseline) and
  three `gated` (each blocked by a specific `implementation_requirement`).
- `behavior_tests` has 6 entries, classified `primary` (2),
  `supporting` (1), `diagnostic` (3).
- `behavior_tests[0].calibration_anchor` carries the Phase 1
  walkthrough finding that v2ecoli's monomer_counts[DnaA] = 256k vs
  the literature 300-800 band — the divergence is documented as an
  expert resolution question rather than a hard failure.
- `implementation_requirements` lists 6 concrete pieces of work, each
  with effort estimate and `unblocks:` targets.
