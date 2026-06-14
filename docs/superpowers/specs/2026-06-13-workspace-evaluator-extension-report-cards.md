# Design: workspace-pluggable evaluators — report cards as first-class acceptance evidence

**Date:** 2026-06-13
**Status:** approved design, pre-implementation
**Repos touched:** `pbg-superpowers` (small generic seam), `v2ecoli` / `pbg_v2ecoli` (the first consumer)

## Problem

A behavioral **report card** (PR #134: `population_phenotype_basal` in its `vs_vecoli`
equivalence mode) grades a model across 21 typed-criteria axes (Welch t-test / R² /
flux-scatter) earning per-axis verdicts `within_tol` / `drift` / `mismatch` / `ungraded`.
Today that grading dead-ends in a self-contained HTML artifact. The investigation/study
**evaluation spine** never sees it:

- `study_evaluator.compute_outcomes` turns a study's `tests[]` into `computed_outcomes`
  only for a closed set of native `measure.kind` values; anything else is bucketed
  `evaluated_by: agent`.
- `study_verdict.roll_up_verdict` rolls per-test outcomes into `gate_status`.
- `investigation_status.roll_up_acceptance` rolls study outcomes into investigation
  acceptance criteria.

Concretely, `showcase-6-equivalence-large`'s five group-level tests are all
`evaluated_by: agent` — the card *does* the rigorous grading, but the spine treats the
verdicts as hand-asserted prose, not machine evidence. We want the card's verdicts to
**flow into `computed_outcomes` → gate → acceptance criteria**, and we want this added in
a way that **augments the framework from within the workspace** rather than baking a
v2ecoli-specific concept into shared `pbg-superpowers` core — mirroring the existing
`build_core()` workspace-local registration pattern.

## Decisions (from brainstorming)

1. **Attach point:** the card grades a study **test** (populates `computed_outcomes` via a
   workspace-registered evaluator); the existing `acceptance_criteria → study-outcome`
   roll-up then works unchanged. The AC layer is *not* modified.
2. **Extension API:** a **generic** custom-evaluator registry. The framework knows nothing
   about report cards; it exposes a pluggable seam and the workspace registers a
   `report_card_axis` evaluator. Any future workspace evaluator plugs in the same way.
3. **Grain:** **per group** — each study test maps to one of the card's 5 groups
   (Physiology / Composition / Ribosomes / Exchange fluxes / Gene expression). Matches the
   five tests `showcase-6` already carries.
4. **Wiring:** a dedicated evaluator-registry hook discovered exactly like `build_core()`
   (workspace `pbg_<slug>` package). Not `importlib.metadata` entry points (the ecosystem
   deliberately uses `build_core()` registration, not metadata discovery). Not a
   process-bigraph `Step`/`Analysis` (that conflates evaluation with the composite runtime).

## Architecture

```
sim runs ─► sweep parquet ─► render report card ──► report_card.html        (human view)
                                              └────► report_card_verdict.json (machine view)
                                                                │
study_outcomes.sync(study_dir):                                 │
  record_runs → compute_outcomes ───────────────────────────────┤
     evaluate_test(kind=report_card_axis)                        │
        └─ load_workspace_evaluators(ws_root)                    │
             └─ pbg_<slug>.register_evaluators(registry)         │
                  └─ evaluate_report_card_group(test,run,ws) ◄───┘  reads verdict.json[group]
                       └─ computed_outcome {result, evaluated_by: report_card, provenance}
  → roll_up_verdict (gate_status) → roll_up_acceptance (investigation AC)
```

The framework ships an **empty** registry; everything report-card-specific lives in
`pbg_v2ecoli/`. A workspace with no `register_evaluators` hook behaves exactly as today.

### Component 1 — framework seam (`pbg-superpowers/pbg_superpowers/study_evaluator.py`)

- Module-level `CUSTOM_EVALUATORS: dict[str, Callable]` — empty by default.
- `load_workspace_evaluators(ws_root) -> dict[str, Callable]`:
  - read `workspace.yaml` `name` → import `pbg_<name>` (same resolution the dashboard uses
    to find `build_core`);
  - if the package exposes `register_evaluators(registry: dict) -> None`, call it with a
    fresh dict and return it; else return `{}`.
  - Result cached per `ws_root`. Import / hook errors are caught and logged, returning `{}`
    (a broken workspace hook must never crash the evaluator — degrade to `agent`).
- In `evaluate_test(test, reader, ws_root)`: resolution order becomes
  1. native `measure.kind ∈ RUN_DATA_KINDS` → existing code evaluation;
  2. else `measure.kind ∈ load_workspace_evaluators(ws_root)` → call
     `evaluator(test, run, ws_root)` and use its returned outcome dict;
  3. else → existing `evaluated_by: agent` fallback.
- The evaluator's returned dict MUST carry `result` and `evaluated_by`; MAY carry
  `measured_value`, `detail`, `provenance`. `reconcile` against authored `outcomes[test]`
  is applied by the existing reconciliation code, unchanged.

**Scope of core change:** ~20 lines + the cached loader. Fully backward-compatible.

### Component 2 — workspace evaluator (`pbg_v2ecoli/evaluators.py`, new)

```python
def register_evaluators(registry: dict) -> None:
    registry["report_card_axis"] = evaluate_report_card_group

def evaluate_report_card_group(test, run, ws_root) -> dict:
    # measure: {kind: report_card_axis, card: <dir>, group: <name>}
    # load <ws_root>/<card>/report_card_verdict.json; read groups[group]; aggregate.
    ...
```

`evaluate_report_card_group` reads the verdict JSON (never parses HTML), looks up the named
group, applies the aggregation rule (Component 4), and returns:
```python
{"result": "PASS"|"FAIL"|"ungraded", "evaluated_by": "report_card",
 "detail": "...", "provenance": {"card": ..., "group": ...,
            "axis_verdicts": [...], "overall": ...}}
```
Missing file / missing group → `result: ungraded` with a reason (never raises).

### Component 3 — verdict emission (`reports/population_phenotype_basal_report.py`)

The renderer already computes per-axis verdicts to build the HTML. Add one side-effect:
write `<out-dir>/report_card_verdict.json` next to `report_card.html`.

Contract (`schema: "report_card_verdict/v1"`):
```json
{
  "schema": "report_card_verdict/v1",
  "model_ref": "bd2123d2", "reference_model": "vEcoli (v1)", "generated": "...",
  "overall": "mismatch",
  "groups": {
    "physiology": {"verdict": "within_tol",
      "axes": [{"id": "doubling_time", "verdict": "within_tol",
                "delta": -0.022, "p": 0.014, "d": -0.26}, ...]},
    "exchange_fluxes": {"verdict": "mismatch", "axes": [...]},
    ...
  }
}
```
Versioned `schema` so the framework seam and the v2ecoli reader evolve independently.

### Component 4 — group → outcome aggregation

| Group's axis verdicts | `computed_outcome.result` | rolls up as |
|---|---|---|
| any `mismatch` | **FAIL** | gate `failed` |
| ≥1 `drift`, no `mismatch` | **PASS** (+ `caveat: drift`) | `passing-with-caveats` |
| only `within_tol` (+ `ungraded`) | **PASS** | gate `passed` |
| all `ungraded` | **ungraded** (skip) | `needs_calibration` |

The group's own `verdict` field in the JSON is advisory; the spine outcome is computed from
the axis verdicts by this rule so the policy is explicit and lives in one place.

### Measure schema (study.yaml)

```yaml
- name: physiology-equivalent-to-vecoli
  classification: primary
  measure:
    kind: report_card_axis
    card: docs/report_cards/population_phenotype_basal/vs_vecoli   # dir holding verdict JSON
    group: physiology
  pass_if:
    op: report_card_group_within_tol
```

## Data flow & dashboard surfacing

`study_outcomes.sync(study_dir)` is unchanged as an entry point: `record_runs →
compute_outcomes → …`. Because the card-backed tests now resolve through the registry,
`sync` grades them automatically. **No new dashboard UI:** card-graded tests render as
ordinary `computed_outcomes` (with `evaluated_by: report_card` + the `provenance` axis
breakdown), and the HTML card remains the embedded iframe already wired into the study's
Visualizations tab. The investigation acceptance-criteria roll-up consumes them unchanged.

## Migration: showcase-6

Rewrite `showcase-6-equivalence-large`'s five tests from bare `evaluated_by: agent` to
`measure.kind: report_card_axis` (one per group). After `sync`, their `computed_outcomes`
become `evaluated_by: report_card` with `reconcile` against the authored outcomes, and the
gate/AC roll-up is computed from the card rather than hand-asserted. Expected results match
the rendered card: physiology/composition/ribosomes PASS, exchange_fluxes FAIL (O₂/CO₂
mismatch), gene_expression FAIL or PASS-with-caveat per the aggregation rule (transcriptome
`mismatch` ⇒ FAIL under "any mismatch → FAIL"; this corrects the hand-authored "partial").

## Testing

- **Framework seam (pbg-superpowers):** `load_workspace_evaluators` — hook present / absent
  / raises (→ `{}`). `evaluate_test` dispatch — registered kind vs. `agent` fallback, using
  a fixture workspace with a toy `register_evaluators` (no dependency on v2ecoli).
- **Workspace evaluator (pbg_v2ecoli):** `evaluate_report_card_group` against fixture
  `report_card_verdict.json`s covering each aggregation branch + missing file/group → skip.
- **Contract:** schema-validate `report_card_verdict/v1`; round-trip test that the
  renderer's emitted JSON matches what the evaluator reads.
- **End-to-end:** `sync` on showcase-6 → assert the 5 tests come back
  `evaluated_by: report_card` with the expected per-group results and matching gate/AC roll-up.

## Non-goals (YAGNI)

- No investigation-level "card → acceptance criterion directly" path (Section-1 decision:
  cards grade study tests; AC rolls up unchanged).
- No per-axis (21-test) or single-overall grain — per-group only.
- No generic multi-workspace registry beyond the `build_core`-style hook; report card is
  the only evaluator implemented now (others can register later via the same seam).
- No new dashboard UI.

## Open questions

- Exact `pbg_<slug>` resolution helper to reuse for the loader (whatever the dashboard
  already uses to locate `build_core`) — confirm at implementation time.
- Whether `report_card_group_within_tol` needs to be a registered `pass_if` op or whether
  the evaluator returns a fully-formed outcome and `pass_if` is advisory for card-backed
  tests — lean toward the latter (evaluator owns the verdict).
