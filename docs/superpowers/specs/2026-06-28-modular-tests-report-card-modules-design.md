# Modular Study "Tests" — report-card test modules — Design

**Status:** spec (approved in brainstorm 2026-06-28). Cross-repo:
`vivarium-dashboard` (consumer) + `v2ecoli` (producer). This is the
generalization the comparison↔investigation unification was aiming at: a study's
**Tests section becomes a modular list of evaluation modules**, with the current
Behavioral Tests as the default kind and **report cards as a pluggable kind**.
The comparison investigation declares report-card modules in place of behavioral
tests.

## Goal

Let a study declare what it needs for its "Tests" section as a modular list of
evaluation modules. The default module (`behavioral`) renders exactly as today.
A study can **replace or add** modules; `report_card` modules — each of which
knows how to process a composite's run data — render an embedded card + verdict.
The v2ecoli↔vEcoli comparison studies use `report_card` modules (config / parca /
standard / statistical) as their Tests.

## Scope & decomposition

Two subsystems, one small contract:

1. **Dashboard framework (vivarium-dashboard, consumer):** the modular,
   `kind`-dispatched Tests section. Reusable by ANY study.
2. **Comparison producer (v2ecoli):** emit each study's cards to
   `viz/report_card/`, declare the `report_card` modules, and a scaffold.

**Contract (the only coupling):**
- the test-module `kind` discriminator in the study's tests list, and
- the artifact convention `studies/<name>/viz/report_card/<card>.html` +
  `studies/<name>/viz/report_card/<card>.verdict.json`, which the dashboard
  ALREADY auto-discovers (`saved_visualizations.py`).

The implementation **plan** splits by repo; this spec defines the shared
contract so both sides agree.

## Decisions (brainstorm 2026-06-28)

1. **Report-card modules are pre-rendered by the producer** (harness), not run
   live by the dashboard. The dashboard embeds the HTML + reads the verdict
   sidecar. Reuses existing auto-discovery; the dashboard stays a thin renderer.
2. **One modular tests list, each entry tagged by `kind`.** No `kind` (or
   `kind: behavioral`) = today's Behavioral Test, unchanged. `kind: report_card`
   = a report-card module. Behavioral and report-card modules **mix freely** in
   one Tests section.
3. **A report-card module's outcome = its `<card>.verdict.json` `overall`**,
   mapped within_tol→PASS, drift→PARTIAL, mismatch→FAIL, ungraded→PENDING.
4. **Comparison studies keep a real `baseline:`** (the `v2ecoli-baseline`
   composite for the condition) — it is the genuine v2ecoli side and keeps the
   study re-runnable; no dashboard-schema relaxation needed.

## Test-module schema (the contract)

A study's Tests section is the union of `tests:` (v4, preferred) and the legacy
`behavior_tests:` (both already recognized by the dashboard). Each entry:

```yaml
tests:
# behavioral module — kind absent or 'behavioral'; renders EXACTLY as today
- name: rna-mass-in-band
  measure: {kind: listener_path, path: bulk#RNA, ...}
  pass_if: {op: within, ...}
# report_card module — renders an embedded card + verdict
- name: standard-vs-vecoli
  kind: report_card
  card: standard          # -> viz/report_card/standard.html + standard.verdict.json
  classification: primary # optional, as today
```

- `kind`: `behavioral` (default) | `report_card`.
- `report_card` entries require `card:` (the module/card name); its rendered
  artifact + verdict live at `viz/report_card/<card>.{html,verdict.json}`.
- Existing `behavior_tests:` studies are byte-for-byte unaffected (no `kind`).

## Dashboard framework (consumer)

### Payload
The study-detail payload is already pass-through (`StudyDetail`, `extra="allow"`),
so `kind`/`card` reach the SPA untouched. The Tests collector
(`study_spec.py`) already scans both `tests` and `behavior_tests`; no payload
change beyond ensuring `kind`/`card` survive (they do).

### Render dispatch
- `static/study-detail.html` (the Tests loop, ~1771-1836) and
  `static/study-detail.js::loadTestsTab` (~839-963) gain a **switch on the
  entry's `kind`**:
  - `behavioral` → the existing pill + measure-assertion block (unchanged).
  - `report_card` → an embedded `<iframe>` of `viz/report_card/<card>.html`
    (reuse `walkthrough.js::_renderReportCardCard`'s embed markup) + a verdict
    pill from the sidecar.
- This is the ONLY structural dashboard change: a `kind` branch where today the
  loop renders every entry as a pill.

### Verdict / gate
- `compute_outcomes` / the evaluator seam gains a `report_card` branch: for a
  `kind: report_card` test, read `viz/report_card/<card>.verdict.json`'s
  `overall` and map to a result (within_tol→PASS, drift→PARTIAL, mismatch→FAIL,
  else PENDING). No live data crunching — the producer pre-rendered it.
- Behavioral modules evaluate exactly as today (unchanged).

### Reuse (do NOT rebuild)
- Report-card auto-discovery + iframe embedding already exist
  (`saved_visualizations.py`, `walkthrough.js`); the change moves the embed into
  the Tests section keyed by the declared module, instead of a hardcoded
  Visualizations-tab block.

## Comparison producer (v2ecoli)

### Card emission
- `scripts/comparison_report_card.py::assemble_from_studies` (or a new
  per-study writer it calls) writes, per study, per assigned card, a
  self-contained `studies/<study>/viz/report_card/<card>.html` plus a
  `<card>.verdict.json` (`{schema, overall, groups}`, the existing verdict
  shape). The combined `standardized_comparison_report.html` may remain as an
  index but is no longer the only artifact.
- Cards are the existing `scripts/_compare/report_cards/` registry
  (config/parca/standard/statistical) — each already "knows how to process the
  composite's data" (it builds from `per_obs` read off the zarr stores). The
  registry stays in v2ecoli; the dashboard never imports it (decision 1).

### Materialize the tests
- `scripts/_compare/materialize.py` declares each study's `tests:` as
  `report_card` modules — one per assigned card (graded AND ungraded; config/
  parca render as informational cards too) — replacing the `report_card_axis`
  `behavior_tests`. Keeps the real `baseline:` (decision 4), `pipeline_gate`,
  and the canonical run.
- The per-test gate outcome now comes from the dashboard's `report_card` branch
  reading the sidecar, so the materializer no longer needs to pre-write
  `runs[].outcomes` (the dashboard derives them). It MAY still write them for a
  static (non-dashboard) view.

### Scaffold
- A `v2e-compare scaffold <investigation>` verb (or `run --scaffold-only`) that
  stands up the investigation + one study per configuration with its
  `report_card` modules declared and a placeholder `viz/report_card/` — so the
  structure exists before a run produces the card artifacts. (The existing
  `materialize_study` is the per-study half; scaffold is the investigation-level
  wrapper.)

## Data flow

```
v2e-compare run <investigation>
  -> per study: run both engines -> zarr stores under out/<name>
  -> assemble_from_studies: per card, render viz/report_card/<card>.html
                            + <card>.verdict.json
  -> materialize_study: study.yaml tests: [{kind: report_card, card: <c>}, ...]
                        + baseline: [v2ecoli-baseline]
Dashboard study-detail:
  -> payload passes tests[] through (kind/card intact)
  -> loadTestsTab dispatches by kind:
       behavioral -> pill (today)
       report_card -> iframe(viz/report_card/<card>.html) + verdict(sidecar.overall)
  -> compute_outcomes report_card branch -> gate from sidecar overall
```

## Error handling

- Missing `viz/report_card/<card>.html` or sidecar → the module renders as
  `pending`/`ungraded` (no crash), with a one-line note; the study still loads.
- A `report_card` entry without `card:` → a clear payload-time warning, rendered
  as an error chip, not a 500.
- All reads/writes `encoding="utf-8"`.
- A behavioral-only study (no `kind` anywhere) renders byte-identically to today
  (explicit regression guard).

## Testing

**Dashboard (vivarium-dashboard):**
- payload: a study with mixed `tests` (one behavioral + one report_card) passes
  `kind`/`card` through to the payload.
- render dispatch (JS/template unit or DOM test): behavioral → pill;
  report_card → iframe with the right `src` + verdict pill; mixed list renders
  both; a behavioral-only study is unchanged (snapshot/regression).
- evaluator: `compute_outcomes` report_card branch maps sidecar
  within_tol/drift/mismatch → PASS/PARTIAL/FAIL; missing sidecar → PENDING.

**Producer (v2ecoli):**
- `assemble_from_studies` writes `viz/report_card/<card>.{html,verdict.json}`
  per study per card; HTML is non-empty; verdict matches the card's overall.
- `materialize_study` emits `tests:` as `report_card` modules matching the
  study's cards (one per card), keeps `baseline:`/`pipeline_gate`.
- scaffold stands up the investigation + per-config studies with report_card
  modules declared; idempotent.
- integration: render from existing stores → each study dir has
  `viz/report_card/<card>.html` + sidecar, and its `tests:` reference them.

## Out of scope (YAGNI)

- The dashboard running card modules live (decision 1).
- Moving the report-card registry out of v2ecoli (the dashboard never imports it).
- Relaxing the dashboard's `baseline:` requirement (decision 4 keeps a real one).
- New card kinds beyond the existing config/parca/standard/statistical.
