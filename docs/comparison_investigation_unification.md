# Design note: unify the comparison harness with the investigation structure

**Status:** design sketch (not yet specced/planned). 2026-06-27.

**Goal:** Make the v2ecoli↔vEcoli comparison a first-class **dashboard
investigation** — each condition a **study**, the comparison **report cards** its
**gating tests** — so the comparison shows up natively in the dashboard's
investigations view as a set of report cards with pass/drift/fail status.

## Why this is tractable

The bridge already exists. `pbg_v2ecoli/evaluators.py::evaluate_report_card_group`
(the registered `report_card_axis` evaluator) already turns a per-study
`report_card_verdict.json` into a study gating verdict, and the dashboard already
renders studies with their `report_cards:` + verdicts (see
`workspace/investigations/ketchup-baseline-comparison/.../study.yaml` and the
`beulig_batch` studies). So unification is mostly **data-shape + generation**, not
new framework code.

## The mapping

| Comparison harness (PR #301) | Investigation structure |
| --- | --- |
| manifest (`comparison_spec.json`) | an **investigation** (`investigation.yaml`) |
| each condition (basal, with_aa, …) | a **study** (`studies/<condition>/study.yaml`) |
| `scripts/run_comparison.py` (both engines per config) | the study's **canonical runs** |
| per-config report cards (`config`/`parca`/`standard`/`statistical`) | the study's `report_cards:` + `report_card_verdict.json` |
| card verdicts (within_tol / drift / mismatch) | the study's **gating tests** (`behavior_tests` via `report_card_axis`) |
| `standardized_comparison_report.html` | the dashboard **investigation view** |

## Phases

1. **Per-condition verdict emission.** The renderer
   (`scripts/comparison_report_card.py` / `run_comparison.py`) writes, per
   condition, `docs/report_cards/v2ecoli-vecoli-comparison/<condition>/` with the
   card HTML(s) **plus a `report_card_verdict.json`**. The `statistical` /
   `report_card_section.build_report_card` path already produces `verdict_json` —
   route it per condition; add a verdict to the `standard` card from the
   matched-time grading.

2. **Manifest → investigation/study generator** (new
   `scripts/comparison_to_investigation.py`, or extend `run_comparison.py`): from
   the manifest, emit `investigation.yaml` + one `study.yaml` per condition with
   `claim` ("v2ecoli reproduces vEcoli on `<condition>`"), `runs` (the two
   engines), `report_cards:` (that condition's cards), and `behavior_tests` using
   `report_card_axis` pointing at the condition's `report_card_verdict.json`.

3. **Gating wires itself** — the existing `report_card_axis` evaluator reads each
   `report_card_verdict.json` → the study's `gate_status`; the dashboard renders
   the investigation → per-condition studies → report cards + pass/drift/fail
   pills.

4. **Verify dashboard pickup** — add a root symlink if the nested
   `workspace/investigations/` layout needs it (known scanner gap; see
   `reference_dashboard_isetlist_root_layout_gap`).

## Open decisions (settle in a brainstorm before planning)

- One gate per condition (the `statistical` card's verdict) vs one test per
  card or per axis?
- Conditions as independent studies vs a pipeline DAG (basal gates the rest)?
  — leaning **independent** (each condition is a standalone comparison).
- Manifest **generates** the studies (single source of truth) vs studies
  reference the manifest? — leaning **generate**.
- Does `run_comparison.py` become the study's canonical-run command, so the
  dashboard "re-run study" re-runs the comparison for that condition?

## Next step

Formalize via brainstorming → spec → writing-plans. The heavy lifting is the
manifest→investigation/study generator + per-condition `report_card_verdict.json`
emission; the gating + dashboard rendering reuse existing machinery.

Related: the modular comparison harness ([PR #301], `comparison_spec.json` +
`scripts/_compare/report_cards/` + `run_comparison.py`).
