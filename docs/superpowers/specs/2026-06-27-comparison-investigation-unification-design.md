# Comparison ↔ Investigation Unification — Design

**Status:** spec (approved in brainstorm 2026-06-27). Supersedes the sketch in
`docs/comparison_investigation_unification.md` where they differ (this spec is
authoritative).

**Goal:** Make the v2ecoli↔vEcoli comparison harness (PR #301) a first-class
dashboard **investigation** — each condition a **study**, each comparison
**report card** a **gating test** — so the comparison renders natively in the
dashboard's investigations view with pass/drift/fail pills, reusing the existing
`report_card_axis` evaluator with no framework change.

## Settled decisions (brainstorm 2026-06-27)

1. **Gate granularity: one gating test per report card** (the card's *overall*
   verdict). Not per internal axis, not one per condition.
2. **Topology: independent studies.** Each condition study has empty
   `pipeline_gate.prerequisites`; the investigation gate aggregates them.
3. **Direction: studies reference the manifest.** `study.yaml` files are
   hand-authored, persistent artifacts carrying `comparison_manifest` +
   `condition`; cards/run-shape resolve from the manifest at run time. A
   **validator** (not a generator) guards against drift.
4. **Bootstrap: one-time scaffold + hand-own.** A scaffold script writes the
   initial `study.yaml` skeletons once; thereafter they are hand-authored.
5. **Verdict emission: both render and run paths.** Both
   `comparison_report_card.assemble_from_manifest` (render-only) and
   `run_comparison.py` (full run) write `report_card_verdict.json`, so a gating
   verdict exists after either path.

## The verdict mapping (why this needs no evaluator change)

The registered `report_card_axis` evaluator
(`pbg_v2ecoli/evaluators.py::evaluate_report_card_group`) reads
`<card_dir>/report_card_verdict.json`, looks up `groups[<group>].axes`, and
returns the worst-of-axes verdict (`mismatch → FAIL`, `drift → PASS+caveat`,
`within_tol → PASS`, else `ungraded`). We map onto it directly:

| Unification concept | report_card_verdict.json element |
| --- | --- |
| a **condition** (basal, with_aa, basal_4x4) | the **card dir**: `docs/report_cards/v2ecoli-vecoli-comparison/<condition>/` |
| a **report card** (standard, statistical) | a **group** in that JSON |
| a graded card's internal axes | the group's `axes` (worst-of = card overall) |
| `config`/`parca` cards | an **ungraded** group (renders, never gates) |
| one **gating test** per graded card | one `behavior_test` `{kind: report_card_axis, card: <condition dir>, group: <card_name>}` |

One verdict JSON per condition; one group per card; one behavior_test per graded
card. Study `gate_status` = worst across its card-groups (existing aggregation).

## Architecture

```
comparison_spec.json  (single source of truth: repos, configs, cards, run-shape)
        │  referenced by
        ▼
workspace/investigations/v2ecoli-vecoli-comparison/
  investigation.yaml                         (hand-authored; lists studies)
  studies/<condition>/study.yaml             (hand-authored; references manifest)
        │  behavior_tests point at
        ▼
docs/report_cards/v2ecoli-vecoli-comparison/<condition>/
  index.html, ...                            (card HTML, existing)
  report_card_verdict.json                   (NEW: groups = cards)
        ▲  written by both
        │
  comparison_report_card.assemble_from_manifest()   (render path)
  run_comparison.py                                 (run path)
```

The dashboard scanner reads `study.yaml` (`report_cards:`, `behavior_tests:`,
`runs:`) and the evaluator reads the verdict JSON. Neither needs to know about
the manifest; the manifest is consumed only by `run_comparison.py` and the
validator/scaffold.

## Components

### 1. Verdict emission (`scripts/_compare/verdict.py`, new)

A single function that takes the assembled condition sections (each card's
in-memory verdict data) and writes one `report_card_verdict.json`:

```
build_condition_verdict(condition, cards) -> dict   # schema report_card_verdict/v1
write_condition_verdict(out_dir, condition, cards) -> Path
```

- `statistical` card: its `build_report_card` already computes a full `vjson`
  (internal groups growth/mass/rna). Capture it (currently discarded in
  `statistical.py`) and flatten its internal axes into the `statistical` group's
  `axes`; group verdict = `vjson["overall"]`.
- `standard` card: surface a verdict from `eval_section`'s matched-time grading
  (currently emits none) into the `standard` group's `axes`.
- `config` / `parca`: emit a group with `verdict: ungraded`, no axes.
- top-level `overall` = worst across groups (reuse the `_RANK`/severity order
  already in `report_card.py` / the evaluator).

To carry the per-card verdict data out of the cards, `CardContext`/`Section`
gains an optional `verdict` payload (the `statistical` Section already returns
`verdict`; extend `standard` to return its graded axes). The assembler collects
these and calls `write_condition_verdict`.

Wired into **both** `assemble_from_manifest` (render-only) and
`run_comparison.py` (after a full run) so the verdict is refreshed on either
path.

### 2. Authored studies + investigation (hand-owned)

`workspace/investigations/v2ecoli-vecoli-comparison/investigation.yaml` and one
`studies/<condition>/study.yaml` per manifest config. Each study:

```yaml
schema_version: 4
name: <condition>
investigation: v2ecoli-vecoli-comparison
comparison_manifest: comparison_spec.json     # the reference
condition: <condition>                          # manifest config (name if disambiguated)
report_cards:
- docs/report_cards/v2ecoli-vecoli-comparison/<condition>/index.html
behavior_tests:
- name: <card>-vs-vecoli                        # one per graded card
  measure:
    kind: report_card_axis
    card: docs/report_cards/v2ecoli-vecoli-comparison/<condition>
    group: <card>
runs:
- name: <condition>-comparison
  kind: analysis
  canonical: true
  description: "run_comparison.py <manifest> for condition <condition>"
pipeline_gate: {prerequisites: [], enables: []}
```

### 3. Scaffold (`scripts/scaffold_comparison_studies.py`, new, one-time)

Reads the manifest, writes initial `investigation.yaml` + per-condition
`study.yaml` skeletons (idempotent: refuses to overwrite an existing study
unless `--force`). After scaffolding, the files are hand-owned; the scaffold is
not part of the run/render loop.

### 4. Validator (`scripts/validate_comparison_studies.py`, new)

Asserts, for the investigation:
- every study's `condition` exists as a manifest config (by `config` stem or
  `name`);
- every study's `behavior_tests` groups exactly match the manifest's assigned
  cards for that condition (graded cards only);
- each referenced `card` dir path is well-formed.

Exits non-zero on drift (run in CI / pre-merge). This is how "studies reference
the manifest" stays honest without auto-generating.

### 5. Dashboard pickup (Phase 4)

Nested `workspace/investigations/` layouts need a root symlink for the
list/sidebar scanner (known gap, `reference_dashboard_isetlist_root_layout_gap`).
Add the symlink; verify the investigation renders with per-condition studies and
card-group pills.

## Data flow

1. `run_comparison.py <manifest>` (or `--render-only`) runs/loads both engines
   per condition, assembles cards, and writes
   `docs/report_cards/v2ecoli-vecoli-comparison/<condition>/report_card_verdict.json`.
2. The dashboard scans `workspace/investigations/v2ecoli-vecoli-comparison/`,
   loads each `study.yaml`, and for each `report_card_axis` behavior_test calls
   the evaluator, which reads the verdict JSON's `groups[<card>]`.
3. Study `gate_status` = worst card-group; investigation aggregates studies.

## Error handling

- **Missing verdict JSON:** evaluator already returns `ungraded` (not a crash).
- **Group absent in card:** evaluator already returns `ungraded`.
- **Study/manifest drift:** the validator fails loudly before the dashboard
  silently shows `ungraded`.
- **Unicode:** all verdict/manifest/study reads+writes use `encoding="utf-8"`
  (CI ASCII-locale guard, per the PR #301 lesson).

## Testing

- `tests/test_comparison_verdict.py`: `build_condition_verdict` groups shape;
  `overall` = worst across groups; statistical card flatten; standard card
  graded axes present; config/parca → ungraded.
- `tests/test_validate_comparison_studies.py`: passes on a matching
  study+manifest; fails on a renamed condition, on a card-group mismatch, on a
  malformed card path.
- `tests/test_scaffold_comparison_studies.py`: scaffold writes expected
  skeletons; refuses overwrite without `--force`.
- Integration: render from an existing zarr store (the `out/smoke5` 4×4×5
  mediafix stores) → assert each condition dir has a verdict JSON whose groups
  match its assigned cards, and the evaluator returns a non-`ungraded` verdict
  for the graded card.

## Out of scope (YAGNI)

- No DAG/prerequisites between conditions (decision 2).
- No auto-generation of studies on every run (decision 3); scaffold is one-time.
- No new dashboard evaluator or renderer code (the bridge already exists).
- No change to which cards exist or how they grade (PR #301 owns that).

## Phasing

- **Phase 1:** verdict emission (`verdict.py` + card verdict payloads + wire into
  both paths) + tests. Independently testable (verdict JSON appears, graded).
- **Phase 2:** scaffold + authored studies + investigation.yaml + validator +
  tests.
- **Phase 3:** gating — no code; confirmed by the evaluator reading the verdicts.
- **Phase 4:** dashboard root symlink + manual render verification.
