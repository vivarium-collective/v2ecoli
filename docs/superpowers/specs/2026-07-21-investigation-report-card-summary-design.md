# Investigation Report-Card Summary — Design

**Date:** 2026-07-21
**Branch:** `feat/investigation-report-card-summary`
**Status:** Design (awaiting review → implementation plan)

## Problem

An investigation (e.g. `v2ecoli-vecoli-comparison`) groups several studies, each of
which produces one or more **report cards** — a rendered `viz/report_card/<card>.html`
plus a machine-readable `viz/report_card/<card>.verdict.json`
(`report_card_verdict/v1`). Today the only investigation-level views are the
workbench's **full** per-investigation report (`report_views.py`,
`single_study_report.py`) and the live dashboard SPA. There is no lightweight,
sendable, **high-level summary** that pulls every study's report cards together —
an overview up front, an at-a-glance verdict matrix, and easy access to each card.

The pre-investigation ancestor of this idea was `reports/compare_report.py` (removed
in PR #117), a self-contained HTML report with a Summary/verdict box, section
headers, an n-way divergence table, and iframe'd sub-views. We rework that scaffold
into a proper investigation-structured generator.

## Goal

A standalone Python generator that scans **any** investigation whose studies carry
`viz/report_card/*` artifacts and emits **one self-contained HTML page** summarizing
the report cards: overview section in front, a study × observable verdict matrix, and
per-study collapsible sections that inline-embed the rendered cards.

Non-goal: re-running simulations, changing the dashboard/workbench, or standing up a
server. Pure static generation over already-committed artifacts.

## Interface

```
python reports/investigation_summary.py --investigation <slug> [--out PATH] [--no-open]
  → reports/summaries/<slug>_summary.html   (auto-opens in browser unless --no-open)
```

- `--investigation <slug>` — required; resolves `workspace/investigations/<slug>/`.
- `--out PATH` — optional override of the default output path.
- Default output: `reports/summaries/<slug>_summary.html` (new folder, one file per
  investigation).
- Output is **fully self-contained**: all card HTML + styles inlined, so the file
  works when emailed or moved anywhere with no repo present.

## Data sources (read-only, no sims)

- **`workspace/investigations/<slug>/investigation.yaml`** → `title`, `question`,
  study order.
- Each **`study.yaml`** → per-study `title`/`name`, `status`, canonical run `result`
  (PASS / PARTIAL / FAIL), `pipeline_gate.prerequisites` (for the DAG), the
  `report_cards:` list, and top-line `findings[].statement`.
- Each **`viz/report_card/<card>.verdict.json`** (`report_card_verdict/v1`) →
  `overall` (`within_tol` / `drift` / `mismatch` / `ungraded`) and
  `groups[<g>].axes[]` with `label`, `verdict`, `value`, `meter`. Drives the roll-up
  counts and the observable matrix.
- Each rendered **`viz/report_card/<card>.html`** → embedded per study.

If a study declares a card in `report_cards:` whose `.html` or `.verdict.json` is
missing, the generator records it as a "missing card" placeholder rather than
crashing (renders a muted note in that study's section).

## Page structure

1. **Overview (front)**
   - Investigation title + question.
   - Roll-up strip: verdict-colored counts (e.g. `2 FAIL · 3 PARTIAL · 2 PASS`),
     derived from each study's canonical `result`.
   - Pipeline DAG rendered from `pipeline_gate.prerequisites`
     (`parca → {basal, with_aa, succinate, no_oxygen, acetate} → statistical`).
   - Sticky in-page nav linking to each study section.

2. **Verdict matrix**
   - Rows = studies (DAG order); columns = the **union of axis labels** across all
     studies' graded cards (`overall != ungraded`), ordered by first appearance in
     DAG order (so `standard`-card observables — `cell mass (fg)`, `dry mass (fg)`,
     `protein mass (fg)`, `RNA mass (fg)`, `growth rate` — lead, and any
     `parca`/`statistical`-only axes append after). Cross-card axes with identical
     labels collapse into one column.
   - Each cell colored `within_tol` / `drift` / `mismatch` from that study's matching
     axis `verdict`; rendered empty (muted) when the study has no axis with that
     label. Config cards (`ungraded`) contribute no columns.
   - Cells anchor-link to the owning study's section. This is the "everything at a
     glance" table (the n-way-divergence-table analog).

3. **Per-study sections** (ordered by the DAG)
   - One collapsible `<details>` per study. Header: study title + overall verdict
     badge + one-line finding statement.
   - Body inline-embeds each rendered report card. Config cards
     (`overall: ungraded`, config-diff) collapsed by default; graded cards open.

## Card embedding

Rendered cards are HTML **fragments** using only inline styles + inline SVG (no
`<style>`/`<script>`, so no CSS/JS collision when inlined) — **except**
`statistical.html`, which is a full `<html>` document with its own `<style>`.

- Fragment cards: inline the markup directly into the study section.
- Full-document cards: embed via `<iframe srcdoc="...">` for style isolation
  (matches the original `compare_report.py` iframe idiom; `srcdoc` keeps the output
  self-contained). A small auto-height script sizes the iframe to its content.

Styling: reuse `reports/assets/style.css` tokens inlined into a `<style>` block so
the page matches the report family while remaining a single portable file.

## Module structure

- `reports/investigation_summary.py` — thin CLI (argparse → aggregate → render →
  write/open).
- `reports/_summary/aggregate.py` — **pure** function: filesystem → a plain
  `InvestigationSummary` dict (`{investigation, question, studies[], matrix, rollup}`).
  No rendering, no I/O beyond reads. Independently testable and reusable (could later
  back a dashboard view).
- `reports/_summary/render.py` — pure function: `InvestigationSummary` dict → HTML
  string. No filesystem reads except pulling the referenced card HTML/CSS to inline.

Each unit has one purpose and a well-defined interface (dict boundary between
aggregate and render), so aggregation logic can be tested without HTML assertions.

## Testing

`tests/test_investigation_summary.py`:

1. Run `aggregate()` on `v2ecoli-vecoli-comparison`; assert all 7 studies discovered
   in DAG order (`parca` first, `statistical` last).
2. Assert the matrix cell verdicts equal the values read directly from the source
   `verdict.json` axes (no transformation drift).
3. Assert the roll-up counts equal the studies' canonical `result` tallies.
4. Assert every declared card resolves to an existing file, and a synthetic
   missing-card study yields a "missing card" placeholder rather than an exception.
5. Smoke: `render()` on the aggregated dict returns a non-empty string containing
   each study `name` and is a single self-contained document (no external `<link>`
   / `<script src>` to repo paths).

## Out of scope (YAGNI)

- Re-running any simulation.
- Dashboard / workbench (`vivarium_workbench`) changes.
- A live server or interactivity beyond native `<details>` collapse and the iframe
  auto-height shim.
- Cross-investigation index pages.
