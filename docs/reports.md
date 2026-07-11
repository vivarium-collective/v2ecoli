# Published outputs: dashboard, investigation reports, and HTML reports

Everything v2ecoli publishes is served from GitHub Pages at
**[vivarium-collective.github.io/v2ecoli](https://vivarium-collective.github.io/v2ecoli/)**
and runs in the browser with no install. There are **three** kinds of output,
each produced by a different pipeline.

| Output | Where it's served | How it's published |
|---|---|---|
| **[Interactive dashboard](#1-the-interactive-dashboard)** | `/dashboard/` | Static snapshot of the workspace SPA, auto-rebuilt from `main` (`.github/workflows/publish-dashboard.yml`) |
| **[Investigation reports](#2-investigation-reports-auto-generated)** | `/investigations/<slug>.html` (one per investigation) | Auto-rebuilt from `main` (`.github/workflows/publish-reports.yml`) |
| **[Standalone HTML reports](#3-standalone-html-reports)** | gallery root — `/<name>.html` | Generated on demand by `reports/*.py` / `scripts/*.py`, committed under `docs/` |

The [report gallery landing page](https://vivarium-collective.github.io/v2ecoli/)
links into all three.

---

## 1. The interactive dashboard

**[→ vivarium-collective.github.io/v2ecoli/dashboard/](https://vivarium-collective.github.io/v2ecoli/dashboard/)**

A **read-only** snapshot of the v2ecoli workspace rendered by the
[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench)
SPA. Browse:

- **Investigations & studies** — the research questions and the runs that answer
  them, with verdicts and figures.
- **Registry** — the Processes, Steps, Emitters, and Types this workspace
  registers (scoped to v2ecoli + its installed modules).
- **Composites** — navigable process/store wiring graphs (e.g. `baseline`,
  `parca`) via the embedded [bigraph-loom](https://github.com/vivarium-collective/bigraph-loom) viewer.
- **Sources** — the ecoli-sources input bundle, each with a link.

**How it's built.** `vivarium-workbench-publish` exports the workspace into a
self-contained static bundle (per-resource JSON + page shells + assets) that any
static server can host. The `publish-dashboard.yml` workflow rebuilds it from
`main` and publishes it to `gh-pages:dashboard/`.

> The hosted version is **view-only**. For the full interactive dashboard —
> authoring investigations, running studies, editing wiring — clone the repo and
> run `vivarium-workbench serve` locally (see the
> [vivarium-workbench README](https://github.com/vivarium-collective/vivarium-workbench#readme)).

---

## 2. Investigation reports (auto-generated)

Each investigation under `workspace/investigations/<slug>/` (any with an
`investigation.yaml`) publishes a single, **self-contained** interactive HTML
report — overview → studies → figures → reviewer decisions — at
`investigations/<slug>.html`.

**How it's built.** The report is rendered **client-side** by the dashboard SPA
(`_generateInvestigationReport`), embedding each study's figures inline. The
`publish-reports.yml` workflow drives that generator headlessly (Playwright) and
publishes the result to `gh-pages:investigations/` on every push to `main`.

| Investigation | Report | Research question |
|---|---|---|
| **Baseline Showcase** ⭐ | [v2ecoli-baseline-showcase.html](https://vivarium-collective.github.io/v2ecoli/investigations/v2ecoli-baseline-showcase.html) | Can v2ecoli rebuild the ParCa from raw ecoli-sources and reproduce the vEcoli baseline cell cycle? |
| **PDMP reformulation** | [v2ecoli-pdmp.html](https://vivarium-collective.github.io/v2ecoli/investigations/v2ecoli-pdmp.html) | Can the hybrid WCM be incrementally transformed into a piecewise-deterministic (PDMP) formulation, incl. the Millard kinetic-ODE → FBA flux bridge? |
| **Colonies** | [colonies.html](https://vivarium-collective.github.io/v2ecoli/investigations/colonies.html) | How many whole-cell *E. coli* agents fit per HPC node, and at what per-cell wall-time? |
| **KETCHUP baseline comparison** | [ketchup-baseline-comparison.html](https://vivarium-collective.github.io/v2ecoli/investigations/ketchup-baseline-comparison.html) | Does the baseline central-carbon exchange match the KETCHUP core-metabolism kinetic models? |
| **Parameter UQ** | [parameter-uq.html](https://vivarium-collective.github.io/v2ecoli/investigations/parameter-uq.html) | Forward uncertainty quantification — which parameters drive growth-rate variance? |
| **Units Atlas** | [units-atlas.html](https://vivarium-collective.github.io/v2ecoli/investigations/units-atlas.html) | A units-aware readout inventory across the model's ports. |

---

## 3. Standalone HTML reports

Generated on demand by a script under `reports/` or `scripts/`, committed under
`docs/`, and served at the gallery root. Each writes a self-contained HTML file.

### Interactive model viewers — *explore the model in your browser*

The [**baseline whole-cell composite viewer**](https://vivarium-collective.github.io/v2ecoli/baseline-viewer/)
opens the full process/store wiring in an interactive
[bigraph-loom](https://github.com/vivarium-collective/bigraph-loom) viewer — pan
the bigraph, expand stores, click any process for its formal `describe()`
description. Runs entirely in-browser.

| Report | Published | Generate locally |
|---|---|---|
| [Baseline composite](https://vivarium-collective.github.io/v2ecoli/bigraph_baseline.html) — processes, stores, wiring, port schemas (with units), per-process equations | `bigraph_baseline.html` | `scripts/viz_baseline_interactive.py` |
| [ParCa pipeline](https://vivarium-collective.github.io/v2ecoli/bigraph_parca.html) — the nine-Step parameter calculator, same viewer | `bigraph_parca.html` | `scripts/viz_parca_interactive.py` |
| [ParCa network](https://vivarium-collective.github.io/v2ecoli/parca_network.html) / [Baseline network](https://vivarium-collective.github.io/v2ecoli/network_baseline.html) — Cytoscape topology; click a process for ports/schema/docstring/math | `parca_network.html`, `network_baseline.html` | `scripts/parca_network.py` |

### Simulation result reports — *what the cell actually did*

| Report | Published | Generate locally |
|---|---|---|
| [Cell lifecycle](https://vivarium-collective.github.io/v2ecoli/workflow_report.html) — one cell, mother → division → both daughters | `workflow_report.html` | `reports/workflow_report.py` |
| [Multigeneration lineage](https://vivarium-collective.github.io/v2ecoli/multigeneration_report.html) — N-generation single lineage, mass trajectories & fold-change | `multigeneration_report.html` | `reports/multigeneration_report.py --generations 3` |
| [Colony](https://vivarium-collective.github.io/v2ecoli/colony_report.html) — mixed colony with pymunk physics, growth & division, synced animations | `colony_report.html` | `reports/colony_report.py --n-adder 9` |

### Comparison & benchmark — *v2ecoli vs vEcoli*

| Report | Published | Generate locally |
|---|---|---|
| [v1 vs v2](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html) — vEcoli 1.0 vs v2ecoli baseline: wall/sim time, dry mass, growth | `v1_v2_comparison.html` | `reports/v1_v2_report.py` |
| [Composite comparison](https://vivarium-collective.github.io/v2ecoli/composite_comparison.html) — any set of engines side-by-side (load/wall/sim, composition, sparklines) | `composite_comparison.html` | `reports/composite_comparison.py --engines baseline millard_pdmp_baseline` |
| Benchmark — v2ecoli vs the vEcoli composite (local-only) | — | `reports/benchmark_report.py` |

### Model structure, ParCa & sources — *the math, the parameters, the inputs*

| Report | Published | Generate locally |
|---|---|---|
| [Mathematical structure](https://vivarium-collective.github.io/v2ecoli/math_structure.html) — every process's governing equations by subsystem, per-tick execution flow, partition→allocate→evolve contract | `math_structure.html` | `reports/math_structure_report.py` |
| [ParCa workflow](https://vivarium-collective.github.io/v2ecoli/parca_workflow_report.html) — the nine-Step run with per-step runtimes, port manifests, raw-data stats | `parca_workflow_report.html` | (ParCa pipeline) |
| [ecoli-sources](https://vivarium-collective.github.io/v2ecoli/ecoli_sources_report.html) — the ParCa input bundle (inherited + v2ecoli overrides), each with a download link | `ecoli_sources_report.html` | `reports/ecoli_sources_report.py` |

> **Provenance banners.** PR-evidence reports (`scripts/pr_session_report.py`,
> `scripts/sweep_report.py`) embed a self-describing header — ISO timestamp, git
> SHA/branch, dirty-tree badge, last commit, host/OS/Python — so an HTML file
> stays meaningful months later. Attach these to PRs that change biology; see
> [AGENTS.md → Reports](../AGENTS.md).
