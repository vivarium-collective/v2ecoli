# Composite Viewers Hub — design

## Problem

Today the only public, serverless view of a v2ecoli composite is
`docs/baseline-viewer/` — a standalone bigraph-loom build frozen on the
**baseline** composite, with a manually-regenerated cached state
(`data/baseline.state.json` + a saved view). It covers one composite, one
viewer tool, and the regeneration steps are hand-run curl/`npm build` commands
documented in `REGENERATE.md`. Separately, loose cached states
(`baseline_state.json`, `biological_state.json`) and ad-hoc viewer HTMLs
(`bigraph_baseline.html`, `bigraph_parca.html`, …) sit scattered at the
gh-pages root.

We want a single, organized, **standalone public showcase** where each of a
curated set of composites is viewable through all three rendering tools, backed
by one cached state per composite, kept fresh by one command.

## The three viewers (all consume a process-bigraph composite state dict)

- **bigraph-loom** — interactive React Flow viewer; loads a state JSON at
  runtime via `?stateUrl=` (the tool baseline-viewer uses today). Per-node
  inspector (config / schemas / `describe()` docs).
- **bigraph-viz2** — lightweight interactive read-only renderer, no graphviz;
  `emit_html(state)` produces a self-contained snippet, or its IIFE bundle
  renders client-side from the same state JSON.
- **bigraph-viz** — original Graphviz-based static plotter;
  `plot_bigraph(state, schema, …, file_format='svg')` → static SVG/PNG
  (regen-time, needs graphviz).

Because all three ingest the same composite **state**, one cached state per
composite feeds all three: loom and viz2 read it at runtime; viz is
pre-rendered to a static SVG at regen time.

## Scope

**Curated set (5):** `baseline`, `biological`, `millard_fba_bridge_harness`,
`colony`, `reactor_bird_coupled`. The set is data-driven by a manifest, so
adding/removing a composite is a one-line edit, not a code change.

Out of scope: auto-generating viewers for all ~16 composites; CI
auto-regeneration; modifying the viewer tools themselves; feeding the
interactive dashboard's Composite Explorer (this hub is independent).

## Architecture

Published gh-pages layout (sourced from `docs/viewers/` in the repo):

```
docs/viewers/
  index.html                    hub page — one row per composite × 3 viewer links
  viewers.json                  manifest: [{id, title, blurb, state, svg}] curated set
  loom/                         ONE shared bigraph-loom static bundle (assets + index.html)
  viz2/                         ONE shared bigraph-viz2 IIFE bundle (.iife.min.js + .css)
  data/<slug>.state.json        cached composite state — feeds loom + viz2
  data/<slug>.view.json         optional saved loom arrangement (baseline has one; others auto-layout until exported)
  img/<slug>.svg                bigraph-viz static render (graphviz)
```

`<slug>` is a short, stable per-composite id (e.g. `baseline`, `biological`).
The full registry id (e.g. `v2ecoli.composites.ecoli_baseline.ecoli_baseline`) is carried
in the manifest and passed to loom as `?id=`.

### Hub page (`index.html`)

Static page generated from `viewers.json`. One card/row per composite: title,
one-line blurb, and three buttons:

- **Loom** → `loom/index.html?static=1&id=<id>&stateUrl=../data/<slug>.state.json&viewUrl=../data/<slug>.view.json`
  (omit `viewUrl` when no saved view exists → loom auto-layouts).
- **Viz2** → a thin `viz2/render.html?state=../data/<slug>.state.json` that
  loads the IIFE bundle and renders the fetched state client-side. (One shared
  render shell, parameterized by `?state=` — mirrors loom's single-bundle reuse.)
- **Viz** → `img/<slug>.svg` (static image, opens directly).

All viewer links for a composite resolve to the **same** `data/<slug>.state.json`
— a single source of truth per composite.

### Regeneration (`scripts/regenerate_viewers.py`) — the one command

Reads `viewers.json` and, for each curated composite:

1. **Resolve state + SVG** by reusing the dashboard's
   `vivarium_workbench.server._composite_resolve_data(<id>)`, which returns the
   loom state JSON (with `describe()` docs) **and** a bigraph-viz SVG in one
   call. Write `data/<slug>.state.json` and `img/<slug>.svg`.
   - Runs **locally**, where the on-disk ParCa cache exists, so heavy composites
     (baseline, biological, colony) resolve. (This is exactly why the published
     dashboard shows `has_wiring=false` for baseline — CI has no cache; local
     regen does.)
   - A composite that fails to resolve is **skipped with a logged warning** and
     left out of the regenerated hub, rather than aborting the whole run.
2. **Refresh shared bundles** — copy the current bigraph-loom `_dist/` into
   `loom/` and the bigraph-viz2 IIFE bundle into `viz2/` (idempotent; only the
   two shared bundles, not per-composite copies).
3. **Regenerate `index.html`** from the manifest + whichever composites
   resolved.

`data/<slug>.view.json` (saved loom arrangements) are **not** overwritten by
regen — they are committed artifacts, hand-exported from loom's `Views ▾` menu.
Baseline keeps its existing saved view; others start auto-layout and gain saved
views incrementally.

### Publish (`scripts/publish_viewers.sh`)

Mirrors `scripts/publish_dashboard.sh`: a surgical worktree push that replaces
only `viewers/` (and the `baseline-viewer/` redirect) on the gh-pages branch,
leaving `dashboard/`, `investigations/`, and the docs mirror untouched.

### baseline-viewer migration

Replace the standalone `docs/baseline-viewer/` bundle with a **redirect** so the
existing public URL and QR keep working:

```html
<!-- docs/baseline-viewer/index.html -->
<script>location.replace('../viewers/loom/index.html?static=1'
  + '&id=v2ecoli.composites.ecoli_baseline.ecoli_baseline'
  + '&stateUrl=../../viewers/data/baseline.state.json'
  + '&viewUrl=../../viewers/data/baseline.view.json');</script>
```

(Exact relative paths finalized against the gh-pages root layout during
implementation.) The baseline saved view (`baseline_default.view.json`) moves
into `docs/viewers/data/` as `baseline.view.json`.

## Components & boundaries

| Unit | Purpose | Depends on |
|------|---------|------------|
| `viewers.json` | declares the curated set + per-composite metadata | — (hand-edited) |
| `regenerate_viewers.py` | manifest → cached states + SVGs + refreshed bundles + hub html | dashboard `_composite_resolve_data`, bigraph-loom `_dist`, bigraph-viz2 bundle |
| `publish_viewers.sh` | push `docs/viewers/` to gh-pages | git worktree, gh-pages branch |
| `index.html` (generated) | the hub UI | `viewers.json`, the shared bundles, `data/`, `img/` |
| `viz2/render.html` | shared client-side viz2 render shell | viz2 IIFE bundle, `?state=` |

Each composite's three viewers are independent and all read the one
`data/<slug>.state.json`; changing a composite means re-running regen, nothing
else.

## Error handling

- Unresolvable composite at regen → warn, skip, exclude from hub (no abort).
- Missing `data/<slug>.view.json` → loom link omits `viewUrl` (auto-layout).
- Missing graphviz → viz SVG step warns and the Viz button is omitted for that
  run (loom + viz2 still work).

## Testing

- `regenerate_viewers.py` unit: a tiny fake manifest + a stub
  `_composite_resolve_data` returning a known state/SVG → asserts `data/*.json`,
  `img/*.svg`, and a hub `index.html` listing exactly the resolved composites
  are written; an unresolvable entry is skipped, not fatal.
- Hub html generation: asserts each resolved composite yields three correctly-
  formed viewer URLs (loom `?stateUrl=`, viz2 `?state=`, viz `.svg`) and that
  the manifest's full `id` is passed to loom.
- baseline-viewer redirect: assert the committed redirect targets the hub's
  baseline loom URL.

## Success criteria

1. `python scripts/regenerate_viewers.py` produces, from the curated manifest,
   a complete `docs/viewers/` (states, SVGs, shared bundles, hub) in one command.
2. The hub opens each curated composite in loom, viz2, and viz, all from one
   shared cached state per composite.
3. `/baseline-viewer/` (and its QR) still resolves — now via redirect into the
   hub.
4. Adding a composite to the showcase is a one-line `viewers.json` edit + a
   regen run.
