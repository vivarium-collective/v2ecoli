# Composite viewers hub — regenerate & publish

The public composite viewers live in `docs/viewers/` (served at
`https://vivarium-collective.github.io/v2ecoli/viewers/`). Each curated
composite is viewable in three tools — **bigraph-loom** (interactive),
**bigraph-viz2** (interactive, light), and **bigraph-viz** (static SVG) — all
fed from one cached state per composite.

This `baseline-viewer/` directory is now a thin redirect into that hub
(preserving the old `/baseline-viewer/` URL + its QR).

## Prerequisites (run locally)

- The ParCa cache must be on disk so the composites resolve. If resolution
  fails with a config `KeyError` (e.g. `tf_ids`), the cache is stale — rebuild:
  ```bash
  PYTHONPATH=. .venv/bin/python scripts/build_cache.py
  ```
- `bigraph-viz2` must be importable. It is not always installed in the workspace
  venv; the easiest path is to add its `py/` dir to `PYTHONPATH` for the run
  (the regen command below does this). bigraph-loom and bigraph-viz are normal
  workspace deps.

## Regenerate (one command)

```bash
PYTHONPATH=.:/path/to/bigraph-viz2/py .venv/bin/python scripts/regenerate_viewers.py
```

This rewrites `docs/viewers/{data/*.state.json, img/*.svg, viz2/*.html,
loom/**, index.html}` from `docs/viewers/viewers.json`. It:
- resolves each composite to a cached state (reusing the dashboard's
  `_composite_resolve_data`), trimmed of bulk molecule-data arrays the viewers
  don't draw,
- renders a bigraph-viz SVG and a self-contained bigraph-viz2 snippet,
- copies the shared bigraph-loom bundle (source maps stripped),
- regenerates the hub `index.html`.

A composite that fails to resolve is skipped with a logged reason (and a
build_cache hint) — it never aborts the run. Saved loom arrangements
(`docs/viewers/data/<slug>.view.json`) are NOT overwritten: export one from the
loom `Views ▾` menu and commit it to give a composite a default arrangement.

## Add a composite to the showcase

Add one line to `docs/viewers/viewers.json` (`slug`, `id`, `title`, `blurb`) and
re-run the regen command.

## Publish to GitHub Pages

```bash
bash scripts/publish_viewers.sh
```

Surgically replaces only `viewers/` and `baseline-viewer/` on the `gh-pages`
branch (leaving `dashboard/`, `investigations/`, and the docs mirror untouched).
