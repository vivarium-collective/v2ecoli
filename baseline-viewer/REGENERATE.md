# baseline-viewer — static bigraph-loom deployment

A serverless, view-only build of
[bigraph-loom](https://github.com/vivarium-collective/bigraph-loom) that opens
the v2ecoli **baseline** whole-cell composite on GitHub Pages, with a preset
default arrangement. Linked from the repo README.

Live: `https://vivarium-collective.github.io/v2ecoli/docs/baseline-viewer/?static=1&stateUrl=data/baseline.state.json&viewUrl=data/baseline_default.view.json`

`?static=1` shows only the View tab (no Configure/Run/Results/Visualizations —
those need the dashboard server). `?stateUrl=` + `?viewUrl=` load the committed
snapshots below instead of any `/api/*`.

## Files

- `index.html` + `assets/` — the bigraph-loom static bundle (`npm run build`).
- `data/baseline.state.json` — point-in-time snapshot of the baseline composite
  state **with per-process `describe()` docs attached** (the dashboard's
  `/api/composite-state?ref=v2ecoli.composites.baseline.baseline` response).
- `data/baseline_default.view.json` — the default view (node positions +
  collapsed groups + hidden nodes), exported from the viewer's `Views ▾` menu.

## Regenerate

```bash
# 1. Rebuild the viewer bundle (from the bigraph-loom repo) and copy it here,
#    excluding sourcemaps:
( cd /path/to/bigraph-loom && npm run build )
find /path/to/bigraph-loom/bigraph_loom/_dist -type f ! -name '*.map' \
  | while read f; do install -D "$f" "docs/baseline-viewer/${f#*/_dist/}"; done

# 2. Re-snapshot the composite state from a running dashboard:
curl -s "http://localhost:<port>/api/composite-state?ref=v2ecoli.composites.baseline.baseline" \
  -o docs/baseline-viewer/data/baseline.state.json

# 3. Update the default view: arrange it in the viewer -> Views ▾ -> Export
#    view file -> save as docs/baseline-viewer/data/baseline_default.view.json
```

The snapshot is point-in-time: regenerate `baseline.state.json` if the baseline
composite's structure changes.

## Short URL redirect (re-apply after a rebuild)

`index.html` carries a small inline `<script>` (before the module script) that
redirects a bare visit to the full query string, so
`…github.io/v2ecoli/baseline-viewer/` (and the QR built from it) loads the
featured composite + saved view for **every** visitor. Step 1's `npm run build`
overwrites `index.html`, so re-add the block:

```html
<script>
  if (!location.search) {
    location.replace(location.pathname +
      '?static=1&id=v2ecoli.composites.baseline.baseline' +
      '&stateUrl=data/baseline.state.json&viewUrl=data/baseline_default.view.json');
  }
</script>
```

The default view (`data/baseline_default.view.json`) is committed here, NOT read
from any visitor's localStorage — so the online viewer shows the saved
arrangement to everyone.
