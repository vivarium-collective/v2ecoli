# Local patches to vivarium-workbench 0.3.78

Applied against the pip-installed package in `.venv`. **These are lost on any
reinstall or upgrade** — re-apply with `patches/apply.sh`, or drop them once
upstream carries the fix.

## investigation-graph edge relations were unreachable

The DAG renderer defines a five-value edge vocabulary — `leads-to`,
`model-input`, `evidence`, `calibrates-threshold`, `refutes-alternative` — and
draws a legend for all five. No code path could produce four of them:

1. `lib/investigation_graph_views.py` built each legacy-path edge as
   `{source, target, rel: "prerequisite", condition}` and never passed the
   `relation` declared in `pipeline_gate.prerequisites[]`.
2. `static/walkthrough.js` then set `relation: e.artifact ? 'model-input' :
   'leads-to'`, ignoring any server-supplied relation.

So a study declaring `relation: evidence` rendered as `leads-to`, and the
investigation graph showed a flat chain regardless of the declared semantics.
`_dagEdges()` reads the canonical field correctly, but `dagEdgesFor()` prefers
the server's edges and only falls back when the server returns none — which the
legacy path never does.

**Patch:** emit `relation` (and `outputs_used`) server-side; honour
`e.relation` client-side, keeping the artifact-based default as the fallback.

Both changes are additive and back-compatible: an edge with no declared
relation behaves exactly as before.

Originals: `*.orig` in this directory.
