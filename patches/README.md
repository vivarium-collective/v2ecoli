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

## authored `confidence:` was unreachable

Same shape as the edge bug, one field over. `/viva-study` documents `confidence:`
as **"Authored in: Decide (when the derived value is wrong)"**, but no code path
could read an authored value:

1. `lib/investigation_graph_views.py` sets `_node["confidence"] =
   derive_confidence(study_spec)` unconditionally.
2. `viva_superpowers.study_verdict.derive_confidence()` reads gate verdict ->
   `report.verdict` -> lifecycle `status`. It never consults
   `spec["confidence"]`.
3. `static/walkthrough.js:10407` does `s.confidence || <fallback>` — but the
   server always populates `s.confidence`, so the fallback never fires and the
   authored value never reaches the renderer.

The source comment at (1) claims "the frontend prefers this over its own
status-fallback derivation (`s.confidence || ...`)", which reads as though the
authored field wins. It does not; it is overwritten before the frontend sees it.

Consequence: `confidence` is a pure function of `gate_status`, so **any study
whose pre-registered hypothesis failed renders `Refuted`** — even when the study
executed exactly as designed and its `claim` (the knowledge it produced) stands.
That conflates *"this gate failed"* with *"this claim is false"*, which are
different things, and makes a healthy pre-registered investigation look like a
wall of failures.

**Patch:** prefer an authored `confidence:` when it is one of the four enums
(`Accepted | Investigating | Refuted | Planned`); fall back to
`derive_confidence()` when absent or invalid. Additive — a study with no
authored `confidence:` behaves exactly as before.

Originals: `*.orig` in this directory.
