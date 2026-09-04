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

## authored `confidence:` is unreachable — left unpatched, deliberately

`derive_confidence()` reads gate verdict -> `report.verdict` -> lifecycle
`status` and never consults `spec["confidence"]`, while the graph view assigns
it unconditionally. So `/viva-study`'s documented "Authored in: **Decide** (when
the derived value is wrong)" path cannot be exercised. That is a real upstream
bug and worth reporting.

**We are NOT patching it here.** A patch was written and then reverted on the
2026-09-04 review: making the authored value win lets a stale scaffold entry
(e.g. a `confidence: Planned` left from study creation) outlive the run that
superseded it and mask a real gate change. For an investigation with studies
actively re-running, derivation staying the single source of truth is worth more
than the ability to hand-author a badge.

The consequence is accepted knowingly: a study whose pre-registered hypothesis
failed renders `Refuted` even when its `claim` stands. That conflates "this gate
failed" with "this claim is false" — which are different things — but the fix
belongs upstream, in a form where a divergence carries a recorded warrant rather
than resting on an editor's judgment.

## the left rail painted every failed gate red

Third instance of the same class. `derive_confidence()`'s docstring states the
intent plainly — the value exists so that "the left-rail dot, the
investigation-graph node, and the study badge" all read one thing "**so they
never disagree**". The rail does not read it:

1. `static/walkthrough.js:_railStudyItem()` colours its dot with
   `_railStatusColor(s.status)` — the raw gate outcome.
2. `lib/investigations_index.py` never put a `confidence` key in the study row
   at all, so the rail *could* not read one.
3. `_railStatusColor()` also can't render the confidence vocabulary: `Refuted`
   and `Investigating` match none of its substrings and both fall through to
   gray, so naively passing confidence into it silently loses two of four
   states.

Consequence: a pre-registered study whose hypothesis failed carries
`status: failed` — accurate — and the rail painted it the same red it uses for
`invalid` and `blocked`. A healthy investigation that had done its job read as a
wall of errors.

**Patch:** ship the DERIVED `confidence` on the study row; colour the rail dot by it via a new
`_railConfidenceColor()` covering all four enums; keep the gate outcome in the
tooltip (`Accepted (gate: failed)`) so nothing is hidden. Falls back to the old
status colour when no confidence is shipped.

Originals: `*.orig` in this directory.
