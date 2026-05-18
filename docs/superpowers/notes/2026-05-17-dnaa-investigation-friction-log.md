# Notes for the pbg-superpowers / vivarium-dashboard / pbg-template improvement agent

**Session:** v2ecoli, dnaa-replication investigation, 2026-05-17
**Author:** Claude (Opus 4.7, 1M context), executing the investigation as a working Claude
**Purpose:** Record friction points I hit while running this investigation so the
follow-on improvement agent can decide what to upstream.

The investigation is mid-flight — dnaa-01 baseline re-ran, dnaa-02 implementation
started. I will keep appending to this file as I work; treat the chronological
order as priority order (earliest = most painful = highest ROI to fix).

---

## What this investigation looks like in practice

Four sequential studies (dnaa-01..04) that incrementally replace the heuristic
chromosome-replication trigger with a DnaA-driven mechanism. Each study
defines `purpose`, `simulation_set`, `behavior_tests`, `implementation_requirements`,
`pipeline_gate`, `key_assumptions`, etc., in a single `study.yaml`. The
workspace dashboard renders all of this as a planning report; the investigation
"begins" when I start implementing the requirements.

---

## Friction points

### 1. Studies declare `composite: v2ecoli.composites.baseline_recipes.<name>` but no recipe file exists

dnaa-02..04 each `baseline:` block points to a composite like
`v2ecoli.composites.baseline_recipes.dnaa_02_with_intrinsic_hydrolysis`. This
module doesn't exist in `v2ecoli/composites/`. The convention seems to be
"the planner names it; the implementer creates it." Three problems:

- No template / scaffold. I have to read existing composites (baseline.py,
  departitioned.py) to figure out the structure, then hand-write a near-clone
  that adds the new Step in the right execution layer.
- No way to check from the dashboard that the recipe is missing (the study
  page shows the composite path as plain text — clicking it should at least
  say "not found, scaffold one?").
- The hand-written recipes will drift from baseline.py over time. Adding
  features to the upstream baseline (like ppgpp_regulation) won't propagate
  to dnaa_02_with_intrinsic_hydrolysis without a manual sync.

**Improvement suggestion:** A `composite_recipe` generator that takes a
diff-style spec (e.g., `extends: baseline; adds_steps: [DnaaIntrinsicHydrolysis];
insert_after: ecoli-equilibrium; new_params: {dnaA_intrinsic_hydrolysis_rate_per_min: 0.046}`)
and emits the composite module. The dashboard could have a "Scaffold recipe"
button on the study page that auto-creates the file from this spec.

### 2. The simulation runner is hand-written per study

The dnaa-01 YAML referenced `studies/dnaa-01-expression-dynamics/sims/run_baseline.py`
which didn't exist; I had to write it from scratch (~200 lines) to capture
DnaA-specific readouts. There's `scripts/run_v2.py` but it only captures cell
mass, not the per-study observables.

Friction:
- Each study will need a near-identical runner with different readouts.
- No way to declare `readouts:` in study.yaml and have a runner generated.
- No way to declare `simulation_set:` in study.yaml and have the dashboard
  execute it (the "Run" button on the investigation page runs ONE recipe,
  not a sweep).

**Improvement suggestion:** A study-level "Run all simulations" action that
reads `simulation_set:` entries, instantiates each as a composite with the
declared perturbation/params, and emits the declared `readouts:` to a
study-scoped output dir. Should also persist a `runs.db` per study so the
dashboard can pick up results without further glue.

### 3. Listener output schema is a footgun (the `overwrite[]` bug)

The walkthrough notes recorded a bug where `monomer_counts_listener` emitted
DnaA = 51,781 instead of ~100. Root cause: output type declared as
`array[N,integer]` instead of `overwrite[array[N,integer]]`, so each timestep
ADDED to the previous value. The same pattern exists on at least two sibling
listeners (`rna_synth_prob`, `replication_data`). Sister fix `req-7` notes
the dnaa-mRNA TU index lookup is also broken silently.

**Improvement suggestion:** A linter that flags listener outputs lacking an
`overwrite[]` wrapper unless they explicitly declare a delta/accumulator
semantics. Could be a bigraph-schema type rule; could be a pre-commit hook
under `scripts/lint-workspace.py`.

### 4. No "audit a composite for X" tool

The first thing I needed before starting dnaa-02 was: "where does v2ecoli
read/write `MONOMER0-160[c]`?" Took me three `grep` invocations across
the source tree + a runtime probe of the cached configs to find the answer
(MONOMER0-160_RXN in equilibrium, plus a parallel MONOMER0-4565_RXN for
the ADP form that I didn't know existed).

The audit took ~10 minutes. It could have taken 30 seconds:

**Improvement suggestion:** A dashboard panel "What touches this bulk id?"
or `pbg-audit <bulk_id>`: grep + sim_data introspection + emit a table of
processes that read/write the species, plus equilibrium/complexation
reactions that include it.

This is exactly the discovery a dnaa-02-EQ-02 expert question is asking for;
the dashboard could resolve a chunk of those automatically before they ever
go to a human expert.

### 5. The "before state" is invisible until a biologist explicitly asks

The user prompted: "maybe we can run the baseline, and use it in visualizations
for the planning phase, to see the before state". This was extremely valuable —
running the baseline once, before any new study runs, surfaced a calibration
issue (DnaA = 115/cell vs literature 300-800) AND a design clarification
(all DnaA is in MONOMER0-160 form, free monomer ≈ 0). Both of these
materially changed how dnaa-02 will be implemented.

But nothing in the workflow prompted this. The dashboard's planning report
defaults to showing the spec without any "what does the baseline cell look
like right now?" panel.

**Improvement suggestion:** A "Baseline snapshot" panel on the investigation
page that runs the workspace's baseline composite once (or shows the last
snapshot if cached) and renders 3-5 standard charts: mass trajectory, key
molecule trajectories, chromosome state. Investigations gain a "what does
the cell already do?" anchor before any study runs.

### 6. The expert-decisions-needed loop is one-way

Each study yaml lists `expert_decisions_needed:` with `status: open, asked_to: TBD`.
There's no surface in the dashboard for an expert to ANSWER one (it'd have
to be a YAML edit). And there's no inbox view that says "these 7 questions
are open across this investigation."

I just resolved one (dnaa-02-EQ-02) via code audit. Editing the YAML manually
was fine but not discoverable — an expert would not know to do this.

**Improvement suggestion:** A `Pre-run expert review` inbox on the investigation
page that aggregates open questions across all studies, with a "Resolve"
modal that writes back to the source study.yaml (`status: resolved`,
`resolution: <text>`). I've already added the per-study panel in this
session; the inbox view is a natural extension.

### 7. The investigation-plan report is biologist-unfriendly by default

The user explicitly called out that the default study-detail view leads
with implementation details (listener paths, JSON test specs, code refs)
and a biologist has to dig past them. I patched this in-session by adding
`biological_summary`, `study_card`, `literature_anchors` fields to the
study YAMLs + corresponding template blocks. But:

- The fields are not part of any schema; nothing prompts a study author
  to add them.
- The migration would need a one-time pass over existing studies.
- The shareable HTML report (`_buildInvestigationReportHtml` in
  `walkthrough.js`) duplicates large chunks of the live-dashboard rendering
  logic. I had to patch the same biology blocks in TWO places (template
  for live view, JS function for downloadable).

**Improvement suggestion:**
- Add the three biology fields to the study.yaml schema with examples.
- Either generate the shareable HTML server-side from the same Jinja2
  templates (so one source of truth), or extract a shared rendering library
  used by both paths.

### 8. Re-running the dashboard to pick up template/JS changes is manual

When I patched `study-detail.html` and `walkthrough.js`, I had to:
1. Kill the running dashboard process by PID
2. Restart `bash scripts/serve.sh`
3. Find the new port (each restart picks a new random port)
4. Verify the change took via curl

This is ~3-5 manual operations per template iteration. Hot-reload would help
a lot — auto-restart on template/static change, and keep a stable port.

### 9. Each study's runner needs its own out/ subdirectory convention

I put dnaa-01 baseline outputs at `out/dnaa-01/baseline_seed{0,1,2}.json`.
Nothing told me this was the convention; nothing enforces it. The
`baseline_preview/data.json` lives at `reports/baseline_preview/data.json`.
A future study could land its outputs anywhere.

**Improvement suggestion:** A `study_outputs(study_name, run_name) -> Path`
helper in `pbg_superpowers` that returns the canonical location. Used by
both runners and the dashboard's runs view.

### 10. Bulk-id discovery requires composite instantiation

Finding "what is the bulk id for DnaA-ADP?" required:
1. `build_composite('baseline', ...)` (7+ seconds load time)
2. Iterate `cell['bulk']['id']` for matches

There's no offline registry. A dashboard panel `Bulk-id browser`
(searchable list of all bulk ids, with their initial counts, with the
processes that touch each one) would save a lot of probe scripts.

---

## Wins (worth preserving)

Not all friction — these things worked well and shouldn't regress:

- **The cache_version check.** `build_cache.py` rebuilds in ~3 seconds and
  the StaleCacheError gives a one-line fix command. This is exactly the
  right ergonomics.
- **The study.yaml schema flexibility.** Adding new fields (`biological_summary`,
  `literature_anchors`, `downstream_design_implications`) didn't break any
  loader. The dashboard rendered them once I patched the template.
- **The dashboard's auto-discovery of investigations + studies.** Dropping
  new files at `studies/<slug>/study.yaml` and reloading the dashboard
  was enough to surface them.
- **The `/api/iset/<name>` + `/api/study/<name>` shape.** Clean JSON; easy
  to extend (I added one field per endpoint).

---

## Patterns I evolved during this session that might generalize

### Pattern A: "Audit-before-implement" for any cross-study assumption

The first ~15 minutes of dnaa-02 implementation went to auditing what was
already there. This saved a 4-hour mistake (manually creating species that
already existed). Convention: any `expert_decisions_needed` entry of the
form "should X / should Y / should Z" should trigger a code audit BEFORE
the expert is even asked, because the audit may resolve the question
mechanically.

### Pattern B: "Decision card" in the YAML when an audit resolves an EQ

I added `status: resolved-by-audit` + `resolution:` to the YAML when I
resolved dnaa-02-EQ-02. This is a one-shot schema addition; the dashboard
could render resolved entries differently (green, collapsed) without changing
the data model further.

### Pattern C: Baseline-preview-as-evidence

I generated `reports/baseline_preview/index.html` with four planning-phase
viz from the dnaa-01 baseline. The biologist can open it before any study
runs. Equivalent: the workspace dashboard should default to showing this
preview on the investigations page when no study has run yet.

---

## What I'm doing next (so you know the state when you read this)

Right now I'm implementing `v2ecoli/steps/dnaa_intrinsic_hydrolysis.py` (req-2)
and the composite recipe `v2ecoli/composites/baseline_recipes.py`. Will
append to this file as I hit more friction.

---

## Friction (continued): Real-time tug-of-war between new Steps and existing equilibrium

This one is interesting because the bug surfaced from running, not from reading.

I implemented `DnaaIntrinsicHydrolysis` (MONOMER0-160 → MONOMER0-4565 at k = 0.046/min)
and `DnaaAtpFractionClamp` (forces ATP-fraction into [0.2, 0.5]). Each timestep
the clamp transferred ~57 molecules from MONOMER0-160 to MONOMER0-4565
(verified via the clamp's own listener emit: `xfer=57`, `direction=atp_to_adp`).

But the resulting bulk state OSCILLATED between two regimes across ticks:

| Tick | atp_bulk | adp_bulk | who won |
|------|----------|----------|---------|
|  60  | 56       | 57       | clamp   |
| 120  | 58       | 59       | clamp   |
| 240  | 115      | 1        | equilibrium |
| 300  | 57       | 57       | clamp   |
| ...  | ...      | ...      | alternates |

Root cause: process-bigraph uses **snapshot semantics within a tick**. All Steps
read inputs from the same tick-start state, accumulate updates independently,
and the framework applies all deltas at end-of-tick. So:

1. Tick starts: bulk[MONOMER0-160] = 115, bulk[MONOMER0-4565] = 0
2. My clamp computes Δ = (-57, +57) based on starting state
3. ecoli-equilibrium runs the ODE on starting state, computes its own delta
   that wants to drive back to its thermodynamic steady state (115/0)
4. Deltas SUM, end of tick: bulk = 115/0 + my_delta + eq_delta = 57/58 (lucky)
5. Next tick: equilibrium reads 57/58, sees it's far from steady state, computes
   a HUGE delta to drag it back to 115/0. Clamp reads 57/58, sees atp_frac=0.50
   (already at the high edge of the band) → noop.
6. End of tick: bulk = 115/0 (equilibrium wins).
7. Cycle repeats.

**Within-tick priority hints (which I tried) don't help** because they don't
change which state Steps READ from. Priorities only break ties for tasks the
framework hasn't already serialized via input/output dependency analysis.

### What broke my mental model

I assumed: "give my Steps lower priority and they'll run last, reading the
already-equilibrated bulk, then make their transfers stick."

What actually happens: ALL Steps read the same start-of-tick snapshot. The
equilibrium's delta is computed against that snapshot too, so it always
"unwinds" my transfers in the very same tick — by COMPUTING what the
equilibrium SHOULD look like and writing the difference.

### What would have prevented this

1. **A process-bigraph FAQ or troubleshooting note**: "Why are my Step's
   bulk transfers being reverted each tick?" → explain snapshot semantics
   + show pattern for cross-step ordering (token chains, locked species,
   or join the equilibrium reaction system itself).

2. **A "what-if I add a Step" simulator** in the dashboard: drop in a Step
   stub, declare its read/write store paths, and the dashboard shows you
   the predicted dependency graph + warns "this Step writes a store that's
   also written by ecoli-equilibrium — they will combine deltas, not chain."

3. **A worked example in pbg-superpowers** of "how to add a Step that
   modifies a species also under equilibrium / complexation control".
   The two natural options are (a) introduce a new "locked" species that
   isn't in the equilibrium reaction set, or (b) modify the equilibrium
   reaction set itself. Both are non-trivial; a recipe would save days.

### My next decision

I documented this in the study yaml's `conclusion_verdicts.biological_validation`
as a FAIL with a clear root-cause line + three options for the user to pick.
The work I did is preserved (Steps + recipe + runner); we just need a design
decision before validation can pass.

---

## Pattern: validate against the runner BEFORE updating study-status

The flow I evolved naturally:
1. Implement Steps.
2. Add to composite via recipe.
3. Run via study sims/ runner.
4. Inspect trajectory + listener emit.
5. ONLY THEN update study.yaml status fields.

Step 5 prevents claiming "PASS" on a study whose code superficially ran but
produced biologically wrong results. The dashboard could enforce this by
requiring a run-id + non-empty results before any study.status moves out of
`planning` / `design`. Right now, nothing stops me from manually editing the
status field to "complete" with no runs to back it up.

---

## Friction (continued): Visualizations tab silently empty for studies with completed runs

User caught this during dnaa-01's review: the study's Visualizations tab in
the dashboard was empty even though I had already rendered baseline trajectory
charts at ``reports/baseline_preview/index.html`` and the study had successful
runs in ``out/dnaa-01/baseline_seed*.json``.

Two reasons:

1. The dashboard's "Latest-run visualizations" panel reads from ``runs.db``,
   not from arbitrary JSON output files. My runner wrote raw JSON; the
   dashboard never saw it.

2. The dashboard's "Registered visualization modules" panel reads
   ``study.visualizations`` from the YAML. dnaa-01 didn't declare any
   (dnaa-02..04 do, with addresses pointing at ``DnaAStateVisualization`` /
   ``DnaABoxOccupancyVisualization`` classes that don't exist yet).

The user's reasonable expectation was: "I ran the study. The dashboard
should show me what it produced — automatically."

**Improvement suggestions for the meta-agent:**

- **Auto-discover study outputs.** The dashboard should scan ``out/<study-slug>/``
  for JSON or NPZ files and offer a "View raw output" link, plus a generic
  trajectory plotter for numeric arrays (one chart per array, time-series if
  there's a time column).

- **Allow ``visualizations:`` entries to point at static HTML files** (or a
  URL) for the case where the runner already produced a self-contained
  chart. Schema extension:
    ```yaml
    visualizations:
      - name: baseline_preview
        embed: reports/baseline_preview/index.html    # NEW
        description: pre-execution baseline preview
    ```

- **Persist runner outputs to ``runs.db``** so the auto-panel picks them up.
  Currently my runner writes JSON in ``out/dnaa-01/``; the dashboard's
  auto-viz code looks at ``studies/<slug>/runs.db``. There's no shared
  convention for "the runner shipped here writes there."

For now I patched study-detail.html to render a study.embed_visualizations
list as iframes, and pointed dnaa-01 at its baseline_preview HTML. That
makes the tab non-empty, but the underlying problem is real: nothing
prompted me to do this, and the next study will hit the same wall.

---

## Friction (continued): Per-study Visualizations are STILL not automatic

After fixing dnaa-01's embed (added `embed_visualizations:` to the YAML +
patched study-detail.html to render iframes), user immediately hit the same
wall on dnaa-03: "where are the figures, why don't they show automatically?"

The literal answer is: I produced JSON output (`out/dnaa-03/seed0_v1_500dnaa.json`)
but never rendered it as charts. The dashboard cannot auto-generate Plotly
panels from arbitrary JSON; it requires either:

  (a) a Visualization class registered at the `address:` declared in study.yaml
      (e.g. `local:DnaAStateVisualization`) — none of dnaa-02/03/04's declared
      classes actually exist yet, so the "Registered visualization modules"
      panel renders just the *name* with no chart
  (b) a hand-authored HTML preview I plug in via `embed_visualizations:`
      (what I did for dnaa-01)
  (c) a runs.db with the auto-viz Step pipeline wired through — none of my
      hand-rolled runners populate runs.db

So every study I run produces a manual chore: write the preview HTML, add
the embed entry. This is on-brand for the user's irritation — it should be
automatic.

**Highest-ROI improvements for the meta-agent:**

1. **Auto-discover `out/<study-slug>/*.json` and render generic charts.**
   The runner-output JSON I'm writing has a uniform shape (snapshots list
   with numeric fields). A dashboard endpoint
   `/api/study/<slug>/auto-charts` could:
     - Walk `out/<slug>/`, find JSON files with a `snapshots` key
     - Infer numeric series per snapshot field
     - Render a generic time-series plot per numeric field
     - Show as a "Latest output" panel below the Visualizations tab header
   Even a UGLY generic plot beats an empty tab.

2. **Auto-derive `embed_visualizations:` from `simulation_set:` outputs.**
   If study.yaml declares `simulation_set: [- name: X, readouts: [a,b,c]]`,
   the dashboard could auto-generate one Plotly chart per readout from
   the latest run's output file.

3. **Scaffold a runner stub from study.yaml.**
   The runner I hand-wrote for each study is ~150 lines of near-identical
   boilerplate. A `pbg-scaffold-runner <study-slug>` command should write
   it from the YAML's `simulation_set` + `readouts` declarations. Combined
   with (1), every study would auto-light-up.

For now: hand-authoring two more preview HTMLs (dnaa-03 from the seed0
run; dnaa-04 as a planning-phase placeholder).

---

## Friction (continued): Simulations DB silently empty until backfill

User caught this after the dnaa-03 run: "the Simulations DB is also not
showing any simulations ran. This can't be right." Indeed — the dashboard's
``/api/simulations`` endpoint walks two SQLite sources:

  - ``.pbg/composite-runs.db`` (workspace-wide)
  - ``studies/<name>/runs.db`` (per-study)

My hand-rolled runners (``studies/<slug>/sims/run_*.py``) wrote raw JSON
to ``out/<study>/`` and never opened either SQLite. The dashboard
correctly reported "0 simulations" even though three studies had produced
13 distinct runs spanning hours of wall time.

I wrote ``scripts/backfill_runs_db.py`` to walk the JSON outputs and
insert equivalent ``runs_meta`` rows. Result: 13 entries now visible in
the Simulations tab. But:

  - This is a one-shot backfill. The next run from a fresh runner will
    again be invisible until I re-run the script.
  - The per-step ``history`` table (which SQLiteEmitter normally owns)
    is still empty. The dashboard's "auto-viz from runs.db" path can't
    render charts without it. My ``embed_visualizations`` workaround
    covers the figures, but the in-dashboard auto-viz Step pipeline
    stays unused.

**Improvement suggestions for the meta-agent:**

1. **A runner mixin / decorator** (``@pbg_runner``) that wraps a runner
   function to: (a) generate a run_id, (b) open studies/<slug>/runs.db
   via composite_runs.connect, (c) save_metadata at start, (d) install
   an SQLiteEmitter that captures per-step state into the history
   table, (e) complete_metadata at end. The runner author writes ONLY
   the per-step physics; the bookkeeping is automatic.

2. **A "register existing JSON output" CLI**: ``pbg-register-run <study>
   <json-path>``. So if a one-off runner produced JSON without the
   mixin, you can still backfill in one command. (Effectively my
   backfill_runs_db.py but as a first-class workflow tool.)

3. **A dashboard warning panel** on the Simulations tab when
   ``out/<study>/*.json`` files exist that aren't represented in
   runs.db: "Detected 5 unregistered runs. Run ``pbg-register-run`` to
   backfill, or rerun via the runner mixin to capture automatically."

The pattern is consistent: the dashboard has rich machinery to display
data, but the runners I'm writing don't know how to feed that machinery.
Closing that loop would let every study auto-light-up.

---

## Session 2 (2026-05-17, evening) — infrastructure pass on the runner ↔ runs.db ↔ viz pipeline

**Context for the listening agent.** A second Claude session resumed this worktree
to address the "visualizations stay empty" friction. I implemented the
`@pbg_runner` decorator + two `local:DnaAStateVisualization` /
`DnaABoxOccupancyVisualization` classes + fixed two latent bugs in the
vivarium-dashboard library. End-to-end now produces a 9.7KB Plotly chart with
real data. The friction points below are what surfaced *while doing that work*
— they are recommended infrastructural improvements ordered by ROI.

Files touched (for cross-reference):
- `pbg-superpowers/pbg_superpowers/runner.py` (new, upstream)
- `vivarium-dashboard/.../lib/simulations_index.py` (bug fix)
- `vivarium-dashboard/.../lib/investigations.py` (bug fix + helper)
- `v2ecoli/visualizations/dnaa.py` (new, workspace)
- `v2ecoli/visualizations/__init__.py` (registration)
- `studies/dnaa-0{1,2}/sims/run_*.py` (converted)
- `studies/dnaa-02-atp-hydrolysis/study.yaml` (`inputs_map` blocks)

### Session-2 / Friction #11 — Workspace pins upstream pbg-superpowers to a git SHA, not an editable install

`pyproject.toml` declares `pbg-superpowers = { git = ".../pbg-superpowers.git", branch = "main" }`
and `uv.lock` pins a specific commit. The local
`/Users/eranagmon/code/pbg-superpowers/` editable clone is invisible to the
workspace `.venv` until you manually `.venv/bin/python -m pip install -e ...`
(no pip in a uv venv — need `uv pip install -e <path> --python .venv/bin/python`).

For an AI agent iterating on upstream APIs, this is a hidden multi-step
prerequisite. The workspace had no `make dev-install-upstream` or `scripts/dev-link.sh`
target to switch from pinned-git to editable-local for in-flight dev.

**Improvement suggestions:**
- Add `scripts/dev-link.sh <package-name>` that locates the editable clone in
  a configured search path (e.g., `$HOME/code/<pkg>`) and `uv pip install -e`s it.
- Document the "editable-install dance" in AGENTS.md or a top-level `DEV.md`.
- Even better: a pre-flight check in `pbg-workspace`-style scaffolding that
  detects when `pbg_superpowers` or `vivarium_dashboard` has a sibling clone
  under `~/code` and offers to swap to editable on workspace creation.

### Session-2 / Friction #12 — `address: local:Foo` references to non-existent classes silent-fail

dnaa-02/03/04 declared `address: local:DnaAStateVisualization` /
`local:DnaABoxOccupancyVisualization` before those classes existed. The dashboard's
behavior:
- "Registered visualization modules" panel: renders the *name* as plain text
  (the address resolves to None, but the template doesn't differentiate).
- Auto-viz pipeline: silently picks up no Visualization instance, contributes
  nothing to the Visualizations tab.
- Downloadable HTML report: same; chart panel just stays empty.

A biologist sees "DnaA state" as a viz name with no chart and can't tell whether
they're waiting for a run or whether the class itself is missing.

**Improvement suggestion:** a workspace lint (`pbg-report` lint, pre-commit hook,
or dashboard render-time check) that walks every `study.visualizations[].address`,
resolves the class against the workspace's `core.link_registry`, and FAILS LOUDLY
with "address `local:DnaAStateVisualization` is not registered — did you mean to
add it to `v2ecoli/visualizations/`?" Lint should also flag empty `inputs_map`
when the class declares non-trivially-typed inputs, since flat port names rarely
match nested observable paths (see #14 below).

### Session-2 / Friction #13 — Pre-existing bug: `simulations_index._ts` mixes REAL float and TEXT ISO

When a `runs.db` accumulates rows from both `runs_meta` (`started_at REAL`
inserted by `composite_runs.py`) and `simulations` (`started_at TEXT ISO`
inserted by SQLiteEmitter), the merged-list sort in
`simulations_index.list_simulations` raises `TypeError: '<' not supported between
instances of 'str' and 'float'`. Frontend gets `{"error": "simulations index failed: …"}`
and the Simulations tab is *empty across the entire workspace*.

This bug was latent under the friction-log #11 / Session-1 backfill (which only
wrote `runs_meta`). The moment `@pbg_runner` produced both kinds of rows in the
same db, the dashboard's whole sims tab broke.

**Why this matters for future work:** mixed-source SQLite aggregation needs a
normalising layer at the application boundary. Any helper that aggregates rows
written by two different writers (SQLiteEmitter vs `composite_runs`) is a
bug magnet.

**Improvement suggestion:** convergence on a single schema. Either:
- Promote `composite_runs.py` to pbg-superpowers and have SQLiteEmitter / the
  dashboard / pbg_runner all use its `save_metadata` / `complete_metadata`
  conventions, OR
- Add a normalised TIMESTAMP view to the runs.db schema so consumers don't
  re-encode this case-by-case.

The session's quick fix in `simulations_index._ts` (parse ISO → float) is a
band-aid, not the architectural fix.

### Session-2 / Friction #14 — `build_viz_composite` resolved only top-level observables

The observable-gather pipeline (`investigations.py::gather_runs`) records
per-tick state as `{top_level_key: [tick_0_value, tick_1_value, …]}`. Top-level
only. So `{listeners: [{dnaA_cycle: {atp_count: 113}}, …]}` ends up as a single
"listeners" series of opaque blobs.

`build_viz_composite` then did `observables.get(inputs_map_value)` — flat key
lookup. So `inputs_map: listeners.dnaA_cycle.atp_count` returned None and the
Visualization saw an empty input port. The class's "no data" branch rendered, the
chart was blank, no error logged. *Extremely* hard to debug — the data IS in
`history.state`, the inputs_map IS spelled correctly, but the resolver can't reach it.

The session's fix added a `_resolve_observable(observables, path)` helper that
walks dotted segments into per-tick dicts.

**Why this matters:** the *natural* place a biologist writes an observable is the
listener field path. The fact that the dashboard handled only flat keys made
nested-listener inputs invisible. This was the single biggest "silent failure"
of the session.

**Improvement suggestions:**
- Promote `_resolve_observable` from a helper to a documented contract in the
  investigations API.
- Add a render-time warning when an `inputs_map` value resolves to a series
  whose ALL values are None — that's the signal that "the resolver couldn't walk
  the path" vs "the run had no data."
- Per-viz "missing inputs" panel in the dashboard, listing each declared input
  port and whether it bound to ≥1 non-null value.

### Session-2 / Friction #15 — Viz rendering only fires on `/api/study-run-baseline`

`_render_study_visualizations` is called from the dashboard's run-baseline
handler (around line 1342 in server.py). Runs launched any other way — CLI
runner via pbg_runner, future MCP tool, batch-mode rerun — never trigger viz
generation. The runs.db has the data, the viz classes are registered, but no
HTML lands under `studies/<study>/viz/`.

Net effect: an AI agent that runs simulations programmatically (as I did in this
session) sees a perfectly populated `runs.db` and a perfectly empty Visualizations
tab. The plumbing that connects them only fires from one specific UI button.

**Improvement suggestions:**
- Fold viz rendering into the `pbg_runner` exit path: at end-of-run, look up
  the study spec, call `render_visualizations(spec, study_dir, name, core_registry, build_and_run)`,
  write the HTML files alongside the data. Closes the loop for every CLI run.
- Or: a `pbg-render-viz <study> [run_id]` CLI in pbg-superpowers that renders
  on demand against an existing run.
- Either way, decouple viz rendering from "the dashboard ran the simulation."

### Session-2 / Friction #16 — Dashboard URL routing intercepts `/studies/*` before static serving

`do_GET()` checks `self.path.startswith("/studies/")` FIRST and dispatches to
the study-detail page handler. The generic static-file fallback at the bottom
of `do_GET()` is never reached for `/studies/<x>/viz/<name>.html`. So even
though `_render_study_visualizations` writes those files, no URL serves them.

A biologist (or AI agent) naturally types `http://.../studies/dnaa-02-atp-hydrolysis/viz/dnaa_state.html`
expecting to see the file. Gets the study-detail HTML instead.

**Improvement suggestion:** require the `/studies/` route to match *exactly*
`/studies/<slug>` (or with query params), not any deeper subpath. Fall through
to static serving for `/studies/<slug>/<anything-else>`. One-line dispatch tweak.

### Session-2 / Friction #17 — Three-repo edit dance with no atomic-coordination story

The session edited three repos:
- `pbg-superpowers` (`runner.py` — new module)
- `vivarium-dashboard` (`simulations_index.py`, `investigations.py` — bug fixes)
- `v2ecoli` worktree (viz classes, study.yaml, runners)

Each will need its own commit/PR. If a workspace bump pulls the new dashboard
SHA but not the new pbg-superpowers SHA (or vice versa), the workspace will
break in subtle ways (e.g., `_resolve_observable` import not found, OR
`pbg_runner` import not found in a study runner that depends on it).

**Improvement suggestions:**
- A `make-coupled-release` script in pbg-superpowers that bumps both itself
  and vivarium-dashboard in lockstep and emits coordinated PR drafts.
- Workspace `uv.lock` should explicitly tie pbg-superpowers + vivarium-dashboard
  to compatible commits, with a documented compatibility-matrix entry.
- Integration test in pbg-superpowers that requires *both* its own and
  vivarium-dashboard's expected interfaces to coexist (i.e., test
  `_resolve_observable` symbol presence, `inject_sqlite_emitter` signature,
  etc., so a breaking change in one repo CI-fails the other).

### Session-2 / Friction #18 — The cross-tier observable contract is unwritten

The data path from Step output to rendered viz is FIVE hops:

1. `Step.outputs()` declares schema (with `overwrite[…]` or not — friction #3 of session 1)
2. Composite's emit step records selected fields per tick
3. SQLiteEmitter writes `history.state` as JSON blob
4. `investigations.gather_runs` flattens TOP-LEVEL keys into per-tick series
5. `build_viz_composite._resolve_observable` walks dotted paths into per-tick dicts
6. `Visualization.update(state)` consumes the resolved arrays

Each hop can silently drop data. The friction-log entries #3 (overwrite[]), #14
(nested resolution), and the new fixes all live on this path. There is no
end-to-end test or doc that asserts "if I declare a listener field X, a viz
declaring `inputs_map: foo: listeners.X` will receive non-empty data."

**Improvement suggestion:** an integration test fixture in pbg-superpowers that:
- Constructs a minimal composite with one Step emitting `listeners.demo.count = 42` per tick
- Wraps a Composite + SQLiteEmitter via `pbg_runner`-style setup
- Builds a `local:DemoViz` with `inputs_map: count: listeners.demo.count`
- Renders and asserts the resulting HTML contains "count: 42" or similar marker

Any regression in *any* of the 5 hops fails this test. Catches all 4
session-2 bugs (#13, #14, #16, #17 dep-skew) AND friction #3, #5, #9 from
session 1.

### Session-2 / Friction #19 — Inferring the editable / pip / uv install layout

When I needed to install `pbg-superpowers` as editable, the workspace `.venv/bin/`
had no `pip`. Took two cycles (`pip` not found → check if uv-managed venv → use
`uv pip install -e <path> --python <venv-python>`). The correct command isn't
obvious from the venv contents alone.

**Improvement suggestion:** include either `pip` or a `dev-install` shim in
the workspace `.venv` so AI agents and humans don't have to recognise the uv
convention separately. Or a top-level `Makefile` target
`dev-install-editable PKG=pbg-superpowers` that abstracts this.

---

## Wins from this session worth preserving

- **Parallel Explore-agent dispatch** for two independent surveys (runs.db
  schema + viz-class interface). 30-second 90% design in front of me. This is
  the right pattern for "I need to understand two orthogonal subsystems
  before writing code."
- **`_resolve_observable` is small and testable** — promote it to a documented
  helper rather than a private function.
- **The `pbg_runner` context-manager shape** (yields `RunContext` with `.run_id`,
  `.db_path`, `.emitter_config`, `.heartbeat()`, `.n_steps`) keeps workspace-
  specific emitter setup (v2ecoli's `set_emitter_override` vs vivarium-dashboard's
  `inject_sqlite_emitter`) out of pbg-superpowers — pbg_runner only owns the
  `runs_meta` row + the emitter config dict, hands the runner author the dict
  and lets them pick how to install the emitter. Right separation of concerns.

---

## Biology / modelling friction (deferred but documented)

### Bio-friction #B1 — dnaa-02 passes via a monkey-patch with no alternative evaluated

The dnaa-02 biological_validation verdict is currently PASS, but only because
`_disable_dnaa_adp_equilibrium()` mutates the live ecoli-equilibrium Step's
`stoichMatrix` to zero out MONOMER0-4565_RXN. This is option B in the
EQ-04 design-pivot block. Options A (locked DnaA-ADP-locked[c] species) and
B' (re-calibrate the equilibrium reverse-rate instead of disabling) are
documented but not tested.

dnaa-03/04 will inherit this pivot. If option A or B' would have produced
materially different DnaA-ADP dynamics, the downstream studies are calibrating
against the wrong baseline.

**Recommended:** a planned follow-up study `dnaa-02f-equilibrium-cleanup`
(scope: three-variant cleanup, runs in parallel with dnaa-03) that compares
all three pivots on the ATP-fraction band test AND a downstream-isolation
test for MONOMER0-160 dynamics. User approved this in session 2 but we
pivoted to the dashboard work before scaffolding it.

### Bio-friction #B2 — Nothing forces structural alternatives to be evaluated before downstream studies start

The investigation chain (dnaa-01 → dnaa-02 → dnaa-03 → dnaa-04) treats each
study as "passing the gate ⇒ next study can start." But a *passing* verdict
that rests on an unvalidated structural choice (like the monkey-patch above)
silently propagates risk forward.

**Improvement suggestion:** when a study's verdict cites a
`design_pivot_required` entry as "resolved-by-pivot," automatically spawn a
parallel-track follow-up study seed in `follow_up_studies` block listing the
unchosen alternatives. The downstream studies' `pipeline_gate.prerequisites`
should include "either dnaa-02f resolved OR dnaa-02 pivot validated by
expert sign-off."

### Bio-friction #B3 — `expert_decisions_needed` blocks rot without a UI surface

Both `dnaa-02-EQ-01` (equilibrium rate calibration) and `dnaa-02-EQ-04`
(snapshot-semantics tug-of-war) are documented with full context, alternatives,
and `asked_to: TBD`. There's no inbox UI for an expert to find these, and no
notification when a downstream study reaches a `blocks:` entry that names them.

**Improvement suggestion:** see session-1 friction #6 ("expert-decisions-needed
loop is one-way"). Still valid; even higher priority now that there are 2+
unresolved EQs blocking biological_validation across studies.

