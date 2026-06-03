# Friction log: report figures + prose go stale when a study is re-run (2026-06-01)

## What happened (recurring)

Reviewer (Rashmi) feedback, repeated across rounds:

- 2026-05-31: "tests still show pending" → studies declared done but the report
  derived status from `runs[]`, which were absent. Fixed with
  `study_clarity_summary` + report-linter guards (pbg-superpowers#87).
- 2026-06-01: "It says the tests passed for study 3 but the plots have not been
  updated. Most sections of the report still carry results/conclusions from
  previous runs. The visualizations haven't been updated." — i.e. dnaa-2's
  verdict/tests were flipped to PASS (DnaA-ATP ~0.45) but:
  - the **figures** were still the Step-2 renders (~0.997, all seeds, May 30);
  - the **body prose** (report.result / interpretation / findings /
    conclusion_logic) still narrated the Step-2 "~0.997, blocked on Haochen"
    story as current.

The common root cause: **a study's run, its figures, and its derived prose are
three decoupled artifacts.** Re-running updates the run data; a human then hand-
edits the verdict; but nothing re-renders the figures from the new run or
reconciles the body prose. The report ends up internally contradictory (badge:
PASS; plot + text: the old failing result).

## Why the current flow makes this easy to get wrong

1. **Figures are rendered by bespoke per-study scripts** (`render_dnaa2_sixpanel.py`,
   etc.) pointed at a specific run dir, run **manually**. Nothing ties the
   rendered `reports/figures/<study>/<name>.html` to the study's *current*
   canonical run, so they silently persist from an old run.
2. **`study.yaml.visualizations`** lists figures by name; the dashboard auto-
   discovers the matching HTML. There's no check that the HTML is newer than the
   run, or that it came from the canonical run/seed.
3. **Body prose** (report.result/interpretation/findings) is free text; updating
   the verdict/tests doesn't flag the now-contradictory narrative.
4. `/pbg-study run-*` and the workflow driver finish a run but do **not** trigger
   a viz re-render or a prose-staleness check.

## Proposed framework improvements (pbg-superpowers / vivarium-dashboard)

1. **Run → viz coupling (auto-refresh on run).** On run completion,
   `/pbg-study run-*` and `v2ecoli.workflow.run` should re-render the study's
   declared `visualizations:` from the just-produced (canonical) run via a
   generic `render_study_viz`, writing to `reports/figures/<study>/`. Figures
   then track runs by construction. Bespoke render scripts become the
   per-viz implementation the generic refresher calls, not a manual step.

2. **Figure provenance + staleness audit (pbg-report Pass A).** Stamp every
   rendered figure with a sidecar `<name>.meta.json` carrying
   `{run_id, run_mtime, seed, rendered_at}`. Add a Pass A check that flags any
   declared visualization whose figure is **older than the study's canonical
   run** or whose `run_id`/`seed` ≠ the canonical one — "figure stale vs run."
   This would have caught dnaa-2 directly (figures from Step-2 while canonical =
   Step-3).

3. **Verdict ↔ figure ↔ prose numeric-drift check (extend Pass A).** The audit
   already does "verdict↔chart drift." Extend to: (a) a test marked PASS whose
   illustrating figure predates the passing run; (b) body prose
   (report.result/findings) containing numbers that disagree with the recorded
   test/verdict numbers (e.g. result says "0.997" but the test result is
   "~0.45"). Flag as stale prose.

4. **Reader-facing figure freshness.** The per-study report block should show
   "figures: rendered <date> from run <id> (seed N)" next to the
   Ran·Tests·Verdict strip, so a reviewer sees figure freshness at a glance
   (the visual analog of the status-clarity strip).

5. **One canonical viz per study by default; compactness.** Reviewer wants one
   seed shown, not all. The viz refresher should default to the canonical seed
   and not emit a figure per seed unless explicitly asked — keeps the
   visualization section short.

## Smallest first step

Item 2 (figure-staleness audit) is the cheapest high-value guard: it turns this
recurring silent failure into a loud pre-send finding, without changing the run
pipeline. Item 1 (auto-refresh on run) is the durable fix that prevents it.

## Progress (2026-06-01, this session)

- **Item 2 SHIPPED** — `figure_stale_vs_run` lint check (pbg-superpowers#91).
  Already earned its keep: caught dnaa-1's stale figure during this round.
- **Item 1 PARTIAL:**
  - Fixed `v2ecoli.library.parquet_viz.load_run_history` to find the hive in the
    workflow meta-composite's NESTED layout (`<run>/parquet/<exp>/history/`),
    which previously raised "no history/ directory" — the generic renderer no
    longer crashes on workflow output.
  - Added a post-sweep AUTO-RENDER hook to the workflow driver
    (`run_dnaa2_multiseed_isolated.py`): after a run it re-renders the canonical
    seed's figure from THAT run via the bespoke sixpanel renderer, so the plot
    can't lag the data.
- **Architectural finding that scopes the REAL framework fix:** the generic
  `render_study_visualizations` is **inputs_map-based** (wires named ports to
  parquet columns, calls the Visualization). Viz that read parquet directly
  (the dnaa sixpanel/forms — no `inputs_map`) can't be rendered by it, and a
  multiseed hive would mix seeds. So a fully-general auto-refresh-on-run needs
  (a) seed-awareness in the generic renderer and (b) a study-declared
  **`figure_refresh:`** command list that the framework run-completion
  (`/pbg-study run-*`, `v2ecoli.workflow.run`) invokes for bespoke viz. That
  convention + hook is the remaining framework work; the per-driver hook above
  is the concrete stop-gap.

- **Item 1 SHIPPED (the durable fix)** — the study-declared **`figure_refresh:`**
  convention (pbg-superpowers#92). A study declares HOW to render its figures in
  `study.yaml` (command templates with `{run}`/`{study}`/`{figdir}`/`{ws}`/`{py}`
  placeholders); `pbg_superpowers.figure_refresh.refresh_study_figures()`
  substitutes + runs them on run-completion. dnaa-2 adopted it: the workflow
  driver's hardcoded importlib render hook is gone, replaced by a call that just
  names the finished run — the study owns the render command. Figures now track
  runs **by construction**, so #91 (the detector) has nothing to flag.
  - `{py}` = the invoking interpreter (`sys.executable`) closes the
    bare-`python`-vs-venv footgun: the subprocess shell otherwise resolves
    `python` from PATH, which in v2ecoli lacks `polars`/`unum`.
  - Verified end-to-end: real refresh re-rendered the canonical Step-3 figure
    from `parquet-runs/dnaa2-step3-mechanism` via the venv interpreter (ran 1,
    failed 0), mtime advanced, byte-stable content (same run data).
- **Remaining (smaller, deferred):** wire `refresh_study_figures` into the
  generic `/pbg-study run-*` + `v2ecoli.workflow.run` completion paths (dnaa-2
  proves the API; generalizing the call site is the next step), and the
  reader-facing "figures rendered <date> from run <id>" freshness line (item 4).

See also: [[pbg-editable-install-gap]], pbg-superpowers
`docs/conventions/handling-investigation-feedback.md` (verify-the-rendered-artifact).
