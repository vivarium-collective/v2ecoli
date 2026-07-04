# Report cards as process-bigraph `as_step` types — Design

**Status:** spec (approved in brainstorm 2026-06-28). v2ecoli-only. Supersedes the
bespoke `@report_card` registry (merged in #303/#305) and the proposed
"extract the registry to pbg-superpowers" path — both are replaced by making
report cards first-class process-bigraph **Steps** registered in the
bigraph-schema **core**.

## Goal

Make each report card a proper process-bigraph type: an `@as_step`-decorated
Step with a typed `inputs`/`outputs` interface, registered in the core, resolved
by name. This replaces the bespoke `@report_card`/`REGISTRY`/`render`/`CardContext`
machinery. The shareable infrastructure becomes process-bigraph itself
(`as_step` + the core) — any project whose `build_core` registers the card Steps
gets them; nothing bespoke to extract.

## Decisions (brainstorm 2026-06-28)

1. **Execution model: typed Step, harness-invoked.** The card is an `as_step`
   Step (typed I/O, core-registered), but the harness orchestrates: it loads the
   v2ecoli+vEcoli zarr → builds the comparison data, resolves the card via
   `core.access`, instantiates it, and calls `update(state)`. Not a full analysis
   composite (can be wired into one later); not the v2ecoli `Analysis` framework.
2. **Inputs: pragmatically typed.** Type the structural contract
   (`condition`/`seeds`/`generations`/`variant` scalars, `observables` as a
   tree of per-seed stat records) but leave each per-seed stat record a loose
   `map`/`tree[float]` — meaningful, introspectable, not brittle.
3. **Migration: full replacement.** Convert all 5 cards to `as_step`; drop
   `REGISTRY`/`@report_card`/`render`/`get`/`all_names`/`CardContext`; the harness
   resolves cards via the core.
4. **Pure Step.** `update()` returns `{card_html, verdict, axes}`; the *harness*
   writes them to `viz/report_card/<card>.{html,verdict.json}` and aggregates the
   per-condition verdict. The dashboard + the `viz/report_card` contract are
   unchanged.
5. **No pbg-superpowers change.** `as_step` is in process-bigraph (already a
   dep); registration is explicit `core.register_link` in `build_core` (the
   mechanism v2ecoli already uses for emitters/`ShapeStep`/`KetchupEstimator`).
   `discover_packages` is NOT used here.

## The card Step

Each card module (`config`, `parca`, `standard`, `statistical`, `config_diff`)
becomes an `update_<name>_report_card(state)` function decorated with `as_step`:

```python
from process_bigraph.composite import as_step

@as_step(
    inputs={
        "condition": "string", "seeds": "integer", "generations": "integer",
        "variant": "integer",
        "observables": "tree[list[map]]",   # observable -> [per-seed stat record] (loose map)
        "plot_trajs": "tree[any]", "v2_bounds": "list[float]", "config": "tree[any]",
    },
    outputs={
        "card_html": "overwrite[string]",
        "verdict": "overwrite[string]",          # within_tol|drift|mismatch|ungraded
        "axes": "overwrite[list[tree[any]]]",    # the verdict_axes (id/label/verdict/value/meter/detail)
    },
    name="standard_report_card", aliases=["standard"],
)
def update_standard_report_card(state):
    # the existing standard-card logic, reading from `state` not a CardContext
    ...
    return {"card_html": html, "verdict": verdict, "axes": axes}
```

- The function name MUST be `update_*` (as_step requirement); `name=` gives the
  core key (`<card>_report_card`), `aliases=["<card>"]` the short alias.
- Existing logic is reused verbatim where possible: `standard` wraps
  `runs_section`+`eval_section`; `statistical` wraps `build_report_card`;
  `parca` grades `parca_section` rows; `config`/`config_diff` render the config
  and output `verdict="ungraded"` with `axes=[]`.
- The body reads `state["observables"]` (the former `ctx.per_obs`),
  `state["config"]`, `state["seeds"]`, etc.
- The exact bigraph-schema **type strings above are illustrative** — the plan's
  first task confirms the real syntax for nested/loose types (e.g. whether it's
  `tree[list[map]]`, `map[list[map]]`, or `tree[any]`) against the installed
  `bigraph_schema` core and uses the validated forms verbatim thereafter.

## Registration & resolution

The card Step classes register into the core in `v2ecoli/core.py::build_core`,
alongside the existing `register_link` calls:

```python
    from scripts._compare.report_cards import REPORT_CARD_STEPS  # {name: StepCls}
    core.register_links(REPORT_CARD_STEPS)
```

`REPORT_CARD_STEPS` is assembled by the card package's `__init__` (importing each
card module collects its `as_step` class under its `name`). Resolution:
`core.access("standard_report_card")` returns the Step class. Any project's
`build_core` that calls `register_links(REPORT_CARD_STEPS)` gets the cards —
that's the infrastructure-wide sharing, via the core.

## Harness invocation (replaces `render()`)

`scripts/comparison_report_card.py::assemble_from_studies` — for each card the
study declares:

```python
    core = build_core()
    ...
    for card in spec.cards:
        step_cls = core.access(f"{card}_report_card")
        step = step_cls(config={}, core=core)
        state = {
            "condition": spec.condition, "seeds": spec.seeds,
            "generations": spec.gens, "variant": 0,
            "observables": per_obs, "plot_trajs": plot_trajs,
            "v2_bounds": v2_bounds,
            "config": {"condition": spec.condition, "seeds": spec.seeds,
                       "generations": spec.gens, "cards": spec.cards},
        }
        out = step.update(state)        # {card_html, verdict, axes}
        # collect sections for the combined report (out["card_html"]),
        # card_verdicts[card] = {"verdict": out["verdict"], "axes": out["axes"]}
        viz_cards.append({"name": card, "html": out["card_html"],
                          "verdict": out["verdict"], "axes": out["axes"]})
    write_report_cards(study_dir, viz_cards)            # unchanged helper (signature updated)
    write_condition_verdict(verdict_root, name, card_verdicts)  # unchanged
```

`viz_cards.write_report_cards` is adjusted to take pre-rendered `html` per card
(it previously rebuilt HTML from `sections`); the verdict aggregation
(`verdict.build_condition_verdict`) is unchanged.

## Removed / changed

- **Removed** from `scripts/_compare/report_cards/__init__.py`: `REGISTRY`,
  `@report_card`, `render`, `get`, `all_names`, `CardContext`. Replaced by
  `REPORT_CARD_STEPS` (the `{name: StepCls}` map) for core registration.
- **Changed:** the 5 card modules (→ `as_step` `update_*` functions);
  `assemble_from_studies` (→ core invocation); `viz_cards.write_report_cards`
  (takes `html` not `sections`); the comparison tests (→ Step interface).
- **Unchanged:** `verdict.py`, `materialize.py`, `study_spec.py`, the
  `runner`/`scaffold`/CLI, the dashboard, and the `viz/report_card` +
  `tests: [{kind: report_card}]` contract.

## Data flow

```
v2e-compare run -> harness loads zarr -> comparison data per study
  core = build_core()  (REPORT_CARD_STEPS registered)
  for card in study.cards:
     core.access("<card>_report_card") -> step.update(state) -> {card_html, verdict, axes}
  harness writes viz/report_card/<card>.{html,verdict.json} + per-condition verdict
Dashboard: unchanged (embeds viz/report_card by the contract)
```

## Error handling

- Missing observable in `state["observables"]` → the card logic already maps to
  `not_compared`/`ungraded`; the loose `tree`/`map` input does not reject.
- `core.access` on an unknown card → a clear KeyError-style failure naming the
  card (surfaced by the harness, not a silent skip).
- A card whose `update` omits an output key → the `overwrite[...]` outputs schema
  + a harness assertion catch it.
- All reads/writes `encoding="utf-8"`.

## Testing

- **Per card Step:** `core = build_core(); step = core.access("standard_report_card")(config={}, core=core); out = step.update(synthetic_state)` → assert `out["verdict"] in {within_tol,drift,mismatch,ungraded}`, `out["axes"]` shape, and (for graded cards) at least one non-ungraded axis on a gradeable fixture. Mirrors today's `test_card_verdicts` via the Step interface.
- **Registration:** `core.access("standard_report_card")` and the `"standard"` alias both resolve; all 5 names registered.
- **Harness:** `assemble_from_studies` renders via core invocation and writes the viz cards + per-condition verdict (adapt the existing `test_assemble_studies`).
- **viz_cards:** `write_report_cards` writes the supplied `html` + verdict sidecar (adapt `test_viz_cards`).
- The 5-card conversion keeps the verdict values identical for the same input (a graded card with a 1% Δ still grades `within_tol`).

## Scope (YAGNI)

- v2ecoli-only. No dashboard/contract change, no pbg-superpowers change, no new
  dependency (`as_step` ships with process-bigraph).
- Not building a full analysis *composite* (decision 1) — the Steps are
  harness-invoked; wiring them into a composite is a later, separate step.
- Not converting the v2ecoli `Analysis` framework — out of scope.
