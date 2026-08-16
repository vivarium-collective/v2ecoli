# Panel-screen ranking readout + report card — design

Status: **proposed** · 2026-08-16 · builds directly on `#500` / `#503` / `#508`.

## Problem

`variant-sweep-phenotype` (`#508`) reads a **sweep along one axis** — vary one perturbation,
watch an observable move, dose-response style. Its member template sweeps three variant
indices and compares them.

A **panel screen** is a different readout over the same machinery: **N distinct designs, each
possibly crossed with M environmental conditions, ranked against each other.** The question is
not "how does the observable move as this knob turns" but "which of these designs wins, by how
much, and at what cost."

Nothing about the *execution* differs — the whole-config node already runs it, one variant
index per arm. What is missing is the **readout**: aggregation, contrast, ranking and grading
across a panel.

## What this proposes

Two artifacts, both perturbation-agnostic. Neither adds a sweep mechanism.

### 1. A ranking readout (analysis)

Per-arm summary over the panel:

- **Cell-level aggregation.** Time-average within a cell first, then aggregate across cells —
  never statistics over raw timepoints. Report `n` cells per arm.
- **Contrast against a named reference arm**, declared by the study rather than assumed to be
  index 0.
- **Rank order** on a chosen objective observable, with mean ± SEM per arm.
- **A cost view** — objective against growth rate, with the Pareto front drawn, so a design
  that wins on product but collapses growth is visibly distinguishable from one that doesn't.

### 2. A panel-screen report card (Gen-2 `ReportCardStep`)

Grades the panel's top-level outcome. Follows the existing card conventions exactly:

- **Fixture-graded**, like `acetate_overflow_card` reads a committed
  `model_overflow_baseline.json` — so the card needs **no live run** to build or test.
- **Parameterised through `report_card_refs`**, like `genotype_build_integrity` takes
  `gene_ids`. The objective observable and the reference arm are **study inputs, not
  constants**, so the same card serves any panel.
- Grades through the existing **`report_card_axis`** evaluator. **No new evaluator kind and no
  new `pass_if` op** — the closed sets stay closed.

## ★ One methodological requirement worth stating up front

**When a panel is crossed with two or more environmental conditions, both the reference arm
and the multiple-testing family must resolve _within_ a condition.**

If a panel of designs is swept in, say, two media and every arm is compared against a single
reference from one of them, the comparison attributes the **condition** effect to the
**design**. In a real case this made ~all arms in the second condition come out "significant"
against a reference from the first. The same applies to any FDR correction: the family is per
(observable × condition), not global.

There is a second, quieter version of the same failure: if an arm's display label is derived
only from its design vector, then the same design in two conditions produces **two arms with
one label**, and they silently share a row in any plot or table keyed on it. **The arm
identity must carry the condition** — which a study's named variants give for free, and a
label derived at render time does not.

Both are cheap to get right at design time and expensive to find later.

## Non-goals

- Not a new sweep or execution mechanism — `#500`/`#503` already provide it.
- Not a new evaluator kind, `pass_if` op, or measure kind.
- Not tied to any product, pathway or organism-specific content. A downstream (possibly
  private) repo supplies the config, the observables and the reference arm.

## Open

- Whether the ranking readout is a registered analysis, a visualization, or both.
- Whether the card grades the panel's *best* arm against a target, the *separation* between
  arms, or both — likely a study-level choice rather than a card-level one.
- The public test fixture: synthetic, or captured from a public sweep once a public whole-config
  run is available.
