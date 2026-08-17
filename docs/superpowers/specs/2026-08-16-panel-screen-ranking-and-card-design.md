# Panel-screen grading — design

Status: **proposed** · 2026-08-16, **rescoped 2026-08-17** · builds on `#500` / `#503` / `#508`.

## Problem

`variant-sweep-phenotype` reads a **sweep along one axis** — vary one perturbation, watch an
observable move. A **panel screen** is a different readout over the same machinery: **N
distinct designs, optionally crossed with M environmental conditions, ranked against each
other.** Not "how does the observable move as this knob turns" but "which design wins, by how
much, and at what cost."

Execution needs nothing new — the whole-config node already runs it, one variant index per arm.

## What is already here (checked, not assumed)

The first draft of this note proposed a statistics module. **Most of it already exists**, so
this rescope removes it:

| Capability | Where |
|---|---|
| Cell-level aggregation — `by_cell` as `[[seed, gen, value]]`, `build_cell_records` | the population-phenotype card path |
| Seed/gen variance decomposition — `eta_seed`, `eta_gen`, `rho_gen` | `workflow/analysis._variance_decomposition`, mirrored in `library/report_card._decompose` |
| Welch two-sample test | `viva_superpowers.card_criteria._welch` |
| Grading vocabulary — `grade_axis`, `_band`, `_r2`, `_fit_threshold_linear` | `viva_superpowers.card_criteria` |
| Sweep reshaping / endpoints | `library/phenotype_sweep` |

**Genuinely absent:** BH / FDR **q-values** (no such function in this repo or in
`card_criteria`), and a **panel-shaped contrast** — N arms each against a *named reference arm*
with the testing family tracked. Welch is available as an axis-vs-band comparison, not a panel
one.

## What this proposes

**1. A panel-screen report card** (Gen-2 `ReportCardStep`) — the substance.

- **Fixture-graded**, like `acetate_overflow_card` reads a committed
  `model_overflow_baseline.json`. No live run to build or test.
- **Parameterised through `report_card_refs`**, like `genotype_build_integrity` takes
  `gene_ids`: the objective observable, the **reference arm**, and the **strata keys** are
  study inputs, not constants.
- Grades through the existing **`report_card_axis`** evaluator. **No new evaluator kind, no new
  `pass_if` op.**

**2. A q-value helper** — BH-FDR over a family of p-values. Small.

Nothing else. Aggregation, Welch and variance decomposition are **consumed** where they live.

## ★ Why the card's contract is the actual deliverable

The failure this is built to prevent was **not a missing function.** In a prior panel analysis
Welch was already being applied correctly per arm — the defect was **how the family was
constructed**: q-values pooled across two media, so nearly every arm in the second medium came
out "significant" when most of that signal was the medium, not the design.

A library function cannot prevent that, because the *caller* decides what the family is. A card
**contract** can: if the card requires `strata` keys, a panel cannot be graded without
declaring what the family is. Hence the emphasis on the card rather than on statistics.

There is a quieter sibling worth designing out at the same time: if an arm's label is derived
only from its design vector, the same design in two conditions yields **two arms with one
label**, and they silently share a row in anything keyed on it. **Arm identity must carry the
condition** — named study variants give that for free; a label computed at render time does not.

## Deliberately not in scope

- **No generic rendering layer.** Study-local `sims/*.py` is the convention here (14 of 59
  studies), and `genotype-01`'s panel table/chart is a good example of it. Tables, strip plots
  and Pareto views stay with the consuming study.
- No new sweep or execution mechanism.
- No product, pathway or organism content — a downstream (possibly private) repo supplies the
  config, observables and reference arm.

## Open

- **Where the q-value helper belongs** — here, or upstream in `viva_superpowers.card_criteria`
  beside `_welch`, where the rest of the statistics already live. Upstream seems more natural;
  flagged for review rather than decided.
- Whether the card grades *best-arm-vs-reference*, *ranking resolvability* (separation exceeds
  within-arm noise), or both — plausibly a study-level choice via `report_card_refs`.
- The public test fixture: synthetic, or captured from a public sweep.
- Whether `genotype-01` would consume the same card, as the first panel study in the repo.
