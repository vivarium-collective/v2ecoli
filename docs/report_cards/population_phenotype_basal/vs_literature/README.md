# `vs_literature` — experimental/theoretical reference mode

A third reference mode for the `population_phenotype_basal` card, alongside the
existing two:

| mode | reference | question |
|---|---|---|
| self-pin | a blessed prior run of this model | has behavior **drifted**? |
| `vs_vecoli` | a matched vEcoli (v1) ensemble | are v1 and v2 **equivalent**? |
| **`vs_literature`** | curated experimental + theoretical values | does the model match **reality**? |

The grader is reference-agnostic, so this mode reuses the same typed-criteria
machinery pointed at curated reference claims from the **ecoli-sources
validation-data bundle** (`VALIDATION_BUNDLE_PATH`) rather than a pinned run.

## First axis: basal biomass yield

`Yxs = μ / (q_glc · M_glc)` — derived from the μ and glucose-exchange flux the
basal ensemble already extracts. Graded against the `basal__biomass_yield`
slot, which distinguishes reference `kind`:

- **`measured`** — an experimental yield; deviation is a soft mismatch (a
  tracked gap, not necessarily a defect).
- **`theoretical_max`** — a first-principles ceiling (e.g. Varma 1993,
  0.538 gDW/g glucose). A model output **above** the ceiling is a *harder,
  differentiated failure* — it exceeds the network's stoichiometric limit, no
  adequacy judgment required.

## Status

Work in progress. This document seeds the mode; the yield axis, the
`pin_literature_reference.py` script, and the rendered card follow. Depends on
the ecoli-sources validation-data subsystem (companion PR
vivarium-collective/ecoli-sources#3). Sibling perturbation-response work: #235.
