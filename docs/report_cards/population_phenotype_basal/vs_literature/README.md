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
validation-data bundle** (`ecoli_sources.VALIDATION_BUNDLE_PATH`) rather than a
pinned run.

## Physiology axes (this card)

Three scalar physiology axes, graded by the **`literature`** criterion (measured
band + an optional first-principles `theoretical_max` ceiling):

| axis | model derivation | verdict |
|---|---|---|
| **Growth rate (μ)** | ensemble of per-cell `ln 2 / cell-cycle time` | within_tol — μ ≈ 0.81/h sits in the measured 0.68–0.81 band |
| **Biomass yield (Yxs)** | **direct mass balance** per cell: `ΔDW / ∫(q_glc·DW)dt` | **mismatch — first-principles violation** (0.83 > the 0.538 stoichiometric ceiling) |
| **Glucose uptake (q_glc)** | ensemble-mean `\|GLC[p]\| exchange flux` | mismatch — 5.1 vs measured 8.5–10.6 (model under-consumes) |

Read together: **the model grows at the right rate but takes up too little
glucose, producing a yield above the thermodynamic limit.**

**Yield is the *direct* mass-balance ratio** (g dry weight made / g glucose
consumed), integrated per cell — not the steady-state `μ/(q·M)` shortcut (which
ran ~7% higher and noisier). A carbon check falls out of the same integration:
the **implied biomass carbon ≈ 0.46 gC/gDW is physically plausible, so carbon is
conserved** — the violation is *energetic* (the model under-respires, routing
~99% of glucose carbon into biomass), not mass creation. A formal `=1`
carbon-closure axis is deferred (it needs the model's exact biomass C-content,
which isn't a single sim_data field).

### The plot

Each axis carries a `literature` plot: **green** = sim (per-cell violin + mean
diamond), **black** = experimental measurements (one marker per source, ±
uncertainty), **red dashed** = the theoretical-max ceiling (cited). Sources are
labelled on the plot, so each reference traces to its origin.

## Reference data

Curated in **ecoli-sources** (`validation_data/`, draft PR
vivarium-collective/ecoli-sources#3): measured basal physiology from Varma &
Palsson 1994, LaCroix 2015, Long 2017, Kavvas 2022 (3 of 4 strain-matched
MG1655) and the Varma 1993 theoretical-max ceiling. v2ecoli pins ecoli-sources
by git SHA in `pyproject.toml`.

## Regenerate

```bash
# re-render from the committed golden fixtures + current bundle (no sweep needed):
python scripts/render_basal_vs_literature.py

# re-bake the model fixtures from a blessed sweep (stamps fresh provenance):
python scripts/render_basal_vs_literature.py --from-sweep out/population_phenotype_basal  # physiology
python scripts/bake_model_metabolism.py --from-sweep out/population_phenotype_basal       # metabolism/proteome/composition/pools
```

The model values are **golden fixtures** — per-cell aggregates baked from one
blessed sweep (organized sim output, not reference data) — committed under
**`tests/fixtures/population_phenotype_basal/model_*.json`** so the card + tests
stay independent of the gitignored sweep. Each carries a `provenance` block
(commit + dirty/diff-sha + source-sweep identity; see `scripts/_provenance.py`).
The rendered outputs here (`report_card.html`, `*_reference.json`,
`report_card_verdict.json`) are **gitignored** — regenerated deterministically
from the fixtures + `ecoli_sources.VALIDATION_BUNDLE_PATH`. No new run required.

## Follow-ups (not in this card)

- Exchange-flux and proteome/fluxome axes vs literature (the reference data
  already supports them; the 13C fluxome is staged in ecoli-sources).
- A per-cell *yield* spread (currently a single ratio-of-means point).
