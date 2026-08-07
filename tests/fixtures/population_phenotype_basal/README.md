# `population_phenotype_basal` model fixtures — FROZEN GOLDENS

These `model_*.json` files are **frozen regression goldens**. They are the
model side of the basal `vs_literature` card, baked once from one particular
ensemble and committed so the card and its tests grade **run-free** — CI has no
sweep, and `out/` is gitignored.

**They are not the live sim side of a comparison.** That distinction is the
whole point of this file, because these two roles pull in opposite directions:

| | frozen golden | live sim vector |
|---|---|---|
| should change when the model changes | **no** — a change fails a test | **yes** — it tracks the run |
| identity | a pinned ensemble, stated below | the run that produced it |
| lives in | this directory (committed) | `<out_dir>/sim_vectors/` (run-keyed, gitignored) |

An artifact doing both jobs at once is how a provenance string drifts from the
values it describes: a downstream consumer once read one of these files as the
live sim side while asserting a *different* ensemble in its provenance panel,
and rendered "104 cells" over numbers from the 20-cell bake below. Nothing in
the stack could detect it, because the file could not say which run it was.

## If you want the live sim vector

Use the run-keyed cache — it is keyed by
`(experiment_id, generation_lower_bound, extractor_version)`, so it always
states which run it came from and a new sweep yields a new entry automatically:

```python
from v2ecoli.library.sim_vector_cache import load_or_extract
env = load_or_extract("out/<sweep>", generation_lower_bound=3)
env["run"]["experiment_id"]      # which run these numbers are from
env["vectors"]["omics"]["proteome"]["vector"]
```

## What these were baked from

All five files: **4 seeds × 8 generations, `generation_lower_bound=3` → 20
cells**, model ref `b162243` (recorded in
`../population_phenotype_basal_reference.json`'s `stimulus`). Their own
`provenance` blocks are `reconstructed: true` — the bake predates provenance
stamping, so the producing commit is genuinely unknown rather than merely
unrecorded. Re-bake for an authoritative stamp.

Note the source sweep for that 20-cell ensemble is **no longer on disk**, so
these values cannot currently be reproduced by re-running the bake; a re-bake
would use a different ensemble and change every value the card grades. That is
a deliberate, separate decision, not a cleanup.

| file | contents | baked by |
|---|---|---|
| `model_physiology.json` | μ, q_glc, biomass yield by direct mass balance | `scripts/render_basal_vs_literature.py --from-sweep` |
| `model_metabolism.json` | G6P branch point + boundary exchanges | `scripts/bake_model_metabolism.py --from-sweep` |
| `model_proteome.json` | ensemble-mean copies/cell, `by_id` **and** `by_symbol` | `scripts/bake_model_metabolism.py --from-sweep` |
| `model_transcriptome.json` | ensemble-mean counts/cell `by_gene_id` | `scripts/bake_model_omics.py --transcriptome` |
| `model_composition.json` | macromolecular dry-mass fractions | `scripts/bake_model_metabolism.py --from-sweep` |
| `model_metabolite_pools.json` | intracellular pools vs Bennett | `scripts/bake_model_metabolism.py --from-sweep` |

## Joining to another data source

Join on **`by_id`** (EcoCyc monomer id) or **`by_gene_id`** (EcoCyc gene id),
never on `by_symbol`. Symbol spaces differ between databases and are not
injective across them; `by_symbol` exists because the literature reference this
card grades against is itself symbol-keyed, which is a single known source, not
a general join key.
