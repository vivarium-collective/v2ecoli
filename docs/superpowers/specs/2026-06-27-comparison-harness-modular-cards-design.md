# Comparison Harness v2 — Modular, Config-Driven Report Cards (Design)

**Status:** approved design → writing-plans next
**Author:** Eran Agmon (with Claude)
**Date:** 2026-06-27

## Goal

Clean up how the v2ecoli ↔ vEcoli comparison harness is *defined* and *reported*:
a single JSON manifest that pins the repos to compare, lists the vEcoli configs
to run, and assigns **modular, reusable report cards** to each config — rendered
into a report whose structure mirrors the manifest (overview + a section per
config, each showing its assigned cards).

## Background / current state

- The harness already exists in `vivarium-collective/v2ecoli`:
  `scripts/run_comparison_ensemble.py` (per-engine runner), `scripts/comparison_harness.sh`
  (orchestrator), `scripts/_read_spec.py` (parser), `scripts/comparison_report_card.py`
  (report assembly), `comparison_spec.json` (manifest).
- The report is already a list-of-sections assembly with an `overview_section` at
  the top and per-condition sections (parca, config, eval). It is **close** to the
  target structure.
- **Chris Long's report-card library is already on `main` and tracked** —
  `v2ecoli/library/report_card.py` (grading + HTML render with `<details>`
  collapsible "dropdown" viz bars + `within_tol/drift/mismatch` verdict pills),
  `card_criteria.py` (`rel_tol`/`ttest`/`flux_scatter`/`r2`), `card_plots.py`
  (`violin_strip`/`literature_strip`/`loglog_scatter` → inline SVG), `card_vectors.py`.
  The harness already calls it via `scripts/_compare/report_card_section.build_report_card`.
  Reference investigation: PR #235 (cplong90).
- Rendered cards live under **`docs/report_cards/`** (e.g.
  `vecoli_v2ecoli_conditions/{basal,with_aa,no_oxygen,succinate}`,
  `population_phenotype_basal/`). This output convention is established.

## What's wrong today (the cleanup targets)

1. The manifest **duplicates** `seeds`/`gens` per condition (separate entry) and
   **ignores variants** — runs are variant=0 only. vEcoli configs already carry
   `n_init_sims` (seeds), `generations` (gens), and `variants`.
2. Report cards are **not assignable** — the report runs a fixed per-condition
   section sequence; you cannot say "use the statistical card for baseline, the
   standard card elsewhere."
3. Card **code** is flat in `v2ecoli/library/`, not a modular package keyed by
   card name.

## Design

### 1. Manifest schema (`comparison.json`)

```jsonc
{
  "v2ecoli": { "repo": "https://github.com/vivarium-collective/v2ecoli", "commit": "<sha>" },
  "vecoli":  { "repo": "https://github.com/CovertLab/vEcoli",          "commit": "<sha>" },
  "defaults": { "cards": ["standard"] },
  "configs": [
    { "config": "configs/cond_basal.json", "cards": ["standard"] }
  ],
  "report": { "out": "out/report", "title": "..." }
}
```

- **`v2ecoli` / `vecoli`**: each `{ repo, commit }`. `commit` is the pin (the old
  `branch` field is dropped). A run records the resolved commits in the report.
- **`configs[]`**: each `{ config, cards?, note? }`. `config` is a path to a vEcoli
  config file (relative to the vEcoli fork root). **`seeds` (`n_init_sims`), `gens`
  (`generations`), and `variants` are read from that config — never specified in
  the manifest.** To change scale, point at a different config (vEcoli's
  `inherit_from` makes a 1-seed vs 4-seed sibling cheap).
- **`cards`**: list of report-card names for this config; falls back to
  `defaults.cards` when omitted.
- **`report`**: `{ out, title }` — output dir + report title.

`_read_spec.py` gains a `configs` mode that emits `config<TAB>cards(csv)` per entry
and resolves `seeds/gens/variants` by reading each referenced vEcoli config (via
the existing `config_adapter.resolve_vecoli_config_local`). The old `conditions`
mode + `defaults.seeds/gens` are removed.

### 2. Report-card registry (modular code package)

New package `scripts/_compare/report_cards/`:

- `__init__.py` — the registry: `REGISTRY: dict[str, Card]`, a `@report_card(name)`
  decorator, and `get(name) -> Card`. Importing the package registers all cards.
- One module per card:
  - `standard.py` → `standard_card(ctx)` — matched-time evaluation + run
    trajectories (today's `eval_section` + `runs_section`), the lighter card.
  - `statistical.py` → `statistical_card(ctx)` — wraps
    `report_card_section.build_report_card(...)` → `v2ecoli.library.report_card`:
    Chris's graded card with violin/strip distribution plots, the `<details>`
    dropdown viz bars, and `within_tol/drift/mismatch` verdict pills. **No new viz
    code** — this exposes the on-main library as an assignable card.
  - `parca.py` → `parca_card(ctx)` — ParCa / initial-state match (today's
    `parca_section`).
  - `config_diff.py` → `config_diff_card(ctx)` — vEcoli-vs-v2 config diff (today's
    `config_diff_section` / `config_sections_for`).

A **card** is `Callable[[CardContext], Section | list[Section]]`.

```python
@dataclass
class CardContext:
    config_name: str          # e.g. "basal"
    variant: int              # variant index (0 for the baseline variant)
    v2_dir: str               # dir holding v2ecoli_seed*.zarr for this (config, variant)
    ve_dir: str               # dir holding vecoli_seed*.zarr
    seeds: int                # from the config's n_init_sims
    gens: int                 # from the config's generations
    per_obs: dict             # extracted per-observable trajectories/values
    config: dict              # the resolved vEcoli config
```

`Section = {"title": str, "kind": "content", "html": str, "anchor": str,
"verdict": str | None}` — the existing section dict shape (so the assembler is
unchanged structurally).

Adding a future card = one decorated module; assigning it = a name in the manifest.

### 3. Full variant execution

- For each `configs[]` entry, read the config's `variants` dict and expand it into
  the variant matrix (variant index 0..N-1). `variants: {}` → a single variant
  (index 0), identical to today.
- `run_comparison_ensemble.py` gains a variant dimension: it runs **each
  (config, variant)** through both engines, writing stores to
  `<out>/<config_name>/variant_<i>/{v2ecoli,vecoli}_seed<NN>.zarr`. Variant
  application mirrors vEcoli's variant mechanism (apply the variant's parameter
  overrides to sim_data/config before the run); v2ecoli applies the same overrides
  so both engines run the matched variant.
- Stores are addressed by `(config, variant, engine, seed)` throughout.

### 4. Report assembly (manifest-mirroring)

`comparison_report_card.py main()` is rewritten to be manifest-driven:

1. **Overview** section at the top — one row per `(config, variant)` with its
   headline verdict(s), as today's `overview_section` (extended with a variant
   column).
2. For each `configs[]` entry, in manifest order:
   - a **config section group** titled by the config name;
   - for each variant, that config's **assigned cards** rendered in `cards` order,
     each card producing its Section(s).
3. The assembled report is written to `report.out` as
   `standardized_comparison_report.html`; per-card rendered bundles are also
   written under `docs/report_cards/<run_slug>/<config>/variant_<i>/` following the
   existing `docs/report_cards/` convention.

### 5. Example manifests (tracked, in-repo)

- **`comparison.5cond_1x4.json`** — 5 conditions
  (`configs/cond_{basal,with_aa,succinate,no_oxygen,acetate}.json`), each
  `cards: ["standard"]`, full variant execution. The condition configs carry
  `n_init_sims: 1, generations: 4`.
- **`comparison.baseline_4x4_statistical.json`** — baseline only
  (`configs/cond_basal_4x4.json`, an `inherit_from: cond_basal.json` sibling with
  `n_init_sims: 4, generations: 4`), `cards: ["statistical"]` (Chris's
  violin/dropdown graded card).

### Reuse vs new

- **Reuse (tracked, on `main`):** Chris's `report_card.py` + `card_criteria.py` +
  `card_plots.py` + `card_vectors.py` (violin/strip + `<details>` dropdown +
  grading); the `docs/report_cards/` output convention; the existing
  `run_comparison_ensemble` / `comparison_harness.sh` / `config_adapter` plumbing.
- **New:** the slimmed manifest schema (cards-per-config, no seeds/gens/variants
  duplication), the `scripts/_compare/report_cards/` registry package, variant
  expansion in the runner, the manifest-mirroring report assembly, the two example
  manifests.

## Testing

- **`_read_spec` / schema**: unit tests that a manifest resolves to the right
  `(config → cards)` map and that seeds/gens/variants come from the referenced
  config (extend `tests/test_read_spec.py`).
- **Registry**: a card registers under its name; `get("statistical")` returns a
  callable; an unknown card name fails loud.
- **Each card**: given a small fixture `CardContext`, the card returns a Section
  with a non-empty `html` and a verdict in `{within_tol, drift, mismatch,
  ungraded}` (reuse existing `report_card` fixtures).
- **Variant expansion**: a config with `variants: {}` yields one variant; a config
  with a real variant dict yields the expected matrix (fixture, no full sim).
- **Report assembly**: an overview at the top + one section group per config in
  manifest order, each containing exactly its assigned cards (assert on the
  assembled section list, not a full render).

## Out of scope

- Cloud/Ray orchestration changes (the existing route is unaffected; the manifest
  is the only new control surface).
- New grading criteria or plot types — cards reuse Chris's existing
  `card_criteria` / `card_plots`.
- Migrating other report consumers (sweep reports, s3 reports) onto the registry.
