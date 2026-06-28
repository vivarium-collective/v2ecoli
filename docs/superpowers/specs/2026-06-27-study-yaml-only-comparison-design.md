# Study-YAML-only comparison framework — Design

**Status:** spec (approved 2026-06-27). Supersedes the manifest-JSON model from
`2026-06-27-comparison-investigation-unification-design.md`: the comparison
manifest JSON, the scaffold, and the validator are **removed**. The
investigation/study YAML is now the single source of truth, triggered directly
by the CLI. Folds into PR #303 (unmerged).

## Goal

One artifact type — the **study YAML** (grouped under an **investigation YAML**)
— specifies both *how to run* a v2ecoli↔vEcoli comparison and *how the dashboard
shows it*. A CLI runs a single study or a whole investigation directly from YAML.
No parallel manifest JSON; no scaffold; no drift validator (nothing to match).

## Why (the redundancy this removes)

The manifest+study model duplicated the `(condition → cards)` mapping in two
files and policed it with a validator. Collapsing to one artifact removes the
duplication by construction.

## Artifacts

**Investigation YAML** — shared execution context + study list:
```yaml
schema_version: 4
name: v2ecoli-vecoli-comparison
title: v2ecoli ↔ vEcoli comparison
comparison:                       # shared execution context (provenance + defaults)
  v2ecoli: {repo: https://github.com/vivarium-collective/v2ecoli, commit: ""}
  vecoli:  {repo: https://github.com/CovertLab/vEcoli, commit: ""}
  vecoli_dir_env: V2E_VECOLI_DIR  # env var naming the local vEcoli fork checkout
  v2_cache: out/cache_full
  ve_cache: out/compare_harness/vecoli_parca
  defaults: {cards: [config, parca, standard]}
studies: [basal, with_aa, succinate, no_oxygen, acetate, basal_4x4]
```

**Study YAML** — the run spec AND the dashboard fields in one file:
```yaml
schema_version: 4
name: basal_4x4                   # study identity = store/verdict/card key
investigation: v2ecoli-vecoli-comparison
condition: basal                  # biological vEcoli condition simulated (--condition)
comparison:
  seeds: 4
  generations: 4
  cards: [config, parca, statistical]   # the ONLY structural knob you author
# --- materialized by the CLI from `cards` (do not hand-edit) ---
report_cards: [docs/report_cards/v2ecoli-vecoli-comparison/basal_4x4/index.html]
behavior_tests:
- name: statistical-vs-vecoli
  classification: primary
  measure: {kind: report_card_axis, card: docs/report_cards/v2ecoli-vecoli-comparison/basal_4x4, group: statistical}
pipeline_gate: {prerequisites: [], enables: []}
# --- hand-authored narrative (preserved across materialize) ---
claim: ...
question: ...
```

`name` = the **store key** (output dir, verdict dir, card dir). `condition` =
the **biological vEcoli condition** passed to the ensemble runner. These differ
for disambiguated studies (`basal_4x4` runs the `basal` condition with 4 seeds).
This is the existing `store_key`/`sim_condition` split.

## Cards → tests (materialization)

You author only `comparison.cards`. When the CLI runs a study it **materializes**
`report_cards` + `behavior_tests` into that same `study.yaml` — one
`report_card_axis` behavior_test per *graded* card (`standard`/`statistical`;
`config`/`parca` render but don't gate) — pointing at
`docs/report_cards/v2ecoli-vecoli-comparison/<name>`, group `<card>`. Narrative
fields already in the file (`claim`, `question`, `title`, `bibliography`, …) are
preserved. Materialization is idempotent.

## CLI (replaces the manifest framework)

```
v2e-compare study <name|path> [--ray] [--out DIR] [--render-only]
v2e-compare run <investigation|path> [--ray] [--out DIR] [--render-only]
```
- `study`: load one `study.yaml` (by name under the investigation's `studies/`,
  or by path) → run both engines for `seeds×generations` of its `condition`
  (v2ecoli matched-initial-state on `v2_cache`; genuine vEcoli via
  `vivarium-process` on `ve_cache`) → assemble its cards → write the
  per-condition `report_card_verdict.json` → materialize the study's
  `report_cards`/`behavior_tests`.
- `run`: load an investigation → do the above for each listed study.
- serial+local default; `--ray`/`V2E_MODE=ray` fans seeds/studies out;
  `--render-only` reuses existing stores.

## Components

- **`scripts/_compare/study_spec.py`** (new) — `load_investigation(ref)` and
  `load_study(ref)`; resolve a name or path to the YAML; merge the
  investigation's `comparison` context (caches, repos, default cards, fork env)
  into each study spec; default `cards` from the investigation when the study
  omits them. Returns plain dataclasses/dicts: `{name, condition, seeds, gens,
  cards, v2_cache, ve_cache, fork, invest_name}`.
- **`scripts/_compare/materialize.py`** (new) — `materialize_study(study_path,
  spec)`: rewrite `report_cards` + `behavior_tests` from the spec's cards
  (graded subset), preserving all other keys. (Absorbs the useful part of the
  deleted scaffold's `build_study`.)
- **`scripts/_compare/runner.py`** (new) — `run_study(spec, out, mode,
  render_only)` and `run_investigation(inv_ref, out, mode, render_only)`: the
  per-study engine subprocess wiring (moved out of the deleted
  `run_comparison.py`), then render + verdict + materialize. Reuses
  `run_comparison_ensemble.py` unchanged.
- **`scripts/comparison_report_card.py`** — `assemble_from_manifest` →
  `assemble_from_studies(specs, cond_data, conds)`: render overview + per-study
  cards from the study specs (not a manifest). Verdict emission
  (`write_condition_verdict`) unchanged. The `config` card renders the study's
  `comparison` block (no config-file dependency).
- **`scripts/compare_cli.py`** — `run`/`study` read YAML via `study_spec` and
  call `runner`. Drops the scaffold + validate steps.

## Removed

- `comparison*.json` manifests; `configs/cond_*_1x4.json` / `cond_*_4x4.json`.
- `scripts/run_comparison.py` (manifest engine), `scripts/scaffold_comparison_studies.py`,
  `scripts/validate_comparison_studies.py`, and the manifest helpers in
  `scripts/_read_spec.py` / `config_run_shape` / manifest-shaped `store_key`.
- Their tests (`test_scaffold_comparison_studies.py`, `test_validate_comparison_studies.py`),
  and the manifest-mode branch of `comparison_report_card.py`.

## Unchanged

- `scripts/_compare/verdict.py` (verdict builder/writer) and the
  `report_card_axis` evaluator — the gating chain is identical.
- `run_comparison_ensemble.py` (per-engine worker) and the card registry.
- The dashboard reads `study.yaml`'s `report_cards`/`behavior_tests` exactly as
  before — those fields are now CLI-materialized rather than hand-scaffolded.

## Error handling

- Unknown study/investigation name or path → clear `sys.exit`.
- A study with no `condition` or non-positive `seeds`/`generations` → fail loud.
- `--render-only` with no stores under `out/<name>` → the assembler skips that
  study (as today) and logs it.
- All YAML/JSON reads+writes `encoding="utf-8"`.

## Testing

- `study_spec`: name/path resolution; investigation-context merge; card default;
  store-key vs condition separation (basal_4x4 → name basal_4x4, condition basal).
- `materialize`: cards → one graded behavior_test each; config/parca ungated;
  narrative fields preserved; idempotent.
- `runner`: study runs both engines with the right `--condition`/`--n-seeds`/
  `--max-generations` and stores under `out/<name>` (subprocess mocked);
  `--render-only` skips sims; `--ray` selects ray mode; investigation loops studies.
- `compare_cli`: `study`/`run` resolve YAML and invoke the runner; non-zero on
  load error.
- Integration: regenerate the 6 studies in the new schema; `v2e-compare run
  --render-only` against existing stores writes a verdict per study and
  materializes tests matching each study's cards.
