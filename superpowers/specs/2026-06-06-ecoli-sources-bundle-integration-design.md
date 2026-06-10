# Design: ecoli-sources bundle integration + multi-ParCa (v2ecoli)

**Date:** 2026-06-06
**Author:** Eran (with Claude)
**Status:** Draft — awaiting review

## Goal

Replace most of v2ecoli's in-repo ParCa source data with the shared
[`ecoli-sources`](https://github.com/vivarium-collective/ecoli-sources) data
package, mirroring CovertLab/vEcoli's
[#426](https://github.com/CovertLab/vEcoli/pull/426)
(`data-bundle-migration`), and then add the **multi-ParCa** capability from
CovertLab's `multi-parca-workflow` branch — re-expressed on v2ecoli's own
process-bigraph workflow (v2ecoli has no Nextflow).

Delivered as **two stacked PRs**:

- **PR 1 — bundle integration** (this spec's primary focus).
- **PR 2 — multi-ParCa runner** (sketched here; its own spec later).

## Background (established during brainstorming)

- `ecoli-sources` is a data package: `reference_bundle.tsv` (135
  `canonical_key → source_path` rows) + the flat data + Pandera schemas,
  exposed via `ecoli_sources.BUNDLE_PATH` / `ecoli_sources.DATA_DIR`.
  Installing it also installs the top-level `schemas` package
  (`ReferenceBundleSchema`); transitive deps `pandas>=2.0`, `pandera>=0.19`.
- vEcoli #426 adds a `SourceBundle` resolver (`wholecell/io/sources.py`),
  rewires `KnowledgeBaseEcoli` to resolve flat files via canonical keys, adds
  `--bundle-manifest-path`, and deletes local `reconstruction/ecoli/flat/`.
- v2ecoli today: 133 flat files at
  `v2ecoli/processes/parca/reconstruction/ecoli/flat/` (~13 MB, shipped as
  package-data); `KnowledgeBaseEcoli`
  (`.../reconstruction/ecoli/knowledge_base_raw.py`) loads them via a hardcoded
  `LIST_OF_DICT_FILENAMES` + `REMOVED_DATA`/`MODIFIED_DATA`/`ADDED_DATA` maps +
  variant flags (`operons_on`, `remove_rrna_operons`, `stable_rrna`,
  `new_genes_option`). ParCa entry point: `v2ecoli-parca` CLI →
  `build_parca_composite` (9-step composite). No multi-ParCa.

### Content audit (decisive)

Comparing git blob SHAs of the 133 flat files against ecoli-sources:

- **130 of 133 are byte-identical.**
- **3 differ — and they are v2ecoli's own intentional biology:**
  - `equilibrium_reactions.tsv`, `equilibrium_reaction_rates.tsv` — from
    **PR #123** (DnaA-ATP hydrolysis as kinetic equilibrium, commit `c44e013`).
  - `metabolic_reactions_added.tsv` — from the v2parca merge (#16).

Adopting the stock default bundle as-is would silently revert this work. The
design must preserve v2ecoli's divergent files.

### Variant-key coverage (verified)

The ecoli-sources bundle already contains every variant-relevant canonical key
v2ecoli's KB uses: `transcription_units`, `transcription_units_{added,removed,
modified}`, `rrna_options__remove_rrff__*`,
`rrna_options__remove_rrna_operons__*`, `new_gene_data__{gfp,template}__*`, and
all `*_{added,removed,modified}` keys. Key convention: path separators become
`__`; this maps cleanly to v2ecoli's dotted refs (e.g.
`rrna_options.remove_rrna_operons.transcription_units_removed`).

## Decisions (locked)

1. **Two stacked PRs.** PR 1 standalone-valuable; PR 2 stacked on it.
2. **v2ecoli override bundle.** Keep the 3 divergent files in v2ecoli; the
   resolver layers a tiny v2ecoli override onto the ecoli-sources default
   bundle so the 130 identical keys come from ecoli-sources and the 3 diverged
   keys come from local copies. Preserves byte-identity; no upstream blocker.

## PR 1 — bundle integration

### Acceptance gate

**ParCa output is byte-identical before/after the migration** (default bundle +
v2ecoli overrides). Concretely: `raw_data` field-for-field equal, and the
produced `sim_data` / cache bundle equal to the current frozen cache. This
fits v2ecoli's existing parity-harness culture and the gzipped CI cache at
`tests/fixtures/cache/`.

### Components

#### 1. Dependency

Add `ecoli-sources` git-pinned to a **specific commit** (not a branch) in
`pyproject.toml`:

```toml
[project]
dependencies = [ ..., "ecoli-sources" ]

[tool.uv.sources]
ecoli-sources = { git = "https://github.com/vivarium-collective/ecoli-sources.git", rev = "<pinned-sha>" }
```

Rationale: ParCa cache parity requires the input data to be frozen — a moving
`branch=main` would silently change ParCa output and break the parity gate.
`pandas`/`pandera` arrive transitively.

#### 2. `SourceBundle` resolver

New module `v2ecoli/processes/parca/reconstruction/ecoli/sources.py` (next to
`knowledge_base_raw.py`). Port of vEcoli's `wholecell/io/sources.py`, extended
for the override layering:

```python
class SourceBundle:
    def __init__(self, base_manifest=None, overrides=None):
        # base_manifest defaults to ecoli_sources.BUNDLE_PATH
        # overrides defaults to the checked-in v2ecoli override spec
        ...
    def path(self, canonical_key) -> Path: ...
    def get(self, canonical_key) -> Path:  # alias; raises KeyError naming the key
```

Resolution model (**load-time merge, one effective manifest in memory**):

- Read the base manifest (ecoli-sources). Its `source_path`s resolve against
  `ecoli_sources.DATA_DIR`.
- Read the v2ecoli override spec (small TSV, same columns). Its `source_path`s
  resolve against the v2ecoli override root.
- For each `canonical_key`, the override row replaces the base row. The result
  is a single `{canonical_key: absolute_path}` index.

This deviates from ecoli-sources' "no runtime merge" guidance, deliberately:
v2ecoli keeps a **3-row** override file instead of a drifting 135-row copy.
Validation (below) runs on the *effective merged* manifest, so the canonical-key
contract is still enforced.

Validation: run `schemas.ReferenceBundleSchema.validate(...)` (from the
installed ecoli-sources `schemas` package) on the effective manifest at
construction; fail loudly naming any missing/extra key.

#### 3. v2ecoli override spec + override data

- `v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv` — 3 rows
  (`equilibrium_reactions`, `equilibrium_reaction_rates`,
  `metabolic_reactions_added`) pointing at local files.
- `v2ecoli/processes/parca/reconstruction/ecoli/flat_overrides/` — the 3
  divergent files, moved out of the deleted `flat/` tree. Shipped as
  package-data.

#### 4. Rewire `KnowledgeBaseEcoli`

- Replace `FLAT_DIR` path joins / `_load_tsv(FLAT_DIR, ...)` with
  `SourceBundle` lookups keyed by canonical key.
- Translate `LIST_OF_DICT_FILENAMES` filenames → canonical keys
  (filename `a/b/c.tsv` → key `a__b__c`); keep the list as the *set of keys to
  load* so behaviour is unchanged.
- `REMOVED_DATA`/`MODIFIED_DATA`/`ADDED_DATA` and the variant flags
  (`operons_on`, `remove_rrna_operons`, `new_genes_option`, `stable_rrna`) now
  select **canonical keys** (the dotted refs already match `__` keys 1:1).
- `_load_parameters` and `_load_sequence` (FASTA) resolve their inputs through
  the bundle too (the bundle has `sequence`/parameter keys).

This is the **highest-risk** part: the variant/REMOVED/MODIFIED/ADDED mapping
must reproduce the exact files the old code loaded for every flag combination
exercised by the test matrix.

#### 5. CLI / config

- `v2ecoli-parca --bundle-manifest-path <path>` (default: ecoli-sources
  default + v2ecoli overrides).
- Config key `parca_options.bundle_manifest_path` (null = default).
- Thread through to `KnowledgeBaseEcoli` construction.
- **Interaction with overrides:** `--bundle-manifest-path` replaces the *base*
  manifest only; the v2ecoli 3-file overrides still layer on top (they are
  v2ecoli biology that must persist across variant runs). A variant manifest
  that itself defines one of those 3 keys wins for that key (override of the
  override), so a variant can still deliberately change equilibrium/metabolism
  data when intended.

#### 6. Delete local flat data

- Remove the 130 identical files from
  `v2ecoli/processes/parca/reconstruction/ecoli/flat/`.
- Update `[tool.setuptools.package-data]` (drop `flat/**/*`, add
  `flat_overrides/*` and `parca_overrides.tsv`).
- Keep only the 3 override files (relocated).

### Testing (PR 1)

1. **Bundle parity test:** assert the 130 inherited keys resolve to files whose
   content SHA equals the ecoli-sources blob, and the 3 override keys resolve to
   the v2ecoli local files. (Drift guard against ecoli-sources updates.)
2. **Resolver unit tests:** missing-key error names the key; override replaces
   base; validation rejects an incomplete manifest.
3. **ParCa byte-identity:** run ParCa (fast mode at minimum; full in CI cache
   build) and assert `raw_data`/`sim_data` equal the pre-migration baseline.
   Reuse the existing cache-fingerprint / parity harness.
4. **Variant-flag coverage:** for each KB variant flag combination in current
   use, assert the resolved key set equals the pre-migration filename set.

### Risks / mitigations

- **Variant-key mismatch** (highest): a flag combo resolves a different file
  set than before → caught by test 4 and the byte-identity gate.
- **ecoli-sources drift** under the pin: mitigated by commit-pin + test 1.
- **`schemas` package name collision** (generic top-level name from
  ecoli-sources): import as needed; if it bites, vendor a minimal validator.
- **pandera as a runtime dep** in the ParCa path: acceptable (transitive);
  validation only at bundle load.

## PR 2 — multi-ParCa runner (sketch)

Re-express CovertLab `multi-parca-workflow` on v2ecoli's stack (no Nextflow):

- Config: top-level `parca_variants: []` — each entry a dict of
  `parca_options` overrides merged on the baseline (empty → single run,
  backward-compatible). The natural v2ecoli override is `bundle_manifest_path`
  (each variant = a different ecoli-sources bundle).
- Runner: run ParCa once per variant → N cache bundles
  (`models/parca/parca_<idx>/...`).
- Two-level indexing `global = parca_idx * pickles_per_parca + variant_idx`
  so caches/outputs never collide; wire into the existing
  `v2ecoli/workflow/meta_composite.py` branches (which already do
  variant×seed).
- Merge per-ParCa metadata → single metadata for the existing multi-variant
  comparison/analysis layer.
- A failed ParCa variant is isolated; the rest proceed (matches upstream
  behaviour).

Detailed design deferred to PR 2's own spec after PR 1 lands.

## Out of scope

- vEcoli #426's sim_data field additions (`cistron_data.common_name`,
  `antibiotics` namespace, `strand_term_p`) — vEcoli-specific, not part of this
  migration.
- Migrating data-prep scripts into ecoli-sources (upstream's ecoli-sources#2).
- Any change to the 3 divergent biology files (preserved as-is via overrides).
