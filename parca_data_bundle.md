# ParCa Data Bundle

This document describes where v2ecoli's ParCa flat-file data now comes from and
how overrides are layered on top of the upstream package.

## Where the data lives

ParCa flat-file data comes from the
[`ecoli-sources`](https://github.com/vivarium-collective/ecoli-sources) Python
package, which is declared as a git-pinned dependency in `pyproject.toml`.
Installing the project installs `ecoli-sources` and places it in the active
virtual environment.

At load time the resolver reads the package's canonical manifest:

```
ecoli_sources.BUNDLE_PATH
# → <venv>/site-packages/ecoli_sources/data/reference_bundle.tsv
```

The manifest has 135 rows mapping `canonical_key → source_path`.  All 135
source files ship inside the `ecoli-sources` package under its `data/`
directory.

The resolver is implemented in:

```
v2ecoli/processes/parca/reconstruction/ecoli/sources.py  # SourceBundle class
```

## Override mechanism

v2ecoli keeps three files that deliberately differ from the `ecoli-sources`
defaults:

| File | Reason |
|---|---|
| `equilibrium_reactions.tsv` | PR #123 — DnaA-ATP hydrolysis as kinetic equilibrium |
| `equilibrium_reaction_rates.tsv` | PR #123 — DnaA-ATP hydrolysis rates |
| `metabolic_reactions_added.tsv` | v2parca merge #16 — additional metabolic reactions |

These live under `flat_overrides/` (sibling of `sources.py`).  The override
spec at:

```
v2ecoli/processes/parca/reconstruction/ecoli/parca_overrides.tsv
```

lists the three keys with `source_path` pointing at `flat_overrides/<file>`.
When `SourceBundle` is constructed it reads the base manifest first, then layers
the override manifest on top: any key present in both is won by the override.
All other 132 keys resolve to the `ecoli-sources` package files.

The 130 files that were previously in
`v2ecoli/processes/parca/reconstruction/ecoli/flat/` and are byte-identical to
`ecoli-sources` were deleted (PR #1 `ecoli-sources-bundle`).

## How `KnowledgeBaseEcoli` consumes the bundle

`KnowledgeBaseEcoli` (`knowledge_base_raw.py`) accepts an optional `bundle=`
parameter:

```python
KnowledgeBaseEcoli(operons_on=True, ..., bundle=SourceBundle())
```

When `bundle` is supplied, every flat-file load goes through
`bundle.resolve_relpath(rel_path)` which converts a legacy relative path to the
resolved absolute path for that key.  If `bundle` is `None` the old
`FLAT_DIR`-based fallback is used (kept for offline development / debugging).

The default is `bundle=None`; the v2ecoli `build_parca_composite` entry-point
constructs a `SourceBundle()` and passes it in automatically, so normal ParCa
runs always use the bundle.

## CLI usage (`v2ecoli-parca`)

```
v2ecoli-parca [--bundle-manifest-path /path/to/custom_bundle.tsv] ...
```

`--bundle-manifest-path` replaces the *base* manifest (default:
`ecoli_sources.BUNDLE_PATH`).  The v2ecoli overrides in `parca_overrides.tsv`
are still layered on top regardless of which base manifest is used.  Passing a
custom manifest is useful for testing alternative data sets without modifying the
installed `ecoli-sources` package.

## Canonical-key convention

Bundle keys are derived from the flat-relative file path by:

1. Replacing path separators with `__`
2. Stripping the trailing file extension from the basename only

Examples:

| Flat relative path | Canonical key |
|---|---|
| `genes.tsv` | `genes` |
| `adjustments/amino_acid_pathways.tsv` | `adjustments__amino_acid_pathways` |
| `condition/media/MIX0-55.tsv` | `condition__media__MIX0-55` |

The conversion is implemented in `sources.relpath_to_key()`.

## Spec and plan

- Design spec: `docs/superpowers/specs/2026-06-06-ecoli-sources-bundle-integration-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-06-ecoli-sources-bundle-integration-pr1.md`
