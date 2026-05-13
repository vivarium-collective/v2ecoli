# Standardize v2ecoli composites on the pbg-superpowers framework

**Status:** Design approved, awaiting implementation plan.
**Date:** 2026-05-12.
**Branch:** `standardize-composites` (cut from `main`).
**Related:** This is sub-project A of a larger effort. Sub-projects B (whole-duration sim runs), C (workflow framework), D (daughter sim modes) are deferred to future specs.

## Goal

Replace v2ecoli's three hand-rolled `make_*_composite()` factory functions and three matched `generate*.py` document builders with a uniform, registered `@composite_generator` framework borrowed from `pbg-superpowers`. Rename the default architecture from "partitioned" to "baseline" so every architecture has a consistent first-class identity. Consolidate shared infrastructure into a single `v2ecoli/core.py` module and expose one top-level entry point — `v2ecoli.build_composite(name, **kwargs) -> Composite` — for callers that just want a composite by architecture name.

## Non-goals

- Workflow orchestration (experiments × variants × seeds × generations). Sub-project C.
- Whole-duration simulation runs (replacing the controlled-interval stepping pattern). Sub-project B.
- Daughter simulation modes (single-daughter vs multi-daughter). Sub-project D.
- Promoting the colony composite to a registered architecture.
- Adding new architectures (only the three that exist today get standardized).
- Behavior changes. The migration is supposed to be document-equivalent — same processes, same wiring, same outputs.
- Expanding the parameter surface of each generator beyond `seed` and `cache_dir`. The current code reads further configuration from the cache bundle and internal defaults; promoting more parameters to the typed surface is deferred.

## Architecture

### Final file layout

```
v2ecoli/
├── __init__.py              # exposes build_composite; imports composites/ so decorators fire
├── core.py                  # build_core, load_cache_bundle[_cached], save_cache
├── composites/
│   ├── __init__.py          # forces import of all three generators so @composite_generator fires
│   ├── baseline.py          # @composite_generator(name="baseline")
│   ├── departitioned.py     # @composite_generator(name="departitioned")
│   └── reconciled.py        # @composite_generator(name="reconciled")
├── bridge.py                # updated imports
├── steps/division.py        # updated imports
├── library/cache_version.py # updated to fingerprint the new files
└── ... (everything else unchanged)
```

### Deleted

Six legacy files, all logic migrated into the new layout:

- `v2ecoli/composite.py`
- `v2ecoli/composite_departitioned.py`
- `v2ecoli/composite_reconciled.py`
- `v2ecoli/generate.py`
- `v2ecoli/generate_departitioned.py`
- `v2ecoli/generate_reconciled.py`

### Dependency added

`pbg-superpowers` becomes a runtime dependency of `v2ecoli`. It hosts the `@composite_generator` decorator, `GeneratorEntry` dataclass, `_REGISTRY`, `build_generator`, and `discover_generators`. Source repo is `vivarium-collective/pbg-superpowers`; consumed via `uv` in `[tool.uv.sources]` the same way `process-bigraph` and `bigraph-schema` are consumed today. Version constraint: `pbg-superpowers` pinned to whatever version is currently published (the implementation plan resolves the exact pin from `uv` at install time).

## API contract

### Composite generator (one per architecture)

```python
# v2ecoli/composites/baseline.py
from pbg_superpowers.composite_generator import composite_generator
from v2ecoli.core import load_cache_bundle


@composite_generator(
    name="baseline",
    description="55-process partitioned whole-cell E. coli model — upstream-parity architecture",
    parameters={
        "seed":      {"type": "integer", "default": 0,
                      "description": "RNG seed for stochastic initialization"},
        "cache_dir": {"type": "string",  "default": "out/cache",
                      "description": "Path to ParCa cache directory"},
    },
)
def baseline(core=None, *, seed=0, cache_dir="out/cache") -> dict:
    """Build the process-bigraph state document for the baseline architecture.

    Returns a state document (a plain dict) suitable for passing to
    ``Composite(doc, core=core)``. Use ``v2ecoli.build_composite("baseline", ...)``
    for the one-line "give me a Composite" path; this function is the lower-level
    entry point for callers that want to inspect or mutate the document before
    Composite construction.
    """
    bundle = load_cache_bundle(cache_dir)
    # ... migrated body from current v2ecoli/generate.py:build_document ...
    return state  # plain dict; {state, schema} envelope used only if upstream needs it
```

Same shape for `departitioned` and `reconciled`. The function name matches the architecture name (and matches the decorator's `name=` argument). Each generator returns a state document. Cache loading happens inside the generator from `cache_dir`; the legacy `(state, configs, unique_names, dry_mass_inc_dict)` argument tuple is no longer surfaced.

### Top-level helper

```python
# v2ecoli/__init__.py
from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator

from v2ecoli.core import build_core
from v2ecoli import composites  # noqa: F401  — forces decorator side-effects


def build_composite(name: str, *, core=None, **kwargs) -> Composite:
    """Build a Composite by architecture name.

    ``name`` is one of ``"baseline"``, ``"departitioned"``, ``"reconciled"``.
    ``**kwargs`` are passed through to the generator's declared parameters
    (currently ``seed`` and ``cache_dir`` for all three architectures).

    Unknown parameter names raise ``ValueError`` via ``build_generator``;
    unknown architecture names raise ``ValueError`` with the list of available
    architectures.
    """
    if core is None:
        core = build_core()
    matches = [e for e in _REGISTRY.values() if e.name == name]
    if not matches:
        available = sorted({e.name for e in _REGISTRY.values()})
        raise ValueError(
            f"unknown composite architecture {name!r}; available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous architecture name {name!r}; multiple generators registered: "
            f"{[e.id for e in matches]}"
        )
    doc = build_generator(matches[0], overrides=kwargs, core=core)
    return Composite(doc, core=core)


__all__ = ["build_composite", "build_core"]
```

### Public surface after the migration

| Symbol | Module | Purpose |
|---|---|---|
| `build_composite(name, *, core=None, **kwargs)` | `v2ecoli` | One-line "give me a Composite by architecture name" |
| `build_core()` | `v2ecoli.core` | Construct the v2ecoli-specific bigraph-schema core |
| `load_cache_bundle(cache_dir)` | `v2ecoli.core` | Load `(state, configs, unique_names, dry_mass_inc_dict)` from the ParCa cache |
| `save_cache(sim_data_path, cache_dir, seed)` | `v2ecoli.core` | Write the ParCa cache |
| `baseline(core, *, seed, cache_dir)` | `v2ecoli.composites.baseline` | Build the baseline document |
| `departitioned(core, *, seed, cache_dir)` | `v2ecoli.composites.departitioned` | Build the departitioned document |
| `reconciled(core, *, seed, cache_dir)` | `v2ecoli.composites.reconciled` | Build the reconciled document |

### Retired (no longer importable)

- `v2ecoli.composite.make_composite`
- `v2ecoli.composite._build_core`
- `v2ecoli.composite._load_cache_bundle`
- `v2ecoli.composite._load_cache_bundle_cached`
- `v2ecoli.composite._build_from_cache`
- `v2ecoli.composite.save_cache`
- `v2ecoli.composite_departitioned.make_departitioned_composite`
- `v2ecoli.composite_reconciled.make_reconciled_composite`
- `v2ecoli.generate.build_document`
- `v2ecoli.generate_departitioned.build_document_departitioned`
- `v2ecoli.generate_reconciled.build_document_reconciled`

## Migration mechanics

### Per-symbol mapping

| Legacy symbol | New home |
|---|---|
| `v2ecoli.composite._build_core` | `v2ecoli.core.build_core` (renamed; public) |
| `v2ecoli.composite._load_cache_bundle` | `v2ecoli.core.load_cache_bundle` (public) |
| `v2ecoli.composite._load_cache_bundle_cached` | `v2ecoli.core._load_cache_bundle_cached` (kept private) |
| `v2ecoli.composite.save_cache` | `v2ecoli.core.save_cache` |
| `v2ecoli.composite.make_composite` | retired → `v2ecoli.build_composite("baseline", ...)` |
| `v2ecoli.composite._build_from_cache` | absorbed into `v2ecoli/composites/baseline.py` (private) |
| `v2ecoli.composite_departitioned.make_departitioned_composite` | retired → `v2ecoli.build_composite("departitioned", ...)` |
| `v2ecoli.composite_reconciled.make_reconciled_composite` | retired → `v2ecoli.build_composite("reconciled", ...)` |
| `v2ecoli.generate.build_document` | body → `v2ecoli.composites.baseline.baseline` |
| `v2ecoli.generate_departitioned.build_document_departitioned` | body → `v2ecoli.composites.departitioned.departitioned` |
| `v2ecoli.generate_reconciled.build_document_reconciled` | body → `v2ecoli.composites.reconciled.reconciled` |

### Call-site update patterns

Every caller falls into one of three patterns. The migration is mechanical and uniform.

**Pattern 1 — "I want a Composite":**
```python
# Before
from v2ecoli.composite import make_composite
comp = make_composite(cache_dir="out/cache", seed=0)
# or
from v2ecoli.composite_departitioned import make_departitioned_composite
comp = make_departitioned_composite(cache_dir="out/cache", seed=0)

# After
from v2ecoli import build_composite
comp = build_composite("baseline", cache_dir="out/cache", seed=0)
# or
comp = build_composite("departitioned", cache_dir="out/cache", seed=0)
```

**Pattern 2 — "I want the document so I can inspect/modify before Composite construction":**
```python
# Before
from v2ecoli.composite import _build_core
from v2ecoli.generate import build_document
core = _build_core()
doc = build_document(state, configs, unique_names, dry_mass_inc_dict=dry_mass_inc, seed=0)
comp = Composite(doc, core=core)

# After
from v2ecoli.core import build_core
from v2ecoli.composites.baseline import baseline
core = build_core()
doc = baseline(core=core, seed=0)   # cache loading is now inside baseline()
comp = Composite(doc, core=core)
```
Callers no longer pass `state, configs, unique_names, dry_mass_inc_dict` — those are loaded internally from `cache_dir`. Tests that hand-mutate any of these mid-build (none currently identified in the codebase) drop one level lower to `v2ecoli.core.load_cache_bundle(cache_dir)` and assemble the document by reading the body of the relevant generator function.

**Pattern 3 — "I need the core only":**
```python
# Before
from v2ecoli.composite import _build_core
core = _build_core()

# After
from v2ecoli.core import build_core
core = build_core()
```

### Internal-module updates

- `v2ecoli/bridge.py` — colony bridge per-tick composite construction. Imports updated to `v2ecoli.core.build_core` and `v2ecoli.build_composite`; the per-tick `build_composite("baseline", core=core, ...)` is functionally identical to the old `make_composite(core=core, ...)`. The bridge already caches a single core; that pattern is preserved.
- `v2ecoli/steps/division.py` — daughter-cell rebuild logic. Same migration pattern.
- `v2ecoli/library/cache_version.py` — the cache fingerprint mechanism reads specific source files by path. Updated to fingerprint `v2ecoli/composites/*.py` and `v2ecoli/core.py` instead of the deleted `composite*.py` / `generate*.py`.

### Test file updates

The 11 `tests/test_*.py` files that import the legacy API are migrated in lockstep:

`test_cell_cycle_regressions.py`, `test_composite_run_no_refire.py`, `test_model_behavior.py`, `test_seed_determinism.py`, `test_sim_speed.py`, `test_sustained_growth.py`, `test_initialize.py`, `test_cache_version.py`, `test_growth_parity.py`, `test_architectures_grow.py`, plus the helper `tests/_state_equal.py`.

Each is a Pattern 1 or Pattern 2 migration. Behavior assertions (mass thresholds, division timings, parity comparisons against vEcoli) **don't change** — same numbers, same tolerances.

### Scripts, reports, docs

- `scripts/build_cache.py` — `save_cache` import updated to `v2ecoli.core.save_cache`.
- `scripts/run_v2.py`, `scripts/viz_baseline.py`, `scripts/viz_network.py` — Pattern 1.
- `reports/benchmark_report.py`, `reports/multigeneration_report.py`, `reports/workflow_report.py`, `reports/network_report.py`, and other affected report scripts — Pattern 1.
- `README.md`, `AGENTS.md`, `docs/generate_full_parca.md` — prose + import-example updates. `AGENTS.md` gets a short paragraph on the `composites/` package and how to add a new architecture (decorate a function with `@composite_generator`, drop it in `v2ecoli/composites/`, add the import to `composites/__init__.py`).

### Discovery vs eager registration

The `pbg-superpowers` contract is "register on import." `v2ecoli/__init__.py` adds `from v2ecoli import composites  # noqa: F401` to force the three modules under `composites/` to import (which fires the three `@composite_generator` decorators). `v2ecoli/composites/__init__.py` does the same for its own submodules: `from v2ecoli.composites import baseline, departitioned, reconciled  # noqa: F401`. After `import v2ecoli`, all three are in `_REGISTRY` and `build_composite("baseline" / "departitioned" / "reconciled")` resolves locally without invoking `discover_generators()` (which is expensive — it imports every installed `bigraph-schema`-dependent distribution).

Downstream consumers that need to see *all* `@composite_generator`-decorated functions across multiple packages call `discover_generators()` themselves; v2ecoli does not.

## Testing

### 1. Existing behavior tests are the primary gate

All 11 test files that exercise the legacy API are migrated to the new API in this PR. Behavior thresholds and assertions are unchanged. The gate is:

- `pytest -m "not sim" tests/` — fast suite, currently 50 tests on `main`, ~55 after the migration adds smoke tests.
- `pytest -m sim tests/test_model_behavior.py` — 7 definitive behavior tests (mass, growth rate, division timing). This is the primary equivalence check: if the refactor accidentally changed the produced document, these mass/timing assertions catch it.
- `tests/test_architectures_grow.py` ensures all three generators produce growing cells; covers all three architectures.
- `tests/test_growth_parity.py` ensures continued vEcoli parity at 60s.
- `tests/test_cache_version.py` needs updating to point at the new fingerprint paths (`v2ecoli/composites/*.py` and `v2ecoli/core.py`).

### 2. New smoke tests — `tests/test_build_composite.py`

Five `@pytest.mark.fast` tests covering the new API surface:

```python
def test_build_composite_baseline_returns_composite():
    from v2ecoli import build_composite
    from process_bigraph.composite import Composite
    comp = build_composite("baseline", seed=0)
    assert isinstance(comp, Composite)

def test_build_composite_each_architecture():
    from v2ecoli import build_composite
    for name in ("baseline", "departitioned", "reconciled"):
        comp = build_composite(name, seed=0)
        assert comp is not None

def test_build_composite_unknown_name_raises():
    from v2ecoli import build_composite
    with pytest.raises(ValueError, match="unknown composite architecture"):
        build_composite("nonexistent", seed=0)

def test_build_composite_accepts_core_override():
    from v2ecoli import build_composite
    from v2ecoli.core import build_core
    core = build_core()
    comp = build_composite("baseline", seed=0, core=core)
    assert comp.core is core

def test_generators_registered_under_short_names():
    from pbg_superpowers.composite_generator import _REGISTRY
    import v2ecoli  # noqa: F401
    names = {e.name for e in _REGISTRY.values() if e.module.startswith("v2ecoli.")}
    assert {"baseline", "departitioned", "reconciled"} <= names
```

No ParCa cache rebuild, no sim runs — pure import-and-construct.

### 3. Document equivalence is enforced through behavior tests, not a separate gate

A structural diff against a golden document from `main` was considered and rejected: the behavior tests (mass trajectories at specific times) span the document's content tightly enough that any meaningful divergence is caught, and a structural hash would be fragile to dict ordering and default rendering. The behavior tests are the equivalence proof.

### 4. Lower-level helper coverage

`v2ecoli.core.build_core`, `v2ecoli.core.load_cache_bundle`, and the three per-architecture functions don't get dedicated unit tests. They're either trivial passthroughs of the old code, or covered transitively by `test_build_composite` and `test_model_behavior`.

### 5. Fixture refresh

`tests/fixtures/cache/` was fingerprinted against the old source paths. After `cache_version.py` is updated to point at the new paths, the committed cache is regenerated as a one-time fixture refresh. Per AGENTS.md, fixture changes are deliberate and called out in the PR description.

## Risks

1. **Behavior drift from refactor.** Migration is document-equivalent in intent; behavior tests are the only enforcement. If `test_model_behavior.py` fails after migration, the answer is to find where the document drifted — not to weaken the threshold.
2. **Cache fingerprint paths.** `cache_version.py` must be updated AND the committed cache fixture regenerated. If only one is done, every CI run fails with `StaleCacheError` or misses real cache invalidations.
3. **Reports regenerate.** Per AGENTS.md, PRs that touch composite wiring rerun the affected reports and attach the output HTML. This PR touches the public composite-construction API, so the affected reports (`workflow_report`, `compare_report`, `network_report`, `v1_v2_report`) get rerun and the diff attached.
4. **`pbg-superpowers` runtime dep.** Stable enough today (the `@composite_generator` decorator surface is small and frozen by upstream tests in pbg-superpowers itself), but the package is still in active development. We pin a lower-bound rather than a tight pin, and call out the addition in the PR description's "Dependency changes" section.
5. **Discovery ordering.** `v2ecoli/__init__.py` imports `v2ecoli.composites` to fire the three decorator side-effects. Standard Python import semantics make this robust (any `import v2ecoli.*` runs `v2ecoli/__init__.py` first), but the `test_generators_registered_under_short_names` smoke test guards against regressions.
6. **`bridge.py` per-tick overhead.** The colony bridge rebuilds the inner composite each tick; `build_composite` does a registry lookup on each call. The registry is a dict (O(1)), so the overhead is negligible. Flagged for awareness; if profiling later shows it matters, the bridge caches the resolved `GeneratorEntry` in its instance.

## Open items (deferred to follow-up)

- **Richer parameter surface on each generator.** Other knobs (feature flags, time-step overrides, per-process configuration) stay where they live today. Revisited when sub-project C lands.
- **`pbg_superpowers.composite_discovery.discover_composites`** (YAML/JSON spec discovery). v2ecoli has no `*.composite.yaml` files today; out of scope until a static-spec consumer needs them.
- **`v2ecoli/colony.py` promotion to registered architecture.** The colony composite wraps a baseline composite + pymunk physics — different shape from the three single-cell architectures. Whether to register it as `@composite_generator(name="colony")` later is a follow-up decision.

## Sequencing

**Single PR.** Atomic refactor per the chosen approach. Estimated size:

| Bucket | Count |
|---|---|
| New files | 5 (`v2ecoli/composites/{__init__,baseline,departitioned,reconciled}.py`, `v2ecoli/core.py`) |
| Deleted files | 6 (legacy `composite*.py` + `generate*.py`) |
| Modified — internal v2ecoli | ~4 (`v2ecoli/__init__.py`, `bridge.py`, `steps/division.py`, `library/cache_version.py`) |
| Modified — tests | ~12 |
| Modified — scripts | ~4 |
| Modified — reports | ~5 |
| Modified — docs | 3 (`README.md`, `AGENTS.md`, `docs/generate_full_parca.md`) |
| New tests | 1 (`tests/test_build_composite.py`, 5 smoke tests) |
| Fixture refresh | 1 (`tests/fixtures/cache/`, called out in PR) |
| `pyproject.toml` + `uv.lock` | 2 (add `pbg-superpowers` dep) |

Roughly 35–40 file changes. Big PR, but each change is small and mechanical. The implementation plan will TDD-shape into tasks: scaffolding (`core.py`, `composites/`), per-architecture migrations, call-site updates in waves (tests → internal → scripts → reports → docs), then the deletes and the acceptance sweep.

## Acceptance criteria

- [ ] `pytest -m "not sim" tests/` passes (50 existing + 5 new = ~55 tests).
- [ ] `pytest -m sim tests/test_model_behavior.py` passes (7 behavior tests).
- [ ] `pytest -m sim tests/test_growth_parity.py` passes.
- [ ] `pytest -m sim tests/test_architectures_grow.py` passes for all three architectures.
- [ ] `rg "from v2ecoli\.composite[_a-z]*|from v2ecoli\.generate" v2ecoli/ tests/ scripts/ reports/` returns zero hits.
- [ ] `rg "make_composite|make_departitioned_composite|make_reconciled_composite|build_document" v2ecoli/ tests/ scripts/ reports/` returns zero hits.
- [ ] The six legacy files are deleted from `v2ecoli/`.
- [ ] `python -c "from v2ecoli import build_composite; print(build_composite)"` succeeds.
- [ ] `python -c "from pbg_superpowers.composite_generator import _REGISTRY; import v2ecoli; print({e.name for e in _REGISTRY.values()})"` includes `{"baseline", "departitioned", "reconciled"}`.
- [ ] `python scripts/build_cache.py` runs cleanly (cache rebuild against new source paths).
- [ ] `uv.lock` regenerated and committed with `pbg-superpowers` resolved.
- [ ] PR description has a "Dependency changes" section (per AGENTS.md).
- [ ] Affected reports rerun and the output HTML attached to the PR (per AGENTS.md).
