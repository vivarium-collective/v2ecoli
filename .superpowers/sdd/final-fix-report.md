# Final Fix Report — comparison↔investigation unify
Branch: `feat/comparison-investigation-unify`  
Date: 2026-06-27

---

## FIX 1 — Single-source `store_key`

### 1a — `scripts/_compare/config_adapter.py`
Added `import os` and `import re` at the top. Added `store_key(entry, fork_dir="") -> str` implementing the resolution order: explicit `name` → fork-resolved `condition` → filename-stem (leading `cond_` + trailing `_NxN` stripped).

### 1b — `scripts/run_comparison.py`
- Imported `store_key` from `scripts._compare.config_adapter`.
- Separated `sim_condition = condition_of(cfg, fork)` (biological condition passed as `--condition` to subprocesses) from `store = store_key(entry, fork)` (store dir, progress prints).
- Updated `--condition` filter at the top of main to use `store_key(e, fork)` instead of the old `(e.get("name") or condition_of(...))` expression.

### 1c — `scripts/comparison_report_card.py`
- In the `manifest_mode` block: imported `store_key` alongside `resolve_vecoli_config_local`.
- Replaced the `_config_names` dict comprehension from `_entry.get("name") or _cond_name(...)` to `store_key(_entry, _fork)`.
- Left `_cond_name` defined (still referenced by its own internal fallback logic, removed the dead `import re as _re` module reference by inlining a local `_re2`).
- Added one-line comment on the `cardv` last-graded-section-wins invariant in `assemble_from_manifest`.

### 1d — `scripts/scaffold_comparison_studies.py`
- Replaced `condition_name()` body to delegate to `store_key(entry, os.environ.get("V2E_VECOLI_DIR", ""))`.
- Added `import sys` + `sys.path.insert(0, str(REPO))` so the script is runnable standalone (required for the lazy `from scripts._compare.config_adapter import store_key` inside `condition_name`).
- Removed unused `import re` and `import sys` (then re-added `sys` for the path bootstrap), leaving `import os` (used by `os.path.relpath`).

---

## FIX 2 — Validator runnable standalone

### `scripts/validate_comparison_studies.py`
Added `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` before `from scripts.scaffold_comparison_studies import ...`.

---

## FIX 3 — Minors

### `scripts/validate_comparison_studies.py`
Changed `validate(...)` return annotation from `-> list` to `-> list[str]`.

### `tests/test_validate_comparison_studies.py`
`test_validate_flags_group_mismatch`: replaced `s["behavior_tests"][0]` with a `next(...)` lookup finding the first test whose `measure.kind == "report_card_axis"`, so the test is robust to non-axis tests appearing first.

### `tests/test_card_verdicts.py`
`test_statistical_card_emits_verdict_and_axes`: added assertion that every entry in `sec["verdict_axes"]` has exactly the 6 required keys `{"id","label","verdict","value","meter","detail"}`.

### `scripts/comparison_report_card.py`
Added comment `# last-graded-section-wins; relies on the one-graded-section-per-card invariant` in `assemble_from_manifest` at the `cardv` collection line. No hard assert added.

---

## Verification Results

### 1. Full feature suite
```
30 passed, 3 warnings in 1.62s
```

### 2. Standalone validator
```
$ .venv/bin/python scripts/validate_comparison_studies.py comparison.5cond_1x4.json
comparison studies OK (match manifest)
exit: 0
```

### 3. Scaffold idempotency
```
$ .venv/bin/python scripts/scaffold_comparison_studies.py comparison.5cond_1x4.json
nothing to write (all studies exist; use --force to overwrite)
$ git status --porcelain workspace/investigations/v2ecoli-vecoli-comparison
(no output — zero modifications)
```

### 4. `store_key` sanity
```
$ .venv/bin/python -c "from scripts._compare.config_adapter import store_key; ..."
basal_4x4
basal
```

---

## Files Touched
- `scripts/_compare/config_adapter.py` — added `store_key`
- `scripts/run_comparison.py` — import + separate sim_condition/store
- `scripts/comparison_report_card.py` — `_config_names` + cardv comment
- `scripts/scaffold_comparison_studies.py` — delegate `condition_name`, sys.path bootstrap
- `scripts/validate_comparison_studies.py` — sys.path bootstrap + return annotation
- `tests/test_validate_comparison_studies.py` — axis-test lookup fix
- `tests/test_card_verdicts.py` — verdict_axes key assertion
