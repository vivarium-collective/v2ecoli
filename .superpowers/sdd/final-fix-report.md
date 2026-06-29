# Final Fix Report: graceful-skip covers applies()/instantiation and per-study failures

Branch: `feat/study-report-card-modules`
Date: 2026-06-29

---

## Changes

### 1. `v2ecoli/workflow/report_cards/__init__.py` — `applicable()` per-card resilience

Wrapped the `step = cls({}, core=core)` + `step.applies(ctx)` block in a try/except inside the loop:

```python
try:
    step = cls({}, core=core)
    if step.applies(ctx):
        out.append(step)
except Exception:  # noqa: BLE001 — one broken card never aborts selection
    continue
```

A card whose `__init__` or `applies()` raises is silently skipped (continue to the next); all previously-passing tests are byte-identical in behavior.

### 2. `scripts/study_report_cards.py` — `run_studies()` helper + per-study resilience

Extracted a new `run_studies(ws_root, study_names, core, only, do_prune) -> int` helper that wraps each `generate_study` call in a try/except:

```python
def run_studies(ws_root, study_names, core, only, do_prune) -> int:
    total = 0
    for s in study_names:
        try:
            total += len(generate_study(ws_root, s, core, only, do_prune)["written"])
        except Exception as e:  # noqa: BLE001 — one study never aborts the run
            print(f"  ! {s}: skipped ({e})")
    return total
```

`main()` now delegates to `run_studies()` instead of the inline loop.

---

## New Tests

### `tests/test_report_card_step.py`

**Added `_BoomApplies` card** (globally registered as `"boom_applies"`) whose `applies()` raises `RuntimeError("boom")`.

**`test_applicable_skips_card_whose_applies_raises`** — RED → GREEN:
- Calls `applicable(ctx, core, only="boom_applies")`.
- Asserts result is `[]` (no raise, card skipped).

### `tests/test_study_report_cards_cli.py`

**`test_run_studies_skips_failing_study_and_continues`** — RED → GREEN:
- Creates two studies: `"good"` (valid `tests:` list) and `"bad"`.
- Monkeypatches `cli.generate_study` to raise for `"bad"`.
- Calls `run_studies(tmp_path, ["good", "bad"], core, only=None, do_prune=False)`.
- Asserts `total >= 1` and `good/viz/report_card/tests.html` exists.

---

## Full Suite Results

```
26 passed, 4 warnings in 3.68s
```

Prior suite: 24 tests. New tests: 2. All 26 green.

---

## No-churn Regeneration Check

Ran `python scripts/study_report_cards.py --study all --prune` → 24 cards across 23 studies.

`git status --porcelain workspace/studies/*/viz/report_card/` → **empty** (no modified files).

---

## Files Touched

- `v2ecoli/workflow/report_cards/__init__.py` — try/except guard in `applicable()`
- `scripts/study_report_cards.py` — `run_studies()` helper + delegating `main()`
- `tests/test_report_card_step.py` — `_BoomApplies` card + `test_applicable_skips_card_whose_applies_raises`
- `tests/test_study_report_cards_cli.py` — `run_studies` import + `test_run_studies_skips_failing_study_and_continues`
