# Plan-B Minor Sweep Report

## Fixes applied

1. **scripts/_compare/materialize.py** — `materialize_study` docstring updated: "report_cards + behavior_tests" → "report_cards + the modular `tests` list".

2. **scripts/_compare/viz_cards.py** — `write_report_cards` return annotation updated from `-> list` to `-> list[Path]`.

3. **scripts/comparison_report_card.py** — In `assemble_from_studies`, `from scripts._compare.viz_cards import write_report_cards` moved out of the unconditional function preamble and into the `if studies_root:` block (just before the `write_report_cards(...)` call). The other two imports (`report_cards as rc`, `write_condition_verdict`) remain at the top of the function.

4. **tests/test_modular_tests_integration.py** — Rewritten as read-only: loads `v2ecoli-vecoli-comparison` and for each spec reads the EXISTING `Path(s.study_path)` study.yaml WITHOUT calling `materialize_study`. Asserts: (a) `report_card` module cards == `sorted(s.cards)`; (b) every `report_cards` entry startswith `viz/report_card/`; (c) `len(data["report_cards"]) == len(s.cards)`.

5. **tests/test_compare_cli.py** — Removed the redundant inner `import scripts.compare_cli as cli` from `test_scaffold_materializes_all_studies` (already imported at module level).

6. **tests/test_viz_cards.py** — Added assertions that `config.verdict.json` is written and its `overall` == `"ungraded"` (the config card has no graded verdict → defaults to ungraded).

## Pytest result

```
.venv/bin/python -m pytest tests/test_viz_cards.py tests/test_materialize.py tests/test_compare_cli.py tests/test_modular_tests_integration.py tests/test_assemble_studies.py -q
14 passed, 3 warnings in 1.60s
```

All 14 tests pass. Warnings are pre-existing (unknown config options + vivarium deprecation), not introduced by this sweep.

## git-status-clean confirmation

```
git status --porcelain workspace/investigations/v2ecoli-vecoli-comparison
(no output — clean)
```

The integration test no longer mutates tracked study YAML files.
