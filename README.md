# v2ecoli

A process-bigraph research workspace. Use the `/pbg-*` skills to drive the
canonical PR flow:

1. `/pbg-add-model <name>` — register a new model (creates a submodule)
2. `/pbg-pull-processes` — pull required pbg-* wrappers
3. `/pbg-data <model>` — curate datasets + references
4. `/pbg-expert-input <model>` — capture expert-stated expected behavior
5. `/pbg-baseline <model>` — build minimal end-to-end composite
6. `/pbg-phase-plan <model>` — lay out multi-phase plan
7. `/pbg-phase <n> <model>` — implement each phase

Optional dashboard:

- `/pbg-server start` — launch local HTTP server for live guidance + progress

See `docs/` and `reports/index.html` for details.
