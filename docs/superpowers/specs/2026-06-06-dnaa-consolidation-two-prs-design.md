# DnaA investigation consolidation → two PRs

**Date:** 2026-06-06
**Author:** Eran (+ Claude)
**Status:** approved design, pre-implementation

## Goal

Make `feat/aim2-dnaa-oric` the single source of truth for the DnaA / replication-initiation
investigation (it generated the canonical report `investigation-dnaa-replication-2026-06-05-7.html`),
and consolidate its 124-commits-vs-`main` of content into **two PRs**:

1. **Infra PR → `main`** (ready for review): general v2ecoli engine/ParCa/test changes the
   investigation depends on, prepared to merge.
2. **Investigation Draft PR** (stacked on the infra branch): the dnaa investigation artifacts —
   studies, feedback rounds, reports, figures, and dnaa-specific analysis scripts.

## Decisions (locked)

- **Authoritative source:** `feat/aim2-dnaa-oric` @ `ef4be45`. Other dnaa branches
  (`dnaa-replication`, `consolidate/dnaa-replication` = PR #99, `investigation/dnaa-replication`,
  `feat/dnaa-atp-equilibrium-mechanism`) are treated as superseded; their unique commits are
  assumed captured by content in aim2. Rashmi's work (50 commits, incl. her tip `5ce2651`) is
  fully contained in aim2; her recovery branch has 0 unique commits.
- **Split rule:** path-default, content-snapshot (not commit replay).
- **Topology:** stacked. `infra/dnaa-replication-support` off `origin/main`;
  `investigation/dnaa-replication-v3` off the infra branch. After infra merges, investigation
  rebases onto `main`.
- **Existing PRs/branches:** `feat/aim2-dnaa-oric`, `dnaa-replication`, PR #97, PR #99 — **left
  fully intact** as a safety net. No history rewrite, no deletion, no force-push. Disposition
  decided later, after Rashmi confirms nothing unpushed. This is the explicit "don't disrupt
  Rashmi's work" guarantee.
- **Plasmid work (19 files):** completely separate investigation — **excluded from both PRs**,
  untouched. Ships via the plasmids investigation's own PR.
- **dnaa-specific scripts (12) + `dnaa_box_binding.py` deriver:** → investigation PR, not `main`.

## File sets (200 files vs `origin/main`, no overlap)

- **INFRA → main (61):** `v2ecoli/{core.py,__init__.py,types,cli/parca.py}`,
  `library/{division,schema,schema_types,sim_data,initial_conditions,function_registry}.py`,
  `composites/{baseline,_helpers,__init__}.py`, `processes/{chromosome_replication,equilibrium}.py`,
  ParCa dataclasses/steps + non-plasmid flat TSVs, `steps/{allocator,division}.py`,
  general run/compare scripts, `tests/`, `uv.lock`, `pyproject.toml`, `workspace.yaml`, `models/parca/*`.
- **INVESTIGATION → Draft (120):** `investigations/dnaa-replication/**` (incl. 8 feedback rounds),
  `studies/dnaa-*/**`, dnaa `reports/**`, dnaa `docs/**`, figures, the 12 dnaa-specific scripts,
  `v2ecoli/steps/derivers/dnaa_box_binding.py`.
- **PLASMID — untouched (19):** `processes/plasmid_replication.py`, `composites/plasmids.py`,
  plasmid ParCa flat files, `scripts/*plasmid*`, `scripts/plot_colE1*`, plasmid reports, plasmid docs.

## Mechanism

1. `git branch infra/dnaa-replication-support origin/main`; in a fresh worktree, `git checkout
   feat/aim2-dnaa-oric -- <INFRA set>`; one commit with `Co-authored-by: RashmiKaldera
   <kalderadissasekara@uchc.edu>`.
2. `git branch investigation/dnaa-replication-v3 infra/dnaa-replication-support`; checkout the
   INVESTIGATION set from aim2; one commit (+ this spec).
3. Push both. Open infra PR (base `main`, ready) and investigation PR (base
   `infra/dnaa-replication-support`, **Draft**), each documenting the stack.

## Verification gates (before infra PR marked ready)

- Infra branch: `uv` env imports cleanly; `v2ecoli` package imports with no plasmid dependency
  errors (if a plasmid import breaks, revisit set A).
- `tests/test_composites_baseline.py` passes.
- ParCa flat-file parameter changes (esp. `transcription_units_removed.tsv` TU00434/35 removal,
  `equilibrium_reactions.tsv`, `per_promoter_ratios.tsv`) explicitly called out in the PR body
  for human review — confirm they are intended permanent calibration, not investigation-only
  overrides, before merge.

## Out of scope

Retiring PR #97/#99, repointing the dnaa dashboards, importing the next Rashmi/Haochen feedback
round. Tracked separately.
