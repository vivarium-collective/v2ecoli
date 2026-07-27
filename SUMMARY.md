# colonies hardening — SUMMARY

## Load-bearing gap picked
The investigation's headline has two halves: a **compute foundation that is
HPC-deployable** (Part A) and the **phenotype science** it enables (Part B). The
quantitative expression of "HPC-deployable" is the **cells-per-node RAM budget**.
On disk that number was internally inconsistent across the investigation:

| source | per-cell RSS | cells/node |
|---|---|---|
| colonies-01 F-03 (commit 2f950d9) | ~450 MB/cell | **384** |
| colonies-02 charts (text) | "~1+ GB/cell, each loads its own sim_data" | (would be ~230) |
| investigation executive | "RSS re-derived from the re-run" | **~1000** |

The `~1000 cells/node` in the executive is backed by **nothing** — no member
study derives it, and the re-run it cites (`f69f9690`, 2026-07-26) re-derived
per-cell *wall* (~57 ms/tick), not RSS. This 2.6× overclaim of the headline
HPC-capacity figure is the single gap with the most leverage that is also
closeable in one bounded headless run on this mini.

## Mode (per skill step 2)
**Overclaimed verdict + passed-but-thin measurement.** The measurement existed
(F-03 N-sweep) but its interpretation was contradicted and the executive inflated
it. Hardening = re-measure per-cell RSS cleanly on current main, isolate the
sim_data-sharing question, and reconcile every citation to one evidence-backed
number.

## What I did
- Traced the mechanism: `v2ecoli/core.py::_load_cache_bundle_cached` is
  `@lru_cache`-keyed by `cache_dir`, and `ecoli_baseline` deep-copies only
  `initial_state` per cell → **within one process, all cells share one sim_data
  bundle by reference**; the per-cell increment is mutable state, not a fresh
  ~1 GB bundle.
- Re-measured per-cell RSS on **current main** (bounded `ColonyPhenotypeRecorder`,
  `emit_cells=False`) by growing a WCM colony N=1→2→4 via forced division within
  one process, decomposing RSS into gc-visible numpy vs native:
  `sims/percell_rss.py` → `runs/percell_rss.csv`.

  | quantity | value |
  |---|---|
  | fixed baseline (imports + sim_data, once/process) | ~888 MB |
  | incremental per-cell RSS (N1→N2, N2→N4) | 282 / 295 → **~291 MB/cell** |
  | numpy delta N1→N4 | +39 MB (flat → **sim_data shared** confirmed) |

  Per-cell RSS **dropped ~450 → ~291 MB** since 2f950d9: the bounded phenotype
  recorder replaced the in-RAM cells-map accumulation.
- Recomputed the budget with the **Ray topology made explicit** (each actor is a
  separate OS process paying its own ~888 MB sim_data): packing ~11 cells/actor
  on 64 actors / 256 GB (per-actor 888 + 11·291 ≈ 4.08 GB < 4.096; compute
  11·57 ms = 627 ms < 1 s realtime) → **~700 cells/node** on current main. This
  supersedes 384 (stale per-cell) and the unsupported ~1000. One-cell-per-actor
  (max GIL parallelism) is RAM-bound at ~215 cells/node.
- Figure: `charts/percell_rss_budget.svg` (RSS-vs-N + budget-vs-assumption),
  via `sims/make_percell_fig.py`.

## Deliverables (committed)
- `colonies-01` F-03: rewritten with the mechanism, the current-main
  re-measurement, and the topology-explicit ~700 cells/node budget; new
  `percell-rss-budget` visualization; description line updated.
- `colonies-02`: corrected the misleading "~1 GB/cell" chart text (that is the
  per-**actor** baseline, not the per-cell increment).
- `investigation.yaml`: executive verdict reconciled (removed unsupported ~1000
  → ~700 with derivation); new `evidence_for` bullet; added `decisions_needed`
  (`ray-actor-cell-packing`) and `followups`
  (`wcm-phenotype-distributions-on-linux`, `percell-rss-under-ray`).
- `sims/percell_rss.py`, `sims/make_percell_fig.py`, `runs/percell_rss.csv`,
  `charts/percell_rss_budget.svg`.

## Verdict
The report card and the verdict now tell the same story as the data: per-cell
RSS ~291 MB (current main), sim_data shared within a process, **~700 cells/node**
with the Ray packing stated — no more 384-vs-1000 contradiction.

## Residual gaps (recorded as followups)
- **WCM phenotype distributions (colonies-08/09) remain PRELIMINARY (n=1–2).**
  A multi-generation WCM colony OOMs on the macOS mini before enough divisions
  (arena artifact), so real distributions are genuinely HPC-blocked here — logged
  as `followups: wcm-phenotype-distributions-on-linux`, not silently capped.
- **Per-cell RSS under the real Ray protocol** (separate processes) is
  extrapolated, not measured — `followups: percell-rss-under-ray`, which also
  resolves `decisions_needed: ray-actor-cell-packing`.

## Compute budget used
One bounded WCM-colony run to N=4 on the mini (~20-tick plateaus; peak RSS
~2.1 GB, well under the ~19 GB pre-division ceiling). No HPC, single seed (0).

## Env note
Shared venv (`~/code/v2ecoli/.venv`) lacked `bigraph_schema.contract` required by
current-main v2ecoli. Shadowed the pinned `bigraph-schema` commit `4b208e13`
(from uv.lock) into a git-ignored `.env-shadow/` and prepended it to PYTHONPATH —
shared venv untouched. Run with:
`PYTHONPATH=~/code/v2e-hcolonies/.env-shadow:~/code/v2e-hcolonies ~/code/v2ecoli/.venv/bin/python ...`
