# v2ecoli ↔ vEcoli — multiseed/multigen performance

**Bottom line: v2ecoli and vEcoli run at parity.** Per simulation step they are
identical (~84 ms/step), and on a matched workload their end-to-end wall times are
within 1% (v2ecoli 455 s vs vEcoli 459 s). The headline report is
[`perf_compare.html`](./perf_compare.html) (self-contained; open in a browser).

This document records the full investigation — what we measured, the wins we found,
the dead ends we ruled out, and the v2ecoli internals we learned along the way — so
none of it has to be re-discovered.

---

## TL;DR

| | total wall | peak RAM | per-step |
|---|---|---|---|
| **v2ecoli** (Ray fan-out, step-matched) | **454.8 s** | **3.7 GB** | **83.8 ms** |
| **vEcoli** (Nextflow, 2 natural generations) | 458.6 s | n/a¹ | 84.3 ms |

¹ macOS-local Nextflow (no container engine) doesn't emit per-task RAM.

- **There is no per-cell code gap.** v2ecoli: 670.8 s / 8000 steps = 83.8 ms/step.
  vEcoli: 213 s / 2527 steps = 84.3 ms/step. Same forked science, same speed.
- An earlier draft reported a "3.35× → 1.77×" gap. **That was a benchmark artifact**
  (see below), not a real difference.

---

## What the comparison runs

The harness (`scripts/perf_compare.py`) drives the **same** N-seed × N-generation
job on both engines and captures wall-clock, peak RSS, and per-task timing:

- **v2ecoli** — `scripts/_perf_v2_driver_ray.py`: one `@ray.remote` worker per seed,
  each building the cell and running `run_multigen_sqlite(single_daughters=True)`,
  reporting its own peak RSS via `resource.getrusage`. Threads/worker = cores ÷ seeds
  to avoid BLAS oversubscription. (Sequential variant: `_perf_v2_driver.py`.)
- **vEcoli** — `python -m runscripts.workflow` over a stripped 2-seed × 2-gen config
  (`configs/two_generations.json`), reusing a prebuilt `simData.cPickle` so ParCa is
  excluded; per-task metrics harvested from Nextflow's trace CSV.

Both reuse a prebuilt ParCa `sim_data` and skip analyses, so this isolates
**simulation** execution. Regenerate with
`python scripts/perf_compare.py --engine both --v2-mode ray` then
`python scripts/perf_report.py`.

---

## The artifact that created the illusion of a gap

vEcoli divides **naturally** — each generation runs until the cell divides
(`division_threshold`), ~2527–2882 steps/gen. The first benchmark ran v2ecoli with a
fixed **`max_steps = 8000` cap** for "2 generations," so:

| | steps run (2 seeds × 2 gens) |
|---|---|
| vEcoli (natural division) | **10,834** |
| v2ecoli (8000-step cap) | **16,000** |

v2ecoli simulated **1.48× more sim-time**. The remaining ~1.2× came from BLAS
oversubscription on an un-thread-balanced parallel run. Step-matched
(v2ecoli ~5400 steps/seed = vEcoli's actual 2-generation sim-time), the engines are
at parity.

> **Lesson for future benchmarks:** compare *natural division* or *matched step
> counts*, never a fixed step cap against natural division.

## Is Nextflow making vEcoli faster? No.

vEcoli's sim task is literally
`PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=1 python ecoli/experiments/ecoli_master_sim.py`
— it **caps threads to 1** and gives no `cpus` boost. Nextflow only schedules tasks;
it cannot and does not change per-cell execution speed.

---

## The real, bankable wins

1. **Parallelism via the process-bigraph Ray protocol — 1.89× (2 seeds), scales
   ~linearly with seeds up to core count.** v2ecoli's WCM is GIL/Python-bound per cell
   (≈ 1 core/cell), so the win is running more cells concurrently, not making one cell
   multi-threaded. The external `@ray.remote`-per-seed fan-out is the right model
   (matches vEcoli's Nextflow); an *in-engine* parallel composite matches it but does
   not beat it (independent seeds gain nothing from being coupled in one bigraph).

2. **Peak RAM: 57 GB → ~4–5 GB.** Two fixes:
   - `set_null_emitter_override(True)` disables the redundant **internal** full-state
     `ParquetEmitter` (the runner drives its *own* external emitter); the internal one
     was capturing the entire WCM state every step (falling back to a RAM emitter that
     keeps `tree_copy(state)` forever).
   - Removed an unbounded `chromosome_history` accumulation in `v2ecoli/bridge.py`
     (appended every step, never read).

   Pure memory wins — per-step time is unchanged.

---

## Dead ends (measured and ruled out — don't re-try)

- **FBA dual simplex.** The metabolism LP re-solves ~6000 simplex iterations/step.
  Switching from primal to dual to "warm-start" the bound-changed re-solve is
  **3.5× *worse*** and numerically unstable (floods "numerical instability" warnings).
  Primal (`GLP_PRIMAL`, presolve off) is already the right method.
- **A custom in-place `BulkArray` bigraph type** (vEcoli's `composite` branch has one).
  v2ecoli **already has it** — `bulk_array` (`v2ecoli/types/bulk_numpy.py`), used by all
  16 processes with in-place `count[idx] += value`. The bigraph marshaling tax is only
  ~3–7% of step time (the 4.3M `isinstance` calls are ~1.3% — nanoseconds each).
- **Running the whole sweep as one parallel composite.** Works (bridged
  `LineageProcess` branches on `ray:` addresses + `parallel_processes=True`), but
  matches the external fan-out rather than beating it.
- **Polymerization** is already `@njit` (numba); no headroom there.

The per-step time is ~90% genuine science (FBA solve + JIT'd polymerization + numpy);
the process-bigraph engine itself is ~0.9%. **A large per-cell speedup is not hiding in
the framework** — it requires the structural PDMP / compiled-runtime reformulation.

---

## v2ecoli internals learned

- **Cell = composite-as-process via bridge.** `EcoliWCM` (`v2ecoli/bridge.py`) wraps
  the inner 55-process WCM `Composite` and exposes a narrow bridge
  (mass/length/volume/exchange/chromosome_state).
- **FBA runs via the TOP-LEVEL `wholecell` package**, not
  `v2ecoli.processes.parca.wholecell` (two roots exist — important when patching).
  `Metabolism` is a `Step`, built lazily on first solve.
- **Custom bigraph types already in use:** `bulk_array`, `unique_array`, `units_array`
  (`v2ecoli/types/`), registered via `build_core()` → `ECOLI_TYPES`.
- **Emitters:** the internal `local:ParquetEmitter` is a RAM trap for externally-driven
  runners — minimise it with `set_null_emitter_override`. The implementation lives in
  the separate `pbg-emitters` package (`v2ecoli[parquet]`).
- **Runners:** `run_multigen_sqlite` (single lineage past divisions);
  `workflow/meta_composite.py` + `LineageProcess` (one composite of N seed-branches).
- **Biology:** first natural division ≈ step 2580; `cell_mass` is a unit-wrapped `unum`
  `Quantity`. Open question: a `single_daughters` gen-2 daughter didn't re-divide by
  ~5400 steps while vEcoli's did at ~2851 — a division-timing nuance worth a look.

## Operational gotchas

- Nextflow needs **Java 17+** (`brew install openjdk@17`; set `JAVA_HOME` for the call).
- vEcoli's Nextflow tasks run a bare `python`, so prepend vEcoli's **own** `.venv/bin`
  to `PATH` or they fail on `fsspec`.
- The local `pbg-emitters` editable-install hook is broken under uv (imports only via
  explicit path) → add the checkout to `PYTHONPATH` for the driver and Ray workers.
- Run v2ecoli via `.venv/bin/python` (bare `python` lacks `unum`); share the ParCa
  `out/cache` across worktrees via symlink (rebuild is ~30 min).
