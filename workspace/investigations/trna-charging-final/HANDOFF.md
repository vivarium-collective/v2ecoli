# trna_charging_final port — session handoff

**Branch:** `trna_charging_final` (local, not pushed). Last commit: (Task 2c — about to commit).

**Upstream reference:** `CovertLab/vEcoli@trna_charging_final` at `/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli` (HEAD `330ee3f4`).

**Audit doc:** `workspace/investigations/trna-charging-final/audit.md` — file-by-file map of upstream changes ↔ v2ecoli destinations, status legend, and per-file porting notes. Read this first in a new session.

**Architectural decisions locked from the previous session:**
- `validation/` lives at top-level `v2ecoli/validation/` (mirror upstream).
- `KineticTrnaChargingModel` lands as a **new composite architecture** called `kinetic_charging_baseline` (a fourth one alongside `baseline`, `colony`, `millard_pdmp_baseline`), registered via `@composite_generator`.
- Cython kernel `_trna_charging.pyx` translates to **pure NumPy + numba `@njit`** — precedent is `v2ecoli/processes/polypeptide/kinetics.py`. Reroute `libc rand()` through `numpy.random.RandomState` so `seed_rng` semantics survive.

**Env setup (do this first in every new session):**

```bash
cd /Users/arnabmutsuddy/projects/v2ecoli
uv sync --extra dev --no-install-package vivarium-dashboard
```

`vivarium-dashboard` is excluded because its wheel build fails under strict hatchling — same workaround as CI (PR #141, commit `7579ded`). Without `--no-install-package`, `uv sync` fails with `ValueError: A second file is being added to the wheel archive at the same path: vivarium_dashboard/static/client.js`.

## Remaining tasks (ordered, with dependencies)

The order matters — each task is gated by the ones above it.

| # | Task | Sizing | Notes |
|---|---|---|---|
| ~~2a~~ | ~~Parity-test scaffold~~ | **Done** | 25 cases × 9 functions captured to `tests/fixtures/trna_charging_kernel_golden.json.gz` from upstream Cython kernel built in `vEcoli_trna/.venv`. `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` has the RNG wrapper + 10 NotImplementedError stubs. `tests/test_kinetic_charging_kernel_scaffold.py` (10 tests, all green) gates the golden round-trip + RNG determinism + signature parity. RNG policy documented: stochastic functions parity per-RNG, not byte-identical vs libc rand. See audit.md "Task #2a progress log". |
| ~~2b~~ | ~~7 deterministic kernel functions~~ | **Done** | All 7 (`get_initiations`, `get_codon_at`, `get_candidates_to_C/N`, `select_candidate`, `is_initial_state`, `get_codons_read`) ported as `@njit(error_model="numpy")` and verified bit-identical against the golden via `tests/test_kinetic_charging_kernel.py` (18 passed, 3 skipped for 2c/2d/2e). Notable: `select_candidate` is purely deterministic — `rand()` is called by upstream's *caller*, not by `select_candidate` itself — so the RNG seam is first exercised in 2c. See audit.md "Task #2b progress log". |
| ~~2c~~ | ~~Port `reconcile_via_ribosome_positions`~~ | **Done** | ~140 LOC pure-Python orchestration calling the 2b `@njit`'d helpers. Two non-obvious upstream behaviors preserved: `disagreements_remaining` state leak across attempts (skips phase 1 on attempt 2+), and phase 2's lack of an `exhausted` array. Parity strategy: byte-identity vs committed numpy-RandomState golden, plus invariants (kinetics_codons immutable, non-negativity, conservation, convergence) checked against the libc-rand golden. 20 passed, 2 skipped. See audit.md "Task #2c progress log". |
| 2d | Port `reconcile_via_trna_pools` | 1 session, 3–4 hr | Lines 350–463 (~114 LOC). Pool-balance accounting with stochastic rounding. Route stochastic rounding through the seeded RandomState from 2a, not numpy.random.binomial directly inside `@njit`. Parity-test. |
| 2e | Port `get_elongation_rate` + companion 580-line test | 1 session, 3–4 hr | Lines 464–622 (~159 LOC) — the per-tick rate solver. Then port `wholecell/tests/utils/test_trna_charging.py` (580 lines) → `tests/test_trna_charging_kernel.py`. Must pass under `pytest -m 'not sim'`. |
| 3 | Refresh `polypeptide_elongation.py` + add `KineticTrnaChargingModel` class | 2–3 days | Class is at `polypeptide_elongation.py:2198` upstream. Implement alongside (not replacing) the existing `SteadyStateElongationModel` inside v2ecoli's `polypeptide/` subpackage. Composite wiring goes in a new `v2ecoli/composites/kinetic_charging_baseline.py`. Behavior test `tests/test_behavior_kinetic_charging.py`. |
| 4 | Other process deltas | 1 day | `polypeptide_initiation.py` (+60), `protein_degradation.py` (+19), `transcript_elongation.py` (+30), `tf_binding.py` (+5), `chromosome_structure.py` (+58), `cell_division.py` (+22), `metabolism.py` (+8), `listeners/monomer_counts.py` (+69), `listeners/ribosome_data.py` (+2). |
| 5 | Library deltas | 1 day | `library/sim_data.py` (+212) — **touching this forces `python scripts/build_cache.py` re-run** because it's part of the cache-version fingerprint. `library/initial_conditions.py` (+61), `library/schema.py` (+65). `parquet_emitter.py` deltas may already be covered by recent `feat/default-baseline-parquet`. |
| ~~6~~ | ~~ParCa dataclass deltas~~ | **Done in 518768d** | translation.py, molecule_groups.py, simulation_data.py, transcription.py, growth_rate_dependent_parameters.py applied. two_component_system.py and scripts/nca/run_all.py skipped (upstream-master infra reversion, not tRNA — see audit.md). |
| 8 | Run full ParCa pipeline | hours compute | After #2–#6. See `docs/generate_full_parca.md`. Regenerates `models/parca/parca_state.pkl.gz` with the kinetic re-optimization. |
| 9 | Rebuild `out/cache` | minutes | `python scripts/build_cache.py`. Refingerprints against new `parca_state.pkl.gz`. |
| 10 | Fast tests | minutes | `pytest -m 'not sim' -n auto`. |
| 11 | Behavior tests | tens of minutes | `pytest -m sim tests/test_model_behavior.py`. The 7 gating tests. |
| 12 | Parity gate vs main golden | minutes | `PYTHONPATH=$PWD .venv/bin/python scripts/parity_check.py --seconds 120 --compare tests/golden/baseline_parity_signature.json --build-check`. Expected to **not** be identical for `kinetic_charging_baseline` (this is a new model). The `baseline` arch should remain bit-identical — that's the actual gate. |
| 13 | Reports | hours | `reports/workflow_report.py`, `reports/multigeneration_report.py`, plus a dedicated tRNA-charging HTML with provenance banner via `scripts/pr_session_report.py`. Archive copies per `AGENTS.md`. |

## Suggested session boundaries

Each session should land one logical commit. Recommended split:

- ~~**Session 2:** Tasks #6~~ — Done in 518768d.
- ~~**Session 3:** Task #2a~~ — Done.
- ~~**Session 4:** Task #2b~~ — Done.
- ~~**Session 5:** Task #2c~~ — Done.
- **Session 6 (next):** Task #2d (`reconcile_via_trna_pools`, ~114 LOC). Same pattern as 2c — Python orchestration, `randint_below` for the 3 RNG draws (codon pick + free-vs-charged tRNA pick + ribosome pick). After landing the function, re-run `workspace/investigations/trna-charging-final/capture_numpy_randomstate_golden.py` to refresh the golden with the new cases (the existing one has 2 skipped placeholders waiting), then enable the `test_reconcile_via_trna_pools_parity` test that's already stubbed in `tests/test_kinetic_charging_kernel.py`. Invariants to assert vs libc golden: free+charged tRNA conservation, chargings count never goes negative, amino_acids_used parallel to chargings.
- **Sessions 5 & 6:** Tasks #2c (`reconcile_via_ribosome_positions`) and #2d (`reconcile_via_trna_pools`). Independent — could run in parallel across two sessions if you have the bandwidth.
- **Session 7:** Task #2e (`get_elongation_rate` + companion 580-line test).
- **Session 8:** Task #3 (`KineticTrnaChargingModel` + composite arch + behavior test).
- **Session 9:** Tasks #4 and #5 (process + library deltas — likely intertwined).
- **Session 10:** Task #8 (ParCa run) — mostly compute, can run in background.
- **Session 11:** Tasks #9–#13 (cache rebuild, tests, parity, reports).

## Prompt template for a new session

Paste this verbatim:

```
Continue the trna_charging_final port in v2ecoli. Branch is already
checked out at `trna_charging_final` (last commit 518768d). Read
`workspace/investigations/trna-charging-final/HANDOFF.md` and
`workspace/investigations/trna-charging-final/audit.md` first — they
have the full state, architectural decisions, and remaining task list.

This session: tackle Task <N> from the HANDOFF.md table.
<paste the task's row from the table — subject + sizing + notes>

Remember the structural rule: 2a's golden fixture is what makes 2b–2e
mechanically verifiable. Don't skip it.

Reference clone of upstream is at /Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli
(already on trna_charging_final branch). Run
`uv sync --extra dev --no-install-package vivarium-dashboard` first if
the venv needs rehydrating.
```

Replace `<N>` with the task number and `<paste task description>` with the corresponding row.
