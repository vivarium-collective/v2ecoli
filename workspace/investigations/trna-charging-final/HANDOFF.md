# trna_charging_final port — session handoff

**Branch:** `trna_charging_final` (local, not pushed). Last commit: (Task 3d — about to commit). **Task #2 complete; Task #3 split into 3a–3f, 3a–3d done. 3e collapsed into 3d (listener paths folded into the `evolve_state` override).**

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
| ~~2d~~ | ~~Port `reconcile_via_trna_pools`~~ | **Done** | ~95 LOC pure-Python orchestration. Two-branch structure (free-tRNA branch returns free→charged; charged-tRNA branch undoes charging+read with no net tRNA-state change). Key invariant: `kinetics_codons` final state is RNG-invariant (loop runs until disagreements=0). Numpy-RandomState golden refreshed to 10 cases. 22 passed, 1 skipped. See audit.md "Task #2d progress log". |
| ~~2e~~ | ~~`get_elongation_rate` + companion test~~ | **Done** | ~110 LOC `@njit` binary search. Companion 580-line upstream test effectively ported via the golden-fixture approach in earlier sub-tasks; `test_get_elongation_rate_parity` and `test_reconcile_seed_propagates_to_kernel_output` are the remaining 2e-specific cases. Surprising finding: upstream's big_seed_12345 vs big_seed_54321 inputs converge to identical state under numpy.RandomState (real RNG-coincidence, not a bug); used `attempts_threshold` inputs for the divergence test. 24 passed, 0 skipped, runs under `-m 'not sim'`. See audit.md "Task #2e progress log". |
| ~~3a~~ | ~~Class scaffold~~ | **Done** | `v2ecoli/processes/polypeptide/kinetic_charging.py` defines `KineticTrnaChargingPolypeptideElongation` as a peer subclass of `BasePolypeptideElongation`. `config_schema` dict-merges 12 new kinetic-charging-specific keys onto the base. 14 method stubs raise `NotImplementedError` with an explicit task marker per owning sub-task. `tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` has 20 tests (structural + parametric marker checks). 20 passed in 3.80 s. See audit.md "Task #3a progress log". |
| ~~3b~~ | ~~`__init__` + `get_kinetic_constants`~~ | **Done** | ~95 LOC across two methods. `initialize` calls `super().initialize(config)` then unpacks all 12 kinetic-charging keys + derives slice layout + `trnas_to_amino_acid_indexes`. `get_kinetic_constants(cell_mass)` returns volume-scaled K_M arrays. 23 scaffold tests pass (5 new for 3b incl. cache-gated end-to-end instantiation + K_M-doubling-with-mass check). Test helper `_make_kinetic_extensions` shipped for 3c–3e to reuse. See audit.md "Task #3b progress log". |
| ~~3c~~ | ~~`elongation_rate` + `request` + helpers~~ | **Done** | ~430 LOC across 6 methods + `_init_bulk_indices` override that adds ATP/AMP/PPi/Met/MAP indices. `elongation_rate` re-derives `(protein_indexes, peptide_lengths)` from `states["active_ribosome"]` (v2ecoli's contract). `request` ignores the amino-acid-domain `aasInSequences` arg, recomputes the codon-domain version from `self.longer_sequences`, and emits bulk requests for AAs (+1% buffer), ATP, tRNAs, synthetases, MAP, water. `run_model` is the ~270-LOC RK45 ODE driver with sin/sin² roll-offs and per-tRNA reconciliation. 28 scaffold tests pass (5 new for 3c, all source-scan). See audit.md "Task #3c progress log". |
| ~~3d~~ | ~~`evolve` + reconcile + protein_maturation + evolve_state override~~ | **Done** | ~450 LOC across 7 method bodies + the ~225-LOC `evolve_state` override (v2ecoli's base is amino-acid-centric; kinetic needs codon-based polymerize → reconcile → protein_maturation → evolve). `final_amino_acids` is kept as a NotImplementedError raise with an explanatory bypass message. `monomer_to_aa`, `monomer_limit`, `next_amino_acids` pulled forward from 3e because the override consumes them. 31 scaffold tests pass (8 new for 3d, all source-scan). See audit.md "Task #3d progress log". |
| ~~3e~~ | ~~Per-codon cap + listeners~~ | **Collapsed into 3d** | `monomer_to_aa`, `monomer_limit`, `next_amino_acids` pulled into 3d; listener emission folded into the `evolve_state` override. Nothing left for a separate 3e session. |
| 3f | Composite arch + behavior test | 1 session, 2 hr (port) + blocked on #5 | `v2ecoli/composites/kinetic_charging_baseline.py` with `@composite_generator`. Update `__init__.py` + `cache_version.py`. Behavior test in `tests/test_behavior_kinetic_charging.py`. Blocked on Task #5 for end-to-end run. |
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
- ~~**Session 6:** Task #2d~~ — Done.
- ~~**Session 7:** Task #2e~~ — Done. **Task #2 fully complete.**
- ~~**Session 8:** Task #3a~~ — Done. `KineticTrnaChargingPolypeptideElongation` scaffold + 20 smoke tests in `v2ecoli/processes/polypeptide/kinetic_charging.py`.
- ~~**Session 9:** Task #3b~~ — Done. `initialize` + `get_kinetic_constants` ported; 23 scaffold tests pass (5 new for 3b, incl. cache-gated end-to-end instantiation). Synthetic config helper `_make_kinetic_extensions` available to 3c–3e.
- ~~**Session 10:** Task #3c~~ — Done. 6 request-side methods + `_init_bulk_indices` override; 5 new scaffold tests (28 total pass, 50 in the wider kinetic-charging fast-test bucket).
- ~~**Session 11:** Task #3d~~ — Done. `evolve_state` override + 7 evolve-side methods; 8 new scaffold tests (31 total pass, 53 in the wider kinetic-charging fast-test bucket).
- ~~**Session 12:** Task #3e~~ — Collapsed into 3d.
- **Session 13 (next):** Task #5 (library/sim_data deltas) — populates the new `config_schema` keys from `sim_data.relation`. Unblocks 3f's behavior test. Most of the data already exists on `sim_data.relation` after Task #6 ported the Relation dataclass and its `_build_trna_charging_kinetics` method; the work is mapping those attrs into v2ecoli's config-build path.
- **Session 14:** Task #3f — composite arch + behavior test. Build `v2ecoli/composites/kinetic_charging_baseline.py` via `@composite_generator`, append to `composites/__init__.py`, update `library/cache_version.py:INPUT_FILES`, then a `tests/test_behavior_kinetic_charging.py` that builds the composite and asserts one-tick growth.
- **Session 15:** Task #4 — `metabolism_redux_classic` + other process deltas (independent of #3/#5).
- **Session 16:** Task #8 — full ParCa rerun. Mostly compute; can run in background.
- **Session 17:** Tasks #9–#13 — cache rebuild + tests + parity + reports.

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
