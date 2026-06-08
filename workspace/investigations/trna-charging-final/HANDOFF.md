# trna_charging_final port — session handoff

**Branch:** `trna_charging_final` (local, not pushed). Last commit: `885df04 infra(trna-charging): port Relation dataclass + ParCa flat data + validation tree`.

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
| 2 | Port `wholecell/utils/_trna_charging.pyx` (638 lines Cython) to NumPy+numba | 1–2 days | Lands in `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py`. Companion test `wholecell/tests/utils/test_trna_charging.py` (580 lines) → `tests/test_trna_charging_kernel.py`. Capture golden outputs from the upstream Cython kernel on a fixed seed for parity. |
| 3 | Refresh `polypeptide_elongation.py` + add `KineticTrnaChargingModel` class | 2–3 days | Class is at `polypeptide_elongation.py:2198` upstream. Implement alongside (not replacing) the existing `SteadyStateElongationModel` inside v2ecoli's `polypeptide/` subpackage. Composite wiring goes in a new `v2ecoli/composites/kinetic_charging_baseline.py`. Behavior test `tests/test_behavior_kinetic_charging.py`. |
| 4 | Other process deltas | 1 day | `polypeptide_initiation.py` (+60), `protein_degradation.py` (+19), `transcript_elongation.py` (+30), `tf_binding.py` (+5), `chromosome_structure.py` (+58), `cell_division.py` (+22), `metabolism.py` (+8), `listeners/monomer_counts.py` (+69), `listeners/ribosome_data.py` (+2). |
| 5 | Library deltas | 1 day | `library/sim_data.py` (+212) — **touching this forces `python scripts/build_cache.py` re-run** because it's part of the cache-version fingerprint. `library/initial_conditions.py` (+61), `library/schema.py` (+65). `parquet_emitter.py` deltas may already be covered by recent `feat/default-baseline-parquet`. |
| 6 | Remaining ParCa dataclass deltas | 1 day | `dataclasses/process/transcription.py` (+70), `process/two_component_system.py` (+103), `process/translation.py` (+9), `dataclasses/molecule_groups.py` (+21), `dataclasses/getter_functions.py` (small), `simulation_data.py` (+27), `growth_rate_dependent_parameters.py` (+169), `scripts/nca/run_all.py` (+12). |
| 8 | Run full ParCa pipeline | hours compute | After #2–#6. See `docs/generate_full_parca.md`. Regenerates `models/parca/parca_state.pkl.gz` with the kinetic re-optimization. |
| 9 | Rebuild `out/cache` | minutes | `python scripts/build_cache.py`. Refingerprints against new `parca_state.pkl.gz`. |
| 10 | Fast tests | minutes | `pytest -m 'not sim' -n auto`. |
| 11 | Behavior tests | tens of minutes | `pytest -m sim tests/test_model_behavior.py`. The 7 gating tests. |
| 12 | Parity gate vs main golden | minutes | `PYTHONPATH=$PWD .venv/bin/python scripts/parity_check.py --seconds 120 --compare tests/golden/baseline_parity_signature.json --build-check`. Expected to **not** be identical for `kinetic_charging_baseline` (this is a new model). The `baseline` arch should remain bit-identical — that's the actual gate. |
| 13 | Reports | hours | `reports/workflow_report.py`, `reports/multigeneration_report.py`, plus a dedicated tRNA-charging HTML with provenance banner via `scripts/pr_session_report.py`. Archive copies per `AGENTS.md`. |

## Suggested session boundaries

Each session should land one logical commit. Recommended split:

- **Session 2:** Tasks #6 (the small ParCa dataclass deltas — quickest, useful warm-up).
- **Session 3:** Task #2 (Cython → numba kernel + companion test). This is the highest-risk piece; isolate it.
- **Session 4:** Task #3 (KineticTrnaChargingModel + composite arch + behavior test).
- **Session 5:** Tasks #4 and #5 (process + library deltas — likely intertwined).
- **Session 6:** Task #8 (ParCa run) — mostly compute, can run in background.
- **Session 7:** Tasks #9–#13 (cache rebuild, tests, parity, reports).

## Prompt template for a new session

Paste this verbatim:

```
Continue the trna_charging_final port in v2ecoli. Branch is already
checked out at `trna_charging_final` (last commit 885df04). Read
`workspace/investigations/trna-charging-final/HANDOFF.md` and
`workspace/investigations/trna-charging-final/audit.md` first — they
have the full state, architectural decisions, and remaining task list.

This session: tackle Task <N> from the HANDOFF.md table.
<paste task description>

Reference clone of upstream is at /Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli
(already on trna_charging_final branch). Run
`uv sync --extra dev --no-install-package vivarium-dashboard` first if
the venv needs rehydrating.
```

Replace `<N>` with the task number and `<paste task description>` with the corresponding row.
