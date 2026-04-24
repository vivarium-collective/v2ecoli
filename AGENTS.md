# AGENTS.md

Guidance for AI coding assistants working in this repo. Read this before
editing biological process code, composite wiring, or the type system.
Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## What this repo is

v2ecoli is a whole-cell *E. coli* model. It ports 55 biological processes
from [CovertLab/vEcoli](https://github.com/CovertLab/vEcoli) onto
[process-bigraph](https://github.com/vivarium-collective/process-bigraph)
and [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema).
The biology must match upstream unless a PR explicitly justifies a change.

## Frameworks you must understand

### process-bigraph

- A **Composite** = processes + stores (state) + wires (edges from ports to
  store paths). Composites are assembled in `v2ecoli/composite*.py` and the
  corresponding `generate*.py` files build them for each architecture.
- A **Process** declares `inputs` and `outputs` schemas and implements an
  `update(state, interval) -> update` method. Updates are applied to the
  shared store after each timestep.
- A **Step** runs to convergence within a timestep rather than stepping
  through time. Departitioned fuses partitioned requester/evolver halves
  into Steps.
- Don't reinvent composition patterns — copy from an existing process in
  `v2ecoli/processes/` or `v2ecoli/steps/` and adapt.

### bigraph-schema

- Types describe the shape of stores and ports. Project-specific types live
  in `v2ecoli/types/`.
- Types register serializers via `@_serialize.dispatch`. A save state that
  cannot round-trip through the serializer is broken. Test this.
- Units are carried as `pint.Quantity` values; schemas enforce dimension.

For deeper questions about either framework, invoke the `pbg-expert` skill.

## Checks you must apply when adding or editing a process

1. **Schema round-trip.** `serialize(state) -> JSON -> deserialize` must
   reproduce the original state. No pickle anywhere in this path.
2. **Port contract.** Everything `update` reads must be declared in `inputs`;
   everything it writes must be declared in `outputs`. Mismatches are silent
   bugs in process-bigraph.
3. **Units.** Every dimensioned quantity at a port is a `pint.Quantity`.
   No bare floats, no `Unum`. The Unum bridge at
   `v2ecoli/library/unit_bridge.py` exists only for upstream vEcoli interop.
4. **Conservation.** Mass, molecule counts, and charge must balance across
   the update unless the process has an explicit source/sink with a stated
   biological justification.
5. **Behavior test.** A new process requires `tests/test_behavior_<name>.py`
   that runs the process inside a composite and asserts on an outcome
   (growth rate, molecule count, concentration, etc.). A unit test of a
   helper function does not substitute.

## E. coli domain details

- **EcoCyc** is the reference database for E. coli pathways, genes,
  transcription units, and regulatory sites. Processes representing a known
  biological entity should cite the EcoCyc ID (e.g., `EG10001`, `TU0-42`) in
  the process docstring. Parameters for metabolism, transcription, and
  translation all descend from EcoCyc via the ParCa pipeline.
- **ParCa** (Parameter Calculator) builds `sim_data` from raw EcoCyc-derived
  knowledge bases. It's expensive (minutes to hours). Never run ParCa in CI —
  CI uses a frozen gzipped cache at `tests/fixtures/cache/`.
- **Three architectures** all simulate the same cell:
  - `baseline` — partitioned, 55 processes, upstream-parity.
  - `departitioned` — 41 steps, requester+evolver halves fused.
  - `reconciled` — hybrid.
  A change to a process must work across all three, or the PR must explain
  why it's scoped. Use the architecture comparison report to verify.

## Reports

Regenerate the relevant report and inspect it before opening a PR that
touches processes, steps, or composite wiring.

- `reports/workflow_report.py` → `out/reports/workflow_report.html` — full cell lifecycle,
  division at ~42 min.
- `reports/multigeneration_report.py` → `multigeneration_report.html` — N-generation single
  lineage with mass trajectories and fold-change.
- `reports/colony_report.py` → `colony_report.html` — mixed colony with pymunk
  physics, growth, and division.
- `reports/compare_report.py` → `comparison_report.html` — baseline vs departitioned
  vs reconciled, 42-min side-by-side.
- `reports/network_report.py` → `network_*.html` — per-architecture Cytoscape
  topology. Click a process to see ports, schemas, config, docstring, math.
- `reports/v1_v2_report.py` → `v1_v2_comparison.html` — vEcoli 1.0 vs 2.0 vs v2ecoli.
- `reports/benchmark_report.py` — v2ecoli vs vEcoli composite subprocess benchmark.

Published at https://vivarium-collective.github.io/v2ecoli/.

## Tests you must know about

- `tests/test_model_behavior.py` — 7 definitive behavior tests. Gate every
  PR. Do not weaken thresholds without a reviewed justification.
- `tests/test_cell_cycle_regressions.py` — slow full-cycle tests. Run in a
  nightly job, not on every PR. Marked with `@pytest.mark.slow`.
- `@pytest.mark.sim` separates behavior tests from fast tests. CI splits
  them into parallel jobs. Don't remove the marker.
- Behavior fixtures live in `tests/fixtures/`. The pre-division state
  (`pre_division_state.json.gz`) and ParCa cache (`cache/`) are load-bearing.
- `out/cache/` is fingerprinted by `v2ecoli/library/cache_version.py`.
  `make_composite` calls `verify_cache_version` before loading, so a cache
  that was built from a different `models/parca/parca_state.pkl.gz`,
  `v2ecoli/library/sim_data.py`, or unit-bridge raises `StaleCacheError`
  with a one-line rebuild command instead of a 10-frame-deep `AttributeError`.
  Rebuild with `python scripts/build_cache.py` (fast; reuses the committed
  ParCa fixture — no ParCa re-run). See `docs/generate_full_parca.md` for
  the full ParCa path.

## What NOT to do

- Do not modify anything under `tests/fixtures/`. Regenerating a fixture is a
  deliberate act with its own PR.
- Do not edit `.github/workflows/` or bump `pyproject.toml` dependencies
  without explicitly calling it out in the PR description.
- Do not introduce `pickle`, `dill`, or `cloudpickle` in save-state paths.
  Save states round-trip through bigraph-schema JSON. ParCa caches are the
  only exception.
- Do not bypass the Unum/pint boundary in `v2ecoli/library/unit_bridge.py`.
  Unum should appear nowhere else.
- Do not add a process to one architecture without a plan for the other two.
- Do not commit generated reports (`out/`) or ParCa scratch output.

## Before opening a PR

```bash
python scripts/build_cache.py                # regenerate out/cache if stale
pytest -m "not sim"                          # fast tests
pytest -m sim tests/test_model_behavior.py   # behavior tests
```

If you changed a process, also run the relevant report script and attach the
output HTML (or the diff against `main`) to the PR.
