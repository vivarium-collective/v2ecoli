# Contributing to v2ecoli

Thanks for your interest in contributing. v2ecoli is a whole-cell *E. coli*
model built on [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
and [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema).
Because changes here alter simulated biology, we hold PRs to a higher bar than
most software projects: behavior must be preserved (or the deviation
justified), units must be consistent, and new processes must come with tests.

If you are an AI coding assistant, please also read [AGENTS.md](AGENTS.md).

## Getting started

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
python -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
```

The first ParCa (parameter calculator) run builds `sim_data` from raw
EcoCyc-derived knowledge bases and is cached under `out/cache/`. CI uses a
frozen gzipped cache in `tests/fixtures/cache/` — don't regenerate it in CI.

## Running tests

Fast tests (unit-level, no simulation):

```bash
pytest -m "not sim"
```

Behavior tests (run short simulations, validate model output):

```bash
pytest -m sim tests/test_model_behavior.py
```

Full CI-equivalent locally:

```bash
pytest
```

Slow end-to-end regressions (`tests/test_cell_cycle_regressions.py`) run in a
nightly job, not on every PR.

## PR checklist

Before opening a PR:

- [ ] Link to a tracking issue (or open one) describing the motivation.
- [ ] Tests added or updated. New processes require a
      `tests/test_behavior_<process>.py`.
- [ ] Behavior preserved. If a change alters simulated trajectories, re-run
      `reports/workflow_report.py` and attach the generated report to the PR, explaining the
      expected effect.
- [ ] Units consistent. All quantities at process boundaries use `pint`.
- [ ] No `pickle`/`dill` in save-state paths. Save states are bigraph-schema
      JSON (optionally gzipped); see `v2ecoli/cache.py`. (ParCa outputs are
      the one exception, and live under `out/cache/` / `tests/fixtures/cache/`.)
- [ ] New processes cite their EcoCyc ID where applicable and reference the
      upstream vEcoli source when porting.
- [ ] CI is green.

## Hard rules

These are load-bearing and enforced on review (and, over time, in CI):

1. **Save states are bigraph-schema JSON, never pickle/dill.** Pickled state
   is not portable across bigraph-schema versions and caused past silent
   corruption. ParCa caches are the only exception.
2. **`pint` at every process boundary.** Upstream vEcoli is Unum-native; the
   translation layer lives in `v2ecoli/library/unit_bridge.py`. Do not
   re-introduce Unum elsewhere, and do not pass bare floats through ports
   that carry dimensioned quantities.
3. **New processes need a behavior test.** A process without a test will not
   be merged. Unit tests of helpers don't count — the test must run the
   process inside a composite and check an outcome.
4. **Three architectures.** v2ecoli runs baseline (partitioned, 55 processes),
   departitioned (41 steps), and reconciled (hybrid). A change to a process
   should work across all three, or the PR must explain why it's scoped to
   one.

## Reports

PRs that touch `v2ecoli/processes/`, `v2ecoli/steps/`, or composite wiring
should regenerate and sanity-check the relevant reports:

- **Workflow report** (`reports/workflow_report.py`) — full cell lifecycle, division at
  ~42 min.
- **Multigeneration report** (`reports/multigeneration_report.py`) — N-generation single
  lineage.
- **Colony report** (`reports/colony_report.py`) — mixed colony with pymunk physics.
- **Architecture comparison** (`reports/compare_report.py`) — baseline vs
  departitioned vs reconciled.
- **Network views** (`reports/network_report.py`) — per-architecture Cytoscape
  topology with ports, schemas, and math.

Published versions live at https://vivarium-collective.github.io/v2ecoli/ and
are republished from `main` on merge.

## Review

`CODEOWNERS` automatically requests review from the maintainer for changes
in biological process code, the type system, composite wiring, CI, and
fixtures. Expect a 1–3 day turnaround. For larger changes, open a draft PR
early so we can align on approach before the work is done.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant](CODE_OF_CONDUCT.md). Please report concerns to
Eran Agmon at `agmon@uchc.edu`.
