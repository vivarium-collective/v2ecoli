# Polypeptide Elongation: strategy classes → wireable processes

**Date:** 2026-05-30
**Status:** design (awaiting review)
**Scope:** `v2ecoli/processes/polypeptide_elongation.py` + `v2ecoli/processes/polypeptide/`

## Problem

Polypeptide elongation has three behavioural variants —
`BaseElongationModel`, `TranslationSupplyElongationModel`,
`SteadyStateElongationModel` — implemented today as a strategy-pattern
inheritance chain of plain Python classes (`object` subclasses) living in
`v2ecoli/processes/polypeptide/elongation_models.py`. A single host process,
`PolypeptideElongation(PartitionedProcess)`, instantiates one of them in
`initialize()` based on two config flags (`translation_supply`,
`trna_charging`) and delegates five hooks to it:
`elongation_rate`, `amino_acid_counts`, `request`, `final_amino_acids`,
`evolve`.

Consequences we want to fix:
- The model choice is a hidden config flag, not a visible wiring decision.
- The variants' true data dependencies are invisible: the host declares one
  union of ports even though only `SteadyState` touches charged-tRNA / ppGpp
  stores.
- `elongation_models.py` (850 lines) + the host (726 lines) are a single
  tangled unit; the strategy indirection (`self.process.X` back-references)
  adds friction.

## Goals

- Turn the three variants into **three sibling process-bigraph
  `PartitionedProcess` classes** that are wired into a composite directly.
- Model choice becomes a **wiring choice** (which class is registered for
  `ecoli-polypeptide-elongation`), not a config flag.
- Each variant declares **only the ports its math uses**.
- **Bit-identical baseline behaviour**: the default-wired variant
  (`SteadyState`) reproduces today's baseline trajectory exactly.
- Delete the `translation_supply` / `trna_charging` selector flags.

## Non-goals

- Transcript elongation — **explicitly skipped** this pass (it has no
  Base/Supply/SteadyState trio; its only variant axes are a
  `variable_elongation` flag and tRNA attenuation, which is already modular
  via the `attenuation_config` store + `trna-attenuation-config` step +
  `trna_attenuation` feature module).
- No change to the partition/allocator scheme, the polymerize machinery, the
  ppGpp/charging **math**, or the `ppgpp-initiation` step.
- No decomposition of `SteadyState` into separate charging/ppGpp/supply
  processes (the "decompose by concern" option was considered and declined;
  we keep swap-of-three).

## Approach — Base process + subclass chain (chosen)

Fold the strategy classes *up* into a small inheritance chain of real
`PartitionedProcess`es that mirrors the existing model chain:

```
PartitionedProcess
  └─ BasePolypeptideElongation              # was PolypeptideElongation host
       │                                    #  + BaseElongationModel hooks inlined
       └─ TranslationSupplyPolypeptideElongation
            └─ SteadyStatePolypeptideElongation   # default wired variant
```

- The host's shared scaffolding — `initialize()` (minus model selection),
  `_init_bulk_indices()`, base `inputs()`/`outputs()`, the
  `calculate_request()` / `evolve_state()` skeletons — lives on
  `BasePolypeptideElongation`.
- The five hook methods become **methods on the base** (with `Base`
  behaviour) that subclasses override. The current model-class method bodies
  move in **verbatim**, with `self.process.X` rewritten to `self.X`.
- `elongation_models.py` is **deleted**. Pure helpers it leaned on
  (`polypeptide/kinetics.py`, `polypeptide/common.py`) stay and are imported
  by the process classes.

### Why not the alternative (three flat independent classes)
Rejected: it would duplicate ~200 lines of host scaffolding across three
classes or push it into free functions, raising churn and parity risk for no
benefit — the variants are genuinely related (Supply refines Base; SteadyState
refines Supply), so the chain is the honest structure.

## Target module layout

- `v2ecoli/processes/polypeptide_elongation.py`
  - `BasePolypeptideElongation(PartitionedProcess)` — scaffolding + Base hooks.
  - `TranslationSupplyPolypeptideElongation(BasePolypeptideElongation)` —
    overrides `elongation_rate`, `amino_acid_counts`.
  - `SteadyStatePolypeptideElongation(TranslationSupplyPolypeptideElongation)`
    — overrides `request`, `final_amino_acids`, `evolve`; carries the
    `_amino_acid_supply`, `_ppgpp_request`, `_ppgpp_evolve`,
    `distribution_from_aa` helpers.
  - (If the file grows unwieldy, `SteadyState` may move to
    `polypeptide/steady_state.py`; decide during implementation by file size.)
- `v2ecoli/processes/polypeptide/elongation_models.py` — **deleted**.
- `v2ecoli/processes/polypeptide/{kinetics,common}.py` — unchanged.

## Ports diverge by variant (the payoff)

- `BasePolypeptideElongation` / `TranslationSupply…` declare the base
  inputs/outputs (bulk AAs, GTP, ribosomes, listeners, partition stores).
- `SteadyState…` **overrides** `inputs()`/`outputs()` to add the stores only
  it reads/writes: charged/uncharged tRNA, synthetases, ppGpp, and the
  `process_state.polypeptide_elongation` AA-exchange channel. The wiring then
  shows each variant's real dependency surface.

## Selection & wiring migration

- Register all three classes (bigraph-schema discovery / `local:` addresses).
- `v2ecoli/composites/_helpers.py` `PARTITIONED_PROCESSES` maps
  `'ecoli-polypeptide-elongation' → SteadyStatePolypeptideElongation`
  (the default — preserves baseline parity). `departitioned` / `reconciled`
  reuse this map, so they inherit the same default automatically.
- A study selects a different variant by overriding that registry entry (or a
  generator-level `elongation_variant` param resolving to the class). The
  hidden flags are gone; the wired class *is* the choice.

## Config-flag removal & ParCa/sim_data migration

- Remove `translation_supply` and `trna_charging` from the elongation
  process `config_schema`.
- `v2ecoli/library/sim_data.py:1095-1096` currently injects both flags into
  the elongation process config (and imports the model classes). Update it to
  **stop emitting the two keys** into that config dict and drop the now-unused
  model-class import. The ParCa-internal `self.translation_supply` /
  `self.trna_charging` attributes (which also drive ParCa fitting, separate
  concern) stay.
- The ParCa cache (`out/cache`) currently stores the two keys in the
  elongation config. The loader must tolerate/strip unknown keys so existing
  caches still build (filter the two keys on load, or rebuild the cache).
  Pick the filter approach so old caches keep working.

## Behaviour parity & tests (AGENTS.md gate)

- **Golden parity test:** `SteadyStatePolypeptideElongation` wired in the
  baseline reproduces the pre-refactor trajectory — assert `dry_mass` at
  `t=200` and at division, plus `monomer_counts` totals, match a captured
  golden within float-noise (target bit-identical). RNG call **order** must be
  preserved when moving method bodies, or trajectories drift.
- **Per-variant behaviour tests:** `tests/test_behavior_polypeptide_elongation_<variant>.py`
  — each of Base / Supply / SteadyState runs inside a minimal composite and
  asserts a sane outcome (ribosomes elongate, protein mass increases).
- Migrate existing tests that import the deleted symbols:
  `tests/test_kinetics_units.py`, `tests/test_parca_fixture_roundtrip.py`.

## Blast radius (files to touch)

- `v2ecoli/processes/polypeptide_elongation.py` — rewrite into 3 classes.
- `v2ecoli/processes/polypeptide/elongation_models.py` — delete.
- `v2ecoli/composites/_helpers.py` — registry entry → SteadyState class.
- `v2ecoli/library/sim_data.py` — stop emitting flags; fix imports.
- cache config loader — strip the two removed keys.
- `tests/test_kinetics_units.py`, `tests/test_parca_fixture_roundtrip.py` —
  update imports/refs.
- `scripts/baseline_report_html.py` — update the elongation doc text
  (mentions the two flags as config knobs); non-blocking.

## Risks & mitigations

- **Parity drift from reordering** → move method bodies verbatim; diff the
  AA-exchange / RNG call sequence; gate on the golden test.
- **Cache-config breakage** from deleted schema keys → tolerant loader that
  filters the two keys.
- **Hidden coupling in `sim_data.py`** importing model classes → audit why it
  imports them before deleting `elongation_models.py`.
- **Discovery/registration** of three partitioned processes → verify all
  three resolve via the registry and `build_composite('baseline')` still
  builds.

## Done when

- `build_composite('baseline')` wires `SteadyStatePolypeptideElongation` and
  reproduces the golden trajectory.
- The three variants are independently wireable; the two flags are gone.
- `elongation_models.py` deleted; full test suite (incl. new behaviour +
  parity tests) green.
