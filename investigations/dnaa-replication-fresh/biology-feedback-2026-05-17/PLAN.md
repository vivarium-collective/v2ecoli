# Plan: applying biology-review feedback to the dnaA investigation

**Source:** `references/expert/dnaa_biology_feedback.pdf` (registered in `workspace.yaml.expert_docs[dnaa_biology_feedback]`)
**Date:** 2026-05-17
**Author:** External biology reviewer
**Workspace:** v2ecoli

## TL;DR of the feedback

The decomposition (dnaa-01..06) is sound. The current biological-interpretation layer is not. Three changes are non-negotiable before any biological claim should stand:

1. **Separate target classes.** Every gate test must declare one of `regression_compatibility` / `biological_validation` / `explanatory_gain`. A heuristic-match test cannot count as biological validation.
2. **Map literature → model observable.** Every numeric threshold must carry a `measurement_mapping` (literature_observable + model_observable + transformation + caveats). No "DnaA in [300, 800]" without saying which pool and which transformation.
3. **Perturbations validate mechanisms.** Wild-type baseline alone cannot close biological validation. Each study must declare a perturbation panel.

Plus four more P1 items: explicit DnaA pool decomposition, autorepression test-SUITE (not just Pearson r), DnaA-ATP decomposition (free / DNA-bound / oriC-bound / non-oriC-bound), and uncertainty as actionable expert questions.

## Where this changes our findings

Specific findings that are now overclaimed and need re-labeling:

| Finding | Current status | After review |
|---|---|---|
| dnaa-01 F-01: "DnaA count 115 vs literature [300, 800]" | `biological/contradicts` | **observable mismatch** — which pool is the literature 300-800? mass-spec → total cellular DnaA; v2ecoli's `monomer_counts[3861]` is ambiguous (apo? all-three-form sum?). Needs `measurement_mapping`. |
| dnaa-01 F-02: "Autorepression Pearson r = -0.594 PASSES" | `biological/confirms` | Pearson is the WEAKEST autorepression test. PASS-by-r is insufficient evidence. Promote to "promoter-response test suite ran, 1 of 4 tests passed". |
| dnaa-01g (TE=20×, fc=0.7) "both gates pass" | `gate_status: open` | downgrade to `regression_compatibility: PASS` and `biological_validation: PENDING perturbation panel`. The fc=0.7 number is heuristic regression; doesn't say WHY autorepression is over-strong in v2ecoli. |
| dnaa-02 F-04/F-05: "k=4.6/min PASSES Boesen band" | `biological/novel` | **regression_compatibility** — the 100× ratio is what's needed to FIT the band given v2ecoli's equilibrium reverse-rate. Doesn't validate biology; could equally mean the equilibrium reverse-rate is mis-calibrated by 100×. |
| dnaa-03 F-05: "Simple Hill model fails 2-step" | `biological/contradicts` | correct — but the next-step "cooperative Hill n=5-10" is itself a `regression_compatibility` target until tested with literature DnaA-ATP perturbations. |

## TODO matrix (P1 → P3)

### P1 — must fix, schema changes touch every study

| # | TODO | Files touched | Owner |
|---|---|---|---|
| P1-1 | `target_class:` field on every `behavior_tests[]` entry | all 10 study.yaml | this task |
| P1-2 | `measurement_mapping:` block on every numeric threshold | all 10 study.yaml (in `behavior_tests` + `findings.evidence`) | this task |
| P1-3 | Replace Pearson-only autorepression test with `autorepression_tests[]` suite | dnaa-01, dnaa-01g, dnaa-02..06 readouts | this task + dnaa-01 |
| P1-4 | Explicit DnaA pool decomposition in readouts: `total / free / DNA-bound / oriC-bound / non-oriC-bound / ATP-bound / ADP-bound / apo` | dnaa-01 readouts; dnaa-02 readouts (this is where the split lives); enforce across dnaa-03..06 | this task |
| P1-5 | `perturbation_panel:` block on each study; tests that REQUIRE perturbation evidence get tagged `requires_perturbations: [...]` | dnaa-01..06 | this task |

### P2 — next cycle

| # | TODO | Notes |
|---|---|---|
| P2-1 | DnaA-box catalog expert curation: each box gets `{region_type, box_class, affinity, nucleotide_preference, cooperative_group, source, expert_review_status}` | dnaa-03's box catalog work — extend DNAA_BOX_ARRAY schema in `v2ecoli/library/schema_types.py` and the ParCa initial-conditions setup that populates it |
| P2-2 | Version SeqA abstractions: `seqa_v0` (fixed 10-min timer), `seqa_v1` (methylation-state + Dam re-methylation kinetics), `seqa_v2` (cooperative SeqA-Dam) | dnaa-06 study; each version is its own recipe (see baseline_recipes.py) |
| P2-3 | RIDA/DDAH/DARS as RESET MECHANISMS not "hydrolysis knobs" — each gets a mechanism card | dnaa-05 study |
| P2-4 | Test-failure classifier: each FAIL labels `failure_cause: [missing_mechanism, miscalibration, observable_mismatch, insufficient_statistics]` | evaluator in `tests/_behaviors.py` |

### P3 — improvements

| # | TODO | Notes |
|---|---|---|
| P3-1 | Mechanism cards (`mechanism_cards/` per study) for expert-editable assumptions | small templated md file per assumption |
| P3-2 | "What would change my mind" boxes on every major conclusion | add to `conclusion_logic` |
| P3-3 | Expert references → answerable expert questions with `alternatives, impact_if_wrong, blocking_status, requested_response_type` | replace flat `expert_references` |

## Schema additions to study.yaml

These are the minimal additions; the per-study application below shows usage.

```yaml
behavior_tests:
- name: dnaA-count-in-range
  target_class: biological_validation    # NEW (P1-1)
  classification: primary
  description: ...
  measurement_mapping:                   # NEW (P1-2)
    literature_observable:
      name: total cellular DnaA per cell
      biological_pool: total DnaA across apo + ATP + ADP + DNA-bound complexes
      condition: M9 + glucose, exponential growth, doubling ~30 min
      statistic: median across population
      source_ids: [Schmidt2016NatBiotechnol, Sekimizu1991JBacteriol]
    model_observable:
      listener_path: listeners.monomer_counts
      state_path: bulk[PD03831[c]] + bulk[MONOMER0-160[c]] + bulk[MONOMER0-4565[c]]
      molecular_pool: total DnaA (sum across three forms)
      units: molecules/cell
      includes_bound_species: true       # critical: clarifies whether DNA-bound is in pool
    transformation:
      formula: median(sum_three_forms over t in [duration/2, duration])
      time_window: second_half
      aggregation: median across seeds
    caveats:
      - 'v2ecoli does not yet model DnaA bound to RIDA replisome complexes; if literature 300-800 includes those, sum understates by ~10%'
      - 'Schmidt 2016 mass-spec excludes DnaA bound during fixation washouts; ratio uncertain'
  measure: {kind: median_window, ...}
  expect: {op: in_range, range: [300, 800]}

readouts:
- name: dnaA_pool_total
  pool_kind: total                # NEW (P1-4): one of total | free | ATP_bound | ADP_bound | apo | DNA_bound | oriC_bound | non_oriC_bound
  store_path: bulk
  expansion: ['PD03831[c]', 'MONOMER0-160[c]', 'MONOMER0-4565[c]']
- name: dnaA_pool_ATP_bound
  pool_kind: ATP_bound
  store_path: bulk
  index_by: 'MONOMER0-160[c]'

autorepression_tests:                  # NEW (P1-3): replaces single Pearson
- name: lagged_binding_mrna_cross_correlation
  description: 'Cross-correlation between DnaA-TF binding(t-Δt) and dnaA mRNA(t) at Δt ∈ {0, 30, 60, 120}s; expects negative peak at biologically-plausible Δt.'
- name: conditional_transcription_probability_given_promoter_bound
- name: transcription_burst_rate_by_promoter_state
- name: dnaA_gene_dosage_corrected_expression

perturbation_panel:                  # NEW (P1-5)
- name: dnaA-overexpression
  type: expression_step
  description: 'Double dnaA expression at t=300s; expect autorepression to attenuate transient.'
  expected_response: 'mRNA drops within 60s; protein steady state increases <1.5×'
- name: dnaA-knockdown
  type: expression_step
  description: 'Halve dnaA TE; expect promoter de-repression in <60s.'
- name: autorepression-ablation
  type: parameter_swap
  description: 'Set fc=0; expect uncontrolled DnaA accumulation.'
- name: ATP-cycle-ablation
  type: pathway_swap
  description: 'Disable IntrinsicHydrolysis Step; expect ATP-fraction → 1.0.'
# ... per the feedback list: oriC affinity, titration sites, RIDA, DDAH, DARS, SeqA
```

## Per-study application

| Study | Most critical P1 change |
|---|---|
| **dnaa-01** | F-01 needs full `measurement_mapping` for [300, 800]; F-02 must split out `autorepression_tests[]` and drop Pearson-only PASS claim |
| **dnaa-01f-recalibrate** | Re-classify (TE=20×, fc=0.7) as `regression_compatibility` not biological validation; add perturbation panel |
| **dnaa-01g-joint-te-fold-change-sweep** | Same — the win is regression, NOT biological validation |
| **dnaa-02** | F-04/F-05 (k=4.6/min) re-labeled `regression_compatibility`; add `measurement_mapping` for the Boesen [0.20, 0.50] band including which-pool clarification |
| **dnaa-03** | `DnaA_box` schema extension (P2-1) — region_type, box_class, affinity, nucleotide_preference, cooperative_group, source, expert_review_status |
| **dnaa-04** | Initiation trigger test gets a perturbation panel: dnaA-overexpression, autorepression-ablation, ATP-cycle-ablation |
| **dnaa-05** | RIDA/DDAH/DARS each get mechanism cards (P2-3); hydrolysis-rate becomes "model-implied" until per-pathway biology lands |
| **dnaa-06** | SeqA versioning (P2-2) — three recipes: seqa_v0_fixed_timer, seqa_v1_methylation_state, seqa_v2_cooperative |

## Execution order (this PR + next)

1. **This task** (today): Apply P1-1..P1-5 to dnaa-01 as proof of pattern. Document the schema extensions inline.
2. **Next task**: Roll out P1-1..P1-5 to dnaa-01g, dnaa-02, then dnaa-03..06.
3. **Then**: P2-1 (DnaA-box schema) — drives dnaa-03 implementation.
4. **Then**: P2-2 (SeqA versions) — three recipes in baseline_recipes.py.
5. **Then**: P3 items as polish.

## How this interacts with the recipe chain

Good news — the cascading baseline_recipes.py infrastructure already supports versioning. SeqA v0/v1/v2 become three child recipes of `dnaa_04_with_dnaa_initiation_trigger`. Perturbation panels become loop_patches that flip a single parameter mid-sim.

## How this interacts with the live Visualization Steps

`DnaAStateVisualization` already exposes apo / ATP / ADP separately. P1-4 (explicit pool decomposition) is largely already presented in viz output — just needs the readout-schema metadata to match.

`DnaABoxOccupancyVisualization` partitions by oriC-proximal vs chromosomal. P2-1 (box catalog) will let it partition by `region_type` (oriC / dnaAp / datA / DARS / titration) instead of distance-based heuristic.

## ★ Deeper integration: lift these into bigraph-schema types

These schema additions should NOT just be YAML keys in study.yaml — they should be **first-class bigraph-schema types** registered in `v2ecoli/types/__init__.py:ECOLI_TYPES`. Three immediate wins:

1. **Type-checking at construction.** If `target_class` is a registered enum type, attempting to set an invalid value fails at composite-build time rather than silently slipping through to the evaluator.
2. **Discoverability via the type registry.** The dashboard's Registry tab, the test runner, and the report generator can all introspect the registered types to (e.g.) find every observable tagged `biological_validation`, or list every site whose `region_type == 'oriC'`.
3. **Composability.** A `DnaA_pool_kind` enum referenced from a Listener's output port and from a `behavior_tests.measure.pool_kind` field guarantees the names agree.

### Concrete bigraph-schema additions

Five new types to register in `v2ecoli/types/__init__.py`:

```python
# v2ecoli/types/biology.py  (new file)

TARGET_CLASS = {
    '_type': 'enum',
    '_values': [
        'regression_compatibility',
        'biological_validation',
        'explanatory_gain',
    ],
    '_description': 'How a behavior_test should be interpreted. Heuristic-match cannot count as biological validation. Per dnaa_biology_feedback (2026-05-17).',
}

DNAA_POOL_KIND = {
    '_type': 'enum',
    '_values': [
        'total',           # apo + ATP-bound + ADP-bound + DNA-bound + complexed
        'free',            # apo + ATP + ADP, not DNA-bound
        'ATP_bound',       # MONOMER0-160 only
        'ADP_bound',       # MONOMER0-4565 only
        'apo',             # PD03831, neither nucleotide
        'DNA_bound',       # bound to chromosomal boxes
        'oriC_bound',      # subset of DNA_bound: at oriC sites
        'non_oriC_bound',  # subset of DNA_bound: chromosomal + dnaAp + datA
    ],
    '_description': 'Which DnaA pool a count refers to. dnaA-count-in-range gate cannot fire without explicit pool selection. Per feedback P1-4.',
}

DNAA_BOX_REGION_TYPE = {
    '_type': 'enum',
    '_values': ['oriC', 'dnaAp', 'datA', 'DARS1', 'DARS2', 'chromosomal_titration', 'other'],
    '_description': 'Per feedback P2-1 — oriC / dnaAp / datA / DARS / titration must not be conflated.',
}

DNAA_BOX_AFFINITY_CLASS = {
    '_type': 'enum',
    '_values': ['high', 'medium', 'low'],
    '_description': 'Roth1998 + Hansen 2018 affinity classification. Maps to Kd ~1-5, ~20-100, ~100-500 (DnaA-ATP molecules per cell).',
}

FAILURE_CAUSE = {
    '_type': 'enum',
    '_values': [
        'missing_mechanism',       # the biology required isn\'t in the model yet
        'miscalibration',          # mechanism present, parameter wrong
        'observable_mismatch',     # literature vs model measure different pools/conditions
        'insufficient_statistics', # data exists but not enough samples
    ],
    '_description': 'Per feedback P2-4. Required field on any FAIL outcome.',
}

# Compound type — a measurement_mapping is a struct attached to a behavior_test:
MEASUREMENT_MAPPING = {
    '_type': 'tree[any]',  # bigraph-schema struct
    '_inputs': {
        'literature_observable': {
            '_type': 'tree[any]',
            'name': 'string',
            'biological_pool': 'dnaa_pool_kind',  # ← references the enum above
            'condition': 'string',
            'statistic': 'string',
            'source_ids': 'list[string]',
        },
        'model_observable': {
            '_type': 'tree[any]',
            'listener_path': 'string',
            'state_path': 'string',
            'molecular_pool': 'dnaa_pool_kind',
            'units': 'string',
            'includes_bound_species': 'boolean',
        },
        'transformation': {
            '_type': 'tree[any]',
            'formula': 'string',
            'time_window': 'string',
            'aggregation': 'string',
        },
        'caveats': 'list[string]',
    },
}

# Extend the existing DNAA_BOX_ARRAY (P2-1):
# Current: 'unique_array[coordinates:integer|domain_index:integer|DnaA_bound:boolean|...]'
# Becomes:
DNAA_BOX_ARRAY_RICH = (
    'unique_array[coordinates:integer'
    '|domain_index:integer'
    '|DnaA_bound:boolean'
    '|region_type:dnaa_box_region_type'        # ← NEW
    '|affinity_class:dnaa_box_affinity_class'  # ← NEW
    '|nucleotide_preference:string'            # ← NEW: ATP-only / ATP-preferred / either
    '|cooperative_group:integer'               # ← NEW: index into cooperative-binding cluster
    '|source:string'                            # ← NEW: roth1998 / hansen2018 / parca_pwm / ...
    '|expert_review_status:string'             # ← NEW: confirmed / pending / disputed
    '|_entryState:integer|...]'
)
```

Then register them:
```python
# v2ecoli/types/__init__.py — extend ECOLI_TYPES
from .biology import (
    TARGET_CLASS, DNAA_POOL_KIND, DNAA_BOX_REGION_TYPE,
    DNAA_BOX_AFFINITY_CLASS, FAILURE_CAUSE, MEASUREMENT_MAPPING,
)
ECOLI_TYPES['target_class'] = TARGET_CLASS
ECOLI_TYPES['dnaa_pool_kind'] = DNAA_POOL_KIND
ECOLI_TYPES['dnaa_box_region_type'] = DNAA_BOX_REGION_TYPE
ECOLI_TYPES['dnaa_box_affinity_class'] = DNAA_BOX_AFFINITY_CLASS
ECOLI_TYPES['failure_cause'] = FAILURE_CAUSE
ECOLI_TYPES['measurement_mapping'] = MEASUREMENT_MAPPING
```

### Wins from doing this at the type level

| Capability | Without bigraph-schema integration | With it |
|---|---|---|
| Invalid `target_class` value | Slips through to evaluator → silent miscategorization | Composite-build raises TypeError at registration time |
| Listener output port for DnaA-ATP count | Just a number | A `{value: int, pool_kind: 'ATP_bound', units: 'molecules/cell'}` struct that the test runner unwraps with full schema knowledge |
| DnaA-box query "all oriC boxes" | Have to read coordinates and infer | `boxes[where region_type == 'oriC']` — type-aware indexing |
| Test runner classifying a FAIL | Free-text in the report | `failure_cause` enum on the result; aggregator groups failures by cause |
| `DnaABoxOccupancyVisualization` color-by-class | Hard-coded distance heuristic | Reads `region_type` directly from the unique array; works for any box catalog |
| Cross-study consistency | Each study.yaml could use slightly different pool names | Single registered type → mismatches fail fast |

### Migration path

1. **Now (P1)**: Add the enums + MEASUREMENT_MAPPING type to `v2ecoli/types/biology.py`. Register in ECOLI_TYPES. study.yamls reference them by name (string values still readable to humans).
2. **Next (P2-1)**: Extend `DNAA_BOX_ARRAY` to `DNAA_BOX_ARRAY_RICH`. ParCa's initial-conditions populates `region_type` and `affinity_class` from the curated catalog. Existing dnaa-03 probe + viz Step start using these attributes.
3. **Then (P2-2 / P2-3)**: Mechanism cards become structured types too — `SeqA_version: enum[v0_fixed_timer, v1_methylation_state, v2_cooperative]` and similar for RIDA/DDAH/DARS.
4. **Then**: The test runner introspects type-registered metadata to render the report's per-class breakdown automatically (target_class buckets) and to surface failure_cause histograms.

## What changes in this PR vs follow-ups

- **This PR (current)**: register `dnaa_biology_feedback` expert doc, write this plan, apply P1-1..P1-5 to dnaa-01 as proof of pattern (with bigraph-schema type usage from the start).
- **Next PR**: roll out P1 to dnaa-01g, dnaa-02, dnaa-03..06; create `v2ecoli/types/biology.py` if not done here.
- **Then**: P2 items as separate PRs (DnaA-box schema extension is its own substantial implementation).
