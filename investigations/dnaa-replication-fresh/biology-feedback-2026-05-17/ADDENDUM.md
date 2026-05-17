# Addendum: items missed in the first PLAN pass

The first pass at the biology-feedback (`PLAN.md`) captured the P1/P2/P3
items at face value but missed several SUBSTANTIVE insights from the
review's section 3 acceptance criteria + section 5 desired-behavior
checklist. Documenting them here as P1b / P2b / P3b that extend the
matrix in `PLAN.md`.

## P1b — must fix (architectural)

### P1b-1 ★ Active-initiation-signal constraint (dnaa-04 architectural change)

> Section 3, DnaA-ATP decomposition acceptance: **"Initiation studies cannot
> use total DnaA-ATP alone as the active initiation signal."**

This is the deepest specific architectural finding in the entire review.
The dnaa-04 implementation must read **`DnaA_box.DnaA_bound AND
region_type == 'oriC'`** as the initiation trigger, not the total
`bulk[MONOMER0-160]` count. My drafted swap-point design at
`chromosome_replication.py:244` correctly mentions "n_DnaA_bound_at_oriC"
but the **fallback** text says "use bulk[MONOMER0-160] count as a proxy"
— that's explicitly forbidden. Remove the fallback wording.

Required changes:
- `studies/dnaa-04-initiation-mechanism/study.yaml` — strip "fallback to
  total DnaA-ATP" from F-01 and from `gate_status_summary`. Make it clear
  the trigger READS `oriC_bound` exclusively. Without dnaa-03's binding
  mechanism + region_type catalog, dnaa-04 is **blocked**, not "ready
  with degraded fallback."
- `gate_status` for dnaa-04 changes from `ready` (cascaded via recipe
  chain) back to `blocked` (on dnaa-03's cooperative binding + box
  region_type catalog).
- `v2ecoli/composites/baseline_recipes.py` —
  `dnaa_04_with_dnaa_initiation_trigger` recipe must depend on the
  dnaa-03 box catalog being populated, not just inherit from the
  hydrolysis-bearing parent.

### P1b-2 ★ Triple verdict in `conclusion_logic` (not collapsed)

> Section 3, Target class split acceptance: **"A heuristic-match test
> cannot count as biological validation. Final decisions show all three
> outcomes independently."**

`conclusion_logic` currently emits one verdict. It must emit three:

```yaml
conclusion_verdicts:
  regression_compatibility:
    result: PASS                  # PASS | FAIL | PENDING | NOT_APPLICABLE
    basis: '(TE=20×, fc=0.7) matches Schmidt 2016 count'
    contributing_tests: [dnaA-count-in-range, autorepression-correlation]
  biological_validation:
    result: PENDING
    basis: 'no perturbation panel runs yet'
    blocking_tests: [dnaA-overexpression, autorepression-ablation]
  explanatory_gain:
    result: NOT_CLAIMED
    basis: 'study is calibration, not mechanism'
```

The dashboard's `gate_status` badge should split into 3 sub-badges per
study. Currently one `🟢 open` collapses all three classes — misleading.

### P1b-3 ★ `model_implied: true` tag on calibration-fit findings

> Section 3, Reset mechanism cards acceptance: **"Hydrolysis-rate
> requirements are labeled model-implied unless supported by direct
> biology."**

Generalize: any finding whose evidence is "the model matches literature
because we fit a parameter to make it match" must be tagged
`model_implied: true`. The tag prevents these from being read as
measured biology.

Findings to re-tag:

| Finding | Add tag | Clarifying note |
|---|---|---|
| dnaa-01g recipe `dnaa_01g_calibrated` | `model_implied: true` | "TE=20×, fc=0.7 fit to dnaa-01 count target; not measured" |
| dnaa-02 F-04 (k=4.6/min) | `model_implied: true` | "Compensates for v2ecoli's equilibrium reverse-rate; NOT an in-vivo measured rate" |
| dnaa-02 F-05 (recipe-chain validation) | partially `model_implied` | "Mechanism architecture validated; specific rate is a fit" |
| dnaa-03 F-05 (Hill n=5-10 prediction) | `model_implied: true` | "Predicted to reproduce two-step pattern; alternatives: HU/IHF/Fis accessory factors, SeqA sequestration" |

### P1b-4 INSUFFICIENT_EVIDENCE enforcement

> Section 3, Autorepression suite acceptance: **"Sparse mRNA data can
> return insufficient evidence rather than false pass/fail."**

`AUTOREPRESSION_TEST_RESULT` enum has the `INSUFFICIENT_EVIDENCE` value
but `tests/_behaviors.py` doesn't return it. The current behavior is to
return `r=N/A` and call it FAIL or to silently produce a pearson r with
N=2 samples.

Required changes:
- `tests/_behaviors.py` pearson-correlation evaluator emits
  `INSUFFICIENT_EVIDENCE` when `len(samples) < 30` OR when the input
  signal has zero variance.
- `tests/_behaviors.py` adds a generic `_insufficient(reason)` helper
  used by every measure kind that can produce one.
- Outcome aggregation in study.yaml `runs[].outcomes` accepts
  `INSUFFICIENT_EVIDENCE` and propagates it to `conclusion_verdicts`
  (NOT a fail, NOT a pass).

## P2b — schema extensions

### P2b-5 `decision_log[]` versioning

> Section 3, Expert questions acceptance: **"Expert input changes
> thresholds, mappings, or mechanism cards in a versioned way."**

Append-only log entries on each `pass_if` threshold, each
`measurement_mapping`, each `mechanism_card`:

```yaml
behavior_tests:
- name: dnaA-count-in-range
  pass_if: {op: in_range, low: 300, high: 800}
  decision_log:
  - timestamp: 2026-05-01T00:00Z
    actor: 'initial setup (auto)'
    field_changed: pass_if.high
    old_value: 1000
    new_value: 800
    rationale: 'Schmidt 2016 updated mass-spec upper bound'
  - timestamp: 2026-05-17T11:40Z
    actor: 'biology review (external)'
    field_changed: measurement_mapping.literature_observable.biological_pool
    old_value: '(implicit: total)'
    new_value: total                  # now explicit
    rationale: 'feedback P1-2 — pool must be explicit before threshold can be used'
```

### P2b-6 `DnaAStateVisualization` extended to 8 pools

> Section 3, DnaA-ATP decomposition specification: **"Report total, free,
> DNA-bound, oriC-bound, non-oriC-bound, ADP-bound, and apo-DnaA pools."**

The viz Step currently emits 3 pools + total. Missing splits:
- `free` vs `DNA-bound` (requires dnaa-03 binding + the
  `DnaA_box.DnaA_bound` state being populated)
- `oriC-bound` vs `non-oriC-bound` (requires the `region_type` attribute
  from P2-1 box-catalog extension)

The Step should be extended NOW with placeholder series that draw zero
until the supporting mechanism lands. Then when dnaa-03 wires in box
binding, the viz starts showing real data without further Step changes.

### P2b-7 Cross-study `biological_pool` consistency validator — **LANDED**

When two studies reference the same biological measurement but with
different `biological_pool` values, that's a quiet conflict the
aggregator should flag.

**Implementation** (commit on PR #56): `scripts/validate_study_biology.py`.
Walks every `studies/*/study.yaml` and:

1. Validates enum-typed fields against `BIOLOGY_TYPES` (the registered
   bigraph-schema enums in `v2ecoli/types/biology.py`):
   `target_class`, `biological_pool`, `molecular_pool`, `failure_cause`,
   `region_type`, `affinity_class`, `nucleotide_preference`, `result`
   (verdict_result / autorepression_test_result by context),
   `narrative_confidence`, `autorepression_test_kind`,
   `perturbation_kind`, `seqa_version`, `reset_mechanism_kind`.
   Contextual rules use strict shape matching (`list_item` vs
   `dict_value`) so e.g. `autorepression_test_suite[i].measure.kind`
   is NOT wrongly checked against the autorepression-test-kind enum.
2. Cross-study consistency: same `literature_observable.name` must map
   to the same `biological_pool` across studies (ERROR if not); same
   `model_observable.state_path` must map to the same `molecular_pool`
   (ERROR if not); same `literature_observable.name` cited with
   completely disjoint `source_ids` across studies emits a WARNING.

Hooked into `scripts/lint-workspace.py` — failed enum or cross-study
checks fail the workspace lint. Standalone usage:
`python scripts/validate_study_biology.py [--json] [--strict]`.

## P3b — process / reporting

### P3b-8 `abstraction_level:` field per study and per test

> Section 5 desired behavior: "what mechanism is being tested, **what
> abstraction represents it**, ..."

Each study declares its abstraction level explicitly so claims at
abstraction N don't get extrapolated to N+1 inappropriately:

```yaml
abstraction_level:
  name: 'lumped total-DnaA pool, quasi-steady-state'
  resolves:
  - 'total DnaA homeostasis'
  - 'autorepression as a correlation signal'
  does_not_resolve:
  - 'per-form (ATP/ADP/apo) dynamics — that\'s dnaa-02'
  - 'spatial DnaA distribution — that\'s dnaa-03+'
  - 'cell-cycle timing — that\'s dnaa-04+'
```

Per-study abstraction-level declarations:

- dnaa-01: "lumped total-DnaA pool, quasi-steady-state"
- dnaa-02: "three-state nucleotide cycle (apo/ATP/ADP), well-mixed cytoplasm"
- dnaa-03: "DnaA-box occupancy with per-class affinity, no cooperativity (v1)"
- dnaa-04: "DnaA-occupancy-triggered initiation, mass-threshold removed"
- dnaa-05: "extrinsic conversion via RIDA + DDAH + DARS"
- dnaa-06 v0: "SeqA as fixed 10-min post-initiation timer"
- dnaa-06 v1: "Dam re-methylation kinetics + SeqA"
- dnaa-06 v2: "cooperative SeqA-Dam dynamics"

### P3b-9 `narrative_confidence:` flag on mechanism cards

> Section 1 framing: **"must not treat a staged textbook narrative as
> more settled than the current evidence warrants."**

Each mechanism card carries a textbook-consensus rating + the specific
sub-claims that ARE contested:

```yaml
mechanism_cards:
- name: dnaA_autorepression
  textbook_consensus_strength: strong          # strong | moderate | contested | open
  evidence_for_strength:
  - {ref: Speck1999EMBO, evidence_type: in_vitro_binding}
  - {ref: Saggioro2013BiochemJ, evidence_type: in_vivo_dosage_correction}
  - {ref: Katayama2017Frontiers, evidence_type: review}
  contested_aspects:
  - aspect: 'Hill cooperativity coefficient for dnaAp1/dnaAp2 binding'
    range_in_literature: '[1, 4]'
    review_note: 'current evidence does not warrant claiming a specific value'
  - aspect: 'Relative strength of dnaAp1 vs dnaAp2 binding'
    review_note: 'dnaAp2 dominant in some studies, comparable in others'
```

### P3b-10 Structured `expert_decisions_needed[]` block

> Section 5: "...what expert decision is needed."
> Section 2 P3: "Convert uncertainties into answerable questions with
> alternatives, impact if wrong, blocking status, requested response type."

Already mentioned in PLAN.md P3-3 but not given a structured shape:

```yaml
expert_decisions_needed:
- id: dnaa-01-EQ-01
  question: 'Does Schmidt 2016 mass-spec 300-800 cellular DnaA include the chromosomally-bound DnaA population?'
  alternatives: [yes, no, partially]
  impact_if_wrong: 'The current 4.8× count gap (115 vs 300-800) may be an observable mismatch (different pool) rather than a miscalibration. If yes, dnaA-01 fails biological_validation for the wrong reason; if no, the gap is real and dnaa-01g calibration is justified.'
  blocks:
  - dnaa-01 conclusion_verdicts.biological_validation
  - dnaa-02 calibration target derivation
  requested_response: 'one paragraph from a mass-spec expert + DOI to extraction-protocol section of the 2016 paper'
  status: open
  asked_to: TBD
  asked_at: TBD
  answer: null
```

## Cross-cutting: updates to the bigraph-schema types

These additions imply two enum extensions in `v2ecoli/types/biology.py`:

```python
VERDICT_RESULT = {
    '_type': 'enum',
    '_values': ['PASS', 'FAIL', 'PENDING', 'NOT_APPLICABLE', 'NOT_CLAIMED'],
}

NARRATIVE_CONFIDENCE = {
    '_type': 'enum',
    '_values': ['strong', 'moderate', 'contested', 'open'],
}
```

(These can ship in the same PR as the per-study application of P1b-3
`model_implied` tags.)

## Updated execution order

1. **Now (this iteration)**: Apply P1b-1 / P1b-2 / P1b-3 to dnaa-01,
   dnaa-02, dnaa-04, dnaa-01g (the studies with actual outcomes to
   re-classify). Add VERDICT_RESULT and NARRATIVE_CONFIDENCE enums.
2. **Next iteration**: Roll out P1b-1..P1b-3 to remaining studies (-03, -05, -06).
3. **Then**: P1b-4 INSUFFICIENT_EVIDENCE enforcement in tests/_behaviors.py.
4. **Then**: P2b-5 decision_log[] versioning across all behavior_tests.
5. **Then**: P2b-6 viz Step extension to 8 pools (requires dnaa-03 binding to land for real data).
6. **Then**: P2b-7 cross-study consistency validator.
7. **Then**: P3b items as polish.

## What changes in the dashboard

When the triple-verdict (P1b-2) lands, the dashboard's gate-status badge
on each study card should split into three smaller badges:
`reg ✓ · bio ⏸ · expl —`. The DAG SVG in `viz/08_investigation_dag.svg`
should color each node's three sub-strips independently. Cleaner signal
than the current single-emoji collapse.
