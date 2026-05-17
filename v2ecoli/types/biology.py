"""Bigraph-schema types for the study-evaluation framework.

A small registry of enums that any v2ecoli study can reuse to declare
how its behavior_tests should be interpreted, how its conclusions
break down across target classes, and how its outcomes get classified.
These are first-class bigraph-schema types so:

  1. Listener output ports + ``behavior_tests`` blocks can declare
     ``target_class``, ``failure_cause``, etc. and have the registry
     reject invalid values at composite-build time.
  2. The dashboard, test runner, and viz Steps can introspect the
     registry to filter / group by these categories (e.g., "show all
     biological_validation tests across the workspace").
  3. Cross-study consistency: every place that talks about a verdict
     result or failure cause uses the same enum members; a workspace-
     level validator can catch divergence early
     (``scripts/validate_study_biology.py``).

All types are registered into ``ECOLI_TYPES`` in
``v2ecoli/types/__init__.py`` and become available as type strings:
``target_class``, ``verdict_result``, ``failure_cause``, etc.

The shape is deliberately enum-light: each entry is a registered
bigraph-schema ``enum`` type. Domain-specific enums (e.g. the
DnaA-pool-kind catalog, the autorepression-test-kind catalog) are
left for the per-investigation modules to add on top — this file
ships only the framework primitives.
"""
from __future__ import annotations


# ─── verdict_result: per-target-class outcome in conclusion_verdicts ────

VERDICT_RESULT = {
    '_type': 'enum',
    '_values': [
        'PASS',            # mechanism / target satisfied
        'FAIL',            # mechanism / target violated
        'PENDING',         # required evidence not yet collected
        'NOT_APPLICABLE',  # this class doesn't apply to this study
        'NOT_CLAIMED',     # study deliberately doesn't make this claim
    ],
    '_description': (
        'Outcome for one of the three target_class verdicts in '
        "conclusion_verdicts. A study's final verdict should report "
        'regression_compatibility, biological_validation, and '
        'explanatory_gain INDEPENDENTLY — not collapsed to a single '
        'pass/fail.'
    ),
}


# ─── narrative_confidence: textbook-consensus rating on mechanism cards ─

NARRATIVE_CONFIDENCE = {
    '_type': 'enum',
    '_values': ['strong', 'moderate', 'contested', 'open'],
    '_description': (
        'Rating attached to mechanism_cards (or other narrative claims) '
        'so the framework can distinguish "settled textbook" from '
        '"actively contested" assertions and flag claims built on '
        'contested narrative.'
    ),
}


# ─── target_class: how to interpret a behavior_test result ──────────────

TARGET_CLASS = {
    '_type': 'enum',
    '_values': [
        # Test passes when the model matches a heuristic / regression target.
        # Does NOT validate the underlying biology — it just says the model
        # reproduces the result we already had (or that the literature is
        # consistent at a coarse level).
        'regression_compatibility',

        # Test passes ONLY when the mechanism is genuinely doing the right
        # biology. Requires perturbation panel evidence (wild-type baseline
        # alone is insufficient).
        'biological_validation',

        # Test passes when the model EXPLAINS something previously unexplained
        # (a quantitative relationship, an emergent timing, a robustness
        # property). Highest bar — requires both validation and a counterfactual.
        'explanatory_gain',
    ],
    '_description': (
        'How a behavior_test should be interpreted. Heuristic regression '
        'cannot count as biological validation; biological validation '
        'cannot count as explanatory gain. Studies must declare which '
        'bar each test is reaching for.'
    ),
}


# ─── failure_cause: classify a FAIL outcome ─────────────────────────────

FAILURE_CAUSE = {
    '_type': 'enum',
    '_values': [
        'missing_mechanism',         # biology required isn't in the model yet
        'miscalibration',            # mechanism present, parameter wrong
        'observable_mismatch',       # literature and model measure different things
        'insufficient_statistics',   # data exists but not enough samples
    ],
    '_description': (
        'Every FAIL outcome should declare the likely cause. Drives '
        'downstream triage: a miscalibration is solvable in place; a '
        'missing mechanism opens a new study; an observable mismatch '
        'asks for a measurement_mapping fix.'
    ),
}


# ─── autorepression_test_result: PASS/FAIL/INSUFFICIENT_EVIDENCE shape ──
# Generic result tier for tests where the underlying statistic may be
# uncomputable for reasons unrelated to PASS/FAIL (too few samples,
# zero-variance signal, etc.). Originally introduced for the
# autorepression test suite; reusable by any sample-bounded test.

AUTOREPRESSION_TEST_RESULT = {
    '_type': 'enum',
    '_values': ['PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE', 'NOT_RUN'],
    '_description': (
        'Outcome of a sample-bounded behavior_test. Distinguishes '
        '"the data exists and the test failed" (FAIL) from "the data is '
        'too sparse or degenerate for the test to be meaningful" '
        '(INSUFFICIENT_EVIDENCE). The evaluator must emit '
        'INSUFFICIENT_EVIDENCE rather than a false PASS or FAIL when '
        'sample count is below a per-test threshold or when the input '
        'signal has zero variance.'
    ),
}


# ─── perturbation panel: required for biological_validation ─────────────

PERTURBATION_KIND = {
    '_type': 'enum',
    '_values': [
        'expression_step',     # toggle gene expression up/down mid-sim
        'parameter_swap',      # change a parameter at t=0
        'pathway_swap',        # disable a whole Step
        'site_mutation',       # remove a binding / regulatory site
        'media_shift',         # change growth condition
        'inducible_promoter',  # toggle expression at a specific time
    ],
    '_description': (
        'Categories of perturbation a study may run to discharge a '
        'biological_validation claim. Wild-type baseline alone cannot '
        'close validation — the study must run at least one perturbation '
        'and report the prediction-vs-observation delta.'
    ),
}


# ─── measurement_mapping: literature → model observable correspondence ──
# Compound dict-shaped block; documented here for reference but NOT a
# bigraph-schema-registered type (bigraph-schema's tree[any] parser doesn't
# accept the nested-fields form cleanly). Validated at the study-loader
# layer instead (see scripts/validate_study_biology.py). The validator
# checks: literature_observable + model_observable blocks both exist;
# biological_pool / molecular_pool reference a registered pool-kind enum
# if the workspace has one; source_ids non-empty.
#
# Shape (use this in study.yaml):
#   measurement_mapping:
#     literature_observable:
#       name:            <str>
#       biological_pool: <enum value, workspace-specific>
#       condition:       <str>
#       statistic:       <str>
#       source_ids:      [<bib_key>, ...]
#     model_observable:
#       listener_path:         <str>
#       state_path:            <str>
#       molecular_pool:        <enum value, workspace-specific>
#       units:                 <str>
#       includes_bound_species:<bool>
#     transformation:
#       formula:     <str>
#       time_window: <str>
#       aggregation: <str>
#     caveats: [<str>, ...]
MEASUREMENT_MAPPING_SHAPE = {
    'description': (
        'Documented YAML shape for behavior_tests[].measurement_mapping. '
        'Every numeric literature threshold should carry a '
        'measurement_mapping showing how the literature quantity maps to '
        'a concrete model quantity.'
    ),
    'required_top_level_keys': ['literature_observable', 'model_observable',
                                'transformation'],
    'pool_kind_fields': ['literature_observable.biological_pool',
                         'model_observable.molecular_pool'],
}


# ─── Public registry ────────────────────────────────────────────────────

BIOLOGY_TYPES = {
    # All enum types — registered as first-class bigraph-schema types.
    # Listener output ports / behavior_tests can reference these by name
    # ('target_class', 'verdict_result', ...); invalid values are rejected
    # at composite-build time.
    'target_class':                  TARGET_CLASS,
    'verdict_result':                VERDICT_RESULT,
    'narrative_confidence':          NARRATIVE_CONFIDENCE,
    'failure_cause':                 FAILURE_CAUSE,
    'autorepression_test_result':    AUTOREPRESSION_TEST_RESULT,
    'perturbation_kind':             PERTURBATION_KIND,
}

# Compound shapes that aren't registered as bigraph types but are validated
# at the study-loader layer:
COMPOUND_SHAPES = {
    'measurement_mapping': MEASUREMENT_MAPPING_SHAPE,
}


# ═══════════════════════════════════════════════════════════════════════
# Per-investigation extension: dnaA replication-initiation studies
# ═══════════════════════════════════════════════════════════════════════
# Domain-specific enums for the dnaA investigation. These ride on top of
# the framework primitives above. Source: external biology review
# (2026-05-17), registered as expert doc ``dnaa_biology_feedback`` in
# workspace.yaml. The reviewer's P1 items mandate explicit DnaA pool
# decomposition, DnaA-box catalog enrichments, autorepression test suite
# kinds, SeqA version tracking, and reset-mechanism classification.


# ─── dnaa_pool_kind: which DnaA pool a count or fraction refers to ──────

DNAA_POOL_KIND = {
    '_type': 'enum',
    '_values': [
        'total',           # apo + ATP-bound + ADP-bound + DNA-bound + complexed
        'free',            # apo + ATP + ADP, but NOT DNA-bound
        'apo',             # PD03831 — neither nucleotide bound
        'ATP_bound',       # MONOMER0-160 — DnaA-ATP complex
        'ADP_bound',       # MONOMER0-4565 — DnaA-ADP complex
        'DNA_bound',       # bound at any DnaA-box site (sum across regions)
        'oriC_bound',      # subset of DNA_bound: at oriC sites (high-affinity)
        'non_oriC_bound',  # subset of DNA_bound: chromosomal titration + dnaAp + datA
    ],
    '_description': (
        'Which DnaA pool a count or fraction refers to. Per feedback P1-4: '
        'dnaA-count-in-range pass/fail is blocked if pool mapping is unknown '
        'or expert-disputed. Used by Listener output ports, behavior_test '
        'readouts, and measurement_mapping blocks.'
    ),
}


# ─── DnaA-box catalog enrichments ───────────────────────────────────────

DNAA_BOX_REGION_TYPE = {
    '_type': 'enum',
    '_values': [
        'oriC',                    # the 11 canonical sites at the origin
        'dnaAp',                   # 4 dnaA-promoter sites (auto-regulation)
        'datA',                    # DDAH GATC cluster (post-initiation hydrolysis)
        'DARS1',                   # first DnaA-reactivating site
        'DARS2',                   # second DnaA-reactivating site
        'chromosomal_titration',   # generic high-low affinity sites across genome
        'other',                   # uncurated or unknown function
    ],
    '_description': (
        'Functional class of a DnaA-box site. Per feedback P2-1: oriC, dnaAp, '
        'datA, DARS, and titration boxes MUST be distinguishable in analysis '
        'and plots. Default region_type for unknown sites is "other".'
    ),
}

DNAA_BOX_AFFINITY_CLASS = {
    '_type': 'enum',
    '_values': ['high', 'medium', 'low'],
    '_description': (
        'Affinity classification per Roth1998 + Hansen 2018. Maps to '
        'approximate Kd ranges: high ~1-5 molecules/cell, medium ~20-100, '
        'low ~100-500 (in DnaA-ATP-molecule-equivalent units).'
    ),
}

DNAA_BOX_NUCLEOTIDE_PREFERENCE = {
    '_type': 'enum',
    '_values': [
        'ATP_only',       # DnaA-ATP binds; DnaA-ADP does not (oriC-type)
        'ATP_preferred',  # binds both but ATP more strongly
        'either',         # nucleotide-independent binding
        'ADP_preferred',  # rare; some literature suggests dnaAp ADP preference
    ],
    '_description': (
        'Which DnaA nucleotide form a site preferentially binds. Required '
        'for dnaa-03 cooperative-binding model: oriC sites are ATP_only, '
        'chromosomal_titration is mostly either.'
    ),
}


# ─── autorepression test suite (replaces single Pearson r) ──────────────

AUTOREPRESSION_TEST_KIND = {
    '_type': 'enum',
    '_values': [
        'lagged_binding_mrna_cross_correlation',
        'conditional_transcription_probability_given_promoter_bound',
        'transcription_burst_rate_by_promoter_state',
        'dnaA_gene_dosage_corrected_expression',
        'dnaAp1_response',                 # if dnaAp1/p2 are resolved separately
        'dnaAp2_response',
    ],
    '_description': (
        'Per feedback P1-3: replace Pearson-only autorepression testing with '
        'this suite. Each test reports PASS / FAIL / INSUFFICIENT_EVIDENCE.'
    ),
}


# ─── SeqA + reset-mechanism versioning ──────────────────────────────────

SEQA_VERSION = {
    '_type': 'enum',
    '_values': [
        'seqa_v0_fixed_timer',
        'seqa_v1_methylation_state',
        'seqa_v2_cooperative_seqa_dam',
    ],
    '_description': (
        'Per feedback P2-2: claims about SeqA are LIMITED BY the implemented '
        'abstraction version. v0 = fixed 10-min timer; v1 = methylation-state '
        'machine + Dam re-methylation kinetics; v2 = cooperative SeqA-Dam.'
    ),
}

RESET_MECHANISM_KIND = {
    '_type': 'enum',
    '_values': ['RIDA', 'DDAH', 'DARS1', 'DARS2'],
    '_description': (
        'Per feedback P2-3: RIDA / DDAH / DARS are reset mechanisms, NOT '
        'generic hydrolysis knobs. Each gets a mechanism card with biological '
        'role, model abstraction, included/omitted behaviors, perturbations.'
    ),
}


# Extend the framework registry with the dnaA-specific enums.
BIOLOGY_TYPES.update({
    'dnaa_pool_kind':                 DNAA_POOL_KIND,
    'dnaa_box_region_type':           DNAA_BOX_REGION_TYPE,
    'dnaa_box_affinity_class':        DNAA_BOX_AFFINITY_CLASS,
    'dnaa_box_nucleotide_preference': DNAA_BOX_NUCLEOTIDE_PREFERENCE,
    'autorepression_test_kind':       AUTOREPRESSION_TEST_KIND,
    'seqa_version':                   SEQA_VERSION,
    'reset_mechanism_kind':           RESET_MECHANISM_KIND,
})
