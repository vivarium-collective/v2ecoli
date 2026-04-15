"""
v2ecoli.processes.parca — process-bigraph migration of the E. coli Parameter Calculator.

Layout
======

    v2ecoli.processes.parca/
        composite.py          build_parca_composite / run_parca / register_parca_steps
        state.py              ParcaState bigraph-schema type + register_parca_types
        types.py              Input/Output dataclasses for every stage
        fitting.py            pure math + sim_data-reading fitting helpers
        promoter_fitting.py   matrix builders + CVXPY optimization (stages 6/7)
        steps/
            __init__.py       ALL_STEP_CLASSES registry
            step_01_initialize.py
            step_02_input_adjustments.py      (Extract → Compute → Merge, PURE)
            step_03_basal_specs.py            (COUPLED)
            step_04_tf_condition_specs.py     (COUPLED)
            step_05_fit_condition.py          (READ-ONLY)
            step_06_promoter_binding.py       (COUPLED)
            step_07_adjust_promoters.py       (COUPLED)
            step_08_set_conditions.py         (Extract → Compute → Merge, PURE)
            step_09_final_adjustments.py      (COUPLED)

Each stage module (2–9) provides:

    extract_input(sim_data, cell_specs, **kwargs) -> StageInput
    compute_<name>(inp: StageInput)              -> StageOutput
    merge_output(sim_data, cell_specs, out)      -> None
    <StageName>Step(Step)                         — process-bigraph wrapper

Pure stages (2, 8) additionally expose ExtractFor... and MergeAfter...
Step classes so the compute Step has only explicit typed ports.

Purity legend
=============
  PURE       compute has no sim_data/cell_specs access
  READ-ONLY  compute reads sim_data via ref but does not mutate it
  COUPLED    compute still mutates sim_data via ref (future refactor target)
"""
