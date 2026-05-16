"""
Process-bigraph Step classes for the ParCa pipeline — one per step.

Each step module exposes its Step class and any pure helpers.  Every Step
declares its ports at the **subsystem-object / top-level-dict** level: one
port per sim_data subsystem it reads/writes, plus one port per top-level
pure-data dict.  The composite wires each port to its canonical path in a
nested bigraph store that mirrors sim_data's own structure.

``ALL_STEP_CLASSES`` is the flat registry for
``process_bigraph.allocate_core(top=...)``.
"""

from v2ecoli.processes.parca.steps.step_01_initialize            import InitializeStep
from v2ecoli.processes.parca.steps.step_02_input_adjustments     import InputAdjustmentsStep
from v2ecoli.processes.parca.steps.step_03_basal_specs           import BasalSpecsStep
from v2ecoli.processes.parca.steps.step_04_tf_condition_specs    import TfConditionSpecsStep
from v2ecoli.processes.parca.steps.step_05_fit_condition         import FitConditionStep
from v2ecoli.processes.parca.steps.step_06_promoter_binding      import PromoterBindingStep
from v2ecoli.processes.parca.steps.step_07_adjust_promoters      import AdjustPromotersStep
from v2ecoli.processes.parca.steps.step_08_set_conditions        import SetConditionsStep
from v2ecoli.processes.parca.steps.step_09_final_adjustments     import FinalAdjustmentsStep


ALL_STEP_CLASSES = {
    'InitializeStep':       InitializeStep,
    'InputAdjustmentsStep': InputAdjustmentsStep,
    'BasalSpecsStep':       BasalSpecsStep,
    'TfConditionSpecsStep': TfConditionSpecsStep,
    'FitConditionStep':     FitConditionStep,
    'PromoterBindingStep':  PromoterBindingStep,
    'AdjustPromotersStep':  AdjustPromotersStep,
    'SetConditionsStep':    SetConditionsStep,
    'FinalAdjustmentsStep': FinalAdjustmentsStep,
}


__all__ = [
    'ALL_STEP_CLASSES',
    'InitializeStep', 'InputAdjustmentsStep', 'BasalSpecsStep',
    'TfConditionSpecsStep', 'FitConditionStep', 'PromoterBindingStep',
    'AdjustPromotersStep', 'SetConditionsStep', 'FinalAdjustmentsStep',
]
