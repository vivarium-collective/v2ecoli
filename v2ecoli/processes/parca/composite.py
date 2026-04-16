"""
Composite builder for the ParCa pipeline.

Port-first, nested-store design.  The composite's state is a bigraph tree
that mirrors ``SimulationDataEcoli``'s own structure: subsystem objects
(``Transcription``, ``Mass``, ``Constants``, …) sit at natural paths like
``process/transcription`` or ``constants``, and pure-data top-level dicts
(``tf_to_active_inactive_conditions``, ``conditions``, …) sit at sibling
paths.  Each Step's ports wire to the subsystems and leaves it actually
touches — nothing travels through a monolithic ``sim_data`` blob.

**This PoC composite** runs Step 1 (scatter) and Step 2 (input_adjustments).
Steps 3–9 will be wired in after they're converted to the object/leaf
port style.
"""

from process_bigraph import Composite, allocate_core

from v2ecoli.processes.parca.schema import register_parca_schema
from v2ecoli.processes.parca.steps import ALL_STEP_CLASSES


# ---------------------------------------------------------------------------
# Port-to-store-path table — single source of truth for every wire in the
# composite.  Port names are kept globally unique across Steps so this one
# table suffices for all wiring.
# ---------------------------------------------------------------------------

# Sequencing tokens that enforce serial ordering across Steps 1-9.  Each
# Step writes ``tick_N`` on output and reads ``tick_{N-1}`` on input; the
# resulting dependency graph serializes Steps 1 -> 2 -> ... -> 9 even when
# their data-level port surfaces overlap.
_TICKS = [f'tick_{n}' for n in range(10)]  # tick_0 .. tick_9
_TICK_PATHS = {name: [name] for name in _TICKS}


STORE_PATH = {}
STORE_PATH.update(_TICK_PATHS)
STORE_PATH.update({
    # subsystem object leaves
    'transcription':                    ['process', 'transcription'],
    'translation':                      ['process', 'translation'],
    'metabolism':                       ['process', 'metabolism'],
    'rna_decay':                        ['process', 'rna_decay'],
    'complexation':                     ['process', 'complexation'],
    'equilibrium':                      ['process', 'equilibrium'],
    'two_component_system':             ['process', 'two_component_system'],
    'transcription_regulation':         ['process', 'transcription_regulation'],
    'replication':                      ['process', 'replication'],
    'mass':                             ['mass'],
    'constants':                        ['constants'],
    'growth_rate_parameters':           ['growth_rate_parameters'],
    'adjustments':                      ['adjustments'],
    'molecule_groups':                  ['molecule_groups'],
    'molecule_ids':                     ['molecule_ids'],
    'relation':                         ['relation'],
    'getter':                           ['getter'],
    'bulk_molecules':                   ['internal_state', 'bulk_molecules'],

    # pure-data top-level dicts
    'tf_to_active_inactive_conditions': ['tf_to_active_inactive_conditions'],
    'conditions':                       ['conditions'],
    'condition_to_doubling_time':       ['condition_to_doubling_time'],
    'tf_to_fold_change':                ['tf_to_fold_change'],
    'tf_to_direction':                  ['tf_to_direction'],
    'condition_active_tfs':             ['condition_active_tfs'],
    'condition_inactive_tfs':           ['condition_inactive_tfs'],
    'cell_specs':                       ['cell_specs'],
    'translation_supply_rate':          ['translation_supply_rate'],
    'expected_dry_mass_increase_dict':  ['expected_dry_mass_increase_dict'],
    'pPromoterBound':                   ['pPromoterBound'],
    'external_state':                   ['external_state'],
    'condition':                        ['condition'],
    'sim_data_root':                    ['sim_data_root'],
})


def _wires(port_names):
    """Produce a composite ``wires`` dict for the given ports."""
    return {name: STORE_PATH[name] for name in port_names}


def build_parca_composite(raw_data, debug=False, cpus=1,
                          cache_dir='', core=None,
                          variable_elongation_transcription=True,
                          variable_elongation_translation=False,
                          disable_ribosome_capacity_fitting=False,
                          disable_rnapoly_capacity_fitting=False,
                          resume_from_step=1, resume_state=None):
    """Build a Composite that runs the 9-step ParCa pipeline.

    Args:
        raw_data: a ``KnowledgeBaseEcoli`` instance.  Passed through
            InitializeStep's config to keep bigraph-schema from walking
            its nested KB internals at composite construction time.
            Required when ``resume_from_step <= 1``; ignored otherwise.
        debug, cpus, cache_dir, *_elongation*, *_capacity_fitting:
            forwarded to the relevant Step configs.
        core: optional pre-built core.
        resume_from_step: 1-9.  When > 1, steps 1..(N-1) are skipped and
            the composite's initial store state is seeded from
            ``resume_state`` (which must be a dict produced by a prior
            run's ``composite.state``).  Used to debug late-pipeline
            steps without re-running the expensive Step 5.
        resume_state: store-state dict from a prior run, required when
            ``resume_from_step > 1``.
    Returns:
        The ``Composite`` instance with the pipeline already executed.
        The final store state is at ``composite.state``.
    """
    if resume_from_step > 1 and resume_state is None:
        raise ValueError(
            "resume_from_step > 1 requires resume_state from a prior run")
    if core is None:
        core = allocate_core(top=ALL_STEP_CLASSES)
        register_parca_schema(core)

    from v2ecoli.processes.parca.steps.step_01_initialize        import OUTPUT_PORTS as _s1_out
    from v2ecoli.processes.parca.steps.step_02_input_adjustments import (
        INPUT_PORTS as _s2_in, OUTPUT_PORTS as _s2_out)
    from v2ecoli.processes.parca.steps.step_03_basal_specs       import (
        INPUT_PORTS as _s3_in, OUTPUT_PORTS as _s3_out)
    from v2ecoli.processes.parca.steps.step_04_tf_condition_specs import (
        INPUT_PORTS as _s4_in, OUTPUT_PORTS as _s4_out)
    from v2ecoli.processes.parca.steps.step_05_fit_condition     import (
        INPUT_PORTS as _s5_in, OUTPUT_PORTS as _s5_out)
    from v2ecoli.processes.parca.steps.step_06_promoter_binding  import (
        INPUT_PORTS as _s6_in, OUTPUT_PORTS as _s6_out)
    from v2ecoli.processes.parca.steps.step_07_adjust_promoters  import (
        INPUT_PORTS as _s7_in, OUTPUT_PORTS as _s7_out)
    from v2ecoli.processes.parca.steps.step_08_set_conditions    import (
        INPUT_PORTS as _s8_in, OUTPUT_PORTS as _s8_out)
    from v2ecoli.processes.parca.steps.step_09_final_adjustments import (
        INPUT_PORTS as _s9_in, OUTPUT_PORTS as _s9_out)

    _elong = dict(
        variable_elongation_transcription=variable_elongation_transcription,
        variable_elongation_translation=variable_elongation_translation,
        disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
        disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
    )

    # Ticks are already baked into each step's INPUT_PORTS / OUTPUT_PORTS
    # so _wires() picks them up automatically.
    s1i, s1o = {}, _wires(_s1_out.keys())
    s2i, s2o = _wires(_s2_in.keys()), _wires(_s2_out.keys())
    s3i, s3o = _wires(_s3_in.keys()), _wires(_s3_out.keys())
    s4i, s4o = _wires(_s4_in.keys()), _wires(_s4_out.keys())
    s5i, s5o = _wires(_s5_in.keys()), _wires(_s5_out.keys())
    s6i, s6o = _wires(_s6_in.keys()), _wires(_s6_out.keys())
    s7i, s7o = _wires(_s7_in.keys()), _wires(_s7_out.keys())
    s8i, s8o = _wires(_s8_in.keys()), _wires(_s8_out.keys())
    s9i, s9o = _wires(_s9_in.keys()), _wires(_s9_out.keys())

    # Each step slot.  We build them once and selectively include them
    # below according to ``resume_from_step``.
    step_slots = {
        'initialize': {
            '_type': 'step', 'address': 'local:InitializeStep',
            'config': {'raw_data': raw_data},
            'inputs': s1i, 'outputs': s1o,
        },
        'input_adjustments': {
            '_type': 'step', 'address': 'local:InputAdjustmentsStep',
            'config': {'debug': debug},
            'inputs': s2i, 'outputs': s2o,
        },
        'basal_specs': {
            '_type': 'step', 'address': 'local:BasalSpecsStep',
            'config': {**_elong, 'cache_dir': cache_dir},
            'inputs': s3i, 'outputs': s3o,
        },
        'tf_condition_specs': {
            '_type': 'step', 'address': 'local:TfConditionSpecsStep',
            'config': {**_elong, 'cpus': cpus},
            'inputs': s4i, 'outputs': s4o,
        },
        'fit_condition': {
            '_type': 'step', 'address': 'local:FitConditionStep',
            'config': {'cpus': cpus},
            'inputs': s5i, 'outputs': s5o,
        },
        'promoter_binding': {
            '_type': 'step', 'address': 'local:PromoterBindingStep',
            'config': {},
            'inputs': s6i, 'outputs': s6o,
        },
        'adjust_promoters': {
            '_type': 'step', 'address': 'local:AdjustPromotersStep',
            'config': {},
            'inputs': s7i, 'outputs': s7o,
        },
        'set_conditions': {
            '_type': 'step', 'address': 'local:SetConditionsStep',
            'config': {'verbose': 1},
            'inputs': s8i, 'outputs': s8o,
        },
        'final_adjustments': {
            '_type': 'step', 'address': 'local:FinalAdjustmentsStep',
            'config': {},
            'inputs': s9i, 'outputs': s9o,
        },
    }

    STEP_ORDER = [
        'initialize', 'input_adjustments', 'basal_specs', 'tf_condition_specs',
        'fit_condition', 'promoter_binding', 'adjust_promoters',
        'set_conditions', 'final_adjustments',
    ]

    state = {}
    # Seed leaves from a prior run when resuming.  Skip composite-internal
    # bookkeeping keys and the step slot keys themselves.  Checkpoints are
    # keyed by port name (from Step.update return dicts), but the composite's
    # store expects values at their nested STORE_PATH locations — e.g. the
    # 'transcription' port lives at ['process', 'transcription'].  Seed via
    # STORE_PATH so step inputs find the values where their wires point.
    if resume_from_step > 1 and resume_state:
        for k, v in resume_state.items():
            if k in STEP_ORDER or k.startswith('_') or k == 'global_time':
                continue
            path = STORE_PATH.get(k, [k])
            cursor = state
            for seg in path[:-1]:
                cursor = cursor.setdefault(seg, {})
            cursor[path[-1]] = v

    # Include only steps from resume_from_step onward.
    for i, name in enumerate(STEP_ORDER, start=1):
        if i >= resume_from_step:
            state[name] = step_slots[name]

    spec = {'run_steps_on_init': True, 'state': state}

    return Composite(spec, core=core)


def run_parca(raw_data, debug=False):
    """Build the PoC composite, let the Step DAG execute, return store state."""
    composite = build_parca_composite(raw_data, debug=debug)
    return composite.state
