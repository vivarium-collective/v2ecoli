"""Step 1 — initialize (scatter).  Bootstrap the pipeline from raw KB data.

Takes the ``KnowledgeBaseEcoli`` loaded from the flat-file TSVs and builds
the nested ``SimulationDataEcoli`` object, then scatters its nine subsystem
dataclasses and twenty-odd top-level dicts onto their natural store paths
in the bigraph. No ``sim_data`` blob is passed downstream; every subsequent
step wires only the ports it needs.

Mathematical Model
------------------

Inputs:
- raw_data: a ``KnowledgeBaseEcoli`` instance holding every flat-file table
  (genes, RNAs, proteins, reactions, media, mass fractions, ...) parsed
  from ``v2ecoli.processes.parca/reconstruction/ecoli/flat/``. Delivered as a ``config``
  parameter (not a port wire) so bigraph-schema does not introspect the
  KB's deep internals at composite-construction time.

Parameters:
- debug (bool): forwarded to downstream steps; no effect on this one.

Calculation:
- sim_data = SimulationDataEcoli()
- sim_data.initialize(raw_data=raw_data, basal_expression_condition=...)
- split sim_data into its constituent subsystem objects and data dicts.

Outputs:
- Subsystem objects on ``process/*`` stores: transcription, translation,
  metabolism, rna_decay, complexation, equilibrium, two_component_system,
  transcription_regulation, replication.
- Top-level dataclass stores: mass, constants, growth_rate_parameters,
  adjustments, molecule_groups, molecule_ids, relation, getter, external_state.
- Pure-data dicts: conditions, condition_to_doubling_time, tf_to_fold_change,
  tf_to_active_inactive_conditions, condition_active_tfs, condition_inactive_tfs.
- bulk_molecules at ``internal_state/bulk_molecules``.
"""

import time

from process_bigraph import Step

from v2ecoli.processes.parca.reconstruction.ecoli.simulation_data import SimulationDataEcoli


# Subsystem object outputs — each typed by its corresponding schema entry.
_SUBSYSTEM_PORTS = {
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'rna_decay':                'sim_data.rna_decay',
    'complexation':             'sim_data.complexation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    # Not every sim_data subsystem has a dedicated schema type registered;
    # use overwrite for the remainder.
    'adjustments':              'overwrite',
    'molecule_groups':          'overwrite',
    'molecule_ids':             'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'external_state':           'overwrite',
    # Escape-hatch port — the live SimulationDataEcoli instance.  Useful
    # for the handful of sub-functions that call methods defined on
    # sim_data itself (e.g. sim_data.calculate_ppgpp_expression) rather
    # than on one of its subsystems.  Mutations to the subsystems carried
    # by other ports propagate through this reference automatically.
    'sim_data_root':            'overwrite',
}

# Pure-data top-level dict outputs.
_DATA_LEAF_PORTS = {
    'tf_to_active_inactive_conditions': 'overwrite',
    'conditions':                       'overwrite',
    'condition_to_doubling_time':       'overwrite',
    'tf_to_fold_change':                'overwrite',
    'tf_to_direction':                  'overwrite',
    'condition_active_tfs':             'overwrite',
    'condition_inactive_tfs':           'overwrite',
    # Seed cell_specs as an empty dict — steps 3–9 populate per-condition
    # entries into it.  Tracked as its own leaf so every step's cell_specs
    # read/write is visible in the composite wires.
    'cell_specs':                       'overwrite',
    # Seeded empty; step 5 writes per-nutrient entries.
    'translation_supply_rate':          'overwrite',
    # Seeded empty; step 8 populates.
    'expected_dry_mass_increase_dict':  'overwrite',
    # Seeded empty; step 6 populates, step 7 consumes.
    'pPromoterBound':                   'overwrite',
    # sim_data.condition is a mutable runtime attr (default "basal")
    # used by create_bulk_container to pick the right nutrient media.
    'condition':                        'overwrite',
}

OUTPUT_PORTS = {
    'tick_1': 'overwrite',
    **_SUBSYSTEM_PORTS,
    **_DATA_LEAF_PORTS,
}


class InitializeStep(Step):
    """Run ``sim_data.initialize(raw_data=...)`` and scatter its subsystems."""

    config_schema = {
        'raw_data':                   'overwrite',
        'basal_expression_condition': {
            '_type': 'string',
            '_default': 'M9 Glucose minus AAs',
        },
    }

    def inputs(self):
        return {}

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()
        raw_data = self.config['raw_data']

        sim_data = SimulationDataEcoli()
        sim_data.initialize(
            raw_data=raw_data,
            basal_expression_condition=self.config.get(
                'basal_expression_condition', 'M9 Glucose minus AAs'),
        )

        # Scatter subsystems as live object references (no copies) so
        # downstream steps can mutate them in place and the mutations
        # persist in the store.
        out = {
            # subsystems
            'transcription':            sim_data.process.transcription,
            'translation':              sim_data.process.translation,
            'metabolism':               sim_data.process.metabolism,
            'rna_decay':                sim_data.process.rna_decay,
            'complexation':             sim_data.process.complexation,
            'equilibrium':              sim_data.process.equilibrium,
            'two_component_system':     sim_data.process.two_component_system,
            'transcription_regulation': sim_data.process.transcription_regulation,
            'replication':              sim_data.process.replication,
            'mass':                     sim_data.mass,
            'constants':                sim_data.constants,
            'growth_rate_parameters':   sim_data.growth_rate_parameters,
            'adjustments':              sim_data.adjustments,
            'molecule_groups':          sim_data.molecule_groups,
            'molecule_ids':             sim_data.molecule_ids,
            'relation':                 sim_data.relation,
            'getter':                   sim_data.getter,
            'bulk_molecules':           sim_data.internal_state.bulk_molecules,
            'external_state':           sim_data.external_state,
            'sim_data_root':            sim_data,
            # pure-data top-level dicts (copied — callers may mutate)
            'tf_to_active_inactive_conditions':
                dict(sim_data.tf_to_active_inactive_conditions),
            'conditions':                 dict(sim_data.conditions),
            'condition_to_doubling_time': dict(sim_data.condition_to_doubling_time),
            'tf_to_fold_change':          dict(sim_data.tf_to_fold_change),
            'tf_to_direction':            dict(sim_data.tf_to_direction),
            'condition_active_tfs':       dict(sim_data.condition_active_tfs),
            'condition_inactive_tfs':     dict(sim_data.condition_inactive_tfs),
            'cell_specs':                 {},
            'translation_supply_rate':    dict(sim_data.translation_supply_rate),
            'expected_dry_mass_increase_dict': {},
            'pPromoterBound':             {},
            'condition':                  sim_data.condition,
            'tick_1': True,
        }

        print(f"  Step 1 (initialize + scatter) completed in {time.time() - t0:.1f}s")
        return out
