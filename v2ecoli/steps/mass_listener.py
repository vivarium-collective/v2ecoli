"""
=============
Mass Listener
=============

v2-native wrapper for the vEcoli MassListener step.

Computes total cellular mass from bulk molecule counts and unique
molecule states.  Outputs mass breakdown to listeners/mass.
"""

from bigraph_schema.schema import Node, Float, Overwrite
from process_bigraph import Step

from ecoli.processes.listeners.mass_listener import (
    MassListener as V1MassListener,
    NAME,
    TOPOLOGY,
)
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate


def _mass_output_schema():
    """Schema for each mass output: float with 'set' semantics."""
    return Overwrite(_value=Float(), _default=0.0)


class MassListener(Step):
    """v2-native mass listener step.

    Wraps the v1 MassListener biological logic, exposing typed
    inputs/outputs for the Composite pipeline.
    """

    name = NAME
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._v1 = V1MassListener(parameters=config)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'unique': Node(),
            'listeners': {'mass': Node()},
            'global_time': Float(_default=0.0),
            'timestep': Float(_default=1.0),
        }

    def outputs(self):
        mass_keys = [
            'cell_mass', 'water_mass', 'dry_mass', 'volume',
            'protein_mass', 'rna_mass', 'rRna_mass', 'tRna_mass',
            'mRna_mass', 'dna_mass', 'smallMolecule_mass',
            'protein_mass_fraction', 'rna_mass_fraction',
            'growth', 'instantaneous_growth_rate',
            'dry_mass_fold_change', 'protein_mass_fold_change',
            'rna_mass_fold_change', 'small_molecule_fold_change',
            'projection_mass', 'cytosol_mass', 'extracellular_mass',
            'flagellum_mass', 'membrane_mass', 'outer_membrane_mass',
            'periplasm_mass', 'pilus_mass', 'inner_membrane_mass',
            'expected_mass_fold_change',
        ]
        return {
            'listeners': {
                'mass': {key: _mass_output_schema() for key in mass_keys}
            },
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        timestep = state.get('timestep', 1.0)
        delta = self._v1.next_update(timestep, state)
        return delta if delta else {}
