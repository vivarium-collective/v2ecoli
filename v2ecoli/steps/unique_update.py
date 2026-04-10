from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import numpy_schema
from v2ecoli.library.schema_types import UNIQUE_TYPES


class UniqueUpdate(Step):
    """Placed after all Steps of each execution layer (see :ref:`partitioning`)
    to ensure that unique molecules are completely up-to-date"""

    name = "unique-update"

    config_schema = {
        'emit_unique': 'boolean{false}',
        'unique_topo': 'map[string]',
    }

    def inputs(self):
        return {mol: UNIQUE_TYPES.get(mol, 'node') for mol in self.unique_topo}

    def outputs(self):
        return {mol: UNIQUE_TYPES.get(mol, 'node') for mol in self.unique_topo}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.unique_topo = self.parameters["unique_topo"]

    def update(self, states, interval=None):
        return {unique_mol: {"update": True} for unique_mol in self.unique_topo.keys()}
