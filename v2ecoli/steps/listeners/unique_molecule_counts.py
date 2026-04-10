"""
===============================
Unique Molecule Counts Listener
===============================

Counts unique molecules
"""

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import numpy_schema, listener_schema
# topology_registry removed — topology defined as class attribute

# Register default topology for this process, associating it with process name
NAME = "unique_molecule_counts"
TOPOLOGY = {
    "unique": ("unique",),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class UniqueMoleculeCounts(Step):
    """UniqueMoleculeCounts"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
        'unique_ids': 'list[string]',
    }


    def inputs(self):
        return {
            'unique': 'map[node]',
            'global_time': 'float',
            'timestep': 'float',
        }

    def outputs(self):
        return {
            'listeners': {
                'unique_molecule_counts': 'map[integer]',
            },
        }


    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.unique_ids = self.parameters["unique_ids"]

    def port_defaults(self):
        """Default values for ports that need pre-population."""
        return {
            'unique': {
                'active_RNAP': [],
                'RNA': [],
                'full_chromosome': [],
                'chromosome_domain': [],
                'active_replisome': [],
                'oriC': [],
                'promoter': [],
                'gene': [],
                'chromosomal_segment': [],
                'active_ribosome': [],
                'DnaA_box': [],
            },
            'listeners': {
                'unique_molecule_counts': {
                    'active_RNAP': 0,
                    'RNA': 0,
                    'active_ribosome': 0,
                    'full_chromosome': 0,
                    'chromosome_domain': 0,
                    'active_replisome': 0,
                    'oriC': 0,
                    'promoter': 0,
                    'gene': 0,
                    'chromosomal_segment': 0,
                    'DnaA_box': 0,
                },
            },
            'global_time': 0.0,
            'timestep': 1.0,
        }

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on first tick if data not yet populated
        return {
            "listeners": {
                "unique_molecule_counts": {
                    str(unique_id): states["unique"][unique_id]["_entryState"].sum()
                    for unique_id in self.unique_ids
                }
            }
        }
