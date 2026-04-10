"""
=========================
DNA Supercoiling Listener
=========================
"""

import numpy as np
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import attrs
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from bigraph_schema.schema import Float


class DnaSupercoiling(Step):
    """
    Listener for DNA supercoiling data.
    """

    name = "dna_supercoiling_listener"
    config_schema = {
        "relaxed_DNA_base_pairs_per_turn": {"_default": 0},
        "emit_unique": {"_default": False},
        "time_step": {"_default": 1},
    }
    topology = {
        "listeners": ("listeners",),
        "chromosomal_segments": ("unique", "chromosomal_segment"),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.relaxed_DNA_base_pairs_per_turn = self.parameters.get(
            "relaxed_DNA_base_pairs_per_turn", 0
        )

    def inputs(self):
        return {
            "listeners": ListenerStore(),
            "chromosomal_segments": UniqueNumpyUpdate(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        boundary_coordinates, domain_indexes, linking_numbers = attrs(
            states["chromosomal_segments"],
            ["boundary_coordinates", "domain_index", "linking_number"],
        )

        if len(boundary_coordinates) == 0:
            return {
                "listeners": {
                    "dna_supercoiling": {
                        "segment_left_boundary_coordinates": [],
                        "segment_right_boundary_coordinates": [],
                        "segment_domain_indexes": [],
                        "segment_superhelical_densities": [],
                    }
                }
            }

        # Get mask for segments with nonzero lengths
        segment_lengths = boundary_coordinates[:, 1] - boundary_coordinates[:, 0]

        assert np.all(segment_lengths >= 0)
        nonzero_length_mask = segment_lengths > 0

        # Calculate superhelical densities
        linking_numbers_relaxed_DNA = (
            segment_lengths[nonzero_length_mask] / self.relaxed_DNA_base_pairs_per_turn
        )

        update = {
            "listeners": {
                "dna_supercoiling": {
                    "segment_left_boundary_coordinates": boundary_coordinates[
                        nonzero_length_mask, 0
                    ],
                    "segment_right_boundary_coordinates": boundary_coordinates[
                        nonzero_length_mask, 1
                    ],
                    "segment_domain_indexes": domain_indexes[nonzero_length_mask],
                    "segment_superhelical_densities": np.divide(
                        linking_numbers[nonzero_length_mask]
                        - linking_numbers_relaxed_DNA,
                        linking_numbers_relaxed_DNA,
                    ),
                }
            }
        }
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
