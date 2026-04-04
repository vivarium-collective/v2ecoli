"""
=========================
DNA Supercoiling Listener
=========================
"""

import numpy as np
from process_bigraph import Step
from v2ecoli.library.schema import numpy_schema, listener_schema, attrs


class DnaSupercoiling(Step):
    """
    Listener for DNA supercoiling data.
    """

    name = "dna_supercoiling_listener"
    config_schema = {}

    defaults = {
        "relaxed_DNA_base_pairs_per_turn": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.relaxed_DNA_base_pairs_per_turn = self.parameters.get(
            "relaxed_DNA_base_pairs_per_turn", 0
        )

    def ports_schema(self):
        return {
            "listeners": {
                "dna_supercoiling": listener_schema(
                    {
                        "segment_left_boundary_coordinates": [],
                        "segment_right_boundary_coordinates": [],
                        "segment_domain_indexes": [],
                        "segment_superhelical_densities": [],
                    }
                )
            },
            "chromosomal_segments": numpy_schema(
                "chromosomal_segments", emit=self.parameters.get("emit_unique", False)
            ),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters.get("time_step", 1)},
        }

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
