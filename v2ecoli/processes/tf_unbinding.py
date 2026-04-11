"""
TfUnbinding
Unbind transcription factors from DNA to allow signaling processes before
binding back to DNA.
"""

import numpy as np
import warnings

from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed
from v2ecoli.library.schema import bulk_name_to_idx, attrs, numpy_schema
from v2ecoli.library.schema_types import PROMOTER_ARRAY

# Register default topology for this process, associating it with process name
NAME = "ecoli-tf-unbinding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "promoters": (
        "unique",
        "promoter",
    ),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_unbinding"),
}


class TfUnbinding(Step):
    """TfUnbinding"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_tf_masses': 'array[float]',
        'emit_unique': {'_type': 'boolean', '_default': False},
        'submass_indices': 'map[integer]',
        'tf_ids': 'list[string]',
        'time_step': {'_type': 'integer', '_default': 1},
    }


    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'integer', '_default': 1},
            'next_update_time': {'_type': 'overwrite[float]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'promoters': PROMOTER_ARRAY,
            'next_update_time': 'overwrite[float]',
        }


    def initialize(self, config):
        self.tf_ids = self.parameters["tf_ids"]
        self.submass_indices = self.parameters["submass_indices"]
        self.active_tf_masses = self.parameters["active_tf_masses"]

        # Numpy indices for bulk molecules
        self.active_tf_idx = None

    def update_condition(self, timestep, states):
        """
        See :py:meth:`~.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def update(self, states, interval=None):
        # At t=0, convert all strings to indices
        if self.active_tf_idx is None:
            self.active_tf_idx = bulk_name_to_idx(
                [tf + "[c]" for tf in self.tf_ids], states["bulk"]["id"]
            )

        # Get attributes of all promoters
        (bound_TF,) = attrs(states["promoters"], ["bound_TF"])
        # If there are no promoters, return immediately
        if len(bound_TF) == 0:
            return {}

        # Calculate number of bound TFs for each TF prior to changes
        n_bound_TF = bound_TF.sum(axis=0)

        update = {
            # Free all DNA-bound TFs into free active TFs
            "bulk": [(self.active_tf_idx, n_bound_TF)],
            "promoters": {
                # Reset bound_TF attribute of promoters
                "set": {"bound_TF": np.zeros_like(bound_TF)}
            },
        }

        # Add mass_diffs array to promoter submass
        mass_diffs = bound_TF @ -self.active_tf_masses
        for submass, idx in self.submass_indices.items():
            update["promoters"]["set"][submass] = (
                attrs(states["promoters"], [submass])[0] + mass_diffs[:, idx]
            )

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update
