"""
TfUnbinding
Unbind transcription factors from DNA to allow signaling processes before
binding back to DNA.
"""

import numpy as np
import warnings

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Node, Float, Overwrite

from v2ecoli.library.schema import bulk_name_to_idx, attrs
from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict


class TfUnbindingLogic:
    """TfUnbinding logic — pure computation, no Step inheritance."""

    name = "ecoli-tf-unbinding"
    topology = {
    "bulk": ("bulk",),
    "promoters": (
        "unique",
        "promoter",
    ),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_unbinding"),
}

    defaults = {"time_step": 1, "emit_unique": False}

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

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

    def next_update(self, timestep, states):
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


class TfUnbindingStep(_SafeInvokeMixin, Step):
    """Single Step that runs TfUnbindingLogic."""

    config_schema = {}

    def initialize(self, config):
        self.logic = TfUnbindingLogic(parameters=config)
        self.topology = self.logic.topology

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'global_time': InPlaceDict(),
            'timestep': InPlaceDict(),
            'next_update_time': Overwrite(_value=Node()),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'global_time': InPlaceDict(),
            'timestep': InPlaceDict(),
            'next_update_time': Overwrite(_value=Node()),
        }

    def update(self, state, interval=None):
        state = _protect_state(state)
        timestep = state.get('timestep', self.logic.parameters.get('time_step', 1.0))
        return self.logic.next_update(timestep, state)
