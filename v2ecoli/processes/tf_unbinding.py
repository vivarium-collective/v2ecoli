"""
TfUnbinding
Unbind transcription factors from DNA to allow signaling processes before
binding back to DNA.
"""

import numpy as np
import warnings

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import bulk_name_to_idx, attrs, numpy_schema
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


def _protect_standalone_state(state):
    """Protect state for standalone processes.

    Unlike partitioned processes whose unique molecules live under a
    'unique' key, standalone processes may have unique molecule ports
    (promoters, active_RNAPs, etc.) at the top level.  Copy any numpy
    structured arrays so biological logic can mutate them safely.
    """
    import numpy as np
    protected = dict(state)
    if 'bulk' in protected and hasattr(protected['bulk'], 'copy'):
        protected['bulk'] = protected['bulk'].copy()
        protected['bulk'].flags.writeable = True
    # Copy any top-level structured arrays (unique molecule tables)
    for key, val in protected.items():
        if key == 'bulk':
            continue
        if hasattr(val, 'dtype') and hasattr(val, 'copy'):
            protected[key] = val.copy()
            protected[key].flags.writeable = True
    return protected


class TfUnbinding(Step):
    """TfUnbinding Step — standalone (no request/allocate partition)."""

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

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)
        self.parameters = {**self.defaults, **(config or {})}
        self.tf_ids = self.parameters["tf_ids"]
        self.submass_indices = self.parameters["submass_indices"]
        self.active_tf_masses = self.parameters["active_tf_masses"]

        # Numpy indices for bulk molecules
        self.active_tf_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'global_time': Float(_default=0.0),
            'timestep': Float(_default=self.parameters.get('time_step', 1.0)),
            'next_update_time': Float(_default=self.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        # Timing guard (replaces update_condition)
        next_t = state.get('next_update_time', 0)
        global_t = state.get('global_time', 0)
        if next_t > global_t:
            return {}
        if next_t < global_t:
            warnings.warn(
                f"{self.name} updated at t="
                f"{global_t} instead of t="
                f"{next_t}. Decrease the "
                "timestep for the global clock process for more "
                "accurate timekeeping."
            )

        state = _protect_standalone_state(state)
        return self.next_update(state.get('timestep', 1.0), state)

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
