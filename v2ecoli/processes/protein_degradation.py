"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.
"""

import numpy as np

from process_bigraph import Step

from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict


class ProteinDegradationStep(_SafeInvokeMixin, Step):
    """Protein degradation — single-step Poisson sampling + degradation."""

    config_schema = {}

    topology = {"bulk": ("bulk",), "timestep": ("timestep",)}

    def initialize(self, config):
        defaults = {
            "raw_degradation_rate": [],
            "water_id": "h2o",
            "amino_acid_ids": [],
            "amino_acid_counts": [],
            "protein_ids": [],
            "protein_lengths": [],
            "seed": 0,
            "time_step": 1,
        }
        params = {**defaults, **config}

        self.raw_degradation_rate = params["raw_degradation_rate"]
        self.water_id = params["water_id"]
        self.amino_acid_ids = params["amino_acid_ids"]
        self.amino_acid_counts = params["amino_acid_counts"]
        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)
        self.protein_ids = params["protein_ids"]
        self.protein_lengths = params["protein_lengths"]
        self.random_state = np.random.RandomState(seed=params["seed"])
        self.metabolite_idx = None

        # Build S matrix
        self.degradation_matrix = np.zeros(
            (len(self.metabolite_ids), len(self.protein_ids)), np.int64
        )
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts
        )
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)

        # Initialize indices on first call
        if self.metabolite_idx is None:
            bulk_ids = state["bulk"]["id"]
            self.water_idx = bulk_name_to_idx(self.water_id, bulk_ids)
            self.protein_idx = bulk_name_to_idx(self.protein_ids, bulk_ids)
            self.metabolite_idx = bulk_name_to_idx(self.metabolite_ids, bulk_ids)

        # Poisson-sample proteins to degrade, capped at available
        protein_data = counts(state["bulk"], self.protein_idx)
        nProteinsToDegrade = np.fmin(
            self.random_state.poisson(
                self.raw_degradation_rate * timestep * protein_data
            ),
            protein_data,
        )

        # Compute metabolite changes from degradation
        metabolites_delta = np.dot(self.degradation_matrix, nProteinsToDegrade)

        return {
            "bulk": [
                (self.metabolite_idx, metabolites_delta),
                (self.protein_idx, -nProteinsToDegrade),
            ],
            "next_update_time": global_time + timestep,
        }
