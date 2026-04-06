"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.

TODO:
 - protein complexes
 - add protease functionality
"""

import numpy as np

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx
from v2ecoli.steps.partition import PartitionedProcess, _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict
class ProteinDegradation(PartitionedProcess):
    """Protein Degradation PartitionedProcess"""

    name = "ecoli-protein-degradation"
    topology = {"bulk": ("bulk",), "timestep": ("timestep",)}
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

    # Constructor
    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)

        self.raw_degradation_rate = self.parameters["raw_degradation_rate"]

        self.water_id = self.parameters["water_id"]
        self.amino_acid_ids = self.parameters["amino_acid_ids"]
        self.amino_acid_counts = self.parameters["amino_acid_counts"]

        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)

        # Build protein IDs for S matrix
        self.protein_ids = self.parameters["protein_ids"]
        self.protein_lengths = self.parameters["protein_lengths"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.metabolite_idx = None

        # Build S matrix
        self.degradation_matrix = np.zeros(
            (len(self.metabolite_ids), len(self.protein_ids)), np.int64
        )
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts
        )
        # Assuming N-1 H2O is required per peptide chain length N
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        # In first timestep, convert all strings to indices
        if self.metabolite_idx is None:
            self.water_idx = bulk_name_to_idx(self.water_id, states["bulk"]["id"])
            self.protein_idx = bulk_name_to_idx(self.protein_ids, states["bulk"]["id"])
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_ids, states["bulk"]["id"]
            )

        protein_data = counts(states["bulk"], self.protein_idx)
        # Determine how many proteins to degrade based on the degradation rates
        # and counts of each protein
        nProteinsToDegrade = np.fmin(
            self.random_state.poisson(
                self._proteinDegRates(states["timestep"]) * protein_data
            ),
            protein_data,
        )

        # Determine the number of hydrolysis reactions
        # TODO(vivarium): Missing asNumber() and other unit-related things
        nReactions = np.dot(self.protein_lengths, nProteinsToDegrade)

        # Determine the amount of water required to degrade the selected proteins
        # Assuming one N-1 H2O is required per peptide chain length N
        requests = {
            "bulk": [
                (self.protein_idx, nProteinsToDegrade),
                (self.water_idx, nReactions - np.sum(nProteinsToDegrade)),
            ]
        }
        return requests

    def evolve_state(self, timestep, states):
        # Degrade selected proteins, release amino acids from those proteins
        # back into the cell, and consume H_2O that is required for the
        # degradation process
        allocated_proteins = counts(states["bulk"], self.protein_idx)
        metabolites_delta = np.dot(self.degradation_matrix, allocated_proteins)

        update = {
            "bulk": [
                (self.metabolite_idx, metabolites_delta),
                (self.protein_idx, -allocated_proteins),
            ]
        }

        return update

    def _proteinDegRates(self, timestep):
        return self.raw_degradation_rate * timestep


class ProteinDegradationRequester(_SafeInvokeMixin, Step):
    """Explicit requester Step for ProteinDegradation.

    Reads bulk to compute degradation request. Writes only to request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        request = self.process.calculate_request(timestep, state)
        self.process.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {self.process_name: {}}}
        if bulk_request is not None:
            result['request'][self.process_name]['bulk'] = bulk_request

        return result


class ProteinDegradationEvolver(_SafeInvokeMixin, Step):
    """Explicit evolver Step for ProteinDegradation.

    Reads allocation, writes bulk updates.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        # Apply allocation: replace bulk counts with allocated amounts
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        update = self.process.evolve_state(timestep, state)
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
