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
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict
from v2ecoli.steps.partition import _protect_state


class ProteinDegradationLogic:
    """Biological logic for protein degradation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**ProteinDegradation.defaults, **(parameters or {})}
        parameters = self.parameters
        self.raw_degradation_rate = parameters["raw_degradation_rate"]
        self.water_id = parameters["water_id"]
        self.amino_acid_ids = parameters["amino_acid_ids"]
        self.amino_acid_counts = parameters["amino_acid_counts"]

        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)

        self.protein_ids = parameters["protein_ids"]
        self.protein_lengths = parameters["protein_lengths"]

        self.seed = parameters.get("seed", 0)
        self.random_state = np.random.RandomState(seed=self.seed)

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

    def _init_indices(self, bulk_ids):
        if self.metabolite_idx is None:
            self.water_idx = bulk_name_to_idx(self.water_id, bulk_ids)
            self.protein_idx = bulk_name_to_idx(self.protein_ids, bulk_ids)
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_ids, bulk_ids)

    def _proteinDegRates(self, timestep):
        return self.raw_degradation_rate * timestep


class ProteinDegradationRequester(Step):
    """Requester step for protein degradation.

    Calculates which proteins to degrade via Poisson process,
    writes bulk molecule requests to the request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ProteinDegradationLogic(config)
        self.process_name = 'ecoli-protein-degradation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])
        timestep = state.get('timestep', 1.0)

        protein_data = counts(state['bulk'], proc.protein_idx)
        nProteinsToDegrade = np.fmin(
            proc.random_state.poisson(
                proc._proteinDegRates(timestep) * protein_data),
            protein_data)
        nReactions = np.dot(proc.protein_lengths, nProteinsToDegrade)

        bulk_request = [
            (proc.protein_idx, nProteinsToDegrade),
            (proc.water_idx, nReactions - np.sum(nProteinsToDegrade)),
        ]

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class ProteinDegradationEvolver(Step):
    """Evolver step for protein degradation.

    Reads allocated bulk molecules from the allocate store,
    degrades proteins and releases amino acids.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ProteinDegradationLogic(config)
        self.process_name = 'ecoli-protein-degradation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        # Evolve: degrade allocated proteins, release amino acids
        allocated_proteins = counts(state['bulk'], proc.protein_idx)
        metabolites_delta = np.dot(proc.degradation_matrix, allocated_proteins)

        return {
            'bulk': [
                (proc.metabolite_idx, metabolites_delta),
                (proc.protein_idx, -allocated_proteins),
            ],
            'next_update_time': state.get('global_time', 0) + state.get('timestep', 1.0),
        }


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class ProteinDegradation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = ProteinDegradationLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        self._logic._init_indices(states["bulk"]["id"])
        proc = self._logic
        protein_data = counts(states["bulk"], proc.protein_idx)
        nProteinsToDegrade = np.fmin(
            proc.random_state.poisson(
                proc._proteinDegRates(timestep) * protein_data),
            protein_data)
        nReactions = np.dot(proc.protein_lengths, nProteinsToDegrade)
        return {
            "bulk": [
                (proc.protein_idx, nProteinsToDegrade),
                (proc.water_idx, nReactions - np.sum(nProteinsToDegrade)),
            ]
        }

    def evolve_state(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])
        allocated_proteins = counts(states["bulk"], proc.protein_idx)
        metabolites_delta = np.dot(proc.degradation_matrix, allocated_proteins)
        return {
            "bulk": [
                (proc.metabolite_idx, metabolites_delta),
                (proc.protein_idx, -allocated_proteins),
            ]
        }
