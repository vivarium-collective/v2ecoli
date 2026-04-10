"""Process-bigraph partitioned process: protein_degradation."""

from typing import Any, Callable, Optional, Tuple, cast

from numba import njit
import numpy as np
import numpy.typing as npt
import scipy.sparse
import warnings
from scipy.integrate import solve_ivp

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.fitting import normalize
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.random import stochasticRound
from v2ecoli.library.schema import (
    create_unique_indices,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
    zero_listener,
)
from v2ecoli.library.unit_defs import units
from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore, SetStore


def _apply_config_defaults(config_schema, parameters):
    """Merge config_schema defaults with provided parameters."""
    merged = {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and "_default" in spec:
            merged[key] = spec["_default"]
    merged.update(parameters or {})
    return merged

class ProteinDegradationLogic:
    """Protein degradation — shared state container for Requester/Evolver."""

    name = "ecoli-protein-degradation"
    topology = {"bulk": ("bulk",), "timestep": ("timestep",)}
    config_schema = {
        "raw_degradation_rate": {"_default": []},
        "water_id": {"_default": "h2o"},
        "amino_acid_ids": {"_default": []},
        "amino_acid_counts": {"_default": []},
        "protein_ids": {"_default": []},
        "protein_lengths": {"_default": []},
        "seed": {"_default": 0},
        "time_step": {"_default": 1},
    }

    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
        self.request_set = False

        self.raw_degradation_rate = self.parameters["raw_degradation_rate"]
        self.water_id = self.parameters["water_id"]
        self.amino_acid_ids = self.parameters["amino_acid_ids"]
        self.amino_acid_counts = self.parameters["amino_acid_counts"]
        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)
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
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )


class ProteinDegradationRequester(_SafeInvokeMixin, Step):
    """Reads bulk to compute degradation request. Writes to request store."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from calculate_request ---
        if p.metabolite_idx is None:
            p.water_idx = bulk_name_to_idx(p.water_id, state["bulk"]["id"])
            p.protein_idx = bulk_name_to_idx(p.protein_ids, state["bulk"]["id"])
            p.metabolite_idx = bulk_name_to_idx(
                p.metabolite_ids, state["bulk"]["id"]
            )

        protein_data = counts(state["bulk"], p.protein_idx)
        nProteinsToDegrade = np.fmin(
            p.random_state.poisson(
                p.raw_degradation_rate * timestep * protein_data
            ),
            protein_data,
        )
        nReactions = np.dot(p.protein_lengths, nProteinsToDegrade)

        request = {
            "bulk": [
                (p.protein_idx, nProteinsToDegrade),
                (p.water_idx, nReactions - np.sum(nProteinsToDegrade)),
            ]
        }
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class ProteinDegradationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': SetStore(),
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        p = self.process
        timestep = state.get('timestep', 1.0)

        # --- inlined from evolve_state ---
        allocated_proteins = counts(state["bulk"], p.protein_idx)
        metabolites_delta = np.dot(p.degradation_matrix, allocated_proteins)

        update = {
            "bulk": [
                (p.metabolite_idx, metabolites_delta),
                (p.protein_idx, -allocated_proteins),
            ]
        }
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
