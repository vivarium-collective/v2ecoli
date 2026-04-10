"""Process-bigraph partitioned process: rna_maturation."""

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
from v2ecoli.library.units import units
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

class RnaMaturationLogic:
    """RnaMaturation — shared state container for Requester/Evolver."""

    name = "ecoli-rna-maturation"
    topology = {"bulk": ("bulk",), "bulk_total": ("bulk",), "listeners": ("listeners",)}
    config_schema = {}

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
        self.request_set = False
        # Get matrices and vectors that describe maturation reactions
        self.stoich_matrix = self.parameters["stoich_matrix"]
        self.enzyme_matrix = self.parameters["enzyme_matrix"]
        self.n_required_enzymes = self.parameters["n_required_enzymes"]
        self.degraded_nt_counts = self.parameters["degraded_nt_counts"]
        self.n_ppi_added = self.parameters["n_ppi_added"]

        # Calculate number of NMPs that should be added when consolidating rRNA
        # molecules
        self.main_23s_rRNA_id = self.parameters["main_23s_rRNA_id"]
        self.main_16s_rRNA_id = self.parameters["main_16s_rRNA_id"]
        self.main_5s_rRNA_id = self.parameters["main_5s_rRNA_id"]

        self.variant_23s_rRNA_ids = self.parameters["variant_23s_rRNA_ids"]
        self.variant_16s_rRNA_ids = self.parameters["variant_16s_rRNA_ids"]
        self.variant_5s_rRNA_ids = self.parameters["variant_5s_rRNA_ids"]

        self.delta_nt_counts_23s = self.parameters["delta_nt_counts_23s"]
        self.delta_nt_counts_16s = self.parameters["delta_nt_counts_16s"]
        self.delta_nt_counts_5s = self.parameters["delta_nt_counts_5s"]

        # Bulk molecule IDs
        self.unprocessed_rna_ids = self.parameters["unprocessed_rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.rna_maturation_enzyme_ids = self.parameters["rna_maturation_enzyme_ids"]
        self.fragment_bases = self.parameters["fragment_bases"]
        self.ppi = self.parameters["ppi"]
        self.water = self.parameters["water"]
        self.nmps = self.parameters["nmps"]
        self.proton = self.parameters["proton"]

        # Numpy indices for bulk molecules
        self.ppi_idx = None


class RnaMaturationRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
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
        timestep = 1.0  # RnaMaturation has no timestep in topology
        p = self.process

        # --- inlined from calculate_request ---
        # Get bulk indices
        if p.ppi_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.unprocessed_rna_idx = bulk_name_to_idx(
                p.unprocessed_rna_ids, bulk_ids
            )
            p.mature_rna_idx = bulk_name_to_idx(p.mature_rna_ids, bulk_ids)
            p.rna_maturation_enzyme_idx = bulk_name_to_idx(
                p.rna_maturation_enzyme_ids, bulk_ids
            )
            p.fragment_base_idx = bulk_name_to_idx(p.fragment_bases, bulk_ids)
            p.ppi_idx = bulk_name_to_idx(p.ppi, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water, bulk_ids)
            p.nmps_idx = bulk_name_to_idx(p.nmps, bulk_ids)
            p.proton_idx = bulk_name_to_idx(p.proton, bulk_ids)
            p.main_23s_rRNA_idx = bulk_name_to_idx(p.main_23s_rRNA_id, bulk_ids)
            p.main_16s_rRNA_idx = bulk_name_to_idx(p.main_16s_rRNA_id, bulk_ids)
            p.main_5s_rRNA_idx = bulk_name_to_idx(p.main_5s_rRNA_id, bulk_ids)
            p.variant_23s_rRNA_idx = bulk_name_to_idx(
                p.variant_23s_rRNA_ids, bulk_ids
            )
            p.variant_16s_rRNA_idx = bulk_name_to_idx(
                p.variant_16s_rRNA_ids, bulk_ids
            )
            p.variant_5s_rRNA_idx = bulk_name_to_idx(
                p.variant_5s_rRNA_ids, bulk_ids
            )

        unprocessed_rna_counts = counts(state["bulk_total"], p.unprocessed_rna_idx)
        variant_23s_rRNA_counts = counts(
            state["bulk_total"], p.variant_23s_rRNA_idx
        )
        variant_16s_rRNA_counts = counts(
            state["bulk_total"], p.variant_16s_rRNA_idx
        )
        variant_5s_rRNA_counts = counts(state["bulk_total"], p.variant_5s_rRNA_idx)
        p.enzyme_availability = counts(
            state["bulk_total"], p.rna_maturation_enzyme_idx
        ).astype(bool)

        # Determine which maturation reactions to turn off based on enzyme
        # availability
        reaction_is_off = (
            p.enzyme_matrix.dot(p.enzyme_availability) < p.n_required_enzymes
        )
        unprocessed_rna_counts[reaction_is_off] = 0

        # Calculate NMPs, water, and proton needed to balance mass
        n_added_bases_from_maturation = np.dot(
            p.degraded_nt_counts.T, unprocessed_rna_counts
        )
        n_added_bases_from_consolidation = (
            p.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + p.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + p.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )
        n_added_bases = n_added_bases_from_maturation + n_added_bases_from_consolidation
        n_total_added_bases = int(n_added_bases.sum())

        # Request all unprocessed RNAs, ppis that need to be added to the
        # 5'-ends of mature RNAs, all variant rRNAs, and NMPs/water/protons
        # needed to balance mass
        request = {
            "bulk": [
                (p.unprocessed_rna_idx, unprocessed_rna_counts),
                (p.ppi_idx, p.n_ppi_added.dot(unprocessed_rna_counts)),
                (p.variant_23s_rRNA_idx, variant_23s_rRNA_counts),
                (p.variant_16s_rRNA_idx, variant_16s_rRNA_counts),
                (p.variant_5s_rRNA_idx, variant_5s_rRNA_counts),
                (p.nmps_idx, np.abs(-n_added_bases).astype(int)),
            ]
        }

        if n_total_added_bases > 0:
            request["bulk"].append((p.water_idx, n_total_added_bases))
        else:
            request["bulk"].append((p.proton_idx, -n_total_added_bases))
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class RnaMaturationEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': SetStore(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
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
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)
        if not self.process.request_set:
            return {}
        timestep = 1.0
        p = self.process

        # --- inlined from evolve_state ---
        # Create copy of bulk counts so can update in real-time
        state["bulk"] = counts(state["bulk"], range(len(state["bulk"])))

        # Get counts of unprocessed RNAs
        unprocessed_rna_counts = counts(state["bulk"], p.unprocessed_rna_idx)

        # Calculate numbers of mature RNAs and fragment bases that are generated
        # upon maturation
        n_mature_rnas = p.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            p.degraded_nt_counts.T, unprocessed_rna_counts
        )

        state["bulk"][p.mature_rna_idx] += n_mature_rnas
        state["bulk"][p.unprocessed_rna_idx] += -unprocessed_rna_counts
        ppi_update = p.n_ppi_added.dot(unprocessed_rna_counts)
        state["bulk"][p.ppi_idx] += -ppi_update
        update = {
            "bulk": [
                (p.mature_rna_idx, n_mature_rnas),
                (p.unprocessed_rna_idx, -unprocessed_rna_counts),
                (p.ppi_idx, -ppi_update),
            ],
            "listeners": {
                "rna_maturation_listener": {
                    "total_maturation_events": unprocessed_rna_counts.sum(),
                    "total_degraded_ntps": n_added_bases_from_maturation.sum(dtype=int),
                    "unprocessed_rnas_consumed": unprocessed_rna_counts,
                    "mature_rnas_generated": n_mature_rnas,
                    "maturation_enzyme_counts": counts(
                        state["bulk_total"], p.rna_maturation_enzyme_idx
                    ),
                }
            },
        }

        # Get counts of variant rRNAs
        variant_23s_rRNA_counts = counts(state["bulk"], p.variant_23s_rRNA_idx)
        variant_16s_rRNA_counts = counts(state["bulk"], p.variant_16s_rRNA_idx)
        variant_5s_rRNA_counts = counts(state["bulk"], p.variant_5s_rRNA_idx)

        # Calculate number of NMPs that should be added to balance out the mass
        # difference during the consolidation
        n_added_bases_from_consolidation = (
            p.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + p.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + p.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        # Evolve states
        update["bulk"].extend(
            [
                (p.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
                (p.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
                (p.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
                (p.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
                (p.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
                (p.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
            ]
        )

        # Consume or add NMPs to balance out mass
        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = n_added_bases.sum()

        update["bulk"].extend(
            [
                (p.nmps_idx, n_added_bases),
                (p.water_idx, -n_total_added_bases),
                (p.proton_idx, n_total_added_bases),
            ]
        )
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
