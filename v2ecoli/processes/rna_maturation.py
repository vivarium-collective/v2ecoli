"""
RnaMaturation process
=====================
- Converts unprocessed tRNA/rRNA molecules into mature tRNA/rRNAs
- Consolidates the different variants of 23S, 16S, and 5S rRNAs into the single
  variant that is used for ribosomal subunits
"""

import numpy as np

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import listener_schema, numpy_schema, counts, bulk_name_to_idx
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


class RnaMaturationLogic:
    """Biological logic for RNA maturation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**RnaMaturation.defaults, **(parameters or {})}
        parameters = self.parameters

        # Get matrices and vectors that describe maturation reactions
        self.stoich_matrix = parameters["stoich_matrix"]
        self.enzyme_matrix = parameters["enzyme_matrix"]
        self.n_required_enzymes = parameters["n_required_enzymes"]
        self.degraded_nt_counts = parameters["degraded_nt_counts"]
        self.n_ppi_added = parameters["n_ppi_added"]

        # rRNA variant IDs
        self.main_23s_rRNA_id = parameters["main_23s_rRNA_id"]
        self.main_16s_rRNA_id = parameters["main_16s_rRNA_id"]
        self.main_5s_rRNA_id = parameters["main_5s_rRNA_id"]

        self.variant_23s_rRNA_ids = parameters["variant_23s_rRNA_ids"]
        self.variant_16s_rRNA_ids = parameters["variant_16s_rRNA_ids"]
        self.variant_5s_rRNA_ids = parameters["variant_5s_rRNA_ids"]

        self.delta_nt_counts_23s = parameters["delta_nt_counts_23s"]
        self.delta_nt_counts_16s = parameters["delta_nt_counts_16s"]
        self.delta_nt_counts_5s = parameters["delta_nt_counts_5s"]

        # Bulk molecule IDs
        self.unprocessed_rna_ids = parameters["unprocessed_rna_ids"]
        self.mature_rna_ids = parameters["mature_rna_ids"]
        self.rna_maturation_enzyme_ids = parameters["rna_maturation_enzyme_ids"]
        self.fragment_bases = parameters["fragment_bases"]
        self.ppi = parameters["ppi"]
        self.water = parameters["water"]
        self.nmps = parameters["nmps"]
        self.proton = parameters["proton"]

        # Numpy indices for bulk molecules (lazy init)
        self.ppi_idx = None

    def _init_indices(self, bulk_ids):
        if self.ppi_idx is None:
            self.unprocessed_rna_idx = bulk_name_to_idx(
                self.unprocessed_rna_ids, bulk_ids
            )
            self.mature_rna_idx = bulk_name_to_idx(self.mature_rna_ids, bulk_ids)
            self.rna_maturation_enzyme_idx = bulk_name_to_idx(
                self.rna_maturation_enzyme_ids, bulk_ids
            )
            self.fragment_base_idx = bulk_name_to_idx(self.fragment_bases, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
            self.nmps_idx = bulk_name_to_idx(self.nmps, bulk_ids)
            self.proton_idx = bulk_name_to_idx(self.proton, bulk_ids)
            self.main_23s_rRNA_idx = bulk_name_to_idx(self.main_23s_rRNA_id, bulk_ids)
            self.main_16s_rRNA_idx = bulk_name_to_idx(self.main_16s_rRNA_id, bulk_ids)
            self.main_5s_rRNA_idx = bulk_name_to_idx(self.main_5s_rRNA_id, bulk_ids)
            self.variant_23s_rRNA_idx = bulk_name_to_idx(
                self.variant_23s_rRNA_ids, bulk_ids
            )
            self.variant_16s_rRNA_idx = bulk_name_to_idx(
                self.variant_16s_rRNA_ids, bulk_ids
            )
            self.variant_5s_rRNA_idx = bulk_name_to_idx(
                self.variant_5s_rRNA_ids, bulk_ids
            )


class RnaMaturationRequester(Step):
    """Requester step for RNA maturation.

    Requests unprocessed RNAs, ppi, variant rRNAs, and NMPs/water/protons
    needed for maturation and consolidation.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = RnaMaturationLogic(config)
        self.process_name = 'ecoli-rna-maturation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        unprocessed_rna_counts = counts(state['bulk_total'], proc.unprocessed_rna_idx)
        variant_23s_rRNA_counts = counts(
            state['bulk_total'], proc.variant_23s_rRNA_idx
        )
        variant_16s_rRNA_counts = counts(
            state['bulk_total'], proc.variant_16s_rRNA_idx
        )
        variant_5s_rRNA_counts = counts(state['bulk_total'], proc.variant_5s_rRNA_idx)
        enzyme_availability = counts(
            state['bulk_total'], proc.rna_maturation_enzyme_idx
        ).astype(bool)

        # Determine which maturation reactions to turn off based on enzyme
        # availability
        reaction_is_off = (
            proc.enzyme_matrix.dot(enzyme_availability) < proc.n_required_enzymes
        )
        unprocessed_rna_counts[reaction_is_off] = 0

        # Calculate NMPs, water, and proton needed to balance mass
        n_added_bases_from_maturation = np.dot(
            proc.degraded_nt_counts.T, unprocessed_rna_counts
        )
        n_added_bases_from_consolidation = (
            proc.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + proc.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + proc.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )
        n_added_bases = n_added_bases_from_maturation + n_added_bases_from_consolidation
        n_total_added_bases = int(n_added_bases.sum())

        # Request all unprocessed RNAs, ppis, variant rRNAs, and NMPs/water/protons
        bulk_request = [
            (proc.unprocessed_rna_idx, unprocessed_rna_counts),
            (proc.ppi_idx, proc.n_ppi_added.dot(unprocessed_rna_counts)),
            (proc.variant_23s_rRNA_idx, variant_23s_rRNA_counts),
            (proc.variant_16s_rRNA_idx, variant_16s_rRNA_counts),
            (proc.variant_5s_rRNA_idx, variant_5s_rRNA_counts),
            (proc.nmps_idx, np.abs(-n_added_bases).astype(int)),
        ]

        if n_total_added_bases > 0:
            bulk_request.append((proc.water_idx, n_total_added_bases))
        else:
            bulk_request.append((proc.proton_idx, -n_total_added_bases))

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class RnaMaturationEvolver(Step):
    """Evolver step for RNA maturation.

    Reads allocated bulk molecules, matures RNAs, consolidates
    rRNA variants, and balances mass with NMPs.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = RnaMaturationLogic(config)
        self.process_name = 'ecoli-rna-maturation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'listeners': InPlaceDict(),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        # Create copy of bulk counts so can update in real-time
        bulk_counts = counts(state['bulk'], range(len(state['bulk'])))

        # Get counts of unprocessed RNAs
        unprocessed_rna_counts = bulk_counts[proc.unprocessed_rna_idx]

        # Calculate numbers of mature RNAs and fragment bases that are generated
        # upon maturation
        n_mature_rnas = proc.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            proc.degraded_nt_counts.T, unprocessed_rna_counts
        )

        bulk_counts[proc.mature_rna_idx] += n_mature_rnas
        bulk_counts[proc.unprocessed_rna_idx] += -unprocessed_rna_counts
        ppi_update = proc.n_ppi_added.dot(unprocessed_rna_counts)
        bulk_counts[proc.ppi_idx] += -ppi_update

        update = {
            'bulk': [
                (proc.mature_rna_idx, n_mature_rnas),
                (proc.unprocessed_rna_idx, -unprocessed_rna_counts),
                (proc.ppi_idx, -ppi_update),
            ],
            'listeners': {
                'rna_maturation_listener': {
                    'total_maturation_events': unprocessed_rna_counts.sum(),
                    'total_degraded_ntps': n_added_bases_from_maturation.sum(dtype=int),
                    'unprocessed_rnas_consumed': unprocessed_rna_counts,
                    'mature_rnas_generated': n_mature_rnas,
                    'maturation_enzyme_counts': counts(
                        state['bulk_total'], proc.rna_maturation_enzyme_idx
                    ),
                }
            },
        }

        # Get counts of variant rRNAs
        variant_23s_rRNA_counts = bulk_counts[proc.variant_23s_rRNA_idx]
        variant_16s_rRNA_counts = bulk_counts[proc.variant_16s_rRNA_idx]
        variant_5s_rRNA_counts = bulk_counts[proc.variant_5s_rRNA_idx]

        # Calculate number of NMPs that should be added to balance out the mass
        # difference during the consolidation
        n_added_bases_from_consolidation = (
            proc.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + proc.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + proc.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        # Evolve states
        update['bulk'].extend(
            [
                (proc.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
                (proc.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
                (proc.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
                (proc.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
                (proc.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
                (proc.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
            ]
        )

        # Consume or add NMPs to balance out mass
        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = n_added_bases.sum()

        update['bulk'].extend(
            [
                (proc.nmps_idx, n_added_bases),
                (proc.water_idx, -n_total_added_bases),
                (proc.proton_idx, n_total_added_bases),
            ]
        )

        timestep = state.get('timestep', 1.0)
        update['next_update_time'] = state.get('global_time', 0) + timestep

        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class RnaMaturation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

    name = "ecoli-rna-maturation"
    topology = {"bulk": ("bulk",), "bulk_total": ("bulk",), "listeners": ("listeners",)}

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = RnaMaturationLogic(self.parameters)

    def ports_schema(self):
        proc = self._logic
        return {
            "bulk": numpy_schema("bulk"),
            "bulk_total": numpy_schema("bulk"),
            "listeners": {
                "rna_maturation_listener": listener_schema(
                    {
                        "total_maturation_events": 0,
                        "total_degraded_ntps": 0,
                        "unprocessed_rnas_consumed": (
                            [0] * len(proc.unprocessed_rna_ids),
                            proc.unprocessed_rna_ids,
                        ),
                        "mature_rnas_generated": (
                            [0] * len(proc.mature_rna_ids),
                            proc.mature_rna_ids,
                        ),
                        "maturation_enzyme_counts": (
                            [0] * len(proc.rna_maturation_enzyme_ids),
                            proc.rna_maturation_enzyme_ids,
                        ),
                    }
                )
            },
        }

    def calculate_request(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])

        unprocessed_rna_counts = counts(states["bulk_total"], proc.unprocessed_rna_idx)
        variant_23s_rRNA_counts = counts(
            states["bulk_total"], proc.variant_23s_rRNA_idx
        )
        variant_16s_rRNA_counts = counts(
            states["bulk_total"], proc.variant_16s_rRNA_idx
        )
        variant_5s_rRNA_counts = counts(states["bulk_total"], proc.variant_5s_rRNA_idx)
        self.enzyme_availability = counts(
            states["bulk_total"], proc.rna_maturation_enzyme_idx
        ).astype(bool)

        reaction_is_off = (
            proc.enzyme_matrix.dot(self.enzyme_availability) < proc.n_required_enzymes
        )
        unprocessed_rna_counts[reaction_is_off] = 0

        n_added_bases_from_maturation = np.dot(
            proc.degraded_nt_counts.T, unprocessed_rna_counts
        )
        n_added_bases_from_consolidation = (
            proc.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + proc.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + proc.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )
        n_added_bases = n_added_bases_from_maturation + n_added_bases_from_consolidation
        n_total_added_bases = int(n_added_bases.sum())

        request = {
            "bulk": [
                (proc.unprocessed_rna_idx, unprocessed_rna_counts),
                (proc.ppi_idx, proc.n_ppi_added.dot(unprocessed_rna_counts)),
                (proc.variant_23s_rRNA_idx, variant_23s_rRNA_counts),
                (proc.variant_16s_rRNA_idx, variant_16s_rRNA_counts),
                (proc.variant_5s_rRNA_idx, variant_5s_rRNA_counts),
                (proc.nmps_idx, np.abs(-n_added_bases).astype(int)),
            ]
        }

        if n_total_added_bases > 0:
            request["bulk"].append((proc.water_idx, n_total_added_bases))
        else:
            request["bulk"].append((proc.proton_idx, -n_total_added_bases))

        return request

    def evolve_state(self, timestep, states):
        proc = self._logic

        states["bulk"] = counts(states["bulk"], range(len(states["bulk"])))
        unprocessed_rna_counts = counts(states["bulk"], proc.unprocessed_rna_idx)

        n_mature_rnas = proc.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            proc.degraded_nt_counts.T, unprocessed_rna_counts
        )

        states["bulk"][proc.mature_rna_idx] += n_mature_rnas
        states["bulk"][proc.unprocessed_rna_idx] += -unprocessed_rna_counts
        ppi_update = proc.n_ppi_added.dot(unprocessed_rna_counts)
        states["bulk"][proc.ppi_idx] += -ppi_update

        update = {
            "bulk": [
                (proc.mature_rna_idx, n_mature_rnas),
                (proc.unprocessed_rna_idx, -unprocessed_rna_counts),
                (proc.ppi_idx, -ppi_update),
            ],
            "listeners": {
                "rna_maturation_listener": {
                    "total_maturation_events": unprocessed_rna_counts.sum(),
                    "total_degraded_ntps": n_added_bases_from_maturation.sum(dtype=int),
                    "unprocessed_rnas_consumed": unprocessed_rna_counts,
                    "mature_rnas_generated": n_mature_rnas,
                    "maturation_enzyme_counts": counts(
                        states["bulk_total"], proc.rna_maturation_enzyme_idx
                    ),
                }
            },
        }

        variant_23s_rRNA_counts = counts(states["bulk"], proc.variant_23s_rRNA_idx)
        variant_16s_rRNA_counts = counts(states["bulk"], proc.variant_16s_rRNA_idx)
        variant_5s_rRNA_counts = counts(states["bulk"], proc.variant_5s_rRNA_idx)

        n_added_bases_from_consolidation = (
            proc.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + proc.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + proc.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        update["bulk"].extend(
            [
                (proc.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
                (proc.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
                (proc.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
                (proc.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
                (proc.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
                (proc.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
            ]
        )

        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = n_added_bases.sum()

        update["bulk"].extend(
            [
                (proc.nmps_idx, n_added_bases),
                (proc.water_idx, -n_total_added_bases),
                (proc.proton_idx, n_total_added_bases),
            ]
        )

        return update
