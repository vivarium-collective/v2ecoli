"""
Allocator step for v2ecoli.

Reads requests from PartitionedProcesses and allocates molecules
according to process priorities. Proper process-bigraph Step.
"""

import numpy as np
from process_bigraph import Step
from bigraph_schema.schema import Node, Overwrite

from v2ecoli.library.schema import counts, bulk_name_to_idx, listener_schema
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate


ASSERT_POSITIVE_COUNTS = True


class NegativeCountsError(Exception):
    pass


class Allocator(Step):
    """Allocator — arbitrates bulk molecule allocation."""

    name = "allocator"
    topology = {
        'request': ('request',),
        'allocate': ('allocate',),
        'bulk': ('bulk',),
        'listeners': ('listeners',),
        'allocator_rng': ('allocator_rng',),
    }
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        params = config or {}
        self.parameters = params
        self.moleculeNames = params.get("molecule_names", [])
        self.n_molecules = len(self.moleculeNames)
        self.mol_name_to_idx = {
            name: idx for idx, name in enumerate(self.moleculeNames)}
        self.mol_idx_to_name = {
            idx: name for idx, name in enumerate(self.moleculeNames)}
        self.processNames = params.get("process_names", [])
        self.n_processes = len(self.processNames)
        self.proc_name_to_idx = {
            name: idx for idx, name in enumerate(self.processNames)}
        self.proc_idx_to_name = {
            idx: name for idx, name in enumerate(self.processNames)}
        self.processPriorities = np.zeros(len(self.processNames))
        for process, custom_priority in params.get("custom_priorities", {}).items():
            if process in self.proc_name_to_idx:
                self.processPriorities[self.proc_name_to_idx[process]] = custom_priority
        self.seed = params.get("seed", 0)
        self.molecule_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'request': Node(),
            'listeners': Node(),
            'allocator_rng': Node(),
        }

    def outputs(self):
        return {
            'request': Overwrite(_value=Node()),
            'allocate': Overwrite(_value=Node()),
            'listeners': Node(),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, state["bulk"]["id"])
            self.atp_idx = bulk_name_to_idx("ATP[c]", state["bulk"]["id"])

        total_counts = counts(state["bulk"], self.molecule_idx)
        original_totals = total_counts.copy()
        counts_requested = np.zeros((self.n_molecules, self.n_processes), dtype=int)

        proc_idx_in_layer = []
        for process in state.get("request", {}):
            if process not in self.proc_name_to_idx:
                continue
            proc_idx = self.proc_name_to_idx[process]
            req_bulk = state["request"][process].get("bulk", [])
            if len(req_bulk) > 0:
                proc_idx_in_layer.append(proc_idx)
            for req_idx, req in req_bulk:
                counts_requested[req_idx, proc_idx] += req

        if ASSERT_POSITIVE_COUNTS and np.any(counts_requested < 0):
            raise NegativeCountsError("Negative counts_requested")

        partitioned_counts = calculate_partition(
            self.processPriorities, counts_requested,
            total_counts, state.get("allocator_rng", np.random.RandomState(self.seed)))
        partitioned_counts.astype(int, copy=False)

        # ATP listener
        non_zero_mask = counts_requested[self.atp_idx, :] != 0
        curr_atp_req = np.array(
            state.get("listeners", {}).get("atp", {}).get(
                "atp_requested", [0] * self.n_processes)).copy()
        curr_atp_alloc = np.array(
            state.get("listeners", {}).get("atp", {}).get(
                "atp_allocated_initial", [0] * self.n_processes)).copy()
        curr_atp_req[non_zero_mask] = counts_requested[self.atp_idx, non_zero_mask]
        curr_atp_alloc[non_zero_mask] = partitioned_counts[self.atp_idx, non_zero_mask]

        return {
            "request": {process: {"bulk": []} for process in state.get("request", {})},
            "allocate": {
                process: {"bulk": partitioned_counts[:, self.proc_name_to_idx[process]]}
                for process in state.get("request", {})
                if process in self.proc_name_to_idx},
            "listeners": {"atp": {
                "atp_requested": curr_atp_req,
                "atp_allocated_initial": curr_atp_alloc}},
        }


def calculate_partition(process_priorities, counts_requested, total_counts, random_state):
    """Partition molecules across processes by priority."""
    priorityLevels = np.sort(np.unique(process_priorities))[::-1]
    partitioned_counts = np.zeros_like(counts_requested)

    for priorityLevel in priorityLevels:
        processHasPriority = priorityLevel == process_priorities
        requests = counts_requested[:, processHasPriority].copy()
        total_requested = requests.sum(axis=1)
        excess_request_mask = (total_requested > total_counts) & (total_requested > 0)

        fractional_requests = (
            requests[excess_request_mask, :]
            * total_counts[excess_request_mask, np.newaxis]
            / total_requested[excess_request_mask, np.newaxis])

        remainders = fractional_requests % 1
        options = np.arange(remainders.shape[1])
        for idx, remainder in enumerate(remainders):
            total_remainder = remainder.sum()
            count = int(np.round(total_remainder))
            if count > 0:
                allocated_indices = random_state.choice(
                    options, size=count,
                    p=remainder / total_remainder, replace=False)
                fractional_requests[idx, allocated_indices] += 1
        requests[excess_request_mask, :] = fractional_requests

        allocations = requests.astype(np.int64)
        partitioned_counts[:, processHasPriority] = allocations
        total_counts -= allocations.sum(axis=1)

    return partitioned_counts
