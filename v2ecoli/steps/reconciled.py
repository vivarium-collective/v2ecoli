"""
=============================
Reconciled Step (Departitioned)
=============================

Wraps a group of PartitionedProcesses so they run as a single Step
with reconcile-based allocation. Each process computes its request;
the requests are reconciled against available bulk counts using
proportional scaling (matching the Allocator's fairness logic but
without the separate Requester/Allocator/Evolver machinery).

This sits between the full partitioned architecture (3 steps per process
+ allocator) and the naive departitioned architecture (each process takes
what it wants, no mediation). It provides the same fairness guarantees
as the Allocator while being a single step per layer.

Relationship to bigraph-schema reconcile:
    reconcile([request_1, request_2, ...]) produces a single combined
    allocation that respects available supply. This is the same principle
    as bigraph-schema's reconcile — combining multiple updates into one
    consistent update — but applied to resource allocation rather than
    state deltas.
"""

import warnings

import numpy as np
from bigraph_schema import deep_merge

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.steps.partition import PartitionedProcess


def reconcile_requests(requests_per_process, total_counts, random_state=None):
    """Reconcile bulk molecule requests from multiple processes.

    Given a list of per-process requests and available molecule counts,
    distribute molecules proportionally when total requests exceed supply.
    This matches the Allocator's calculatePartition logic (at a single
    priority level) but expressed as a reconciliation operation:

        reconcile([req_1, req_2, ...], available) -> [alloc_1, alloc_2, ...]

    Args:
        requests_per_process: List of dicts mapping molecule_index -> count.
            Each dict is one process's request.
        total_counts: 1D array of available molecule counts.
        random_state: Optional numpy RandomState for remainder distribution.

    Returns:
        List of dicts mapping molecule_index -> allocated_count, one per
        process. Sum of allocations <= total_counts for each molecule.
    """
    if not requests_per_process:
        return []

    n_procs = len(requests_per_process)

    # Collect all molecule indices that were requested
    all_indices = set()
    for req in requests_per_process:
        all_indices.update(req.keys())

    if not all_indices:
        return [{} for _ in range(n_procs)]

    all_indices = sorted(all_indices)
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
    n_mols = len(all_indices)

    # Build request matrix: (n_molecules, n_processes)
    req_matrix = np.zeros((n_mols, n_procs), dtype=np.int64)
    for p, req in enumerate(requests_per_process):
        for mol_idx, count in req.items():
            req_matrix[idx_to_pos[mol_idx], p] = count

    # Get available counts for these molecules
    avail = np.array([int(total_counts[idx]) for idx in all_indices],
                     dtype=np.int64)

    # Reconcile: proportional scaling where requests exceed supply
    total_requested = req_matrix.sum(axis=1)
    excess_mask = (total_requested > avail) & (total_requested > 0)

    alloc_matrix = req_matrix.copy()

    if excess_mask.any():
        # Scale down proportionally where there's scarcity
        fractional = (
            req_matrix[excess_mask, :].astype(np.float64)
            * avail[excess_mask, np.newaxis]
            / total_requested[excess_mask, np.newaxis]
        )

        # Distribute remainders randomly (matching Allocator behavior)
        remainders = fractional % 1
        options = np.arange(n_procs)
        rng = random_state or np.random.RandomState()
        for i, remainder in enumerate(remainders):
            total_remainder = remainder.sum()
            count = int(np.round(total_remainder))
            if count > 0 and total_remainder > 0:
                count = min(count, len(options))
                allocated = rng.choice(
                    options, size=count,
                    p=remainder / total_remainder,
                    replace=False)
                fractional[i, allocated] += 1

        alloc_matrix[excess_mask, :] = fractional.astype(np.int64)

    # Convert back to per-process dicts
    allocations = []
    for p in range(n_procs):
        alloc = {}
        for pos, mol_idx in enumerate(all_indices):
            if alloc_matrix[pos, p] > 0 or req_matrix[pos, p] > 0:
                alloc[mol_idx] = int(alloc_matrix[pos, p])
        allocations.append(alloc)

    return allocations


class ReconciledStep(Step):
    """Single step that reconciles requests from multiple PartitionedProcesses.

    Given a list of PartitionedProcess objects (e.g. one allocator layer),
    this step:
    1. Calls calculate_request on each process
    2. Reconciles bulk requests proportionally against available supply
    3. Calls evolve_state on each with reconciled allocations
    4. Merges all returned updates

    This replaces both the DepartitionedStep wrappers AND the Allocator
    for a group of processes, producing results equivalent to the full
    partitioned architecture.
    """

    config_schema = {
        'processes': 'node',
        'seed': 'integer{0}',
        'evolve_only': 'list[string]',  # process names that skip request
    }

    def inputs(self):
        # Deep union of all process inputs
        ports = {}
        for proc in self.parameters.get('processes', []):
            proc_inputs = proc.inputs()
            for key, schema in proc_inputs.items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        ports['global_time'] = 'float'
        ports['timestep'] = 'integer'
        ports['next_update_time'] = 'node'
        return ports

    def outputs(self):
        # Deep union of all process outputs
        ports = {}
        for proc in self.parameters.get('processes', []):
            proc_outputs = proc.outputs()
            for key, schema in proc_outputs.items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        ports['next_update_time'] = 'node'
        return ports

    def initialize(self, config):
        processes = self.parameters.get('processes', [])
        assert all(isinstance(p, PartitionedProcess) for p in processes)
        names = [p.name for p in processes]
        self.parameters['name'] = f"reconciled_{'_'.join(names)}"
        self.random_state = np.random.RandomState(
            seed=self.parameters.get('seed', 0))
        self.evolve_only_names = set(
            self.parameters.get('evolve_only', []))

    def port_defaults(self):
        """Merge port_defaults from all wrapped processes."""
        merged = {}
        for proc in self.parameters.get('processes', []):
            if hasattr(proc, 'port_defaults'):
                defaults = proc.port_defaults()
                if defaults:
                    from bigraph_schema import deep_merge
                    merged = deep_merge(merged, defaults)
        return merged

    def update_condition(self, timestep, states):
        """Run if any wrapped process is due."""
        nut = states.get('next_update_time', {})
        gt = states.get('global_time', 0)
        for proc in self.parameters.get('processes', []):
            proc_nut = nut.get(proc.name, 0)
            if proc_nut <= gt:
                return True
        return False

    def update(self, states, interval=None):
        processes = self.parameters.get('processes', [])
        timestep = states.get('timestep', 1)
        global_time = states.get('global_time', 0)
        nut = states.get('next_update_time', {})
        bulk = states.get('bulk')

        # Separate active processes into reconciled vs evolve-only
        reconciled_procs = []   # need request + reconcile + evolve
        evolve_only_procs = []  # skip request, run evolve with full bulk

        for proc in processes:
            proc_nut = nut.get(proc.name, 0)
            if proc_nut > global_time:
                continue
            if proc.name in self.evolve_only_names:
                evolve_only_procs.append(proc)
            else:
                reconciled_procs.append(proc)

        if not reconciled_procs and not evolve_only_procs:
            return {}

        # Get bulk counts for reconciliation
        if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
            total_counts = bulk['count'].copy()
        elif bulk is not None:
            total_counts = np.array(bulk, dtype=np.int64)
        else:
            total_counts = np.array([], dtype=np.int64)

        # Phase 1: Collect requests from reconciled processes
        proc_requests = []
        proc_bulk_requests = []

        for proc in reconciled_procs:
            request = proc.calculate_request(timestep, states)

            # Extract bulk requests into {idx: count} dict
            bulk_req_list = request.pop('bulk', [])
            bulk_req_dict = {}
            for idx_arr, count_arr in bulk_req_list:
                idx_arr = np.atleast_1d(idx_arr)
                count_arr = np.atleast_1d(count_arr)
                for idx, cnt in zip(idx_arr, count_arr):
                    bulk_req_dict[int(idx)] = bulk_req_dict.get(
                        int(idx), 0) + int(cnt)

            proc_requests.append(request)
            proc_bulk_requests.append(bulk_req_dict)

        # Phase 2: Reconcile bulk requests against available supply
        allocations = reconcile_requests(
            proc_bulk_requests, total_counts, self.random_state)

        # Phase 3: Run all processes
        combined_bulk = []
        combined_listeners = {}
        combined_other = {}
        nut_update = {}

        # 3a: Reconciled processes — evolve with allocated bulk
        for i, proc in enumerate(reconciled_procs):
            proc_states = dict(states)
            alloc_array = np.zeros(len(total_counts), dtype=np.int64)
            for mol_idx, count in allocations[i].items():
                alloc_array[mol_idx] = count
            proc_states['bulk'] = alloc_array

            # Merge non-bulk request outputs into states
            proc_states = deep_merge(proc_states, proc_requests[i])

            # Run evolve
            update = proc.evolve_state(timestep, proc_states)

            # Merge listener updates from request phase
            if 'listeners' in proc_requests[i]:
                if 'listeners' in update:
                    update['listeners'] = deep_merge(
                        update['listeners'], proc_requests[i]['listeners'])
                else:
                    update['listeners'] = proc_requests[i]['listeners']

            self._accumulate_update(
                update, combined_bulk, combined_listeners, combined_other)
            nut_update[proc.name] = global_time + timestep

        # 3b: Evolve-only processes — skip request, use _do_update with
        #     full bulk (like departitioned, but within the reconciled step)
        for proc in evolve_only_procs:
            update = proc._do_update(timestep, dict(states))

            self._accumulate_update(
                update, combined_bulk, combined_listeners, combined_other)
            nut_update[proc.name] = global_time + timestep

        # Phase 4: Reconcile combined bulk deltas to prevent negative counts
        if combined_bulk and len(total_counts) > 0:
            combined_bulk = self._clamp_bulk_deltas(
                combined_bulk, total_counts)

        # Assemble final update
        result = dict(combined_other)
        if combined_bulk:
            result['bulk'] = combined_bulk
        if combined_listeners:
            result['listeners'] = combined_listeners
        result['next_update_time'] = nut_update
        return result

    @staticmethod
    def _clamp_bulk_deltas(bulk_deltas, current_counts):
        """Reconcile bulk deltas so no molecule count goes negative.

        Sums all deltas per molecule index. If the net delta would push
        a count below zero, scales back the negative (consumption) deltas
        proportionally — the same fairness principle as the Allocator.
        """
        # Accumulate net delta per molecule index
        net = np.zeros(len(current_counts), dtype=np.float64)
        for idx_arr, delta_arr in bulk_deltas:
            idx_arr = np.atleast_1d(idx_arr)
            delta_arr = np.atleast_1d(delta_arr).astype(np.float64)
            np.add.at(net, idx_arr, delta_arr)

        # Find molecules that would go negative
        projected = current_counts.astype(np.float64) + net
        negative_mask = projected < 0

        if not negative_mask.any():
            return bulk_deltas  # no clamping needed

        # For negative molecules, scale the net delta to exactly zero out
        # the count (consume everything available, no more)
        clamped_net = net.copy()
        clamped_net[negative_mask] = -current_counts[negative_mask].astype(
            np.float64)

        # Rebuild bulk deltas with per-entry scaling
        # For efficiency, apply a correction delta instead of rewriting
        correction = clamped_net - net
        correction_indices = np.where(correction != 0)[0]

        if len(correction_indices) > 0:
            bulk_deltas = list(bulk_deltas)
            bulk_deltas.append(
                (correction_indices, correction[correction_indices].astype(
                    np.int64)))

        return bulk_deltas

    @staticmethod
    def _accumulate_update(update, combined_bulk, combined_listeners,
                           combined_other):
        """Split an update dict and accumulate into running totals."""
        bulk_deltas = update.pop('bulk', [])
        if isinstance(bulk_deltas, list):
            combined_bulk.extend(bulk_deltas)

        listener_update = update.pop('listeners', {})
        if listener_update:
            combined_listeners.update(
                deep_merge(combined_listeners, listener_update))

        if update:
            combined_other.update(
                deep_merge(combined_other, update))
