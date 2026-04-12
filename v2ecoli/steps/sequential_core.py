"""
Sequential core step — biological architecture's resource-arbitration step.

Runs an ordered list of sub-groups of PartitionedProcesses as one Step.
Within each sub-group, proportional reconciliation is applied (same math
as ReconciledStep). Between sub-groups, the running bulk snapshot is
updated with the prior sub-group's deltas before the next sub-group's
request phase — i.e. a flush happens inside the step, not between
scheduler layers.

Effect: equivalent numerics to running each sub-group as its own
ReconciledStep on its own layer (the reconciled architecture) but fewer
scheduler-level layers.

Biological layout
-----------------
  sub-groups = [[rna_degradation], [polypeptide_elongation, transcript_elongation]]
  → rna_degradation releases NMPs and fragment bases first; then the two
    elongations run with the post-degradation bulk view and reconcile
    against each other for any shared consumable pool.
"""

import numpy as np
from bigraph_schema import deep_merge

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.steps.partition import PartitionedProcess
from v2ecoli.steps.reconciled import reconcile_requests


class SequentialCoreStep(Step):
    """Runs ordered sub-groups of PartitionedProcesses as one step.

    Config:
        subgroups: list of lists of PartitionedProcess instances. Each
            inner list is a reconciled group; outer list is execution
            order.
        seed:      RNG seed for the proportional-remainder draws.
    """

    name = 'sequential-core'

    config_schema = {
        'subgroups': 'node',
        'seed': 'integer{0}',
    }

    def _all_processes(self):
        for sg in self.parameters.get('subgroups', []):
            yield from sg

    def inputs(self):
        ports = {}
        for proc in self._all_processes():
            for key, schema in proc.inputs().items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        ports['global_time'] = 'float'
        ports['timestep'] = 'integer'
        ports['next_update_time'] = 'node'
        return ports

    def outputs(self):
        ports = {}
        for proc in self._all_processes():
            for key, schema in proc.outputs().items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        ports['next_update_time'] = 'node'
        return ports

    def initialize(self, config):
        subgroups = self.parameters.get('subgroups', [])
        for sg in subgroups:
            assert all(isinstance(p, PartitionedProcess) for p in sg)
        flat_names = [p.name for sg in subgroups for p in sg]
        self.parameters['name'] = f"sequential_{'_'.join(flat_names)}"
        self.random_state = np.random.RandomState(
            seed=self.parameters.get('seed', 0))

    def port_defaults(self):
        merged = {}
        for proc in self._all_processes():
            if hasattr(proc, 'port_defaults'):
                defaults = proc.port_defaults()
                if defaults:
                    merged = deep_merge(merged, defaults)
        return merged

    def update_condition(self, timestep, states):
        nut = states.get('next_update_time', {})
        gt = states.get('global_time', 0)
        for proc in self._all_processes():
            if nut.get(proc.name, 0) <= gt:
                return True
        return False

    def update(self, states, interval=None):
        subgroups = self.parameters.get('subgroups', [])
        timestep = states.get('timestep', 1)
        global_time = states.get('global_time', 0)
        nut = states.get('next_update_time', {})
        bulk = states.get('bulk')

        if bulk is not None and hasattr(bulk, 'dtype') and \
                'count' in bulk.dtype.names:
            running_bulk = bulk.copy()
            running_counts = running_bulk['count']
        else:
            running_bulk = None
            running_counts = np.array([], dtype=np.int64)

        combined_bulk = []
        combined_listeners = {}
        combined_other = {}
        nut_update = {}

        for subgroup in subgroups:
            # Skip processes whose next_update_time is in the future.
            active = [p for p in subgroup
                      if nut.get(p.name, 0) <= global_time]
            if not active:
                continue

            # ------------------------------------------------------------
            # Request phase: each process calculates its bulk requests
            # against the CURRENT running bulk (which reflects prior
            # sub-groups' deltas).
            # ------------------------------------------------------------
            proc_req_dicts = []
            proc_non_bulk = []
            for proc in active:
                proc_states = dict(states)
                if running_bulk is not None:
                    proc_states['bulk'] = running_bulk

                request = proc.calculate_request(timestep, proc_states)
                bulk_reqs = request.pop('bulk', []) or []

                req_dict = {}
                for idx_arr, count_arr in bulk_reqs:
                    idx_arr = np.atleast_1d(idx_arr)
                    count_arr = np.atleast_1d(count_arr)
                    for idx, cnt in zip(idx_arr, count_arr):
                        req_dict[int(idx)] = req_dict.get(
                            int(idx), 0) + int(cnt)

                proc_req_dicts.append(req_dict)
                proc_non_bulk.append(request)

            # ------------------------------------------------------------
            # Reconcile phase: proportional scaling against running supply.
            # ------------------------------------------------------------
            allocations = reconcile_requests(
                proc_req_dicts, running_counts, self.random_state)

            # ------------------------------------------------------------
            # Evolve phase: each process runs against its allocation.
            # Apply its bulk deltas to running_bulk so the NEXT sub-group
            # sees them.
            # ------------------------------------------------------------
            for i, proc in enumerate(active):
                alloc_array = np.zeros(len(running_counts), dtype=np.int64)
                for mol_idx, count in allocations[i].items():
                    alloc_array[mol_idx] = count

                evolve_states = dict(states)
                if running_bulk is not None:
                    evolve_states['bulk'] = alloc_array
                    # Preserve bulk_total = original (for processes that
                    # read it for concentration calculations).
                else:
                    evolve_states['bulk'] = alloc_array
                evolve_states = deep_merge(evolve_states, proc_non_bulk[i])

                update = proc.evolve_state(timestep, evolve_states)

                if 'listeners' in proc_non_bulk[i]:
                    lreq = proc_non_bulk[i]['listeners']
                    if 'listeners' in update:
                        update['listeners'] = deep_merge(
                            update['listeners'], lreq)
                    else:
                        update['listeners'] = lreq

                # Apply bulk deltas to running_bulk for downstream sub-groups.
                bulk_deltas = update.pop('bulk', []) or []
                if running_bulk is not None:
                    for idx_arr, delta_arr in bulk_deltas:
                        idx_arr = np.atleast_1d(idx_arr)
                        delta_arr = np.atleast_1d(delta_arr).astype(np.int64)
                        np.add.at(running_counts, idx_arr, delta_arr)
                        combined_bulk.append((idx_arr, delta_arr))

                listener_update = update.pop('listeners', {})
                if listener_update:
                    combined_listeners = deep_merge(
                        combined_listeners, listener_update)

                if update:
                    combined_other = deep_merge(combined_other, update)

                nut_update[proc.name] = global_time + timestep

        # Safety: clamp negative final counts (same invariant as
        # ReconciledStep._clamp_bulk_deltas).
        if combined_bulk and running_bulk is not None:
            neg = running_counts < 0
            if neg.any():
                correction = np.where(neg, -running_counts, 0).astype(np.int64)
                ci = np.where(correction != 0)[0]
                if len(ci) > 0:
                    combined_bulk.append((ci, correction[ci]))

        result = dict(combined_other)
        if combined_bulk:
            result['bulk'] = combined_bulk
        if combined_listeners:
            result['listeners'] = combined_listeners
        result['next_update_time'] = nut_update
        return result
