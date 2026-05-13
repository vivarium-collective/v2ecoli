"""Baseline whole-cell E. coli composite (55 processes, partitioned).

Upstream-parity architecture: the partitioned model matches the
vivarium-collective/vEcoli composite tick-for-tick. See AGENTS.md.

Migration note: the document-building body was migrated from
``v2ecoli/generate.py:build_document`` and
``v2ecoli/composite.py:_build_from_cache``.  Both legacy files still exist
and are unchanged; they will be deleted in Task 14.

Helper functions (``make_edge``, ``inject_flow_dependencies``,
``_seed_state_from_defaults``, ``_seed_mass_listener``,
``_normalize_boundary_units``, ``_make_instance``, ``_get_special_step``,
``_expand_flushes``, module-level constants) are imported from
``v2ecoli.generate`` rather than duplicated here because
``generate_departitioned.py`` and ``generate_reconciled.py`` import the same
helpers.  Task 4 will resolve the shared-helpers concern when it migrates
those two files.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle

# ---------------------------------------------------------------------------
# Re-use shared helpers and constants from generate.py.
# generate_departitioned.py and generate_reconciled.py import the same names,
# so duplication here would conflict in Task 14 when generate.py is deleted.
# Task 4 is responsible for deciding the final home for these helpers.
# ---------------------------------------------------------------------------
from v2ecoli.generate import (
    # helpers
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    _seed_mass_listener,
    _normalize_boundary_units,
    _get_step_config,
    _expand_flushes,
    build_execution_layers,
    # constants
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
    DEFAULT_FEATURES,
)

# Convenience re-export: ordered list of all step names in execution order.
# Computed once at module import so callers don't need to import from
# v2ecoli.generate directly.
FLOW_ORDER = [step for layer in build_execution_layers(DEFAULT_FEATURES) for step in layer]


@composite_generator(
    name="baseline",
    description="55-process partitioned whole-cell E. coli model — upstream-parity architecture",
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
    },
)
def baseline(core: Any = None, *, seed: int = 0, cache_dir: str = "out/cache") -> dict:
    """Build the process-bigraph state document for the baseline architecture.

    Migrated from ``v2ecoli/generate.py:build_document`` +
    ``v2ecoli/composite.py:_build_from_cache``.  Returns a plain dict
    suitable for ``Composite(doc, core=core)``; does NOT wrap in Composite.

    Note: ``features`` is fixed to ``DEFAULT_FEATURES`` and is not a caller-
    visible parameter.  To run with a different feature set, use the
    departitioned or reconciled generator (when they land).

    Args:
        core: bigraph-schema core.  If None, one is created via build_core().
        seed: Random seed for stochastic initialisation.
        cache_dir: Path to the ParCa cache directory (must contain
            ``initial_state.json`` and ``sim_data_cache.dill``).

    Returns:
        Process-bigraph document dict with keys ``state``,
        ``skip_initial_steps``, ``sequential_steps``, ``flow_order``.
    """
    if core is None:
        core = build_core()

    bundle = load_cache_bundle(cache_dir)
    # bundle is a flat dict: initial_state, configs, unique_names,
    # dry_mass_inc_dict (and possibly others from the dill cache).
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

    # ------------------------------------------------------------------
    # Body migrated verbatim from v2ecoli/generate.py:build_document
    # (signature arguments replaced with bundle keys above).
    # features defaults to DEFAULT_FEATURES (same as build_document).
    # ------------------------------------------------------------------
    features = DEFAULT_FEATURES

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    # Pre-create virtual stores
    for store in ['listeners', 'process',
                  'allocator_rng', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault('mass', {'dry_mass': 0.0, 'cell_mass': 0.0})
    cell_state.setdefault('allocator_rng', np.random.RandomState(seed=seed))

    # Pre-create feature module stores
    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'enabled': False,
    })

    # Initialize next_update_time for all partitioned processes
    # (requesters/evolvers check this to decide whether to run)
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)  # Run on first tick

    # Pre-create shared request/allocate/process stores with per-process sub-keys
    cell_state.setdefault('request', {})
    cell_state.setdefault('allocate', {})
    for proc_name in ALL_PARTITIONED:
        cell_state['request'].setdefault(proc_name, {'bulk': {}})
        cell_state['allocate'].setdefault(proc_name, {'bulk': {}})
        # process store starts empty — populated when requester runs
    # Pre-create listener sub-stores expected by various steps
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })
    # Listener sub-stores are populated by _seed_state_from_ports below

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Create a mock loader that returns configs from the cache
    class _CachedLoader:
        def __init__(self, configs, unique_names, dry_mass_inc_dict, cache_dir='out/cache'):
            self._configs = configs
            self.unique_names = unique_names
            self.cache_dir = cache_dir

            class _SimData:
                class _InternalState:
                    class _UniqueMolecule:
                        def __init__(self, names):
                            self.unique_molecule_definitions = {
                                n: {} for n in names}
                    unique_molecule = None
                    def __init__(self, names):
                        self.unique_molecule = self._UniqueMolecule(names)
                internal_state = None
                expectedDryMassIncreaseDict = {}

            self.sim_data = _SimData()
            self.sim_data.internal_state = _SimData._InternalState(unique_names)
            self.sim_data.expectedDryMassIncreaseDict = dry_mass_inc_dict or {}

        def get_config_by_name(self, name):
            if name in self._configs:
                return self._configs[name]
            raise KeyError(f'Unknown: {name}')

    loader = _CachedLoader(configs, unique_names, dry_mass_inc_dict, cache_dir=cache_dir)

    # Build execution layers for the requested feature set
    execution_layers = build_execution_layers(features)
    flow_order = [step for layer in execution_layers for step in layer]

    _process_cache = {}
    for step_name in flow_order:
        config = _get_step_config(
            loader, step_name, core, _process_cache)
        if config is not None:
            if len(config) == 5:
                instance, topology, edge_type, in_topo, out_topo = config
                cell_state[step_name] = make_edge(
                    instance, topology, input_topology=in_topo,
                    output_topology=out_topo, edge_type=edge_type)
            else:
                instance, topology, edge_type = config
                cell_state[step_name] = make_edge(
                    instance, topology, edge_type=edge_type)

    # Place shared PartitionedProcess instances in the process store
    # (Requester/Evolver read them from state via the 'process' port)
    for proc_name, proc_instance in _process_cache.items():
        cell_state['process'][proc_name] = (proc_instance,)

    _seed_state_from_defaults(cell_state)
    _seed_mass_listener(cell_state, core)

    inject_flow_dependencies(
        cell_state, flow_order, layers=execution_layers)

    state = {
        'agents': {'0': cell_state},
        'global_time': 0.0,
    }

    return {
        'state': state,
        'skip_initial_steps': True,
        'sequential_steps': False,
        'flow_order': flow_order,
    }


