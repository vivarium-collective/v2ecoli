"""Departitioned whole-cell E. coli composite.

41 steps with requester+evolver halves fused into single Steps. See AGENTS.md.

Body migrated from:
  - v2ecoli/generate_departitioned.py:build_departitioned_document — top-level document builder
  - v2ecoli/composite_departitioned.py:make_departitioned_composite — thin Composite wrapper
    (no meaningful post-processing; just passes args through to build_departitioned_document)

Helpers from v2ecoli.generate (and architecture-specific helpers from
v2ecoli.generate_departitioned) are imported rather than duplicated here.
Task 14's deletion of the legacy generate*.py files will need to resolve
where the shared helpers ultimately live.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle

# ---------------------------------------------------------------------------
# Re-use shared helpers and constants from generate.py
# (same set that generate_departitioned.py imports)
# ---------------------------------------------------------------------------
from v2ecoli.generate import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    _seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    _expand_flushes,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
)

# Architecture-specific helpers from generate_departitioned.py
from v2ecoli.generate_departitioned import (
    build_execution_layers,
    _get_step_config,
    DEFAULT_FEATURES,
)


@composite_generator(
    name="departitioned",
    description="41-step departitioned whole-cell E. coli model — fused requester+evolver halves",
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
def departitioned(core: Any = None, *, seed: int = 0, cache_dir: str = "out/cache") -> dict:
    """Build the process-bigraph state document for the departitioned architecture.

    Migrated from ``v2ecoli/generate_departitioned.py:build_departitioned_document`` +
    ``v2ecoli/composite_departitioned.py:make_departitioned_composite``.  Returns a
    plain dict suitable for ``Composite(doc, core=core)``; does NOT wrap in Composite.

    Note: ``features`` is fixed to ``DEFAULT_FEATURES`` and is not a caller-visible
    parameter. Adjust by switching to a different architecture's generator.

    Args:
        core: bigraph-schema core. If None, one is created via build_core().
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
    # Body migrated verbatim from
    # v2ecoli/generate_departitioned.py:build_departitioned_document
    # (signature arguments replaced with bundle keys above).
    # features defaults to DEFAULT_FEATURES.
    # ------------------------------------------------------------------

    cell_state = {}
    cell_state.update(initial_state)

    _normalize_boundary_units(cell_state)

    # Pre-create virtual stores (no request/allocate needed)
    for store in ['listeners', 'process_state', 'exchange',
                  'next_update_time']:
        if store not in cell_state:
            cell_state[store] = {}
    cell_state.setdefault('global_time', 0.0)
    cell_state.setdefault('timestep', 1.0)
    cell_state.setdefault('divide', False)
    cell_state.setdefault('division_threshold', 'mass_distribution')
    cell_state.setdefault('listeners', {})
    cell_state['listeners'].setdefault(
        'mass', {'dry_mass': 0.0, 'cell_mass': 0.0})

    # Initialize next_update_time for all processes
    nut = cell_state.setdefault('next_update_time', {})
    for proc_name in ALL_PARTITIONED:
        nut.setdefault(proc_name, 0.0)

    # Listener sub-stores
    n_part = len(ALL_PARTITIONED)
    cell_state['listeners'].setdefault('atp', {
        'atp_requested': np.zeros(n_part, dtype=int),
        'atp_allocated_initial': np.zeros(n_part, dtype=int),
    })

    # Feature module stores
    cell_state.setdefault('ppgpp_state', {
        'basal_prob': [],
        'frac_active_rnap': 0.0,
    })
    cell_state.setdefault('attenuation_config', {
        'attenuation_probability': [],
    })

    cell_state.setdefault('process_state', {})
    cell_state['process_state'].setdefault('polypeptide_elongation', {
        'aa_exchange_rates': np.zeros(21),
        'gtp_to_hydrolyze': 0,
        'aa_count_diff': np.zeros(21),
    })

    # Mock loader
    class _CachedLoader:
        def __init__(self, configs, unique_names, dry_mass_inc_dict):
            self._configs = configs
            self.unique_names = unique_names

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
            self.sim_data.internal_state = _SimData._InternalState(
                unique_names)
            self.sim_data.expectedDryMassIncreaseDict = (
                dry_mass_inc_dict or {})

        def get_config_by_name(self, name):
            if name in self._configs:
                return self._configs[name]
            raise KeyError(f'Unknown: {name}')

    loader = _CachedLoader(configs, unique_names, dry_mass_inc_dict)

    # Build execution layers for the default feature set
    execution_layers = build_execution_layers(DEFAULT_FEATURES)
    flow_order = [step for layer in execution_layers for step in layer]

    for step_name in flow_order:
        config = _get_step_config(loader, step_name, core)
        if config is None:
            # Match baseline behavior: silently skip steps without a resolvable
            # config (e.g. exchange_data, which has no LoadSimData entry).
            continue
        if len(config) == 5:
            instance, topology, edge_type, in_topo, out_topo = config
            cell_state[step_name] = make_edge(
                instance, topology, input_topology=in_topo,
                output_topology=out_topo, edge_type=edge_type)
        else:
            instance, topology, edge_type = config
            cell_state[step_name] = make_edge(
                instance, topology, edge_type=edge_type)

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
