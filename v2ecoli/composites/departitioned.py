"""Departitioned whole-cell E. coli composite.

41 steps with requester+evolver halves fused into single Steps. See AGENTS.md.

Body migrated from:
  - v2ecoli/generate_departitioned.py:build_departitioned_document
  - v2ecoli/composite_departitioned.py:make_departitioned_composite

Shared helpers live in ``v2ecoli.composites._helpers``.
Architecture-specific helpers (``build_execution_layers``, ``DEFAULT_FEATURES``,
``_get_step_config``) are defined as private module-level functions here.
Both legacy files were deleted in Task 14.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core, load_cache_bundle

# ---------------------------------------------------------------------------
# Shared helpers and constants
# ---------------------------------------------------------------------------
from v2ecoli.composites._helpers import (
    make_edge,
    inject_flow_dependencies,
    _seed_state_from_defaults,
    seed_mass_listener,
    _normalize_boundary_units,
    _make_instance,
    _get_special_step,
    _expand_flushes,
    FLUSH,
    PARTITIONED_PROCESSES,
    ALL_PARTITIONED,
    DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)

# Process imports needed by _instantiate_departitioned_step
from v2ecoli.processes.equilibrium import Equilibrium
from v2ecoli.processes.two_component_system import TwoComponentSystem
from v2ecoli.processes.rna_maturation import RnaMaturation
from v2ecoli.processes.complexation import Complexation
from v2ecoli.processes.protein_degradation import ProteinDegradation
from v2ecoli.processes.transcript_initiation import TranscriptInitiation
from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from v2ecoli.processes.chromosome_replication import ChromosomeReplication
from v2ecoli.processes.plasmid_replication import PlasmidReplication
from v2ecoli.steps.departitioned import DepartitionedStep


# ---------------------------------------------------------------------------
# Feature modules (departitioned step names)
# ---------------------------------------------------------------------------

FEATURE_MODULES = {
    'supercoiling': {
        'insert_after': 'ecoli-chromosome-structure',
        'steps': ['dna-supercoiling-step'],
        'listeners': ['dna_supercoiling_listener'],
    },
    'ppgpp_regulation': {
        'insert_before': 'ecoli-transcript-initiation',
        'steps': ['ppgpp-initiation'],
    },
    'trna_attenuation': {
        'insert_before': 'ecoli-transcript-elongation',
        'steps': ['trna-attenuation-config'],
    },
}

DEFAULT_FEATURES = ['ppgpp_regulation']  # trna_attenuation disabled to match v1 default


# ---------------------------------------------------------------------------
# Execution layers (departitioned — no allocators)
# ---------------------------------------------------------------------------

BASE_EXECUTION_LAYERS = [
    # Layer 0: post-division mass
    ['post-division-mass-listener'], FLUSH,

    # Layer 1: media/environment
    ['media_update'], FLUSH,
    ['ecoli-tf-unbinding'],
    ['exchange_data'], FLUSH,

    # Layer 2: partition layer 1 (sequential standalone)
    ['ecoli-equilibrium'],
    ['ecoli-rna-maturation'],
    ['ecoli-two-component-system'], FLUSH,

    # Layer 3: TF binding
    ['ecoli-tf-binding'], FLUSH,

    # Layer 4: partition layer 2 (sequential standalone)
    ['ecoli-complexation'],
    ['ecoli-protein-degradation'],
    ['ecoli-rna-degradation'],
    ['ecoli-transcript-initiation'],
    ['ecoli-polypeptide-initiation'],
    ['ecoli-chromosome-replication'], FLUSH,

    # Layer 5: partition layer 3 (sequential standalone)
    ['ecoli-transcript-elongation'],
    ['ecoli-polypeptide-elongation'], FLUSH,

    # Layer 6: chromosome structure + metabolism
    ['ecoli-chromosome-structure'], FLUSH,
    ['ecoli-metabolism'], FLUSH,

    # Layer 7: listeners (parallel)
    ['RNA_counts_listener', 'dna_supercoiling_listener', 'ecoli-mass-listener',
     'monomer_counts_listener', 'replication_data_listener', 'ribosome_data_listener',
     'rna_synth_prob_listener', 'rnap_data_listener', 'unique_molecule_counts'], FLUSH,

    # Emitter + clock
    ['emitter'],
    ['global_clock'],

    # Layer 8: division check
    ['mark_d_period'], FLUSH,
    ['division'],
]


def build_execution_layers(features=None):
    """Build EXECUTION_LAYERS from base + enabled feature modules."""
    layers = copy.deepcopy(BASE_EXECUTION_LAYERS)
    for feat_name in (features or []):
        feat = FEATURE_MODULES.get(feat_name)
        if feat is None:
            continue
        if 'insert_after' in feat:
            ref = feat['insert_after']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in feat.get('steps', []):
                        layers.insert(i + 1, [step_name])
                    break
        if 'insert_before' in feat:
            ref = feat['insert_before']
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and ref in layer:
                    for step_name in reversed(feat.get('steps', [])):
                        layers.insert(i, [step_name])
                    break
        for listener in feat.get('listeners', []):
            for layer in layers:
                if isinstance(layer, list) and 'ecoli-mass-listener' in layer:
                    if listener not in layer:
                        layer.append(listener)
                    break
    return _expand_flushes(layers)


# ---------------------------------------------------------------------------
# Step instantiation (departitioned)
# ---------------------------------------------------------------------------

def _instantiate_departitioned_step(step_name, config, loader, core):
    """Instantiate a step for the departitioned architecture."""
    from v2ecoli.processes.tf_binding import TfBinding
    from v2ecoli.processes.tf_unbinding import TfUnbinding
    from v2ecoli.processes.chromosome_structure import ChromosomeStructure
    from v2ecoli.processes.metabolism import Metabolism
    from v2ecoli.steps.listeners.mass_listener import (
        MassListener, PostDivisionMassListener)
    from v2ecoli.steps.listeners.rna_counts import RNACounts
    from v2ecoli.steps.listeners.rna_synth_prob import RnaSynthProb
    from v2ecoli.steps.listeners.monomer_counts import MonomerCounts
    from v2ecoli.steps.listeners.dna_supercoiling import DnaSupercoiling
    from v2ecoli.steps.listeners.replication_data import ReplicationData
    from v2ecoli.steps.listeners.rnap_data import RnapData
    from v2ecoli.steps.listeners.unique_molecule_counts import UniqueMoleculeCounts
    from v2ecoli.steps.listeners.ribosome_data import RibosomeData
    from v2ecoli.steps.media_update import MediaUpdate
    from v2ecoli.steps.exchange_data import ExchangeData

    STANDALONE_STEPS = {
        'ecoli-tf-binding': TfBinding,
        'ecoli-tf-unbinding': TfUnbinding,
        'ecoli-chromosome-structure': ChromosomeStructure,
        'ecoli-metabolism': Metabolism,
        'ecoli-equilibrium': Equilibrium,
        'ecoli-two-component-system': TwoComponentSystem,
        'ecoli-rna-maturation': RnaMaturation,
        'ecoli-complexation': Complexation,
        'ecoli-protein-degradation': ProteinDegradation,
        'ecoli-transcript-initiation': TranscriptInitiation,
        'ecoli-polypeptide-initiation': PolypeptideInitiation,
        'ecoli-chromosome-replication': ChromosomeReplication,
        'ecoli-plasmid-replication': PlasmidReplication,
    }

    SIMPLE_STEPS = {
        'ecoli-mass-listener': MassListener,
        'post-division-mass-listener': PostDivisionMassListener,
        'RNA_counts_listener': RNACounts,
        'rna_synth_prob_listener': RnaSynthProb,
        'monomer_counts_listener': MonomerCounts,
        'dna_supercoiling_listener': DnaSupercoiling,
        'replication_data_listener': ReplicationData,
        'rnap_data_listener': RnapData,
        'unique_molecule_counts': UniqueMoleculeCounts,
        'ribosome_data_listener': RibosomeData,
        'media_update': MediaUpdate,
        'exchange_data': ExchangeData,
    }

    from v2ecoli.library.config_resolver import resolve_config
    config = resolve_config(config) if config else config

    EVOLVE_ONLY = {
        'ecoli-rna-maturation',
        'ecoli-complexation',
    }

    # Departitioned: wrap PartitionedProcess in DepartitionedStep
    if step_name in PARTITIONED_PROCESSES:
        proc_cls = PARTITIONED_PROCESSES[step_name]
        from v2ecoli.library.ecoli_step import set_current_core
        set_current_core(core)
        process = proc_cls(config)
        set_current_core(None)

        if step_name in EVOLVE_ONLY:
            process.evolve_only = True

        topology = dict(config.get('topology', {}) or {})
        if not topology:
            topology = getattr(process, 'topology',
                               getattr(proc_cls, 'topology', {}))
            if callable(topology):
                topology = topology()
            topology = dict(topology)

        instance = DepartitionedStep({
            'time_step': config.get('time_step', 1),
            'process': process,
        })

        in_topo = dict(topology)
        in_topo['global_time'] = ('global_time',)
        in_topo.setdefault('timestep', ('timestep',))
        in_topo['next_update_time'] = ('next_update_time', step_name)

        out_topo = {
            'next_update_time': ('next_update_time', step_name),
        }
        out_ports = set(instance.outputs().keys())
        for port in out_ports:
            if port in ('next_update_time', 'global_time', 'timestep'):
                continue
            if port in topology:
                out_topo[port] = topology[port]
            elif port == 'listeners':
                out_topo['listeners'] = topology.get(
                    'listeners', ('listeners',))

        return instance, topology, 'step', in_topo, out_topo

    if step_name in STANDALONE_STEPS:
        step_cls = STANDALONE_STEPS[step_name]
        instance = _make_instance(step_cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    if step_name in SIMPLE_STEPS:
        cls = SIMPLE_STEPS[step_name]
        instance = _make_instance(cls, config, core)
        topology = getattr(instance, 'topology', {})
        if callable(topology):
            topology = topology()
        return instance, topology, 'step'

    return None


def _get_step_config(loader, step_name, core):
    """Get step config — skip allocators."""
    if step_name.startswith('allocator'):
        return None

    try:
        config = loader.get_config_by_name(step_name)
    except (KeyError, AttributeError):
        return _get_special_step(loader, step_name, core)

    if config is None:
        return _get_special_step(loader, step_name, core)

    return _instantiate_departitioned_step(step_name, config, loader, core)


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
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
def departitioned(core: Any = None, *, seed: int = 0, cache_dir: str = "out/cache") -> dict:
    """Build the process-bigraph state document for the departitioned architecture.

    Migrated from ``v2ecoli/generate_departitioned.py:build_departitioned_document`` +
    ``v2ecoli/composite_departitioned.py:make_departitioned_composite``.  Returns a
    plain dict suitable for ``Composite(doc, core=core)``; does NOT wrap in Composite.

    Note: ``features`` is fixed to ``DEFAULT_FEATURES`` and is not a caller-visible
    parameter. Adjust by switching to a different architecture's generator.

    Note: ``seed`` is accepted for signature uniformity across all three
    architectures, but the departitioned document is deterministic; the
    seed is not consumed during construction.

    Args:
        core: bigraph-schema core. If None, one is created via build_core().
        seed: Random seed for stochastic initialisation (not consumed at
            build time; accepted for signature uniformity only).
        cache_dir: Path to the ParCa cache directory (must contain
            ``initial_state.json`` and ``sim_data_cache.dill``).

    Returns:
        Process-bigraph document dict with keys ``state``,
        ``skip_initial_steps``, ``sequential_steps``, ``flow_order``.
    """
    if core is None:
        core = build_core()

    bundle = load_cache_bundle(cache_dir)
    initial_state = bundle["initial_state"]
    configs = bundle["configs"]
    unique_names = bundle["unique_names"]
    dry_mass_inc_dict = bundle.get("dry_mass_inc_dict", {})

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
    seed_mass_listener(cell_state, core)

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
