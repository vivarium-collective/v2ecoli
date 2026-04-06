"""
Composite loading for v2ecoli.

Uses process-bigraph's Composite directly — no custom simulation engine.

Supports three loading modes:
1. From simData pickle: make_composite(sim_data_path='...')
2. From cache directory: make_composite(cache_dir='out/cache')
3. From pre-built document: make_composite(document={...})
"""

import os
import time

import dill
from bigraph_schema import allocate_core
from process_bigraph import Composite

from v2ecoli.types import ECOLI_TYPES
from v2ecoli.cache import load_initial_state, save_initial_state, save_json


def _build_core():
    """Create and configure a bigraph-schema core with ecoli types."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


def make_composite(document=None, sim_data_path=None, cache_dir=None,
                   seed=0, core=None):
    """Create a Composite from a document, cache, or simData.

    Loading modes (in priority order):
    1. document: Pre-built document dict
    2. cache_dir: Directory with initial_state.json + sim_data_cache.dill
    3. sim_data_path: Path to simData pickle (runs build_document)

    Args:
        document: Pre-built document dict.
        sim_data_path: Path to simData pickle.
        cache_dir: Path to cache directory.
        seed: Random seed for initial state generation.
        core: Pre-configured core. If None, creates one.

    Returns:
        A Composite ready for .run(interval).
    """
    if core is None:
        core = _build_core()

    if document is None:
        if cache_dir and os.path.isdir(cache_dir):
            document = _build_from_cache(cache_dir, core, seed)
        else:
            from v2ecoli.generate import build_document
            document = build_document(sim_data_path=sim_data_path, seed=seed)

    composite = Composite(document, core=core)
    return composite


def _build_from_cache(cache_dir, core, seed=0):
    """Build a document from cached initial state and process configs."""
    from v2ecoli.generate import build_document_from_configs  # deferred: circular

    # Load cached data
    initial_state = load_initial_state(
        os.path.join(cache_dir, 'initial_state.json'))

    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    with open(cache_path, 'rb') as f:
        cache = dill.load(f)

    return build_document_from_configs(
        initial_state=initial_state,
        configs=cache['configs'],
        unique_names=cache['unique_names'],
        dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
        core=core,
        seed=seed,
    )


def save_cache(sim_data_path, cache_dir='out/cache', seed=0):
    """Generate and save cache files from simData.

    Creates:
    - cache_dir/initial_state.json — E. coli initial state (10MB JSON)
    - cache_dir/sim_data_cache.dill — process configs (190MB pickle)
    - cache_dir/metadata.json — unique molecule names etc.
    """
    from v2ecoli.library.sim_data import LoadSimData  # deferred: heavy import

    os.makedirs(cache_dir, exist_ok=True)

    loader = LoadSimData(sim_data_path=sim_data_path, seed=seed)

    # Save initial state as JSON
    state = loader.generate_initial_state()
    save_initial_state(state, os.path.join(cache_dir, 'initial_state.json'))

    # Save configs as pickle
    configs = {}
    for name in [
        'post-division-mass-listener', 'ecoli-mass-listener', 'media_update',
        'exchange_data', 'ecoli-tf-unbinding', 'ecoli-tf-binding',
        'ecoli-equilibrium', 'ecoli-two-component-system', 'ecoli-rna-maturation',
        'ecoli-transcript-initiation', 'ecoli-polypeptide-initiation',
        'ecoli-chromosome-replication', 'ecoli-protein-degradation',
        'ecoli-rna-degradation', 'ecoli-complexation',
        'ecoli-transcript-elongation', 'ecoli-polypeptide-elongation',
        'ecoli-chromosome-structure', 'ecoli-metabolism',
        'RNA_counts_listener', 'rna_synth_prob_listener',
        'monomer_counts_listener', 'dna_supercoiling_listener',
        'ribosome_data_listener', 'rnap_data_listener',
        'unique_molecule_counts', 'allocator',
    ]:
        try:
            configs[name] = loader.get_config_by_name(name)
        except Exception:
            pass

    unique_names = list(
        loader.sim_data.internal_state.unique_molecule.unique_molecule_definitions.keys())

    # Also save division parameters
    dry_mass_inc = getattr(loader.sim_data, 'expectedDryMassIncreaseDict', {})

    cache = {
        'configs': configs,
        'unique_names': unique_names,
        'dry_mass_inc_dict': dry_mass_inc,
    }
    cache_path = os.path.join(cache_dir, 'sim_data_cache.dill')
    with open(cache_path, 'wb') as f:
        dill.dump(cache, f)

    save_json({'unique_names': unique_names}, os.path.join(cache_dir, 'metadata.json'))
    print(f"Cache saved to {cache_dir}")


def save_state(composite, path='out/checkpoint.dill'):
    """Save the full simulation state for later resumption."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # Extract just the cell state (no instances)
    state = composite.state
    flow_order = composite.config.get('flow_order', [])

    with open(path, 'wb') as f:
        dill.dump({
            'state': state,
            'flow_order': flow_order,
            'global_time': state.get('global_time', 0.0),
        }, f)
    print(f"State saved to {path} (t={state.get('global_time', 0)})")


def load_state(path='out/checkpoint.dill', core=None):
    """Load a saved state and create a new Composite from it.

    This rebuilds the Composite with fresh process instances
    but restores the saved state values.
    """
    with open(path, 'rb') as f:
        checkpoint = dill.load(f)

    # The checkpoint has the full state including process instances
    # Just wrap it in a Composite
    if core is None:
        core = _build_core()

    document = {
        'state': checkpoint['state'],
        'skip_initial_steps': True,
        'sequential_steps': False,
    }

    composite = Composite(document, core=core)
    print(f"Loaded state from {path} (t={checkpoint['global_time']})")
    return composite


def run_and_cache(cache_dir='out/cache', checkpoint_dir='out/checkpoints',
                  intervals=None, seed=0):
    """Run simulation with periodic checkpoints.

    Args:
        cache_dir: Path to initial cache.
        checkpoint_dir: Where to save checkpoints.
        intervals: List of simulation times to checkpoint at.
        seed: Random seed.

    Returns:
        Final composite.
    """

    if intervals is None:
        intervals = [100, 500, 1000, 1500, 1800, 2000, 2500, 3000]

    os.makedirs(checkpoint_dir, exist_ok=True)
    composite = make_composite(cache_dir=cache_dir, seed=seed)

    for target in intervals:
        cell = composite.state['agents']['0']
        current_t = cell.get('global_time', 0.0)
        remaining = target - current_t
        if remaining <= 0:
            continue

        t0 = time.time()
        composite.run(remaining)
        elapsed = time.time() - t0

        cell = composite.state['agents']['0']
        mass = cell.get('listeners', {}).get('mass', {})
        dm = mass.get('dry_mass', 0)
        thresh = cell.get('division_threshold', '?')
        n_chrom = 0
        fc = cell.get('unique', {}).get('full_chromosome')
        if fc is not None and hasattr(fc, 'dtype'):
            n_chrom = fc['_entryState'].view(bool).sum()

        path = os.path.join(checkpoint_dir, f't{target}.dill')
        save_state(composite, path)
        print(f'  t={target}: dry_mass={float(dm):.1f}, threshold={thresh}, '
              f'chroms={n_chrom} ({elapsed:.0f}s wall)')

    return composite
