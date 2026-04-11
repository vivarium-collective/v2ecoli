"""
EcoliWCM — whole-cell E. coli model as a Process with bridge.

Wraps the full v2ecoli simulation (55 biological steps) as a single
Process node with inputs/outputs connected to internal stores via
a Composite bridge. This allows the whole-cell model to be embedded
inside multi_cell colony simulations or any other bigraph composite.

The bridge maps:
    external inputs  →  internal stores
    internal stores  →  external outputs

Usage in a multi_cell document::

    'ecoli': {
        '_type': 'process',
        'address': 'local:EcoliWCM',
        'config': {'cache_dir': 'out/cache'},
        'interval': 1.0,
        'inputs': {'local': ['local']},
        'outputs': {
            'mass': ['mass'],
            'volume': ['volume'],
        },
    }
"""

from process_bigraph import Process, Composite


class EcoliWCM(Process):
    """Whole-cell E. coli as a single Process with bridge to internals.

    Holds an internal Composite (the full v2ecoli model) and exposes
    selected internal stores as external ports. Built lazily on first
    update() to avoid heavy imports at discovery time.
    """

    config_schema = {
        'cache_dir':      {'_type': 'string', '_default': ''},
        'seed':           {'_type': 'integer', '_default': 0},
        'molecule_map':   {'_default': {}},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._composite = None
        self._prev_mass = 0.0
        self._prev_volume = 0.0

    def inputs(self):
        return {
            'local': 'map[float]',
        }

    def outputs(self):
        return {
            'mass': 'float',
            'volume': 'float',
            'exchange': 'map[float]',
        }

    def _build_composite(self):
        """Lazily construct the internal v2ecoli Composite with bridge."""
        from v2ecoli.composite import _build_core
        from v2ecoli.generate import build_document
        from v2ecoli.cache import load_initial_state
        # Import types to trigger resolve dispatch registration
        import v2ecoli.types  # noqa: F401
        import dill, os

        cache_dir = self.config.get('cache_dir', '') or 'out/cache'
        seed = int(self.config.get('seed', 0))

        internal_core = _build_core()

        # Build from cache
        initial_state = load_initial_state(
            os.path.join(cache_dir, 'initial_state.json'))
        with open(os.path.join(cache_dir, 'sim_data_cache.dill'), 'rb') as f:
            cache = dill.load(f)

        document = build_document(
            initial_state=initial_state,
            configs=cache['configs'],
            unique_names=cache['unique_names'],
            dry_mass_inc_dict=cache.get('dry_mass_inc_dict', {}),
            core=internal_core,
            seed=seed,
        )

        # Unwrap cell state from agents container
        cell_state = document['state']['agents']['0']

        # Remove edges with '..' references (division tries to access parent)
        for key in list(cell_state.keys()):
            val = cell_state[key]
            if isinstance(val, dict) and 'inputs' in val:
                wires = {**val.get('inputs', {}), **val.get('outputs', {})}
                for port, wire in wires.items():
                    if isinstance(wire, list) and '..' in wire:
                        del cell_state[key]
                        break

        # Bridge: maps external ports to internal v2ecoli store paths
        bridge = {
            'inputs': {
                'local': ['boundary', 'external'],
            },
            'outputs': {
                'mass': ['listeners', 'mass', 'dry_mass'],
                'volume': ['listeners', 'mass', 'volume'],
                'exchange': ['listeners', 'mass', 'growth'],
            },
        }

        composite_config = {
            'state': cell_state,
            'bridge': bridge,
            'flow_order': document.get('flow_order', []),
            'skip_initial_steps': document.get('skip_initial_steps', True),
            'sequential_steps': document.get('sequential_steps', False),
        }

        self._composite = Composite(
            config=composite_config, core=internal_core)

        # Seed previous values for delta computation
        mass_data = cell_state.get('listeners', {}).get('mass', {})
        self._prev_mass = float(mass_data.get('dry_mass', 0.0))
        self._prev_volume = float(mass_data.get('volume', 0.0))

    def _read_outputs(self):
        """Read mass/volume from internal composite.

        Returns the delta to reach the current whole-cell mass from
        the *physical* mass set by the colony. Since the physical mass
        starts small (~0.04fg) and the whole-cell mass is ~380fg, we
        compute: delta = wcm_mass * scale_factor - current_physical_mass.
        """
        cell = self._composite.state
        mass_data = cell.get('listeners', {}).get('mass', {})
        cur_mass = float(mass_data.get('dry_mass', 0.0))
        cur_volume = float(mass_data.get('volume', 0.0))
        growth = float(mass_data.get('growth', 0.0))

        # Compute mass delta, clamped to non-negative to prevent
        # pymunk crashes from negative physical mass
        d_mass = cur_mass - self._prev_mass
        d_volume = cur_volume - self._prev_volume
        self._prev_mass = cur_mass
        self._prev_volume = cur_volume

        # Pass whole-cell mass delta directly (in fg)
        # Clamp to non-negative to prevent pymunk crashes
        d_mass_physical = max(0.0, d_mass)

        exchange = {}
        if growth != 0.0:
            exchange['biomass'] = growth

        return d_mass_physical, d_volume, exchange

    def update(self, state, interval):
        if self._composite is None:
            try:
                self._build_composite()
            except Exception as e:
                # Build failed — return zeros
                return {'mass': 0.0, 'volume': 0.0, 'exchange': {}}

        # Push external concentrations directly into internal boundary
        local = state.get('local') or {}
        mol_map = self.config.get('molecule_map', {})
        boundary = self._composite.state.get('boundary', {}).get('external', {})
        if isinstance(boundary, dict):
            for mc_name, conc in local.items():
                ecoli_name = mol_map.get(mc_name, mc_name)
                boundary[ecoli_name] = float(conc)

        # Run the internal composite directly
        try:
            self._composite.run(interval)
        except Exception:
            # Internal sim error — return zeros for this tick
            return {'mass': 0.0, 'volume': 0.0, 'exchange': {}}

        # Read outputs and return deltas
        d_mass, d_volume, exchange = self._read_outputs()
        return {
            'mass': d_mass,
            'volume': d_volume,
            'exchange': exchange,
        }


def ecoli_document(cache_dir='out/cache', seed=0):
    """Return a process-bigraph document spec for an EcoliWCM process.

    This is a convenience function for embedding E. coli in larger
    composites. The returned dict can be placed directly into a
    composite document::

        doc = {
            'cells': {
                'cell_0': {
                    **ecoli_document(),
                    # additional cell state...
                }
            }
        }
    """
    return {
        '_type': 'process',
        'address': 'local:EcoliWCM',
        'config': {
            'cache_dir': cache_dir,
            'seed': seed,
        },
        'interval': 1.0,
        'inputs': {
            'local': ['local'],
        },
        'outputs': {
            'mass': ['mass'],
            'volume': ['volume'],
            'exchange': ['exchange'],
        },
    }
