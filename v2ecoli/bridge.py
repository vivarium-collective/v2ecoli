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
        self._prev_length = 2.0

    def inputs(self):
        return {
            'local': 'map[float]',
            'agent_id': 'string',
            'location': 'tuple[float,float]',
            'angle': 'float',
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
        """Read mass/volume from internal composite, compute length."""
        cell = self._composite.state
        mass_data = cell.get('listeners', {}).get('mass', {})
        cur_mass = float(mass_data.get('dry_mass', 0.0))
        cur_volume = float(mass_data.get('volume', 0.0))
        growth = float(mass_data.get('growth', 0.0))

        d_mass = cur_mass - self._prev_mass
        d_volume = cur_volume - self._prev_volume
        self._prev_mass = cur_mass
        self._prev_volume = cur_volume

        d_mass_physical = max(0.0, d_mass)

        # Compute length from volume using capsule geometry:
        # V = (4/3)πr³ + πr²a, l = a + 2r
        import math
        radius = 0.5  # µm, E. coli radius
        if cur_volume > 0:
            vol_um3 = cur_volume  # volume in fL ≈ µm³
            cylinder_a = (vol_um3 - (4/3) * math.pi * radius**3) / (math.pi * radius**2)
            length = max(2 * radius, cylinder_a + 2 * radius)
        else:
            length = 2.0
        d_length = length - self._prev_length
        self._prev_length = length

        exchange = {}
        if growth != 0.0:
            exchange['biomass'] = growth

        return d_mass_physical, d_length, d_volume, exchange

    def outputs(self):
        return {
            'mass': 'float',
            'length': 'float',
            'volume': 'float',
            'exchange': 'map[float]',
            'agents': 'map[map]',  # for division: writing new cells
        }

    def update(self, state, interval):
        if self._composite is None:
            try:
                self._build_composite()
            except Exception as e:
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
        division_fired = False
        try:
            self._composite.run(interval)
        except Exception as e:
            # Check if this was a division event
            err_str = str(e).lower()
            if 'divide' in err_str or 'division' in err_str or 'DIVISION' in str(e):
                division_fired = True
            # Otherwise return zeros
            if not division_fired:
                return {'mass': 0.0, 'volume': 0.0, 'exchange': {}}

        # Check for division flag in internal state
        cell = self._composite.state
        if cell.get('divide', False):
            division_fired = True

        if division_fired:
            return self._handle_division(state)

        # Normal: read outputs and return deltas
        d_mass, d_length, d_volume, exchange = self._read_outputs()
        return {
            'mass': d_mass,
            'length': d_length,
            'volume': d_volume,
            'exchange': exchange,
        }

    def _handle_division(self, state):
        """Handle WCM division: produce two daughter cells in the colony."""
        cell = self._composite.state
        mass_data = cell.get('listeners', {}).get('mass', {})
        mother_mass = float(mass_data.get('dry_mass', 380.0))
        half_mass = mother_mass / 2

        # Get mother cell's physical state from the colony
        agent_id = state.get('agent_id', 'unknown')

        # Build two daughter EcoliWCM specs
        cache_dir = self.config.get('cache_dir', 'out/cache')
        seed = int(self.config.get('seed', 0))

        from multi_cell.processes.multibody import make_rng, build_microbe
        rng = make_rng(seed + hash(agent_id) % 10000)

        # Place daughters near mother
        mother_loc = state.get('location', (15, 15))
        if isinstance(mother_loc, (list, tuple)) and len(mother_loc) >= 2:
            mx, my = float(mother_loc[0]), float(mother_loc[1])
        else:
            mx, my = 15.0, 15.0
        mother_angle = float(state.get('angle', 0))

        import math
        offset = 1.5  # µm apart
        dx = offset * math.cos(mother_angle)
        dy = offset * math.sin(mother_angle)

        daughters = {}
        for i, (ox, oy) in enumerate([(dx, dy), (-dx, -dy)]):
            d_id = f'{agent_id}_{i}'
            _, d_body = build_microbe(
                rng, env_size=40,
                x=mx + ox, y=my + oy, angle=mother_angle + rng.uniform(-0.3, 0.3),
                length=2.0, radius=0.5, density=0.02,
            )
            d_body['mass'] = half_mass
            # Each daughter gets its own EcoliWCM
            d_body['ecoli'] = {
                '_type': 'process',
                'address': 'local:EcoliWCM',
                'config': {
                    'cache_dir': cache_dir,
                    'seed': seed + i + 1,
                },
                'interval': self.config.get('interval', 60.0),
                'inputs': {
                    'local': ['local'],
                    'agent_id': ['id'],
                    'location': ['location'],
                    'angle': ['angle'],
                },
                'outputs': {
                    'mass': ['mass'],
                    'length': ['length'],
                    'volume': ['volume'],
                    'exchange': ['exchange'],
                    'agents': ['..', '..', 'cells'],
                },
            }
            d_body['local'] = {}
            d_body['volume'] = 0.0
            d_body['exchange'] = {}
            daughters[d_id] = d_body

        # Return structural update: remove mother, add daughters
        return {
            'mass': 0.0,
            'length': 0.0,
            'volume': 0.0,
            'exchange': {},
            'agents': {
                '_remove': [agent_id] if agent_id != 'unknown' else [],
                '_add': list(daughters.items()),
            },
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
