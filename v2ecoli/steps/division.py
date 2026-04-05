"""
Cell division steps for v2ecoli.

MarkDPeriod: Sets division flag after D period has elapsed.
Division: Detects division condition, splits state, produces daughter
    cells via process-bigraph's _add/_remove structural updates.
"""

import numpy as np
import binascii

from process_bigraph import Step
from bigraph_schema.schema import Node, Float, Overwrite

from v2ecoli.library.schema import attrs
from v2ecoli.library.units import units
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + '0', str(mother_id) + '1']


class MarkDPeriod(Step):
    """Set division flag after D period has elapsed."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)

    def inputs(self):
        return {
            'full_chromosome': UniqueNumpyUpdate(),
            'global_time': Float(),
            'divide': Overwrite(_value=Node()),
        }

    def outputs(self):
        return {
            'full_chromosome': UniqueNumpyUpdate(),
            'divide': Overwrite(_value=Node()),
        }

    def update(self, state):
        full_chrom = state.get('full_chromosome')
        if full_chrom is None or not hasattr(full_chrom, 'dtype'):
            return {}

        division_time, has_triggered_division = attrs(
            full_chrom, ['division_time', 'has_triggered_division'])

        if len(division_time) < 2:
            return {}

        untriggered = division_time[~has_triggered_division]
        if len(untriggered) == 0:
            return {}
        divide_at_time = untriggered.min()
        if state.get('global_time', 0) >= divide_at_time:
            divide_at_time_index = np.where(division_time == divide_at_time)[0][0]
            has_triggered_division = has_triggered_division.copy()
            has_triggered_division[divide_at_time_index] = True
            return {
                'full_chromosome': {
                    'set': {'has_triggered_division': has_triggered_division}
                },
                'divide': True,
            }
        return {}


class Division(Step):
    """Detect division and produce daughter cells via _add/_remove.

    When the division condition is met (dry mass >= threshold with 2+
    chromosomes), this step:
    1. Divides the mother cell's state (bulk binomial, unique by domain)
    2. Builds complete daughter cell states with fresh process instances
    3. Returns structural update to remove mother and add daughters
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)
        self.parameters = config or {}
        self.agent_id = self.parameters.get('agent_id', '0')
        self.dry_mass_inc_dict = self.parameters.get('dry_mass_inc_dict', {})
        self.division_detected = False

        # Configs needed to rebuild daughter cell states
        self._configs = self.parameters.get('configs', {})
        self._unique_names = self.parameters.get('unique_names', [])
        self._seed = self.parameters.get('seed', 0)

        # Division mass multiplier
        seed = self.parameters.get('seed', 0)
        self.division_mass_multiplier = 1
        if self.parameters.get('division_threshold') == 'mass_distribution':
            div_seed = binascii.crc32(b'CellDivision', seed) & 0xFFFFFFFF
            div_rng = np.random.RandomState(seed=div_seed)
            self.division_mass_multiplier = div_rng.normal(loc=1.0, scale=0.1)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'unique': InPlaceDict(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'global_time': Float(),
            'division_threshold': Overwrite(_value=Node()),
            'media_id': Overwrite(_value=Node()),
        }

    def outputs(self):
        return {
            'agents': {
                '_type': 'map',
                '_value': 'node',
            },
            'division_threshold': Overwrite(_value=Node()),
        }

    def update(self, state):
        # --- Threshold initialization (first timestep) ---
        if state.get('division_threshold') == 'mass_distribution':
            media_id = state.get('media_id', 'minimal')
            dry_mass_inc = self.dry_mass_inc_dict.get(media_id)
            if dry_mass_inc is not None:
                dry_mass = 0.0
                listeners = state.get('listeners')
                if isinstance(listeners, dict):
                    mass = listeners.get('mass', {})
                    if isinstance(mass, dict):
                        dry_mass = mass.get('dry_mass', 0.0)
                return {
                    'division_threshold': (
                        dry_mass
                        + dry_mass_inc.asNumber(units.fg)
                        * self.division_mass_multiplier
                    ),
                    'agents': {},
                }
            return {}

        # --- Division check ---
        # Get dry mass from listeners
        division_variable = 0.0
        listeners = state.get('listeners')
        if isinstance(listeners, dict):
            mass = listeners.get('mass', {})
            if isinstance(mass, dict):
                division_variable = mass.get('dry_mass', 0.0)

        threshold = state.get('division_threshold', float('inf'))

        full_chrom = state.get('unique', {}).get('full_chromosome')
        n_chromosomes = 0
        if full_chrom is not None and hasattr(full_chrom, 'dtype'):
            if '_entryState' in full_chrom.dtype.names:
                n_chromosomes = full_chrom['_entryState'].sum()

        if division_variable < threshold or n_chromosomes < 2:
            return {}

        if self.division_detected:
            return {}

        # --- DIVISION ---
        self.division_detected = True
        division_time = state.get('global_time', 0)
        print(f'DIVISION at t={division_time:.0f}s '
              f'(dry_mass={division_variable:.1f} fg, '
              f'threshold={threshold:.1f} fg, '
              f'chromosomes={n_chromosomes})')

        # Build a cell_state-like dict from our inputs for divide_cell
        cell_data = {
            'bulk': state['bulk'],
            'unique': state['unique'],
            'environment': state.get('environment', {}),
            'boundary': state.get('boundary', {}),
        }

        from v2ecoli.library.division import divide_cell, daughter_phylogeny_id
        d1_data, d2_data = divide_cell(cell_data)

        # Preserve global_time in daughters
        d1_data['global_time'] = division_time
        d2_data['global_time'] = division_time

        # Build complete daughter cell states with fresh process instances
        from v2ecoli.generate import build_cell_state
        d1_id, d2_id = daughter_phylogeny_id(self.agent_id)
        d1_seed = (self._seed + 1) % (2**31)
        d2_seed = (self._seed + 2) % (2**31)

        d1_cell = build_cell_state(
            d1_data, self._configs, self._unique_names,
            dry_mass_inc_dict=self.dry_mass_inc_dict,
            core=self.core, seed=d1_seed)

        d2_cell = build_cell_state(
            d2_data, self._configs, self._unique_names,
            dry_mass_inc_dict=self.dry_mass_inc_dict,
            core=self.core, seed=d2_seed)

        print(f'  DAUGHTERS: {d1_id} (bulk={d1_data["bulk"]["count"].sum()}) '
              f'+ {d2_id} (bulk={d2_data["bulk"]["count"].sum()})')

        return {
            'agents': {
                '_remove': [self.agent_id],
                '_add': [
                    (d1_id, d1_cell),
                    (d2_id, d2_cell),
                ],
            },
        }
