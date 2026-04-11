"""
Cell division steps for v2ecoli.

MarkDPeriod: Sets division flag after D period has elapsed.
Division: Detects division condition, splits state, produces daughter
    cells via process-bigraph's _add/_remove structural updates.
"""

import numpy as np
import binascii

from bigraph_schema.schema import Overwrite, Float

from v2ecoli.steps.base import V2Step
from v2ecoli.types.stores import InPlaceDict
from v2ecoli.library.schema import attrs
from v2ecoli.library.unit_defs import units


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + '0', str(mother_id) + '1']


class MarkDPeriod(V2Step):
    """Set division flag after D period has elapsed."""

    name = "mark_d_period"
    config_schema = {}

    def initialize(self, config):
        self.parameters = config or {}

    def inputs(self):
        return {
            "full_chromosome": InPlaceDict(),
            "global_time": Float(_default=0.0),
            "divide": Overwrite(),
        }

    def outputs(self):
        return self.inputs()

    def next_update(self, timestep, states):
        full_chrom = states.get("full_chromosome")
        if full_chrom is None or not hasattr(full_chrom, 'dtype'):
            return {}

        division_time, has_triggered_division = attrs(
            full_chrom, ["division_time", "has_triggered_division"])

        if len(division_time) < 2:
            return {}

        untriggered = division_time[~has_triggered_division]
        if len(untriggered) == 0:
            return {}
        divide_at_time = untriggered.min()
        if states.get("global_time", 0) >= divide_at_time:
            divide_at_time_index = np.where(division_time == divide_at_time)[0][0]
            has_triggered_division = has_triggered_division.copy()
            has_triggered_division[divide_at_time_index] = True
            return {
                "full_chromosome": {
                    "set": {"has_triggered_division": has_triggered_division}
                },
                "divide": True,
            }
        return {}

    def update(self, state, interval=None):
        return self.next_update(1.0, state)


class Division(V2Step):
    """Detect division and produce daughter cells via _add/_remove.

    When the division condition is met (dry mass >= threshold with 2+
    chromosomes), this step:
    1. Divides the mother cell's state (bulk binomial, unique by domain)
    2. Builds complete daughter cell states with fresh process instances
    3. Returns structural update to remove mother and add daughters
    """

    name = "division"
    config_schema = {}

    def initialize(self, config):
        self.parameters = config or {}
        self.agent_id = self.parameters.get('agent_id', '0')
        self.dry_mass_inc_dict = self.parameters.get('dry_mass_inc_dict', {})
        self.division_detected = False

        # Configs for rebuilding daughter cell states
        self._configs = self.parameters.get('configs', {})
        self._unique_names = self.parameters.get('unique_names', [])
        self._seed = self.parameters.get('seed', 0)

        # Division mass multiplier
        seed = self._seed
        self.division_mass_multiplier = 1
        if self.parameters.get('division_threshold') == 'mass_distribution':
            div_seed = binascii.crc32(b'CellDivision', seed) & 0xFFFFFFFF
            div_rng = np.random.RandomState(seed=div_seed)
            self.division_mass_multiplier = div_rng.normal(loc=1.0, scale=0.1)

    def inputs(self):
        return {
            "bulk": InPlaceDict(),
            "unique": InPlaceDict(),
            "listeners": InPlaceDict(),
            "environment": InPlaceDict(),
            "boundary": InPlaceDict(),
            "global_time": Float(_default=0.0),
            "division_threshold": Overwrite(),
            "media_id": InPlaceDict(),
        }

    def outputs(self):
        ports = self.inputs()
        ports['agents'] = {'_type': 'map', '_value': 'node'}
        return ports

    def next_update(self, timestep, states):
        # --- Threshold initialization ---
        if states.get("division_threshold") == "mass_distribution":
            media_id = states.get("media_id", "minimal")
            dry_mass_inc = self.dry_mass_inc_dict.get(media_id)
            if dry_mass_inc is not None:
                dry_mass = 0.0
                listeners = states.get('listeners')
                if isinstance(listeners, dict):
                    mass = listeners.get('mass', {})
                    if isinstance(mass, dict):
                        dry_mass = mass.get('dry_mass', 0.0)
                return {
                    "division_threshold": (
                        dry_mass
                        + dry_mass_inc.asNumber(units.fg)
                        * self.division_mass_multiplier
                    )
                }
            return {}

        # --- Division check ---
        dry_mass = 0.0
        listeners = states.get('listeners')
        if isinstance(listeners, dict):
            mass = listeners.get('mass', {})
            if isinstance(mass, dict):
                dry_mass = mass.get('dry_mass', 0.0)

        threshold = states.get("division_threshold", float('inf'))

        full_chrom = states.get('unique', {}).get('full_chromosome')
        n_chromosomes = 0
        if full_chrom is not None and hasattr(full_chrom, 'dtype'):
            if '_entryState' in full_chrom.dtype.names:
                n_chromosomes = full_chrom['_entryState'].sum()

        if dry_mass < threshold or n_chromosomes < 2:
            return {}

        if self.division_detected:
            return {}

        # --- DIVISION ---
        self.division_detected = True
        division_time = states.get("global_time", 0)
        print(f'DIVISION at t={division_time:.0f}s '
              f'(dry_mass={dry_mass:.1f} fg, '
              f'threshold={threshold:.1f} fg, '
              f'chromosomes={n_chromosomes})')

        # Split cell state
        cell_data = {
            'bulk': states['bulk'],
            'unique': states['unique'],
            'environment': states.get('environment', {}),
            'boundary': states.get('boundary', {}),
        }

        from v2ecoli.library.division import divide_cell
        d1_data, d2_data = divide_cell(cell_data)

        # Preserve global_time in daughters
        d1_data['global_time'] = division_time
        d2_data['global_time'] = division_time

        # Build complete daughter cell states
        d1_id, d2_id = daughter_phylogeny_id(self.agent_id)

        if self._configs:
            from v2ecoli.partitioned.generate_partitioned import (
                build_partitioned_document_from_configs)
            d1_seed = (self._seed + 1) % (2**31)
            d2_seed = (self._seed + 2) % (2**31)

            d1_doc = build_partitioned_document_from_configs(
                d1_data, self._configs, self._unique_names,
                dry_mass_inc_dict=self.dry_mass_inc_dict,
                core=self.core, seed=d1_seed)
            d2_doc = build_partitioned_document_from_configs(
                d2_data, self._configs, self._unique_names,
                dry_mass_inc_dict=self.dry_mass_inc_dict,
                core=self.core, seed=d2_seed)

            d1_cell = d1_doc['state']['agents']['0']
            d2_cell = d2_doc['state']['agents']['0']
        else:
            # No configs — can't build daughters, just report
            print(f'  No configs available for daughter generation')
            return {}

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

    def update(self, state, interval=None):
        return self.next_update(1.0, state)
