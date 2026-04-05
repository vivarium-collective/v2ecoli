"""
Cell division steps for v2ecoli.

MarkDPeriod: Sets division flag after D period has elapsed
Division: Detects division condition, saves pre-division state
"""

import numpy as np

from v2ecoli.steps.base import V2Step
from v2ecoli.library.schema import attrs
from v2ecoli.library.units import units


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + "0", str(mother_id) + "1"]


class MarkDPeriod(V2Step):
    """Set division flag after D period has elapsed."""

    name = "mark_d_period"
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)
        self.parameters = config or {}

    def ports_schema(self):
        return {
            "full_chromosome": {},
            "global_time": {"_default": 0.0},
            "divide": {"_default": False, "_updater": "set"},
        }

    def next_update(self, timestep, states):
        full_chrom = states.get("full_chromosome")
        if full_chrom is None or not hasattr(full_chrom, 'dtype'):
            return {}

        division_time, has_triggered_division = attrs(
            full_chrom, ["division_time", "has_triggered_division"])

        if len(division_time) < 2:
            return {}

        divide_at_time = division_time[~has_triggered_division].min()
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
    """Detect division condition and report it.

    When the division condition is met (dry mass >= threshold with
    2+ chromosomes), prints a message and sets a flag. Full daughter
    cell generation is not yet implemented.
    """

    name = "division"
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)
        self.parameters = config or {}
        self.agent_id = self.parameters.get('agent_id', '0')
        self.dry_mass_inc_dict = self.parameters.get('dry_mass_inc_dict', {})
        self.division_detected = False
        self.division_time = None

        # Division mass multiplier
        import binascii
        seed = self.parameters.get('seed', 0)
        self.division_mass_multiplier = 1
        if self.parameters.get('division_threshold') == 'mass_distribution':
            div_seed = binascii.crc32(b"CellDivision", seed) & 0xFFFFFFFF
            div_rng = np.random.RandomState(seed=div_seed)
            self.division_mass_multiplier = div_rng.normal(loc=1.0, scale=0.1)

    def ports_schema(self):
        return {
            "division_variable": {"_default": 0.0},
            "full_chromosome": {},
            "media_id": {"_default": "minimal"},
            "division_threshold": {
                "_default": self.parameters.get('division_threshold', 2000.0),
                "_updater": "set",
            },
            "global_time": {"_default": 0.0},
        }

    def next_update(self, timestep, states):
        # Set threshold on first timestep if using mass_distribution
        if states.get("division_threshold") == "mass_distribution":
            media_id = states.get("media_id", "minimal")
            dry_mass_inc = self.dry_mass_inc_dict.get(media_id)
            if dry_mass_inc is not None:
                return {
                    "division_threshold": (
                        states["division_variable"]
                        + dry_mass_inc.asNumber(units.fg)
                        * self.division_mass_multiplier
                    )
                }
            return {}

        division_variable = states.get("division_variable", 0)
        threshold = states.get("division_threshold", float('inf'))

        full_chrom = states.get("full_chromosome")
        n_chromosomes = 0
        if full_chrom is not None and hasattr(full_chrom, 'dtype'):
            n_chromosomes = full_chrom["_entryState"].sum()

        if division_variable >= threshold and n_chromosomes >= 2:
            if not self.division_detected:
                self.division_detected = True
                self.division_time = states.get("global_time", 0)
                print(f"DIVISION DETECTED at t={self.division_time:.0f}s "
                      f"(dry_mass={division_variable:.1f} fg, "
                      f"threshold={threshold:.1f} fg, "
                      f"chromosomes={n_chromosomes})")

        return {}

    def update(self, state, interval=None):
        return self.next_update(1.0, state)
