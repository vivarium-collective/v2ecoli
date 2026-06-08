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
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.types.quantity import ureg as units


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
        self._cache_dir = self.parameters.get('cache_dir', 'out/cache')

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
                # Under units-on-ports, dry_mass is a pint Quantity[fg]; coerce
                # to a plain fg float so the arithmetic below doesn't raise a
                # DimensionalityError (which process-bigraph silently swallows,
                # leaving division_threshold stuck on "mass_distribution").
                dry_mass = fg_magnitude(dry_mass)
                return {
                    "division_threshold": (
                        dry_mass
                        + dry_mass_inc.to(units.fg).magnitude
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
        # Coerce both sides to plain fg floats — dry_mass may be a Quantity[fg]
        # (units-on-ports); a Quantity vs float comparison raises (and is
        # swallowed), which would silently prevent division.
        dry_mass = fg_magnitude(dry_mass)
        threshold = fg_magnitude(states.get("division_threshold", float('inf')))

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

        try:
            cell_data = {
                'bulk': states['bulk'],
                'unique': states['unique'],
                'environment': states.get('environment', {}),
                'boundary': states.get('boundary', {}),
            }

            from v2ecoli.library.division import divide_cell
            d1_data, d2_data = divide_cell(cell_data)

            d1_data['global_time'] = division_time
            d2_data['global_time'] = division_time

            d1_id, d2_id = daughter_phylogeny_id(self.agent_id)

            # Build daughter docs via the new composite generator API.
            # baseline() loads wiring from the cache; we then overlay the
            # daughter's divided biological state on top.
            from v2ecoli.composites.baseline import baseline, seed_mass_listener
            from v2ecoli.composites._helpers import (
                _PARQUET_EMITTER_OVERRIDE, set_parquet_emitter_override)
            d1_seed = (self._seed + 1) % (2**31)
            d2_seed = (self._seed + 2) % (2**31)

            def _build_daughter_doc(d_data, seed, daughter_id):
                # When a parquet-emitter override is active, the daughter's
                # emitter would otherwise inherit the PARENT's static partition
                # metadata (generation, agent_id) from the global override, and
                # its _write_configuration (run in __init__) would DELETE the
                # parent's fully-written history partition at division — the
                # parent emits its whole life to generation=N/agent_id=<id>, the
                # daughter spawns on the SAME path and wipes it, leaving only
                # the daughter's birth rows (the early-cycle-slice symptom).
                # Re-point the override to the daughter's OWN slot
                # (generation=N+1, agent_id=<parent><suffix>) for the duration
                # of the build so the daughter partitions to
                # generation=N+1/agent_id=P0|P1 and never touches the parent's
                # data. The next runner-driven generation re-wipes and rewrites
                # that daughter slot cleanly.
                import copy as _copy
                _saved = _PARQUET_EMITTER_OVERRIDE
                if _saved is not None:
                    _ovr = _copy.deepcopy(_saved)
                    _meta = _ovr.setdefault('metadata', {})
                    # Derive the daughter slot from the PARENT's override
                    # metadata (the runner's per-generation identity), NOT from
                    # self.agent_id — inside a composite the cell is always the
                    # 'agents/0' key, so self.agent_id is "0" for every runner
                    # generation. Using it would give generation=2/agent_id=00
                    # for ALL gens, colliding with (and wiping) runner gen 2.
                    _parent_aid = str(_meta.get('agent_id', self.agent_id))
                    _parent_gen = _meta.get('generation')
                    if _parent_gen is None:
                        _parent_gen = len(_parent_aid)
                    _suffix = daughter_id[len(str(self.agent_id)):] or '0'
                    _meta['agent_id'] = _parent_aid + _suffix
                    _meta['generation'] = int(_parent_gen) + 1
                    set_parquet_emitter_override(_ovr)
                try:
                    doc = baseline(
                        core=self.core, seed=seed, cache_dir=self._cache_dir)
                finally:
                    if _saved is not None:
                        set_parquet_emitter_override(_saved)
                agent = doc['state']['agents']['0']
                for key in ('bulk', 'unique', 'environment', 'boundary'):
                    if key in d_data:
                        agent[key] = d_data[key]
                agent['listeners']['mass'] = {'dry_mass': 0.0, 'cell_mass': 0.0}
                seed_mass_listener(agent, self.core)
                return doc

            d1_doc = _build_daughter_doc(d1_data, d1_seed, d1_id)
            d2_doc = _build_daughter_doc(d2_data, d2_seed, d2_id)

            d1_cell = d1_doc['state']['agents']['0']
            d2_cell = d2_doc['state']['agents']['0']

            print(f'  DAUGHTERS: {d1_id} (bulk={d1_data["bulk"]["count"].sum()}) '
                  f'+ {d2_id} (bulk={d2_data["bulk"]["count"].sum()})')

            # Mirror vEcoli's pre-divide ``emitter.finalize()`` hook
            # (ecoli_master_sim.py DivisionDetected branch: success=True then
            # finalize()). The parent ParquetEmitter has rows buffered since its
            # last batch flush; without an explicit close() they vanish — and
            # no success sentinel is written — when this _remove update tears
            # the agent subtree down. The close-flush-and-unregister itself is
            # the FRAMEWORK-generic ``finalize_emitter_for_agent`` from
            # pbg-emitters; the v2ecoli-specific glue is only *which key* to
            # finalize: the parent is registered under the parquet override's
            # metadata.agent_id (the runner's per-generation identity: "0",
            # "00", ...). self.agent_id inside the composite is always "0" (the
            # agents/0 key), so for gens >= 2 a self.agent_id lookup misses;
            # use the override metadata as the canonical key.
            from v2ecoli.composites._helpers import finalize_emitter_for_agent
            _ovr_meta = (_PARQUET_EMITTER_OVERRIDE or {}).get('metadata') or {}
            _emitter_key = str(_ovr_meta.get('agent_id', self.agent_id))
            finalize_emitter_for_agent(_emitter_key, success=True)

            return {
                'agents': {
                    '_remove': [self.agent_id],
                    '_add': [
                        (d1_id, d1_cell),
                        (d2_id, d2_cell),
                    ],
                },
            }
        except Exception as e:
            # Surface failures loudly — process-bigraph otherwise swallows
            # exceptions raised from a step's next_update.
            import traceback
            print(f'  DIVISION FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            raise

    def update(self, state, interval=None):
        return self.next_update(1.0, state)
