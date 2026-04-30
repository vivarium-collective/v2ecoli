"""
================
DnaA Box Binding
================

Binds DnaA-ATP / DnaA-ADP to chromosomal DnaA boxes using a
**per-affinity-tier** model that mirrors the curated reference. At
each named regulatory region (oriC, dnaA promoter, datA, DARS1,
DARS2), the curated PDF reports a mix of high- and low-affinity
boxes; this Step samples each tier separately so the load-and-trigger
biology of oriC (3 high-affinity + 8 cooperative low-affinity boxes)
is preserved instead of being collapsed into a single Kd.

Why per-tier matters: at fast-growth E. coli the DnaA-ATP pool sits
at hundreds of nM. With Kd_high ≈ 1 nM, ``p = c/(Kd+c) ≈ 0.99``, so
high-affinity boxes appear saturated essentially from t=0. The
low-affinity boxes (Kd > 100 nM, DnaA-ATP-preferential) only fill as
the pool rises further — that gradient is what makes initiation a
sharp switch. Treating every oriC box at Kd_high (the prior
collapsed model) over-reports oriC occupancy and erases the switch.

Mathematical model (per region, per tick):

    [DnaA-ATP]    = n_atp / (V * N_A) * 1e9      [nM]
    [DnaA-ADP]    = n_adp / (V * N_A) * 1e9
    [DnaA-total]  = [DnaA-ATP] + [DnaA-ADP]

For each named region with profile (n_high, n_low, n_very_low):

    high tier (binds both ATP and ADP at Kd_high):
        p_h = [DnaA-total] / (Kd_high + [DnaA-total])
        bound_h ~ Binomial(n_high, p_h)

    low tier (DnaA-ATP-preferential at Kd_low):
        p_l = [DnaA-ATP] / (Kd_low + [DnaA-ATP])
        bound_l ~ Binomial(n_low, p_l)

    very_low tier (binds both forms at ~10 × Kd_low):
        p_v = [DnaA-total] / (10·Kd_low + [DnaA-total])
        bound_v ~ Binomial(n_very_low, p_v)

    bound_<region> = bound_h + bound_l + bound_v

For the genomic-background "other" boxes (the bioinformatic
strict-consensus matches outside any named region), the Step
iterates over the active coordinates and samples each at Kd_low,
ATP-preferential — same as before, so the background titration
behavior is unchanged.

Source of the per-region tier counts:
``v2ecoli.data.replication_initiation.PER_REGION_AFFINITY_PROFILE``,
derived from the per-box ``affinity_class`` annotations in
``ORIC.dnaA_boxes`` / ``DNAA_PROMOTER.dnaA_boxes`` (curated against
the PDF; see docs/references/replication_initiation.md).

References (in the curated PDF):
    Katayama et al. 2017, Frontiers in Microbiology 8:2496.
    Kasho, Ozaki, Katayama 2023, Int. J. Mol. Sci. 24(14):11572.
    Speck, Weigel, Messer 1999, EMBO J. 18(21):6169–6176.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import (
    DEFAULT_REGION_BINDING_RULE,
    DNAA_ADP_BULK_ID, DNAA_ATP_BULK_ID,
    DNAA_BOX_HIGH_AFFINITY_KD_NM,
    DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND,
    PER_REGION_AFFINITY_PROFILE,
    REGION_BINDING_RULES,
    REGION_BOUNDARIES,
    region_for_coord,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY


NAME = "dnaA_box_binding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# Avogadro's number; cellular volume default (typical fast-growth E. coli).
_N_AVOGADRO = 6.022e23
_DEFAULT_CELL_VOLUME_L = 1e-15  # 1 fL


class DnaABoxBinding(Step):
    """DnaA Box Binding Step

    Per active DnaA box, sample bound/unbound from the equilibrium
    occupancy probability. Updates the ``DnaA_bound`` attribute via
    the unique-array ``set`` interface. Emits per-region bound counts.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'seed': 'integer{0}',
        'cell_volume_L': f'float{{{_DEFAULT_CELL_VOLUME_L:e}}}',
        'kd_high_nM': f'float{{{DNAA_BOX_HIGH_AFFINITY_KD_NM}}}',
        'kd_low_nM': f'float{{{DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND}}}',
        # When True, the binding step decrements bulk[DnaA-ATP] /
        # bulk[DnaA-ADP] by the net change in bound count each tick,
        # so the genomic-background boxes (the largest population)
        # actually titrate the cytoplasmic pool. Default True; set
        # False to recover the listener-only behavior.
        'enable_titration': 'boolean{true}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.cell_volume_L = float(self.parameters.get(
            'cell_volume_L', _DEFAULT_CELL_VOLUME_L))
        self.kd_high = float(self.parameters.get(
            'kd_high_nM', DNAA_BOX_HIGH_AFFINITY_KD_NM))
        self.kd_low = float(self.parameters.get(
            'kd_low_nM', DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND))
        self.enable_titration = bool(self.parameters.get(
            'enable_titration', True))
        self.random_state = np.random.RandomState(
            seed=int(self.parameters.get('seed', 0)))
        self._bulk_idx = None
        # ``other`` boxes (genomic background) — count cached across
        # ticks. Recomputed only when the active-box count changes
        # (i.e. fork passage adds or removes boxes), since
        # region_for_coord is purely a function of the coordinate.
        self._cached_n_active: int = -1
        self._cached_n_other: int = 0
        # Pre-build sorted (lo, hi) bounds arrays for vectorized
        # named-region classification.
        bounds = list(REGION_BOUNDARIES.values())
        self._region_lo = np.array(
            [lo for lo, _ in bounds], dtype=np.int64)
        self._region_hi = np.array(
            [hi for _, hi in bounds], dtype=np.int64)
        # Titration state: count of DnaA-ATP / DnaA-ADP currently
        # sequestered on chromosomal boxes (across all regions). Each
        # tick we recompute the sequestered count and apply the delta
        # to the cytoplasmic bulk pool.
        self._prev_atp_sequestered: int = 0
        self._prev_adp_sequestered: int = 0

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            # Read-only on DnaA_boxes — the listener emits sampled
            # occupancy without writing back to DnaA_bound; see the
            # comment in update().
            #
            # Titration: the step decrements bulk[DnaA-ATP] /
            # bulk[DnaA-ADP] each tick by the net change in
            # sequestered count, capped at the available free pool
            # to avoid negative counts.
            'bulk': 'bulk_array',
            'listeners': {
                'dnaA_binding': {
                    'total_bound': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'total_active': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'fraction_bound': {
                        '_type': 'overwrite[float]', '_default': []},
                    'bound_oric': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_oric_high': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_oric_low': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_dnaA_promoter': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_datA': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_DARS1': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_DARS2': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'bound_other': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'atp_sequestered': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'adp_sequestered': {
                        '_type': 'overwrite[integer]', '_default': []},
                },
            },
        }

    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def _sample_tier(self, n: int, kd: float, atp_only: bool,
                      atp_nM: float, adp_nM: float) -> tuple[int, int]:
        """Sample one affinity tier of ``n`` boxes, returning
        ``(bound_atp, bound_adp)`` — how many bound a DnaA-ATP vs a
        DnaA-ADP molecule. The split is needed for titration: each
        bound molecule reduces its respective bulk pool.

        ``atp_only`` selects between two cases:
          * True (low-affinity, ATP-preferential per the curated PDF):
            sample at ``p = atp_nM / (kd + atp_nM)``; all bound
            molecules are DnaA-ATP.
          * False (high / very_low, both forms): sample at
            ``p = (atp + adp) / (kd + atp + adp)``; assign each bound
            molecule's nucleotide form proportionally to atp / total
            via a second binomial.
        """
        if n <= 0:
            return 0, 0
        if atp_only:
            if atp_nM <= 0:
                return 0, 0
            p = atp_nM / (kd + atp_nM)
            return int(self.random_state.binomial(n, p)), 0
        c = atp_nM + adp_nM
        if c <= 0:
            return 0, 0
        p = c / (kd + c)
        bound = int(self.random_state.binomial(n, p))
        if bound == 0:
            return 0, 0
        atp_share = atp_nM / c
        bound_atp = int(self.random_state.binomial(bound, atp_share))
        return bound_atp, bound - bound_atp

    def _sample_named_region(self, region: str, atp_nM: float,
                              adp_nM: float):
        """Sample bound counts for one named region using its
        per-tier affinity profile. Returns a ``(b_h, b_l, b_vl,
        bound_atp, bound_adp)`` 5-tuple where the first three are
        per-tier totals (high / low / very_low) and the last two are
        per-form totals across the region (used for titration).

        Tier semantics, sourced from the curated PDF:
          * high — Kd ~1 nM, binds both DnaA-ATP and DnaA-ADP.
          * low — Kd > 100 nM, DnaA-ATP-preferential, cooperative.
          * very_low — Kd ≫ 100 nM, both forms (rare; e.g.
            box3 at the dnaA promoter). Modeled at 10 × Kd_low.
        """
        profile = PER_REGION_AFFINITY_PROFILE.get(region, {})
        n_h = int(profile.get('high', 0))
        n_l = int(profile.get('low', 0))
        n_vl = int(profile.get('very_low', 0))

        h_atp, h_adp = self._sample_tier(
            n_h, self.kd_high, atp_only=False,
            atp_nM=atp_nM, adp_nM=adp_nM)
        # Low-affinity boxes are DnaA-ATP-preferential per the curated
        # reference (Speck/Weigel/Messer 1999, in PDF).
        l_atp, l_adp = self._sample_tier(
            n_l, self.kd_low, atp_only=True,
            atp_nM=atp_nM, adp_nM=adp_nM)
        v_atp, v_adp = self._sample_tier(
            n_vl, self.kd_low * 10.0, atp_only=False,
            atp_nM=atp_nM, adp_nM=adp_nM)

        bound_h = h_atp + h_adp
        bound_l = l_atp + l_adp
        bound_vl = v_atp + v_adp
        bound_atp = h_atp + l_atp + v_atp
        bound_adp = h_adp + l_adp + v_adp
        return bound_h, bound_l, bound_vl, bound_atp, bound_adp

    def update(self, states, interval=None):
        if self._bulk_idx is None:
            ids = states['bulk']['id']
            self._bulk_idx = bulk_name_to_idx(
                [DNAA_ATP_BULK_ID, DNAA_ADP_BULK_ID], ids)

        atp_count, adp_count = counts(states['bulk'], self._bulk_idx)
        atp_count = int(atp_count)
        adp_count = int(adp_count)

        # Concentrations in nM. M = count / (V_L * N_A); nM = M * 1e9.
        denom = self.cell_volume_L * _N_AVOGADRO
        atp_nM = (atp_count / denom) * 1e9 if denom > 0 else 0.0
        adp_nM = (adp_count / denom) * 1e9 if denom > 0 else 0.0

        # ----- Stage 1: sample independent equilibrium demand per
        # tier per region (and for 'other'). This is what each tier
        # would bind if unconstrained by total DnaA. Stage 2 then
        # allocates the actual DnaA budget to those samples in
        # priority order (lowest Kd first = highest affinity wins).
        #
        # Why staged? At fast-growth concentrations the independent
        # equilibrium samples can sum to more DnaA than the cell has —
        # the literature DnaA pool is much smaller than the total box
        # count × occupancy probability. Real biology resolves this
        # through competitive binding (high-affinity boxes win); we
        # approximate that with a priority-ordered allocation.
        demand: dict[tuple[str, str], tuple[int, int]] = {}
        for region, profile in PER_REGION_AFFINITY_PROFILE.items():
            n_h = int(profile.get('high', 0))
            n_l = int(profile.get('low', 0))
            n_vl = int(profile.get('very_low', 0))
            demand[(region, 'high')] = self._sample_tier(
                n_h, self.kd_high, atp_only=False,
                atp_nM=atp_nM, adp_nM=adp_nM)
            demand[(region, 'low')] = self._sample_tier(
                n_l, self.kd_low, atp_only=True,
                atp_nM=atp_nM, adp_nM=adp_nM)
            demand[(region, 'very_low')] = self._sample_tier(
                n_vl, self.kd_low * 10.0, atp_only=False,
                atp_nM=atp_nM, adp_nM=adp_nM)

        # Total active boxes in named regions = curated PDF totals.
        # Used to normalize fraction_bound. Background "other" boxes
        # are added below.
        n_active_named = sum(
            sum(p.values()) for p in PER_REGION_AFFINITY_PROFILE.values())

        # ----- "Other" (genomic background) — vectorized count of
        # active boxes outside any named region, then a single
        # binomial draw at Kd_low (DnaA-ATP-preferential). Per-coord
        # identity doesn't matter for the bound count, only n_other.
        # The count is cached across ticks since it only changes when
        # chromosome_structure adds/removes boxes (fork passage),
        # and that's reflected in active_mask.sum().
        n_active_other = 0
        boxes = states.get('DnaA_boxes')
        if (boxes is not None and hasattr(boxes, 'dtype')
                and '_entryState' in boxes.dtype.names):
            active_mask = boxes['_entryState'].view(np.bool_)
            n_active = int(active_mask.sum())
            if n_active != self._cached_n_active:
                # Recompute n_other. Use the same sorted-bounds arrays
                # built in initialize() so the per-region check is a
                # vector op.
                coords = boxes['coordinates'][active_mask].astype(
                    np.int64, copy=False)
                # In_named[i] = True if coords[i] falls in any named
                # region. Broadcast: (N, 1) <= (R,) -> (N, R), then
                # ANY along the region axis.
                if coords.size > 0:
                    in_named = np.any(
                        (coords[:, None] >= self._region_lo[None, :])
                        & (coords[:, None] <= self._region_hi[None, :]),
                        axis=1)
                    self._cached_n_other = int((~in_named).sum())
                else:
                    self._cached_n_other = 0
                self._cached_n_active = n_active
            n_active_other = self._cached_n_other

        # 'other' boxes (genomic background) are low-affinity,
        # DnaA-ATP-preferential.
        demand[('other', 'low')] = self._sample_tier(
            n_active_other, self.kd_low, atp_only=True,
            atp_nM=atp_nM, adp_nM=adp_nM)
        total_active = int(n_active_named + n_active_other)

        # ----- Stage 2: allocate budget in priority order. Highest
        # affinity (lowest Kd) wins when DnaA is scarce. With the
        # cache's ~120-molecule DnaA pool and ~500 active boxes, this
        # ensures the high-affinity sites at oriC, dnaA_promoter,
        # datA, DARS1, DARS2 saturate before background boxes get any
        # DnaA — which is what the literature describes for
        # competitive binding under titration.
        if self.enable_titration:
            atp_budget = atp_count + self._prev_atp_sequestered
            adp_budget = adp_count + self._prev_adp_sequestered
        else:
            # Listener-only behavior: pretend we have unlimited DnaA
            # so every tier gets its full equilibrium sample.
            atp_budget = float('inf')
            adp_budget = float('inf')

        # Tier priority: lower Kd first.
        TIER_PRIORITY = {'high': 0, 'low': 1, 'very_low': 2}
        ordered_keys = sorted(
            demand.keys(),
            key=lambda k: (TIER_PRIORITY[k[1]], 0 if k[0] != 'other' else 1))

        allocated: dict[tuple[str, str], tuple[int, int]] = {}
        for key in ordered_keys:
            want_atp, want_adp = demand[key]
            got_atp = min(want_atp, int(atp_budget))
            got_adp = min(want_adp, int(adp_budget))
            atp_budget -= got_atp
            adp_budget -= got_adp
            allocated[key] = (got_atp, got_adp)

        # Aggregate per region from the allocation.
        region_bound_counts: dict[str, int] = {
            r: 0 for r in PER_REGION_AFFINITY_PROFILE}
        region_bound_counts['other'] = 0
        oriC_high = oriC_low = 0
        atp_sequestered = 0
        adp_sequestered = 0
        for (region, tier), (b_atp, b_adp) in allocated.items():
            region_bound_counts[region] += b_atp + b_adp
            atp_sequestered += b_atp
            adp_sequestered += b_adp
            if region == 'oriC' and tier == 'high':
                oriC_high = b_atp + b_adp
            elif region == 'oriC' and tier == 'low':
                oriC_low = b_atp + b_adp

        total_bound = int(sum(region_bound_counts.values()))
        fraction_bound = (total_bound / total_active
                           if total_active > 0 else 0.0)

        # ----- Apply titration delta to the cytoplasmic pool.
        bulk_update_entries = []
        if self.enable_titration:
            delta_atp = atp_sequestered - self._prev_atp_sequestered
            delta_adp = adp_sequestered - self._prev_adp_sequestered
            self._prev_atp_sequestered = atp_sequestered
            self._prev_adp_sequestered = adp_sequestered
            if delta_atp != 0 or delta_adp != 0:
                bulk_update_entries.append(
                    (self._bulk_idx,
                     np.array([-delta_atp, -delta_adp], dtype=np.int64)))

        # We do *not* write back to the DnaA_bound field on the unique
        # store. The 'set' update mode requires the new value array to
        # match the current active-box count exactly, but
        # chromosome_structure adds and deletes boxes during fork
        # passage in the same tick, so the count at apply-time can
        # differ from sample-time and the set raises a numpy size
        # error. Instead, the binding process reports occupancy via
        # the listener and titrates the cytoplasmic pool via the bulk
        # update (above).
        return {
            'bulk': bulk_update_entries,
            'listeners': {
                'dnaA_binding': {
                    'total_bound': total_bound,
                    'total_active': total_active,
                    'fraction_bound': fraction_bound,
                    'bound_oric': region_bound_counts['oriC'],
                    'bound_oric_high': oriC_high,
                    'bound_oric_low': oriC_low,
                    'bound_dnaA_promoter': region_bound_counts['dnaA_promoter'],
                    'bound_datA': region_bound_counts['datA'],
                    'bound_DARS1': region_bound_counts['DARS1'],
                    'bound_DARS2': region_bound_counts['DARS2'],
                    'bound_other': region_bound_counts['other'],
                    # Per-form sequestration for the titration listener.
                    'atp_sequestered': int(atp_sequestered),
                    'adp_sequestered': int(adp_sequestered),
                },
            },
        }
