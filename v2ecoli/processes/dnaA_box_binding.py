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
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.cell_volume_L = float(self.parameters.get(
            'cell_volume_L', _DEFAULT_CELL_VOLUME_L))
        self.kd_high = float(self.parameters.get(
            'kd_high_nM', DNAA_BOX_HIGH_AFFINITY_KD_NM))
        self.kd_low = float(self.parameters.get(
            'kd_low_nM', DNAA_BOX_LOW_AFFINITY_KD_NM_LOWER_BOUND))
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
                },
            },
        }

    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def _sample_named_region(self, region: str, atp_nM: float,
                              adp_nM: float) -> tuple[int, int, int]:
        """Sample bound counts for one named region using its
        per-tier affinity profile. Returns (bound_high, bound_low,
        bound_very_low). The total occupancy of the region is the sum.

        Tier semantics, sourced from the curated PDF:
          * high — Kd ~1 nM, binds both DnaA-ATP and DnaA-ADP.
          * low — Kd > 100 nM, DnaA-ATP-preferential, cooperative.
          * very_low — Kd ≫ 100 nM, both forms (rare; e.g.
            box3 at the dnaA promoter). Modeled at 10 × Kd_low.

        ``np.random.RandomState.binomial(n, p)`` returns a sample
        from the binomial distribution — equivalent to the sum of
        ``n`` independent Bernoulli(p) samples but ~10× faster than
        looping when n is large.
        """
        profile = PER_REGION_AFFINITY_PROFILE.get(region, {})
        n_h = int(profile.get('high', 0))
        n_l = int(profile.get('low', 0))
        n_vl = int(profile.get('very_low', 0))

        c_total = atp_nM + adp_nM
        bound_h = bound_l = bound_vl = 0

        if n_h > 0 and c_total > 0:
            p_h = c_total / (self.kd_high + c_total)
            bound_h = int(self.random_state.binomial(n_h, p_h))
        if n_l > 0 and atp_nM > 0:
            # Low-affinity oriC boxes are DnaA-ATP-preferential per
            # the curated reference (Speck/Weigel/Messer 1999, in PDF).
            p_l = atp_nM / (self.kd_low + atp_nM)
            bound_l = int(self.random_state.binomial(n_l, p_l))
        if n_vl > 0 and c_total > 0:
            kd_vl = self.kd_low * 10.0
            p_vl = c_total / (kd_vl + c_total)
            bound_vl = int(self.random_state.binomial(n_vl, p_vl))
        return bound_h, bound_l, bound_vl

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

        # ----- Named regions: per-tier sampling from curated profile.
        # Each named region has a (high, low, very_low) box count from
        # the curated PDF; we sample each tier separately so the
        # load-and-trigger biology is preserved. The numbers in
        # PER_REGION_AFFINITY_PROFILE are independent of how many
        # strict-consensus matches the bioinformatic search found —
        # the curated count is the source of truth.
        region_bound_counts: dict[str, int] = {
            r: 0 for r in PER_REGION_AFFINITY_PROFILE}
        oriC_high = oriC_low = 0
        for region in PER_REGION_AFFINITY_PROFILE:
            b_h, b_l, b_vl = self._sample_named_region(
                region, atp_nM, adp_nM)
            region_bound_counts[region] = b_h + b_l + b_vl
            if region == 'oriC':
                oriC_high, oriC_low = b_h, b_l

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

        bound_other = 0
        if n_active_other > 0 and atp_nM > 0:
            p_other = atp_nM / (self.kd_low + atp_nM)
            bound_other = int(
                self.random_state.binomial(n_active_other, p_other))
        region_bound_counts['other'] = bound_other

        total_bound = int(sum(region_bound_counts.values()))
        total_active = int(n_active_named + n_active_other)
        fraction_bound = (total_bound / total_active
                           if total_active > 0 else 0.0)

        # We do *not* write back to the DnaA_bound field on the unique
        # store. The 'set' update mode requires the new value array to
        # match the current active-box count exactly, but
        # chromosome_structure adds and deletes boxes during fork
        # passage in the same tick, so the count at apply-time can
        # differ from sample-time and the set raises a numpy size
        # error. Instead, the binding process is a *listener-only*
        # report on equilibrium occupancy. Phase 3's initiation gate
        # could read bound_oric / bound_oric_low directly to capture
        # the load-and-trigger switch — that's a follow-up.
        return {
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
                },
            },
        }
