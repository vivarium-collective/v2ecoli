"""
===============
DnaA-box binding
===============

A Process that tracks per-site DnaA-box occupancy across the chromosome,
oriC, and the dnaA promoter. Each tick:

  1. Reads the current DnaA-ATP and DnaA-ADP bulk counts.
  2. Computes the FREE DnaA-ATP / DnaA-ADP pools (bulk count minus the
     fraction already bound to a DnaA-box).
  3. Updates the per-site occupancy probabilistically using equilibrium
     binding: P_bound = [free DnaA] / (Kd + [free DnaA]), where the
     free DnaA species pool is filtered by the site's form_preference
     (high-affinity sites accept both ATP- and ADP-forms; low-affinity
     sites accept ATP-form only).
  4. Emits region/affinity-class aggregates to
     ``listeners.dnaA_binding.*``.

This is the v1 (equilibrium-occupancy, non-cooperative) model called
out in the dnaa-03 study yaml. v2 (Hill cooperativity at oriC) and v3
(IHF/Fis accessory factors) belong to the cooperative-oriC follow-up
study tracked as ``dnaa-03-EQ-02``.

Internal state
--------------
Per-site occupancy + bound_form is held in a numpy structured array on the
Process instance (``self._occupancy``). It is NOT exposed in the composite's
state tree as a unique-molecule store; the YAML's v1 spec acknowledges
this simplification ("does_not_resolve: per-site cooperativity"). If a
later phase needs per-site introspection from outside the Process, the
listener already emits aggregates by region+affinity_class; a richer
listener emit is a small extension.

Listener emit
-------------
  listeners.dnaA_binding.chromosome.occupied_fraction       float
  listeners.dnaA_binding.chromosome.occupied_count          int
  listeners.dnaA_binding.chromosome.total_sites             int
  listeners.dnaA_binding.oric.high_affinity_occupied        float
  listeners.dnaA_binding.oric.low_affinity_occupied         float
  listeners.dnaA_binding.oric.occupied_count                int
  listeners.dnaA_binding.dnaap.occupied                     float
  listeners.dnaA_binding.dnaap.occupied_count               int
  listeners.dnaA_binding.free_atp                           int
  listeners.dnaA_binding.free_adp                           int
  listeners.dnaA_binding.free_total                         int  (free_atp + free_adp)
  listeners.dnaA_binding.bound_total                        int

Biology notes
-------------
- Free-DnaA-ATP and free-DnaA-ADP pools come from the dnaa-02 species
  split. The Process is composite-recipe-paired with dnaa-02; running
  dnaa-03 without dnaa-02's Steps means DnaA-ADP is always 0 and the
  low-affinity oriC sites (ATP-preferential) will never gate properly.
- Kd values are the v1 defaults from molecular_info doc:
    high-affinity: 1 nM
    low-affinity:  100 nM
  Effective on-rate per tick is computed from the box-class Kd plus
  the current free DnaA pool size and cell volume.
- The MONOMER0-160[c] / MONOMER0-4565[c] bulk values continue to count
  TOTAL DnaA-ATP / DnaA-ADP (bound + free). The Process never modifies
  bulk; it tracks the bound count internally and exposes the free count
  via the listener emit. This is a deliberate v1 simplification — a
  later phase may move binding into a bulk-modifying flow when fork
  passage (dnaa-04 req-2) needs to release bound DnaA back into the
  free pool.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-box-binding"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
    "listeners_mass": ("listeners", "mass"),
}

DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"

# Cell-physics constants for converting count <-> concentration.
N_AVOGADRO = 6.022e23  # /mol


class DnaaBoxBinding(Step):
    """Equilibrium DnaA-box binding Process (v1: non-cooperative)."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "kd_high_nM": {"_type": "float", "_default": 1.0},
        "kd_low_nM":  {"_type": "float", "_default": 100.0},
        "seed":       {"_type": "integer", "_default": 0},
        "enable_oric_binding":  {"_type": "boolean", "_default": True},
        "enable_dnaap_binding": {"_type": "boolean", "_default": True},
        # oriC low-affinity cooperativity (Stage-1, expert round-1 2026-05-21):
        # the 8 weak oriC DnaA-ATP boxes have a per-site Kd GRADIENT decreasing
        # with occupancy — 50 nM (first site to fill) → 1 nM (last) linearly
        # (wcm_stage1_parameters). This "differential binding affinity by
        # occupancy" replaces a flat low Kd / Hill coefficient (which can't
        # track per-box occupancy). Disable to fall back to flat kd_low_nM.
        "oric_low_cooperative":   {"_type": "boolean", "_default": True},
        "kd_oric_low_max_nM":     {"_type": "float", "_default": 50.0},
        "kd_oric_low_min_nM":     {"_type": "float", "_default": 1.0},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        from v2ecoli.data.dnaa_box_catalog import active_boxes

        self.kd_high_nM = float(self.parameters["kd_high_nM"])
        self.kd_low_nM = float(self.parameters["kd_low_nM"])
        self.oric_low_cooperative = bool(self.parameters.get("oric_low_cooperative", True))
        self.kd_oric_low_max_nM = float(self.parameters.get("kd_oric_low_max_nM", 50.0))
        self.kd_oric_low_min_nM = float(self.parameters.get("kd_oric_low_min_nM", 1.0))
        self.enable_oric = bool(self.parameters["enable_oric_binding"])
        self.enable_dnaap = bool(self.parameters["enable_dnaap_binding"])
        self.seed = int(self.parameters["seed"])
        self.random_state = np.random.RandomState(seed=self.seed)

        boxes = active_boxes()
        # Optionally drop oriC / dnaAp from the active set (for diagnostic
        # variants — see dnaa-03 simulation_set.chromosomal-boxes-only).
        if not self.enable_oric:
            boxes = [b for b in boxes if b.region_type != "ORIC"]
        if not self.enable_dnaap:
            boxes = [b for b in boxes if b.region_type != "DNAAP"]

        n = len(boxes)
        self._n_sites = n
        # Parallel arrays (faster than per-site Python).
        self._region = np.array([b.region_type for b in boxes])
        self._affinity = np.array([b.affinity_class for b in boxes])
        self._form_pref = np.array([b.form_preference for b in boxes])
        # Bookkeeping arrays mutated each tick.
        self._occupied = np.zeros(n, dtype=bool)
        # bound_form: 0 = unbound, 1 = ATP, 2 = ADP
        self._bound_form = np.zeros(n, dtype=np.int8)

        # Precompute per-site Kd (nM) from affinity class.
        kd = np.where(self._affinity == "high",
                      self.kd_high_nM, self.kd_low_nM)
        self._kd_nM = kd
        # Boolean masks for emit-time aggregates.
        self._mask_chrom = self._region == "CHROMOSOMAL_TITRATION"
        self._mask_oric = self._region == "ORIC"
        self._mask_dnaap = self._region == "DNAAP"
        self._mask_oric_high = self._mask_oric & (self._affinity == "high")
        self._mask_oric_low = self._mask_oric & (self._affinity == "low")
        # Mask: does this site accept ADP-bound DnaA?
        self._accepts_adp = self._form_pref == "both"

        # oriC low-affinity cooperativity: give the weak oriC boxes a per-site
        # Kd GRADIENT (50→1 nM linear) instead of the flat kd_low. The tightest
        # site (Kd≈1) fills last as occupancy rises, so occupancy is a graded,
        # cooperative function of [DnaA-ATP] — the "differential affinity by
        # occupancy" the reviewer asked for (vs a Hill coefficient that can't
        # resolve per-box occupancy). Indices ordered by the catalog; the
        # gradient is over the count, not genomic position.
        oric_low_idx = np.where(self._mask_oric_low)[0]
        if self.oric_low_cooperative and len(oric_low_idx) > 0:
            n = len(oric_low_idx)
            grad = (np.linspace(self.kd_oric_low_max_nM, self.kd_oric_low_min_nM, n)
                    if n > 1 else np.array([self.kd_oric_low_min_nM]))
            self._kd_nM = np.array(self._kd_nM, dtype=float)
            self._kd_nM[oric_low_idx] = grad

        self._atp_idx: int | None = None
        self._adp_idx: int | None = None

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "listeners_mass": {
                "cell_mass": {"_type": "float", "_default": 0.0},
                "volume":    {"_type": "float", "_default": 0.0},
            },
        }

    def outputs(self):
        return {
            "listeners": {
                "dnaA_binding": {
                    "chromosome": {
                        "occupied_fraction":  {"_type": "overwrite[float]",   "_default": 0.0},
                        "occupied_count":     {"_type": "overwrite[integer]", "_default": 0},
                        "total_sites":        {"_type": "overwrite[integer]", "_default": 0},
                    },
                    "oric": {
                        "high_affinity_occupied": {"_type": "overwrite[float]",   "_default": 0.0},
                        "low_affinity_occupied":  {"_type": "overwrite[float]",   "_default": 0.0},
                        "occupied_count":         {"_type": "overwrite[integer]", "_default": 0},
                    },
                    "dnaap": {
                        "occupied":         {"_type": "overwrite[float]",   "_default": 0.0},
                        "occupied_count":   {"_type": "overwrite[integer]", "_default": 0},
                    },
                    "free_atp":    {"_type": "overwrite[integer]", "_default": 0},
                    "free_adp":    {"_type": "overwrite[integer]", "_default": 0},
                    "free_total":  {"_type": "overwrite[integer]", "_default": 0},
                    "bound_total": {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    def _free_concentration_nM(self, free_count, cell_volume_L):
        """Convert free count to nM concentration."""
        if free_count <= 0 or cell_volume_L <= 0:
            return 0.0
        moles = free_count / N_AVOGADRO
        molar = moles / cell_volume_L
        return molar * 1e9  # nM

    def _equilibrium_occupied(self, n_sites_class, kd_nM, free_concentration_nM):
        """Compute expected number of sites occupied at equilibrium.

        P_bound = [free] / (Kd + [free]); expected occupied = n * P_bound.
        """
        if n_sites_class <= 0 or kd_nM <= 0:
            return 0
        denom = kd_nM + free_concentration_nM
        if denom <= 0:
            return 0
        p_bound = free_concentration_nM / denom
        return p_bound

    def update(self, states, interval=None):
        if self._atp_idx is None:
            self._atp_idx = int(bulk_name_to_idx(DNAA_ATP_ID, states["bulk"]["id"]))
            self._adp_idx = int(bulk_name_to_idx(DNAA_ADP_ID, states["bulk"]["id"]))

        # Total DnaA in each form (free + bound).
        total_atp = int(counts(states["bulk"], self._atp_idx))
        total_adp = int(counts(states["bulk"], self._adp_idx))

        # Subtract currently-bound from total to get the free pool.
        bound_atp = int(np.sum(self._bound_form == 1))
        bound_adp = int(np.sum(self._bound_form == 2))
        free_atp = max(0, total_atp - bound_atp)
        free_adp = max(0, total_adp - bound_adp)

        # Read cell volume (L) for concentration math.
        try:
            volume_fl = float(states["listeners_mass"]["volume"])
        except (KeyError, TypeError):
            volume_fl = 1.0
        cell_volume_L = max(volume_fl, 1e-3) * 1e-15

        # Free concentrations (nM).
        c_atp_nM = self._free_concentration_nM(free_atp, cell_volume_L)
        c_adp_nM = self._free_concentration_nM(free_adp, cell_volume_L)

        # Update each site stochastically using its PER-SITE Kd
        # (self._kd_nM). This is fully per-site rather than per-class so the
        # oriC low-affinity Kd gradient (50→1 nM) takes effect — each weak
        # oriC box has its own affinity, giving graded/cooperative occupancy.
        #   - atp_preferential sites: P_bound = c_atp / (Kd_i + c_atp)
        #   - "both" sites: pool the two free concentrations
        new_occupied = np.zeros(self._n_sites, dtype=bool)
        new_form = np.zeros(self._n_sites, dtype=np.int8)
        kd_site = np.asarray(self._kd_nM, dtype=float)

        atp_only_mask = self._form_pref == "atp_preferential"
        both_mask = self._form_pref == "both"

        # ATP-only sites: per-site p_bound from c_atp and per-site Kd.
        if atp_only_mask.any():
            kds = kd_site[atp_only_mask]
            denom = kds + c_atp_nM
            p = np.where(denom > 0, c_atp_nM / denom, 0.0)
            occ = self.random_state.random(p.shape[0]) < p
            new_occupied[atp_only_mask] = occ
            new_form[atp_only_mask] = np.where(occ, 1, 0)

        # "both" sites: pool free ATP+ADP; per-site p_bound; assign form by
        # the free-pool ratio.
        if both_mask.any():
            kds = kd_site[both_mask]
            total_free = c_atp_nM + c_adp_nM
            denom = kds + total_free
            p = np.where(denom > 0, total_free / denom, 0.0)
            occ = self.random_state.random(p.shape[0]) < p
            p_atp_given_bound = (c_atp_nM / total_free) if total_free > 0 else 0.0
            form_draws = self.random_state.random(p.shape[0])
            forms_for_bound = np.where(form_draws < p_atp_given_bound, 1, 2)
            new_form[both_mask] = np.where(occ, forms_for_bound, 0)
            new_occupied[both_mask] = occ

        # Cap by free pool. If we accidentally tried to bind more than the
        # free pool, randomly evict the excess. Iterate at most a few times
        # because the equilibrium draw is unlikely to overshoot massively.
        for form_id, free_pool in ((1, free_atp), (2, free_adp)):
            requested = int(np.sum(new_form == form_id))
            if requested > free_pool:
                excess = requested - free_pool
                indices = np.where(new_form == form_id)[0]
                evict = self.random_state.choice(indices, excess, replace=False)
                new_form[evict] = 0
                new_occupied[evict] = False

        self._occupied = new_occupied
        self._bound_form = new_form

        # Compute listener aggregates
        bound_atp_now = int(np.sum(new_form == 1))
        bound_adp_now = int(np.sum(new_form == 2))
        bound_total = bound_atp_now + bound_adp_now
        free_atp_emit = max(0, total_atp - bound_atp_now)
        free_adp_emit = max(0, total_adp - bound_adp_now)
        free_total_emit = free_atp_emit + free_adp_emit

        chrom_total = int(self._mask_chrom.sum())
        chrom_occupied = int(np.sum(new_occupied & self._mask_chrom))
        chrom_frac = chrom_occupied / chrom_total if chrom_total else 0.0

        oric_total = int(self._mask_oric.sum())
        oric_high_total = int(self._mask_oric_high.sum())
        oric_low_total = int(self._mask_oric_low.sum())
        oric_occupied = int(np.sum(new_occupied & self._mask_oric))
        oric_high_occ = int(np.sum(new_occupied & self._mask_oric_high))
        oric_low_occ = int(np.sum(new_occupied & self._mask_oric_low))
        oric_high_frac = oric_high_occ / oric_high_total if oric_high_total else 0.0
        oric_low_frac = oric_low_occ / oric_low_total if oric_low_total else 0.0

        dnaap_total = int(self._mask_dnaap.sum())
        dnaap_occupied = int(np.sum(new_occupied & self._mask_dnaap))
        dnaap_frac = dnaap_occupied / dnaap_total if dnaap_total else 0.0

        return {
            "listeners": {
                "dnaA_binding": {
                    "chromosome": {
                        "occupied_fraction": float(chrom_frac),
                        "occupied_count":    int(chrom_occupied),
                        "total_sites":       int(chrom_total),
                    },
                    "oric": {
                        "high_affinity_occupied": float(oric_high_frac),
                        "low_affinity_occupied":  float(oric_low_frac),
                        "occupied_count":         int(oric_occupied),
                    },
                    "dnaap": {
                        "occupied":       float(dnaap_frac),
                        "occupied_count": int(dnaap_occupied),
                    },
                    "free_atp":    int(free_atp_emit),
                    "free_adp":    int(free_adp_emit),
                    "free_total":  int(free_total_emit),
                    "bound_total": int(bound_total),
                },
            },
        }
