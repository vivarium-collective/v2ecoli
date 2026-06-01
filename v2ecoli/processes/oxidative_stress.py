"""
=============================================================================
Oxidative Stress Response — E. coli Whole-Cell Model Process
=============================================================================

Models the complete H2O2-centred oxidative stress response including:

  1. H2O2 PRODUCTION (Imlay 2013 Nat Rev Microbiol)
     Endogenous: ~14 µM/s from flavoenzyme autoxidation
     External:   configurable challenge (µM/s)

  2. H2O2 SCAVENGING — Michaelis-Menten kinetics
     AhpCF proxy (TrxReduct): kcat=428 s-1, Km=1.4 µM
       (calibrated to reproduce ~20 nM SS, Imlay 2013)
     KatG:  kcat=200 s-1, Km=3.9 mM  (Switala & Loewen 2002)
     KatE:  kcat=53 s-1,  Km=40 mM   (Switala & Loewen 2002)

  3. OxyR TRANSCRIPTIONAL RESPONSE (Zheng et al. 2001 Science)
     Hill activation: Kox=5 µM, n=2
     Upregulates: katG, katE, ahpCF, dps, grxA, trxC (up to 10-fold)
     OxyR reduction rate (kred) controls recovery kinetics.

  4. SoxRS TRANSCRIPTIONAL RESPONSE (Pomposiello & Demple 2001)
     Activated by ROS (proxy: H2O2), K=50 µM, n=1.5
     Upregulates: sodA, sodB, zwf, fumC, nfo (up to 5-fold)

  5. NADPH CONSUMPTION
     AhpCF: 1 NADPH per H2O2 reduced

  6. DNA DAMAGE — Fenton chemistry (Imlay & Linn 1988 Science)
     Rate = k_fenton * [H2O2] * [Fe2+]  (k=76 M-1s-1, [Fe2+]=10 µM)

References
----------
Imlay JA (2013) Nat Rev Microbiol 11:443-454
Zheng M et al. (2001) Science 292:2083-2086
Switala J, Loewen PC (2002) Biochemistry 41:4348-4353
Imlay JA, Linn S (1988) Science 240:1302-1309
"""

import numpy as np
from scipy.optimize import brentq

from v2ecoli.library.schema import numpy_schema, listener_schema, counts, bulk_name_to_idx
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.types.quantity import ureg as units

# ---------------------------------------------------------------------------
NAME = "ecoli-oxidative-stress"
TOPOLOGY = {
    "bulk":       ("bulk",),
    "bulk_total": ("bulk",),
    "listeners":  ("listeners",),
    "timestep":   ("timestep",),
}

# ---------------------------------------------------------------------------
_N_A    = 6.022e23          # Avogadro
_V_CELL = 1.1e-15           # L fallback (Imlay 2013) — overridden by mass listener


def _count_to_uM(n, volume_L):
    """Convert molecule count to µM given cell volume in litres."""
    if volume_L <= 0:
        volume_L = _V_CELL
    return n / (_N_A * volume_L) * 1e6


def _uM_to_count(uM, volume_L):
    """Convert µM concentration to molecule count."""
    if volume_L <= 0:
        volume_L = _V_CELL
    return uM * 1e-6 * _N_A * volume_L


# ---------------------------------------------------------------------------
# Molecule bulk IDs
# ---------------------------------------------------------------------------
_H2O2_ID  = "HYDROGEN-PEROXIDE[c]"
_NADPH_ID = "NADPH[c]"
_AHPC_ID  = "THIOREDOXIN-REDUCT-NADPH-CPLX[c]"   # AhpCF proxy
_KATG_ID  = "HYDROPEROXIDI-CPLX[c]"               # KatG
_KATE_ID  = "HYDROPEROXIDII-CPLX[c]"              # KatE
_H2O_ID   = "WATER[c]"
_PPGPP_ID = "GUANOSINE-5DP-3DP[c]"                # ppGpp (stringent response)
_GTP_ID   = "GTP[c]"                               # GTP (ppGpp substrate)


class OxidativeStress(Step):
    """Oxidative stress response Step process.

    Uses Step (not PartitionedProcess) because it doesn't compete for
    resources — it reads bulk_total and writes deltas directly.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "h2o2_production_rate_uM_per_s": {"_type": "float", "_default": 14.0},
        "external_h2o2_uM":              {"_type": "float", "_default": 0.0},
        "ahpcf_kcat":     {"_type": "float", "_default": 428.0},
        "ahpcf_Km_uM":    {"_type": "float", "_default": 1.4},
        "katg_kcat":      {"_type": "float", "_default": 200.0},
        "katg_Km_uM":     {"_type": "float", "_default": 3900.0},
        "kate_kcat":      {"_type": "float", "_default": 53.0},
        "kate_Km_uM":     {"_type": "float", "_default": 40000.0},
        "oxyr_Kox_uM":    {"_type": "float", "_default": 0.2},
        "oxyr_hill":      {"_type": "float", "_default": 1.5},
        "oxyr_kred":      {"_type": "float", "_default": 0.1},
        "oxyr_max_fold_change": {"_type": "float", "_default": 10.0},
        "soxr_K_uM":      {"_type": "float", "_default": 10.0},
        "soxr_hill":      {"_type": "float", "_default": 1.5},
        "soxrs_max_fold_change": {"_type": "float", "_default": 5.0},
        "k_fenton_per_uM_per_s": {"_type": "float", "_default": 0.076},
        "fe2_free_uM":    {"_type": "float", "_default": 10.0},
        "k_ppgpp_per_uM_h2o2":  {"_type": "float", "_default": 150.0},
        "ppgpp_Km_h2o2_uM":     {"_type": "float", "_default": 1.0},
        "cell_density":   {"_type": "float", "_default": 1100.0},
        "n_avogadro":     {"_type": "float", "_default": _N_A},
        "seed":           {"_type": "integer", "_default": 0},
        "time_step":      {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        p = self.parameters

        self.h2o2_prod_rate_uM_s = float(p["h2o2_production_rate_uM_per_s"])
        self.ext_h2o2_uM         = float(p["external_h2o2_uM"])

        self.ahpcf_kcat  = float(p["ahpcf_kcat"]);  self.ahpcf_Km = float(p["ahpcf_Km_uM"])
        self.katg_kcat   = float(p["katg_kcat"]);   self.katg_Km  = float(p["katg_Km_uM"])
        self.kate_kcat   = float(p["kate_kcat"]);   self.kate_Km  = float(p["kate_Km_uM"])

        self.oxyr_Kox    = float(p["oxyr_Kox_uM"])
        self.oxyr_hill   = float(p["oxyr_hill"])
        self.oxyr_kred   = float(p["oxyr_kred"])
        self.oxyr_fc_max = float(p["oxyr_max_fold_change"])

        self.soxr_K      = float(p["soxr_K_uM"])
        self.soxr_hill   = float(p["soxr_hill"])
        self.soxrs_fc_max = float(p["soxrs_max_fold_change"])

        self.k_fenton    = float(p["k_fenton_per_uM_per_s"])
        self.fe2_free    = float(p["fe2_free_uM"])

        self.k_ppgpp     = float(p["k_ppgpp_per_uM_h2o2"])
        self.ppgpp_Km    = float(p["ppgpp_Km_h2o2_uM"])

        self.cell_density_val = float(p["cell_density"])  # g/L as float
        self.n_avogadro_val   = float(p["n_avogadro"])

        # State (carried across timesteps)
        self.oxyr_ox  = 0.0
        self.soxr_ox  = 0.0
        self.cum_dmg  = 0.0

        # Bulk indices — resolved lazily on first timestep
        self._h2o2_idx  = None
        self._nadph_idx = None
        self._ahpc_idx  = None
        self._katg_idx  = None
        self._kate_idx  = None
        self._h2o_idx   = None
        self._ppgpp_idx = None
        self._gtp_idx   = None
        self._indices_resolved = False

    # ------------------------------------------------------------------
    def inputs(self):
        return {
            "bulk":       {"_type": "bulk_array", "_default": []},
            "bulk_total": {"_type": "bulk_array", "_default": []},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "float[fg]", "_default": 0.0},
                },
                "timed_stress": {
                    "h2o2_signal_uM_s": {"_type": "float", "_default": 0.0},
                },
            },
            "timestep": {"_type": "float", "_default": self.parameters.get("time_step", 1.0)},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "oxidative_stress": {
                    "h2o2_uM":                   {"_type": "overwrite[float]", "_default": 0.0},
                    "oxyr_ox_fraction":          {"_type": "overwrite[float]", "_default": 0.0},
                    "soxr_ox_fraction":          {"_type": "overwrite[float]", "_default": 0.0},
                    "ahpcf_flux_uM_per_s":       {"_type": "overwrite[float]", "_default": 0.0},
                    "katg_flux_uM_per_s":        {"_type": "overwrite[float]", "_default": 0.0},
                    "kate_flux_uM_per_s":        {"_type": "overwrite[float]", "_default": 0.0},
                    "total_scavenging_uM_per_s": {"_type": "overwrite[float]", "_default": 0.0},
                    "nadph_consumed":            {"_type": "overwrite[integer]", "_default": 0},
                    "ppgpp_produced":            {"_type": "overwrite[integer]", "_default": 0},
                    "dna_damage_rate":           {"_type": "overwrite[float]", "_default": 0.0},
                    "cumulative_dna_damage":     {"_type": "overwrite[float]", "_default": 0.0},
                    "oxyr_fold_change":          {"_type": "overwrite[float]", "_default": 1.0},
                    "soxrs_fold_change":         {"_type": "overwrite[float]", "_default": 1.0},
                },
            },
        }

    # ------------------------------------------------------------------
    def _resolve_indices(self, bulk):
        """Lazily resolve all bulk molecule indices."""
        if hasattr(bulk, "dtype") and bulk.dtype.names and "id" in bulk.dtype.names:
            self._h2o2_idx  = bulk_name_to_idx(_H2O2_ID,  bulk["id"])
            self._nadph_idx = bulk_name_to_idx(_NADPH_ID, bulk["id"])
            self._ahpc_idx  = bulk_name_to_idx(_AHPC_ID,  bulk["id"])
            self._katg_idx  = bulk_name_to_idx(_KATG_ID,  bulk["id"])
            self._kate_idx  = bulk_name_to_idx(_KATE_ID,  bulk["id"])
            self._h2o_idx   = bulk_name_to_idx(_H2O_ID,   bulk["id"])
            self._ppgpp_idx = bulk_name_to_idx(_PPGPP_ID, bulk["id"])
            self._gtp_idx   = bulk_name_to_idx(_GTP_ID,   bulk["id"])
        else:
            # Plain int64 array (WCM mode) — use hardcoded indices
            self._h2o2_idx  = 10149
            self._nadph_idx = 11201
            self._ahpc_idx  = 12341
            self._katg_idx  = 10152
            self._kate_idx  = 10154
            self._h2o_idx   = 15614
            self._ppgpp_idx = 10012
            self._gtp_idx   = 10000
        self._indices_resolved = True

    def _cell_volume(self, states):
        """Cell volume in litres from mass listener."""
        try:
            cell_mass_fg = float(states["listeners"]["mass"]["cell_mass"])
            if cell_mass_fg > 0:
                # cell_mass in fg → grams, then divide by density (g/L) → litres
                vol_L = (cell_mass_fg * 1e-15) / self.cell_density_val
                return vol_L
        except Exception:
            pass
        return _V_CELL

    def _mm(self, s_uM, E, kcat, Km):
        """Michaelis-Menten rate in molecules/s."""
        if E <= 0 or s_uM <= 0:
            return 0.0
        return kcat * E * s_uM / (Km + s_uM)

    def _total_scavenging(self, h2o2_uM, ahpc, katg, kate):
        """Total scavenging rate in molecules/s at given [H2O2]."""
        return (self._mm(h2o2_uM, ahpc, self.ahpcf_kcat, self.ahpcf_Km)
                + self._mm(h2o2_uM, katg, self.katg_kcat,  self.katg_Km)
                + self._mm(h2o2_uM, kate, self.kate_kcat,  self.kate_Km))

    def _h2o2_qss_uM(self, prod_rate_molec_s, ahpc, katg, kate, vol):
        """H2O2 quasi-steady-state concentration (µM).

        H2O2 kinetics (ms timescale) are much faster than the WCM timestep
        (2 s), so we solve for the concentration where production = scavenging
        directly instead of Euler-stepping, which would oscillate.
        """
        if prod_rate_molec_s <= 0.0:
            return 0.0

        def net(h_uM):
            return prod_rate_molec_s - self._total_scavenging(h_uM, ahpc, katg, kate)

        vmax_ahpcf = self.ahpcf_kcat * ahpc if ahpc > 0 else 0.0
        vmax_katg  = self.katg_kcat  * katg  if katg  > 0 else 0.0
        vmax_kate  = self.kate_kcat  * kate  if kate  > 0 else 0.0
        total_vmax = vmax_ahpcf + vmax_katg + vmax_kate

        if total_vmax <= prod_rate_molec_s:
            net_prod_molec_s = prod_rate_molec_s - total_vmax
            return _count_to_uM(net_prod_molec_s, vol)

        try:
            return float(brentq(net, 0.0, 1e6, xtol=1e-5, rtol=1e-5))
        except Exception:
            slope = self._total_scavenging(1e-4, ahpc, katg, kate) / 1e-4
            if slope > 0:
                return prod_rate_molec_s / slope
            return _count_to_uM(prod_rate_molec_s, vol)

    def _hill(self, x, K, n):
        if x <= 0:
            return 0.0
        xn = x ** n
        return xn / (K ** n + xn)

    # ------------------------------------------------------------------
    def update(self, states, interval=None):
        timestep = interval if interval is not None else states.get("timestep", 1.0)
        # process-bigraph may pass negative intervals; use absolute value
        timestep = abs(timestep) if timestep != 0 else 1.0

        if not self._indices_resolved:
            self._resolve_indices(states["bulk_total"])

        bt      = states["bulk_total"]
        h2o2    = float(counts(bt, self._h2o2_idx))
        ahpc    = float(counts(bt, self._ahpc_idx))
        katg    = float(counts(bt, self._katg_idx))
        kate    = float(counts(bt, self._kate_idx))
        nadph   = float(counts(bt, self._nadph_idx))
        gtp     = float(counts(bt, self._gtp_idx))

        vol     = self._cell_volume(states)

        # Total production rate (molecules/s)
        # Include timed stress H₂O₂ signal if present
        timed_h2o2 = 0.0
        try:
            timed_h2o2 = float(
                states["listeners"]["timed_stress"]["h2o2_signal_uM_s"])
        except (KeyError, TypeError):
            pass
        effective_ext = self.ext_h2o2_uM + timed_h2o2
        prod_rate = (_uM_to_count(self.h2o2_prod_rate_uM_s, vol)
                     + _uM_to_count(effective_ext, vol))

        # Quasi-steady-state [H2O2]
        new_uM  = self._h2o2_qss_uM(prod_rate, ahpc, katg, kate, vol)
        new_h2o2 = max(0.0, _uM_to_count(new_uM, vol))

        # Scavenging flux at the QSS concentration
        ahpcf_rate = self._mm(new_uM, ahpc, self.ahpcf_kcat, self.ahpcf_Km)
        katg_rate  = self._mm(new_uM, katg, self.katg_kcat,  self.katg_Km)
        kate_rate  = self._mm(new_uM, kate, self.kate_kcat,  self.kate_Km)
        total_rate = ahpcf_rate + katg_rate + kate_rate

        # Net H2O2 bulk change
        delta_h2o2 = int(round(new_h2o2)) - int(h2o2)

        # NADPH consumed by AhpCF
        nadph_consumed = min(int(round(ahpcf_rate * timestep)), int(nadph))

        # OxyR oxidation/reduction kinetics
        oxyr_ss = self._hill(new_uM, self.oxyr_Kox, self.oxyr_hill)
        oxyr_kox_eff = oxyr_ss * self.oxyr_kred / max(1.0 - oxyr_ss, 1e-6)
        d_oxyr = (oxyr_kox_eff * (1.0 - self.oxyr_ox)
                  - self.oxyr_kred * self.oxyr_ox) * timestep
        self.oxyr_ox = float(np.clip(self.oxyr_ox + d_oxyr, 0.0, 1.0))

        # SoxRS activation
        soxr_ss = self._hill(new_uM, self.soxr_K, self.soxr_hill)
        self.soxr_ox += (soxr_ss - self.soxr_ox) * min(1.0, timestep * 0.3)
        self.soxr_ox = float(np.clip(self.soxr_ox, 0.0, 1.0))

        # Fold changes
        oxyr_fc  = 1.0 + (self.oxyr_fc_max  - 1.0) * self.oxyr_ox
        soxrs_fc = 1.0 + (self.soxrs_fc_max - 1.0) * self.soxr_ox

        # DNA damage via Fenton chemistry
        dmg_rate = self.k_fenton * new_uM * self.fe2_free
        self.cum_dmg += dmg_rate * timestep

        # ppGpp–growth coupling
        gtp_count = float(counts(bt, self._gtp_idx))
        ppgpp_synth_rate = self.k_ppgpp * new_uM / (self.ppgpp_Km + new_uM)
        ppgpp_produced = int(round(ppgpp_synth_rate * timestep))
        ppgpp_produced = min(ppgpp_produced, int(gtp_count))

        # Bulk updates
        bulk_updates = [(self._h2o2_idx, delta_h2o2)]
        if ppgpp_produced > 0:
            bulk_updates.append((self._ppgpp_idx, ppgpp_produced))
            bulk_updates.append((self._gtp_idx, -ppgpp_produced))
        if nadph_consumed > 0:
            bulk_updates.append((self._nadph_idx, -nadph_consumed))
        cat_consumed = int(round((katg_rate + kate_rate) * timestep))
        if cat_consumed > 0:
            bulk_updates.append((self._h2o_idx, cat_consumed))

        return {
            "bulk": bulk_updates,
            "listeners": {
                "oxidative_stress": {
                    "h2o2_uM":                   float(new_uM),
                    "oxyr_ox_fraction":          float(self.oxyr_ox),
                    "soxr_ox_fraction":          float(self.soxr_ox),
                    "ahpcf_flux_uM_per_s":       float(_count_to_uM(ahpcf_rate, vol)),
                    "katg_flux_uM_per_s":        float(_count_to_uM(katg_rate,  vol)),
                    "kate_flux_uM_per_s":        float(_count_to_uM(kate_rate,  vol)),
                    "total_scavenging_uM_per_s": float(_count_to_uM(total_rate, vol)),
                    "nadph_consumed":            int(nadph_consumed),
                    "ppgpp_produced":            int(ppgpp_produced),
                    "dna_damage_rate":           float(dmg_rate),
                    "cumulative_dna_damage":     float(self.cum_dmg),
                    "oxyr_fold_change":          float(oxyr_fc),
                    "soxrs_fold_change":         float(soxrs_fc),
                },
            },
        }
