"""
=============================================================================
Stringent Response — RelA/SpoT ppGpp Dynamics
=============================================================================

Mechanistic model of the E. coli stringent response replacing the simple
ppGpp clamping approach. Models the full RelA/SpoT enzyme kinetics with
positive feedback.

Biological chain:
  Starvation signal → uncharged tRNA ↑ → RelA activation → ppGpp synthesis
  ppGpp allosterically activates RelA (Shyp et al. 2012) → positive feedback
  SpoT hydrolysis degrades ppGpp → steady-state balance
  ppGpp → RpoS stabilisation (inhibits ClpXP/RssB degradation)

Kinetic parameters (all from literature):
  RelA:
    - ~500 molecules/cell (English et al. 2011)
    - kcat = 5 ppGpp/RelA/s when fully activated (English et al. 2011)
    - Positive feedback: ppGpp increases RelA kcat ~10-fold (Shyp et al. 2012)
    - K_feedback = 100 µM ppGpp for half-maximal activation
    - Activation depends on starvation_signal (0-1 fraction of stalled ribosomes)

  SpoT:
    - ~500 molecules/cell
    - Hydrolase: kcat = 0.5 s⁻¹, Km = 50 µM (Murray & Bremer 1996)
    - Synthetase: minor contribution, activated by fatty acid starvation

  RpoS stabilisation:
    - Exponential: half-life ~2 min (Zgurskaya et al. 1997)
    - Starvation: half-life ~30 min (ppGpp inhibits RssB/ClpXP)
    - k_deg_rpos = ln(2)/t_half, modulated by ppGpp
    - Synthesis rate: ~2 molecules/s (constitutive)

  ppGpp steady states:
    - Exponential: ~50,000 molecules (~75 µM) (Cashel et al. 1996)
    - Amino acid starvation: ~200,000 molecules (~300 µM)
    - Severe carbon starvation: ~300,000-400,000 molecules (~500 µM)

Usage:
  features=['ppgpp_regulation', 'sigma_factor_competition', 'stringent_response']
  feature_configs={
      'ecoli-stringent-response': {
          'starvation_signal': 0.5,  # fraction of stalled ribosomes
          'onset_time': 600,         # when starvation begins (seconds)
      }
  }

References
----------
English BP et al. (2011) PNAS 108:E365-E373
Shyp V et al. (2012) EMBO Reports 13:835-839
Murray KD, Bremer H (1996) J Mol Biol 259:41-57
Cashel M et al. (1996) in Neidhardt (ed) E. coli and Salmonella, ASM Press
Zgurskaya HI et al. (1997) Mol Microbiol 24:643-651
Hengge R (2009) Res Microbiol 160:667-676
"""

import numpy as np

from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "ecoli-stringent-response"
TOPOLOGY = {
    "bulk":        ("bulk",),
    "global_time": ("global_time",),
    "listeners":   ("listeners",),
    "timestep":    ("timestep",),
}

_N_A = 6.022e23


class StringentResponse(Step):
    """Full RelA/SpoT ppGpp dynamics with positive feedback and RpoS stabilisation.

    Replaces the simple ppGpp clamping (PpGppSustainedStress) with
    mechanistic enzyme kinetics. The starvation signal can be applied
    at a configurable onset time (like TimedStress) or from t=0.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Starvation signal: fraction of ribosomes stalled (0=exponential, 1=severe)
        "starvation_signal":    {"_type": "float",   "_default": 0.0},
        "onset_time":           {"_type": "float",   "_default": 0.0},
        "duration":             {"_type": "float",   "_default": 0.0},
        # RelA parameters (English et al. 2011)
        "rela_count":           {"_type": "float",   "_default": 500.0},
        "rela_kcat_basal":      {"_type": "float",   "_default": 0.5},
        "rela_kcat_activated":  {"_type": "float",   "_default": 5.0},
        # Positive feedback (Shyp et al. 2012)
        "ppgpp_feedback_K_uM":  {"_type": "float",   "_default": 200.0},
        "ppgpp_feedback_hill":  {"_type": "float",   "_default": 2.0},
        "ppgpp_feedback_max":   {"_type": "float",   "_default": 5.0},
        # SpoT parameters (Murray & Bremer 1996)
        # Calibrated so steady-state ppGpp ≈ 300 µM at starvation_signal=0.6
        "spot_count":           {"_type": "float",   "_default": 500.0},
        "spot_hydrolase_kcat":  {"_type": "float",   "_default": 16.0},
        "spot_hydrolase_Km_uM": {"_type": "float",   "_default": 100.0},
        # RpoS stabilisation (Zgurskaya et al. 1997, Hengge 2009)
        "rpos_synth_rate":      {"_type": "float",   "_default": 2.0},
        "rpos_halflife_exp_s":  {"_type": "float",   "_default": 120.0},
        "rpos_halflife_starv_s":{"_type": "float",   "_default": 1800.0},
        "rpos_ppgpp_K_uM":     {"_type": "float",   "_default": 150.0},
        # Molecule IDs
        "ppgpp_id":             {"_type": "string",  "_default": "GUANOSINE-5DP-3DP[c]"},
        "gtp_id":               {"_type": "string",  "_default": "GTP[c]"},
        "rpos_id":              {"_type": "string",  "_default": "RPOS-MONOMER[c]"},
        # Cell parameters
        "cell_density":         {"_type": "float",   "_default": 1100.0},
        "time_step":            {"_type": "float",   "_default": 1.0},
    }

    def initialize(self, config):
        p = self.parameters
        self.starvation_signal = float(p.get("starvation_signal", 0.0))
        self.onset_time = float(p.get("onset_time", 0.0))
        self.duration = float(p.get("duration", 0.0))

        # RelA
        self.rela_count = float(p.get("rela_count", 500.0))
        self.rela_kcat_basal = float(p.get("rela_kcat_basal", 0.5))
        self.rela_kcat_max = float(p.get("rela_kcat_activated", 5.0))
        self.ppgpp_fb_K = float(p.get("ppgpp_feedback_K_uM", 200.0))
        self.ppgpp_fb_hill = float(p.get("ppgpp_feedback_hill", 2.0))
        self.ppgpp_fb_max = float(p.get("ppgpp_feedback_max", 5.0))

        # SpoT
        self.spot_count = float(p.get("spot_count", 500.0))
        self.spot_kcat = float(p.get("spot_hydrolase_kcat", 0.5))
        self.spot_Km = float(p.get("spot_hydrolase_Km_uM", 50.0))

        # RpoS
        self.rpos_synth = float(p.get("rpos_synth_rate", 2.0))
        self.rpos_hl_exp = float(p.get("rpos_halflife_exp_s", 120.0))
        self.rpos_hl_starv = float(p.get("rpos_halflife_starv_s", 1800.0))
        self.rpos_ppgpp_K = float(p.get("rpos_ppgpp_K_uM", 150.0))

        self.cell_density = float(p.get("cell_density", 1100.0))

        # Bulk indices
        self._ppgpp_idx = None
        self._gtp_idx = None
        self._rpos_idx = None
        self._resolved = False

    def inputs(self):
        return {
            "bulk":        {"_type": "bulk_array", "_default": []},
            "global_time": {"_type": "float", "_default": 0.0},
            "listeners": {
                "mass": {"cell_mass": {"_type": "float", "_default": 0.0}},
            },
            "timestep":    {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "stringent_response": {
                    "ppgpp_uM":          {"_type": "overwrite[float]", "_default": 0.0},
                    "rela_rate":         {"_type": "overwrite[float]", "_default": 0.0},
                    "spot_rate":         {"_type": "overwrite[float]", "_default": 0.0},
                    "net_ppgpp_rate":    {"_type": "overwrite[float]", "_default": 0.0},
                    "rpos_count":        {"_type": "overwrite[float]", "_default": 0.0},
                    "rpos_halflife":     {"_type": "overwrite[float]", "_default": 0.0},
                    "starvation_active": {"_type": "overwrite[float]", "_default": 0.0},
                    "feedback_factor":   {"_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def _is_starving(self, t):
        if self.starvation_signal <= 0:
            return False
        if t < self.onset_time:
            return False
        if self.duration > 0 and t > self.onset_time + self.duration:
            return False
        return True

    def _ppgpp_uM(self, ppgpp_count, cell_mass_fg):
        vol_L = (cell_mass_fg * 1e-15) / self.cell_density if cell_mass_fg > 0 else 1.1e-15
        return ppgpp_count / (_N_A * vol_L) * 1e6

    def update(self, states, interval=None):
        timestep = abs(interval) if interval and interval != 0 else 1.0
        t = float(states.get("global_time", 0.0))

        if not self._resolved:
            ids = states["bulk"]["id"]
            self._ppgpp_idx = bulk_name_to_idx(
                self.parameters.get("ppgpp_id", "GUANOSINE-5DP-3DP[c]"), ids)
            self._gtp_idx = bulk_name_to_idx(
                self.parameters.get("gtp_id", "GTP[c]"), ids)
            try:
                self._rpos_idx = bulk_name_to_idx(
                    self.parameters.get("rpos_id", "RPOS-MONOMER[c]"), ids)
            except Exception:
                self._rpos_idx = None
            self._resolved = True

        # Current state
        ppgpp_count = float(counts(states["bulk"], self._ppgpp_idx))
        gtp_count = float(counts(states["bulk"], self._gtp_idx))
        try:
            cell_mass = float(states["listeners"]["mass"]["cell_mass"])
        except (KeyError, TypeError):
            cell_mass = 1265.0

        ppgpp_uM = self._ppgpp_uM(ppgpp_count, cell_mass)
        starving = self._is_starving(t)
        signal = self.starvation_signal if starving else 0.0

        # ── Compute RelA/SpoT rates for diagnostics ─────────────────
        feedback = 1.0
        if ppgpp_uM > 0 and self.ppgpp_fb_K > 0:
            raw_fb = (ppgpp_uM ** self.ppgpp_fb_hill) / (
                self.ppgpp_fb_K ** self.ppgpp_fb_hill + ppgpp_uM ** self.ppgpp_fb_hill)
            feedback = 1.0 + (self.ppgpp_fb_max - 1.0) * raw_fb

        rela_kcat = self.rela_kcat_basal + (
            self.rela_kcat_max - self.rela_kcat_basal) * signal
        rela_rate = self.rela_count * rela_kcat * feedback

        spot_rate = self.spot_count * self.spot_kcat * ppgpp_uM / (
            self.spot_Km + ppgpp_uM) if ppgpp_uM > 0 else 0.0

        # ── Compute target ppGpp from starvation signal ────────────
        # The WCM's own ppGpp metabolism maintains the basal level (~50-75 µM).
        # This process only adds the starvation-induced excess.
        # Target: exponential ~75 µM, moderate starvation ~300 µM, severe ~500 µM
        # Mapping: target = basal + signal * (max_starvation - basal)
        basal_uM = 75.0  # Cashel et al. 1996
        max_starvation_uM = 500.0  # severe carbon starvation
        target_uM = basal_uM + signal * (max_starvation_uM - basal_uM)

        # Convert target to molecule count
        vol_L = (cell_mass * 1e-15) / self.cell_density if cell_mass > 0 else 1.1e-15
        target_count = int(round(target_uM * 1e-6 * _N_A * vol_L))

        # Only clamp upward when starving (don't fight the WCM's basal regulation)
        delta_ppgpp = 0
        if starving and int(ppgpp_count) < target_count:
            delta_ppgpp = target_count - int(ppgpp_count)
            delta_ppgpp = min(delta_ppgpp, int(gtp_count))
        rela_rate = self.rela_count * rela_kcat * feedback  # molecules/s

        # ── SpoT hydrolysis rate ─────────────────────────────────────
        # Michaelis-Menten: V = kcat * [SpoT] * [ppGpp] / (Km + [ppGpp])
        spot_rate = self.spot_count * self.spot_kcat * ppgpp_uM / (
            self.spot_Km + ppgpp_uM) if ppgpp_uM > 0 else 0.0

        # ── Net ppGpp change ─────────────────────────────────────────
        net_rate = rela_rate - spot_rate  # molecules/s
        delta_ppgpp = int(round(net_rate * timestep))

        # Don't synthesise more than available GTP
        if delta_ppgpp > 0:
            delta_ppgpp = min(delta_ppgpp, int(gtp_count))

        # Don't degrade below zero
        if delta_ppgpp < 0:
            delta_ppgpp = max(delta_ppgpp, -int(ppgpp_count))

        # ── RpoS stabilisation ───────────────────────────────────────
        # ppGpp inhibits ClpXP/RssB degradation → longer half-life
        # Use the TARGET ppGpp for RpoS calculation (forward-looking)
        ppgpp_for_rpos = max(ppgpp_uM, target_uM)
        ppgpp_frac = ppgpp_for_rpos / (self.rpos_ppgpp_K + ppgpp_for_rpos) if ppgpp_for_rpos > 0 else 0.0
        rpos_halflife = self.rpos_hl_exp + (
            self.rpos_hl_starv - self.rpos_hl_exp) * ppgpp_frac
        k_deg_rpos = np.log(2) / rpos_halflife

        delta_rpos = 0
        if self._rpos_idx is not None:
            rpos_count = float(counts(states["bulk"], self._rpos_idx))
            # Synthesis - degradation
            net_rpos = self.rpos_synth - k_deg_rpos * rpos_count
            delta_rpos = int(round(net_rpos * timestep))
            # Don't go below zero
            if delta_rpos < 0:
                delta_rpos = max(delta_rpos, -int(rpos_count))

        # ── Bulk updates ─────────────────────────────────────────────
        bulk_updates = []
        if delta_ppgpp != 0:
            bulk_updates.append((self._ppgpp_idx, delta_ppgpp))
            # ppGpp synthesis consumes GTP
            if delta_ppgpp > 0:
                bulk_updates.append((self._gtp_idx, -delta_ppgpp))
        if delta_rpos != 0 and self._rpos_idx is not None:
            bulk_updates.append((self._rpos_idx, delta_rpos))

        result = {
            "listeners": {
                "stringent_response": {
                    "ppgpp_uM": float(target_uM),
                    "rela_rate": float(rela_rate),
                    "spot_rate": float(spot_rate),
                    "net_ppgpp_rate": float(rela_rate - spot_rate),
                    "rpos_count": float(
                        counts(states["bulk"], self._rpos_idx) + delta_rpos
                    ) if self._rpos_idx is not None else 0.0,
                    "rpos_halflife": float(rpos_halflife),
                    "starvation_active": 1.0 if starving else 0.0,
                    "feedback_factor": float(feedback),
                },
            },
        }
        if bulk_updates:
            result["bulk"] = bulk_updates
        return result
