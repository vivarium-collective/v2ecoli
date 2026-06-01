"""
OxyR Feedback Loop — Adaptive Scavenging Enhancement
=====================================================

Models the OxyR-mediated adaptive response where H₂O₂-induced
transcriptional upregulation of katG and ahpCF leads to increased
scavenging enzyme levels, which in turn reduce H₂O₂ and deactivate
OxyR (negative feedback homeostasis).

The feedback loop:
  H₂O₂ ↑ → OxyR oxidised → katG/ahpCF transcription ↑
  → KatG/AhpCF protein ↑ → scavenging ↑ → H₂O₂ ↓ → OxyR reduced

This process reads the OxyR fold change from the oxidative stress
listener and directly adds scavenging enzyme molecules to the bulk
store, simulating the accelerated protein synthesis of OxyR regulon
genes. The rate is proportional to the fold change minus 1 (no extra
synthesis at FC=1).

The existing OxidativeStress process reads bulk KatG/AhpCF counts
for its scavenging calculation, so the added enzymes automatically
increase scavenging capacity — closing the feedback loop.

Kinetics (Zheng et al. 2001, Christman et al. 1985):
  - katG mRNA half-life: ~5 min → protein appears ~2-5 min after induction
  - KatG protein half-life: ~60 min (stable)
  - Peak induction: ~10× at 30 min post-challenge
  - We model this as first-order synthesis with delay:
    d[KatG]/dt = k_synth * (FC - 1) * baseline_count - k_deg * [KatG_extra]

References
----------
Zheng M et al. (2001) J Bacteriol 183:4562-4570
Christman MF et al. (1985) Cell 41:753-762
Seaver LC, Imlay JA (2001) J Bacteriol 183:7182-7189
"""

import numpy as np

from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "ecoli-oxyr-feedback"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
    "timestep":  ("timestep",),
}

# Molecule IDs
_KATG_ID = "HYDROPEROXIDI-CPLX[c]"
_KATE_ID = "HYDROPEROXIDII-CPLX[c]"
_AHPC_ID = "THIOREDOXIN-REDUCT-NADPH-CPLX[c]"


class OxyRFeedback(Step):
    """Closes the OxyR feedback loop by synthesising extra scavenging enzymes.

    Reads OxyR fold change from the oxidative stress listener.
    Adds KatG, KatE, and AhpCF molecules proportional to (FC - 1).
    The OxidativeStress process then reads the higher enzyme counts
    and computes increased scavenging, reducing H₂O₂.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Synthesis rate: molecules per second per unit fold-change above 1
        # Calibrated so that 10× FC produces ~200 extra KatG in 10 min
        # (200 / 600s / 9 = 0.037 → round to 0.035)
        "katg_synth_rate":  {"_type": "float", "_default": 0.035},
        "kate_synth_rate":  {"_type": "float", "_default": 0.015},
        "ahpcf_synth_rate": {"_type": "float", "_default": 0.025},
        # Degradation rate (1/s) — protein half-life ~60 min
        # k_deg = ln(2) / (60*60) ≈ 0.000193
        "k_deg":            {"_type": "float", "_default": 0.000193},
        # Baseline enzyme counts (used to scale synthesis)
        "katg_baseline":    {"_type": "integer", "_default": 200},
        "kate_baseline":    {"_type": "integer", "_default": 50},
        "ahpcf_baseline":   {"_type": "integer", "_default": 500},
        # Transcriptional + translational delay (Christman et al. 1985)
        # mRNA appears ~1 min after OxyR activation, protein ~2-3 min later
        "delay_seconds":    {"_type": "float", "_default": 180.0},
        "time_step":        {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        p = self.parameters
        self.katg_synth  = float(p.get("katg_synth_rate", 0.035))
        self.kate_synth  = float(p.get("kate_synth_rate", 0.015))
        self.ahpcf_synth = float(p.get("ahpcf_synth_rate", 0.025))
        self.k_deg       = float(p.get("k_deg", 0.000193))
        self.katg_base   = int(p.get("katg_baseline", 200))
        self.kate_base   = int(p.get("kate_baseline", 50))
        self.ahpcf_base  = int(p.get("ahpcf_baseline", 500))
        self.delay       = float(p.get("delay_seconds", 180.0))

        # Track extra enzyme molecules added (for degradation)
        self.extra_katg  = 0.0
        self.extra_kate  = 0.0
        self.extra_ahpcf = 0.0

        # Delay buffer: store (time, fc_excess) for delayed synthesis
        self._fc_history = []
        self._cumulative_time = 0.0

        self._katg_idx  = None
        self._kate_idx  = None
        self._ahpc_idx  = None
        self._resolved  = False

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "listeners": {
                "oxidative_stress": {
                    "oxyr_fold_change":  {"_type": "float", "_default": 1.0},
                    "soxrs_fold_change": {"_type": "float", "_default": 1.0},
                },
            },
            "timestep": {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "oxyr_feedback": {
                    "extra_katg":  {"_type": "overwrite[float]", "_default": 0.0},
                    "extra_kate":  {"_type": "overwrite[float]", "_default": 0.0},
                    "extra_ahpcf": {"_type": "overwrite[float]", "_default": 0.0},
                    "oxyr_fc_in":  {"_type": "overwrite[float]", "_default": 1.0},
                },
            },
        }

    def update(self, states, interval=None):
        timestep = abs(interval) if interval and interval != 0 else 1.0
        self._cumulative_time += timestep

        # Resolve bulk indices once
        if not self._resolved:
            ids = states["bulk"]["id"]
            self._katg_idx = bulk_name_to_idx(_KATG_ID, ids)
            self._kate_idx = bulk_name_to_idx(_KATE_ID, ids)
            self._ahpc_idx = bulk_name_to_idx(_AHPC_ID, ids)
            self._resolved = True

        # Read OxyR fold change
        try:
            oxyr_fc = float(
                states["listeners"]["oxidative_stress"]["oxyr_fold_change"])
        except (KeyError, TypeError):
            oxyr_fc = 1.0

        # Read SoxRS fold change (SoxRS also upregulates some enzymes)
        try:
            soxrs_fc = float(
                states["listeners"]["oxidative_stress"]["soxrs_fold_change"])
        except (KeyError, TypeError):
            soxrs_fc = 1.0

        # Current fold change excess
        fc_excess = max(0.0, oxyr_fc - 1.0)
        soxrs_excess = max(0.0, soxrs_fc - 1.0)

        # Store current FC in history buffer
        self._fc_history.append((self._cumulative_time, fc_excess, soxrs_excess))

        # Get the DELAYED fold change (from delay_seconds ago)
        # This models the transcription + translation delay
        target_time = self._cumulative_time - self.delay
        delayed_fc = 0.0
        delayed_soxrs = 0.0
        if target_time > 0:
            # Find the FC value from delay_seconds ago
            for t_hist, fc_hist, sox_hist in reversed(self._fc_history):
                if t_hist <= target_time:
                    delayed_fc = fc_hist
                    delayed_soxrs = sox_hist
                    break

        # Trim old history (keep last 2× delay worth)
        cutoff = self._cumulative_time - 2 * self.delay
        self._fc_history = [(t, f, s) for t, f, s in self._fc_history if t > cutoff]

        # Synthesis uses DELAYED fold change (not current)
        synth_katg  = self.katg_synth  * delayed_fc * self.katg_base * timestep
        synth_kate  = self.kate_synth  * (delayed_fc + delayed_soxrs * 0.3) * self.kate_base * timestep
        synth_ahpcf = self.ahpcf_synth * delayed_fc * self.ahpcf_base * timestep

        # Degradation of extra enzymes
        deg_katg  = self.k_deg * self.extra_katg * timestep
        deg_kate  = self.k_deg * self.extra_kate * timestep
        deg_ahpcf = self.k_deg * self.extra_ahpcf * timestep

        # Update tracked extras
        self.extra_katg  = max(0.0, self.extra_katg  + synth_katg  - deg_katg)
        self.extra_kate  = max(0.0, self.extra_kate  + synth_kate  - deg_kate)
        self.extra_ahpcf = max(0.0, self.extra_ahpcf + synth_ahpcf - deg_ahpcf)

        # Net bulk changes (synthesis - degradation, rounded to integers)
        delta_katg  = int(round(synth_katg  - deg_katg))
        delta_kate  = int(round(synth_kate  - deg_kate))
        delta_ahpcf = int(round(synth_ahpcf - deg_ahpcf))

        bulk_updates = []
        if delta_katg != 0:
            bulk_updates.append((self._katg_idx, delta_katg))
        if delta_kate != 0:
            bulk_updates.append((self._kate_idx, delta_kate))
        if delta_ahpcf != 0:
            bulk_updates.append((self._ahpc_idx, delta_ahpcf))

        result = {
            "listeners": {
                "oxyr_feedback": {
                    "extra_katg":  float(self.extra_katg),
                    "extra_kate":  float(self.extra_kate),
                    "extra_ahpcf": float(self.extra_ahpcf),
                    "oxyr_fc_in":  float(oxyr_fc),
                },
            },
        }
        if bulk_updates:
            result["bulk"] = bulk_updates
        return result
