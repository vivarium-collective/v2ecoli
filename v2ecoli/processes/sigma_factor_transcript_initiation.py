"""
=============================================================================
Sigma-Factor-Based Transcript Initiation
=============================================================================

A single process that replaces the original TranscriptInitiation with a
mechanistic sigma factor competition model covering both exponential and
stationary phase transcription.

Biological design
-----------------
Five sigma factors compete for the same pool of core RNAP (E):

    σ70  (RpoD)  — housekeeping, exponential phase
    σ38  (RpoS)  — stationary phase / stress
    σ32  (RpoH)  — heat shock
    σ24  (RpoE)  — envelope / oxidative stress
    σ54  (RpoN)  — nitrogen limitation

Each promoter carries a weight vector [w_σ70, w_σ38, w_σ32, w_σ24, w_σ54]
encoding its intrinsic sigma preference.  Weights are derived from:
  - ppGpp synth_prob response (exponential vs stationary phase)
  - Literature regulon sizes (RegulonDB / EcoCyc)

Competition is solved via the Mauri & Klumpp (2014) iterative fixed-point
scheme (Eq. 3-10), extended to N sigma factors.

ppGpp modulation
----------------
ppGpp weakens RpoD-core affinity (Jishage et al. 2002):
    K_E70_eff = K_E70 * exp(alpha * [ppGpp]_µM)

This automatically shifts the holoenzyme distribution from σ70 to σ38
as ppGpp accumulates during nutrient stress / stationary phase entry.

Phase behaviour
---------------
Exponential phase (ppGpp ~ 0):
    K_E70_eff ≈ K_E70 (1 nM) → σ70 dominates → rRNA/ribosomal genes high
Stationary phase (ppGpp ~ 200-500 µM):
    K_E70_eff ≈ 5-10 nM → σ38 competes effectively → stress genes high

References
----------
Mauri M, Klumpp S (2014) PLoS Comput Biol 10(10):e1003845
Jishage M et al. (2002) Genes Dev 16:1260-1270
Bougdour A et al. (2004) J Biol Chem 279:19540-19550
Gruber TM, Gross CA (2003) Annu Rev Microbiol 57:441-466
Hengge R (2009) Nat Rev Microbiol 7:263-273

Registration
------------
NAME = "ecoli-sigma-factor-transcript-initiation"
Swap in via config:
    "swap_processes": {
        "ecoli-transcript-initiation":
        "ecoli-sigma-factor-transcript-initiation"
    }
"""

import numpy as np
import scipy.sparse
from typing import cast

from v2ecoli.library.schema import (
    bulk_name_to_idx, counts, attrs,
    numpy_schema, listener_schema, create_unique_indices,
)
from v2ecoli.types.quantity import ureg as units
from v2ecoli.processes.transcript_initiation import TranscriptInitiation

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NAME = "ecoli-sigma-factor-transcript-initiation"
TOPOLOGY = {
    "environment":      ("environment",),
    "full_chromosomes": ("unique", "full_chromosome"),
    "RNAs":             ("unique", "RNA"),
    "active_RNAPs":     ("unique", "active_RNAP"),
    "promoters":        ("unique", "promoter"),
    "bulk":             ("bulk",),
    "listeners":        ("listeners",),
    "timestep":         ("timestep",),
    "ppgpp_state":      ("ppgpp_state",),
}

# ---------------------------------------------------------------------------
# Sigma factor metadata
# ---------------------------------------------------------------------------
# (bulk_id, Kd_nM, ppGpp_sensitivity, description)
SIGMA_FACTORS = [
    ("RPOD-MONOMER[c]", 1.0,   0.005, "σ70 RpoD housekeeping"),
    ("RPOS-MONOMER[c]", 20.0,  0.0,   "σ38 RpoS stationary/stress"),
    ("RPOH-MONOMER[c]", 50.0,  0.0,   "σ32 RpoH heat shock"),
    ("RPOE-MONOMER[c]", 80.0,  0.0,   "σ24 RpoE envelope stress"),
    ("RPON-MONOMER[c]", 100.0, 0.0,   "σ54 RpoN nitrogen"),
]
N_SIGMA = len(SIGMA_FACTORS)
SIGMA_NAMES = [s[3] for s in SIGMA_FACTORS]

# ---------------------------------------------------------------------------
# Mauri & Klumpp (2014) N-sigma equilibrium solver
# ---------------------------------------------------------------------------

def _solve_holoenzyme_single(E: float, s: float, K: float) -> float:
    """[Es] for one sigma factor (Eq. 3)."""
    disc = max((K + E + s) ** 2 - 4.0 * E * s, 0.0)
    return 0.5 * (K + E + s - np.sqrt(disc))


def solve_n_sigma_competition(
    E_total: float,
    sigma_counts: np.ndarray,   # shape (N,)
    K_eff: np.ndarray,          # shape (N,) — effective Kd in molecule units
    max_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve holoenzyme counts [E·σᵢ] for N competing sigma factors.

    Extends Mauri & Klumpp (2014) Eq. 4-8 to N sigma factors via
    alternating fixed-point iteration.
    """
    N = len(sigma_counts)
    if E_total <= 0.0 or sigma_counts.sum() <= 0.0:
        return np.zeros(N)

    tot_sigma = sigma_counts.sum()
    Es = np.array([
        _solve_holoenzyme_single(E_total, sigma_counts[i], K_eff[i])
        * sigma_counts[i] / tot_sigma
        for i in range(N)
    ])
    Es = np.maximum(Es, 0.0)

    for _ in range(max_iter):
        Es_prev = Es.copy()
        for i in range(N):
            E_avail = max(0.0, E_total - Es.sum() + Es[i])
            Es_new  = _solve_holoenzyme_single(E_avail, sigma_counts[i], K_eff[i])
            Es[i]   = min(max(0.0, Es_new), E_avail, sigma_counts[i])
        if np.abs(Es - Es_prev).sum() < tol:
            break

    return np.maximum(Es, 0.0)


def holoenzyme_fractions(
    E_total: float,
    sigma_counts: np.ndarray,
    K_eff: np.ndarray,
) -> np.ndarray:
    """Return normalised holoenzyme fractions [f_σ70, f_σ38, ...], shape (N,)."""
    Es = solve_n_sigma_competition(E_total, sigma_counts, K_eff)
    total = Es.sum()
    if total <= 0.0:
        f = np.zeros(N_SIGMA)
        f[0] = 1.0   # fallback: all σ70
        return f
    return Es / total


# ---------------------------------------------------------------------------
# Process class
# ---------------------------------------------------------------------------

class SigmaFactorTranscriptInitiation(TranscriptInitiation):
    """Transcript initiation with full N-sigma factor competition.

    Inherits all original TranscriptInitiation logic and extends it with:
    - Per-promoter sigma preference weights (n_TUs × N_sigma)
    - N-sigma Mauri & Klumpp equilibrium solver
    - ppGpp modulation of σ70 affinity
    - Extended listener outputs for all sigma fractions

    Exponential phase: low ppGpp → σ70 dominates → rRNA/ribosomal genes high
    Stationary phase:  high ppGpp → σ38 rises → stress genes upregulated
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = dict(
        TranscriptInitiation.config_schema,
        sigma_ids={"_type": "list[string]", "_default": [s[0] for s in SIGMA_FACTORS]},
        sigma_Kd_nM={"_type": "array[float]", "_default": np.array([s[1] for s in SIGMA_FACTORS])},
        sigma_ppgpp_sensitivity={"_type": "array[float]", "_default": np.array([s[2] for s in SIGMA_FACTORS])},
        sigma_free_fractions={"_type": "array[float]", "_default": np.array([0.80, 0.03, 0.015, 0.10, 0.05])},
        promoter_sigma_weights={"_type": "array[float]", "_default": np.array([])},
        oxyr_tu_indices={"_type": "list[integer]", "_default": [31, 1572, 1708, 1941, 2077, 2600, 2975, 3010]},
        soxrs_tu_indices={"_type": "list[integer]", "_default": [1608, 1806, 1814, 2621, 2727, 2728]},
    )

    def initialize(self, config):
        super().initialize(config)

        self._sigma_ids  = list(self.parameters.get("sigma_ids", [s[0] for s in SIGMA_FACTORS]))
        self._Kd_nM      = np.asarray(self.parameters.get("sigma_Kd_nM", [s[1] for s in SIGMA_FACTORS]), dtype=float)
        self._ppgpp_sens = np.asarray(self.parameters.get("sigma_ppgpp_sensitivity", [s[2] for s in SIGMA_FACTORS]), dtype=float)
        self._free_fracs = np.asarray(
            self.parameters.get("sigma_free_fractions", np.array([1.0, 0.02, 0.01, 0.05, 0.02])),
            dtype=float,
        )

        raw_w = np.asarray(self.parameters.get("promoter_sigma_weights", np.array([])), dtype=float)
        self._sigma_weights = self._build_weights(raw_w)

        self._oxyr_tu_idx  = np.array(self.parameters.get("oxyr_tu_indices",
                                      [31, 1572, 1708, 1941, 2077, 2600, 2975, 3010]), dtype=int)
        self._soxrs_tu_idx = np.array(self.parameters.get("soxrs_tu_indices",
                                      [1608, 1806, 1814, 2621, 2727, 2728]), dtype=int)

        # Bulk indices — resolved lazily
        self._sigma_idx = None

        # Diagnostics
        self.last_fractions = np.zeros(N_SIGMA)
        self.last_fractions[0] = 1.0
        self.last_Es = np.zeros(N_SIGMA)
        self._last_ppgpp_uM = 0.0

    # ------------------------------------------------------------------
    # Override inputs/outputs to add sigma listener
    # ------------------------------------------------------------------

    def inputs(self):
        schema = super().inputs()
        schema["listeners"]["oxidative_stress"] = {
            "oxyr_fold_change":  {"_type": "float", "_default": 1.0},
            "soxrs_fold_change": {"_type": "float", "_default": 1.0},
        }
        return schema

    def outputs(self):
        schema = super().outputs()
        if isinstance(schema, tuple):
            schema = schema[0]
        schema.setdefault("listeners", {})
        schema["listeners"]["sigma_factors"] = {
            "f_sigma70":  {"_type": "overwrite[float]", "_default": 0.0},
            "f_sigma38":  {"_type": "overwrite[float]", "_default": 0.0},
            "f_sigma32":  {"_type": "overwrite[float]", "_default": 0.0},
            "f_sigma24":  {"_type": "overwrite[float]", "_default": 0.0},
            "f_sigma54":  {"_type": "overwrite[float]", "_default": 0.0},
            "Es70_count": {"_type": "overwrite[float]", "_default": 0.0},
            "EsS_count":  {"_type": "overwrite[float]", "_default": 0.0},
            "ppgpp_uM":   {"_type": "overwrite[float]", "_default": 0.0},
            "K_E70_eff_nM": {"_type": "overwrite[float]", "_default": 0.0},
            "phase":      {"_type": "overwrite[float]", "_default": 0.0},
            "oxyr_fold_change":  {"_type": "overwrite[float]", "_default": 1.0},
            "soxrs_fold_change": {"_type": "overwrite[float]", "_default": 1.0},
        }
        return schema

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_weights(self, raw: np.ndarray) -> np.ndarray:
        """Return row-normalised (n_TUs, N_sigma) weight matrix."""
        if raw.size > 0 and raw.ndim == 2 and raw.shape[1] == N_SIGMA:
            w = raw.copy()
        else:
            # Fallback: all σ70
            w = np.zeros((self.n_TUs, N_SIGMA))
            w[:, 0] = 1.0
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return w / row_sums

    def _resolve_sigma_indices(self, states) -> None:
        bulk_ids = states["bulk"]["id"]
        self._sigma_idx = []
        for sid in self._sigma_ids:
            try:
                self._sigma_idx.append(bulk_name_to_idx(sid, bulk_ids))
            except Exception:
                self._sigma_idx.append(None)

    def _nM_to_mol(self, K_nM: float, states) -> float:
        cell_mass_fg = float(states["listeners"]["mass"]["cell_mass"])
        cell_volume_L = (cell_mass_fg * 1e-15) / self.cell_density.magnitude
        return (1e-9 * self.n_avogadro.magnitude * cell_volume_L) * K_nM

    def _get_ppgpp_uM(self, states) -> float:
        if self.ppgpp_idx is None:
            return 0.0
        cell_mass_fg = float(states["listeners"]["mass"]["cell_mass"])
        cell_volume_L = (cell_mass_fg * 1e-15) / self.cell_density.magnitude
        if cell_volume_L <= 0:
            return 0.0
        ppgpp_count = float(counts(states["bulk"], self.ppgpp_idx))
        conc_M = ppgpp_count / (self.n_avogadro.magnitude * cell_volume_L)
        return conc_M * 1e6

    def _apply_sigma_competition(self, TU_index: np.ndarray, states) -> None:
        """Scale promoter_init_probs by N-sigma holoenzyme fractions and
        OxyR/SoxRS stress-responsive gene upregulation."""
        # Sigma factor counts — apply sequestration fractions
        sigma_counts = np.zeros(N_SIGMA)
        for i, idx in enumerate(self._sigma_idx):
            if idx is not None:
                sigma_counts[i] = float(counts(states["bulk"], idx))
            else:
                sigma_counts[i] = [5700, 300, 50, 30, 100][i]
        sigma_counts = sigma_counts * self._free_fracs

        # Core RNAP
        E_total = float(counts(states["bulk"], self.inactive_RNAP_idx))

        # Effective Kd in molecule units
        ppgpp_uM = self._get_ppgpp_uM(states)
        mol_per_nM = self._nM_to_mol(1.0, states)
        K_eff = np.array([
            self._Kd_nM[i] * mol_per_nM * np.exp(self._ppgpp_sens[i] * ppgpp_uM)
            for i in range(N_SIGMA)
        ])

        # Solve competition
        fractions = holoenzyme_fractions(E_total, sigma_counts, K_eff)
        Es = solve_n_sigma_competition(E_total, sigma_counts, K_eff)

        # Per-promoter modulation
        modulation = self._sigma_weights[TU_index, :].dot(fractions)
        modulation = np.clip(modulation, 0.0, 1.0)
        self.promoter_init_probs *= modulation

        # OxyR/SoxRS regulon upregulation
        try:
            oxyr_fc  = float(states["listeners"]["oxidative_stress"]["oxyr_fold_change"])
        except (KeyError, TypeError):
            oxyr_fc = 1.0
        try:
            soxrs_fc = float(states["listeners"]["oxidative_stress"]["soxrs_fold_change"])
        except (KeyError, TypeError):
            soxrs_fc = 1.0

        if oxyr_fc > 1.0 and len(self._oxyr_tu_idx) > 0:
            for tu_i in self._oxyr_tu_idx:
                mask = (TU_index == tu_i)
                if mask.any():
                    self.promoter_init_probs[mask] *= oxyr_fc

        if soxrs_fc > 1.0 and len(self._soxrs_tu_idx) > 0:
            for tu_i in self._soxrs_tu_idx:
                mask = (TU_index == tu_i)
                if mask.any():
                    self.promoter_init_probs[mask] *= soxrs_fc

        self._last_oxyr_fc  = oxyr_fc
        self._last_soxrs_fc = soxrs_fc

        # NaN guard + renormalise
        self.promoter_init_probs = np.nan_to_num(
            self.promoter_init_probs, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.promoter_init_probs = np.clip(self.promoter_init_probs, 0.0, None)
        prob_sum = self.promoter_init_probs.sum()
        if prob_sum > 0.0:
            self.promoter_init_probs /= prob_sum
        else:
            n = len(self.promoter_init_probs)
            self.promoter_init_probs = np.ones(n) / n

        # Store diagnostics
        self.last_fractions = fractions
        self.last_Es        = Es
        self._last_ppgpp_uM = ppgpp_uM
        self._last_K_E70_eff_nM = float(K_eff[0] / mol_per_nM) if mol_per_nM > 0 else 0.0

    # ------------------------------------------------------------------
    # Override _prepare — parent logic + sigma competition
    # ------------------------------------------------------------------

    def _prepare(self, states):
        # First-call: resolve sigma indices alongside parent indices
        if self._sigma_idx is None:
            self._resolve_sigma_indices(states)

        # Run parent _prepare → sets self.promoter_init_probs
        super()._prepare(states)

        # No chromosomes → nothing to modulate
        if states["full_chromosomes"]["_entryState"].sum() == 0:
            return

        TU_index = attrs(states["promoters"], ["TU_index"])[0]
        self._apply_sigma_competition(TU_index, states)

    # ------------------------------------------------------------------
    # Override _evolve — NaN-safe + sigma listener writes
    # ------------------------------------------------------------------

    def _evolve(self, states):
        # Sanitize before overcrowding loop
        if hasattr(self, "promoter_init_probs") and len(self.promoter_init_probs) > 0:
            self.promoter_init_probs = np.nan_to_num(
                self.promoter_init_probs, nan=0.0, posinf=0.0, neginf=0.0
            )
            self.promoter_init_probs = np.clip(self.promoter_init_probs, 0.0, None)
            s = self.promoter_init_probs.sum()
            if s > 0.0:
                self.promoter_init_probs /= s
            else:
                n = len(self.promoter_init_probs)
                self.promoter_init_probs = np.ones(n) / n

        update = super()._evolve(states)

        # Determine phase: 0=exponential, 1=stationary
        ppgpp_uM = getattr(self, "_last_ppgpp_uM", 0.0)
        phase = float(np.clip(ppgpp_uM / 200.0, 0.0, 1.0))

        update.setdefault("listeners", {})
        update["listeners"]["sigma_factors"] = {
            "f_sigma70":    float(self.last_fractions[0]),
            "f_sigma38":    float(self.last_fractions[1]),
            "f_sigma32":    float(self.last_fractions[2]),
            "f_sigma24":    float(self.last_fractions[3]),
            "f_sigma54":    float(self.last_fractions[4]),
            "Es70_count":   float(self.last_Es[0]),
            "EsS_count":    float(self.last_Es[1]),
            "ppgpp_uM":     ppgpp_uM,
            "K_E70_eff_nM": float(getattr(self, "_last_K_E70_eff_nM", 0.0)),
            "phase":        phase,
            "oxyr_fold_change":  float(getattr(self, "_last_oxyr_fc", 1.0)),
            "soxrs_fold_change": float(getattr(self, "_last_soxrs_fc", 1.0)),
        }
        return update
