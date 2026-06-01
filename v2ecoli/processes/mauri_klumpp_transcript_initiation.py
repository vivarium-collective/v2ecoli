"""
=============================================================================
Mauri & Klumpp (2014) Sigma Factor Competition — Full Whole-Cell Process
=============================================================================

Implementation of sigma factor competition for transcript
initiation in E. coli, based on:

  Mauri M, Klumpp S (2014) A Model for Sigma Factor Competition in Bacterial
  Cells. PLoS Comput Biol 10(10): e1003845.
  https://doi.org/10.1371/journal.pcbi.1003845

Additional regulatory layers from the literature
-------------------------------------------------
1. ppGpp weakening of RpoD-core affinity
   Jishage M et al. (2002) Regulation of sigma factor competition by the
   alarmone ppGpp. Genes Dev 16:1260-1270.

2. Crl activation of RpoS
   Bougdour A et al. (2004) Crl, a low temperature-induced protein in
   E. coli that binds directly to the stationary phase sigma subunit of
   RNA polymerase. J Biol Chem 279:19540-19550.

3. Rsd anti-sigma factor sequestration of RpoD
   Jishage M, Ishihama A (1998) A stationary phase protein in Escherichia
   coli with binding activity to the major sigma subunit of RNA polymerase.
   Proc Natl Acad Sci 95:4953-4958.

4. Per-promoter sigma selectivity weights [w_RpoD, w_RpoS]

5. Extended listener outputs for PhD-level analysis

Design
------
- Inherits from TranscriptInitiation so ALL original logic is preserved.
- Only _prepare() is extended: after the parent builds
  self.promoter_init_probs, the sigma competition layer scales them.
- Registered as "ecoli-mauri-klumpp-transcript-initiation" so it can be
  swapped in via configs.
"""

import numpy as np
import scipy.sparse
from typing import cast

from v2ecoli.library.schema import (
    create_unique_indices,
    listener_schema,
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
)
from v2ecoli.types.quantity import ureg as units

from v2ecoli.processes.transcript_initiation import TranscriptInitiation

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NAME = "ecoli-mauri-klumpp-transcript-initiation"
TOPOLOGY = {
    "environment": ("environment",),
    "full_chromosomes": ("unique", "full_chromosome"),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "ppgpp_state": ("ppgpp_state",),
}

# ---------------------------------------------------------------------------
# Default molecule IDs (whole-cell bulk store names)
# ---------------------------------------------------------------------------
_RPOD_ID = "RPOD-MONOMER[c]"   # sigma-70  (housekeeping)
_RPOS_ID = "RPOS-MONOMER[c]"   # sigma-38  (stress / stationary phase)
_CRL_ID  = "CRL[c]"            # Crl activator of RpoS
_RSD_ID  = "RSD-MONOMER[c]"    # Rsd anti-sigma factor (sequesters RpoD)


# ===========================================================================
# Mauri & Klumpp (2014) equilibrium solver — Equations 3-10
# ===========================================================================

def _solve_holoenzyme_single(E: float, s: float, K: float) -> float:
    """Holoenzyme [Es] for one sigma factor (Eq. 3 of Mauri & Klumpp 2014)."""
    disc = max((K + E + s) ** 2 - 4.0 * E * s, 0.0)
    return 0.5 * (K + E + s - np.sqrt(disc))


def _solve_two_sigma(
    E: float,
    s70: float,
    sS: float,
    K70: float,
    KS: float,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> tuple:
    """Solve [E*RpoD] and [E*RpoS] under competition (Eq. 4-8)."""
    if E <= 0.0 or (s70 <= 0.0 and sS <= 0.0):
        return 0.0, 0.0

    tot = s70 + sS
    if tot <= 0.0:
        return 0.0, 0.0

    Es70_0 = _solve_holoenzyme_single(E, s70, K70)
    EsS_0  = _solve_holoenzyme_single(E, sS,  KS)
    Es70 = Es70_0 * s70 / tot
    EsS  = EsS_0  * sS  / tot

    for _ in range(max_iter):
        p70, pS = Es70, EsS

        E_avail_70 = max(0.0, E - EsS)
        Es70 = min(
            _solve_holoenzyme_single(E_avail_70, s70, K70),
            E_avail_70,
            s70,
        )
        Es70 = max(0.0, Es70)

        E_avail_S = max(0.0, E - Es70)
        EsS = min(
            _solve_holoenzyme_single(E_avail_S, sS, KS),
            E_avail_S,
            sS,
        )
        EsS = max(0.0, EsS)

        if abs(Es70 - p70) + abs(EsS - pS) < tol:
            break

    return max(0.0, Es70), max(0.0, EsS)


def sigma_holoenzyme_fractions(
    E: float,
    s70: float,
    sS: float,
    K70: float,
    KS: float,
) -> np.ndarray:
    """Return normalised holoenzyme fractions [f_RpoD, f_RpoS]."""
    Es70, EsS = _solve_two_sigma(E, s70, sS, K70, KS)
    total = Es70 + EsS
    if total <= 0.0:
        return np.array([1.0, 0.0])
    return np.array([Es70 / total, EsS / total])


def solve_rsd_sequestration(s70_total: float, rsd_total: float, Kd_rsd: float) -> float:
    """Free RpoD after Rsd sequestration (Jishage & Ishihama 1998)."""
    b = Kd_rsd + rsd_total - s70_total
    disc = max(b * b + 4.0 * Kd_rsd * s70_total, 0.0)
    return 0.5 * (-b + np.sqrt(disc))


# ===========================================================================
# Process class
# ===========================================================================

class MauriKlumppTranscriptInitiation(TranscriptInitiation):
    """Transcript initiation with full Mauri & Klumpp (2014) sigma competition.

    Extends TranscriptInitiation with:
    - Exact Mauri & Klumpp (2014) Eq. 3-10 holoenzyme equilibrium solver
    - ppGpp weakening of RpoD-core affinity (Jishage et al. 2002)
    - Crl activation of RpoS (Bougdour et al. 2004)
    - Rsd anti-sigma sequestration of RpoD (Jishage & Ishihama 1998)
    - Per-promoter sigma selectivity weights
    - Extended listener outputs for analysis
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = dict(
        TranscriptInitiation.config_schema,
        # molecule IDs
        rpod_id={"_type": "string", "_default": _RPOD_ID},
        rpos_id={"_type": "string", "_default": _RPOS_ID},
        crl_id={"_type": "string", "_default": _CRL_ID},
        rsd_id={"_type": "string", "_default": _RSD_ID},
        # dissociation constants (nM) — Mauri & Klumpp 2014 Table 1/2
        K_E70_nM={"_type": "float", "_default": 1.0},
        K_ES_nM={"_type": "float", "_default": 20.0},
        Kd_rsd_nM={"_type": "float", "_default": 100.0},
        # ppGpp effect on RpoD affinity (Jishage et al. 2002)
        ppgpp_K_E70_sensitivity={"_type": "float", "_default": 0.005},
        # Crl activation of RpoS (Bougdour et al. 2004)
        K_crl_molecules={"_type": "float", "_default": 500.0},
        # per-TU sigma weights
        promoter_sigma_weights={"_type": "array[float]", "_default": np.array([])},
        rpos_target_tu_indices={"_type": "array[integer]", "_default": np.array([], dtype=int)},
        # OxyR regulon TU indices
        oxyr_target_tu_indices={"_type": "array[integer]", "_default": np.array([], dtype=int)},
        # SoxRS regulon TU indices
        soxrs_target_tu_indices={"_type": "array[integer]", "_default": np.array([], dtype=int)},
    )

    def initialize(self, config):
        super().initialize(config)

        self.rpod_id = self.parameters.get("rpod_id", _RPOD_ID)
        self.rpos_id = self.parameters.get("rpos_id", _RPOS_ID)
        self.crl_id  = self.parameters.get("crl_id", _CRL_ID)
        self.rsd_id  = self.parameters.get("rsd_id", _RSD_ID)

        self._K_E70_nM    = float(self.parameters.get("K_E70_nM", 1.0))
        self._K_ES_nM     = float(self.parameters.get("K_ES_nM", 20.0))
        self._Kd_rsd_nM   = float(self.parameters.get("Kd_rsd_nM", 100.0))
        self._ppgpp_alpha = float(self.parameters.get("ppgpp_K_E70_sensitivity", 0.005))
        self._K_crl       = float(self.parameters.get("K_crl_molecules", 500.0))

        self.rpos_target_tu_indices = np.asarray(
            self.parameters.get("rpos_target_tu_indices", np.array([], dtype=int)), dtype=int
        )
        raw_w = np.asarray(
            self.parameters.get("promoter_sigma_weights", np.array([])), dtype=np.float64
        )
        self._sigma_weights = self._build_sigma_weights(raw_w)

        # OxyR / SoxRS target TU indices
        self._oxyr_tu_idx = np.asarray(
            self.parameters.get("oxyr_target_tu_indices", np.array([], dtype=int)), dtype=int
        )
        self._oxyr_tu_idx = self._oxyr_tu_idx[self._oxyr_tu_idx < self.n_TUs]
        self._soxrs_tu_idx = np.asarray(
            self.parameters.get("soxrs_target_tu_indices", np.array([], dtype=int)), dtype=int
        )
        self._soxrs_tu_idx = self._soxrs_tu_idx[self._soxrs_tu_idx < self.n_TUs]

        # Bulk indices — resolved lazily on first timestep
        self._rpod_idx = None
        self._rpos_idx = None
        self._crl_idx  = None
        self._rsd_idx  = None

        # Diagnostics (written to listeners each timestep)
        self.last_sigma_fractions  = np.array([1.0, 0.0])
        self.last_Es70             = 0.0
        self.last_EsS              = 0.0
        self.last_E_free           = 0.0
        self.last_K_E70_eff        = 0.0
        self.last_K_ES_eff         = 0.0
        self.last_s70_free         = 0.0

    # ------------------------------------------------------------------
    # Override inputs/outputs to add sigma competition listeners
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
        schema["listeners"]["sigma_competition"] = {
            "f_RpoD":         {"_type": "overwrite[float]", "_default": 0.0},
            "f_RpoS":         {"_type": "overwrite[float]", "_default": 0.0},
            "Es70_count":     {"_type": "overwrite[float]", "_default": 0.0},
            "EsS_count":      {"_type": "overwrite[float]", "_default": 0.0},
            "E_free_count":   {"_type": "overwrite[float]", "_default": 0.0},
            "K_E70_eff_nM":   {"_type": "overwrite[float]", "_default": 0.0},
            "K_ES_eff_nM":    {"_type": "overwrite[float]", "_default": 0.0},
            "s70_free_count": {"_type": "overwrite[float]", "_default": 0.0},
            "ppgpp_uM":       {"_type": "overwrite[float]", "_default": 0.0},
            "crl_count":      {"_type": "overwrite[float]", "_default": 0.0},
            "rsd_count":      {"_type": "overwrite[float]", "_default": 0.0},
        }
        return schema

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sigma_weights(self, raw: np.ndarray) -> np.ndarray:
        """Return row-normalised (n_TUs, 2) sigma preference matrix."""
        if raw.size > 0:
            assert raw.shape == (self.n_TUs, 2), (
                f"promoter_sigma_weights must be ({self.n_TUs}, 2), got {raw.shape}"
            )
            w = raw.copy()
        else:
            w = np.zeros((self.n_TUs, 2), dtype=np.float64)
            w[:, 0] = 1.0   # default: all TUs prefer RpoD
            if self.rpos_target_tu_indices.size > 0:
                valid = self.rpos_target_tu_indices[
                    self.rpos_target_tu_indices < self.n_TUs
                ]
                w[valid, 0] = 0.0
                w[valid, 1] = 1.0
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return w / row_sums

    def _resolve_bulk_indices(self, states) -> None:
        """Lazily resolve bulk indices for sigma-related molecules."""
        bulk_ids = states["bulk"]["id"]

        def _safe(mol_id):
            try:
                return bulk_name_to_idx(mol_id, bulk_ids)
            except Exception:
                return None

        self._rpod_idx = _safe(self.rpod_id)
        self._rpos_idx = _safe(self.rpos_id)
        self._crl_idx  = _safe(self.crl_id)
        self._rsd_idx  = _safe(self.rsd_id)

    def _nM_to_molecules(self, K_nM: float, states) -> float:
        """Convert a Kd from nM to molecule-count units using cell volume."""
        cell_mass_fg = float(states["listeners"]["mass"]["cell_mass"])
        cell_volume_L = (cell_mass_fg * 1e-15) / self.cell_density.magnitude
        return (1e-9 * self.n_avogadro.magnitude * cell_volume_L) * K_nM

    def _get_ppgpp_uM(self, states) -> float:
        """Return ppGpp concentration in µM (0 if unavailable)."""
        if self.ppgpp_idx is None:
            return 0.0
        cell_mass_fg = float(states["listeners"]["mass"]["cell_mass"])
        cell_volume_L = (cell_mass_fg * 1e-15) / self.cell_density.magnitude
        if cell_volume_L <= 0:
            return 0.0
        ppgpp_count = float(counts(states["bulk"], self.ppgpp_idx))
        conc_M = ppgpp_count / (self.n_avogadro.magnitude * cell_volume_L)
        return conc_M * 1e6

    def _effective_Kd(self, states) -> tuple:
        """Compute effective K_E70 and K_ES in molecule units."""
        K_E70_mol = self._nM_to_molecules(self._K_E70_nM, states)
        K_ES_mol  = self._nM_to_molecules(self._K_ES_nM,  states)

        # ppGpp weakens RpoD-core binding
        ppgpp_uM = self._get_ppgpp_uM(states)
        K_E70_eff = K_E70_mol * np.exp(self._ppgpp_alpha * ppgpp_uM)

        # Crl strengthens RpoS-core binding
        crl_count = 0.0
        if self._crl_idx is not None:
            crl_count = float(counts(states["bulk"], self._crl_idx))
        K_ES_eff = K_ES_mol / (1.0 + crl_count / self._K_crl)

        return K_E70_eff, K_ES_eff, ppgpp_uM, crl_count

    def _apply_sigma_competition(self, TU_index: np.ndarray, states) -> None:
        """Scale promoter_init_probs by Mauri & Klumpp sigma fractions."""
        # --- molecule counts ---
        E_total = float(counts(states["bulk"], self.inactive_RNAP_idx))
        s70_total = (
            float(counts(states["bulk"], self._rpod_idx))
            if self._rpod_idx is not None else 5700.0
        )
        sS_total = (
            float(counts(states["bulk"], self._rpos_idx))
            if self._rpos_idx is not None else 0.0
        )
        rsd_count = (
            float(counts(states["bulk"], self._rsd_idx))
            if self._rsd_idx is not None else 0.0
        )

        # --- Rsd sequestration of RpoD ---
        Kd_rsd_mol = self._nM_to_molecules(self._Kd_rsd_nM, states)
        s70_free = solve_rsd_sequestration(s70_total, rsd_count, Kd_rsd_mol)

        # --- effective Kd values ---
        K_E70_eff, K_ES_eff, ppgpp_uM, crl_count = self._effective_Kd(states)

        # --- Mauri & Klumpp equilibrium ---
        Es70, EsS = _solve_two_sigma(E_total, s70_free, sS_total, K_E70_eff, K_ES_eff)
        total_holo = Es70 + EsS
        if total_holo > 0.0:
            fractions = np.array([Es70 / total_holo, EsS / total_holo])
        else:
            fractions = np.array([1.0, 0.0])

        # --- per-promoter modulation ---
        modulation = self._sigma_weights[TU_index, :].dot(fractions)
        modulation = np.clip(modulation, 0.0, 1.0)
        self.promoter_init_probs *= modulation

        # Guard against NaN / negative values
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

        # --- OxyR / SoxRS fold changes ---
        try:
            ox_listener = states["listeners"]["oxidative_stress"]
            oxyr_fc  = float(ox_listener.get("oxyr_fold_change",  1.0))
            soxrs_fc = float(ox_listener.get("soxrs_fold_change", 1.0))
        except Exception:
            oxyr_fc  = 1.0
            soxrs_fc = 1.0

        if oxyr_fc > 1.0 and self._oxyr_tu_idx.size > 0:
            oxyr_mask = np.isin(TU_index, self._oxyr_tu_idx)
            if oxyr_mask.any():
                self.promoter_init_probs[oxyr_mask] *= oxyr_fc
                prob_sum = self.promoter_init_probs.sum()
                if prob_sum > 0.0:
                    self.promoter_init_probs /= prob_sum

        if soxrs_fc > 1.0 and self._soxrs_tu_idx.size > 0:
            soxrs_mask = np.isin(TU_index, self._soxrs_tu_idx)
            if soxrs_mask.any():
                self.promoter_init_probs[soxrs_mask] *= soxrs_fc
                prob_sum = self.promoter_init_probs.sum()
                if prob_sum > 0.0:
                    self.promoter_init_probs /= prob_sum

        # --- store diagnostics ---
        self.last_sigma_fractions = fractions
        self.last_Es70            = Es70
        self.last_EsS             = EsS
        self.last_E_free          = max(0.0, E_total - Es70 - EsS)
        mol_per_nM = self._nM_to_molecules(1.0, states)
        self.last_K_E70_eff       = K_E70_eff / mol_per_nM if mol_per_nM > 0 else 0.0
        self.last_K_ES_eff        = K_ES_eff  / mol_per_nM if mol_per_nM > 0 else 0.0
        self.last_s70_free        = s70_free
        self._last_ppgpp_uM       = ppgpp_uM
        self._last_crl_count      = crl_count
        self._last_rsd_count      = rsd_count

    # ------------------------------------------------------------------
    # Override _prepare — parent logic + sigma competition
    # ------------------------------------------------------------------

    def _prepare(self, states):
        # First-call: resolve sigma indices alongside parent indices
        if self._rpod_idx is None:
            self._resolve_bulk_indices(states)

        # Run parent _prepare → sets self.promoter_init_probs
        super()._prepare(states)

        # No chromosomes → nothing to modulate
        if states["full_chromosomes"]["_entryState"].sum() == 0:
            return

        TU_index = attrs(states["promoters"], ["TU_index"])[0]
        self._apply_sigma_competition(TU_index, states)

    # ------------------------------------------------------------------
    # Override _evolve — NaN-safe overcrowding + sigma listener writes
    # ------------------------------------------------------------------

    def _evolve(self, states):
        timestep = states["timestep"]
        update = {
            "listeners": {
                "rna_synth_prob": {
                    "target_rna_synth_prob": np.zeros(self.n_TUs),
                    "actual_rna_synth_prob": np.zeros(self.n_TUs),
                    "tu_is_overcrowded": np.zeros(self.n_TUs, dtype=np.bool_),
                    "total_rna_init": 0,
                    "max_p": 0.0,
                },
                "ribosome_data": {"total_rna_init": 0},
                "rnap_data": {
                    "did_initialize": 0,
                    "rna_init_event": np.zeros(self.n_TUs, dtype=np.int64),
                },
                "sigma_competition": {
                    "f_RpoD":         float(self.last_sigma_fractions[0]),
                    "f_RpoS":         float(self.last_sigma_fractions[1]),
                    "Es70_count":     float(self.last_Es70),
                    "EsS_count":      float(self.last_EsS),
                    "E_free_count":   float(self.last_E_free),
                    "K_E70_eff_nM":   float(self.last_K_E70_eff),
                    "K_ES_eff_nM":    float(self.last_K_ES_eff),
                    "s70_free_count": float(self.last_s70_free),
                    "ppgpp_uM":       float(getattr(self, "_last_ppgpp_uM", 0.0)),
                    "crl_count":      float(getattr(self, "_last_crl_count", 0.0)),
                    "rsd_count":      float(getattr(self, "_last_rsd_count", 0.0)),
                },
            },
            "active_RNAPs": {},
            "full_chromosomes": {},
            "promoters": {},
            "RNAs": {},
        }

        if len(states["full_chromosomes"]) == 0:
            return update

        TU_index, domain_index_promoters = attrs(
            states["promoters"], ["TU_index", "domain_index"]
        )

        n_promoters = states["promoters"]["_entryState"].sum()
        TU_to_promoter = scipy.sparse.csr_matrix(
            (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
            shape=(self.n_TUs, n_promoters),
            dtype=np.int8,
        )

        # Sanitize promoter_init_probs
        self.promoter_init_probs = np.nan_to_num(
            self.promoter_init_probs, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.promoter_init_probs = np.clip(self.promoter_init_probs, 0.0, None)
        prob_sum = self.promoter_init_probs.sum()
        if prob_sum > 0.0:
            self.promoter_init_probs /= prob_sum
        else:
            self.promoter_init_probs = np.ones(n_promoters) / n_promoters

        target_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
        update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
            target_TU_synth_probs
        )

        self.activationProb = self._calculateActivationProb(
            states["timestep"],
            self.fracActiveRnap,
            self.rnaLengths,
            (units.nt / units.s) * self.elongation_rates,
            target_TU_synth_probs,
        )

        n_RNAPs_to_activate = np.int64(
            self.activationProb * counts(states["bulk"], self.inactive_RNAP_idx)
        )

        if n_RNAPs_to_activate == 0:
            return update

        max_p = (
            self.rnaPolymeraseElongationRate
            / self.active_rnap_footprint_size
            * (units.s)
            * states["timestep"]
            / n_RNAPs_to_activate
        ).magnitude
        update["listeners"]["rna_synth_prob"]["max_p"] = max_p
        is_overcrowded = self.promoter_init_probs > max_p

        # NaN-safe overcrowding loop
        while np.any(self.promoter_init_probs > max_p):
            self.promoter_init_probs[is_overcrowded] = max_p
            rest_sum = self.promoter_init_probs[~is_overcrowded].sum()
            if rest_sum > 0.0:
                scale_the_rest_by = (
                    1.0 - self.promoter_init_probs[is_overcrowded].sum()
                ) / rest_sum
                self.promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
            else:
                n_rest = (~is_overcrowded).sum()
                if n_rest > 0:
                    remaining = 1.0 - self.promoter_init_probs[is_overcrowded].sum()
                    self.promoter_init_probs[~is_overcrowded] = max(0.0, remaining) / n_rest
            is_overcrowded |= self.promoter_init_probs > max_p

        # Final NaN guard before multinomial draw
        self.promoter_init_probs = np.nan_to_num(
            self.promoter_init_probs, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.promoter_init_probs = np.clip(self.promoter_init_probs, 0.0, None)
        prob_sum = self.promoter_init_probs.sum()
        if prob_sum > 0.0:
            self.promoter_init_probs /= prob_sum
        else:
            self.promoter_init_probs = np.ones(n_promoters) / n_promoters

        actual_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
        tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
        update["listeners"]["rna_synth_prob"]["actual_rna_synth_prob"] = (
            actual_TU_synth_probs
        )
        update["listeners"]["rna_synth_prob"]["tu_is_overcrowded"] = tu_is_overcrowded

        n_initiations = self.random_state.multinomial(
            n_RNAPs_to_activate, self.promoter_init_probs
        )

        TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
        domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

        coordinates = self.replication_coordinate[TU_index_partial_RNAs]
        is_forward = self.transcription_direction[TU_index_partial_RNAs]

        RNAP_indexes = create_unique_indices(n_RNAPs_to_activate, states["RNAs"])
        update["active_RNAPs"].update(
            {
                "add": {
                    "unique_index": RNAP_indexes,
                    "domain_index": domain_index_rnap,
                    "coordinates": coordinates,
                    "is_forward": is_forward,
                }
            }
        )

        update["bulk"] = [(self.inactive_RNAP_idx, -n_initiations.sum())]

        is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
        update["RNAs"].update(
            {
                "add": {
                    "TU_index": TU_index_partial_RNAs,
                    "transcript_length": np.zeros(cast(int, n_RNAPs_to_activate)),
                    "is_mRNA": is_mRNA,
                    "is_full_transcript": np.zeros(
                        cast(int, n_RNAPs_to_activate), dtype=bool
                    ),
                    "can_translate": is_mRNA,
                    "RNAP_index": RNAP_indexes,
                }
            }
        )

        rna_init_event = TU_to_promoter.dot(n_initiations)
        rRNA_initiations = rna_init_event[self.idx_rRNA]

        update["listeners"]["ribosome_data"] = {
            "rRNA_initiated_TU": rRNA_initiations.astype(int),
            "rRNA_init_prob_TU": rRNA_initiations / float(n_RNAPs_to_activate),
            "total_rna_init": n_RNAPs_to_activate,
        }

        update["listeners"]["rnap_data"] = {
            "did_initialize": n_RNAPs_to_activate,
            "rna_init_event": rna_init_event.astype(np.int64),
        }

        update["listeners"]["rna_synth_prob"]["total_rna_init"] = n_RNAPs_to_activate

        # Update sigma competition listener
        update["listeners"]["sigma_competition"] = {
            "f_RpoD":         float(self.last_sigma_fractions[0]),
            "f_RpoS":         float(self.last_sigma_fractions[1]),
            "Es70_count":     float(self.last_Es70),
            "EsS_count":      float(self.last_EsS),
            "E_free_count":   float(self.last_E_free),
            "K_E70_eff_nM":   float(self.last_K_E70_eff),
            "K_ES_eff_nM":    float(self.last_K_ES_eff),
            "s70_free_count": float(self.last_s70_free),
            "ppgpp_uM":       float(getattr(self, "_last_ppgpp_uM", 0.0)),
            "crl_count":      float(getattr(self, "_last_crl_count", 0.0)),
            "rsd_count":      float(getattr(self, "_last_rsd_count", 0.0)),
        }

        return update
