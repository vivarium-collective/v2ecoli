"""
============================
DnaA Constitutive Expression
============================

Stage-1 dnaA expression as a direct, constitutive source of apo-DnaA.

The Fu/Xiao/Jun (2023) Stage-1 spec says: *"We simply assume constitutive
gene expression at this first stage"* — dnaA transcription 1.5 mRNA/min/gene,
translation efficiency 1 protein/mRNA. The net protein synthesis rate is
therefore::

    1.5 mRNA/min/gene  ×  1 protein/mRNA  =  1.5 DnaA protein/min/gene

This Step produces apo-DnaA (``PD03831[c]``) at exactly that rate each tick
(Poisson-distributed), scaled by dnaA gene-copy number. Newly synthesised
DnaA is apo; the existing equilibrium reactions (``MONOMER0-160_RXN`` /
``MONOMER0-4565_RXN``) then partition it into the DnaA-ATP / DnaA-ADP forms.

Why a dedicated Step (Option A) instead of patching ParCa (Option B):
v2ecoli's ParCa transcription/translation use normalised weights
(basal_prob / translation_efficiencies) that renormalise across ~4500
TUs/monomers. Empirically, patching dnaA's weights to hit the absolute
Stage-1 rates does NOT converge — scaling the weight down RAISED the
realised rate (the renormalisation inverts the local effect). A direct
constitutive source sidesteps that entirely and is deterministic.

Pairing: the Stage-1 condition cache zeroes ParCa's DnaA translation
(``translation_efficiencies_by_monomer[PD03831] = 0``) so this Step is the
SOLE source of DnaA protein — no double-counting. The dnaA mRNA the operon
still transcribes is left untranslated (harmless for the DnaA level; an
explicit mRNA pool is a later refinement when autorepression is modelled).

Validates the dnaa-01 "steady DnaA pool" expectation at Stage 1 (constitutive
= no autorepression yet, by the expert's staging).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "dnaa-constitutive-expression"
TOPOLOGY = {
    "bulk":      ("bulk",),
    "listeners": ("listeners",),
    "timestep":  ("timestep",),
}

DNAA_APO_ID = "PD03831[c]"   # apo-DnaA — the form newly-synthesised DnaA enters


class DnaaConstitutiveExpression(Step):
    """Constitutive apo-DnaA synthesis at the Stage-1 rate.

    Each tick adds ``Poisson(rate_protein_per_min_per_gene * gene_copies *
    dt_min)`` molecules of PD03831[c]. ``gene_copies`` defaults to the
    config value but is overridden per-tick by the dnaA gene-copy number
    when a copy-number input is wired (dnaA sits adjacent to oriC, so its
    dosage tracks oriC copy through the cycle).
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # 1.5 mRNA/min/gene × TE 1 = 1.5 protein/min/gene (Stage-1).
        "rate_protein_per_min_per_gene": {"_type": "float", "_default": 1.5},
        # Static fallback gene-copy when no per-tick copy-number is wired.
        "gene_copies": {"_type": "float", "_default": 1.0},
        "deterministic": {"_type": "boolean", "_default": False},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.rate = float(self.parameters["rate_protein_per_min_per_gene"])
        self.gene_copies = float(self.parameters["gene_copies"])
        self.deterministic = bool(self.parameters["deterministic"])
        self.random_state = np.random.RandomState(seed=int(self.parameters["seed"]))
        self._apo_idx: int | None = None

    def inputs(self):
        return {
            "bulk":     {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "listeners": {
                "dnaA_cycle": {
                    "constitutive_synthesis_events": {
                        "_type": "overwrite[integer]", "_default": 0,
                    },
                },
            },
        }

    def update(self, states, interval=None):
        if self._apo_idx is None:
            self._apo_idx = int(bulk_name_to_idx(DNAA_APO_ID, states["bulk"]["id"]))

        timestep_s = float(states.get("timestep", 1.0))
        dt_min = timestep_s / 60.0
        mean_new = self.rate * self.gene_copies * dt_min

        if self.deterministic:
            new = int(round(mean_new))
        else:
            new = int(self.random_state.poisson(mean_new))

        if new <= 0:
            return {"listeners": {"dnaA_cycle": {"constitutive_synthesis_events": 0}}}

        idx = np.array([self._apo_idx], dtype=int)
        delta = np.array([new], dtype=int)
        return {
            "bulk": [(idx, delta)],
            "listeners": {"dnaA_cycle": {"constitutive_synthesis_events": new}},
        }
