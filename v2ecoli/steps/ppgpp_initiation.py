"""
ppGpp-dependent transcript initiation regulation.

Optional module that computes ppGpp-dependent basal synthesis probabilities
and RNAP active fraction. Writes to a shared ``ppgpp_state`` store that
TranscriptInitiation reads when present.

When this module is not in the composite, TranscriptInitiation falls back
to media-dependent defaults (environment-dependent rescaling).
"""

import numpy as np
from wholecell.utils import units

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts

NAME = "ppgpp-initiation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "ppgpp_state": ("ppgpp_state",),
}


class PpgppInitiation(Step):
    """Compute ppGpp-dependent basal_prob and fracActiveRnap.

    Reads ppGpp concentration from bulk, computes synthesis probabilities
    using the synth_prob callable, and writes results to ppgpp_state store
    for TranscriptInitiation to consume.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'ppgpp': 'string',
        'synth_prob': 'method',
        'copy_number': 'integer{1}',
        'n_avogadro': 'float',
        'cell_density': 'float',
        # Legacy opaque callable (still supported if Hill-fn params not provided)
        'get_rnap_active_fraction_from_ppGpp': {'_type': 'method', '_default': None},
        # Hill-function parameters for RNAP active fraction vs ppGpp:
        #   f_bound = ppgpp^2 / (km_sq + ppgpp^2)
        #   frac_active = f_bound * frac_active_bound + (1 - f_bound) * frac_active_free
        'fraction_active_rnap_bound': {'_type': 'float', '_default': None},
        'fraction_active_rnap_free': {'_type': 'float', '_default': None},
        'ppgpp_km_squared': {'_type': 'float', '_default': None},
        'trna_attenuation': {'_type': 'boolean', '_default': False},
        'attenuated_rna_indices': {'_type': 'array[integer]', '_default': []},
        'attenuation_adjustments': {'_type': 'array[float]', '_default': []},
    }

    def initialize(self, config):
        self.ppgpp = self.parameters["ppgpp"]
        self.synth_prob = self.parameters["synth_prob"]
        self.copy_number = self.parameters["copy_number"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Prefer inline Hill function if its parameters are supplied;
        # fall back to the legacy callable otherwise.
        self._frac_bound = self.parameters.get("fraction_active_rnap_bound")
        self._frac_free = self.parameters.get("fraction_active_rnap_free")
        self._km_sq = self.parameters.get("ppgpp_km_squared")
        if None in (self._frac_bound, self._frac_free, self._km_sq):
            self.get_rnap_active_fraction_from_ppGpp = self.parameters[
                "get_rnap_active_fraction_from_ppGpp"
            ]
        else:
            self.get_rnap_active_fraction_from_ppGpp = None

        self.trna_attenuation = self.parameters.get("trna_attenuation", False)
        if self.trna_attenuation:
            self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
            self.attenuation_adjustments = self.parameters["attenuation_adjustments"]
        self.ppgpp_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'float[fg]', '_default': 0.0},
                },
            },
        }

    def outputs(self):
        return {
            'ppgpp_state': {
                'basal_prob': 'overwrite[array[float]]',
                'frac_active_rnap': 'overwrite[float]',
            },
        }

    def update(self, states, interval=None):
        if self.ppgpp_idx is None:
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, states["bulk"]["id"])

        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        cell_volume = cell_mass / self.cell_density
        counts_to_molar = 1 / (self.n_avogadro * cell_volume)
        ppgpp_conc = counts(states["bulk"], self.ppgpp_idx) * counts_to_molar

        basal_prob, _ = self.synth_prob(ppgpp_conc, self.copy_number)

        if self.trna_attenuation:
            basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        # f_bound = ppgpp^2 / (km_sq + ppgpp^2)
        # frac_active = f_bound * frac_bound + (1 - f_bound) * frac_free
        if self.get_rnap_active_fraction_from_ppGpp is not None:
            frac_active_rnap = float(
                self.get_rnap_active_fraction_from_ppGpp(ppgpp_conc))
        else:
            try:
                ppgpp_val = ppgpp_conc.asNumber(units.umol / units.L)
            except AttributeError:
                ppgpp_val = float(ppgpp_conc)
            f_bound = ppgpp_val ** 2 / (self._km_sq + ppgpp_val ** 2)
            frac_active_rnap = float(
                f_bound * self._frac_bound + (1 - f_bound) * self._frac_free)

        return {
            'ppgpp_state': {
                'basal_prob': basal_prob,
                'frac_active_rnap': frac_active_rnap,
            },
        }
