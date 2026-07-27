"""
ppGpp-dependent transcript initiation regulation.

Optional module that computes ppGpp-dependent basal synthesis probabilities
and RNAP active fraction. Writes to a shared ``ppgpp_state`` store that
TranscriptInitiation reads when present.

When this module is not in the composite, TranscriptInitiation falls back
to media-dependent defaults (environment-dependent rescaling).
"""

import numpy as np
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from bigraph_schema.contract import ProcessContract

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

    contract = ProcessContract(
        summary=(
            "Optional regulator: computes ppGpp-dependent per-TU basal synthesis "
            "probabilities and the RNAP active fraction from the current ppGpp "
            "concentration, and publishes them to the ppgpp_state store that "
            "TranscriptInitiation reads."
        ),
        math=[
            "ppgpp_conc = ppGpp_count / (n_avogadro · cell_volume);  cell_volume = cell_mass / cell_density",
            "basal_prob = synth_prob(ppgpp_conc, copy_number)",
            "f_bound = ppgpp_conc² / (km_sq + ppgpp_conc²)",
            "frac_active_rnap = f_bound · frac_active_rnap_bound + (1 − f_bound) · frac_active_rnap_free",
        ],
        inputs={
            'bulk': "reads the ppGpp molecule count and converts it to a concentration ppgpp_conc.",
            'listeners': "reads listeners.mass.cell_mass to form the cell volume and the count→molar factor.",
        },
        outputs={
            'ppgpp_state': (
                "writes basal_prob (per-TU ppGpp-dependent synthesis probabilities) "
                "and frac_active_rnap for TranscriptInitiation to consume; when absent "
                "TranscriptInitiation falls back to media defaults."
            ),
        },
        config={
            'ppgpp': "bulk id of ppGpp whose count sets the regulatory concentration.",
            'synth_prob': "fitted callable mapping (ppgpp_conc, copy_number) to per-TU basal synthesis probabilities.",
            'copy_number': "expected gene copy number at the reference doubling time, passed to synth_prob.",
            'fraction_active_rnap_bound': "active RNAP fraction in the ppGpp-bound limit (frac_active_rnap_bound).",
            'fraction_active_rnap_free': "active RNAP fraction in the ppGpp-free limit (frac_active_rnap_free).",
            'ppgpp_km_squared': "squared half-saturation ppGpp concentration km_sq for the RNAP-binding Hill term.",
            'get_rnap_active_fraction_from_ppGpp': "legacy opaque callable used for the active fraction when the Hill parameters are not supplied.",
            'cell_density': "cell density used to convert cell_mass to volume.",
            'n_avogadro': "Avogadro constant for the count→molar conversion.",
            'trna_attenuation': "whether fitted attenuation adjustments are added to basal_prob for attenuated TUs.",
            'attenuated_rna_indices': "TU indices receiving the attenuation adjustment when trna_attenuation is on.",
            'attenuation_adjustments': "additive basal_prob adjustments applied at the attenuated TU indices.",
        },
        symbols={
            'ppgpp_conc': "ppGpp concentration derived from its bulk count (converted to µmol/L for the Hill term)",
            'basal_prob': "per-TU ppGpp-dependent baseline synthesis probability (dimensionless)",
            'copy_number': "expected gene copy number at the reference doubling time (dimensionless)",
            'f_bound': "fraction of RNAP in the ppGpp-bound state (dimensionless, 0–1)",
            'km_sq': "squared half-saturation ppGpp concentration for RNAP binding ((µmol/L)²)",
            'frac_active_rnap': "resulting active RNAP fraction published to ppgpp_state (dimensionless, 0–1)",
            'frac_active_rnap_bound': "active RNAP fraction in the fully ppGpp-bound limit (dimensionless)",
            'frac_active_rnap_free': "active RNAP fraction in the ppGpp-free limit (dimensionless)",
            'cell_mass': "cell mass used to form the volume (fg)",
            'cell_density': "cell density used in the volume conversion (g/L)",
        },
        assumptions=[
            "ppGpp concentration is computed from its bulk count and the cell volume (cell_mass / cell_density).",
            "The RNAP active fraction follows a Hill relation in ppGpp (exponent 2) interpolating between the free and bound limiting fractions; a legacy callable is used when Hill parameters are absent.",
            "When tRNA attenuation is enabled, fixed adjustments are added to the basal probabilities of the attenuated TUs.",
        ],
        references=[],
    )

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
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
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

        cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        cell_volume = cell_mass / self.cell_density
        counts_to_molar = 1 / (self.n_avogadro * cell_volume)
        ppgpp_conc = counts(states["bulk"], self.ppgpp_idx) * counts_to_molar

        # synth_prob and get_rnap_active_fraction_from_ppGpp are the
        # function-registry wrappers, which accept pint directly.
        basal_prob, _ = self.synth_prob(ppgpp_conc, self.copy_number)

        if self.trna_attenuation:
            basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        # f_bound = ppgpp^2 / (km_sq + ppgpp^2)
        # frac_active = f_bound * frac_bound + (1 - f_bound) * frac_free
        if self.get_rnap_active_fraction_from_ppGpp is not None:
            frac_active_rnap = float(
                self.get_rnap_active_fraction_from_ppGpp(ppgpp_conc))
        else:
            ppgpp_val = ppgpp_conc.to(units.umol / units.L).magnitude
            f_bound = ppgpp_val ** 2 / (self._km_sq + ppgpp_val ** 2)
            frac_active_rnap = float(
                f_bound * self._frac_bound + (1 - f_bound) * self._frac_free)

        return {
            'ppgpp_state': {
                'basal_prob': basal_prob,
                'frac_active_rnap': frac_active_rnap,
            },
        }
