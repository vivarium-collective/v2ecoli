"""
tRNA attenuation configuration module.

Optional module that provides tRNA attenuation parameters to
TranscriptElongation. When present, TranscriptElongation uses charged
tRNA concentrations to probabilistically terminate transcription of
attenuated genes.

This module writes attenuation parameters (callables and indices) to a
shared ``attenuation_config`` store. TranscriptElongation reads the store
and computes the actual attenuation mask each timestep using current
RNA state.

When this module is not in the composite, TranscriptElongation skips
attenuation entirely (all transcripts elongate).
"""

from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "trna-attenuation-config"
TOPOLOGY = {
    "attenuation_config": ("attenuation_config",),
}


class TrnaAttenuationConfig(Step):
    """Write tRNA attenuation parameters to shared store.

    This is a one-shot step that populates the attenuation_config store
    on first update. TranscriptElongation reads the config each timestep
    to compute stop probabilities from current charged tRNA levels.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'get_attenuation_stop_probabilities': 'method',
        'attenuated_rna_indices': 'array[integer]',
        'location_lookup': 'map[integer]',
        'cell_density': 'float',
        'n_avogadro': 'float',
        'charged_trnas': 'list[string]',
    }

    def initialize(self, config):
        self.stop_probabilities = self.parameters["get_attenuation_stop_probabilities"]
        self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
        self.location_lookup = self.parameters["location_lookup"]
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.charged_trnas = self.parameters["charged_trnas"]
        self._written = False

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'attenuation_config': {
                'stop_probabilities': 'method',
                'attenuated_rna_indices': 'array[integer]',
                'location_lookup': 'map[integer]',
                'cell_density': 'float',
                'n_avogadro': 'float',
                'charged_trnas': 'list[string]',
                'enabled': 'boolean',
            },
        }

    def update(self, states, interval=None):
        if self._written:
            return {}
        self._written = True
        return {
            'attenuation_config': {
                'stop_probabilities': self.stop_probabilities,
                'attenuated_rna_indices': self.attenuated_rna_indices,
                'location_lookup': self.location_lookup,
                'cell_density': self.cell_density,
                'n_avogadro': self.n_avogadro,
                'charged_trnas': self.charged_trnas,
                'enabled': True,
            },
        }
