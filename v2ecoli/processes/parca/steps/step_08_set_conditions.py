"""Step 8 — set_conditions.  Build per-condition mass + expression tables.

Turns the per-condition cell_specs into the flat lookup tables the
simulation consumes each timestep: average cell dry mass, soluble-pool
mass budget, expected dry-mass increase, and per-nutrient transcription
/ translation overrides.

Mathematical Model
------------------

Inputs:
- cell_specs[<condition>] for every fitted condition.
- transcription, translation, metabolism (for getter functions).
- mass, constants, growth_rate_parameters.
- conditions, condition_to_doubling_time.

Calculation:
- Rescale avgCellDryMassInit per condition so that expected growth over
  the doubling time matches observed dry_mass.
- Compute fitAvgSolublePoolMass by subtracting macromolecule mass from
  avgCellDryMassInit.
- bulkContainer: canonical bulk-molecule counts at t=0 for each
  condition, sampled from the fitted distributions.
- expected_dry_mass_increase_dict[nutrient]:
    = avgCellDryMass · (2^(t/τ) - 1) / t    integrated over one cycle.
- Per-nutrient dicts on transcription and translation: which RNAs and
  which tRNAs the online model should emphasize under that medium.

Outputs:
- transcription (mutated): per-nutrient expression overrides.
- translation (mutated): per-nutrient tRNA supply.
- cell_specs (mutated): avgCellDryMassInit, fitAvgSolublePoolMass,
  bulkContainer for every condition.
- expected_dry_mass_increase_dict.
"""

import time

import numpy as np

from process_bigraph import Step

from v2ecoli.processes.parca.fitting import rescale_mass_for_soluble_metabolites
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade
from v2ecoli.processes.parca.wholecell.utils import units


# ============================================================================
# Pure sub-function
# ============================================================================

def compute_synth_prob_fractions(rna_synth_prob, is_mRNA, is_tRNA, is_rRNA):
    """Sum-by-class synthesis probabilities."""
    return {
        "mRna": float(rna_synth_prob[is_mRNA].sum()),
        "tRna": float(rna_synth_prob[is_tRNA].sum()),
        "rRna": float(rna_synth_prob[is_rRNA].sum()),
    }


# ============================================================================
# Step
# ============================================================================

INPUT_PORTS = {
    'tick_7'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    'getter':                   'overwrite',
    'sim_data_root':            'overwrite',
    'conditions':               'overwrite',
    'condition_to_doubling_time': 'overwrite',
    'cell_specs':               'overwrite',
}

OUTPUT_PORTS = {
    'tick_8'                            : 'overwrite',
    'transcription':                  'sim_data.transcription',
    'translation':                    'sim_data.translation',
    'cell_specs':                     'overwrite',
    'expected_dry_mass_increase_dict': 'overwrite',
}


class SetConditionsStep(Step):
    """Step 8 — set_conditions.  See module docstring."""

    config_schema = {'verbose': {'_type': 'integer', '_default': 1}}

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = dict(state['cell_specs'])
        verbose = self.config.get('verbose', 1)

        rna_data = sd.process.transcription.rna_data
        is_mRNA = rna_data["is_mRNA"]
        is_tRNA = rna_data["is_tRNA"]
        is_rRNA = rna_data["is_rRNA"]
        includes_rprotein = rna_data["includes_ribosomal_protein"]
        includes_RNAP     = rna_data["includes_RNAP"]

        rnaSynthProbFraction            = {}
        rnapFractionActiveDict          = {}
        rnaSynthProbRProtein            = {}
        rnaSynthProbRnaPolymerase       = {}
        rnaPolymeraseElongationRateDict = {}
        expectedDryMassIncreaseDict     = {}
        ribosomeElongationRateDict      = {}
        ribosomeFractionActiveDict      = {}

        molar_units = units.mol / units.L

        for condition_label in sorted(cell_specs):
            condition = sd.conditions[condition_label]
            nutrients = condition["nutrients"]
            doubling_time = sd.condition_to_doubling_time[condition_label]
            spec = cell_specs[condition_label]

            conc_dict = (
                sd.process.metabolism.concentration_updates
                  .concentrations_based_on_nutrients(media_id=nutrients)
            )
            conc_dict.update(sd.mass.getBiomassAsConcentrations(doubling_time))

            target_ids = sorted(conc_dict)
            target_concs = molar_units * np.array(
                [conc_dict[k].asNumber(molar_units) for k in target_ids]
            )
            mw = sd.getter.get_masses(target_ids)

            fracs = sd.mass.get_component_masses(doubling_time)
            non_small = (
                fracs["proteinMass"] + fracs["rnaMass"] + fracs["dnaMass"]
            ) / sd.mass.avg_cell_to_initial_cell_conversion_factor

            grp = sd.growth_rate_parameters
            fraction_active_rnap     = grp.get_fraction_active_rnap(doubling_time)
            rnap_elongation_rate     = grp.get_rnap_elongation_rate(doubling_time)
            ribosome_elongation_rate = grp.get_ribosome_elongation_rate(doubling_time)
            fraction_active_ribo     = grp.get_fraction_active_ribosome(doubling_time)

            bulk_container = spec["bulkContainer"].copy()
            rna_synth_prob = sd.process.transcription.rna_synth_prob[condition_label].copy()

            if verbose > 0:
                print(f"Updating mass in condition {condition_label}")

            avg_cell_dry_mass_init, fit_avg_soluble_pool_mass = (
                rescale_mass_for_soluble_metabolites(
                    bulk_container,
                    target_ids,
                    target_concs,
                    mw,
                    non_small,
                    sd.mass.avg_cell_to_initial_cell_conversion_factor,
                    sd.constants.cell_density,
                    sd.constants.n_avogadro,
                )
            )

            spec["avgCellDryMassInit"]     = avg_cell_dry_mass_init
            spec["fitAvgSolublePoolMass"]  = fit_avg_soluble_pool_mass
            spec["bulkContainer"]          = bulk_container

            if condition["perturbations"]:
                continue
            # Populate per-nutrient dicts — first occurrence wins.
            rnaSynthProbFraction.setdefault(
                nutrients,
                compute_synth_prob_fractions(rna_synth_prob, is_mRNA, is_tRNA, is_rRNA))
            rnaSynthProbRProtein.setdefault(
                nutrients, rna_synth_prob[includes_rprotein])
            rnaSynthProbRnaPolymerase.setdefault(
                nutrients, rna_synth_prob[includes_RNAP])
            rnapFractionActiveDict.setdefault(nutrients, fraction_active_rnap)
            rnaPolymeraseElongationRateDict.setdefault(nutrients, rnap_elongation_rate)
            expectedDryMassIncreaseDict.setdefault(nutrients, avg_cell_dry_mass_init)
            ribosomeElongationRateDict.setdefault(nutrients, ribosome_elongation_rate)
            ribosomeFractionActiveDict.setdefault(nutrients, fraction_active_ribo)

        # Install per-nutrient dicts onto the subsystem objects.
        sd.process.transcription.rnaSynthProbFraction            = rnaSynthProbFraction
        sd.process.transcription.rnapFractionActiveDict          = rnapFractionActiveDict
        sd.process.transcription.rnaSynthProbRProtein            = rnaSynthProbRProtein
        sd.process.transcription.rnaSynthProbRnaPolymerase       = rnaSynthProbRnaPolymerase
        sd.process.transcription.rnaPolymeraseElongationRateDict = rnaPolymeraseElongationRateDict
        sd.process.translation.ribosomeElongationRateDict        = ribosomeElongationRateDict
        sd.process.translation.ribosomeFractionActiveDict        = ribosomeFractionActiveDict

        print(f"  Step 8 (set_conditions) completed in {time.time() - t0:.1f}s")
        return {
            'transcription':                   sd.process.transcription,
            'translation':                     sd.process.translation,
            'cell_specs':                      cell_specs,
            'expected_dry_mass_increase_dict': expectedDryMassIncreaseDict,
        
            'tick_8': True,}
