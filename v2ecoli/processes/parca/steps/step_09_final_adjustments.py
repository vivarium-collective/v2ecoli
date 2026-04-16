"""Step 9 — final_adjustments.  Kinetic constants the online model needs.

Fits the last handful of kinetic parameters that don't fit earlier
steps' patterns: amino-acid export/uptake kcats per nutrient,
mechanistic translation supply constants, and ppGpp
synthesis/degradation rates. Also applies any final cross-condition
expression consistency passes.

Mathematical Model
------------------

Inputs:
- All nine subsystems (transcription, translation, metabolism,
  complexation, equilibrium, two_component_system,
  transcription_regulation, replication, plus mass, constants,
  growth_rate_parameters, molecule_ids, molecule_groups, relation,
  bulk_molecules).
- conditions, condition_to_doubling_time, tf_to_fold_change, cell_specs.

Calculation:
- set_mechanistic_supply_constants: solve for amino-acid kcat +
  synthase concentrations so each amino acid's net flux matches the
  translation demand under each condition.
- set_mechanistic_uptake_constants: same for transporter kcats.
- set_mechanistic_export_constants: same for exporter kcats.
- set_ppgpp_kinetics_parameters: fit ppGpp synthase (RelA/SpoT) +
  hydrolase rate constants so steady-state [ppGpp] reproduces the
  measured growth-rate-dependent pool.
- adjust_final_expression: last-pass cross-condition expression
  consistency check.

Outputs:
- transcription (mutated): final expression tables.
- metabolism (mutated): aa_kcats_fwd, aa_kcats_rev,
  aa_enzyme_ids, ppgpp_kinetics.
- constants (mutated): ppGpp synthesis/hydrolysis rates.

Note: ``set_mechanistic_supply_constants`` can hit
``ValueError: Could not find positive forward and reverse kcat for
CYS[c]`` in debug mode — the same numerical corner-case present in the
upstream vEcoli ParCa. The step wraps each mechanistic fit in try /
except so the pickle still lands even if one fails.
"""

import time

from process_bigraph import Step

from v2ecoli.processes.parca.ecoli.library.initial_conditions import create_bulk_container
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade


INPUT_PORTS = {
    'tick_8'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'complexation':             'sim_data.complexation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    'molecule_ids':             'overwrite',
    'molecule_groups':          'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'sim_data_root':            'overwrite',
    'conditions':               'overwrite',
    'condition_to_doubling_time': 'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
    'tf_to_fold_change':        'overwrite',
    'tf_to_direction':          'overwrite',
    'condition_active_tfs':     'overwrite',
    'condition_inactive_tfs':   'overwrite',
    'cell_specs':               'overwrite',
    # set_mechanistic_supply_constants reads sim_data.translation_supply_rate
    # which step 5 populates per-nutrient.
    'translation_supply_rate':  'overwrite',
    # calculate_attenuation reads sim_data.pPromoterBound (set by step 6)
    'pPromoterBound':           'overwrite',
    # create_bulk_container uses external_state.exchange_data_from_media
    # and mutates sim_data.condition temporarily.
    'external_state':           'overwrite',
    'condition':                'overwrite',
}

OUTPUT_PORTS = {
    'tick_9'                            : 'overwrite',
    'transcription': 'sim_data.transcription',
    'metabolism':    'sim_data.metabolism',
    'constants':     'sim_data.constants',
}


class FinalAdjustmentsStep(Step):
    """Step 9 — final_adjustments.  See module docstring."""

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = state['cell_specs']

        # Attenuation + ppGpp expression fixups.
        sd.process.transcription.calculate_attenuation(sd, cell_specs)
        sd.process.transcription.adjust_polymerizing_ppgpp_expression(sd)
        sd.process.transcription.adjust_ppgpp_expression_for_tfs(sd)

        # Amino-acid supply constants — based on average bulk containers.
        average_basal_container   = create_bulk_container(sd, n_seeds=5)
        average_with_aa_container = create_bulk_container(
            sd, condition="with_aa", n_seeds=5)

        sd.process.metabolism.set_phenomological_supply_constants(sd)
        # The three mechanistic_* fits can raise on numerically-marginal
        # kinetics (e.g. "Could not find positive forward and reverse
        # kcat for CYS[c]") in debug mode where the truncated TF set
        # produces edge-case input distributions.  We log and continue
        # so the pipeline still produces a pickle of everything else
        # (including all step-1..8 outputs).  The failure is identical
        # to what the original fit_sim_data_1 raises under the same
        # conditions; debug it with --mode full or by patching the
        # underlying kinetics fit.
        for label, call in [
            ('mechanistic_supply', lambda: sd.process.metabolism
                .set_mechanistic_supply_constants(
                    sd, cell_specs,
                    average_basal_container, average_with_aa_container)),
            ('mechanistic_export', lambda: sd.process.metabolism
                .set_mechanistic_export_constants(
                    sd, cell_specs, average_basal_container)),
            ('mechanistic_uptake', lambda: sd.process.metabolism
                .set_mechanistic_uptake_constants(
                    sd, cell_specs, average_with_aa_container)),
        ]:
            try:
                call()
            except Exception as e:
                print(f"  Step 9 WARNING: {label} failed ({type(e).__name__}: {e}); "
                      "continuing so the pipeline produces a comparable pickle.")

        # ppGpp kinetics.
        sd.process.transcription.set_ppgpp_kinetics_parameters(
            average_basal_container, sd.constants)

        print(f"  Step 9 (final_adjustments) completed in {time.time() - t0:.1f}s")
        return {
            'transcription': sd.process.transcription,
            'metabolism':    sd.process.metabolism,
            'constants':     sd.constants,
        
            'tick_9': True,}
