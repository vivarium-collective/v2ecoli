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
except and records a per-fit ``"ok"``/``"error"`` status (output port
``mechanistic_fit_status``). By default a failed fit aborts the whole
step (PARCA_REVIEW A3) — a partially-fit pickle is byte-shaped exactly
like a complete one, so writing it silently is worse than crashing.
Pass ``allow_partial_fit=True`` in this Step's config (``v2ecoli-parca
--allow-partial-fit`` on the CLI) to opt into writing the pickle anyway,
with the status recorded so the gap is visible instead of silent.
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
    # Per-fit {"mechanistic_supply"|"mechanistic_export"|"mechanistic_uptake":
    # "ok"|"error"} status (PARCA_REVIEW A3) — lands in the composite state
    # (and the pickled parca_state.pkl) so a partial fit is a visible,
    # queryable fact instead of a print statement discarded by the CLI.
    'mechanistic_fit_status': 'overwrite',
}


class FinalAdjustmentsStep(Step):
    """Step 9 — final_adjustments.  See module docstring."""

    description = (
        "Step 9 — final_adjustments.  Fit the last kinetic constants the\n"
        "online model needs, then a final expression-consistency pass.\n\n"
        "  • set_mechanistic_supply_constants — solve aa kcat + synthase conc\n"
        "    so each amino acid's net flux matches translation demand per\n"
        "    condition\n"
        "  • set_mechanistic_uptake/export_constants — transporter / exporter\n"
        "    kcats by the same balance\n"
        "  • set_ppgpp_kinetics_parameters — fit RelA/SpoT synthase + hydrolase\n"
        "    rates so steady-state [ppGpp] tracks the growth-rate-dependent pool\n"
        "  • adjust_final_expression — cross-condition consistency check\n"
        "Each mechanistic fit is wrapped in try/except (a CYS[c] kcat corner\n"
        "case can fail in debug); a failure aborts the step unless\n"
        "allow_partial_fit=True, in which case a per-fit ok/error status is\n"
        "recorded instead of silently landing a partially-fit pickle."
    )

    config_schema = {
        'allow_partial_fit': {'_type': 'boolean', '_default': False},
    }

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
        # produces edge-case input distributions. Record per-fit ok/error
        # status either way (PARCA_REVIEW A3); by default a failure aborts
        # the whole step below rather than silently landing a pickle that
        # is byte-shaped exactly like a complete one. The failure is
        # identical to what the original fit_sim_data_1 raises under the
        # same conditions; debug it with --mode full or by patching the
        # underlying kinetics fit — or pass allow_partial_fit=True to opt
        # into the old "continue with a partial fit" behavior.
        allow_partial_fit = self.config.get('allow_partial_fit', False)
        fit_status: dict = {}
        fit_error_detail: dict = {}
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
                fit_status[label] = 'error'
                fit_error_detail[label] = f'{type(e).__name__}: {e}'
                print(f"  Step 9 WARNING: {label} failed ({type(e).__name__}: {e}); "
                      "continuing so the pipeline produces a comparable pickle."
                      if allow_partial_fit else
                      f"  Step 9 ERROR: {label} failed ({type(e).__name__}: {e})")
            else:
                fit_status[label] = 'ok'

        if fit_error_detail and not allow_partial_fit:
            raise RuntimeError(
                "Step 9 (final_adjustments): mechanistic fit(s) failed: "
                f"{fit_error_detail}. Refusing to write a partially-fit "
                "parca_state.pkl (PARCA_REVIEW A3). Pass --allow-partial-fit "
                "to v2ecoli-parca (or allow_partial_fit=True in this Step's "
                "config) to opt into writing it anyway with the per-fit "
                "status recorded at state['mechanistic_fit_status']."
            )

        # ppGpp kinetics.
        sd.process.transcription.set_ppgpp_kinetics_parameters(
            average_basal_container, sd.constants)

        print(f"  Step 9 (final_adjustments) completed in {time.time() - t0:.1f}s")
        return {
            'transcription': sd.process.transcription,
            'metabolism':    sd.process.metabolism,
            'constants':     sd.constants,
            'mechanistic_fit_status': fit_status,

            'tick_9': True,}
