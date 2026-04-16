"""Step 3 — basal_specs.  Fit the minimal-medium (basal) condition.

Iteratively fits RNA synthesis probabilities so that the steady-state RNA
distribution matches measured RNA-seq expression, then calibrates a few
derived kinetic parameters (endoRNase Michaelis constants, ppGpp-regulated
expression tables, the growth-associated ATP maintenance cost). Everything
in the pipeline downstream of this step assumes these basal fits hold.

Mathematical Model
------------------

Inputs:
- transcription.rna_expression (post–input_adjustments).
- rna_decay.endoRNase counts + KM bounds.
- mass + constants + growth_rate_parameters for the basal doubling time.
- condition_to_doubling_time[basal].
- cell_specs: mutated in place with a new 'basal' entry.

Parameters:
- cpus: parallel seed count for expressionConverge's Monte-Carlo legs.
- debug: short-circuits the outer ppGpp / Km / GAM fits for a quick path.

Calculation:
- expressionConverge (iterative): adjust rna_synth_prob until the implied
  mRNA, tRNA, and rRNA levels reproduce rna_expression within tolerance.
- setPpGppExpression: build ppGpp_aa / ppGpp_basal_expression tables for
  the downstream ppGpp-regulated translation supply.
- fitRnaDecayKm: solve for endoRNase KM values that reproduce the
  measured mRNA half-lives under the fitted synthesis probabilities.
- fitMaintenanceCosts: regress ATP consumption against dry mass to
  recover the growth-associated maintenance coefficient.

Outputs:
- transcription (mutated): rna_synth_prob, ppGpp expression tables.
- rna_decay (mutated): endoRNase KM values.
- mass, constants (mutated): GAM coefficient + derived totals.
- cell_specs['basal']: bulk_average, bulk_distribution, n_avg_copies,
  r_vector placeholder for step 6.
"""

import binascii
import functools
import os
import pickle
import time

import numpy as np
import scipy.optimize

from process_bigraph import Step

from v2ecoli.processes.parca.ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.processes.parca.wholecell.utils import units

from v2ecoli.processes.parca.fitting import VERBOSE, expressionConverge
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade


INPUT_PORTS = {
    'tick_2'                            : 'overwrite',
    # subsystem objects
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'rna_decay':                'sim_data.rna_decay',
    'complexation':             'sim_data.complexation',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    'molecule_groups':          'overwrite',
    'molecule_ids':             'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'sim_data_root':            'overwrite',
    # data leaves
    'condition_to_doubling_time': 'overwrite',
    'cell_specs':                 'overwrite',
}

OUTPUT_PORTS = {
    'tick_3'                            : 'overwrite',
    # subsystems mutated in place; return the objects
    'transcription':  'sim_data.transcription',
    'mass':           'sim_data.mass',
    'constants':      'sim_data.constants',
    'rna_decay':      'sim_data.rna_decay',
    # cell_specs with new ['basal'] entry
    'cell_specs':     'overwrite',
}


class BasalSpecsStep(Step):
    """Step 3 — build basal cell specifications.  See module docstring."""

    config_schema = {
        'variable_elongation_transcription':
            {'_type': 'boolean', '_default': True},
        'variable_elongation_translation':
            {'_type': 'boolean', '_default': False},
        'disable_ribosome_capacity_fitting':
            {'_type': 'boolean', '_default': False},
        'disable_rnapoly_capacity_fitting':
            {'_type': 'boolean', '_default': False},
        'cache_dir': 'string',
    }

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)

        # --- expressionConverge — iterative fit of expression + synth prob
        conc_dict = (
            sd.process.metabolism.concentration_updates
              .concentrations_based_on_nutrients(media_id="minimal")
        )
        expression = sd.process.transcription.rna_expression["basal"].copy()
        doubling_time = state['condition_to_doubling_time']["basal"]

        (
            expression, synth_prob, fit_cistron_expression,
            avg_cell_dry_mass_init, fit_avg_soluble_target_mol_mass,
            bulk_container, _,
        ) = expressionConverge(
            sd, expression, conc_dict, doubling_time, conditionKey="basal",
            variable_elongation_transcription=self.config.get(
                'variable_elongation_transcription', True),
            variable_elongation_translation=self.config.get(
                'variable_elongation_translation', False),
            disable_ribosome_capacity_fitting=self.config.get(
                'disable_ribosome_capacity_fitting', False),
            disable_rnapoly_capacity_fitting=self.config.get(
                'disable_rnapoly_capacity_fitting', False),
        )

        # --- apply mass updates (needed by downstream sub-functions)
        sd.mass.avg_cell_dry_mass_init = avg_cell_dry_mass_init
        sd.mass.avg_cell_dry_mass = (
            sd.mass.avg_cell_dry_mass_init
            * sd.mass.avg_cell_to_initial_cell_conversion_factor
        )
        sd.mass.avg_cell_water_mass_init = (
            sd.mass.avg_cell_dry_mass_init
            / sd.mass.cell_dry_mass_fraction
            * sd.mass.cell_water_mass_fraction
        )
        sd.mass.fitAvgSolubleTargetMolMass = fit_avg_soluble_target_mol_mass

        # --- apply expression updates (needed by set_ppgpp_expression)
        sd.process.transcription.rna_expression["basal"][:] = expression
        sd.process.transcription.rna_synth_prob["basal"][:] = synth_prob
        sd.process.transcription.fit_cistron_expression["basal"] = fit_cistron_expression

        # --- ppGpp regulation
        sd.process.transcription.set_ppgpp_expression(sd)

        # --- endoRNase Km fitting
        Km = setKmCooperativeEndoRNonLinearRNAdecay(
            sd, bulk_container, self.config.get('cache_dir', ''),
        )
        n_transcribed_rnas = len(sd.process.transcription.rna_data)
        sd.process.transcription.rna_data["Km_endoRNase"] = Km[:n_transcribed_rnas]
        sd.process.transcription.mature_rna_data["Km_endoRNase"] = Km[n_transcribed_rnas:]

        # --- maintenance costs
        fitMaintenanceCosts(sd, bulk_container)

        # --- record cell_specs["basal"]
        cell_specs = dict(state['cell_specs'])
        cell_specs["basal"] = {
            "concDict":                  conc_dict,
            "expression":                expression,
            "synthProb":                 synth_prob,
            "fit_cistron_expression":    fit_cistron_expression,
            "doubling_time":             doubling_time,
            "avgCellDryMassInit":        avg_cell_dry_mass_init,
            "fitAvgSolubleTargetMolMass": fit_avg_soluble_target_mol_mass,
            "bulkContainer":             bulk_container,
        }

        print(f"  Step 3 (basal_specs) completed in {time.time() - t0:.1f}s")
        return {
            'transcription': sd.process.transcription,
            'mass':          sd.mass,
            'constants':     sd.constants,
            'rna_decay':     sd.process.rna_decay,
            'cell_specs':    cell_specs,
        
            'tick_3': True,}


# ============================================================================
# Sub-functions (unchanged logic from fit_sim_data_1.py, but operating on
# the sim_data facade assembled above).
# ============================================================================

def _crc32(*arrays: np.ndarray, initial: int = 0) -> int:
    def crc_next(accum, arr):
        shape = str(arr.shape).encode()
        return binascii.crc32(arr.tobytes(), binascii.crc32(shape, accum))
    return functools.reduce(crc_next, arrays, initial)


def setKmCooperativeEndoRNonLinearRNAdecay(sim_data, bulkContainer, cache_dir):
    """Fit Michaelis-Menten constants for RNAs binding to endoRNases."""

    def arrays_differ(a, b):
        return a.shape != b.shape or not np.allclose(a, b, equal_nan=True)

    cellDensity = sim_data.constants.cell_density
    cellVolume = (
        sim_data.mass.avg_cell_dry_mass_init / cellDensity
        / sim_data.mass.cell_dry_mass_fraction
    )
    countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

    degradable_rna_ids = np.concatenate((
        sim_data.process.transcription.rna_data["id"],
        sim_data.process.transcription.mature_rna_data["id"],
    ))
    degradation_rates = (1 / units.s) * np.concatenate((
        sim_data.process.transcription.rna_data["deg_rate"].asNumber(1 / units.s),
        sim_data.process.transcription.mature_rna_data["deg_rate"].asNumber(1 / units.s),
    ))
    endoRNase_idx = bulk_name_to_idx(
        sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"])
    endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
    kcatEndoRNase = sim_data.process.rna_decay.kcats
    totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

    endoRnaseRnaIds = sim_data.molecule_groups.endoRNase_rnas
    isEndoRnase = np.array([(x in endoRnaseRnaIds) for x in degradable_rna_ids])

    degradable_rna_idx = bulk_name_to_idx(degradable_rna_ids, bulkContainer["id"])
    rna_counts = counts(bulkContainer, degradable_rna_idx)
    rna_conc = countsToMolar * rna_counts
    Km_counts = (
        (1 / degradation_rates * totalEndoRnaseCapacity) - rna_conc
    ).asNumber()
    sim_data.process.rna_decay.Km_first_order_decay = Km_counts

    Alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10] if (
        sim_data.constants.sensitivity_analysis_alpha) else []

    total_endo_rnase_capacity_mol_l_s = totalEndoRnaseCapacity.asNumber(
        units.mol / units.L / units.s)
    rna_conc_mol_l = rna_conc.asNumber(units.mol / units.L)
    degradation_rates_s = degradation_rates.asNumber(1 / units.s)

    for alpha in Alphas:
        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            total_endo_rnase_capacity_mol_l_s, rna_conc_mol_l,
            degradation_rates_s, isEndoRnase, alpha)
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(Km_counts), jac=loss_jac).x)
        sim_data.process.rna_decay.sensitivity_analysis_alpha_residual[alpha] = np.sum(
            np.abs(res_aux(Km_cooperative_model)))

    alpha = 0.5
    kcatEndo = [0.0001, 0.001, 0.01, 0.1, 1, 10] if (
        sim_data.constants.sensitivity_analysis_kcat_endo) else []

    for kcat in kcatEndo:
        totalEndoRNcap = units.sum(endoRNaseConc * kcat)
        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            totalEndoRNcap.asNumber(units.mol / units.L), rna_conc_mol_l,
            degradation_rates_s, isEndoRnase, alpha)
        km_counts_ini = (
            (totalEndoRNcap / degradation_rates.asNumber()) - rna_conc
        ).asNumber()
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(km_counts_ini), jac=loss_jac).x)
        sim_data.process.rna_decay.sensitivity_analysis_kcat[kcat] = Km_cooperative_model
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_ini[kcat] = np.sum(
            np.abs(res_aux(km_counts_ini)))
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_opt[kcat] = np.sum(
            np.abs(res_aux(Km_cooperative_model)))

    loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
        total_endo_rnase_capacity_mol_l_s, rna_conc_mol_l,
        degradation_rates_s, isEndoRnase, alpha)

    needToUpdate = ""
    checksum = _crc32(Km_counts, isEndoRnase, np.array(alpha))
    km_filepath = os.path.join(cache_dir, f"parca-km-{checksum}.cPickle") if cache_dir else ""

    if km_filepath and os.path.exists(km_filepath):
        with open(km_filepath, "rb") as f:
            Km_cache = pickle.load(f)
        Km_cooperative_model = Km_cache["Km_cooperative_model"]
        if (
            Km_counts.shape != Km_cooperative_model.shape
            or np.sum(np.abs(res_aux(Km_cooperative_model))) > 1e-15
            or arrays_differ(
                Km_cache["total_endo_rnase_capacity_mol_l_s"],
                total_endo_rnase_capacity_mol_l_s)
            or arrays_differ(Km_cache["rna_conc_mol_l"], rna_conc_mol_l)
            or arrays_differ(Km_cache["degradation_rates_s"], degradation_rates_s)
        ):
            needToUpdate = "recompute"
    else:
        needToUpdate = "compute"

    if needToUpdate:
        if VERBOSE:
            print(f"Running non-linear optimization to {needToUpdate} {km_filepath}")
        sol = scipy.optimize.minimize(
            loss, np.log(Km_counts), jac=loss_jac, tol=1e-8)
        Km_cooperative_model = np.exp(sol.x)
        if km_filepath:
            Km_cache = dict(
                Km_cooperative_model=Km_cooperative_model,
                total_endo_rnase_capacity_mol_l_s=total_endo_rnase_capacity_mol_l_s,
                rna_conc_mol_l=rna_conc_mol_l,
                degradation_rates_s=degradation_rates_s,
            )
            with open(km_filepath, "wb") as f:
                pickle.dump(Km_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    log_Km_cooperative_model = np.log(Km_cooperative_model)
    log_Km_counts = np.log(Km_counts)

    sim_data.process.rna_decay.stats_fit.update({
        "LossKm":         np.sum(np.abs(loss(log_Km_counts))),
        "LossKmOpt":      np.sum(np.abs(loss(log_Km_cooperative_model))),
        "ResKm":          np.sum(np.abs(res(Km_counts))),
        "ResKmOpt":       np.sum(np.abs(res(Km_cooperative_model))),
        "ResEndoRNKm":    np.sum(np.abs(isEndoRnase * res(Km_counts))),
        "ResEndoRNKmOpt": np.sum(np.abs(isEndoRnase * res(Km_cooperative_model))),
        "ResScaledKm":    np.sum(np.abs(res_aux(Km_counts))),
        "ResScaledKmOpt": np.sum(np.abs(res_aux(Km_cooperative_model))),
    })

    return units.mol / units.L * Km_cooperative_model


def fitMaintenanceCosts(sim_data, bulkContainer):
    """Fit growth-associated maintenance (GAM) cost; mutates constants.darkATP."""
    aaCounts = sim_data.process.translation.monomer_data["aa_counts"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"])
    proteinCounts = counts(bulkContainer, protein_idx)
    nAvogadro = sim_data.constants.n_avogadro
    avgCellDryMassInit = sim_data.mass.avg_cell_dry_mass_init
    gtpPerTranslation = sim_data.constants.gtp_per_translation
    atp_per_charge = 2

    aaMmolPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0,
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    aasUsedOverCellCycle = units.sum(aaMmolPerGDCW)
    explicit = (atp_per_charge + gtpPerTranslation) * aasUsedOverCellCycle

    darkATP = sim_data.constants.growth_associated_maintenance - explicit

    if darkATP.asNumber() < 0:
        raise ValueError(
            "GAM has been adjusted too low. Explicit energy accounting should not "
            "exceed GAM. Consider setting darkATP to 0 if energy corrections are accurate."
        )
    sim_data.constants.darkATP = darkATP
