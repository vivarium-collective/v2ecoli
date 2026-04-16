"""Step 5 — fit_condition.  Bulk-molecule distributions + translation supply.

The pipeline's heavy step (~60–70 min at cpus=2). For every condition,
sample N_SEEDS synthetic cells, propagate them through the
complexation / equilibrium / two-component-system solvers to steady
state, and record the mean and standard deviation of every bulk
molecule count. Also computes the per-amino-acid translation supply
rate each condition requires to hit its observed doubling time.

Mathematical Model
------------------

Inputs:
- transcription, translation, complexation, equilibrium,
  two_component_system subsystems (post–step 4).
- mass, constants, molecule_groups, molecule_ids, relation,
  bulk_molecules, growth_rate_parameters.
- cell_specs[<condition>] entries produced by step 4.

Parameters:
- N_SEEDS (module constant): stochastic sample count per condition.
- cpus: conditions distribute across a multiprocessing pool.

Calculation (per condition):
- Sample initial RNA and protein counts from the fitted expression
  distributions conditioned on mass.
- StochasticSystem (arrow) integrates complexation reactions to
  equilibrium.
- Equilibrium + two_component_system solvers produce macromolecule /
  ligand-complex concentrations.
- Aggregate mean_counts, std_counts, n_avg_copies across seeds.
- calculateTranslationSupply: from mean counts + kcats compute the
  amino-acid supply rate that matches the target dry-mass growth.

Outputs:
- cell_specs[<condition>] (mutated): bulk_average, bulk_distribution,
  n_avg_copies filled in.
- translation_supply_rate: {condition: array over 21 amino acids}.

Note: the complexation, equilibrium, and two-component-system solvers
are still sim_data-shaped objects (live references), not pure data — a
pending refactor.
"""

from stochastic_arrow import StochasticSystem
import numpy as np

from v2ecoli.processes.parca.ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.processes.parca.wholecell.utils import units
from v2ecoli.processes.parca.wholecell.utils.fitting import normalize

from v2ecoli.processes.parca.fitting import (
    apply_updates,
    netLossRateFromDilutionAndDegradationProtein,
    proteinDistributionFrommRNA,
    totalCountFromMassesAndRatios,
    totalCountIdDistributionRNA,
    totalCountIdDistributionProtein,
)
import time
from process_bigraph import Step

from v2ecoli.processes.parca.steps._facade import make_sim_data_facade

# Constants (mirrored from fit_sim_data_1.py)
# Per-condition seed count for calculateBulkDistributions.  Default 10
# matches vivarium-ecoli; set V2PARCA_N_SEEDS=3 (or any small int) to
# trade fitting variance for ~3x speed in step 5.
import os as _os
N_SEEDS = int(_os.environ.get('V2PARCA_N_SEEDS', '10'))
VERBOSE = 1


# ============================================================================
# Sub-functions (use sim_data read-only)
# ============================================================================


def calculateBulkDistributions(
    sim_data, expression, concDict, avgCellDryMassInit, doubling_time
):
    """
    Find distributions of copy numbers for macromolecules by instantiating
    N_SEEDS cells, forming complexes, and iterating equilibrium/two-component
    system processes to steady state.

    Returns:
        (bulkAverageContainer, bulkDeviationContainer,
         proteinMonomerAverageContainer, proteinMonomerDeviationContainer)
    """
    totalCount_RNA, ids_rnas, distribution_RNA = totalCountIdDistributionRNA(
        sim_data, expression, doubling_time
    )
    totalCount_protein, ids_protein, distribution_protein = (
        totalCountIdDistributionProtein(sim_data, expression, doubling_time)
    )
    ids_complex = sim_data.process.complexation.molecule_names
    ids_equilibrium = sim_data.process.equilibrium.molecule_names
    ids_twoComponentSystem = sim_data.process.two_component_system.molecule_names
    ids_metabolites = sorted(concDict)
    conc_metabolites = (units.mol / units.L) * np.array(
        [concDict[key].asNumber(units.mol / units.L) for key in ids_metabolites]
    )
    allMoleculesIDs = sorted(
        set(ids_rnas)
        | set(ids_protein)
        | set(ids_complex)
        | set(ids_equilibrium)
        | set(ids_twoComponentSystem)
        | set(ids_metabolites)
    )

    complexationStoichMatrix = sim_data.process.complexation.stoich_matrix().astype(
        np.int64, order="F"
    )

    cellDensity = sim_data.constants.cell_density
    cellVolume = avgCellDryMassInit / cellDensity / sim_data.mass.cell_dry_mass_fraction

    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data.struct_array["id"]
    bulkContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", int)],
    )

    rna_idx = bulk_name_to_idx(ids_rnas, bulkContainer["id"])
    protein_idx = bulk_name_to_idx(ids_protein, bulkContainer["id"])
    complexation_molecules_idx = bulk_name_to_idx(ids_complex, bulkContainer["id"])
    equilibrium_molecules_idx = bulk_name_to_idx(ids_equilibrium, bulkContainer["id"])
    two_component_system_molecules_idx = bulk_name_to_idx(
        ids_twoComponentSystem, bulkContainer["id"]
    )
    metabolites_idx = bulk_name_to_idx(ids_metabolites, bulkContainer["id"])
    all_molecules_idx = bulk_name_to_idx(allMoleculesIDs, bulkContainer["id"])

    allMoleculeCounts = np.empty((N_SEEDS, len(allMoleculesIDs)), np.int64)
    proteinMonomerCounts = np.empty((N_SEEDS, len(ids_protein)), np.int64)

    for seed in range(N_SEEDS):
        bulkContainer["count"][all_molecules_idx] = 0
        bulkContainer["count"][rna_idx] = totalCount_RNA * distribution_RNA
        bulkContainer["count"][protein_idx] = (
            totalCount_protein * distribution_protein
        )

        proteinMonomerCounts[seed, :] = counts(bulkContainer, protein_idx)
        complexationMoleculeCounts = counts(bulkContainer, complexation_molecules_idx)

        # Form complexes
        time_step = 2**31
        complexation_rates = sim_data.process.complexation.rates
        system = StochasticSystem(complexationStoichMatrix.T, random_seed=seed)
        complexation_result = system.evolve(
            time_step, complexationMoleculeCounts, complexation_rates
        )

        updatedCompMoleculeCounts = complexation_result["outcome"]
        bulkContainer["count"][complexation_molecules_idx] = updatedCompMoleculeCounts

        metDiffs = np.inf * np.ones_like(counts(bulkContainer, metabolites_idx))
        nIters = 0

        while np.linalg.norm(metDiffs, np.inf) > 1:
            random_state = np.random.RandomState(seed)
            metCounts = conc_metabolites * cellVolume * sim_data.constants.n_avogadro
            metCounts.normalize()
            metCounts.checkNoUnit()
            bulkContainer["count"][metabolites_idx] = metCounts.asNumber().round()

            rxnFluxes, _ = sim_data.process.equilibrium.fluxes_and_molecules_to_SS(
                bulkContainer["count"][equilibrium_molecules_idx],
                cellVolume.asNumber(units.L),
                sim_data.constants.n_avogadro.asNumber(1 / units.mol),
                random_state,
                jit=False,
            )
            bulkContainer["count"][equilibrium_molecules_idx] += np.dot(
                sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
                rxnFluxes.astype(np.int64),
            )
            assert np.all(bulkContainer["count"][equilibrium_molecules_idx] >= 0)

            _, moleculeCountChanges = (
                sim_data.process.two_component_system.molecules_to_ss(
                    bulkContainer["count"][two_component_system_molecules_idx],
                    cellVolume.asNumber(units.L),
                    sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
                )
            )

            bulkContainer["count"][two_component_system_molecules_idx] += (
                moleculeCountChanges.astype(np.int64)
            )

            metDiffs = (
                bulkContainer["count"][metabolites_idx]
                - metCounts.asNumber().round()
            )

            nIters += 1
            if nIters > 100:
                raise Exception("Equilibrium reactions are not converging!")

        allMoleculeCounts[seed, :] = counts(bulkContainer, all_molecules_idx)

    # Build output containers
    bulkAverageContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    bulkDeviationContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    proteinMonomerAverageContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )
    proteinMonomerDeviationContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )

    bulkAverageContainer["count"][all_molecules_idx] = allMoleculeCounts.mean(0)
    bulkDeviationContainer["count"][all_molecules_idx] = allMoleculeCounts.std(0)
    proteinMonomerAverageContainer["count"] = proteinMonomerCounts.mean(0)
    proteinMonomerDeviationContainer["count"] = proteinMonomerCounts.std(0)

    return (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    )


def calculateTranslationSupply(
    sim_data, doubling_time, bulkContainer, avgCellDryMassInit
):
    """
    Compute supply rates of amino acids to translation given doubling time.

    Returns:
        translation_aa_supply (units array of mol/(mass*time))
    """
    aaCounts = sim_data.process.translation.monomer_data["aa_counts"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)
    nAvogadro = sim_data.constants.n_avogadro

    molAAPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    translation_aa_supply = molAAPerGDCW * np.log(2) / doubling_time

    return translation_aa_supply


def _fit_single_condition(sim_data, spec, condition):
    """
    Fit a single condition: find bulk distributions and translation supply.

    This is the worker function called for each condition (possibly in
    parallel via multiprocessing).

    Args:
        sim_data: read-only SimulationDataEcoli
        spec: dict with 'expression', 'concDict', 'avgCellDryMassInit',
              'doubling_time'
        condition: condition label string

    Returns:
        {condition: updated spec dict}
    """
    if VERBOSE > 0:
        print("Fitting condition {}".format(condition))

    (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    ) = calculateBulkDistributions(
        sim_data,
        spec["expression"],
        spec["concDict"],
        spec["avgCellDryMassInit"],
        spec["doubling_time"],
    )
    spec["bulkAverageContainer"] = bulkAverageContainer
    spec["bulkDeviationContainer"] = bulkDeviationContainer
    spec["proteinMonomerAverageContainer"] = proteinMonomerAverageContainer
    spec["proteinMonomerDeviationContainer"] = proteinMonomerDeviationContainer

    spec["translation_aa_supply"] = calculateTranslationSupply(
        sim_data,
        spec["doubling_time"],
        spec["proteinMonomerAverageContainer"],
        spec["avgCellDryMassInit"],
    )

    return {condition: spec}


# (compute_fit_condition was removed — its logic is now the body of
# FitConditionStep.update().)



# ============================================================================
# Step class — reads subsystem objects + cell_specs; writes cell_specs +
# translation_supply_rate.  Read-only on sim_data subsystems.
# ============================================================================

INPUT_PORTS = {
    'tick_4'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'complexation':             'sim_data.complexation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    'molecule_ids':             'overwrite',
    'relation':                 'overwrite',
    'molecule_groups':          'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'sim_data_root':            'overwrite',
    'conditions':                'overwrite',
    'cell_specs':                'overwrite',
}

OUTPUT_PORTS = {
    'tick_5'                            : 'overwrite',
    'cell_specs':             'overwrite',
    'translation_supply_rate': 'overwrite',
}


class FitConditionStep(Step):
    """Step 5 — fit_condition.  Bulk distributions + translation supply rates."""

    config_schema = {'cpus': {'_type': 'integer', '_default': 1}}

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = dict(state['cell_specs'])

        # Build spec dicts for each condition (working copies).
        condition_labels = sorted(cell_specs)
        working_specs = {}
        for label in condition_labels:
            spec = cell_specs[label]
            working_specs[label] = {
                "expression":         spec["expression"],
                "concDict":           spec["concDict"],
                "avgCellDryMassInit": spec["avgCellDryMassInit"],
                "doubling_time":      spec["doubling_time"],
            }

        # Per-condition fitting (parallelizable).
        args = [
            (sd, working_specs[condition], condition)
            for condition in condition_labels
        ]
        apply_updates(_fit_single_condition, args, condition_labels,
                      working_specs, self.config.get('cpus', 1))

        # Merge per-condition results back into cell_specs.
        for label in condition_labels:
            spec = working_specs[label]
            cell_specs[label]["bulkAverageContainer"] = spec["bulkAverageContainer"]
            cell_specs[label]["bulkDeviationContainer"] = spec["bulkDeviationContainer"]
            cell_specs[label]["proteinMonomerAverageContainer"] = (
                spec["proteinMonomerAverageContainer"])
            cell_specs[label]["proteinMonomerDeviationContainer"] = (
                spec["proteinMonomerDeviationContainer"])
            cell_specs[label]["translation_aa_supply"] = spec["translation_aa_supply"]

        # Build translation_supply_rate dict (first occurrence per nutrient).
        translation_supply_rate = {}
        for label in condition_labels:
            nutrients = sd.conditions[label]["nutrients"]
            if nutrients not in translation_supply_rate:
                translation_supply_rate[nutrients] = (
                    working_specs[label]["translation_aa_supply"])

        print(f"  Step 5 (fit_condition) completed in {time.time() - t0:.1f}s")
        return {
            'cell_specs':             cell_specs,
            'translation_supply_rate': translation_supply_rate,
        
            'tick_5': True,}


