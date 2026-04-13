"""
Elongation models for polypeptide synthesis.

Three models of increasing mechanistic detail:
- BaseElongationModel: max rate elongation, allocates AAs by sequence demand
- TranslationSupplyElongationModel: adds translation_supply gating
- SteadyStateElongationModel: full ppGpp + tRNA-charging kinetics

Each model receives a reference to its parent ``PolypeptideElongation``
process instance via ``self.process`` and uses duck-typed access to its
bulk indices, configuration, and random state.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils.random import stochasticRound
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint, pint_to_unum

from v2ecoli.library.schema import counts, attrs, bulk_name_to_idx
from v2ecoli.processes.metabolism import CONC_UNITS, TIME_UNITS
from v2ecoli.processes.polypeptide.common import MICROMOLAR_UNITS, REMOVED_FROM_CHARGING
from v2ecoli.processes.polypeptide.kinetics import (
    ppgpp_metabolite_changes,
    calculate_trna_charging,
    get_charging_supply_function,
)


class BaseElongationModel(object):
    """
    Base Model: Request amino acids according to upcoming sequence, assuming
    max ribosome elongation.
    """

    def __init__(self, parameters, process):
        self.parameters = parameters
        self.process = process
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.ribosomeElongationRateDict = self.parameters["ribosomeElongationRateDict"]

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        current_media_id = states["environment"]["media_id"]
        rate = self.process.elngRateFactor * unum_to_pint(
            self.ribosomeElongationRateDict[current_media_id]
        ).to(units.aa / units.s).magnitude
        return np.min([self.basal_elongation_rate, rate])

    def amino_acid_counts(self, aasInSequences):
        return aasInSequences

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        aa_counts_for_translation = self.amino_acid_counts(aasInSequences)

        # Bulk requests have to be integers (wcEcoli implicitly casts floats to ints)
        requests = {
            "bulk": [
                (
                    self.process.amino_acid_idx,
                    aa_counts_for_translation.astype(np.int64),
                )
            ]
        }

        # Not modeling charging so set fraction charged to 0 for all tRNA
        fraction_charged = np.zeros(len(self.process.amino_acid_idx))

        return fraction_charged, aa_counts_for_translation.astype(float), requests

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        return total_aa_counts

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        # Update counts of amino acids and water to reflect polymerization
        # reactions
        net_charged = np.zeros(
            len(self.parameters["uncharged_trna_names"]), dtype=np.int64
        )
        return (
            net_charged,
            np.zeros(len(self.process.amino_acids), dtype=np.float64),
            {
                "bulk": [
                    (self.process.amino_acid_idx, -aas_used),
                    (self.process.water_idx, nElongations - nInitialized),
                ]
            },
        )


class TranslationSupplyElongationModel(BaseElongationModel):
    """
    Translation Supply Model: Requests minimum of 1) upcoming amino acid
    sequence assuming max ribosome elongation (ie. Base Model) and 2)
    estimation based on doubling the proteome in one cell cycle (does not
    use ribosome elongation, computed in Parca).
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        # Check if this is required. It is a better request but there may be
        # fewer elongations.
        return np.fmin(self.process.aa_supply, aasInSequences)


class SteadyStateElongationModel(TranslationSupplyElongationModel):
    """
    Steady State Charging Model: Requests amino acids based on the
    Michaelis-Menten competitive inhibition model.
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

        # Cell parameters
        self.cellDensity = unum_to_pint(self.parameters["cellDensity"])

        # Names of molecules associated with tRNA charging
        self.charged_trna_names = self.parameters["charged_trna_names"]
        self.charging_molecule_names = self.parameters["charging_molecule_names"]
        self.synthetase_names = self.parameters["synthetase_names"]

        # Data structures for charging
        self.aa_from_synthetase = self.parameters["aa_from_synthetase"]
        self.charging_stoich_matrix = self.parameters["charging_stoich_matrix"]
        self.charging_molecules_not_aa = np.array(
            [
                mol not in set(self.parameters["amino_acids"])
                for mol in self.charging_molecule_names
            ]
        )

        # ppGpp synthesis
        self.ppgpp_reaction_metabolites = self.parameters["ppgpp_reaction_metabolites"]
        self.elong_rate_by_ppgpp = self.parameters["elong_rate_by_ppgpp"]

        # Parameters for tRNA charging, ribosome elongation and ppGpp reactions
        self.charging_params = {
            "kS": self.parameters["kS"],
            "KMaa": self.parameters["KMaa"],
            "KMtf": self.parameters["KMtf"],
            "krta": self.parameters["krta"],
            "krtf": self.parameters["krtf"],
            "max_elong_rate": float(
                unum_to_pint(self.parameters["elongation_max"]).to(units.aa / units.s).magnitude
            ),
            "charging_mask": np.array(
                [
                    aa not in REMOVED_FROM_CHARGING
                    for aa in self.parameters["amino_acids"]
                ]
            ),
            "unit_conversion": self.parameters["unit_conversion"],
        }
        self.ppgpp_params = {
            "KD_RelA": self.parameters["KD_RelA"],
            "k_RelA": self.parameters["k_RelA"],
            "k_SpoT_syn": self.parameters["k_SpoT_syn"],
            "k_SpoT_deg": self.parameters["k_SpoT_deg"],
            "KI_SpoT": self.parameters["KI_SpoT"],
            "ppgpp_reaction_stoich": self.parameters["ppgpp_reaction_stoich"],
            "synthesis_index": self.parameters["synthesis_index"],
            "degradation_index": self.parameters["degradation_index"],
        }

        # Amino acid supply calculations
        self.aa_supply_scaling = self.parameters["aa_supply_scaling"]

        self.amino_acid_synthesis = self.parameters["amino_acid_synthesis"]
        self.amino_acid_import = self.parameters["amino_acid_import"]
        self.amino_acid_export = self.parameters["amino_acid_export"]
        self.get_pathway_enzyme_counts_per_aa = self.parameters[
            "get_pathway_enzyme_counts_per_aa"
        ]

        # Store as plain float (boundary.external values are plain floats in mM)
        self.import_constraint_threshold = float(
            self.parameters["import_constraint_threshold"]
        )

    def elongation_rate(self, states):
        if (
            self.process.ppgpp_regulation
            and not self.process.disable_ppgpp_elongation_inhibition
        ):
            cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
            cell_volume = cell_mass / self.cellDensity
            counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)
            ppgpp_count = counts(states["bulk"], self.process.ppgpp_idx)
            ppgpp_conc = ppgpp_count * counts_to_molar
            # elong_rate_by_ppgpp is upstream Unum-native
            rate = unum_to_pint(self.elong_rate_by_ppgpp(
                pint_to_unum(ppgpp_conc), self.basal_elongation_rate
            )).to(units.aa / units.s).magnitude
        else:
            rate = super().elongation_rate(states)
        return rate

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        # Conversion from counts to molarity
        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        dry_mass = states["listeners"]["mass"]["dry_mass"] * units.fg
        cell_volume = cell_mass / self.cellDensity
        self.counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)

        # ppGpp related concentrations
        ppgpp_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.ppgpp_idx
        )
        rela_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.rela_idx
        )
        spot_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.spot_idx
        )

        # Get counts and convert synthetase and tRNA to a per AA basis
        synthetase_counts = np.dot(
            self.aa_from_synthetase,
            counts(states["bulk_total"], self.process.synthetase_idx),
        )
        aa_counts = counts(states["bulk_total"], self.process.amino_acid_idx)
        uncharged_trna_array = counts(
            states["bulk_total"], self.process.uncharged_trna_idx
        )
        charged_trna_array = counts(states["bulk_total"], self.process.charged_trna_idx)
        uncharged_trna_counts = np.dot(self.process.aa_from_trna, uncharged_trna_array)
        charged_trna_counts = np.dot(self.process.aa_from_trna, charged_trna_array)
        ribosome_counts = states["active_ribosome"]["_entryState"].sum()

        # Get concentration
        f = aasInSequences / aasInSequences.sum()
        synthetase_conc = self.counts_to_molar * synthetase_counts
        aa_conc = self.counts_to_molar * aa_counts
        uncharged_trna_conc = self.counts_to_molar * uncharged_trna_counts
        charged_trna_conc = self.counts_to_molar * charged_trna_counts
        ribosome_conc = self.counts_to_molar * ribosome_counts

        # Calculate amino acid supply
        aa_in_media = np.array(
            [
                states["boundary"]["external"][aa] > self.import_constraint_threshold
                for aa in self.process.aa_environment_names
            ]
        )
        fwd_enzyme_counts, rev_enzyme_counts = self.get_pathway_enzyme_counts_per_aa(
            counts(states["bulk_total"], self.process.aa_enzyme_idx)
        )
        importer_counts = counts(states["bulk_total"], self.process.aa_importer_idx)
        exporter_counts = counts(states["bulk_total"], self.process.aa_exporter_idx)
        # amino_acid_synthesis/import/export are upstream Numba-compiled and
        # Unum-native; convert pint args at the boundary.
        aa_conc_unum = pint_to_unum(aa_conc)
        dry_mass_unum = pint_to_unum(dry_mass)
        synthesis, fwd_saturation, rev_saturation = self.amino_acid_synthesis(
            fwd_enzyme_counts, rev_enzyme_counts, aa_conc_unum
        )
        import_rates = self.amino_acid_import(
            aa_in_media,
            dry_mass_unum,
            aa_conc_unum,
            importer_counts,
            self.process.mechanistic_aa_transport,
        )
        export_rates = self.amino_acid_export(
            exporter_counts, aa_conc_unum, self.process.mechanistic_aa_transport
        )
        exchange_rates = import_rates - export_rates

        # The closure produced here calls the upstream Unum-native
        # amino_acid_synthesis/import/export with dry_mass and aa_conc, so
        # convert dry_mass at the boundary.
        supply_function = get_charging_supply_function(
            self.process.aa_supply_in_charging,
            self.process.mechanistic_translation_supply,
            self.process.mechanistic_aa_transport,
            self.amino_acid_synthesis,
            self.amino_acid_import,
            self.amino_acid_export,
            self.aa_supply_scaling,
            self.counts_to_molar,
            self.process.aa_supply,
            fwd_enzyme_counts,
            rev_enzyme_counts,
            pint_to_unum(dry_mass),
            importer_counts,
            exporter_counts,
            aa_in_media,
        )

        # Calculate steady state tRNA levels and resulting elongation rate
        self.charging_params["max_elong_rate"] = self.elongation_rate(states)
        (
            fraction_charged,
            v_rib,
            synthesis_in_charging,
            import_in_charging,
            export_in_charging,
        ) = calculate_trna_charging(
            synthetase_conc,
            uncharged_trna_conc,
            charged_trna_conc,
            aa_conc,
            ribosome_conc,
            f,
            self.charging_params,
            supply=supply_function,
            limit_v_rib=True,
            time_limit=states["timestep"],
        )

        # Use the supply calculated from each sub timestep while solving the charging steady state
        if self.process.aa_supply_in_charging:
            conversion = (
                1 / self.counts_to_molar.to(MICROMOLAR_UNITS).magnitude / states["timestep"]
            )
            synthesis = conversion * synthesis_in_charging
            import_rates = conversion * import_in_charging
            export_rates = conversion * export_in_charging
            self.process.aa_supply = synthesis + import_rates - export_rates
        # Use the supply calculated from the starting amino acid concentrations only
        elif self.process.mechanistic_translation_supply:
            # Set supply based on mechanistic synthesis and supply
            self.process.aa_supply = states["timestep"] * (synthesis + exchange_rates)
        else:
            # Adjust aa_supply higher if amino acid concentrations are low
            # Improves stability of charging and mimics amino acid synthesis
            # inhibition and export
            # Polypeptide elongation operates using concentration units of CONC_UNITS (uM)
            # but aa_supply_scaling uses M units, so convert using unit_conversion (1e-6)
            self.process.aa_supply *= self.aa_supply_scaling(
                self.charging_params["unit_conversion"] * aa_conc.to(CONC_UNITS).magnitude,
                aa_in_media,
            )

        aa_counts_for_translation = (
            v_rib
            * f
            * states["timestep"]
            / self.counts_to_molar.to(MICROMOLAR_UNITS).magnitude
        )

        total_trna = charged_trna_array + uncharged_trna_array
        final_charged_trna = stochasticRound(
            self.process.random_state,
            np.dot(fraction_charged, self.process.aa_from_trna * total_trna),
        )

        # Request charged tRNA that will become uncharged
        charged_trna_request = charged_trna_array - final_charged_trna
        charged_trna_request[charged_trna_request < 0] = 0
        uncharged_trna_request = final_charged_trna - charged_trna_array
        uncharged_trna_request[uncharged_trna_request < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        self.aa_counts_for_translation = np.array(aa_counts_for_translation)

        fraction_trna_per_aa = total_trna / np.dot(
            np.dot(self.process.aa_from_trna, total_trna), self.process.aa_from_trna
        )
        total_charging_reactions = stochasticRound(
            self.process.random_state,
            np.dot(aa_counts_for_translation, self.process.aa_from_trna)
            * fraction_trna_per_aa
            + uncharged_trna_request,
        )

        # Only request molecules that will be consumed in the charging reactions
        aa_from_uncharging = -self.charging_stoich_matrix @ charged_trna_request
        aa_from_uncharging[self.charging_molecules_not_aa] = 0
        requested_molecules = (
            -np.dot(self.charging_stoich_matrix, total_charging_reactions)
            - aa_from_uncharging
        )
        requested_molecules[requested_molecules < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        # ppGpp reactions based on charged tRNA
        bulk_request = [
            (
                self.process.charging_molecule_idx,
                requested_molecules.astype(int),
            ),
            (self.process.charged_trna_idx, charged_trna_request.astype(int)),
            # Request water for transfer of AA from tRNA for initial polypeptide.
            # This is severe overestimate assuming the worst case that every
            # elongation is initializing a polypeptide. This excess of water
            # shouldn't matter though.
            (self.process.water_idx, int(aa_counts_for_translation.sum())),
        ]
        if self.process.ppgpp_regulation:
            total_trna_conc = self.counts_to_molar * (
                uncharged_trna_counts + charged_trna_counts
            )
            updated_charged_trna_conc = total_trna_conc * fraction_charged
            updated_uncharged_trna_conc = total_trna_conc - updated_charged_trna_conc
            delta_metabolites, *_ = ppgpp_metabolite_changes(
                updated_uncharged_trna_conc,
                updated_charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                request=True,
                random_state=self.process.random_state,
            )

            request_ppgpp_metabolites = -delta_metabolites.astype(int)
            ppgpp_request = counts(states["bulk"], self.process.ppgpp_idx)
            bulk_request.append((self.process.ppgpp_idx, ppgpp_request))
            bulk_request.append(
                (
                    self.process.ppgpp_rxn_metabolites_idx,
                    request_ppgpp_metabolites,
                )
            )

        return (
            fraction_charged,
            aa_counts_for_translation,
            {
                "bulk": bulk_request,
                "listeners": {
                    "growth_limits": {
                        "original_aa_supply": self.process.aa_supply,
                        "aa_in_media": aa_in_media,
                        "synthetase_conc": synthetase_conc.to(MICROMOLAR_UNITS).magnitude,
                        "uncharged_trna_conc": uncharged_trna_conc.to(
                            MICROMOLAR_UNITS
                        ).magnitude,
                        "charged_trna_conc": charged_trna_conc.to(
                            MICROMOLAR_UNITS
                        ).magnitude,
                        "aa_conc": aa_conc.to(MICROMOLAR_UNITS).magnitude,
                        "ribosome_conc": ribosome_conc.to(MICROMOLAR_UNITS).magnitude,
                        "fraction_aa_to_elongate": f,
                        "aa_supply": self.process.aa_supply,
                        "aa_synthesis": synthesis * states["timestep"],
                        "aa_import": import_rates * states["timestep"],
                        "aa_export": export_rates * states["timestep"],
                        "aa_supply_enzymes_fwd": fwd_enzyme_counts,
                        "aa_supply_enzymes_rev": rev_enzyme_counts,
                        "aa_importers": importer_counts,
                        "aa_exporters": exporter_counts,
                        "aa_supply_aa_conc": aa_conc.to(units.mmol / units.L).magnitude,
                        "aa_supply_fraction_fwd": fwd_saturation,
                        "aa_supply_fraction_rev": rev_saturation,
                        "ppgpp_conc": ppgpp_conc.to(MICROMOLAR_UNITS).magnitude,
                        "rela_conc": rela_conc.to(MICROMOLAR_UNITS).magnitude,
                        "spot_conc": spot_conc.to(MICROMOLAR_UNITS).magnitude,
                    }
                },
                "polypeptide_elongation": {
                    "aa_exchange_rates": (
                        self.counts_to_molar / units.s * (import_rates - export_rates)
                    ).to(CONC_UNITS / TIME_UNITS).magnitude
                },
            },
        )

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        charged_counts_to_uncharge = self.process.aa_from_trna @ charged_trna_counts
        return np.fmin(
            total_aa_counts + charged_counts_to_uncharge, self.aa_counts_for_translation
        )

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        update = {
            "bulk": [],
            "listeners": {"growth_limits": {}},
        }

        # Get tRNA counts
        uncharged_trna = counts(states["bulk"], self.process.uncharged_trna_idx)
        charged_trna = counts(states["bulk"], self.process.charged_trna_idx)
        total_trna = uncharged_trna + charged_trna

        # Adjust molecules for number of charging reactions that occurred
        ## Determine limitations for charging and uncharging reactions
        charged_and_elongated_per_aa = np.fmax(
            0, (aas_used - self.process.aa_from_trna @ charged_trna)
        )
        aa_for_charging = total_aa_counts - charged_and_elongated_per_aa
        n_aa_charged = np.fmin(
            aa_for_charging,
            np.dot(
                self.process.aa_from_trna,
                np.fmin(self.uncharged_trna_to_charge, uncharged_trna),
            ),
        )
        n_uncharged_per_aa = aas_used - charged_and_elongated_per_aa

        ## Calculate changes in tRNA based on limitations
        n_trna_charged = self.distribution_from_aa(n_aa_charged, uncharged_trna, True)
        n_trna_uncharged = self.distribution_from_aa(
            n_uncharged_per_aa, charged_trna, True
        )

        ## Determine reactions that are charged and elongated in same time step without changing
        ## charged or uncharged counts
        charged_and_elongated = self.distribution_from_aa(
            charged_and_elongated_per_aa, total_trna
        )

        ## Determine total number of reactions that occur
        total_uncharging_reactions = charged_and_elongated + n_trna_uncharged
        total_charging_reactions = charged_and_elongated + n_trna_charged
        net_charged = total_charging_reactions - total_uncharging_reactions
        charging_mol_delta = np.dot(
            self.charging_stoich_matrix, total_charging_reactions
        ).astype(int)
        update["bulk"].append((self.process.charging_molecule_idx, charging_mol_delta))
        states["bulk"][self.process.charging_molecule_idx] += charging_mol_delta

        ## Account for uncharging of tRNA during elongation
        update["bulk"].append(
            (self.process.charged_trna_idx, -total_uncharging_reactions)
        )
        update["bulk"].append(
            (self.process.uncharged_trna_idx, total_uncharging_reactions)
        )
        states["bulk"][self.process.charged_trna_idx] += -total_uncharging_reactions
        states["bulk"][self.process.uncharged_trna_idx] += total_uncharging_reactions

        # Update proton counts to reflect polymerization reactions and transfer of AA from tRNA
        # Peptide bond formation releases a water but transferring AA from tRNA consumes a OH-
        # Net production of H+ for each elongation, consume extra water for each initialization
        # since a peptide bond doesn't form
        update["bulk"].append((self.process.proton_idx, nElongations))
        update["bulk"].append((self.process.water_idx, -nInitialized))
        states["bulk"][self.process.proton_idx] += nElongations
        states["bulk"][self.process.water_idx] += -nInitialized

        # Create or degrade ppGpp
        # This should come after all countInc/countDec calls since it shares some molecules with
        # other views and those counts should be updated to get the proper limits on ppGpp reactions
        if self.process.ppgpp_regulation:
            v_rib = (nElongations * self.counts_to_molar).to(
                MICROMOLAR_UNITS
            ).magnitude / states["timestep"]
            ribosome_conc = (
                self.counts_to_molar * states["active_ribosome"]["_entryState"].sum()
            )
            updated_uncharged_trna_counts = (
                counts(states["bulk_total"], self.process.uncharged_trna_idx)
                - net_charged
            )
            updated_charged_trna_counts = (
                counts(states["bulk_total"], self.process.charged_trna_idx)
                + net_charged
            )
            uncharged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_uncharged_trna_counts
            )
            charged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_charged_trna_counts
            )
            ppgpp_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.ppgpp_idx
            )
            rela_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.rela_idx
            )
            spot_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.spot_idx
            )

            # Need to include the next amino acid the ribosome sees for certain
            # cases where elongation does not occur, otherwise f will be NaN
            aa_at_ribosome = aas_used + next_amino_acid_count
            f = aa_at_ribosome / aa_at_ribosome.sum()
            limits = counts(states["bulk"], self.process.ppgpp_rxn_metabolites_idx)
            (
                delta_metabolites,
                ppgpp_syn,
                ppgpp_deg,
                rela_syn,
                spot_syn,
                spot_deg,
                spot_deg_inhibited,
            ) = ppgpp_metabolite_changes(
                uncharged_trna_conc,
                charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                random_state=self.process.random_state,
                limits=limits,
            )

            update["listeners"]["growth_limits"] = {
                "rela_syn": rela_syn,
                "spot_syn": spot_syn,
                "spot_deg": spot_deg,
                "spot_deg_inhibited": spot_deg_inhibited,
            }

            update["bulk"].append(
                (self.process.ppgpp_rxn_metabolites_idx, delta_metabolites.astype(int))
            )
            states["bulk"][self.process.ppgpp_rxn_metabolites_idx] += (
                delta_metabolites.astype(int)
            )

        # Use the difference between (expected AA supply based on expected
        # doubling time and current DCW) and AA used to charge tRNA to update
        # the concentration target in metabolism during the next time step
        aa_used_trna = np.dot(self.process.aa_from_trna, total_charging_reactions)
        aa_diff = self.process.aa_supply - aa_used_trna

        update["listeners"]["growth_limits"]["trna_charged"] = aa_used_trna.astype(int)

        return (
            net_charged,
            aa_diff,
            update,
        )

    def distribution_from_aa(
        self,
        n_aa: npt.NDArray[np.int64],
        n_trna: npt.NDArray[np.int64],
        limited: bool = False,
    ) -> npt.NDArray[np.int64]:
        """
        Distributes counts of amino acids to tRNAs that are associated with
        each amino acid. Uses self.process.aa_from_trna mapping to distribute
        from amino acids to tRNA based on the fraction that each tRNA species
        makes up for all tRNA species that code for the same amino acid.

        Args:
            n_aa: counts of each amino acid to distribute to each tRNA
            n_trna: counts of each tRNA to determine the distribution
            limited: optional, if True, limits the amino acids
                distributed to each tRNA to the number of tRNA that are
                available (n_trna)

        Returns:
            Distributed counts for each tRNA
        """

        # Determine the fraction each tRNA species makes up out of all tRNA of
        # the associated amino acid
        with np.errstate(invalid="ignore"):
            f_trna = n_trna / np.dot(
                np.dot(self.process.aa_from_trna, n_trna), self.process.aa_from_trna
            )
        f_trna[~np.isfinite(f_trna)] = 0

        trna_counts = np.zeros(f_trna.shape, np.int64)
        for count, row in zip(n_aa, self.process.aa_from_trna):
            idx = row == 1
            frac = f_trna[idx]

            counts = np.floor(frac * count)
            diff = int(count - counts.sum())

            # Add additional counts to get up to counts to distribute
            # Prevent adding over the number of tRNA available if limited
            if diff > 0:
                if limited:
                    for _ in range(diff):
                        frac[(n_trna[idx] - counts) == 0] = 0
                        # normalize for multinomial distribution
                        frac /= frac.sum()
                        adjustment = self.process.random_state.multinomial(1, frac)
                        counts += adjustment
                else:
                    adjustment = self.process.random_state.multinomial(diff, frac)
                    counts += adjustment

            trna_counts[idx] = counts

        return trna_counts

