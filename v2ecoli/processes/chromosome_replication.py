"""
======================
Chromosome Replication
======================

Performs initiation, elongation, and termination of active partial chromosomes
that replicate the chromosome.

First, a round of replication is initiated at a fixed cell mass per origin
of replication and generally occurs once per cell cycle. Second, replication
forks are elongated up to the maximal expected elongation rate, dNTP resource
limitations, and template strand sequence but elongation does not take into
account the action of topoisomerases or the enzymes in the replisome. Finally,
replication forks terminate once they reach the end of their template strand
and the chromosome immediately decatenates forming two separate chromosome
molecules.
"""

import numpy as np

from process_bigraph import Step

from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)

from v2ecoli.library.units import units
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease

from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class ChromosomeReplicationStep(_SafeInvokeMixin, Step):
    """Chromosome Replication — merged single Step."""

    config_schema = {}

    topology = {
        "bulk": ("bulk",),
        "active_replisomes": ("unique", "active_replisome"),
        "oriCs": ("unique", "oriC"),
        "chromosome_domains": ("unique", "chromosome_domain"),
        "full_chromosomes": ("unique", "full_chromosome"),
        "listeners": ("listeners",),
        "environment": ("environment",),
        "timestep": ("timestep",),
    }

    def initialize(self, config):
        defaults = {
            "get_dna_critical_mass": lambda doubling_time: units.Unum,
            "criticalInitiationMass": 975 * units.fg,
            "nutrientToDoublingTime": {},
            "replichore_lengths": np.array([]),
            "sequences": np.array([]),
            "polymerized_dntp_weights": [],
            "replication_coordinate": np.array([]),
            "D_period": np.array([]),
            "replisome_protein_mass": 0,
            "no_child_place_holder": -1,
            "basal_elongation_rate": 967,
            "make_elongation_rates": (
                lambda random, replisomes, base, time_step: units.Unum
            ),
            "mechanistic_replisome": True,
            # molecules
            "replisome_trimers_subunits": [],
            "replisome_monomers_subunits": [],
            "dntps": [],
            "ppi": [],
            # random seed
            "seed": 0,
            "emit_unique": False,
            "time_step": 1,
        }
        params = {**defaults, **config}

        self.get_dna_critical_mass = params["get_dna_critical_mass"]
        self.criticalInitiationMass = params["criticalInitiationMass"]
        self.nutrientToDoublingTime = params["nutrientToDoublingTime"]
        self.replichore_lengths = params["replichore_lengths"]
        self.sequences = params["sequences"]
        self.polymerized_dntp_weights = params["polymerized_dntp_weights"]
        self.replication_coordinate = params["replication_coordinate"]
        self.D_period = params["D_period"]
        self.replisome_protein_mass = params["replisome_protein_mass"]
        self.no_child_place_holder = params["no_child_place_holder"]
        self.basal_elongation_rate = params["basal_elongation_rate"]
        self.make_elongation_rates = params["make_elongation_rates"]

        self.mechanistic_replisome = params["mechanistic_replisome"]

        self.seed = params["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = params.get("emit_unique", True)

        self.replisome_trimers_subunits = params["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = params["replisome_monomers_subunits"]
        self.dntps = params["dntps"]
        self.ppi = params["ppi"]

        self.ppi_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        # Time-gating check
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state["timestep"]

        # Lazy index initialization
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, state["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, state["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, state["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, state["bulk"]["id"])

        # Initialize the update dictionary (only add unique molecule keys when needed)
        update = {
            "bulk": [],
            "listeners": {"replication_data": {}},
        }

        # Get total count of existing oriC's
        n_oriC = state["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC == 0:
            update["next_update_time"] = global_time + timestep
            return update

        # Get current cell mass
        cellMass = state["listeners"]["mass"]["cell_mass"] * units.fg

        # Get critical initiation mass for current simulation environment
        current_media_id = state["environment"]["media_id"]
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id]
        )

        # Calculate mass per origin of replication
        massPerOrigin = cellMass / n_oriC
        criticalMassPerOriC = massPerOrigin / self.criticalInitiationMass

        # Module 1: Replication initiation
        n_active_replisomes = state["active_replisomes"]["_entryState"].sum()

        # Get attributes of existing chromosome domains
        domain_index_existing_domain, child_domains = attrs(
            state["chromosome_domains"], ["domain_index", "child_domains"]
        )

        initiate_replication = False
        if criticalMassPerOriC >= 1.0:
            # Check if subunits are actually available (no allocation needed)
            n_replisome_trimers = counts(state["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(state["bulk"], self.replisome_monomers_idx)
            initiate_replication = not self.mechanistic_replisome or (
                np.all(n_replisome_trimers >= 6 * n_oriC)
                and np.all(n_replisome_monomers >= 2 * n_oriC)
            )

        # If all conditions are met, initiate a round of replication on every
        # origin of replication
        if initiate_replication:
            # Get attributes of existing oriCs and domains
            (domain_index_existing_oric,) = attrs(state["oriCs"], ["domain_index"])

            # Get indexes of the domains that would be getting child domains
            new_parent_domains = np.where(
                np.isin(domain_index_existing_domain, domain_index_existing_oric)
            )[0]

            # Calculate counts of new replisomes and domains to add
            n_new_replisome = 2 * n_oriC
            n_new_domain = 2 * n_oriC

            # Calculate the domain indexes of new domains and oriC's
            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_oriC + 1, dtype=np.int32
            )

            # Add new oriC's, and reset attributes of existing oriC's
            update.setdefault("oriCs", {})
            update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
            update["oriCs"]["add"] = {
                "domain_index": domain_index_new[n_oriC:],
            }

            # Add and set attributes of newly created replisomes
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
            right_replichore = right_replichore.tolist()
            domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )
            update.setdefault("active_replisomes", {})
            update["active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            # Add and set attributes of new chromosome domains
            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
            )
            new_domains_update = {
                "add": {
                    "domain_index": domain_index_new,
                    "child_domains": new_child_domains,
                }
            }

            # Add new domains as children of existing domains
            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {"set": {"child_domains": child_domains}}
            update.setdefault("chromosome_domains", {})
            update["chromosome_domains"].update(
                {**new_domains_update, **existing_domains_update}
            )

            # Decrement counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append((self.replisome_trimers_idx, -6 * n_oriC))
                update["bulk"].append((self.replisome_monomers_idx, -2 * n_oriC))

        # Write data from this module to a listener
        update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
            criticalMassPerOriC.asNumber()
        )
        update["listeners"]["replication_data"]["critical_initiation_mass"] = (
            self.criticalInitiationMass.asNumber(units.fg)
        )

        # Module 2: replication elongation
        # Note: the new replication forks added in the previous module are not
        # elongated until the next timestep.
        if n_active_replisomes != 0:
            # Get actual counts of dNTPs (no allocation — use live state)
            dNtpCounts = counts(state["bulk"], self.dntps_idx)

            # Get attributes of existing replisomes
            (
                domain_index_replisome,
                right_replichore,
                coordinates_replisome,
            ) = attrs(
                state["active_replisomes"],
                ["domain_index", "right_replichore", "coordinates"],
            )

            # Compute elongation rates
            elongation_rates = self.make_elongation_rates(
                self.random_state,
                len(self.sequences),
                self.basal_elongation_rate,
                timestep,
            )

            # Build sequences to polymerize
            sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
            sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

            sequences = buildSequences(
                self.sequences, sequence_indexes, sequence_length, elongation_rates
            )

            # Use polymerize algorithm to quickly calculate the number of
            # elongations each fork catalyzes
            reactionLimit = dNtpCounts.sum()

            active_elongation_rates = elongation_rates[sequence_indexes]

            result = polymerize(
                sequences,
                dNtpCounts,
                reactionLimit,
                self.random_state,
                active_elongation_rates,
            )

            sequenceElongations = result.sequenceElongation
            dNtpsUsed = result.monomerUsages

            # Compute mass increase for each elongated sequence
            mass_increase_dna = computeMassIncrease(
                sequences,
                sequenceElongations,
                self.polymerized_dntp_weights.asNumber(units.fg),
            )

            # Compute masses that should be added to each replisome
            added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

            # Update positions of each fork
            updated_length = sequence_length + sequenceElongations
            updated_coordinates = updated_length[0::2]

            # Reverse signs of fork coordinates on left replichore
            updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

            # Update attributes and submasses of replisomes
            (current_dna_mass,) = attrs(state["active_replisomes"], ["massDiff_DNA"])
            update.setdefault("active_replisomes", {})
            update["active_replisomes"].update(
                {
                    "set": {
                        "coordinates": updated_coordinates,
                        "massDiff_DNA": current_dna_mass + added_dna_mass,
                    }
                }
            )

            # Update counts of polymerized metabolites
            update["bulk"].append((self.dntps_idx, -dNtpsUsed))
            update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

            # Module 3: replication termination
            terminal_lengths = self.replichore_lengths[
                np.logical_not(right_replichore).astype(np.int64)
            ]
            terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

            if terminated_replisomes.sum() > 0:
                # Get domain indexes of terminated forks
                terminated_domains = np.unique(
                    domain_index_replisome[terminated_replisomes]
                )

                # Get attributes of existing domains and full chromosomes
                (
                    domain_index_domains,
                    child_domains,
                ) = attrs(state["chromosome_domains"], ["domain_index", "child_domains"])
                (domain_index_full_chroms,) = attrs(
                    state["full_chromosomes"], ["domain_index"]
                )

                # Initialize array of replisomes that should be deleted
                replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)

                # Count number of new full chromosomes that should be created
                n_new_chromosomes = 0

                # Initialize array for domain indexes of new full chromosomes
                domain_index_new_full_chroms = []

                for terminated_domain_index in terminated_domains:
                    # Get all terminated replisomes in the terminated domain
                    terminated_domain_matching_replisomes = np.logical_and(
                        domain_index_replisome == terminated_domain_index,
                        terminated_replisomes,
                    )

                    # If both replisomes in the domain have terminated
                    if terminated_domain_matching_replisomes.sum() == 2:
                        replisomes_to_delete = np.logical_or(
                            replisomes_to_delete, terminated_domain_matching_replisomes
                        )

                        domain_mask = domain_index_domains == terminated_domain_index

                        # Get child domains of deleted domain
                        child_domains_this_domain = child_domains[
                            np.where(domain_mask)[0][0], :
                        ]

                        # Modify domain index of one existing full chromosome
                        domain_index_full_chroms = domain_index_full_chroms.copy()
                        domain_index_full_chroms[
                            np.where(domain_index_full_chroms == terminated_domain_index)[0]
                        ] = child_domains_this_domain[0]

                        # Increment count of new full chromosome
                        n_new_chromosomes += 1

                        # Append chromosome index of new full chromosome
                        domain_index_new_full_chroms.append(child_domains_this_domain[1])

                # Delete terminated replisomes
                update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

                # Generate new full chromosome molecules
                if n_new_chromosomes > 0:
                    chromosome_add_update = {
                        "add": {
                            "domain_index": domain_index_new_full_chroms,
                            "division_time": state["global_time"] + self.D_period,
                            "has_triggered_division": False,
                        }
                    }

                    chromosome_existing_update = {
                        "set": {"domain_index": domain_index_full_chroms}
                    }

                    update.setdefault("full_chromosomes", {})
                    update["full_chromosomes"].update(
                        {**chromosome_add_update, **chromosome_existing_update}
                    )

                # Increment counts of replisome subunits
                if self.mechanistic_replisome:
                    update["bulk"].append(
                        (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                    )
                    update["bulk"].append(
                        (self.replisome_monomers_idx, replisomes_to_delete.sum())
                    )

        update["next_update_time"] = global_time + timestep
        return update
