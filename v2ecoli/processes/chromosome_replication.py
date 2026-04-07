"""
======================
Chromosome Replication
======================

Performs initiation, elongation, and termination of active partial chromosomes
that replicate the chromosome.

First, a round of replication is initiated at a ﬁxed cell mass per origin
of replication and generally occurs once per cell cycle. Second, replication
forks are elongated up to the maximal expected elongation rate, dNTP resource
limitations, and template strand sequence but elongation does not take into
account the action of topoisomerases or the enzymes in the replisome. Finally,
replication forks terminate once they reach the end of their template strand
and the chromosome immediately decatenates forming two separate chromosome
molecules.
"""

import numpy as np

from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)

from v2ecoli.library.units import units
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
class ChromosomeReplicationLogic:
    """Chromosome Replication — shared state container for Requester/Evolver."""

    name = "ecoli-chromosome-replication"
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

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.get_dna_critical_mass = self.parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = self.parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = self.parameters["polymerized_dntp_weights"]
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        # Sim options
        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        # random state
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        # Bulk molecule names
        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters[
            "replisome_monomers_subunits"
        ]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None



class ChromosomeReplicationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute chromosome replication request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        if p.ppi_idx is None:
            p.ppi_idx = bulk_name_to_idx(p.ppi, state["bulk"]["id"])
            p.replisome_trimers_idx = bulk_name_to_idx(
                p.replisome_trimers_subunits, state["bulk"]["id"]
            )
            p.replisome_monomers_idx = bulk_name_to_idx(
                p.replisome_monomers_subunits, state["bulk"]["id"]
            )
            p.dntps_idx = bulk_name_to_idx(p.dntps, state["bulk"]["id"])
        request = {}
        # Get total count of existing oriC's
        n_oriC = state["oriCs"]["_entryState"].sum()
        # If there are no origins, return immediately
        if n_oriC != 0:
            # Get current cell mass
            cellMass = state["listeners"]["mass"]["cell_mass"] * units.fg

            # Get critical initiation mass for current simulation environment
            current_media_id = state["environment"]["media_id"]
            p.criticalInitiationMass = p.get_dna_critical_mass(
                p.nutrientToDoublingTime[current_media_id]
            )

            # Calculate mass per origin of replication, and compare to critical
            # initiation mass. If the cell mass has reached this critical mass,
            # the process will initiate a round of chromosome replication for each
            # origin of replication.
            massPerOrigin = cellMass / n_oriC
            p.criticalMassPerOriC = massPerOrigin / p.criticalInitiationMass

            # If replication should be initiated, request subunits required for
            # building two replisomes per one origin of replication, and edit
            # access to oriC and chromosome domain attributes
            request["bulk"] = []
            if p.criticalMassPerOriC >= 1.0:
                request["bulk"].append((p.replisome_trimers_idx, 6 * n_oriC))
                request["bulk"].append((p.replisome_monomers_idx, 2 * n_oriC))

            # If there are no active forks return
            n_active_replisomes = state["active_replisomes"]["_entryState"].sum()
            if n_active_replisomes != 0:
                # Get current locations of all replication forks
                (fork_coordinates,) = attrs(state["active_replisomes"], ["coordinates"])
                sequence_length = np.abs(np.repeat(fork_coordinates, 2))

                p.elongation_rates = p.make_elongation_rates(
                    p.random_state,
                    len(p.sequences),
                    p.basal_elongation_rate,
                    state["timestep"],
                )

                sequences = buildSequences(
                    p.sequences,
                    np.tile(np.arange(4), n_active_replisomes // 2),
                    sequence_length,
                    p.elongation_rates,
                )

                # Count number of each dNTP in sequences for the next timestep
                sequenceComposition = np.bincount(
                    sequences[sequences != polymerize.PAD_VALUE], minlength=4
                )

                # If one dNTP is limiting then limit the request for the other three by
                # the same ratio
                dNtpsTotal = counts(state["bulk"], p.dntps_idx)
                maxFractionalReactionLimit = (
                    np.fmin(1, dNtpsTotal / sequenceComposition)
                ).min()

                # Request dNTPs
                request["bulk"].append(
                    (
                        p.dntps_idx,
                        (maxFractionalReactionLimit * sequenceComposition).astype(int),
                    )
                )
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class ChromosomeReplicationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        # Initialize the update dictionary
        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        # Module 1: Replication initiation
        # Get number of existing replisomes and oriCs
        n_active_replisomes = state["active_replisomes"]["_entryState"].sum()
        n_oriC = state["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC != 0:
            # Get attributes of existing chromosome domains
            domain_index_existing_domain, child_domains = attrs(
                state["chromosome_domains"], ["domain_index", "child_domains"]
            )

            initiate_replication = False
            if p.criticalMassPerOriC >= 1.0:
                # Get number of available replisome subunits
                n_replisome_trimers = counts(state["bulk"], p.replisome_trimers_idx)
                n_replisome_monomers = counts(state["bulk"], p.replisome_monomers_idx)
                # Initiate replication only when
                # 1) The cell has reached the critical mass per oriC
                # 2) If mechanistic replisome option is on, there are enough
                # replisome subunits to assemble two replisomes per existing OriC.
                # Note that we assume asynchronous initiation does not happen.
                initiate_replication = not p.mechanistic_replisome or (
                    np.all(n_replisome_trimers == 6 * n_oriC)
                    and np.all(n_replisome_monomers == 2 * n_oriC)
                )

            # If all conditions are met, initiate a round of replication on every
            # origin of replication
            if initiate_replication:
                # Get attributes of existing oriCs and domains
                (domain_index_existing_oric,) = attrs(state["oriCs"], ["domain_index"])

                # Get indexes of the domains that would be getting child domains
                # (domains that contain an origin)
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
                # All oriC's must be assigned new domain indexes
                update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
                update["oriCs"]["add"] = {
                    "domain_index": domain_index_new[n_oriC:],
                }

                # Add and set attributes of newly created replisomes.
                # New replisomes inherit the domain indexes of the oriC's they
                # were initiated from. Two replisomes are formed per oriC, one on
                # the right replichore, and one on the left.
                coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
                right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
                right_replichore = right_replichore.tolist()
                domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
                massDiff_protein_new_replisome = np.full(
                    n_new_replisome,
                    p.replisome_protein_mass if p.mechanistic_replisome else 0.0,
                )
                update["active_replisomes"]["add"] = {
                    "coordinates": coordinates_replisome,
                    "right_replichore": right_replichore,
                    "domain_index": domain_index_new_replisome,
                    "massDiff_protein": massDiff_protein_new_replisome,
                }

                # Add and set attributes of new chromosome domains. All new domains
                # should have have no children domains.
                new_child_domains = np.full(
                    (n_new_domain, 2), p.no_child_place_holder, dtype=np.int32
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
                update["chromosome_domains"].update(
                    {**new_domains_update, **existing_domains_update}
                )

                # Decrement counts of replisome subunits
                if p.mechanistic_replisome:
                    update["bulk"].append((p.replisome_trimers_idx, -6 * n_oriC))
                    update["bulk"].append((p.replisome_monomers_idx, -2 * n_oriC))

            # Write data from this module to a listener
            update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
                p.criticalMassPerOriC.asNumber()
            )
            update["listeners"]["replication_data"]["critical_initiation_mass"] = (
                p.criticalInitiationMass.asNumber(units.fg)
            )

            # Module 2: replication elongation
            # If no active replisomes are present, return immediately
            # Note: the new replication forks added in the previous module are not
            # elongated until the next timestep.
            if n_active_replisomes != 0:
                # Get allocated counts of dNTPs
                dNtpCounts = counts(state["bulk"], p.dntps_idx)

                # Get attributes of existing replisomes
                (
                    domain_index_replisome,
                    right_replichore,
                    coordinates_replisome,
                ) = attrs(
                    state["active_replisomes"],
                    ["domain_index", "right_replichore", "coordinates"],
                )

                # Build sequences to polymerize
                sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
                sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

                sequences = buildSequences(
                    p.sequences, sequence_indexes, sequence_length, p.elongation_rates
                )

                # Use polymerize algorithm to quickly calculate the number of
                # elongations each fork catalyzes
                reactionLimit = dNtpCounts.sum()

                active_elongation_rates = p.elongation_rates[sequence_indexes]

                result = polymerize(
                    sequences,
                    dNtpCounts,
                    reactionLimit,
                    p.random_state,
                    active_elongation_rates,
                )

                sequenceElongations = result.sequenceElongation
                dNtpsUsed = result.monomerUsages

                # Compute mass increase for each elongated sequence
                mass_increase_dna = computeMassIncrease(
                    sequences,
                    sequenceElongations,
                    p.polymerized_dntp_weights.asNumber(units.fg),
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
                update["active_replisomes"].update(
                    {
                        "set": {
                            "coordinates": updated_coordinates,
                            "massDiff_DNA": current_dna_mass + added_dna_mass,
                        }
                    }
                )

                # Update counts of polymerized metabolites
                update["bulk"].append((p.dntps_idx, -dNtpsUsed))
                update["bulk"].append((p.ppi_idx, dNtpsUsed.sum()))

                # Module 3: replication termination
                # Determine if any forks have reached the end of their sequences. If
                # so, delete the replisomes and domains that were terminated.
                terminal_lengths = p.replichore_lengths[
                    np.logical_not(right_replichore).astype(np.int64)
                ]
                terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

                # If any forks were terminated,
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

                        # If both replisomes in the domain have terminated, we are
                        # ready to split the chromosome and update the attributes.
                        if terminated_domain_matching_replisomes.sum() == 2:
                            # Tag replisomes and domains with the given domain index
                            # for deletion
                            replisomes_to_delete = np.logical_or(
                                replisomes_to_delete, terminated_domain_matching_replisomes
                            )

                            domain_mask = domain_index_domains == terminated_domain_index

                            # Get child domains of deleted domain
                            child_domains_this_domain = child_domains[
                                np.where(domain_mask)[0][0], :
                            ]

                            # Modify domain index of one existing full chromosome to
                            # index of first child domain
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
                                "division_time": state["global_time"] + p.D_period,
                                "has_triggered_division": False,
                            }
                        }

                        # Reset domain index of existing chromosomes that have finished
                        # replication
                        chromosome_existing_update = {
                            "set": {"domain_index": domain_index_full_chroms}
                        }

                        update["full_chromosomes"].update(
                            {**chromosome_add_update, **chromosome_existing_update}
                        )

                    # Increment counts of replisome subunits
                    if p.mechanistic_replisome:
                        update["bulk"].append(
                            (p.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                        )
                        update["bulk"].append(
                            (p.replisome_monomers_idx, replisomes_to_delete.sum())
                        )
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
