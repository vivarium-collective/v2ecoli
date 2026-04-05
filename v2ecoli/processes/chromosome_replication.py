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

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import (
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    listener_schema,
)

from v2ecoli.library.units import units
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


class ChromosomeReplicationLogic:
    """Biological logic for chromosome replication.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**ChromosomeReplication.defaults, **(parameters or {})}
        parameters = self.parameters
        self.get_dna_critical_mass = parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = parameters["nutrientToDoublingTime"]
        self.replichore_lengths = parameters["replichore_lengths"]
        self.sequences = parameters["sequences"]
        self.polymerized_dntp_weights = parameters["polymerized_dntp_weights"]
        self.replication_coordinate = parameters["replication_coordinate"]
        self.D_period = parameters["D_period"]
        self.replisome_protein_mass = parameters["replisome_protein_mass"]
        self.no_child_place_holder = parameters["no_child_place_holder"]
        self.basal_elongation_rate = parameters["basal_elongation_rate"]
        self.make_elongation_rates = parameters["make_elongation_rates"]
        self.mechanistic_replisome = parameters["mechanistic_replisome"]

        self.seed = parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.replisome_trimers_subunits = parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = parameters["replisome_monomers_subunits"]
        self.dntps = parameters["dntps"]
        self.ppi = parameters["ppi"]

        self.ppi_idx = None
        # Cached values from calculate_request
        self.criticalMassPerOriC = None
        self.elongation_rates = None

    def _init_indices(self, bulk_ids):
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, bulk_ids
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, bulk_ids
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, bulk_ids)

    def calculate_request(self, timestep, states):
        self._init_indices(states["bulk"]["id"])
        requests = {}

        n_oriC = states["oriCs"]["_entryState"].sum()
        if n_oriC == 0:
            self.criticalMassPerOriC = 0
            return requests

        cellMass = states["listeners"]["mass"]["cell_mass"] * units.fg

        current_media_id = states["environment"]["media_id"]
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id]
        )

        massPerOrigin = cellMass / n_oriC
        self.criticalMassPerOriC = massPerOrigin / self.criticalInitiationMass

        requests["bulk"] = []
        if self.criticalMassPerOriC >= 1.0:
            requests["bulk"].append((self.replisome_trimers_idx, 6 * n_oriC))
            requests["bulk"].append((self.replisome_monomers_idx, 2 * n_oriC))

        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        if n_active_replisomes == 0:
            return requests

        (fork_coordinates,) = attrs(states["active_replisomes"], ["coordinates"])
        sequence_length = np.abs(np.repeat(fork_coordinates, 2))

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            len(self.sequences),
            self.basal_elongation_rate,
            states["timestep"],
        )

        sequences = buildSequences(
            self.sequences,
            np.tile(np.arange(4), n_active_replisomes // 2),
            sequence_length,
            self.elongation_rates,
        )

        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        dNtpsTotal = counts(states["bulk"], self.dntps_idx)
        maxFractionalReactionLimit = (
            np.fmin(1, dNtpsTotal / sequenceComposition)
        ).min()

        requests["bulk"].append(
            (
                self.dntps_idx,
                (maxFractionalReactionLimit * sequenceComposition).astype(int),
            )
        )

        return requests

    def evolve_state(self, timestep, states):
        self._init_indices(states["bulk"]["id"])

        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        n_oriC = states["oriCs"]["_entryState"].sum()

        if n_oriC == 0:
            return update

        domain_index_existing_domain, child_domains = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )

        initiate_replication = False
        if self.criticalMassPerOriC >= 1.0:
            n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
            initiate_replication = not self.mechanistic_replisome or (
                np.all(n_replisome_trimers == 6 * n_oriC)
                and np.all(n_replisome_monomers == 2 * n_oriC)
            )

        if initiate_replication:
            (domain_index_existing_oric,) = attrs(states["oriCs"], ["domain_index"])

            new_parent_domains = np.where(
                np.in1d(domain_index_existing_domain, domain_index_existing_oric)
            )[0]

            n_new_replisome = 2 * n_oriC
            n_new_domain = 2 * n_oriC

            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_oriC + 1, dtype=np.int32
            )

            update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
            update["oriCs"]["add"] = {
                "domain_index": domain_index_new[n_oriC:],
            }

            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
            right_replichore = right_replichore.tolist()
            domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )
            update["active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
            )
            new_domains_update = {
                "add": {
                    "domain_index": domain_index_new,
                    "child_domains": new_child_domains,
                }
            }

            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {"set": {"child_domains": child_domains}}
            update["chromosome_domains"].update(
                {**new_domains_update, **existing_domains_update}
            )

            if self.mechanistic_replisome:
                update["bulk"].append((self.replisome_trimers_idx, -6 * n_oriC))
                update["bulk"].append((self.replisome_monomers_idx, -2 * n_oriC))

        update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
            self.criticalMassPerOriC.asNumber()
        )
        update["listeners"]["replication_data"]["critical_initiation_mass"] = (
            self.criticalInitiationMass.asNumber(units.fg)
        )

        # Module 2: elongation
        if n_active_replisomes == 0:
            return update

        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

        sequences = buildSequences(
            self.sequences, sequence_indexes, sequence_length, self.elongation_rates
        )

        reactionLimit = dNtpCounts.sum()
        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates,
        )

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.asNumber(units.fg),
        )

        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

        (current_dna_mass,) = attrs(states["active_replisomes"], ["massDiff_DNA"])
        update["active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: termination
        terminal_lengths = self.replichore_lengths[
            np.logical_not(right_replichore).astype(np.int64)
        ]
        terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

        if terminated_replisomes.sum() > 0:
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            (
                domain_index_domains,
                child_domains,
            ) = attrs(states["chromosome_domains"], ["domain_index", "child_domains"])
            (domain_index_full_chroms,) = attrs(
                states["full_chromosomes"], ["domain_index"]
            )

            replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)
            n_new_chromosomes = 0
            domain_index_new_full_chroms = []

            for terminated_domain_index in terminated_domains:
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                if terminated_domain_matching_replisomes.sum() == 2:
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    domain_index_full_chroms = domain_index_full_chroms.copy()
                    domain_index_full_chroms[
                        np.where(domain_index_full_chroms == terminated_domain_index)[0]
                    ] = child_domains_this_domain[0]

                    n_new_chromosomes += 1
                    domain_index_new_full_chroms.append(child_domains_this_domain[1])

            update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

            if n_new_chromosomes > 0:
                chromosome_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_chroms,
                        "division_time": states["global_time"] + self.D_period,
                        "has_triggered_division": False,
                    }
                }

                chromosome_existing_update = {
                    "set": {"domain_index": domain_index_full_chroms}
                }

                update["full_chromosomes"].update(
                    {**chromosome_add_update, **chromosome_existing_update}
                )

            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


class ChromosomeReplicationRequester(Step):
    """Requester step for chromosome replication."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ChromosomeReplicationLogic(config)
        self.process_name = 'ecoli-chromosome-replication'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': InPlaceDict(),
            'oriCs': InPlaceDict(),
            'listeners': InPlaceDict(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        timestep = state.get('timestep', 1.0)

        requests = proc.calculate_request(timestep, state)

        return {
            'request': {self.process_name: requests},
        }


class ChromosomeReplicationEvolver(Step):
    """Evolver step for chromosome replication.

    RECOMPUTES cached values (criticalMassPerOriC, elongation_rates)
    since Requester and Evolver are separate instances.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance from Requester
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ChromosomeReplicationLogic(config)
        self.process_name = 'ecoli-chromosome-replication'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'active_replisomes': InPlaceDict(),
            'oriCs': InPlaceDict(),
            'chromosome_domains': InPlaceDict(),
            'full_chromosomes': InPlaceDict(),
            'listeners': InPlaceDict(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': InPlaceDict(),
            'oriCs': InPlaceDict(),
            'chromosome_domains': InPlaceDict(),
            'full_chromosomes': InPlaceDict(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        timestep = state.get('timestep', 1.0)

        # Apply allocation
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

                # Evolve
        update = proc.evolve_state(timestep, state)
        update['next_update_time'] = state.get('global_time', 0) + state.get('timestep', 1.0)
        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class ChromosomeReplication(PartitionedProcess):
    """Legacy PartitionedProcess wrapper -- will be removed after migration."""

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
        "replisome_trimers_subunits": [],
        "replisome_monomers_subunits": [],
        "dntps": [],
        "ppi": [],
        "seed": 0,
        "emit_unique": False,
    }

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = ChromosomeReplicationLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0.0}),
                "replication_data": listener_schema(
                    {"critical_initiation_mass": 0.0, "critical_mass_per_oriC": 0.0}
                ),
            },
            "environment": {
                "media_id": {"_default": "", "_updater": "set"},
            },
            "active_replisomes": numpy_schema(
                "active_replisomes", emit=self.parameters["emit_unique"]
            ),
            "oriCs": numpy_schema("oriCs", emit=self.parameters["emit_unique"]),
            "chromosome_domains": numpy_schema(
                "chromosome_domains", emit=self.parameters["emit_unique"]
            ),
            "full_chromosomes": numpy_schema(
                "full_chromosomes", emit=self.parameters["emit_unique"]
            ),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        return self._logic.calculate_request(timestep, states)

    def evolve_state(self, timestep, states):
        return self._logic.evolve_state(timestep, states)
