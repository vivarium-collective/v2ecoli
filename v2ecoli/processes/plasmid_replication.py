"""
======================
Plasmid Replication
======================

Adapted from chromosome replication. Performs initiation, elongation, and
termination of active plasmid molecules that replicate independently of the
chromosome (ColE1 / pBR322).

Replication is initiated asynchronously per plasmid copy when replisome
subunits are available and (optionally) RNA I/II copy number control
permits. The RNA I/II control mechanism (Ataai and Shuler 1986) is
computed inside ``_prepare`` so the updated ``n_rna_initiations`` is
visible to ``_evolve``. Replication forks are elongated unidirectionally;
termination produces a new full plasmid molecule.

Runs as a plain Step, matching the chromosome replication architecture in
v2ecoli. When ``has_plasmid=False`` the unique molecule ports are empty
and this process is a no-op.
"""

import numpy as np

from v2ecoli.library.schema import (
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    listener_schema,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    FULL_PLASMID_ARRAY,
    PLASMID_DOMAIN_ARRAY,
    ORIV_ARRAY,
    PLASMID_ACTIVE_REPLISOME_ARRAY,
)


NAME = "ecoli-plasmid-replication"
TOPOLOGY = {
    "bulk": ("bulk",),
    "plasmid_active_replisomes": ("unique", "plasmid_active_replisome"),
    "oriVs": ("unique", "oriV"),
    "plasmid_domains": ("unique", "plasmid_domain"),
    "full_plasmids": ("unique", "full_plasmid"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "plasmid_rna_control": ("process_state", "plasmid_rna_control"),
    "timestep": ("timestep",),
    "global_time": ("global_time",),
}


class PlasmidReplication(Step):
    """Plasmid Replication Step (ColE1 / pBR322).

    Replisome subunits and dNTPs are shared with chromosome replication;
    because v2ecoli's ChromosomeReplication runs as a plain Step without
    allocator arbitration, this process runs similarly. Resource contention
    is not mediated here.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': {'_type': 'integer', '_default': 1},
        'D_period': {'_type': 'any', '_default': np.array([], dtype=float)},
        'basal_elongation_rate': {'_type': 'integer[nt/s]', '_default': 967},
        'dntps': {'_type': 'list[string]', '_default': []},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'mechanistic_replisome': {'_type': 'boolean', '_default': True},
        'no_child_place_holder': {'_type': 'integer', '_default': -1},
        'polymerized_dntp_weights': {'_type': 'any', '_default': []},
        'ppi': {'_type': 'list[string]', '_default': []},
        'replichore_lengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replisome_monomers_subunits': {'_type': 'list[string]', '_default': []},
        'replisome_protein_mass': {'_type': 'float[fg]', '_default': 0},
        'replisome_trimers_subunits': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'sequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        # RNA I/II copy number control (Ataai-Shuler 1986).
        'use_rna_control': {'_type': 'boolean', '_default': True},
        'rna_I_synthesis_rate': {'_type': 'float', '_default': 63.0 / 3600},
        'rna_I_degradation_rate': {'_type': 'float', '_default': 21.0 / 3600},
        'rna_II_synthesis_rate': {'_type': 'float', '_default': 10.0 / 3600},
        'rna_II_degradation_rate': {'_type': 'float', '_default': 21.0 / 3600},
        'hybridization_rate': {'_type': 'float', '_default': 84.0 / (0.7 * 36000)},
        'hybrid_degradation_rate': {'_type': 'float', '_default': 21.0 / 3600},
        'transcription_time': {'_type': 'float', '_default': 7.0},
        'primer_efficiency': {'_type': 'float', '_default': 0.5},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'plasmid_active_replisomes': {'_type': PLASMID_ACTIVE_REPLISOME_ARRAY, '_default': []},
            'oriVs': {'_type': ORIV_ARRAY, '_default': []},
            'plasmid_domains': {'_type': PLASMID_DOMAIN_ARRAY, '_default': []},
            'full_plasmids': {'_type': FULL_PLASMID_ARRAY, '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'float[fg]', '_default': 0.0},
                },
            },
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'plasmid_rna_control': {
                'rna_I': {'_type': 'float', '_default': 3.0},
                'rna_II': {'_type': 'float', '_default': 0.0},
                'hybrid': {'_type': 'float', '_default': 0.0},
                'time_since_rna_II': {'_type': 'float', '_default': 360.0},
                'PL_fractional': {'_type': 'float', '_default': 0.0},
                'n_rna_initiations': {'_type': 'integer', '_default': 0},
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'global_time': {'_type': 'float', '_default': 0.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'plasmid_active_replisomes': PLASMID_ACTIVE_REPLISOME_ARRAY,
            'oriVs': ORIV_ARRAY,
            'plasmid_domains': PLASMID_DOMAIN_ARRAY,
            'full_plasmids': FULL_PLASMID_ARRAY,
            'plasmid_rna_control': {
                'rna_I': 'overwrite[float]',
                'rna_II': 'overwrite[float]',
                'hybrid': 'overwrite[float]',
                'time_since_rna_II': 'overwrite[float]',
                'PL_fractional': 'overwrite[float]',
                'n_rna_initiations': 'overwrite[integer]',
            },
            'listeners': {
                'replication_data': {
                    'plasmid_critical_mass_per_oriV': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
        }

    def initialize(self, config):
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = unum_to_pint(
            self.parameters["polymerized_dntp_weights"]
        )
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters["replisome_monomers_subunits"]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None

        self.use_rna_control = self.parameters["use_rna_control"]
        if self.use_rna_control:
            self.alpha_I = self.parameters["rna_I_synthesis_rate"]
            self.gamma_I = self.parameters["rna_I_degradation_rate"]
            self.alpha_II = self.parameters["rna_II_synthesis_rate"]
            self.gamma_II = self.parameters["rna_II_degradation_rate"]
            self.k_h = self.parameters["hybridization_rate"]
            self.gamma_H = self.parameters["hybrid_degradation_rate"]
            # 1/K_T_RNAII in seconds — interval between RNA II initiation attempts
            self.rna_II_interval = 1.0 / self.alpha_II
            self.transcription_time = self.parameters["transcription_time"]
            self.primer_efficiency = self.parameters["primer_efficiency"]

    def update(self, states, interval=None):
        self._rna_control_update, self._n_rna_initiations = self._prepare(states)
        return self._evolve(states)

    def _prepare(self, states):
        timestep = states["timestep"]
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, states["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, states["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, states["bulk"]["id"])

        # RNA I/II copy number control (Ataai-Shuler 1986). Update the
        # control state based on current plasmid count; fire integer
        # initiation events when the fractional accumulator crosses 1.0.
        rna_control_update = {}
        n_rna_initiations = 0
        if self.use_rna_control:
            n_plasmids = int(states["full_plasmids"]["_entryState"].sum())
            rna_I = states["plasmid_rna_control"]["rna_I"]
            rna_II = states["plasmid_rna_control"]["rna_II"]
            hybrid = states["plasmid_rna_control"]["hybrid"]
            time_since_rna_II = states["plasmid_rna_control"]["time_since_rna_II"]
            PL_fractional = states["plasmid_rna_control"]["PL_fractional"]

            if n_plasmids > 0:
                # Coupled ODEs for RNA_I, RNA_II, hybrid (Eqs 5, 6, 10):
                # dRNA_I/dt  = K_T_RNAI*N  - k_h*RNA_I*RNA_II - k_d_RNAI*RNA_I
                # dRNA_II/dt = K_T_RNAII*N - k_h*RNA_I*RNA_II - k_d_RNAII*RNA_II
                # dH/dt      = k_h*RNA_I*RNA_II - k_d_H*H
                hybridization = self.k_h * rna_I * rna_II
                d_rna_I = (
                    self.alpha_I * n_plasmids - hybridization - self.gamma_I * rna_I
                ) * timestep
                d_rna_II = (
                    self.alpha_II * n_plasmids - hybridization - self.gamma_II * rna_II
                ) * timestep
                d_hybrid = (hybridization - self.gamma_H * hybrid) * timestep

                new_rna_I = max(0.0, rna_I + d_rna_I)
                new_rna_II = max(0.0, rna_II + d_rna_II)
                new_hybrid = max(0.0, hybrid + d_hybrid)

                new_time = time_since_rna_II + timestep
                if new_time >= self.rna_II_interval:
                    survival = np.exp(-self.k_h * rna_I * self.transcription_time)
                    PL_fractional += n_plasmids * self.primer_efficiency * survival
                    n_rna_initiations = int(PL_fractional)
                    PL_fractional -= n_rna_initiations
                    new_time -= self.rna_II_interval
            else:
                new_rna_I = rna_I
                new_rna_II = rna_II
                new_hybrid = hybrid
                new_time = time_since_rna_II + timestep

            rna_control_update = {
                "rna_I": float(new_rna_I),
                "rna_II": float(new_rna_II),
                "hybrid": float(new_hybrid),
                "time_since_rna_II": float(new_time),
                "PL_fractional": float(PL_fractional),
                "n_rna_initiations": int(n_rna_initiations),
            }

        # Compute elongation rates for the current timestep (used by _evolve).
        n_active_replisomes = states["plasmid_active_replisomes"]["_entryState"].sum()
        if n_active_replisomes > 0:
            self.elongation_rates = self.make_elongation_rates(
                self.random_state,
                len(self.sequences),
                self.basal_elongation_rate,
                timestep,
            )

        return rna_control_update, n_rna_initiations

    def _evolve(self, states):
        update = {
            "bulk": [],
            "plasmid_active_replisomes": {},
            "oriVs": {},
            "plasmid_domains": {},
            "full_plasmids": {},
            "listeners": {"replication_data": {}},
        }
        if self._rna_control_update:
            update["plasmid_rna_control"] = self._rna_control_update

        n_active_replisomes = states["plasmid_active_replisomes"]["_entryState"].sum()
        n_oriV = states["oriVs"]["_entryState"].sum()
        n_full_plasmids = states["full_plasmids"]["_entryState"].sum()

        # If there are no plasmids, return immediately (RNA control has been
        # written; nothing else to do).
        if n_full_plasmids == 0:
            return update

        # Module 1: Replication initiation
        (fork_coordinates, domain_index_replisome) = attrs(
            states["plasmid_active_replisomes"], ["coordinates", "domain_index"]
        )

        domain_index_existing_domain, child_domains = attrs(
            states["plasmid_domains"], ["domain_index", "child_domains"]
        )
        (domain_index_existing_plasmid,) = attrs(
            states["full_plasmids"], ["domain_index"]
        )
        # Plasmid domains that currently have no active replisome.
        idle_plasmid_domains = np.setdiff1d(
            domain_index_existing_plasmid, domain_index_replisome
        )

        initiate_replication = False
        max_new_replisomes = 0
        n_rna_initiations = self._n_rna_initiations
        if len(idle_plasmid_domains) > 0:
            # Gate 1: replisome subunit availability
            n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
            min_trimers = int(np.min(n_replisome_trimers))
            min_monomers = int(np.min(n_replisome_monomers))
            max_by_trimers = min_trimers // 3
            max_by_monomers = min_monomers // 1
            max_new_replisomes = min(max_by_trimers, max_by_monomers)

            subunits_ok = not self.mechanistic_replisome or max_new_replisomes != 0

            # Gate 2: RNA II copy number control (Ataai-Shuler 1986).
            if self.use_rna_control:
                rna_ok = n_rna_initiations > 0
            else:
                n_rna_initiations = len(idle_plasmid_domains)
                rna_ok = True

            initiate_replication = subunits_ok and rna_ok

        if initiate_replication:
            (domain_index_existing_oriv,) = attrs(states["oriVs"], ["domain_index"])

            new_parent_domains = np.where(
                np.isin(domain_index_existing_domain, domain_index_existing_oriv)
            )[0]

            n_new_replisome = 0
            n_new_domain = 0
            domain_index_new = np.array([], dtype=np.int32)

            if max_new_replisomes != 0:
                n_new_replisome = min(
                    len(idle_plasmid_domains),
                    max_new_replisomes,
                    n_rna_initiations,
                )
                n_new_domain = 2 * n_new_replisome

                max_domain_index = domain_index_existing_domain.max()
                domain_index_new = np.arange(
                    max_domain_index + 1,
                    max_domain_index + 2 * n_new_replisome + 1,
                    dtype=np.int32,
                )

            if len(domain_index_new) > 0:
                if n_oriV > n_new_replisome:
                    n_excess_orivs = n_oriV - n_new_replisome
                    domain_index_new_oriv = np.concatenate(
                        (
                            domain_index_existing_oriv[-n_excess_orivs:],
                            domain_index_new,
                        )
                    )
                    update["oriVs"]["set"] = {
                        "domain_index": domain_index_new_oriv[:n_oriV]
                    }
                    update["oriVs"]["add"] = {
                        "domain_index": domain_index_new_oriv[n_oriV:],
                    }
                else:
                    update["oriVs"]["set"] = {
                        "domain_index": domain_index_new[:n_oriV]
                    }
                    update["oriVs"]["add"] = {
                        "domain_index": domain_index_new[n_oriV:],
                    }

            # Plasmid replication is unidirectional: one replisome per oriV.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.full(n_new_replisome, False, dtype=bool)
            domain_index_new_replisome = idle_plasmid_domains[:n_new_replisome]
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )

            update["plasmid_active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            if n_new_domain != 0:
                new_child_domains = np.full(
                    (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
                )
                new_domains_update = {
                    "add": {
                        "domain_index": domain_index_new,
                        "child_domains": new_child_domains,
                    }
                }
                if new_parent_domains.size > 0:
                    if new_parent_domains.size != n_new_replisome:
                        new_parent_domains = new_parent_domains[:n_new_replisome]
                    child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)

                existing_domains_update = {"set": {"child_domains": child_domains}}
                update["plasmid_domains"].update(
                    {**new_domains_update, **existing_domains_update}
                )

            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, -3 * n_new_replisome)
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, -1 * n_new_replisome)
                )

        # Module 2: replication elongation
        if n_active_replisomes == 0:
            return update

        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["plasmid_active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(2), n_active_replisomes)

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
            self.polymerized_dntp_weights.to(units.fg).magnitude,
        )

        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        (current_dna_mass,) = attrs(
            states["plasmid_active_replisomes"], ["massDiff_DNA"]
        )

        update["plasmid_active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: replication termination
        terminated_replisomes = updated_coordinates >= self.replichore_lengths

        if terminated_replisomes.sum() > 0:
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            (
                domain_index_domains,
                child_domains,
            ) = attrs(
                states["plasmid_domains"], ["domain_index", "child_domains"]
            )
            (domain_index_full_plasmid,) = attrs(
                states["full_plasmids"], ["domain_index"]
            )

            replisomes_to_delete = np.zeros_like(
                domain_index_replisome, dtype=np.bool_
            )
            n_new_plasmids = 0
            domain_index_new_full_plasmid = []

            for terminated_domain_index in terminated_domains:
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                # Only a single fork per plasmid domain (unidirectional).
                if terminated_domain_matching_replisomes.any():
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    domain_index_full_plasmid = domain_index_full_plasmid.copy()
                    domain_index_full_plasmid[
                        np.where(
                            domain_index_full_plasmid == terminated_domain_index
                        )[0]
                    ] = child_domains_this_domain[0]

                    n_new_plasmids += 1
                    domain_index_new_full_plasmid.append(
                        child_domains_this_domain[1]
                    )

            update["plasmid_active_replisomes"]["delete"] = np.where(
                replisomes_to_delete
            )[0]

            if n_new_plasmids > 0:
                plasmid_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_plasmid,
                        "division_time": states["global_time"] + self.D_period,
                        "has_triggered_division": False,
                    }
                }
                plasmid_existing_update = {
                    "set": {"domain_index": domain_index_full_plasmid}
                }
                update["full_plasmids"].update(
                    {**plasmid_add_update, **plasmid_existing_update}
                )

            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


def test_plasmid_replication():
    test_config = {}
    process = PlasmidReplication(test_config)
    assert process is not None


if __name__ == "__main__":
    test_plasmid_replication()
