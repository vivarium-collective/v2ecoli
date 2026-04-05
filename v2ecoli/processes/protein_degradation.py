"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.

TODO:
 - protein complexes
 - add protease functionality
"""

import numpy as np

from v2ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx
from v2ecoli.steps.partition import PartitionedProcess
class ProteinDegradation(PartitionedProcess):
    """Protein Degradation PartitionedProcess"""

    name = "ecoli-protein-degradation"
    topology = {"bulk": ("bulk",), "timestep": ("timestep",)}
    defaults = {
        "raw_degradation_rate": [],
        "water_id": "h2o",
        "amino_acid_ids": [],
        "amino_acid_counts": [],
        "protein_ids": [],
        "protein_lengths": [],
        "seed": 0,
        "time_step": 1,
    }

    # Constructor
    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)

        self.raw_degradation_rate = self.parameters["raw_degradation_rate"]

        self.water_id = self.parameters["water_id"]
        self.amino_acid_ids = self.parameters["amino_acid_ids"]
        self.amino_acid_counts = self.parameters["amino_acid_counts"]

        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)

        # Build protein IDs for S matrix
        self.protein_ids = self.parameters["protein_ids"]
        self.protein_lengths = self.parameters["protein_lengths"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.metabolite_idx = None

        # Build S matrix
        self.degradation_matrix = np.zeros(
            (len(self.metabolite_ids), len(self.protein_ids)), np.int64
        )
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts
        )
        # Assuming N-1 H2O is required per peptide chain length N
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        # In first timestep, convert all strings to indices
        if self.metabolite_idx is None:
            self.water_idx = bulk_name_to_idx(self.water_id, states["bulk"]["id"])
            self.protein_idx = bulk_name_to_idx(self.protein_ids, states["bulk"]["id"])
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_ids, states["bulk"]["id"]
            )

        protein_data = counts(states["bulk"], self.protein_idx)
        # Determine how many proteins to degrade based on the degradation rates
        # and counts of each protein
        nProteinsToDegrade = np.fmin(
            self.random_state.poisson(
                self._proteinDegRates(states["timestep"]) * protein_data
            ),
            protein_data,
        )

        # Determine the number of hydrolysis reactions
        # TODO(vivarium): Missing asNumber() and other unit-related things
        nReactions = np.dot(self.protein_lengths, nProteinsToDegrade)

        # Determine the amount of water required to degrade the selected proteins
        # Assuming one N-1 H2O is required per peptide chain length N
        requests = {
            "bulk": [
                (self.protein_idx, nProteinsToDegrade),
                (self.water_idx, nReactions - np.sum(nProteinsToDegrade)),
            ]
        }
        return requests

    def evolve_state(self, timestep, states):
        # Degrade selected proteins, release amino acids from those proteins
        # back into the cell, and consume H_2O that is required for the
        # degradation process
        allocated_proteins = counts(states["bulk"], self.protein_idx)
        metabolites_delta = np.dot(self.degradation_matrix, allocated_proteins)

        update = {
            "bulk": [
                (self.metabolite_idx, metabolites_delta),
                (self.protein_idx, -allocated_proteins),
            ]
        }

        return update

    def _proteinDegRates(self, timestep):
        return self.raw_degradation_rate * timestep
