"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.

Mathematical Model
------------------
First-order stochastic degradation. For each protein species i with
count n_i and first-order degradation rate constant k_i (1/s):

    n_degraded_i ~ Poisson(k_i * dt * n_i)

Each degraded protein of length L_i releases its constituent amino acids
and consumes (L_i - 1) water molecules via hydrolysis:

    Protein_i + (L_i - 1) H2O  -->  sum_j(a_ij * AA_j)

where a_ij is the count of amino acid j in protein i (from the sequence).

The degradation stoichiometry is encoded as a matrix S (metabolites x proteins).
Row indices correspond to amino acid species + water; columns to protein species.
Water consumption per protein = -(sum of amino acid counts - 1), reflecting
(N-1) peptide bond hydrolyses for a chain of length N:

    delta_metabolites = S @ n_degraded

TODO:
 - protein complexes
 - add protease functionality
"""

import numpy as np

# simulate_process removed

from v2ecoli.library.data_predicates import (
    monotonically_increasing,
    monotonically_decreasing,
    all_nonnegative,
)
from v2ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx

# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = "ecoli-protein-degradation"
TOPOLOGY = {"bulk": ("bulk",), "timestep": ("timestep",)}


class ProteinDegradation(PartitionedProcess):
    """Protein Degradation PartitionedProcess"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'amino_acid_counts': {'_type': 'array[integer]', '_default': []},  # (n_proteins x n_amino_acids) composition matrix
        'amino_acid_ids': {'_type': 'list[string]', '_default': []},
        'protein_ids': {'_type': 'list[string]', '_default': []},
        'protein_lengths': {'_type': 'array[integer[aa]]', '_default': []},  # chain length per protein
        'raw_degradation_rate': {'_type': 'array[float[1/s]]', '_default': []},  # first-order rate constants k_i
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'water_id': {'_type': 'string', '_default': 'h2o'},
    }

    def initialize(self, config):

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

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'timestep': {'_type': 'integer[s]', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
        }

    def calculate_request(self, timestep, states):
        # At t=0, convert molecule name strings to bulk array indices
        if self.metabolite_idx is None:
            self.water_idx = bulk_name_to_idx(self.water_id, states["bulk"]["id"])
            self.protein_idx = bulk_name_to_idx(self.protein_ids, states["bulk"]["id"])
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_ids, states["bulk"]["id"]
            )

        dt = states["timestep"]
        protein_counts = counts(states["bulk"], self.protein_idx)

        # Poisson draw: expected degradation count = k_i * dt * n_i,
        # capped at available protein count
        n_to_degrade = np.fmin(
            self.random_state.poisson(
                self.raw_degradation_rate * dt * protein_counts
            ),
            protein_counts,
        )

        # Water required for hydrolysis: (L_i - 1) per protein of length L_i
        # Total water = sum(L_i * n_degrade_i) - sum(n_degrade_i)
        n_peptide_bonds = np.dot(self.protein_lengths, n_to_degrade)
        water_needed = n_peptide_bonds - np.sum(n_to_degrade)

        requests = {
            "bulk": [
                (self.protein_idx, n_to_degrade),
                (self.water_idx, water_needed),
            ]
        }
        return requests

    def evolve_state(self, timestep, states):
        # Apply degradation: delta_metabolites = S @ n_degraded
        # S encodes amino acid release (+) and water consumption (-)
        allocated_proteins = counts(states["bulk"], self.protein_idx)
        delta_metabolites = np.dot(self.degradation_matrix, allocated_proteins)

        update = {
            "bulk": [
                (self.metabolite_idx, delta_metabolites),
                (self.protein_idx, -allocated_proteins),
            ]
        }

        return update


def test_protein_degradation(return_data=False):
    test_config = {
        "raw_degradation_rate": np.array([0.05, 0.08, 0.13, 0.21]),
        "water_id": "H2O",
        "amino_acid_ids": ["A", "B", "C"],
        "amino_acid_counts": np.array([[5, 7, 13], [1, 3, 5], [4, 4, 4], [13, 11, 5]]),
        "protein_ids": ["w", "x", "y", "z"],
        "protein_lengths": np.array([25, 9, 12, 29]),
    }

    protein_degradation = ProteinDegradation(test_config)

    state = {
        "bulk": np.array(
            [
                ("A", 10),
                ("B", 20),
                ("C", 30),
                ("w", 50),
                ("x", 60),
                ("y", 70),
                ("z", 80),
                ("H2O", 10000),
            ],
            dtype=[("id", "U40"), ("count", int)],
        )
    }

    settings = {"total_time": 100, "initial_state": state}

    data = simulate_process(protein_degradation, settings)

    # Assertions =======================================================
    bulk_timeseries = np.array(data["bulk"])
    protein_data = bulk_timeseries[:, 3:7]
    protein_delta = protein_data[1:] - protein_data[:-1]

    aa_data = bulk_timeseries[:, :3]
    aa_delta = aa_data[1:] - aa_data[:-1]

    h2o_data = bulk_timeseries[:, 7]
    h2o_delta = h2o_data[1:] - h2o_data[:-1]

    # Proteins are monotonically decreasing, never <0:
    for i in range(protein_data.shape[1]):
        assert monotonically_decreasing(protein_data[:, i]), (
            f"Protein {test_config['protein_ids'][i]} is not monotonically decreasing."
        )
        assert all_nonnegative(protein_data), (
            f"Protein {test_config['protein_ids'][i]} falls below 0."
        )

    # Amino acids are monotonically increasing
    for i in range(aa_data.shape[1]):
        assert monotonically_increasing(aa_data[:, i]), (
            f"Amino acid {test_config['amino_acid_ids'][i]} is not monotonically increasing."
        )

    # H2O is monotonically decreasing, never < 0
    assert monotonically_decreasing(h2o_data), "H2O is not monotonically decreasing."
    assert all_nonnegative(h2o_data), "H2O falls below 0."

    # Amino acids are released in specified numbers whenever a protein is degraded
    aa_delta_expected = map(
        lambda i: [test_config["amino_acid_counts"].T @ -protein_delta[i, :]],
        range(protein_delta.shape[0]),
    )
    aa_delta_expected = np.concatenate(list(aa_delta_expected))
    np.testing.assert_array_equal(
        aa_delta,
        aa_delta_expected,
        "Mismatch between expected release of amino acids, and counts actually released.",
    )

    # N-1 molecules H2O is consumed whenever a protein of length N is degraded
    h2o_delta_expected = (protein_delta * (test_config["protein_lengths"] - 1)).T
    h2o_delta_expected = np.sum(h2o_delta_expected, axis=0)
    np.testing.assert_array_equal(
        h2o_delta,
        h2o_delta_expected,
        (
            "Mismatch between number of water molecules consumed\n"
            "and expected to be consumed in degradation."
        ),
    )

    print("Passed all tests.")

    if return_data:
        return data


if __name__ == "__main__":
    test_protein_degradation()
