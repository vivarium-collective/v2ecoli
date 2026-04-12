"""
===========
Equilibrium
===========

This process models how ligands are bound to or unbound from their
transcription factor binding partners in a fashion that maintains
equilibrium.

Mathematical Model
------------------
The equilibrium binding/unbinding is computed in two phases:

1. **Steady-state ODE solve**: Given current molecule counts x, cell volume V,
   and Avogadro's number N_A, solve the reaction ODE system to steady state
   to obtain integer reaction fluxes nu_j (net number of times each reaction
   fires):

       x_ss, nu = fluxesAndMoleculesToSS(x, V, N_A)

2. **Greedy flux correction**: If the allocator provides fewer molecules than
   the ODE solution requires, reaction fluxes are iteratively reduced to
   avoid driving any metabolite count negative. The algorithm decrements
   fluxes one unit at a time for reactions consuming limiting metabolites:

       while any (S @ nu + x_allocated < 0):
           reduce offending forward fluxes by 1 (clamp at 0)
           reduce offending reverse fluxes by 1 (clamp at 0)

   Convergence: each iteration reduces at least one flux by 1, so the
   loop terminates in at most sum(|nu|) iterations.

   The final molecule count change is:

       delta_x = S @ nu_corrected

   where S is the stoichiometry matrix (molecules x reactions).
"""

import numpy as np

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess

from wholecell.utils import units


# Register default topology for this process, associating it with process name
NAME = "ecoli-equilibrium"
TOPOLOGY = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}


class Equilibrium(PartitionedProcess):
    """Equilibrium PartitionedProcess

    Models ligand binding/unbinding to maintain equilibrium.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 0.0},
        'complex_ids': {'_type': 'list[string]', '_default': []},
        'fluxesAndMoleculesToSS': {'_type': 'method', '_default': None},  # ODE solver -> steady state
        'jit': {'_type': 'boolean', '_default': False},
        'moleculeNames': {'_type': 'list[string]', '_default': []},
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 0.0},
        'reaction_ids': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'stoichMatrix': {'_type': 'array[integer]', '_default': []},  # (molecules x reactions) stoichiometry matrix S
    }

    def initialize(self, config):

        # Simulation options
        # utilized in the fluxes and molecules function
        self.jit = self.parameters["jit"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create matrix and method
        # stoichMatrix: (94, 33), molecule counts are (94,).
        self.stoichMatrix = self.parameters["stoichMatrix"]

        # fluxesAndMoleculesToSS: solves ODES to get to steady state based off
        # of cell density, volumes and molecule counts
        self.fluxesAndMoleculesToSS = self.parameters["fluxesAndMoleculesToSS"]

        self.product_indices = [
            idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]
        ]

        # Build views
        # moleculeNames: list of molecules that are being iterated over size: 94
        self.moleculeNames = self.parameters["moleculeNames"]
        self.molecule_idx = None

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.complex_ids = self.parameters["complex_ids"]
        self.reaction_ids = self.parameters["reaction_ids"]

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'float[fg]', '_default': 0},
                },
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'equilibrium_listener': {
                    'reaction_rates': {'_type': 'overwrite[array[float[1/s]]]', '_default': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                },
            },
        }

    def calculate_request(self, timestep, states):
        # At t=0, convert molecule name strings to bulk array indices
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )

        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # cell_volume = cell_mass / cell_density  [g / (g/L) = L]
        cell_mass_g = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cell_volume = cell_mass_g / self.cell_density

        # Phase 1: Solve ODE to steady state -> reaction fluxes nu
        self.reaction_fluxes, self.req = self.fluxesAndMoleculesToSS(
            molecule_counts,
            cell_volume,
            self.n_avogadro,
            self.random_state,
            jit=self.jit,
        )

        requests = {"bulk": [(self.molecule_idx, self.req.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        molecule_counts = counts(states["bulk"], self.molecule_idx)
        reaction_fluxes = self.reaction_fluxes.copy()

        # Phase 2: Greedy flux correction when allocation is insufficient.
        # Iteratively reduce fluxes that would drive any metabolite negative.
        # Each iteration decrements at least one flux by 1, so convergence
        # is guaranteed in at most sum(|nu|) iterations.
        max_iterations = int(np.abs(reaction_fluxes).sum()) + 1
        for _ in range(max_iterations):
            # Check: S @ nu + x_allocated >= 0 for all metabolites?
            negative_idxs = np.where(
                np.dot(self.stoichMatrix, reaction_fluxes) + molecule_counts < 0
            )[0]
            if len(negative_idxs) == 0:
                break

            # Identify reactions consuming the limiting metabolites
            limited_stoich = self.stoichMatrix[negative_idxs, :]
            # Forward reactions (nu > 0) that consume limiting metabolites (S < 0)
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_stoich < 0, reaction_fluxes > 0)
            )[1]
            # Reverse reactions (nu < 0) that consume in the reverse direction (S > 0)
            rev_rxn_idxs = np.where(
                np.logical_and(limited_stoich > 0, reaction_fluxes < 0)
            )[1]
            reaction_fluxes[fwd_rxn_idxs] -= 1
            reaction_fluxes[rev_rxn_idxs] += 1
            reaction_fluxes[fwd_rxn_idxs] = np.fmax(0, reaction_fluxes[fwd_rxn_idxs])
            reaction_fluxes[rev_rxn_idxs] = np.fmin(0, reaction_fluxes[rev_rxn_idxs])
        else:
            raise ValueError(
                "Could not get positive counts in equilibrium with allocated molecules."
            )

        # delta_x = S @ nu_corrected
        delta_molecules = np.dot(self.stoichMatrix, reaction_fluxes).astype(int)

        update = {
            "bulk": [(self.molecule_idx, delta_molecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": delta_molecules[self.product_indices]
                    / states["timestep"]
                }
            },
        }

        return update


def test_equilibrium_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    listeners = sim.query()["agents"]["0"]["listeners"]
    assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][0], list)
    assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][1], list)


if __name__ == "__main__":
    test_equilibrium_listener()
