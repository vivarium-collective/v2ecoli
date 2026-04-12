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
from scipy.integrate import solve_ivp

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from v2ecoli.library.ecoli_step import EcoliStep as Step
from wholecell.utils.random import stochasticRound
from wholecell.utils import units


# Register default topology for this process, associating it with process name
NAME = "ecoli-equilibrium"
TOPOLOGY = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}


class Equilibrium(Step):
    """Equilibrium Step

    Models TF-ligand binding/unbinding. Ligands are TF-specific and not
    consumed by any other process, so this process does not compete for
    resources and runs as a plain Step (no request/allocate/evolve cycle).
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 0.0},
        'complex_ids': {'_type': 'list[string]', '_default': []},
        'fluxesAndMoleculesToSS': {'_type': 'method', '_default': None},  # legacy: opaque ODE solver (used if rates_fn not provided)
        'jit': {'_type': 'boolean', '_default': False},
        'moleculeNames': {'_type': 'list[string]', '_default': []},
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 0.0},
        'reaction_ids': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'stoichMatrix': {'_type': 'array[integer]', '_default': []},  # (molecules x reactions) stoichiometry matrix S
        # ODE components (for inline solver -- if provided, fluxesAndMoleculesToSS is ignored)
        'rates_fn': {'_type': 'method', '_default': None},        # callable(t, y, kf, kr) -> rate vector
        'rates_jac_fn': {'_type': 'method', '_default': None},    # callable(t, y, kf, kr) -> jacobian
        'rates_fwd': {'_type': 'array[float]', '_default': []},   # forward rate constants
        'rates_rev': {'_type': 'array[float]', '_default': []},   # reverse rate constants
        'mets_to_rxn_fluxes': {'_type': 'array[float]', '_default': []},  # maps delta_molecules -> reaction_fluxes
        'Rp': {'_type': 'array[float]', '_default': []},          # reactant partition matrix
        'Pp': {'_type': 'array[float]', '_default': []},          # product partition matrix
    }

    def initialize(self, config):
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Stoichiometry matrix S: (n_molecules x n_reactions)
        self.stoichMatrix = self.parameters["stoichMatrix"]
        self.product_indices = [
            idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]
        ]

        self.moleculeNames = self.parameters["moleculeNames"]
        self.molecule_idx = None

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.complex_ids = self.parameters["complex_ids"]
        self.reaction_ids = self.parameters["reaction_ids"]

        # ODE solver components: if rates_fn is provided, use inline solver;
        # otherwise fall back to the legacy opaque callable
        self.rates_fn = self.parameters.get("rates_fn")
        if self.rates_fn is not None:
            self.rates_jac_fn = self.parameters["rates_jac_fn"]
            self.rates_fwd = np.asarray(self.parameters["rates_fwd"])
            self.rates_rev = np.asarray(self.parameters["rates_rev"])
            self.mets_to_rxn_fluxes = np.asarray(self.parameters["mets_to_rxn_fluxes"])
            self.Rp = np.asarray(self.parameters["Rp"])
            self.Pp = np.asarray(self.parameters["Pp"])
        else:
            self.fluxesAndMoleculesToSS = self.parameters["fluxesAndMoleculesToSS"]
            self.jit = self.parameters.get("jit", False)

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

    def _solve_ode_to_steady_state(self, molecule_counts, cell_volume):
        """Solve the reaction ODE system to steady state.

        Converts molecule counts to concentrations, integrates dy/dt = S @ r(y)
        to t=1e20 (effectively steady state), then converts back to integer
        reaction flux counts via stochastic rounding.

        Returns:
            reaction_fluxes: integer reaction flux vector (n_reactions,)
            molecules_needed: molecules required for these fluxes (n_molecules,)
        """
        y_init = molecule_counts / (cell_volume * self.n_avogadro)

        # dy/dt = S @ rates(t, y, k_fwd, k_rev)
        def deriv(t, y):
            return self.stoichMatrix @ self.rates_fn(
                t, y, self.rates_fwd, self.rates_rev)

        def jac(t, y):
            return self.stoichMatrix @ self.rates_jac_fn(
                t, y, self.rates_fwd, self.rates_rev)

        # Solve to steady state (t -> infinity)
        for method in ["LSODA", "BDF"]:
            try:
                sol = solve_ivp(deriv, [0, 1e20], y_init, method=method,
                                t_eval=[0, 1e20], jac=jac)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("Could not solve equilibrium ODE to steady state.")

        # Convert concentration changes back to molecule count changes
        y = sol.y.T
        y[y < 0] = 0
        y_molecules = y * (cell_volume * self.n_avogadro)
        delta_molecules = y_molecules[-1] - y_molecules[0]

        # Round continuous molecule changes to integer reaction fluxes
        for _ in range(100):
            reaction_fluxes = stochasticRound(
                self.random_state, self.mets_to_rxn_fluxes @ delta_molecules)
            if np.all(molecule_counts + self.stoichMatrix @ reaction_fluxes >= 0):
                break
        else:
            raise ValueError("Negative counts in equilibrium steady state.")

        # Compute molecules needed: partition into reactants consumed
        fwd_flux = np.clip(reaction_fluxes, 0, None)
        rev_flux = np.clip(-reaction_fluxes, 0, None)
        molecules_needed = self.Rp @ fwd_flux + self.Pp @ rev_flux

        return reaction_fluxes, molecules_needed

    def update(self, states, interval=None):
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

        # Solve ODE to steady state -> reaction fluxes nu
        if self.rates_fn is not None:
            reaction_fluxes, _ = self._solve_ode_to_steady_state(
                molecule_counts, cell_volume)
        else:
            reaction_fluxes, _ = self.fluxesAndMoleculesToSS(
                molecule_counts, cell_volume, self.n_avogadro,
                self.random_state, jit=self.jit)

        # Greedy flux correction: the steady-state solution may exceed
        # available counts (stochastic rounding overshoot). Reduce fluxes
        # one at a time for reactions that drive any metabolite negative.
        # Converges in at most sum(|nu|) iterations.
        reaction_fluxes = reaction_fluxes.copy()
        max_iterations = int(np.abs(reaction_fluxes).sum()) + 1
        for _ in range(max_iterations):
            negative_idxs = np.where(
                np.dot(self.stoichMatrix, reaction_fluxes) + molecule_counts < 0
            )[0]
            if len(negative_idxs) == 0:
                break
            limited_stoich = self.stoichMatrix[negative_idxs, :]
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_stoich < 0, reaction_fluxes > 0)
            )[1]
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

        return {
            "bulk": [(self.molecule_idx, delta_molecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": delta_molecules[self.product_indices]
                    / states["timestep"]
                }
            },
        }


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
