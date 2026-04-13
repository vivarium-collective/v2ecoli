"""
====================
Two Component System
====================

This process models the phosphotransfer reactions of signal transduction pathways.

Specifically, phosphate groups are transferred from histidine kinases to response
regulators and back in response to counts of ligand stimulants.

Mathematical Model
------------------
The phosphotransfer kinetics are modeled as a system of ODEs:

    dx/dt = f(x)

where x is the vector of molecule counts for histidine kinases,
response regulators, and their phosphorylated forms.

The ODE system is solved from t=0 to t=dt using scipy's BDF
(Backward Differentiation Formula) integrator via ``solve_ivp``.
BDF was empirically found to be the fastest solver for this stiff system.

The molecule count change for the timestep is:

    delta_x = x(dt) - x(0)

If the allocator provides fewer molecules than requested, the system
is re-solved to a long-horizon steady state (10000 s) and the changes
for just the current timestep are extracted, ensuring the system remains
physically consistent with the reduced allocation.
"""

import numpy as np
from scipy.integrate import solve_ivp

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from v2ecoli.library.ecoli_step import EcoliStep as Step

from wholecell.utils import units


# Register default topology for this process, associating it with process name
NAME = "ecoli-two-component-system"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}


class TwoComponentSystem(Step):
    """Two Component System Step

    Phosphotransfer metabolites are system-specific (histidine kinases,
    response regulators, phosphate donors). No other process competes for
    these, so runs as a plain Step.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 0.0},
        'jit': {'_type': 'boolean', '_default': False},
        'moleculeNames': {'_type': 'list[string]', '_default': []},
        'moleculesToNextTimeStep': {'_type': 'method', '_default': None},  # legacy opaque callable
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 0.0},
        'seed': {'_type': 'integer', '_default': 0},
        # ODE components (for inline solver -- if provided, moleculesToNextTimeStep is ignored)
        'rates_fn': {'_type': 'method', '_default': None},          # callable(y, t) -> rate vector
        'rates_jac_fn': {'_type': 'method', '_default': None},      # callable(y, t) -> jacobian
        'stoich_matrix_full': {'_type': 'array[float]', '_default': []},  # full stoichiometry (n_molecules x n_reactions)
        'independent_molecule_indexes': {'_type': 'array[integer]', '_default': []},
        'atp_reaction_reactant_mask': {'_type': 'array[boolean]', '_default': []},
        'independent_molecules_atp_index': {'_type': 'integer', '_default': 0},
        'dependency_matrix': {'_type': 'array[float]', '_default': []},  # maps independent -> all molecule changes
    }

    # ODE solver methods to try, in order of preference
    _IVP_METHODS = ["LSODA", "BDF", "Radau", "RK45", "RK23", "DOP853"]

    def initialize(self, config):
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]
        self.moleculeNames = self.parameters["moleculeNames"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)
        self.molecule_idx = None

        # ODE solver: use inline components if provided, else legacy callable
        self.rates_fn = self.parameters.get("rates_fn")
        if self.rates_fn is not None:
            self.rates_jac_fn = self.parameters["rates_jac_fn"]
            self.stoich_full = np.asarray(self.parameters["stoich_matrix_full"])
            self.indep_idx = np.asarray(self.parameters["independent_molecule_indexes"])
            self.atp_mask = np.asarray(self.parameters["atp_reaction_reactant_mask"])
            self.atp_idx = int(self.parameters["independent_molecules_atp_index"])
            self.dep_matrix = np.asarray(self.parameters["dependency_matrix"])
        else:
            self.moleculesToNextTimeStep = self.parameters["moleculesToNextTimeStep"]
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
        }

    # When allocation is insufficient, re-solve ODE to this long horizon
    # to find the steady-state trajectory, then extract changes for just
    # the current timestep (via min_time_step).
    STEADY_STATE_HORIZON_S = 10_000

    def _solve_ode(self, molecule_counts, cell_volume, dt, method="LSODA",
                   min_time_step=None, methods_tried=None):
        """Solve phosphotransfer ODE: dx/dt = S_full @ rates(x).

        Converts counts to concentrations, integrates, then maps changes
        through the dependency matrix to get all molecule changes and
        molecules needed for allocation.

        Returns:
            molecules_needed: molecules required for allocation (n_molecules,)
            all_molecule_changes: net count changes (n_molecules,)
        """
        y_init = molecule_counts / (cell_volume * self.n_avogadro)

        def deriv(t, y):
            return self.stoich_full @ self.rates_fn(y, t)

        def jac(t, y):
            return self.stoich_full @ self.rates_jac_fn(y, t)

        sol = solve_ivp(deriv, [0, dt], y_init, method=method,
                        t_eval=[0, dt], atol=1e-8, jac=jac)
        y = sol.y.T

        # Check for negative concentrations
        if np.any(y[-1] * (cell_volume * self.n_avogadro) <= -1e-3):
            # Try halving the time horizon
            if min_time_step and dt > min_time_step:
                return self._solve_ode(molecule_counts, cell_volume, dt / 2,
                                       method=method, min_time_step=min_time_step)
            # Try alternative solver methods
            if methods_tried is None:
                methods_tried = set()
            methods_tried.add(method)
            for new_method in self._IVP_METHODS:
                if new_method not in methods_tried:
                    return self._solve_ode(molecule_counts, cell_volume, dt,
                                           method=new_method, min_time_step=min_time_step,
                                           methods_tried=methods_tried)
            raise RuntimeError("TCS ODE produced negative values with all methods.")

        y[y < 0] = 0
        y_molecules = y * (cell_volume * self.n_avogadro)
        delta = y_molecules[-1] - y_molecules[0]

        # Map through independent molecules and dependency matrix
        indep_changes = np.round(delta[self.indep_idx])

        # ATP constraint: cap by available reactants
        max_atp = molecule_counts[self.atp_mask].min()
        non_atp = np.concatenate([indep_changes[:self.atp_idx],
                                  indep_changes[self.atp_idx + 1:]])
        indep_changes[self.atp_idx] = np.fmin(non_atp.sum(), max_atp)

        all_changes = self.dep_matrix @ indep_changes

        # Compute molecules needed from negative and positive changes
        neg = indep_changes.copy()
        neg[neg > 0] = 0
        neg[self.atp_idx] = np.concatenate([neg[:self.atp_idx],
                                             neg[self.atp_idx + 1:]]).sum()
        pos = indep_changes.copy()
        pos[pos < 0] = 0
        needed = np.clip(self.dep_matrix @ (-neg), 0, None) + \
                 np.clip(self.dep_matrix @ (-pos), 0, None)

        return needed, all_changes

    def update(self, states, interval=None):
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

        # Solve dx/dt = S @ rates(x) from t=0 to t=dt. In stationary phase
        # (dark-matter scale → 0) all solvers may fail on depleted counts;
        # fall back to zero changes so the sim can coast forward. Once we've
        # failed, latch into a skip-mode for a while to avoid re-running the
        # expensive multi-method retry loop on every tick.
        n_mol = len(self.molecule_idx)
        zero_changes = np.zeros(n_mol, dtype=int)
        if cell_volume <= 0 or not np.isfinite(cell_volume):
            return {"bulk": [(self.molecule_idx, zero_changes)]}
        if getattr(self, "_skip_ticks_remaining", 0) > 0:
            self._skip_ticks_remaining -= 1
            return {"bulk": [(self.molecule_idx, zero_changes)]}
        try:
            if self.rates_fn is not None:
                molecules_required, all_changes = self._solve_ode(
                    molecule_counts, cell_volume, states["timestep"])
            else:
                molecules_required, all_changes = self.moleculesToNextTimeStep(
                    molecule_counts, cell_volume, self.n_avogadro,
                    states["timestep"], self.random_state,
                    method="BDF", jit=self.jit)

            # If ODE solution exceeds available counts, re-solve at long
            # horizon for a physically consistent trajectory.
            if (molecules_required > molecule_counts).any():
                if self.rates_fn is not None:
                    _, all_changes = self._solve_ode(
                        molecule_counts, cell_volume,
                        self.STEADY_STATE_HORIZON_S,
                        min_time_step=states["timestep"])
                else:
                    _, all_changes = self.moleculesToNextTimeStep(
                        molecule_counts, cell_volume, self.n_avogadro,
                        self.STEADY_STATE_HORIZON_S, self.random_state,
                        method="BDF", min_time_step=states["timestep"],
                        jit=self.jit)
        except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError, Exception):
            all_changes = zero_changes
            # Latch: skip the solver for N ticks so we don't re-run the
            # full multi-method retry loop on every step while the cell is
            # in stationary phase.
            self._skip_ticks_remaining = 60

        return {"bulk": [(self.molecule_idx, all_changes.astype(int))]}
