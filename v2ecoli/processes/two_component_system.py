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

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts

from wholecell.utils import units
# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = "ecoli-two-component-system"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}


class TwoComponentSystem(PartitionedProcess):
    """Two Component System PartitionedProcess"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 0.0},
        'jit': {'_type': 'boolean', '_default': False},
        'moleculeNames': {'_type': 'list[string]', '_default': []},
        'moleculesToNextTimeStep': {'_type': 'method', '_default': None},
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 0.0},
        'seed': {'_type': 'integer', '_default': 0},
    }

    def initialize(self, config):

        # Simulation options
        self.jit = self.parameters["jit"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create method
        self.moleculesToNextTimeStep = self.parameters["moleculesToNextTimeStep"]

        # Build views
        self.moleculeNames = self.parameters["moleculeNames"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.molecule_idx = None

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
        self.cell_volume = cell_mass_g / self.cell_density

        # Solve dx/dt = f(x) from t=0 to t=dt using BDF
        self.molecules_required, self.all_molecule_changes = (
            self.moleculesToNextTimeStep(
                molecule_counts,
                self.cell_volume,
                self.n_avogadro,
                states["timestep"],
                self.random_state,
                method="BDF",
                jit=self.jit,
            )
        )
        requests = {"bulk": [(self.molecule_idx, self.molecules_required.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # If allocation was insufficient, re-solve with reduced counts
        # toward the long-horizon steady state
        if (self.molecules_required > molecule_counts).any():
            _, self.all_molecule_changes = self.moleculesToNextTimeStep(
                molecule_counts,
                self.cell_volume,
                self.n_avogadro,
                self.STEADY_STATE_HORIZON_S,
                self.random_state,
                method="BDF",
                min_time_step=states["timestep"],
                jit=self.jit,
            )

        update = {"bulk": [(self.molecule_idx, self.all_molecule_changes.astype(int))]}
        return update
