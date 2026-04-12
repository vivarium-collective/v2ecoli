"""
============
Complexation
============

This process encodes molecular simulation of macromolecular complexation,
in which monomers are assembled into complexes. Macromolecular complexation
is done by identifying complexation reactions that are possible (which are
reactions that have sufficient counts of all sub-components), performing one
randomly chosen possible reaction, and re-identifying all possible complexation
reactions. This process assumes that macromolecular complexes form spontaneously,
and that complexation reactions are fast and complete within the time step of the
simulation.

Mathematical Model
------------------
Complexation is simulated as a continuous-time Markov chain (Gillespie
algorithm) via the ``StochasticSystem`` class from ``stochastic_arrow``.

Given a stoichiometry matrix S (molecules x reactions) and a rate vector k,
the system evolves molecule counts x(t) over the timestep dt:

    x(t + dt) = StochasticSystem.evolve(dt, x(t), k)

Each reaction j fires stochastically with propensity:

    a_j = k_j * product_i(x_i choose |S_ij|)   for all reactant species i

The net molecule count change is:

    delta_x = x(t + dt) - x(t) = S @ occurrences

Note: ``evolve()`` is called twice -- once in ``calculate_request`` to
determine the maximum molecules that could be consumed (for the allocator),
and again in ``evolve_state`` with the actually allocated counts.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import numpy as np
from stochastic_arrow import StochasticSystem

# simulate_process removed

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess

# Register default topology for this process, associating it with process name
NAME = "ecoli-complexation"
TOPOLOGY = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}


class Complexation(PartitionedProcess):
    """Complexation PartitionedProcess"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'complex_ids': {'_type': 'list[string]', '_default': []},
        'molecule_names': {'_type': 'list[string]', '_default': []},
        'rates': {'_type': 'array[float[1/s]]', '_default': np.array([], dtype=float)},  # reaction propensity rate constants
        'reaction_ids': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'stoichiometry': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},  # (reactions x molecules) stoichiometry matrix S
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }

    def initialize(self, config):

        self.stoichiometry = self.parameters["stoichiometry"]
        self.rates = self.parameters["rates"]
        self.molecule_names = self.parameters["molecule_names"]
        self.molecule_idx = None
        self.reaction_ids = self.parameters["reaction_ids"]
        self.complex_ids = self.parameters["complex_ids"]

        self.randomState = np.random.RandomState(seed=self.parameters["seed"])
        self.seed = self.randomState.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'timestep': {'_type': 'integer[s]', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'complexation_listener': {
                    'complexation_events': {'_type': 'overwrite[array[integer]]', '_default': [0] * 1088},
                },
            },
        }


    def calculate_request(self, timestep, states):
        dt = states["timestep"]
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, states["bulk"]["id"]
            )

        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # Single Gillespie run: cache result for use in evolve_state.
        # This avoids running two independent stochastic simulations,
        # which would produce inconsistent results.
        self._cached_result = self.system.evolve(dt, molecule_counts, self.rates)
        self._cached_initial_counts = molecule_counts.copy()
        consumed = np.fmax(molecule_counts - self._cached_result["outcome"], 0)
        return {"bulk": [(self.molecule_idx, consumed)]}

    def evolve_state(self, timestep, states):
        allocated_counts = counts(states["bulk"], self.molecule_idx)

        # If allocation matches request, use the cached Gillespie result
        # directly. Otherwise, re-run with the reduced allocation.
        if np.array_equal(allocated_counts, self._cached_initial_counts):
            result = self._cached_result
        else:
            dt = states["timestep"]
            result = self.system.evolve(dt, allocated_counts, self.rates)

        complexation_events = result["occurrences"]
        delta_counts = result["outcome"] - allocated_counts

        update = {
            "bulk": [(self.molecule_idx, delta_counts)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": complexation_events.astype(int)
                }
            },
        }

        return update


def test_complexation():
    test_config = {
        "stoichiometry": np.array(
            [[-1, 1, 0], [0, -1, 1], [1, 0, -1], [-1, 0, 1], [1, -1, 0], [0, 1, -1]],
            np.int64,
        ),
        "rates": np.array([1, 1, 1, 1, 1, 1], np.float64),
        "molecule_names": ["A", "B", "C"],
        "seed": 1,
        "reaction_ids": [1, 2, 3, 4, 5, 6],
        "complex_ids": [1, 2, 3, 4, 5, 6],
    }

    complexation = Complexation(test_config)

    state = {
        "bulk": np.array(
            [
                ("A", 10),
                ("B", 20),
                ("C", 30),
            ],
            dtype=[("id", "U40"), ("count", int)],
        )
    }

    settings = {"total_time": 10, "initial_state": state}

    data = simulate_process(complexation, settings)
    complexation_events = data["listeners"]["complexation_listener"][
        "complexation_events"
    ]
    assert isinstance(complexation_events[0], list)
    assert isinstance(complexation_events[1], list)
    print(data)


if __name__ == "__main__":
    test_complexation()
