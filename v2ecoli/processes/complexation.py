"""
============
Complexation
============

This process encodes molecular simulation of macromolecular complexation,
in which monomers are assembled into complexes. Macromolecular complexation
is done by identifying complexation reactions that are possible (which are
reactions that have sufﬁcient counts of all sub-components), performing one
randomly chosen possible reaction, and re-identifying all possible complexation
reactions. This process assumes that macromolecular complexes form spontaneously,
and that complexation reactions are fast and complete within the time step of the
simulation.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import numpy as np
from stochastic_arrow import StochasticSystem

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from v2ecoli.steps.partition import PartitionedProcess
class Complexation(PartitionedProcess):
    """Complexation PartitionedProcess"""

    name = "ecoli-complexation"
    topology = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}
    defaults = {
        "stoichiometry": np.array([[]]),
        "rates": np.array([]),
        "molecule_names": [],
        "seed": 0,
        "reaction_ids": [],
        "complex_ids": [],
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.stoichiometry = self.parameters["stoichiometry"]
        self.rates = self.parameters["rates"]
        self.molecule_names = self.parameters["molecule_names"]
        self.molecule_idx = None
        self.reaction_ids = self.parameters["reaction_ids"]
        self.complex_ids = self.parameters["complex_ids"]

        self.randomState = np.random.RandomState(seed=self.parameters["seed"])
        self.seed = self.randomState.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "complexation_listener": {
                    **listener_schema(
                        {
                            "complexation_events": (
                                [0] * len(self.reaction_ids),
                                self.reaction_ids,
                            )
                        }
                    )
                },
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        timestep = states["timestep"]
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, states["bulk"]["id"]
            )

        moleculeCounts = counts(states["bulk"], self.molecule_idx)

        result = self.system.evolve(timestep, moleculeCounts, self.rates)
        updatedMoleculeCounts = result["outcome"]
        requests = {}
        requests["bulk"] = [
            (self.molecule_idx, np.fmax(moleculeCounts - updatedMoleculeCounts, 0))
        ]
        return requests

    def evolve_state(self, timestep, states):
        timestep = states["timestep"]
        substrate = counts(states["bulk"], self.molecule_idx)

        result = self.system.evolve(timestep, substrate, self.rates)
        complexationEvents = result["occurrences"]
        outcome = result["outcome"] - substrate

        # Write outputs to listeners
        update = {
            "bulk": [(self.molecule_idx, outcome)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": complexationEvents.astype(int)
                }
            },
        }

        return update
