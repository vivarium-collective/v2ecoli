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

from process_bigraph import Step

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import ListenerStore, InPlaceDict


class ComplexationStep(_SafeInvokeMixin, Step):
    """Complexation — single-step Gillespie simulation."""

    config_schema = {}

    topology = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}

    def initialize(self, config):
        defaults = {
            "stoichiometry": np.array([[]]),
            "rates": np.array([]),
            "molecule_names": [],
            "seed": 0,
            "reaction_ids": [],
            "complex_ids": [],
            "time_step": 1,
        }
        params = {**defaults, **config}

        self.stoichiometry = params["stoichiometry"]
        self.rates = params["rates"]
        self.molecule_names = params["molecule_names"]
        self.molecule_idx = None

        random_state = np.random.RandomState(seed=params["seed"])
        seed = random_state.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=seed)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state["timestep"]

        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, state["bulk"]["id"]
            )

        substrate = counts(state["bulk"], self.molecule_idx)
        result = self.system.evolve(timestep, substrate, self.rates)
        complexationEvents = result["occurrences"]
        outcome = result["outcome"] - substrate

        return {
            "bulk": [(self.molecule_idx, outcome)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": complexationEvents.astype(int)
                }
            },
            "next_update_time": global_time + timestep,
        }
