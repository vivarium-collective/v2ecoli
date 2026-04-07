"""
====================
Two Component System
====================

This process models the phosphotransfer reactions of signal transduction pathways.

Specifically, phosphate groups are transferred from histidine kinases to response regulators
and back in response to counts of ligand stimulants.
"""

import numpy as np

from process_bigraph import Step

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.units import units
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class TwoComponentSystemStep(_SafeInvokeMixin, Step):
    """Two Component System — single-step ODE solver for phosphotransfer."""

    config_schema = {}

    topology = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}

    def initialize(self, config):
        defaults = {
            "jit": False,
            "n_avogadro": 0.0,
            "cell_density": 0.0,
            "moleculesToNextTimeStep": (
                lambda counts, volume, avogadro, timestep, random, method, min_step, jit: ([], [])
            ),
            "moleculeNames": [],
            "seed": 0,
        }
        params = {**defaults, **config}

        self.jit = params["jit"]
        self.n_avogadro = params["n_avogadro"]
        self.cell_density = params["cell_density"]
        self.moleculesToNextTimeStep = params["moleculesToNextTimeStep"]
        self.moleculeNames = params["moleculeNames"]
        self.random_state = np.random.RandomState(seed=params["seed"])
        self.molecule_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)

        # Initialize indices on first call
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, state["bulk"]["id"]
            )

        moleculeCounts = counts(state["bulk"], self.molecule_idx)

        # Get cell volume
        cellMass = (state["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / self.cell_density

        # Solve ODEs to next time step
        _, all_molecule_changes = self.moleculesToNextTimeStep(
            moleculeCounts,
            cellVolume,
            self.n_avogadro,
            timestep,
            self.random_state,
            method="BDF",
            jit=self.jit,
        )

        return {
            "bulk": [(self.molecule_idx, all_molecule_changes.astype(int))],
            "next_update_time": global_time + timestep,
        }
