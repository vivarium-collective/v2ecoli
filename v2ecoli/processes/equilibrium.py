"""
===========
Equilibrium
===========

This process models how ligands are bound to or unbound
from their transcription factor binding partners in a fashion
that maintains equilibrium.
"""

import numpy as np

from process_bigraph import Step

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.library.units import units
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class EquilibriumStep(_SafeInvokeMixin, Step):
    """Equilibrium — single-step ODE solve + flux correction."""

    config_schema = {}

    topology = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}

    def initialize(self, config):
        defaults = {
            "jit": False,
            "n_avogadro": 0.0,
            "cell_density": 0.0,
            "stoichMatrix": [[]],
            "fluxesAndMoleculesToSS": lambda counts, volume, avogadro, random, jit: ([], []),
            "moleculeNames": [],
            "seed": 0,
            "complex_ids": [],
            "reaction_ids": [],
        }
        params = {**defaults, **config}

        self.jit = params["jit"]
        self.n_avogadro = params["n_avogadro"]
        self.cell_density = params["cell_density"]
        self.stoichMatrix = params["stoichMatrix"]
        self.fluxesAndMoleculesToSS = params["fluxesAndMoleculesToSS"]
        self.product_indices = [
            idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]
        ]
        self.moleculeNames = params["moleculeNames"]
        self.molecule_idx = None
        self.random_state = np.random.RandomState(seed=params["seed"])

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
            'listeners': ListenerStore(),
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

        # Get molecule counts
        moleculeCounts = counts(state["bulk"], self.molecule_idx)

        # Get cell mass and volume
        cellMass = (state["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / self.cell_density

        # Solve ODEs to steady state
        rxnFluxes, _ = self.fluxesAndMoleculesToSS(
            moleculeCounts,
            cellVolume,
            self.n_avogadro,
            self.random_state,
            jit=self.jit,
        )

        # Correct fluxes if they would cause negative counts
        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            negative_metabolite_idxs = np.where(
                np.dot(self.stoichMatrix, rxnFluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            limited_rxn_stoich = self.stoichMatrix[negative_metabolite_idxs, :]
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich < 0, rxnFluxes > 0)
            )[1]
            rev_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich > 0, rxnFluxes < 0)
            )[1]
            rxnFluxes[fwd_rxn_idxs] -= 1
            rxnFluxes[rev_rxn_idxs] += 1
            rxnFluxes[fwd_rxn_idxs] = np.fmax(0, rxnFluxes[fwd_rxn_idxs])
            rxnFluxes[rev_rxn_idxs] = np.fmin(0, rxnFluxes[rev_rxn_idxs])
        else:
            raise ValueError(
                "Could not get positive counts in equilibrium with allocated molecules."
            )

        deltaMolecules = np.dot(self.stoichMatrix, rxnFluxes).astype(int)

        return {
            "bulk": [(self.molecule_idx, deltaMolecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": deltaMolecules[self.product_indices]
                    / timestep
                }
            },
            "next_update_time": global_time + timestep,
        }
