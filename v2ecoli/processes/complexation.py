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

Each complex's reactants are complex-specific (no shared resource
competition), so this runs as a plain Step with a single Gillespie call.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import os
import numpy as np
from stochastic_arrow import StochasticSystem

# simulate_process removed

from v2ecoli.library.schema import (
    numpy_schema, bulk_name_to_idx, counts, listener_schema, attrs)
from v2ecoli.library.schema_types import ACTIVE_REPLISOME_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step

# Register default topology for this process, associating it with process name
NAME = "ecoli-complexation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "active_replisomes": ("unique", "active_replisome"),
}

# Fork gate for the Hda-β-clamp complexation reaction (RIDA prep).
# The reaction CPLX0-10342_RXN (2×Hda + β-clamp → Hda-β-clamp) fires whenever
# both substrates are in bulk. Biologically the β-clamp must be DNA-loaded
# for Hda to bind it; v2ecoli does not track clamp loading, so with the gate
# off the Hda-β-clamp complex forms even between replication rounds. When
# the gate is on: (a) the reaction rate is set to 0 when no replisomes are
# active, and (b) any existing CPLX0-10342 is decomposed back to Hda + β-clamp
# in the same tick. Opt in with V2ECOLI_RIDA_COMPLEX_FORK_GATE=1.
RIDA_COMPLEX_FORK_GATE = os.environ.get(
    "V2ECOLI_RIDA_COMPLEX_FORK_GATE", "0") in ("1", "true", "True")
RIDA_COMPLEX_RXN_ID = "CPLX0-10342_RXN"
RIDA_COMPLEX_ID = "CPLX0-10342[c]"
RIDA_HDA_ID = "G7313-MONOMER[c]"
RIDA_BETA_CLAMP_ID = "CPLX0-3761[c]"


class Complexation(Step):
    """Complexation Step (Gillespie)"""

    description = (
        "Complexation — spontaneous monomer→complex assembly (Gillespie SSA).\n\n"
        "Continuous-time Markov chain over reactions (stoichiometry S, rates k):\n"
        "    x(t+dt) = StochasticSystem.evolve(dt, x(t), k)\n"
        "    propensity a_j = k_j · ∏_i C(x_i, |S_ij|);   Δx = S·occurrences.\n"
        "Reactants are complex-specific (no shared-resource competition)."
    )

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

        # RIDA-complex fork-gate: cache the rxn index once so update() is O(1).
        # None if this reaction isn't registered (older caches / minimal composites).
        try:
            self._rida_complex_rxn_idx = list(self.reaction_ids).index(
                RIDA_COMPLEX_RXN_ID)
        except ValueError:
            self._rida_complex_rxn_idx = None
        # Bulk indices resolved on first update() (bulk id array not yet built here).
        self._rida_complex_idx = None
        self._rida_hda_idx = None
        self._rida_clamp_idx = None

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'complexation_listener': {
                    'complexation_events': {'_type': 'overwrite[array[integer]]', '_default': [0] * 1088},
                    'rida_complex_n_fork_pairs': {'_type': 'overwrite[integer]', '_default': 0},
                    'rida_complex_decomposed': {'_type': 'overwrite[integer]', '_default': 0},
                },
            },
        }


    def update(self, states, interval=None):
        dt = states["timestep"]
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, states["bulk"]["id"]
            )
            # Resolve bulk indices for RIDA-complex substrates/product (may be None
            # if this cache doesn't have them registered).
            for attr, mol_id in (
                    ("_rida_complex_idx", RIDA_COMPLEX_ID),
                    ("_rida_hda_idx", RIDA_HDA_ID),
                    ("_rida_clamp_idx", RIDA_BETA_CLAMP_ID)):
                try:
                    setattr(self, attr, bulk_name_to_idx(mol_id, states["bulk"]["id"]))
                except Exception:
                    setattr(self, attr, None)

        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # RIDA-complex fork-gate: zero the CPLX0-10342 formation rate when no
        # replisomes are active (β-clamp only recruits Hda when DNA-loaded,
        # which v2ecoli doesn't track). Existing complexes are decomposed
        # back to Hda + β-clamp in the same tick so the count reflects
        # actual RIDA-active enzyme, not resting soluble complex.
        n_fork_pairs = 0
        rida_decomposed = 0
        rates_this_tick = self.rates
        gate_available = (RIDA_COMPLEX_FORK_GATE
                          and self._rida_complex_rxn_idx is not None)
        if gate_available:
            (fork_coords,) = attrs(states["active_replisomes"], ["coordinates"])
            n_fork_pairs = len(fork_coords) // 2
            if n_fork_pairs == 0:
                rates_this_tick = self.rates.copy()
                rates_this_tick[self._rida_complex_rxn_idx] = 0.0

        # Single Gillespie run: x(t + dt) = StochasticSystem.evolve(dt, x(t), k)
        result = self.system.evolve(dt, molecule_counts, rates_this_tick)
        delta_counts = result["outcome"] - molecule_counts

        # Post-Gillespie decomposition: when no forks, dump any residual
        # CPLX0-10342 back into the Hda + β-clamp bulk pools.
        if (gate_available and n_fork_pairs == 0
                and self._rida_complex_idx is not None
                and self._rida_hda_idx is not None
                and self._rida_clamp_idx is not None):
            local_cplx = self._rida_complex_idx
            local_hda = self._rida_hda_idx
            local_clamp = self._rida_clamp_idx
            # Local indices are into the full bulk array; convert to positions
            # in molecule_idx (self.molecule_names is a subset of bulk).
            try:
                cplx_pos = int(np.where(self.molecule_idx == local_cplx)[0][0])
                hda_pos = int(np.where(self.molecule_idx == local_hda)[0][0])
                clamp_pos = int(np.where(self.molecule_idx == local_clamp)[0][0])
                n_cplx = int(molecule_counts[cplx_pos] + delta_counts[cplx_pos])
                if n_cplx > 0:
                    delta_counts[cplx_pos] -= n_cplx
                    delta_counts[hda_pos] += 2 * n_cplx
                    delta_counts[clamp_pos] += n_cplx
                    rida_decomposed = n_cplx
            except (IndexError, ValueError):
                pass

        return {
            "bulk": [(self.molecule_idx, delta_counts)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": result["occurrences"].astype(int),
                    "rida_complex_n_fork_pairs": int(n_fork_pairs),
                    "rida_complex_decomposed": int(rida_decomposed),
                }
            },
        }
