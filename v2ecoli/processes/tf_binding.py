"""
============================
Transcription Factor Binding
============================

This process models how transcription factors bind to promoters on the
DNA sequence.

Mathematical Model
------------------
For each transcription factor (TF) species j:

1. Compute binding probability from active/inactive TF concentrations:

   - For 1CS (one-component system) TFs:
       p_bound_j = f(n_active_j, n_inactive_j)
     where f is a fitted function ``p_promoter_bound_tf`` that maps
     TF counts to the probability that a single promoter is occupied.

   - For 2CS (two-component system) TFs:
       Same function, using phosphorylated (active) vs unphosphorylated
       (inactive) counts.

   - For 0CS (constitutive) TFs:
       p_bound_j = 1.0  (always bound)

2. For each of the N available promoter sites, draw a Bernoulli trial
   with probability p_bound_j (implemented via stochastic rounding):

       n_to_bind_j = min(sum(StochasticRound(p_bound_j, N)), n_available_TF_j)

3. Randomly select n_to_bind_j promoter sites to occupy.

4. Update mass accounting: each bound TF transfers its molecular mass
   (fg) from the bulk pool to the promoter's submass fields:

       delta_mass_i = sum_j(delta_TF[i,j] * m_j)

Special case: MarA/MarR (PD00365) -- MarA activity is modulated by
the fraction of MarR that is complexed with tetracycline (marR-tet),
reflecting antibiotic-mediated de-repression of the marRAB operon.
"""

import numpy as np
import warnings

from v2ecoli.library.ecoli_step import EcoliStep as Step

from v2ecoli.library.schema import (
    listener_schema,
    numpy_schema,
    attrs,
    bulk_name_to_idx,
    counts,
)

from v2ecoli.library.schema_types import PROMOTER_ARRAY

from wholecell.utils.random import stochasticRound
from v2ecoli.types.quantity import ureg as units

# topology_registry removed


# Register default topology for this process, associating it with process name
NAME = "ecoli-tf-binding"
TOPOLOGY = {
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_binding"),
    "global_time": ("global_time",),
}


class TfBinding(Step):
    """Transcription Factor Binding Step"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_to_bound': 'map[string]',
        'active_to_inactive_tf': 'map[string]',
        'bulk_mass_data': 'any',
        'bulk_molecule_ids': 'list[string]',
        'cell_density': {'_type': 'any', '_default': 1100},
        'delta_prob': {'_type': 'node', '_default': {}},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'get_unbound': 'method',
        'n_avogadro': {'_type': 'any', '_default': 6.022141e+23},
        'p_promoter_bound_tf': 'method',
        'rna_ids': 'list[string]',
        'seed': {'_type': 'integer', '_default': 0},
        'submass_indices': 'map[integer]',
        'submass_to_idx': {'_type': 'map[integer]', '_default': {}},
        'tf_ids': 'list[string]',
        'tf_to_tf_type': 'map[string]',
        'time_step': {'_type': 'integer[s]', '_default': 1},
        # Heavy debug listener: n_bound_TF_per_TU is an (n_TU x n_TF) matrix
        # emitting ~900KB/timestep. Off by default; enable only for analyses
        # that need per-TU x per-TF binding resolution.
        'emit_n_bound_TF_per_TU': {'_type': 'boolean', '_default': False},
    }


    def inputs(self):
        return {
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            'bulk_total': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'rna_synth_prob': 'map[float]',
            },
            'timestep': {'_type': 'float[s]', '_default': 0.0},
            'next_update_time': {'_type': 'overwrite[float[s]]', '_default': 1.0},
            'global_time': {'_type': 'float[s]', '_default': 0.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'promoters': PROMOTER_ARRAY,
            'listeners': {
                'rna_synth_prob': {
                    'p_promoter_bound': {'_type': f'array[{self.n_TF},float]', '_default': []},
                    'n_promoter_bound': {'_type': f'array[{self.n_TF},integer]', '_default': []},
                    'n_actual_bound': {'_type': f'array[{self.n_TF},integer]', '_default': []},
                    'n_available_promoters': {'_type': f'array[{self.n_TF},integer]', '_default': []},
                    # Heavy: only populated when emit_n_bound_TF_per_TU is True.
                    # Otherwise emits an empty array to satisfy the schema.
                    'n_bound_TF_per_TU': {'_type': f'array[({self.n_TU}|{self.n_TF}),integer]', '_default': []},
                },
            },
            'next_update_time': 'overwrite[float[s]]',
        }


    def initialize(self, config):

        # Get IDs of transcription factors
        self.tf_ids = self.parameters["tf_ids"]
        self.n_TF = len(self.tf_ids)

        self.rna_ids = self.parameters["rna_ids"]

        # Build dict that maps TFs to transcription units they regulate
        self.delta_prob = self.parameters["delta_prob"]
        self.TF_to_TU_idx = {}

        for i, tf in enumerate(self.tf_ids):
            self.TF_to_TU_idx[tf] = self.delta_prob["deltaI"][
                self.delta_prob["deltaJ"] == i
            ]

        # Get total counts of transcription units
        self.n_TU = self.delta_prob["shape"][0]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create dictionaries and method
        self.p_promoter_bound_tf = self.parameters["p_promoter_bound_tf"]
        self.tf_to_tf_type = self.parameters["tf_to_tf_type"]

        self.active_to_bound = self.parameters["active_to_bound"]
        self.get_unbound = self.parameters["get_unbound"]
        self.active_to_inactive_tf = self.parameters["active_to_inactive_tf"]

        self.active_tfs = {}
        self.inactive_tfs = {}

        for tf in self.tf_ids:
            self.active_tfs[tf] = tf + "[c]"

            if self.tf_to_tf_type[tf] == "1CS":
                if tf == self.active_to_bound[tf]:
                    self.inactive_tfs[tf] = self.get_unbound(tf + "[c]")
                else:
                    self.inactive_tfs[tf] = self.active_to_bound[tf] + "[c]"
            elif self.tf_to_tf_type[tf] == "2CS":
                self.inactive_tfs[tf] = self.active_to_inactive_tf[tf + "[c]"]

        self.bulk_mass_data = self.parameters["bulk_mass_data"]

        # Build array of active TF masses
        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]
        tf_indexes = [
            np.where(self.bulk_molecule_ids == tf_id + "[c]")[0][0]
            for tf_id in self.tf_ids
        ]
        self.active_tf_masses = (
            self.bulk_mass_data[tf_indexes] / self.n_avogadro
        ).to(units.fg).magnitude

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.active_tf_idx = None
        if "PD00365" in self.tf_ids:
            self.marR_name = "CPLX0-7710[c]"
            self.marR_tet = "marR-tet[c]"
        self.submass_indices = self.parameters["submass_indices"]
        self.emit_n_bound_TF_per_TU = self.parameters.get(
            "emit_n_bound_TF_per_TU", False)
    def update_condition(self, timestep, states):
        """
        See :py:meth:`~.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def update(self, states, interval=None):
        # At t=0, convert all strings to indices
        if self.active_tf_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.active_tf_idx = {
                tf_id: bulk_name_to_idx(tf_name, bulk_ids)
                for tf_id, tf_name in self.active_tfs.items()
            }
            self.inactive_tf_idx = {
                tf_id: bulk_name_to_idx(tf_name, bulk_ids)
                for tf_id, tf_name in self.inactive_tfs.items()
            }
            if "PD00365" in self.tf_ids:
                self.marR_idx = bulk_name_to_idx(self.marR_name, bulk_ids)
                self.marR_tet_idx = bulk_name_to_idx(self.marR_tet, bulk_ids)

        # If there are no promoters, return immediately
        if states["promoters"]["_entryState"].sum() == 0:
            return {"promoters": {}}

        # Get attributes of all promoters
        TU_index, bound_TF = attrs(states["promoters"], ["TU_index", "bound_TF"])

        # Calculate number of bound TFs for each TF prior to changes
        n_bound_TF = bound_TF.sum(axis=0)

        # Initialize new bound_TF array
        bound_TF_new = np.zeros_like(bound_TF)

        # Create vectors for storing values
        pPromotersBound = np.zeros(self.n_TF, dtype=np.float64)
        nPromotersBound = np.zeros(self.n_TF, dtype=int)
        nActualBound = np.zeros(self.n_TF, dtype=int)
        n_promoters = np.zeros(self.n_TF, dtype=int)
        n_bound_TF_per_TU = np.zeros((self.n_TU, self.n_TF), dtype=np.int16)

        update = {"bulk": []}

        for tf_idx, tf_id in enumerate(self.tf_ids):
            # Free all DNA-bound transcription factors into free active
            # transcription factors
            curr_tf_idx = self.active_tf_idx[tf_id]
            tf_count = counts(states["bulk"], curr_tf_idx)

            bound_tf_counts = n_bound_TF[tf_idx]
            update["bulk"].append((curr_tf_idx, bound_tf_counts))

            # Get counts of transcription factors
            active_tf_counts = (
                counts(states["bulk_total"], curr_tf_idx) + bound_tf_counts
            )
            n_available_active_tfs = tf_count + bound_tf_counts

            # MarA/MarR special case (PD00365):
            # MarR represses marA transcription. When tetracycline binds
            # MarR (forming marR-tet complex), MarR is sequestered and MarA
            # becomes active. The effective number of active MarA molecules
            # scales with the fraction of MarR that is complexed:
            #   n_active_marA = N_promoters * [marR-tet] / ([marR] + [marR-tet])
            # where N_promoters = 34 (number of MarA-regulated promoter sites).
            if tf_id == "PD00365":
                marR_count = counts(states["bulk_total"], self.marR_idx)
                marR_tet_count = counts(states["bulk_total"], self.marR_tet_idx)
                ratio = marR_tet_count / max(marR_count + marR_tet_count, 1)
                N_MARA_REGULATED_PROMOTERS = 34
                n_available_active_tfs = int(N_MARA_REGULATED_PROMOTERS * ratio)

            # Determine the number of available promoter sites
            available_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
            n_available_promoters = np.count_nonzero(available_promoters)
            n_promoters[tf_idx] = n_available_promoters

            # If there are no active transcription factors to work with,
            # continue to the next transcription factor
            if n_available_active_tfs == 0:
                continue

            # Compute probability of binding the promoter
            if self.tf_to_tf_type[tf_id] == "0CS":
                pPromoterBound = 1.0
            else:
                inactive_tf_counts = counts(
                    states["bulk_total"], self.inactive_tf_idx[tf_id]
                )
                pPromoterBound = self.p_promoter_bound_tf(
                    active_tf_counts, inactive_tf_counts
                )

            # Calculate the number of promoters that should be bound
            n_to_bind = int(
                min(
                    stochasticRound(
                        self.random_state,
                        np.full(n_available_promoters, pPromoterBound),
                    ).sum(),
                    n_available_active_tfs,
                )
            )

            bound_locs = np.zeros(n_available_promoters, dtype=bool)
            if n_to_bind > 0:
                # Determine randomly which DNA targets to bind based on which of
                # the following is more limiting:
                # number of promoter sites to bind, or number of active
                # transcription factors
                bound_locs[
                    self.random_state.choice(
                        n_available_promoters, size=n_to_bind, replace=False
                    )
                ] = True

                # Update count of free transcription factors
                update["bulk"].append((curr_tf_idx, -bound_locs.sum()))

                # Update bound_TF array
                bound_TF_new[available_promoters, tf_idx] = bound_locs

            n_bound_TF_per_TU[:, tf_idx] = np.bincount(
                TU_index[bound_TF_new[:, tf_idx]], minlength=self.n_TU
            )

            # Record values
            pPromotersBound[tf_idx] = pPromoterBound
            nPromotersBound[tf_idx] = n_to_bind
            nActualBound[tf_idx] = bound_locs.sum()

        delta_TF = bound_TF_new.astype(np.int8) - bound_TF.astype(np.int8)
        mass_diffs = delta_TF.dot(self.active_tf_masses)

        submass_update = {
            submass: attrs(states["promoters"], [submass])[0] + mass_diffs[:, i]
            for submass, i in self.submass_indices.items()
        }
        update["promoters"] = {"set": {"bound_TF": bound_TF_new, **submass_update}}

        update["listeners"] = {
            "rna_synth_prob": {
                "p_promoter_bound": pPromotersBound,
                "n_promoter_bound": nPromotersBound,
                "n_actual_bound": nActualBound,
                "n_available_promoters": n_promoters,
                # Heavy (~900KB/ts); emit only when flag is on
                "n_bound_TF_per_TU": (
                    n_bound_TF_per_TU if self.emit_n_bound_TF_per_TU
                    else np.zeros((0, 0), dtype=np.int16)
                ),
            },
        }

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update


def test_tf_binding_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    assert data is not None


if __name__ == "__main__":
    test_tf_binding_listener()
