"""Step 6 — promoter_binding.  Fit TF binding probabilities and recruitment.

For each TF-promoter pair, solve a small convex optimization so that
the RNA synthesis implied by (recruitment × binding) reproduces the
condition-specific rna_synth_prob computed in step 4. This is what
translates "TF is active" into "transcription goes up by X" in the
online model.

Mathematical Model
------------------

Inputs:
- transcription.rna_synth_prob (per condition, from step 4).
- transcription_regulation: TF-to-target matrix + equilibrium Kd's.
- equilibrium, replication, mass, constants, molecule_ids,
  molecule_groups, bulk_molecules (for sanity-checked counts).
- cell_specs (per condition), conditions, condition_to_doubling_time.

Parameters:
- CVXPY solver: ECOS; warm-started per condition.

Calculation:
- For each RNA j and condition c, model:
    synth_prob[j, c] = basal[j] + Σ_tf  r[j, tf] · P[tf, c]
  where P[tf, c] is the probability TF is promoter-bound under c
  (derived from equilibrium with its ligand) and r[j, tf] is the
  recruitment strength.
- Fit r (non-negative) and adjust P so the predicted synth_prob
  matches the step-4 synth_prob in least-squares sense, with
  regularization keeping r sparse.
- fitPromoterBoundProbability does the heavy lifting; returns
  r_vector, r_columns, pPromoterBound[tf, c].

Outputs:
- transcription (mutated): basal_prob placeholder for step 7.
- transcription_regulation (mutated): pPromoterBound filled.
- cell_specs['basal'] gets r_vector / r_columns entries.
- pPromoterBound top-level dict: {(tf, condition): probability}.
"""

import time

from process_bigraph import Step

from v2ecoli.processes.parca.promoter_fitting import fitPromoterBoundProbability
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade


INPUT_PORTS = {
    'tick_5'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'molecule_ids':             'overwrite',
    'molecule_groups':          'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'sim_data_root':            'overwrite',
    'conditions':               'overwrite',
    'condition_to_doubling_time': 'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
    'tf_to_fold_change':        'overwrite',
    'tf_to_direction':          'overwrite',
    'condition_active_tfs':     'overwrite',
    'condition_inactive_tfs':   'overwrite',
    'cell_specs':               'overwrite',
}

OUTPUT_PORTS = {
    'tick_6'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'cell_specs':               'overwrite',
    # fitPromoterBoundProbability sets sim_data.pPromoterBound as a
    # dynamic top-level attr; propagate it so step 7 can read it.
    'pPromoterBound':           'overwrite',
}


class PromoterBindingStep(Step):
    """Step 6 — promoter_binding.  See module docstring."""

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = dict(state['cell_specs'])

        print("Fitting promoter binding")
        r_vector, r_columns = fitPromoterBoundProbability(sd, cell_specs)

        cell_specs.setdefault("basal", {})
        cell_specs["basal"]["r_vector"]  = r_vector
        cell_specs["basal"]["r_columns"] = r_columns

        print(f"  Step 6 (promoter_binding) completed in {time.time() - t0:.1f}s")
        return {
            'transcription':            sd.process.transcription,
            'transcription_regulation': sd.process.transcription_regulation,
            'cell_specs':               cell_specs,
            'pPromoterBound':           getattr(sd, 'pPromoterBound', {}),
            'tick_6': True,
        }
