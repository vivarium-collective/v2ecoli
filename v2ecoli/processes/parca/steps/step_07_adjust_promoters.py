"""Step 7 — adjust_promoters.  Back-solve ligand levels + RNAP recruitment.

Given the fitted ``pPromoterBound`` from step 6, work backwards: find
ligand concentrations and equilibrium reverse-rates that make the
equilibrium solver produce those binding probabilities, and pre-compute
the ``basal_prob`` / ``delta_prob`` columns the online RNAP recruitment
model consumes each timestep.

Mathematical Model
------------------

Inputs:
- pPromoterBound[tf, condition] from step 6.
- transcription, transcription_regulation, equilibrium, metabolism,
  replication, mass, constants, molecule_ids, molecule_groups,
  bulk_molecules.
- cell_specs, conditions, condition_to_doubling_time.

Calculation:
- For each TF with a ligand, solve the equilibrium Hill equation
  backwards: [L] = K_d · (P / (1 - P))^{1/n} given the target P.
- Where the forward equilibrium doesn't match, adjust the equilibrium
  reverse-rate constant to balance.
- basal_prob[j]  = pPromoterBound · r_vector  evaluated at basal.
- delta_prob[j, c] = (pPromoterBound[c] - pPromoterBound[basal]) ·
  r_vector, giving the per-condition deviation that the online model
  adds to basal_prob.

Outputs:
- transcription_regulation (mutated): basal_prob, delta_prob arrays.
- metabolism (mutated): molecule_set_amounts overrides for ligands.
- equilibrium (mutated): reverse_rates balanced against target P.
"""

import time

from process_bigraph import Step

from v2ecoli.processes.parca.promoter_fitting import (
    fitLigandConcentrations,
    calculateRnapRecruitment,
)
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade


INPUT_PORTS = {
    'tick_6'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'metabolism':               'sim_data.metabolism',
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
    # From step 6; fitLigandConcentrations reads sim_data.pPromoterBound.
    'pPromoterBound':           'overwrite',
}

OUTPUT_PORTS = {
    'tick_7'                            : 'overwrite',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'metabolism':               'sim_data.metabolism',
    'equilibrium':              'sim_data.equilibrium',
}


class AdjustPromotersStep(Step):
    """Step 7 — adjust_promoters.  See module docstring."""

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = state['cell_specs']

        fitLigandConcentrations(sd, cell_specs)
        basal_prob, delta_prob = calculateRnapRecruitment(sd, cell_specs)

        sd.process.transcription_regulation.basal_prob = basal_prob
        sd.process.transcription_regulation.delta_prob = delta_prob

        print(f"  Step 7 (adjust_promoters) completed in {time.time() - t0:.1f}s")
        return {
            'transcription_regulation': sd.process.transcription_regulation,
            'metabolism':               sd.process.metabolism,
            'equilibrium':              sd.process.equilibrium,
        
            'tick_7': True,}
