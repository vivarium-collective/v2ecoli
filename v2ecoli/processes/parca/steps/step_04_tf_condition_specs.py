"""Step 4 — tf_condition_specs.  Repeat the basal fit under each TF condition.

For every transcription-factor active/inactive state, and for the
combined growth conditions (with_aa, acetate, succinate, no_oxygen),
apply the fold-change overrides to RNA expression and rerun the basal
fit to produce condition-specific synthesis probabilities.

Mathematical Model
------------------

Inputs:
- transcription post-basal (rna_expression, rna_synth_prob).
- tf_to_fold_change: {tf_id: {rna_id: log2_fc}} from the literature.
- tf_to_active_inactive_conditions: which TFs to perturb (pruned to 1
  in debug mode).
- conditions / condition_active_tfs / condition_inactive_tfs: which TFs
  are on/off in each combined condition.
- condition_to_doubling_time for each condition's growth rate.
- cell_specs['basal'] as the starting point.

Parameters:
- cpus: conditions run in a multiprocessing pool of this size.
- debug: restricts to one TF + the combined conditions.

Calculation:
- For each (tf, state) condition, apply 2^fold_change to rna_expression
  on the TF's target RNAs, then rerun expressionConverge to recover
  self-consistent rna_synth_prob.
- For each combined condition, overlay the active-TF fold changes then
  run the same convergence.
- Condition-specific bulk distributions are sampled from the fitted
  synthesis probabilities.

Outputs:
- transcription (mutated): rna_expression / rna_synth_prob /
  cistron_expression / fit_cistron_expression now carry one entry per
  fitted condition, not just 'basal'.
- cell_specs (mutated): one entry per condition with bulk_average,
  bulk_distribution, n_avg_copies, r_vector placeholder.
"""

import time

from process_bigraph import Step

from v2ecoli.processes.parca.wholecell.utils import parallelization

from v2ecoli.processes.parca.fitting import (
    apply_updates,
    expressionConverge,
    expressionFromConditionAndFoldChange,
)
from v2ecoli.processes.parca.steps._facade import make_sim_data_facade


INPUT_PORTS = {
    'tick_3'                            : 'overwrite',
    'transcription':            'sim_data.transcription',
    'translation':              'sim_data.translation',
    'metabolism':               'sim_data.metabolism',
    'rna_decay':                'sim_data.rna_decay',
    'complexation':             'sim_data.complexation',
    'equilibrium':              'sim_data.equilibrium',
    'two_component_system':     'sim_data.two_component_system',
    'transcription_regulation': 'sim_data.transcription_regulation',
    'replication':              'sim_data.replication',
    'mass':                     'sim_data.mass',
    'constants':                'sim_data.constants',
    'growth_rate_parameters':   'sim_data.growth_rate_parameters',
    'molecule_groups':          'overwrite',
    'molecule_ids':             'overwrite',
    'relation':                 'overwrite',
    'getter':                   'overwrite',
    'bulk_molecules':           'overwrite',
    'sim_data_root':            'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
    'conditions':                        'overwrite',
    'condition_to_doubling_time':        'overwrite',
    'tf_to_fold_change':                 'overwrite',
    'condition_active_tfs':              'overwrite',
    'condition_inactive_tfs':            'overwrite',
    'cell_specs':                        'overwrite',
}

OUTPUT_PORTS = {
    'tick_4'                            : 'overwrite',
    'transcription': 'sim_data.transcription',
    'cell_specs':    'overwrite',
}


class TfConditionSpecsStep(Step):
    """Step 4 — tf_condition_specs.  See module docstring for port wiring."""

    config_schema = {
        'variable_elongation_transcription':
            {'_type': 'boolean', '_default': True},
        'variable_elongation_translation':
            {'_type': 'boolean', '_default': False},
        'disable_ribosome_capacity_fitting':
            {'_type': 'boolean', '_default': False},
        'disable_rnapoly_capacity_fitting':
            {'_type': 'boolean', '_default': False},
        'cpus': {'_type': 'integer', '_default': 1},
    }

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        sd = make_sim_data_facade(state)
        cell_specs = dict(state['cell_specs'])
        cpus = parallelization.cpus(self.config.get('cpus', 1))

        # 1. Per-TF active / inactive fitting (parallelizable).
        tf_conditions = list(sorted(sd.tf_to_active_inactive_conditions))
        args = [
            (sd, tf,
             self.config.get('variable_elongation_transcription', True),
             self.config.get('variable_elongation_translation', False),
             self.config.get('disable_ribosome_capacity_fitting', False),
             self.config.get('disable_rnapoly_capacity_fitting', False))
            for tf in tf_conditions
        ]
        working = {}
        apply_updates(buildTfConditionCellSpecifications, args,
                      tf_conditions, working, cpus)

        # 2. Update transcription's per-condition expression dicts.
        for ckey in working:
            sd.process.transcription.rna_expression[ckey] = working[ckey]["expression"]
            sd.process.transcription.rna_synth_prob[ckey] = working[ckey]["synthProb"]
            sd.process.transcription.cistron_expression[ckey] = (
                working[ckey]["cistron_expression"])
            sd.process.transcription.fit_cistron_expression[ckey] = (
                working[ckey]["fit_cistron_expression"])

        # 3. Combined conditions (with_aa etc.) — further mutates transcription.
        buildCombinedConditionCellSpecifications(
            sd, working,
            self.config.get('variable_elongation_transcription', True),
            self.config.get('variable_elongation_translation', False),
            self.config.get('disable_ribosome_capacity_fitting', False),
            self.config.get('disable_rnapoly_capacity_fitting', False),
        )

        # 4. Merge into cell_specs.
        for label, spec in sorted(working.items()):
            entry = {
                "concDict":                  spec["concDict"],
                "expression":                spec["expression"],
                "synthProb":                 spec["synthProb"],
                "fit_cistron_expression":    spec["fit_cistron_expression"],
                "doubling_time":             spec["doubling_time"],
                "avgCellDryMassInit":        spec["avgCellDryMassInit"],
                "fitAvgSolubleTargetMolMass":spec["fitAvgSolubleTargetMolMass"],
                "bulkContainer":             spec["bulkContainer"],
            }
            if spec.get("cistron_expression") is not None:
                entry["cistron_expression"] = spec["cistron_expression"]
            cell_specs[label] = entry

        print(f"  Step 4 (tf_condition_specs) completed in {time.time() - t0:.1f}s")
        return {
            'transcription': sd.process.transcription,
            'cell_specs':    cell_specs,
        
            'tick_4': True,}


# ============================================================================
# Sub-functions (unchanged logic; take a sim_data-shaped object).
# ============================================================================

def buildTfConditionCellSpecifications(
    sim_data, tf,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """Per-TF active/inactive cell specs.  Returns a dict keyed by
    ``{tf}__active`` / ``{tf}__inactive``."""
    cell_specs = {}
    for choice in ["__active", "__inactive"]:
        conditionKey = tf + choice
        conditionValue = sim_data.conditions[conditionKey]

        fcData = {}
        if choice == "__active" and conditionValue != sim_data.conditions["basal"]:
            fcData = sim_data.tf_to_fold_change[tf]
        if choice == "__inactive" and conditionValue != sim_data.conditions["basal"]:
            fcDataTmp = sim_data.tf_to_fold_change[tf].copy()
            for key, value in fcDataTmp.items():
                fcData[key] = 1.0 / value
        expression, cistron_expression = expressionFromConditionAndFoldChange(
            sim_data.process.transcription, conditionValue["perturbations"], fcData,
        )

        concDict = (
            sim_data.process.metabolism.concentration_updates
              .concentrations_based_on_nutrients(media_id=conditionValue["nutrients"])
        )
        concDict.update(sim_data.mass.getBiomassAsConcentrations(
            sim_data.condition_to_doubling_time[conditionKey]))

        cell_specs[conditionKey] = {
            "concDict":      concDict,
            "expression":    expression,
            "doubling_time": sim_data.condition_to_doubling_time.get(
                conditionKey, sim_data.condition_to_doubling_time["basal"]),
        }

        (expression, synthProb, fit_cistron_expression, avgCellDryMassInit,
         fitAvgSolubleTargetMolMass, bulkContainer, concDict,
        ) = expressionConverge(
            sim_data,
            cell_specs[conditionKey]["expression"],
            cell_specs[conditionKey]["concDict"],
            cell_specs[conditionKey]["doubling_time"],
            sim_data.process.transcription.rna_data["Km_endoRNase"],
            conditionKey=conditionKey,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
        )
        cell_specs[conditionKey].update({
            "expression":                 expression,
            "synthProb":                  synthProb,
            "cistron_expression":         cistron_expression,
            "fit_cistron_expression":     fit_cistron_expression,
            "avgCellDryMassInit":         avgCellDryMassInit,
            "fitAvgSolubleTargetMolMass": fitAvgSolubleTargetMolMass,
            "bulkContainer":              bulkContainer,
        })
    return cell_specs


def buildCombinedConditionCellSpecifications(
    sim_data, cell_specs,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """Combined-condition cell specs.  Mutates cell_specs in place and
    updates sim_data.process.transcription per-condition dicts."""
    for conditionKey in sim_data.condition_active_tfs:
        if conditionKey == "basal" or not conditionKey:
            # Skip basal (already fit) and the empty-string placeholder
            # entry in the KB (perturbations is a bare "" for that one).
            continue

        fcData = {}
        conditionValue = sim_data.conditions[conditionKey]
        for tf in sim_data.condition_active_tfs[conditionKey]:
            for gene, fc in sim_data.tf_to_fold_change[tf].items():
                fcData[gene] = fcData.get(gene, 1) * fc
        for tf in sim_data.condition_inactive_tfs[conditionKey]:
            for gene, fc in sim_data.tf_to_fold_change[tf].items():
                fcData[gene] = fcData.get(gene, 1) / fc

        expression, cistron_expression = expressionFromConditionAndFoldChange(
            sim_data.process.transcription, conditionValue["perturbations"], fcData,
        )

        concDict = (
            sim_data.process.metabolism.concentration_updates
              .concentrations_based_on_nutrients(media_id=conditionValue["nutrients"])
        )
        concDict.update(sim_data.mass.getBiomassAsConcentrations(
            sim_data.condition_to_doubling_time[conditionKey]))

        cell_specs[conditionKey] = {
            "concDict":      concDict,
            "expression":    expression,
            "doubling_time": sim_data.condition_to_doubling_time.get(
                conditionKey, sim_data.condition_to_doubling_time["basal"]),
        }

        (expression, synthProb, fit_cistron_expression, avgCellDryMassInit,
         fitAvgSolubleTargetMolMass, bulkContainer, concDict,
        ) = expressionConverge(
            sim_data,
            cell_specs[conditionKey]["expression"],
            cell_specs[conditionKey]["concDict"],
            cell_specs[conditionKey]["doubling_time"],
            sim_data.process.transcription.rna_data["Km_endoRNase"],
            conditionKey=conditionKey,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
        )
        cell_specs[conditionKey].update({
            "expression":                 expression,
            "synthProb":                  synthProb,
            "cistron_expression":         cistron_expression,
            "fit_cistron_expression":     fit_cistron_expression,
            "avgCellDryMassInit":         avgCellDryMassInit,
            "fitAvgSolubleTargetMolMass": fitAvgSolubleTargetMolMass,
            "bulkContainer":              bulkContainer,
        })

        sim_data.process.transcription.rna_expression[conditionKey] = expression
        sim_data.process.transcription.rna_synth_prob[conditionKey] = synthProb
        sim_data.process.transcription.cistron_expression[conditionKey] = cistron_expression
        sim_data.process.transcription.fit_cistron_expression[conditionKey] = fit_cistron_expression
