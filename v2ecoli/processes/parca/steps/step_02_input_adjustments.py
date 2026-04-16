"""Step 2 — input_adjustments.  Apply literature-curated overrides before fitting.

A handful of genes are known from experiment to need their RNA
expression, translation efficiency, or degradation rate multiplied by a
fixed factor before any fitting begins. This step reads those factors
from the ``adjustments`` dataclass and applies them in place.

Mathematical Model
------------------

Inputs:
- transcription.rna_expression (basal) and transcription.rna_deg_rates.
- translation.translation_efficiencies_by_monomer.
- adjustments: a lookup table of {gene_id: multiplier} for each kind of
  adjustment, shipped in ``flat/adjustments/``.
- tf_to_active_inactive_conditions (reduced to one entry in debug mode).

Parameters:
- debug (bool): when True, prunes tf_to_active_inactive_conditions to a
  single TF so step 4 only runs one condition-fit.

Calculation:
- adjust_translation_efficiencies: scalar multiplier per monomer id.
- balance_translation_efficiencies: renormalize so mean(eff) == 1.
- adjust_rna_expression: scalar multiplier per rna id, then renormalize
  so sum(expression) == 1.
- adjust_rna_deg_rates, adjust_protein_deg_rates: scalar multipliers.

Outputs:
- transcription (mutated): rna_expression + rna_deg_rates updated.
- translation (mutated): translation_efficiencies_by_monomer updated.
- tf_to_active_inactive_conditions (pruned when debug=True).
"""

import time

import numpy as np

from process_bigraph import Step


# ============================================================================
# Pure sub-functions (unchanged — take explicit numpy arrays)
# ============================================================================

def adjust_translation_efficiencies(monomer_ids, efficiencies, adjustments):
    """Multiply translation efficiencies by specified per-monomer factors.

    Args:
        monomer_ids: array of monomer IDs aligned with ``efficiencies``.
        efficiencies: numpy array (may be mutated in place; caller copies).
        adjustments: dict {monomer_id: adjustment_factor}.
    Returns:
        the adjusted numpy array.
    """
    for monomer_id, adjustment in adjustments.items():
        idx = np.where(monomer_ids == monomer_id)[0]
        efficiencies[idx] = efficiencies[idx] * adjustment
    return efficiencies


def balance_translation_efficiencies(monomer_ids, efficiencies, groups):
    """Average translation efficiencies across balanced groups.

    Args:
        monomer_ids: monomer IDs aligned with ``efficiencies``.
        efficiencies: numpy array.
        groups: list of lists — each sub-list is a set of monomer IDs to average.
    Returns:
        the adjusted numpy array.
    """
    for group in groups:
        idx = np.array([
            i for i, m in enumerate(monomer_ids) if m in group
        ])
        if len(idx) > 0:
            efficiencies[idx] = np.mean(efficiencies[idx])
    return efficiencies


def adjust_rna_expression(
    rna_ids, cistron_ids, rna_expression, adjustments, cistron_to_rna_indexes,
):
    """Apply per-cistron adjustments to RNA expression, renormalize.

    Args:
        rna_ids: RNA IDs aligned with ``rna_expression``.
        cistron_ids: cistron IDs.
        rna_expression: numpy array of basal RNA expression (mutated in place).
        adjustments: dict {cistron_id: adjustment_factor}.
        cistron_to_rna_indexes: dict {cistron_id: array of RNA indexes}.
    Returns:
        the adjusted (still-normalized) numpy array.
    """
    for cistron_id, adjustment in adjustments.items():
        rna_indexes = cistron_to_rna_indexes[cistron_id]
        rna_expression[rna_indexes] = rna_expression[rna_indexes] * adjustment
    rna_expression /= rna_expression.sum()
    return rna_expression


def adjust_rna_deg_rates(
    rna_ids, cistron_ids, rna_deg_rates, cistron_deg_rates,
    adjustments, cistron_to_rna_indexes,
):
    """Apply per-cistron degradation-rate adjustments to both the RNA and
    cistron arrays (Unum / structured-array aware, handled by caller).

    Returns:
        (new_rna_deg_rates, new_cistron_deg_rates) pair.
    """
    for cistron_id, adjustment in adjustments.items():
        rna_indexes = cistron_to_rna_indexes[cistron_id]
        rna_deg_rates[rna_indexes] = rna_deg_rates[rna_indexes] * adjustment
        cistron_idx = np.where(cistron_ids == cistron_id)[0]
        cistron_deg_rates[cistron_idx] = cistron_deg_rates[cistron_idx] * adjustment
    return rna_deg_rates, cistron_deg_rates


def adjust_protein_deg_rates(monomer_ids, rates, adjustments):
    """Apply per-monomer degradation-rate adjustments.

    Returns:
        the adjusted numpy array.
    """
    for monomer_id, adjustment in adjustments.items():
        idx = np.where(monomer_ids == monomer_id)[0]
        rates[idx] = rates[idx] * adjustment
    return rates


# ============================================================================
# Step class
# ============================================================================

INPUT_PORTS = {
    'tick_1'                            : 'overwrite',
    'transcription':                    'sim_data.transcription',
    'translation':                      'sim_data.translation',
    'adjustments':                      'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
}

OUTPUT_PORTS = {
    'tick_2'                            : 'overwrite',
    'transcription':                    'sim_data.transcription',
    'translation':                      'sim_data.translation',
    'tf_to_active_inactive_conditions': 'overwrite',
}


class InputAdjustmentsStep(Step):
    """Step 2 — input_adjustments.  See module docstring for port wiring."""

    config_schema = {
        'debug': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        transcription = state['transcription']
        translation   = state['translation']
        adjustments   = state['adjustments']
        tf_cond       = state['tf_to_active_inactive_conditions']

        # --- debug: optionally trim TF conditions ---
        tf_cond_out = None
        if self.config.get('debug', False):
            print(
                "  Step 2: debug mode — reducing tf_to_active_inactive_conditions"
                " to a single key"
            )
            first_key = next(iter(tf_cond))
            tf_cond_out = {first_key: tf_cond[first_key]}

        # --- translation efficiencies ---
        monomer_ids = translation.monomer_data['id']
        efficiencies = translation.translation_efficiencies_by_monomer.copy()
        efficiencies = adjust_translation_efficiencies(
            monomer_ids, efficiencies,
            dict(adjustments.translation_efficiencies_adjustments),
        )
        efficiencies = balance_translation_efficiencies(
            monomer_ids, efficiencies,
            list(adjustments.balanced_translation_efficiencies),
        )
        translation.translation_efficiencies_by_monomer[:] = efficiencies

        # --- RNA expression ---
        rna_ids     = transcription.rna_data['id']
        cistron_ids = transcription.cistron_data['id']
        cistron_to_rna_indexes = {
            cid: transcription.cistron_id_to_rna_indexes(cid)
            for cid in cistron_ids
        }
        new_rna_expr = adjust_rna_expression(
            rna_ids, cistron_ids,
            transcription.rna_expression['basal'].copy(),
            dict(adjustments.rna_expression_adjustments),
            cistron_to_rna_indexes,
        )
        transcription.rna_expression['basal'][:] = new_rna_expr

        # --- RNA + cistron degradation rates ---
        new_rna_deg, new_cistron_deg = adjust_rna_deg_rates(
            rna_ids, cistron_ids,
            transcription.rna_data.struct_array['deg_rate'].copy(),
            transcription.cistron_data.struct_array['deg_rate'].copy(),
            dict(adjustments.rna_deg_rates_adjustments),
            cistron_to_rna_indexes,
        )
        transcription.rna_data.struct_array['deg_rate'][:]     = new_rna_deg
        transcription.cistron_data.struct_array['deg_rate'][:] = new_cistron_deg

        # --- protein degradation rates ---
        new_prot_deg = adjust_protein_deg_rates(
            translation.monomer_data['id'],
            translation.monomer_data.struct_array['deg_rate'].copy(),
            dict(adjustments.protein_deg_rates_adjustments),
        )
        translation.monomer_data.struct_array['deg_rate'][:] = new_prot_deg

        print(f"  Step 2 (input_adjustments) completed in {time.time() - t0:.1f}s")

        out = {
            'transcription': transcription,
            'translation':   translation,
        }
        if tf_cond_out is not None:
            out['tf_to_active_inactive_conditions'] = tf_cond_out
        return out
