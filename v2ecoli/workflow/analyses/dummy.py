"""Native port of vEcoli ``ecoli/analysis/multivariant/dummy.py``.

A column-drift canary: it asserts the sweep's history columns exactly match an
expected set, raising ``RuntimeError`` (fail-loud) on any drift.  The expected
set here is v2ecoli's history schema (176 columns), not vEcoli's — the two
schemas differ (e.g. v2ecoli splits ``bulk`` into ``bulk__id`` + ``bulk__count``
and uses ``global_time`` instead of ``time``), so the canary is re-baselined to
v2ecoli's actual output while preserving the analysis's purpose and behaviour.

Registered as ``"dummy"`` (scale: ``"multivariant"``).
"""

from __future__ import annotations

from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analysis import Analysis

# v2ecoli history schema snapshot (see DESCRIBE of the sweep parquet).
OUTPUT_COLUMN_NAMES: set[str] = {
    "agent_id", "bulk__count", "bulk__id", "experiment_id", "generation",
    "global_time", "lineage_seed", "variant",
    "listeners__atp__atp_allocated_initial", "listeners__atp__atp_requested",
    "listeners__enzyme_kinetics__actual_fluxes",
    "listeners__enzyme_kinetics__counts_to_molar",
    "listeners__enzyme_kinetics__enzyme_counts_init",
    "listeners__enzyme_kinetics__metabolite_counts_final",
    "listeners__enzyme_kinetics__metabolite_counts_init",
    "listeners__enzyme_kinetics__target_aa_conc",
    "listeners__enzyme_kinetics__target_fluxes",
    "listeners__enzyme_kinetics__target_fluxes_lower",
    "listeners__enzyme_kinetics__target_fluxes_upper",
    "listeners__fba_results__base_reaction_fluxes",
    "listeners__fba_results__catalyst_counts",
    "listeners__fba_results__coefficient", "listeners__fba_results__conc_updates",
    "listeners__fba_results__constrained_molecules",
    "listeners__fba_results__delta_metabolites",
    "listeners__fba_results__external_exchange_fluxes",
    "listeners__fba_results__homeostatic_objective_values",
    "listeners__fba_results__kinetic_objective_values",
    "listeners__fba_results__media_id", "listeners__fba_results__objective_value",
    "listeners__fba_results__reaction_fluxes",
    "listeners__fba_results__reduced_costs",
    "listeners__fba_results__shadow_prices",
    "listeners__fba_results__target_concentrations",
    "listeners__fba_results__translation_gtp",
    "listeners__fba_results__unconstrained_molecules",
    "listeners__fba_results__uptake_constraints",
    "listeners__growth_limits__aa_allocated", "listeners__growth_limits__aa_conc",
    "listeners__growth_limits__aa_count_diff", "listeners__growth_limits__aa_export",
    "listeners__growth_limits__aa_exporters", "listeners__growth_limits__aa_import",
    "listeners__growth_limits__aa_importers", "listeners__growth_limits__aa_in_media",
    "listeners__growth_limits__aa_pool_size",
    "listeners__growth_limits__aa_request_size", "listeners__growth_limits__aa_supply",
    "listeners__growth_limits__aa_supply_aa_conc",
    "listeners__growth_limits__aa_supply_enzymes_fwd",
    "listeners__growth_limits__aa_supply_enzymes_rev",
    "listeners__growth_limits__aa_supply_fraction_fwd",
    "listeners__growth_limits__aa_supply_fraction_rev",
    "listeners__growth_limits__aa_synthesis", "listeners__growth_limits__aas_used",
    "listeners__growth_limits__active_ribosome_allocated",
    "listeners__growth_limits__charged_trna_conc",
    "listeners__growth_limits__fraction_aa_to_elongate",
    "listeners__growth_limits__fraction_trna_charged",
    "listeners__growth_limits__net_charged", "listeners__growth_limits__ntp_allocated",
    "listeners__growth_limits__ntp_pool_size",
    "listeners__growth_limits__ntp_request_size", "listeners__growth_limits__ntp_used",
    "listeners__growth_limits__original_aa_supply",
    "listeners__growth_limits__ppgpp_conc", "listeners__growth_limits__rela_conc",
    "listeners__growth_limits__rela_syn", "listeners__growth_limits__ribosome_conc",
    "listeners__growth_limits__spot_conc", "listeners__growth_limits__spot_deg",
    "listeners__growth_limits__spot_deg_inhibited",
    "listeners__growth_limits__spot_syn", "listeners__growth_limits__synthetase_conc",
    "listeners__growth_limits__trna_charged",
    "listeners__growth_limits__uncharged_trna_conc",
    "listeners__mass__cell_mass", "listeners__mass__cytosol_mass",
    "listeners__mass__dna_mass", "listeners__mass__dry_mass",
    "listeners__mass__dry_mass_fold_change",
    "listeners__mass__expected_mass_fold_change",
    "listeners__mass__extracellular_mass", "listeners__mass__flagellum_mass",
    "listeners__mass__growth", "listeners__mass__inner_membrane_mass",
    "listeners__mass__instantaneous_growth_rate", "listeners__mass__mRna_mass",
    "listeners__mass__membrane_mass", "listeners__mass__outer_membrane_mass",
    "listeners__mass__periplasm_mass", "listeners__mass__pilus_mass",
    "listeners__mass__projection_mass", "listeners__mass__protein_mass",
    "listeners__mass__protein_mass_fold_change",
    "listeners__mass__protein_mass_fraction", "listeners__mass__rRna_mass",
    "listeners__mass__rna_mass", "listeners__mass__rna_mass_fold_change",
    "listeners__mass__rna_mass_fraction", "listeners__mass__smallMolecule_mass",
    "listeners__mass__small_molecule_fold_change", "listeners__mass__tRna_mass",
    "listeners__mass__volume", "listeners__mass__water_mass",
    "listeners__monomer_counts",
    "listeners__replication_data__fork_coordinates",
    "listeners__replication_data__fork_domains",
    "listeners__replication_data__fork_unique_index",
    "listeners__replication_data__free_DnaA_boxes",
    "listeners__replication_data__number_of_oric",
    "listeners__replication_data__total_DnaA_boxes",
    "listeners__ribosome_data__aa_count_in_sequence",
    "listeners__ribosome_data__aa_counts",
    "listeners__ribosome_data__actual_elongation_hist",
    "listeners__ribosome_data__actual_elongations",
    "listeners__ribosome_data__did_terminate",
    "listeners__ribosome_data__effective_elongation_rate",
    "listeners__ribosome_data__elongations_non_terminating_hist",
    "listeners__ribosome_data__mRNA_TU_index",
    "listeners__ribosome_data__n_ribosomes_on_each_mRNA",
    "listeners__ribosome_data__n_ribosomes_on_partial_mRNA_per_transcript",
    "listeners__ribosome_data__n_ribosomes_per_transcript",
    "listeners__ribosome_data__num_trpA_terminated",
    "listeners__ribosome_data__process_elongation_rate",
    "listeners__ribosome_data__protein_mass_on_polysomes",
    "listeners__ribosome_data__rRNA16S_init_prob",
    "listeners__ribosome_data__rRNA16S_initiated",
    "listeners__ribosome_data__rRNA23S_init_prob",
    "listeners__ribosome_data__rRNA23S_initiated",
    "listeners__ribosome_data__rRNA5S_init_prob",
    "listeners__ribosome_data__rRNA5S_initiated",
    "listeners__ribosome_data__rRNA_init_prob_TU",
    "listeners__ribosome_data__rRNA_initiated_TU",
    "listeners__ribosome_data__termination_loss",
    "listeners__ribosome_data__total_rRNA_init_prob",
    "listeners__ribosome_data__total_rRNA_initiated",
    "listeners__ribosome_data__translation_supply",
    "listeners__rna_counts__full_mRNA_cistron_counts",
    "listeners__rna_counts__full_mRNA_counts",
    "listeners__rna_counts__mRNA_cistron_counts",
    "listeners__rna_counts__mRNA_counts",
    "listeners__rna_counts__partial_mRNA_cistron_counts",
    "listeners__rna_counts__partial_mRNA_counts",
    "listeners__rna_counts__partial_rRNA_cistron_counts",
    "listeners__rna_counts__partial_rRNA_counts",
    "listeners__rna_synth_prob__actual_rna_synth_prob",
    "listeners__rna_synth_prob__actual_rna_synth_prob_per_cistron",
    "listeners__rna_synth_prob__bound_TF_coordinates",
    "listeners__rna_synth_prob__bound_TF_domains",
    "listeners__rna_synth_prob__bound_TF_indexes",
    "listeners__rna_synth_prob__expected_rna_init_per_cistron",
    "listeners__rna_synth_prob__gene_copy_number",
    "listeners__rna_synth_prob__n_actual_bound",
    "listeners__rna_synth_prob__n_bound_TF_per_TU",
    "listeners__rna_synth_prob__n_bound_TF_per_cistron",
    "listeners__rna_synth_prob__promoter_copy_number",
    "listeners__rna_synth_prob__target_rna_synth_prob",
    "listeners__rna_synth_prob__target_rna_synth_prob_per_cistron",
    "listeners__rna_synth_prob__total_rna_init",
    "listeners__rnap_data__active_rnap_coordinates",
    "listeners__rnap_data__active_rnap_domain_indexes",
    "listeners__rnap_data__active_rnap_n_bound_ribosomes",
    "listeners__rnap_data__active_rnap_on_stable_RNA_indexes",
    "listeners__rnap_data__active_rnap_unique_indexes",
    "listeners__rnap_data__actual_elongations",
    "listeners__rnap_data__did_initialize", "listeners__rnap_data__did_stall",
    "listeners__rnap_data__did_terminate", "listeners__rnap_data__rna_init_event",
    "listeners__rnap_data__rna_init_event_per_cistron",
    "listeners__rnap_data__termination_loss",
    "listeners__transcript_elongation_listener__attenuation_probability",
    "listeners__transcript_elongation_listener__count_NTPs_used",
    "listeners__transcript_elongation_listener__count_rna_synthesized",
    "listeners__transcript_elongation_listener__counts_attenuated",
}


class Dummy(Analysis):
    """Column-drift canary over the sweep history schema (multivariant)."""

    name = "dummy"
    scale = "multivariant"

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data=None,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        current = {
            r[0]
            for r in conn.sql(
                f"SELECT column_name FROM (DESCRIBE ({history_sql}))"
            ).fetchall()
        }
        if current != OUTPUT_COLUMN_NAMES:
            missing = OUTPUT_COLUMN_NAMES - current
            extra = current - OUTPUT_COLUMN_NAMES
            msg = ["Output column names mismatch detected!"]
            if extra:
                msg.append("\nNew columns (not in expected list):")
                msg += [f"  • {c}" for c in sorted(extra)]
            if missing:
                msg.append("\nMissing columns (expected but not found):")
                msg += [f"  • {c}" for c in sorted(missing)]
            msg.append(
                "\nUpdate OUTPUT_COLUMN_NAMES (and any analyses using renamed/"
                "removed columns) if this drift is intentional."
            )
            raise RuntimeError("\n".join(msg))
        text = (
            "Dummy analysis completed successfully.\n"
            "No issues detected with output column names "
            f"({len(current)} columns).\n"
        )
        return {"data": {"filename": "dummy_analysis.txt", "tsv": text}}
