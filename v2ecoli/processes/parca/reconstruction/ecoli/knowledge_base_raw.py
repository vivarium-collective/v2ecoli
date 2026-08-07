"""
KnowledgeBase for Ecoli
Whole-cell knowledge base for Ecoli. Contains all raw, un-fit data processed
directly from CSV flat files.

"""

import io
import os
import json
from typing import List, Dict, Optional
import warnings

from v2ecoli.processes.parca.reconstruction.spreadsheets import read_tsv
from v2ecoli.processes.parca.wholecell.io import tsv
from v2ecoli.processes.parca.wholecell.utils import units  # used by eval()
from v2ecoli.processes.parca.reconstruction.ecoli.sources import relpath_to_key

FLAT_DIR = os.path.join(os.path.dirname(__file__), "flat")
LIST_OF_DICT_FILENAMES = [
    "amino_acid_export_kms.tsv",
    "amino_acid_export_kms_removed.tsv",
    "amino_acid_pathways.tsv",
    "amino_acid_uptake_rates.tsv",
    "amino_acid_uptake_rates_removed.tsv",
    "biomass.tsv",
    "compartments.tsv",
    "complexation_reactions.tsv",
    "complexation_reactions_added.tsv",
    "complexation_reactions_modified.tsv",
    "complexation_reactions_removed.tsv",
    "disabled_kinetic_reactions.tsv",
    "dna_sites.tsv",
    "dry_mass_composition.tsv",
    "endoRNases.tsv",
    "equilibrium_reaction_rates.tsv",
    "equilibrium_reactions.tsv",
    "equilibrium_reactions_added.tsv",
    "equilibrium_reactions_removed.tsv",
    "fold_changes.tsv",
    "fold_changes_nca.tsv",
    "fold_changes_removed.tsv",
    "footprint_sizes.tsv",
    "genes.tsv",
    "growth_rate_dependent_parameters.tsv",
    "linked_metabolites.tsv",
    "metabolic_reactions.tsv",
    "metabolic_reactions_added.tsv",
    "metabolic_reactions_modified.tsv",
    "metabolic_reactions_removed.tsv",
    "metabolism_kinetics.tsv",
    "metabolite_concentrations.tsv",
    "metabolite_concentrations_removed.tsv",
    "metabolites.tsv",
    "metabolites_added.tsv",
    "modified_proteins.tsv",
    "molecular_weight_keys.tsv",
    "ppgpp_fc.tsv",
    "ppgpp_regulation.tsv",
    "ppgpp_regulation_added.tsv",
    "ppgpp_regulation_removed.tsv",
    "protein_half_lives_measured.tsv",
    "protein_half_lives_n_end_rule.tsv",
    "protein_half_lives_pulsed_silac.tsv",
    "proteins.tsv",
    "relative_metabolite_concentrations.tsv",
    "rna_half_lives.tsv",
    "rna_maturation_enzymes.tsv",
    "rnas.tsv",
    "secretions.tsv",
    "sequence_motifs.tsv",
    "transcription_factors.tsv",
    # "transcription_units.tsv",  # special cased in the constructor
    "transcription_units_added.tsv",
    "transcription_units_removed.tsv",
    "transcription_units_modified.tsv",
    "transcriptional_attenuation.tsv",
    "transcriptional_attenuation_removed.tsv",
    "tf_one_component_bound.tsv",
    "translation_efficiency.tsv",
    "trna_charging_reactions.tsv",
    "trna_charging_reactions_added.tsv",
    "trna_charging_reactions_removed.tsv",
    "two_component_systems.tsv",
    "two_component_system_templates.tsv",
    os.path.join("mass_fractions", "glycogen_fractions.tsv"),
    os.path.join("mass_fractions", "ion_fractions.tsv"),
    os.path.join("mass_fractions", "LPS_fractions.tsv"),
    os.path.join("mass_fractions", "lipid_fractions.tsv"),
    os.path.join("mass_fractions", "murein_fractions.tsv"),
    os.path.join("mass_fractions", "soluble_fractions.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_0p4.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_0p7.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_1p6.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_1p07.tsv"),
    os.path.join("trna_data", "trna_ratio_to_16SrRNA_2p5.tsv"),
    os.path.join("trna_data", "trna_growth_rates.tsv"),
    os.path.join("rna_seq_data", "rnaseq_rsem_tpm_mean.tsv"),
    os.path.join("rna_seq_data", "rnaseq_rsem_tpm_std.tsv"),
    os.path.join("rna_seq_data", "rnaseq_seal_rpkm_mean.tsv"),
    os.path.join("rna_seq_data", "rnaseq_seal_rpkm_std.tsv"),
    os.path.join("rrna_options", "remove_rrff", "genes_removed.tsv"),
    os.path.join("rrna_options", "remove_rrff", "rnas_removed.tsv"),
    os.path.join("rrna_options", "remove_rrff", "transcription_units_modified.tsv"),
    os.path.join(
        "rrna_options", "remove_rrna_operons", "transcription_units_added.tsv"
    ),
    os.path.join(
        "rrna_options", "remove_rrna_operons", "transcription_units_removed.tsv"
    ),
    os.path.join("condition", "tf_condition.tsv"),
    os.path.join("condition", "condition_defs.tsv"),
    os.path.join("condition", "environment_molecules.tsv"),
    os.path.join("condition", "timelines_def.tsv"),
    os.path.join("condition", "media_recipes.tsv"),
    os.path.join("condition", "media", "5X_supplement_EZ.tsv"),
    os.path.join("condition", "media", "MIX0-55.tsv"),
    os.path.join("condition", "media", "MIX0-57.tsv"),
    os.path.join("condition", "media", "MIX0-58.tsv"),
    os.path.join("condition", "media", "MIX0-844.tsv"),
    os.path.join("base_codes", "amino_acids.tsv"),
    os.path.join("base_codes", "dntp.tsv"),
    os.path.join("base_codes", "nmp.tsv"),
    os.path.join("base_codes", "ntp.tsv"),
    os.path.join("adjustments", "amino_acid_pathways.tsv"),
    os.path.join("adjustments", "balanced_translation_efficiencies.tsv"),
    os.path.join("adjustments", "translation_efficiencies_adjustments.tsv"),
    os.path.join("adjustments", "rna_expression_adjustments.tsv"),
    os.path.join("adjustments", "rna_deg_rates_adjustments.tsv"),
    os.path.join("adjustments", "protein_deg_rates_adjustments.tsv"),
    os.path.join("adjustments", "relative_metabolite_concentrations_changes.tsv"),
]
SEQUENCE_FILE = "sequence.fasta"
LIST_OF_PARAMETER_FILENAMES = [
    "dna_supercoiling.tsv",
    "parameters.tsv",
    "mass_parameters.tsv",
    os.path.join("new_gene_data", "new_gene_baseline_expression_parameters.tsv"),
]

REMOVED_DATA = {
    "amino_acid_export_kms": "amino_acid_export_kms_removed",
    "amino_acid_uptake_rates": "amino_acid_uptake_rates_removed",
    "complexation_reactions": "complexation_reactions_removed",
    "equilibrium_reactions": "equilibrium_reactions_removed",
    "fold_changes": "fold_changes_removed",
    "fold_changes_nca": "fold_changes_removed",
    "metabolic_reactions": "metabolic_reactions_removed",
    "metabolite_concentrations": "metabolite_concentrations_removed",
    "ppgpp_regulation": "ppgpp_regulation_removed",
    "transcriptional_attenuation": "transcriptional_attenuation_removed",
    "trna_charging_reactions": "trna_charging_reactions_removed",
}
MODIFIED_DATA = {
    "complexation_reactions": "complexation_reactions_modified",
    "metabolic_reactions": "metabolic_reactions_modified",
}

ADDED_DATA = {
    "complexation_reactions": "complexation_reactions_added",
    "equilibrium_reactions": "equilibrium_reactions_added",
    "metabolic_reactions": "metabolic_reactions_added",
    "metabolites": "metabolites_added",
    "ppgpp_regulation": "ppgpp_regulation_added",
    "trna_charging_reactions": "trna_charging_reactions_added",
}


class DataStore(object):
    def __init__(self):
        pass


class KnowledgeBaseEcoli(object):
    """KnowledgeBaseEcoli"""

    def __init__(
        self,
        operons_on: bool,
        remove_rrna_operons: bool,
        remove_rrff: bool,
        stable_rrna: bool,
        new_genes_option: str = "off",
        gene_deletions: Optional[List[str]] = None,
        bundle=None,
    ):
        if bundle is None:
            from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle
            bundle = SourceBundle()
        self._bundle = bundle
        self.operons_on = operons_on
        self.stable_rrna = stable_rrna
        self.new_genes_option = new_genes_option
        self.gene_deletions: List[str] = list(gene_deletions or [])

        if not operons_on and remove_rrna_operons:
            warnings.warn(
                "Setting the 'remove_rrna_operons' option to 'True'"
                " has no effect on the simulations when the 'operon'"
                " option is set to 'off'."
            )

        self.compartments: List[
            dict
        ] = []  # mypy can't track setattr(self, attr_name, rows)
        self.transcription_units: List[dict] = []

        # Make copies to prevent issues with sticky global variables when
        # running multiple operon workflows through Fireworks
        self.list_of_dict_filenames: List[str] = LIST_OF_DICT_FILENAMES.copy()
        self.list_of_parameter_filenames: List[str] = LIST_OF_PARAMETER_FILENAMES.copy()
        self.removed_data: Dict[str, str] = REMOVED_DATA.copy()
        self.modified_data: Dict[str, str] = MODIFIED_DATA.copy()
        self.added_data: Dict[str, str] = ADDED_DATA.copy()

        self.new_gene_added_data: Dict[str, str] = {}
        self.parameter_file_attribute_names: List[str] = [
            os.path.splitext(os.path.basename(filename))[0]
            for filename in self.list_of_parameter_filenames
        ]

        if self.operons_on:
            self.list_of_dict_filenames.append("transcription_units.tsv")
            if remove_rrna_operons:
                # Use alternative file with all rRNA transcription units if
                # remove_rrna_operons option was used
                self.removed_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrna_operons.transcription_units_removed",
                    }
                )
                self.added_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrna_operons.transcription_units_added",
                    }
                )
            else:
                self.removed_data.update(
                    {
                        "transcription_units": "transcription_units_removed",
                    }
                )
                self.added_data.update(
                    {
                        "transcription_units": "transcription_units_added",
                    }
                )
                self.modified_data.update(
                    {
                        "transcription_units": "transcription_units_modified",
                    }
                )

        if remove_rrff:
            self.list_of_parameter_filenames.append(
                os.path.join(
                    "rrna_options", "remove_rrff", "mass_parameters_modified.tsv"
                )
            )
            self.removed_data.update(
                {
                    "genes": "rrna_options.remove_rrff.genes_removed",
                    "rnas": "rrna_options.remove_rrff.rnas_removed",
                }
            )
            self.modified_data.update(
                {
                    "mass_parameters": "rrna_options.remove_rrff.mass_parameters_modified",
                }
            )
            if self.operons_on:
                self.modified_data.update(
                    {
                        "transcription_units": "rrna_options.remove_rrff.transcription_units_modified",
                    }
                )

        if self.new_genes_option != "off":
            new_gene_subdir = new_genes_option
            new_gene_path = os.path.join("new_gene_data", new_gene_subdir)
            if self._bundle is not None:
                assert self._bundle.keys_with_prefix(
                    f"new_gene_data__{new_gene_subdir}__"
                ), "This new_genes_data subdirectory is invalid."
            else:
                assert os.path.isdir(os.path.join(FLAT_DIR, new_gene_path)), (
                    "This new_genes_data subdirectory is invalid."
                )
            nested_attr = "new_gene_data." + new_gene_subdir + "."

            # These files do not need to be joined to existing files
            self.list_of_dict_filenames.append(
                os.path.join(new_gene_path, "insertion_location.tsv")
            )
            self.list_of_dict_filenames.append(
                os.path.join(new_gene_path, "gene_sequences.tsv")
            )

            # These files need to be joined to existing files
            new_gene_shared_files = [
                "genes",
                "rnas",
                "proteins",
                "rna_half_lives",
                "protein_half_lives_measured",
            ]
            for f in new_gene_shared_files:
                file_path = os.path.join(new_gene_path, f + ".tsv")
                # If these files are empty, fill in with default values at a
                # later point
                if self._bundle is not None:
                    present = self._bundle.has_key(relpath_to_key(file_path))
                else:
                    present = os.path.isfile(os.path.join(FLAT_DIR, file_path))
                assert present, (
                    f"File {f}.tsv must be present in the new_genes_data"
                    f" subdirectory {new_gene_subdir}."
                )
                self.list_of_dict_filenames.append(file_path)
                self.new_gene_added_data.update({f: nested_attr + f})

            rnaseq_path = os.path.join(new_gene_path, "rnaseq_rsem_tpm_mean.tsv")
            if (self._bundle.has_key(relpath_to_key(rnaseq_path))
                    if self._bundle is not None
                    else os.path.isfile(os.path.join(FLAT_DIR, rnaseq_path))):
                self.list_of_dict_filenames.append(rnaseq_path)
                self.new_gene_added_data.update(
                    {
                        "rna_seq_data.rnaseq_rsem_tpm_mean": nested_attr
                        + "rnaseq_rsem_tpm_mean"
                    }
                )

        # Load raw data from TSV files
        for filename in self.list_of_dict_filenames:
            self._load_tsv(filename, self._resolve(filename))

        for filename in self.list_of_parameter_filenames:
            self._load_parameters(filename, self._resolve(filename))

        self.genome_sequence = self._load_sequence(self._resolve(SEQUENCE_FILE))

        self._prune_data()

        self._join_data()
        self._modify_data()

        for gene_id in self.gene_deletions:
            self._delete_gene(gene_id)

        if self.new_genes_option != "off":
            self._check_new_gene_ids(nested_attr)

            insert_pos = self._update_gene_insertion_location(nested_attr)

            insertion_sequence = self._get_new_gene_sequence(nested_attr)

            insert_end = self._update_gene_locations(nested_attr, insert_pos)
            self.new_gene_added_data.update({"genes": nested_attr + "genes"})

            self.genome_sequence = (
                self.genome_sequence[:insert_pos]
                + insertion_sequence
                + self.genome_sequence[insert_pos:]
            )
            assert self.genome_sequence[insert_pos:insert_end] == insertion_sequence

            self.added_data = self.new_gene_added_data
            self._join_data()

    def _resolve(self, rel_path):
        return self._bundle.resolve_relpath(rel_path)

    def _load_tsv(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        setattr(path, attr_name, [])

        rows = read_tsv(str(abs_path))
        setattr(path, attr_name, rows)

    def _load_sequence(self, file_path):
        from Bio import SeqIO

        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                return record.seq

    def _load_parameters(self, rel_path, abs_path):
        path = self
        parts = rel_path.replace(os.sep, "/").split("/")
        for sub_path in parts[:-1]:
            if not hasattr(path, sub_path):
                setattr(path, sub_path, DataStore())
            path = getattr(path, sub_path)
        attr_name = parts[-1].split(".")[0]
        param_dict = {}

        with io.open(str(abs_path), "rb") as csvfile:
            reader = tsv.dict_reader(csvfile)
            for row in reader:
                value = json.loads(row["value"])
                if row["units"] != "":
                    unit = eval(row["units"])  # risky!
                    unit = units.getUnit(unit)  # strip
                    value = value * unit
                param_dict[row["name"]] = value

        setattr(path, attr_name, param_dict)

    def _prune_data(self):
        """
        Remove rows that are specified to be removed. Data will only be removed
        if all data in a row in the file specifying rows to be removed matches
        the same columns in the raw data file.
        """

        # Check each pair of files to be removed
        for data_attr, attr_to_remove in self.removed_data.items():
            # Build the set of data to identify rows to be removed
            data_to_remove = getattr(self, attr_to_remove.split(".")[0])
            for attr in attr_to_remove.split(".")[1:]:
                data_to_remove = getattr(data_to_remove, attr)
            removed_cols = list(data_to_remove[0].keys())
            ids_to_remove = set()
            for row in data_to_remove:
                ids_to_remove.add(tuple([row[col] for col in removed_cols]))

            # Remove any matching rows
            data = getattr(self, data_attr)
            n_entries = len(data)
            removed_ids = set()
            for i, row in enumerate(data[::-1]):
                checked_id = tuple([row[col] for col in removed_cols])
                if checked_id in ids_to_remove:
                    data.pop(n_entries - i - 1)
                    removed_ids.add(checked_id)

            # Print warnings for entries that were marked to be removed that
            # does not exist in the original data file. Fold changes are
            # excluded since the original entries are split between two files.
            if not data_attr.startswith("fold_changes"):
                for unremoved_id in ids_to_remove - removed_ids:
                    print(
                        f"Warning: Could not remove row {unremoved_id} "
                        f"in flat file {data_attr} because the row does not "
                        f"exist."
                    )

    def _join_data(self):
        """
        Add rows that are specified in additional files. Data will only be added
        if all the loaded columns from both datasets match.
        """

        # Join data for each file with data to be added
        for data_attr, attr_to_add in self.added_data.items():
            # Get datasets to join
            data = getattr(self, data_attr.split(".")[0])
            for attr in data_attr.split(".")[1:]:
                data = getattr(data, attr)

            added_data = getattr(self, attr_to_add.split(".")[0])
            for attr in attr_to_add.split(".")[1:]:
                added_data = getattr(added_data, attr)

            if added_data:  # Some new gene additional files may be empty
                # Check columns are the same for each dataset
                col_diff = set(data[0].keys()).symmetric_difference(
                    added_data[0].keys()
                )
                if col_diff:
                    raise ValueError(
                        f"Could not join datasets {data_attr} and {attr_to_add} "
                        f"because columns do not match (different columns: {col_diff})."
                    )

            # Join datasets
            for row in added_data:
                data.append(row)

    def _modify_data(self):
        """
        Modify entires in rows that are specified to be modified. Rows must be
        identified by their entries in the first column (usually the ID column).
        """
        # Check each pair of files to be modified
        for data_attr, modify_attr in self.modified_data.items():
            # Build the set of data to identify rows to be modified
            data_to_modify = getattr(self, modify_attr.split(".")[0])
            for attr in modify_attr.split(".")[1:]:
                data_to_modify = getattr(data_to_modify, attr)
            data = getattr(self, data_attr)

            # If modifying a parameter file, replace values in dictionary
            if data_attr in self.parameter_file_attribute_names:
                for key, value in data_to_modify.items():
                    if key not in data:
                        raise ValueError(
                            f"Could not modify data {data_attr}"
                            f"with {modify_attr} because the name {key} does "
                            f"not exist in {data_attr}."
                        )

                    data[key] = value

            # If modifying a table file, replace rows
            else:
                id_col_name = list(data_to_modify[0].keys())[0]

                id_to_modified_cols = {}
                for row in data_to_modify:
                    id_to_modified_cols[row[id_col_name]] = row

                # Modify any matching rows with identical IDs
                if list(data[0].keys())[0] != id_col_name:
                    raise ValueError(
                        f"Could not modify data {data_attr} with "
                        f"{modify_attr} because the names of the first columns "
                        f"do not match."
                    )

                modified_entry_ids = set()
                for i, row in enumerate(data):
                    if row[id_col_name] in id_to_modified_cols:
                        modified_cols = id_to_modified_cols[row[id_col_name]]
                        for col_name in data[i]:
                            if col_name in modified_cols:
                                data[i][col_name] = modified_cols[col_name]
                        modified_entry_ids.add(row[id_col_name])

                # Check for entries in modification data that do not exist in
                # original data
                id_diff = set(id_to_modified_cols.keys()).symmetric_difference(
                    modified_entry_ids
                )
                if id_diff:
                    raise ValueError(
                        f"Could not modify data {data_attr} with "
                        f"{modify_attr} because of one or more entries in "
                        f"{modify_attr} that do not exist in {data_attr} "
                        f"(nonexistent entries: {id_diff})."
                    )

    def _check_new_gene_ids(self, nested_attr):
        """
        Check to ensure each new gene, RNA, and protein id starts with NG.
        """
        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        new_RNA_data = getattr(nested_data, "rnas")
        new_protein_data = getattr(nested_data, "proteins")

        for row in new_genes_data:
            assert row["id"].startswith("NG"), "ids of new genes must start with NG"
        for row in new_RNA_data:
            assert row["id"].startswith("NG"), "ids of new gene rnas must start with NG"
        for row in new_protein_data:
            assert row["id"].startswith("NG"), (
                "ids of new gene proteins must start with NG"
            )
        return

    def _update_gene_insertion_location(self, nested_attr):
        """
        Update insertion location of new genes to prevent conflicts.
        """

        genes_data = getattr(self, "genes")
        tu_data = getattr(self, "transcription_units")
        dna_sites_data = getattr(self, "dna_sites")

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        insert_loc_data = getattr(nested_data, "insertion_location")

        assert len(insert_loc_data) == 1, (
            "each noncontiguous insertion should be in its own directory"
        )
        insert_pos = insert_loc_data[0]["insertion_pos"]

        if not tu_data:
            # Check if specified insertion location is in another gene
            data_to_check = genes_data.copy()
        else:
            # Check if specified insertion location is in a transcription unit
            data_to_check = tu_data.copy()

        # Add important DNA sites to the list of locations to check
        # TODO: Check for other DNA sites if we include any in the future
        sites_data_to_check = [
            site
            for site in dna_sites_data
            if site["common_name"] == "oriC" or site["common_name"] == "TerC"
        ]
        data_to_check += sites_data_to_check

        conflicts = [
            row
            for row in data_to_check
            if ((row["left_end_pos"] is not None) and (row["left_end_pos"] != ""))
            and ((row["right_end_pos"] is not None) and (row["left_end_pos"] != ""))
            and (row["left_end_pos"] < insert_pos)
            and (row["right_end_pos"] >= insert_pos)
        ]
        # Change insertion location to after conflicts
        if conflicts:
            shift = max([sub["right_end_pos"] for sub in conflicts]) - insert_pos + 1
            insert_pos = insert_pos + shift

        return insert_pos

    def _update_global_coordinates(self, data, insert_pos, insert_len):
        """
        Updates the left and right end positions for all elements in data if
        their positions will be impacted by the new gene insertion.

        Args:
            data: Data attribute to update
            insert_pos: Location of new gene insertion
            insert_len: Length of new gene insertion

        """
        for row in data:
            left = row["left_end_pos"]
            right = row["right_end_pos"]
            if (
                (left is not None and left != "")
                and (right is not None and right != "")
                and left >= insert_pos
            ):
                row.update({"left_end_pos": left + insert_len})
                row.update({"right_end_pos": right + insert_len})

    def _update_gene_locations(self, nested_attr, insert_pos):
        """
        Modify positions of original genes based upon the insertion location
        of new genes. Returns end position of the gene insertion.
        """

        genes_data = getattr(self, "genes")
        tu_data = getattr(self, "transcription_units")
        dna_sites_data = getattr(self, "dna_sites")

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        new_genes_data = sorted(new_genes_data, key=lambda d: d["left_end_pos"])

        for i in range(len(new_genes_data) - 1):
            assert (
                new_genes_data[i + 1]["left_end_pos"]
                == new_genes_data[i]["right_end_pos"] + 1
            ), "gaps in new gene insertions are not supported at this time"

        insert_end = new_genes_data[-1]["right_end_pos"] + insert_pos
        insert_len = insert_end - insert_pos

        # Update global positions of original genes
        self._update_global_coordinates(genes_data, insert_pos, insert_len)

        # Update global positions of transcription units
        if tu_data:
            self._update_global_coordinates(tu_data, insert_pos, insert_len)

        # Update DNA site positions
        # (including the origin and terminus of replication)
        self._update_global_coordinates(dna_sites_data, insert_pos, insert_len)

        # Change relative insertion positions to global in reference genome
        for row in new_genes_data:
            left = row["left_end_pos"]
            right = row["right_end_pos"]
            row.update({"left_end_pos": left + insert_pos})
            row.update({"right_end_pos": right + insert_pos})

        return insert_end

    def _get_new_gene_sequence(self, nested_attr):
        """
        Determine genome sequnce for insertion using the sequences and
        relative locations of the new genes.
        """
        from Bio import Seq

        nested_data = getattr(self, nested_attr[:-1].split(".")[0])
        for attr in nested_attr[:-1].split(".")[1:]:
            nested_data = getattr(nested_data, attr)

        new_genes_data = getattr(nested_data, "genes")
        seq_data = getattr(nested_data, "gene_sequences")

        insertion_seq = Seq.Seq("")
        new_genes_data = sorted(new_genes_data, key=lambda d: d["left_end_pos"])
        assert new_genes_data[0]["left_end_pos"] == 1, (
            "first gene in new sequence must start at relative coordinate 1"
        )

        for gene in new_genes_data:
            if gene["direction"] == "+":
                seq_row = next(
                    (row for row in seq_data if row["id"] == gene["id"]), None
                )
                seq_string = seq_row["gene_seq"]
                seq_addition = Seq.Seq(seq_string)
                insertion_seq += seq_addition
            else:
                seq_row = next(
                    (row for row in seq_data if row["id"] == gene["id"]), None
                )
                seq_string = seq_row["gene_seq"]
                seq_addition = Seq.Seq(seq_string)
                insertion_seq += seq_addition.reverse_complement()

            assert len(seq_addition) == (
                gene["right_end_pos"] - gene["left_end_pos"] + 1
            ), (
                "left and right end positions must agree with actual "
                "sequence length for " + gene["id"]
            )

        return insertion_seq

    # --- Chromosome-level gene deletion ------------------------------------
    #
    # COORDINATE CONVENTION: left_end_pos / right_end_pos are 1-based and
    # INCLUSIVE, matching the EcoCyc-derived flat files. A feature spanning
    # [L, R] occupies genome_sequence[L - 1 : R] (see
    # v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/
    # getter_functions.py:194, which slices exactly that way) and has length
    # R - L + 1. Every slice and comparison
    # below assumes this; getting it wrong is silent, because feature LENGTHS
    # stay correct under an off-by-one while the SEQUENCE shifts.

    def _delete_gene(self, del_gene_id):
        """
        Delete a gene from the chromosome: splice it out of the genome
        sequence, null its own coordinates, detach it from the transcription
        units that carry it, and shift every downstream feature left by the
        deleted length.

        Args:
            del_gene_id: id of the gene to delete
        """
        genes_data = getattr(self, "genes")
        tus_data = getattr(self, "transcription_units")

        gene_data = next(
            (gene for gene in genes_data if gene["id"] == del_gene_id), None
        )
        assert gene_data is not None, (
            f"Cannot delete {del_gene_id}: no such gene in the knowledge base."
        )

        del_left_pos = gene_data["left_end_pos"]
        del_right_pos = gene_data["right_end_pos"]
        assert self._has_coordinates(gene_data), (
            f"Cannot delete {del_gene_id}: it has no coordinates (it may "
            f"already have been deleted)."
        )
        del_len = del_right_pos - del_left_pos + 1

        # Splice the gene out of the genome sequence. 1-based inclusive
        # coordinates [L, R] are genome_sequence[L - 1 : R], so the flanks
        # to KEEP are [:L - 1] and [R:].
        original_length = len(self.genome_sequence)
        self.genome_sequence = (
            self.genome_sequence[: del_left_pos - 1]
            + self.genome_sequence[del_right_pos:]
        )
        assert len(self.genome_sequence) == original_length - del_len

        # Detach the gene from the transcription units that carry it. A TU
        # left with no genes has no meaningful coordinates.
        for tu_data in tus_data:
            if del_gene_id not in tu_data["genes"]:
                continue
            if len(tu_data["genes"]) == 1:
                tu_data.update({"left_end_pos": None, "right_end_pos": None})
            else:
                genes_in_tu = [g for g in tu_data["genes"] if g != del_gene_id]
                tu_data.update({"genes": genes_in_tu})
                self._annotate_removed(tu_data, del_gene_id)

        # Null the deleted gene's own coordinates. Doing this BEFORE the
        # coordinate update is what keeps it out of the containment branch
        # below (it exits at the no-coordinate-data guard instead).
        gene_data.update({"left_end_pos": None, "right_end_pos": None})

        self._update_gene_locations_for_deletion(
            del_gene_id, del_left_pos, del_right_pos
        )

    def _update_gene_locations_for_deletion(
        self, del_gene_id, del_left_pos, del_right_pos
    ):
        """
        Modify positions of genes, transcription units, and DNA sites based
        upon the location of a deleted gene.
        """
        genes_data = getattr(self, "genes")
        tu_data = getattr(self, "transcription_units")
        dna_sites_data = getattr(self, "dna_sites")

        # Update global positions of original genes
        self._update_global_coordinates_for_deletion(
            genes_data, "gene", del_gene_id, del_left_pos, del_right_pos
        )

        # Update global positions of transcription units
        if tu_data:
            self._update_global_coordinates_for_deletion(
                tu_data, "tu", del_gene_id, del_left_pos, del_right_pos
            )

        # Update DNA site positions
        # (including the origin and terminus of replication)
        self._update_global_coordinates_for_deletion(
            dna_sites_data, "dna_site", del_gene_id, del_left_pos, del_right_pos
        )

    @staticmethod
    def _has_coordinates(row):
        """True when a row carries usable left AND right coordinates."""
        left = row["left_end_pos"]
        right = row["right_end_pos"]
        return not (
            left is None or right is None or left == "" or right == ""
        )

    @staticmethod
    def _classify_against_deletion(left, right, del_left_pos, del_right_pos):
        """
        Classify a feature's position relative to a deletion. TOTAL over all
        (left <= right, del_left_pos <= del_right_pos) — every input returns
        exactly one of six labels, so callers need no fallthrough case.

        Returns one of:
            'before'        entirely upstream; unaffected
            'after'         entirely downstream; shifts left by del_len
            'contained'     wholly inside the deletion; removed with it
            'spans'         starts before and ends after; loses its middle
            'overlaps_left' starts before, ends inside; truncated at the cut
            'overlaps_right' starts inside, ends after; 5' portion removed
        """
        if right < del_left_pos:
            return "before"
        if left > del_right_pos:
            return "after"
        # Everything below intersects the deletion.
        if left >= del_left_pos and right <= del_right_pos:
            return "contained"
        if left < del_left_pos and right > del_right_pos:
            return "spans"
        if left < del_left_pos:
            return "overlaps_left"
        return "overlaps_right"

    def _update_global_coordinates_for_deletion(
        self, data, data_type, del_gene_id, del_left_pos, del_right_pos
    ):
        """
        Updates the left and right positions for all elements in data if
        their positions will be impacted by the gene deletion. Features lying
        wholly inside the deletion are removed from data.

        Args:
            data: Data attribute to update (mutated in place)
            data_type: One of 'gene', 'tu', 'dna_site' — controls messaging
            del_gene_id: id of the gene being deleted
            del_left_pos: 1-based inclusive left end of the deletion
            del_right_pos: 1-based inclusive right end of the deletion
        """
        del_len = del_right_pos - del_left_pos + 1
        # Collect removals rather than mutating `data` mid-iteration, which
        # would skip the element following each removed one.
        to_remove = []

        for row in data:
            # No coordinate data — nothing to update. Deliberately tolerant of
            # a half-populated row (one end set, the other not), which would
            # otherwise raise a TypeError on comparison.
            if not self._has_coordinates(row):
                continue

            left = row["left_end_pos"]
            right = row["right_end_pos"]
            assert left <= right, (
                f"{data_type} {row['id']} has left_end_pos {left} > "
                f"right_end_pos {right}"
            )

            case = self._classify_against_deletion(
                left, right, del_left_pos, del_right_pos
            )

            if case == "before":
                continue

            if case == "contained":
                if data_type in ("gene", "dna_site"):
                    warnings.warn(
                        f"{row['id']} is contained within the deletion of "
                        f"{del_gene_id} and will also be deleted."
                    )
                to_remove.append(row)
                continue

            if case == "after":
                # Pure translation; the feature itself is untouched.
                row.update(
                    {
                        "left_end_pos": left - del_len,
                        "right_end_pos": right - del_len,
                    }
                )
                continue

            # The remaining cases all LOSE sequence to the deletion.
            if case == "overlaps_left":
                # Starts before, ends inside: truncate at the cut. The kept
                # portion lies entirely upstream, so it does not shift.
                updated_left, updated_right = left, del_left_pos - 1
            elif case == "overlaps_right":
                # Starts inside, ends after: the surviving 3' portion begins
                # where the deletion used to start.
                updated_left, updated_right = del_left_pos, right - del_len
            else:  # 'spans'
                # Starts before, ends after: keeps both flanks, loses del_len.
                updated_left, updated_right = left, right - del_len

            row.update(
                {"left_end_pos": updated_left, "right_end_pos": updated_right}
            )
            if data_type in ("tu", "dna_site"):
                self._annotate_removed(row, del_gene_id)

        for row in to_remove:
            data.remove(row)

    @staticmethod
    def _annotate_removed(row, del_gene_id):
        """
        Mark a feature's common name to record that it lost content to a
        deletion. Applied only to features that were truncated or lost a
        member gene — NOT to features that merely shifted position, which
        would otherwise tag every feature downstream of the deletion.

        Idempotent per deleted gene: a transcription unit that both loses a
        member gene and is truncated by that same deletion is marked once.
        """
        marker = f"_removed_{del_gene_id}"
        previous_common_name = row["common_name"]
        if previous_common_name is None:
            previous_common_name = ""
        if previous_common_name.endswith(marker):
            return
        row.update({"common_name": previous_common_name + marker})
