"""
=============
Mass Listener
=============

Represents the total cellular mass.
"""

import numpy as np
from numpy.lib import recfunctions as rfn

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import counts, attrs, bulk_name_to_idx
from v2ecoli.library.unit_defs import units
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from bigraph_schema.schema import Float


class MassListener(Step):
    """MassListener"""

    name = "ecoli-mass-listener"
    config_schema = {
        "cellDensity": {"_default": 1100.0},
        "bulk_ids": {"_default": []},
        "bulk_masses": {"_default": None},
        "unique_ids": {"_default": []},
        "unique_masses": {"_default": None},
        "submass_to_idx": {"_default": {
            "rRNA": 0,
            "tRNA": 1,
            "mRNA": 2,
            "miscRNA": 3,
            "nonspecific_RNA": 4,
            "protein": 5,
            "metabolite": 6,
            "water": 7,
            "DNA": 8,
        }},
        "compartment_indices": {"_default": {
            "projection": [],
            "cytosol": [],
            "extracellular": [],
            "flagellum": [],
            "membrane": [],
            "outer_membrane": [],
            "periplasm": [],
            "pilus": [],
            "inner_membrane": [],
        }},
        "compartment_id_to_index": {"_default": {}},
        "compartment_abbrev_to_index": {"_default": {}},
        "n_avogadro": {"_default": 6.0221409e23},
        "time_step": {"_default": 1.0},
        "emit_unique": {"_default": False},
        "match_wcecoli": {"_default": False},
    }
    topology = {
        "bulk": ("bulk",),
        "unique": ("unique",),
        "listeners": ("listeners",),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}

        # molecule indexes and masses
        self.bulk_ids = self.parameters.get("bulk_ids", [])
        self.bulk_masses = self.parameters.get("bulk_masses", np.zeros([1, 9]))
        self.unique_ids = self.parameters.get("unique_ids", [])
        self.unique_masses = self.parameters.get("unique_masses", np.zeros([1, 9]))

        _default_submass_to_idx = {
            "rRNA": 0, "tRNA": 1, "mRNA": 2, "miscRNA": 3,
            "nonspecific_RNA": 4, "protein": 5, "metabolite": 6,
            "water": 7, "DNA": 8,
        }
        submass_to_idx = self.parameters.get("submass_to_idx", _default_submass_to_idx)
        self.submass_listener_indices = {
            "rna": np.array(
                [
                    submass_to_idx[name]
                    for name in ["rRNA", "tRNA", "mRNA", "miscRNA", "nonspecific_RNA"]
                ]
            ),
            "rRna": submass_to_idx["rRNA"],
            "tRna": submass_to_idx["tRNA"],
            "mRna": submass_to_idx["mRNA"],
            "dna": submass_to_idx["DNA"],
            "protein": submass_to_idx["protein"],
            "smallMolecule": submass_to_idx["metabolite"],
            "water": submass_to_idx["water"],
        }
        self.ordered_submasses = [0] * len(submass_to_idx)
        for submass, idx in submass_to_idx.items():
            self.ordered_submasses[idx] = f"{submass}_submass"

        # compartment indexes
        _default_compartment_indices = {
            "projection": [], "cytosol": [], "extracellular": [],
            "flagellum": [], "membrane": [], "outer_membrane": [],
            "periplasm": [], "pilus": [], "inner_membrane": [],
        }
        compartment_indices = self.parameters.get("compartment_indices", _default_compartment_indices)
        self.compartment_id_to_index = self.parameters.get("compartment_id_to_index", {})
        self.projection_index = compartment_indices.get("projection", [])
        self.cytosol_index = compartment_indices.get("cytosol", [])
        self.extracellular_index = compartment_indices.get("extracellular", [])
        self.flagellum_index = compartment_indices.get("flagellum", [])
        self.membrane_index = compartment_indices.get("membrane", [])
        self.outer_membrane_index = compartment_indices.get("outer_membrane", [])
        self.periplasm_index = compartment_indices.get("periplasm", [])
        self.pilus_index = compartment_indices.get("pilus", [])
        self.inner_membrane_index = compartment_indices.get("inner_membrane", [])

        # Set up matrix for compartment mass calculation
        self.compartment_abbrev_to_index = self.parameters.get(
            "compartment_abbrev_to_index", {}
        )
        if self.compartment_abbrev_to_index:
            self._bulk_molecule_by_compartment = np.stack(
                [
                    np.core.defchararray.chararray.endswith(self.bulk_ids, abbrev + "]")
                    for abbrev in self.compartment_abbrev_to_index
                ]
            )

        # units and constants
        self.cellDensity = self.parameters.get("cellDensity", 1100.0)
        self.n_avogadro = self.parameters.get("n_avogadro", 6.0221409e23)

        self.time_step = self.parameters.get("time_step", 1.0)
        self.first_time_step = True

        self.massDiff_names = [
            "massDiff_" + submass for submass in submass_to_idx
        ]

        self.cell_cycle_len = self.parameters["condition_to_doubling_time"][
            self.parameters["condition"]
        ].asNumber(units.s)

        # Helper indices for Numpy indexing
        self.bulk_idx = None

        # Enable flag for perfect recapitulation of wcEcoli mass calculations
        self.match_wcecoli = self.parameters.get("match_wcecoli", False)

    def inputs(self):
        return {
            "bulk": BulkNumpyUpdate(),
            "unique": ListenerStore(),
            "listeners": ListenerStore(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        if self.bulk_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.bulk_idx = bulk_name_to_idx(self.bulk_ids, bulk_ids)
            if self.match_wcecoli:
                self.bulk_addon = np.zeros((len(self.bulk_idx), 16))

        mass_update = {}

        # Get previous dry mass, for calculating growth later
        old_dry_mass = states["listeners"]["mass"]["dry_mass"]

        # get submasses from bulk and unique
        bulk_counts = counts(states["bulk"], self.bulk_idx)
        bulk_masses = states["bulk"][self.ordered_submasses][self.bulk_idx]
        bulk_masses = rfn.structured_to_unstructured(bulk_masses)
        bulk_submasses = np.dot(bulk_counts, bulk_masses)
        bulk_compartment_masses = np.dot(
            bulk_counts * self._bulk_molecule_by_compartment, bulk_masses
        )
        if self.match_wcecoli:
            bulk_counts = np.hstack(
                [self.bulk_addon, counts(states["bulk"], self.bulk_idx)[:, np.newaxis]]
            )
            bulk_submasses = np.dot(bulk_counts.T, bulk_masses).sum(axis=0)
            bulk_compartment_masses = np.dot(
                bulk_counts.sum(axis=1) * self._bulk_molecule_by_compartment,
                bulk_masses,
            )

        unique_submasses = np.zeros(len(self.massDiff_names))
        unique_compartment_masses = np.zeros_like(bulk_compartment_masses)
        for unique_id, unique_mass in zip(self.unique_ids, self.unique_masses):
            molecules = states["unique"].get(unique_id)
            n_molecules = molecules["_entryState"].sum()

            if n_molecules == 0:
                continue

            unique_submasses += unique_mass * n_molecules
            unique_compartment_masses[self.compartment_abbrev_to_index["c"], :] += (
                unique_mass * n_molecules
            )

            massDiffs = np.array(list(attrs(molecules, self.massDiff_names))).T
            if self.match_wcecoli:
                massDiffs = np.core.records.fromarrays(
                    attrs(molecules, self.massDiff_names)
                ).view((np.float64, len(self.massDiff_names)))
            unique_submasses += massDiffs.sum(axis=0)
            unique_compartment_masses[self.compartment_abbrev_to_index["c"], :] += (
                massDiffs.sum(axis=0)
            )

        # all of the submasses
        all_submasses = bulk_submasses + unique_submasses

        # save cell mass, water mass, dry mass
        mass_update["cell_mass"] = all_submasses.sum()
        mass_update["water_mass"] = all_submasses[
            self.submass_listener_indices["water"]
        ]
        mass_update["dry_mass"] = mass_update["cell_mass"] - mass_update["water_mass"]

        # Store submasses
        for submass, indices in self.submass_listener_indices.items():
            mass_update[submass + "_mass"] = all_submasses[indices].sum()

        mass_update["volume"] = mass_update["cell_mass"] / self.cellDensity

        if self.first_time_step:
            mass_update["growth"] = 0.0
            self.dryMassInitial = mass_update["dry_mass"]
            self.proteinMassInitial = mass_update["protein_mass"]
            self.rnaMassInitial = mass_update["rna_mass"]
            self.smallMoleculeMassInitial = mass_update["smallMolecule_mass"]
            self.timeInitial = states["global_time"]
        else:
            mass_update["growth"] = mass_update["dry_mass"] - old_dry_mass

        # Compartment submasses
        compartment_submasses = bulk_compartment_masses + unique_compartment_masses
        mass_update["projection_mass"] = compartment_submasses[
            self.projection_index, :
        ].sum()
        mass_update["cytosol_mass"] = compartment_submasses[self.cytosol_index, :].sum()
        mass_update["extracellular_mass"] = compartment_submasses[
            self.extracellular_index, :
        ].sum()
        mass_update["flagellum_mass"] = compartment_submasses[
            self.flagellum_index, :
        ].sum()
        mass_update["membrane_mass"] = compartment_submasses[
            self.membrane_index, :
        ].sum()
        mass_update["outer_membrane_mass"] = compartment_submasses[
            self.outer_membrane_index, :
        ].sum()
        mass_update["periplasm_mass"] = compartment_submasses[
            self.periplasm_index, :
        ].sum()
        mass_update["pilus_mass"] = compartment_submasses[self.pilus_index, :].sum()
        mass_update["inner_membrane_mass"] = compartment_submasses[
            self.inner_membrane_index, :
        ].sum()

        if mass_update["dry_mass"] != 0:
            mass_update["protein_mass_fraction"] = (
                mass_update["protein_mass"] / mass_update["dry_mass"]
            )
            mass_update["rna_mass_fraction"] = (
                mass_update["rna_mass"] / mass_update["dry_mass"]
            )
            mass_update["instantaneous_growth_rate"] = (
                mass_update["growth"] / self.time_step / mass_update["dry_mass"]
            )
            mass_update["dry_mass_fold_change"] = (
                mass_update["dry_mass"] / self.dryMassInitial
            )
            mass_update["protein_mass_fold_change"] = (
                mass_update["protein_mass"] / self.proteinMassInitial
            )
            mass_update["rna_mass_fold_change"] = (
                mass_update["rna_mass"] / self.rnaMassInitial
            )
            mass_update["small_molecule_fold_change"] = (
                mass_update["smallMolecule_mass"] / self.smallMoleculeMassInitial
            )
            mass_update["expected_mass_fold_change"] = np.exp(
                np.log(2)
                * (states["global_time"] - self.timeInitial)
                / self.cell_cycle_len
            )

        self.first_time_step = False

        update = {"listeners": {"mass": mass_update}}
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)


class PostDivisionMassListener(MassListener):
    """
    Normally, the mass listener updates after all other processes and steps
    have run. However, after division, the mass must be updated immediately
    so other processes have access to the accurate mass of their daughter
    cell. This process ensures that the mass seen by other processes following
    division is accurate.
    """

    name = "post-division-mass-listener"

    def update_condition(self, timestep, states):
        return self.first_time_step
