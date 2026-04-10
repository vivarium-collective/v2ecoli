"""
=======================
Monomer Counts Listener
=======================
"""

import numpy as np
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from bigraph_schema.schema import Float


class MonomerCounts(Step):
    """
    Listener for the counts of each protein monomer species.
    """

    name = "monomer_counts_listener"
    config_schema = {
        "bulk_molecule_ids": {"_default": []},
        "unique_ids": {"_default": []},
        "complexation_molecule_ids": {"_default": []},
        "complexation_complex_ids": {"_default": []},
        "equilibrium_molecule_ids": {"_default": []},
        "equilibrium_complex_ids": {"_default": []},
        "monomer_ids": {"_default": []},
        "two_component_system_molecule_ids": {"_default": []},
        "two_component_system_complex_ids": {"_default": []},
        "ribosome_50s_subunits": {"_default": []},
        "ribosome_30s_subunits": {"_default": []},
        "rnap_subunits": {"_default": []},
        "replisome_trimer_subunits": {"_default": []},
        "replisome_monomer_subunits": {"_default": []},
        "complexation_stoich": {"_default": []},
        "equilibrium_stoich": {"_default": []},
        "two_component_system_stoich": {"_default": []},
        "emit_unique": {"_default": False},
        "time_step": {"_default": 1},
    }
    topology = {
        "listeners": ("listeners",),
        "bulk": ("bulk",),
        "unique": ("unique",),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}

        # Get IDs of all bulk molecules
        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]

        # Get IDs of molecules involved in complexation and equilibrium
        self.complexation_molecule_ids = self.parameters["complexation_molecule_ids"]
        self.complexation_complex_ids = self.parameters["complexation_complex_ids"]
        self.equilibrium_molecule_ids = self.parameters["equilibrium_molecule_ids"]
        self.equilibrium_complex_ids = self.parameters["equilibrium_complex_ids"]
        self.monomer_ids = self.parameters["monomer_ids"]

        # Get IDs of complexed molecules monomers involved in two
        # component system
        self.two_component_system_molecule_ids = self.parameters[
            "two_component_system_molecule_ids"
        ]
        self.two_component_system_complex_ids = self.parameters[
            "two_component_system_complex_ids"
        ]

        # Get IDs of ribosome subunits
        ribosome_50s_subunits = self.parameters["ribosome_50s_subunits"]
        ribosome_30s_subunits = self.parameters["ribosome_30s_subunits"]
        self.ribosome_subunit_ids = (
            ribosome_50s_subunits["subunitIds"].tolist()
            + ribosome_30s_subunits["subunitIds"].tolist()
        )

        # Get IDs of RNA polymerase subunits
        rnap_subunits = self.parameters["rnap_subunits"]
        self.rnap_subunit_ids = rnap_subunits["subunitIds"].tolist()

        # Get IDs of replisome subunits
        replisome_trimer_subunits = self.parameters["replisome_trimer_subunits"]
        replisome_monomer_subunits = self.parameters["replisome_monomer_subunits"]
        self.replisome_subunit_ids = (
            replisome_trimer_subunits + replisome_monomer_subunits
        )

        # Get stoichiometric matrices for complexation, equilibrium, two
        # component system and the assembly of unique molecules
        self.complexation_stoich = self.parameters["complexation_stoich"]
        self.equilibrium_stoich = self.parameters["equilibrium_stoich"]
        self.two_component_system_stoich = self.parameters[
            "two_component_system_stoich"
        ]
        self.ribosome_stoich = np.hstack(
            (
                ribosome_50s_subunits["subunitStoich"],
                ribosome_30s_subunits["subunitStoich"],
            )
        )
        self.rnap_stoich = rnap_subunits["subunitStoich"]
        self.replisome_stoich = np.hstack(
            (
                3 * np.ones(len(replisome_trimer_subunits)),
                np.ones(len(replisome_monomer_subunits)),
            )
        )

        # Helper indices for Numpy indexing
        self.monomer_idx = None

    def inputs(self):
        return {
            "listeners": ListenerStore(),
            "bulk": BulkNumpyUpdate(),
            "unique": ListenerStore(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        if self.monomer_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.bulk_molecule_idx = bulk_name_to_idx(self.bulk_molecule_ids, bulk_ids)
            self.monomer_idx = bulk_name_to_idx(self.monomer_ids, bulk_ids)
            self.complexation_molecule_idx = bulk_name_to_idx(
                self.complexation_molecule_ids, bulk_ids
            )
            self.complexation_complex_idx = bulk_name_to_idx(
                self.complexation_complex_ids, bulk_ids
            )
            self.equilibrium_molecule_idx = bulk_name_to_idx(
                self.equilibrium_molecule_ids, bulk_ids
            )
            self.equilibrium_complex_idx = bulk_name_to_idx(
                self.equilibrium_complex_ids, bulk_ids
            )
            self.two_component_system_molecule_idx = bulk_name_to_idx(
                self.two_component_system_molecule_ids, bulk_ids
            )
            self.two_component_system_complex_idx = bulk_name_to_idx(
                self.two_component_system_complex_ids, bulk_ids
            )
            self.ribosome_subunit_idx = bulk_name_to_idx(
                self.ribosome_subunit_ids, bulk_ids
            )
            self.rnap_subunit_idx = bulk_name_to_idx(self.rnap_subunit_ids, bulk_ids)
            self.replisome_subunit_idx = bulk_name_to_idx(
                self.replisome_subunit_ids, bulk_ids
            )

        # Get current counts of bulk and unique molecules
        bulkMoleculeCounts = counts(states["bulk"], self.bulk_molecule_idx)
        n_active_ribosome = states["unique"]["active_ribosome"]["_entryState"].sum()
        n_active_rnap = states["unique"]["active_RNAP"]["_entryState"].sum()
        n_active_replisome = states["unique"]["active_replisome"]["_entryState"].sum()

        # Account for monomers in bulk molecule complexes
        complex_monomer_counts = np.dot(
            self.complexation_stoich,
            np.negative(bulkMoleculeCounts[self.complexation_complex_idx]),
        )
        equilibrium_monomer_counts = np.dot(
            self.equilibrium_stoich,
            np.negative(bulkMoleculeCounts[self.equilibrium_complex_idx]),
        )
        two_component_monomer_counts = np.dot(
            self.two_component_system_stoich,
            np.negative(bulkMoleculeCounts[self.two_component_system_complex_idx]),
        )

        bulkMoleculeCounts[self.complexation_molecule_idx] += (
            complex_monomer_counts.astype(np.int32)
        )
        bulkMoleculeCounts[self.equilibrium_molecule_idx] += (
            equilibrium_monomer_counts.astype(np.int32)
        )
        bulkMoleculeCounts[self.two_component_system_molecule_idx] += (
            two_component_monomer_counts.astype(np.int32)
        )

        # Account for monomers in unique molecule complexes
        n_ribosome_subunit = n_active_ribosome * self.ribosome_stoich
        n_rnap_subunit = n_active_rnap * self.rnap_stoich
        n_replisome_subunit = n_active_replisome * self.replisome_stoich
        bulkMoleculeCounts[self.ribosome_subunit_idx] += n_ribosome_subunit.astype(
            np.int32
        )
        bulkMoleculeCounts[self.rnap_subunit_idx] += n_rnap_subunit.astype(np.int32)
        bulkMoleculeCounts[self.replisome_subunit_idx] += n_replisome_subunit.astype(
            np.int32
        )

        # Update monomerCounts
        monomer_counts = bulkMoleculeCounts[self.monomer_idx]

        update = {"listeners": {"monomer_counts": monomer_counts}}
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
