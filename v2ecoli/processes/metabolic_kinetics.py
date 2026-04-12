"""
Metabolic kinetics — computes every non-FBA input that ``metabolism.py``
consumes each timestep, so that metabolism is reduced to "receive
bounds + targets + counts, call the LP, emit deltas".

What lives here (was inside ``Metabolism._do_update``):

    * exchange-bound computation from environment concentrations
      (the original ``ExchangeData`` responsibility — unchanged)
    * ``counts_to_molar`` and the flux→delta ``coefficient``
    * homeostatic concentration updates: biomass fractions via
      ``getBiomassAsConcentrations``, optional ppGpp target, optional
      tRNA-charging target drift
    * mechanistic amino-acid uptake package (mask + per-AA rates)
    * the persistent ``aa_targets`` dict and its per-step drift
    * bulk-count reads for the four slots FBA needs: metabolites,
      catalysts, kinetic-constraint enzymes, kinetic-constraint
      substrates

What this step writes:

    * ``environment.exchange_data`` — the same {constrained,
      unconstrained} contract metabolism already consumed.
    * ``process_state.metabolism_inputs`` — a flat record with
      scalar-valued unit-free fields plus the four count arrays;
      metabolism reconstructs Unum where ``modular_fba`` requires it.

Metabolism stays the single owner of the FBA model instance; this
step never holds one.
"""

import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import units


COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS


class MetabolicKinetics(Step):
    """Compute every per-step input that metabolism's FBA call needs."""

    name = "metabolic_kinetics"

    config_schema = {
        "external_state": {"_default": None},
        "environment_molecules": {"_default": []},
        "saved_media": {"_default": {}},
        "import_constraint_threshold": {"_type": "float", "_default": 0.0},
        "time_step": {"_default": 1},
        "avogadro": {"_type": "unum", "_default": 6.02214076e23},
        "cell_density": {"_type": "unum", "_default": 1100.0},
        "include_ppgpp": {"_type": "boolean", "_default": False},
        "use_trna_charging": {"_type": "boolean", "_default": False},
        "mechanistic_aa_transport": {"_type": "boolean", "_default": False},
        "media_id": {"_type": "string", "_default": "minimal"},
        "nutrientToDoublingTime": {"_type": "map[float]", "_default": {}},
        "ppgpp_id": {"_type": "string", "_default": "ppgpp"},
        "get_biomass_as_concentrations": {"_type": "method", "_default": None},
        "get_ppGpp_conc": {"_type": "method", "_default": None},
        "aa_names": {"_type": "list[string]", "_default": []},
        "aa_exchange_names": {"_type": "list[string]", "_default": []},
        "aa_targets_not_updated": {"_type": "any", "_default": set()},
        "removed_aa_uptake": {"_type": "any", "_default": None},
        "catalyst_ids": {"_type": "list[string]", "_default": []},
        "kinetic_constraint_enzymes": {"_type": "list[string]", "_default": []},
        "kinetic_constraint_substrates": {"_type": "list[string]", "_default": []},
        "metabolite_names_from_nutrients": {"_type": "list[string]", "_default": []},
        "linked_metabolites": {"_type": "map[node]", "_default": {}},
        # Michaelis-Menten glucose uptake parameters. Defaults: v_max matches
        # the saturating rate the old media lookup produced on minimal+glucose
        # (≈20 mmol/gDCW/h); K_m is the classical PTS-system value (~10 µM).
        # At 22 mM media glucose the MM formula returns essentially v_max,
        # so enabling this is inert at t=0; the smooth falloff only matters
        # once environment-depletion feedback lands in a follow-up.
        "glucose_vmax_mmol_gdcw_h": {"_type": "float", "_default": 20.0},
        "glucose_km_mM": {"_type": "float", "_default": 0.01},
        # Mechanistic biology constraints. These molecules are physically
        # allowed to cross the cell boundary but E. coli cannot grow on
        # them as a net-carbon/energy source. Left uncontrolled, the LP
        # exploits CO2 fixation / H2 oxidation to satisfy the biomass
        # objective after glucose runs out — a failure mode the FBA
        # stoichiometry permits but the underlying biology does not.
        # Override: import bound = 0 (secretion still allowed).
        #
        # Defaults target the known offenders in the E. coli metabolism:
        #   CARBON-DIOXIDE[p]: no autotrophic CO2 fixation in E. coli
        #   HYDROGEN-MOLECULE[c]: no aerobic H2 oxidation as C/energy
        "secretion_only_molecules": {
            "_type": "list[string]",
            "_default": ["CARBON-DIOXIDE[p]", "HYDROGEN-MOLECULE[c]"],
        },
    }

    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
        "bulk": ("bulk",),
        "bulk_total": ("bulk",),
        "listeners": ("listeners",),
        "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
        "metabolism_inputs": ("process_state", "metabolism_inputs"),
    }

    def initialize(self, config):
        self.parameters = config or {}
        p = self.parameters

        self.external_state = p.get("external_state")
        self.environment_molecules = p.get("environment_molecules", [])
        threshold = p.get("import_constraint_threshold", 0)
        if hasattr(threshold, "magnitude"):
            threshold = float(threshold.magnitude)
        elif hasattr(threshold, "asNumber"):
            threshold = float(threshold.asNumber())
        self.import_constraint_threshold = float(threshold)
        if self.external_state is not None:
            self.external_state.import_constraint_threshold = (
                self.import_constraint_threshold)

        self.nAvogadro = p["avogadro"]
        self.cellDensity = p["cell_density"]
        self.include_ppgpp = p["include_ppgpp"]
        self.use_trna_charging = p["use_trna_charging"]
        self.mechanistic_aa_transport = p["mechanistic_aa_transport"]
        self.media_id = p["media_id"]
        self.nutrientToDoublingTime = p["nutrientToDoublingTime"]
        self.ppgpp_id = p["ppgpp_id"]
        self.getBiomassAsConcentrations = p["get_biomass_as_concentrations"]
        self.getppGppConc = p["get_ppGpp_conc"]

        self.aa_names = p["aa_names"]
        self.aa_exchange_names = np.asarray(p["aa_exchange_names"])
        self.aa_environment_names = [aa[:-3] for aa in p["aa_exchange_names"]]
        self.aa_targets_not_updated = p["aa_targets_not_updated"]
        self.removed_aa_uptake = p["removed_aa_uptake"]
        self.linked_metabolites = p.get("linked_metabolites", {})

        self.catalyst_ids = p["catalyst_ids"]
        self.kinetic_constraint_enzymes = p["kinetic_constraint_enzymes"]
        self.kinetic_constraint_substrates = p["kinetic_constraint_substrates"]
        self.metabolite_names_from_nutrients = p["metabolite_names_from_nutrients"]

        self.glucose_vmax = float(p["glucose_vmax_mmol_gdcw_h"])
        self.glucose_km = float(p["glucose_km_mM"])
        self.secretion_only = list(p.get("secretion_only_molecules", []))

        self.aa_targets: dict[str, float] = {}

        self.metabolite_idx = None
        self.catalyst_idx = None
        self.kinetics_enzymes_idx = None
        self.kinetics_substrates_idx = None
        self.aa_idx = None

    def inputs(self):
        return {
            "boundary": "node",
            "environment": {
                "media_id": {"_type": "string", "_default": ""},
            },
            "bulk": {"_type": "bulk_array", "_default": []},
            "bulk_total": {"_type": "bulk_array", "_default": []},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "float[fg]", "_default": 0.0},
                    "dry_mass": {"_type": "float[fg]", "_default": 0.0},
                    "rna_mass": {"_type": "float[fg]", "_default": 0.0},
                    "protein_mass": {"_type": "float[fg]", "_default": 0.0},
                },
            },
            "polypeptide_elongation": {
                "gtp_to_hydrolyze": {"_type": "float", "_default": 0.0},
                "aa_count_diff": {"_type": "array[float]", "_default": []},
                "aa_exchange_rates": {
                    "_type": "array[float[mmol/g/h]]", "_default": []},
            },
        }

    def outputs(self):
        return {
            "environment": {
                "exchange_data": {
                    # Values are Unum quantities (mmol/g/h) produced by
                    # exchange_data_from_concentrations. `node` bypasses
                    # bigraph-schema's float coercion that would fail on Unum.
                    "constrained": {"_type": "node", "_default": {}},
                    "unconstrained": "list[string]",
                },
            },
            # bigraph-schema's default apply for numeric types is *add*;
            # emit everything through overwrite[...] so each step replaces
            # rather than accumulates.
            "metabolism_inputs": {
                "current_media_id": {"_type": "overwrite[string]", "_default": ""},
                "counts_to_molar_mM": {"_type": "overwrite[float]", "_default": 1.0},
                "coefficient_gsL": {"_type": "overwrite[float]", "_default": 0.0},
                "translation_gtp": {"_type": "overwrite[float]", "_default": 0.0},
                "conc_updates_mM": {"_type": "overwrite[map[float]]", "_default": {}},
                "aa_uptake_present": {"_type": "overwrite[boolean]", "_default": False},
                "aa_uptake_rates": {"_type": "overwrite[array[float]]", "_default": []},
                "aa_uptake_names": {"_type": "overwrite[list[string]]", "_default": []},
                "aa_uptake_force": {"_type": "overwrite[boolean]", "_default": True},
                "metabolite_counts": {"_type": "overwrite[array[integer]]", "_default": []},
                "catalyst_counts": {"_type": "overwrite[array[integer]]", "_default": []},
                "kinetic_enzyme_counts": {"_type": "overwrite[array[integer]]", "_default": []},
                "kinetic_substrate_counts": {"_type": "overwrite[array[integer]]", "_default": []},
            },
        }

    def _exchange_bounds(self, external_state):
        env_concs = {}
        for mol in self.environment_molecules:
            val = external_state[mol]
            if hasattr(val, "magnitude"):
                env_concs[mol] = float(val.magnitude)
            elif hasattr(val, "asNumber"):
                env_concs[mol] = float(val.asNumber())
            else:
                env_concs[mol] = float(val)
        ed = self.external_state.exchange_data_from_concentrations(env_concs)
        constrained = dict(ed["importConstrainedExchangeMolecules"])
        unconstrained = list(ed["importUnconstrainedExchangeMolecules"])

        # Michaelis-Menten glucose uptake. Always set GLC[p] from [GLC_ext]
        # so the bound decays smoothly as glucose depletes, rather than
        # dropping off a cliff at import_constraint_threshold.
        glc_ext = float(env_concs.get("GLC", 0.0))
        denom = self.glucose_km + glc_ext
        mm_rate = (self.glucose_vmax * glc_ext / denom) if denom > 0 else 0.0
        constrained["GLC[p]"] = mm_rate * (units.mmol / units.g / units.h)

        # Secretion-only enforcement: force import bound = 0 for molecules
        # whose uptake is biologically disallowed for E. coli (no
        # autotrophic CO2 fixation, no aerobic H2 oxidation as carbon /
        # energy source). Remove from unconstrained list so they don't
        # get infinity-bound downstream, and explicitly write 0 into
        # constrained.
        zero_bound = 0.0 * (units.mmol / units.g / units.h)
        for mol in self.secretion_only:
            if mol in unconstrained:
                unconstrained.remove(mol)
            constrained[mol] = zero_bound

        return constrained, unconstrained

    def update_amino_acid_targets(
        self, counts_to_molar, count_diff, amino_acid_counts,
    ):
        if self.aa_targets:
            for aa, diff in count_diff.items():
                if aa in self.aa_targets_not_updated:
                    continue
                self.aa_targets[aa] += diff
                if self.aa_targets[aa] < 0:
                    print(
                        "Warning: updated amino acid target for "
                        f"{aa} was negative - adjusted to be positive."
                    )
                    self.aa_targets[aa] = 1.0
        else:
            for aa, ct in amino_acid_counts.items():
                if aa in self.aa_targets_not_updated:
                    continue
                self.aa_targets[aa] = float(ct)

        conc_updates = {
            aa: ct * counts_to_molar for aa, ct in self.aa_targets.items()
        }
        for met, link in self.linked_metabolites.items():
            conc_updates[met] = (
                conc_updates.get(link["lead"], 0 * counts_to_molar) * link["ratio"]
            )
        return conc_updates

    def update(self, states, interval=None):
        bulk = states["bulk"]
        if self.metabolite_idx is None:
            self.metabolite_idx = bulk_name_to_idx(
                self.metabolite_names_from_nutrients, bulk["id"])
            self.catalyst_idx = bulk_name_to_idx(self.catalyst_ids, bulk["id"])
            self.kinetics_enzymes_idx = bulk_name_to_idx(
                self.kinetic_constraint_enzymes, bulk["id"])
            self.kinetics_substrates_idx = bulk_name_to_idx(
                self.kinetic_constraint_substrates, bulk["id"])
            self.aa_idx = bulk_name_to_idx(self.aa_names, bulk["id"])

        timestep = states.get("timestep", 1)
        current_media_id = states["environment"].get("media_id", "")

        constrained_bounds, unconstrained_list = self._exchange_bounds(
            states["boundary"]["external"])

        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        dry_mass = states["listeners"]["mass"]["dry_mass"] * units.fg
        cell_volume = cell_mass / self.cellDensity
        counts_to_molar = (1 / (self.nAvogadro * cell_volume)).asUnit(CONC_UNITS)
        coefficient = dry_mass / cell_mass * self.cellDensity * timestep * units.s

        doubling_time = self.nutrientToDoublingTime.get(
            current_media_id,
            self.nutrientToDoublingTime.get(self.media_id))
        if self.include_ppgpp:
            conc_updates = self.getBiomassAsConcentrations(doubling_time)
            conc_updates[self.ppgpp_id] = self.getppGppConc(
                doubling_time).asUnit(CONC_UNITS)
        else:
            rp_ratio = (
                states["listeners"]["mass"]["rna_mass"]
                / states["listeners"]["mass"]["protein_mass"]
            )
            conc_updates = self.getBiomassAsConcentrations(
                doubling_time, rp_ratio=rp_ratio)

        if self.use_trna_charging:
            conc_updates.update(
                self.update_amino_acid_targets(
                    counts_to_molar,
                    dict(zip(self.aa_names,
                             states["polypeptide_elongation"]["aa_count_diff"])),
                    dict(zip(self.aa_names,
                             counts(states["bulk_total"], self.aa_idx))),
                )
            )

        conc_updates_float = {
            met: conc.asNumber(CONC_UNITS) for met, conc in conc_updates.items()
        }

        aa_uptake_present = False
        aa_uptake_rates = np.array([], dtype=float)
        aa_uptake_names: list[str] = []
        if self.mechanistic_aa_transport:
            aa_in_media = np.array([
                states["boundary"]["external"][name]
                > self.import_constraint_threshold
                for name in self.aa_environment_names
            ])
            aa_in_media[self.removed_aa_uptake] = False
            exchange_rates = (
                np.asarray(states["polypeptide_elongation"]["aa_exchange_rates"])
                * timestep
            )
            aa_uptake_rates = exchange_rates[aa_in_media]
            aa_uptake_names = list(self.aa_exchange_names[aa_in_media])
            aa_uptake_present = True

        metabolite_counts = counts(bulk, self.metabolite_idx)
        catalyst_counts = counts(bulk, self.catalyst_idx)
        kinetic_enzyme_counts = counts(bulk, self.kinetics_enzymes_idx)
        kinetic_substrate_counts = counts(bulk, self.kinetics_substrates_idx)

        return {
            "environment": {
                "exchange_data": {
                    "constrained": constrained_bounds,
                    "unconstrained": unconstrained_list,
                },
            },
            "metabolism_inputs": {
                "current_media_id": current_media_id,
                "counts_to_molar_mM": float(counts_to_molar.asNumber(CONC_UNITS)),
                "coefficient_gsL": float(coefficient.asNumber(CONVERSION_UNITS)),
                "translation_gtp": float(
                    states["polypeptide_elongation"]["gtp_to_hydrolyze"]),
                "conc_updates_mM": conc_updates_float,
                "aa_uptake_present": aa_uptake_present,
                "aa_uptake_rates": aa_uptake_rates,
                "aa_uptake_names": aa_uptake_names,
                "aa_uptake_force": True,
                "metabolite_counts": metabolite_counts,
                "catalyst_counts": catalyst_counts,
                "kinetic_enzyme_counts": kinetic_enzyme_counts,
                "kinetic_substrate_counts": kinetic_substrate_counts,
            },
        }

    def next_update(self, timestep, states):
        return self.update(states, interval=timestep)
