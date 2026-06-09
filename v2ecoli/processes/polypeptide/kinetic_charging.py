"""
Kinetic tRNA Charging Polypeptide Elongation
============================================

Port of upstream ``KineticTrnaChargingModel``
(``CovertLab/vEcoli@trna_charging_final::polypeptide_elongation.py:2198``).
Implemented as a peer subclass of :class:`BasePolypeptideElongation` (not a
strategy on a single elongation Process) — v2ecoli's polypeptide subpackage
flattens upstream's ``PolypeptideElongation`` + ``BaseElongationModel`` split
into one class hierarchy per model. See PRs #110 and #117 for the refactor.

The kinetic model elongates polypeptides according to the kinetic limits of
aminoacyl-tRNA synthetases *and* the codon sequence — rather than the
steady-state charged-fraction Michaelis-Menten approach in
:class:`SteadyStatePolypeptideElongation`. Per tick:

1. Pick an elongation rate via binary search over the codon-sequence table
   (:func:`kinetic_charging_kernel.get_elongation_rate`).
2. Simulate codon reading + tRNA charging via the kernel reconcile pair
   (:func:`kinetic_charging_kernel.reconcile_via_ribosome_positions` and
   :func:`kinetic_charging_kernel.reconcile_via_trna_pools`).
3. Request the resulting amino acid / ATP / tRNA / synthetase / MAP counts
   from the partitioner.
4. After allocation, reconcile any tRNA-pool / sequence-position
   disagreements introduced by the realized allocation.
5. Evolve ribosome positions, peptide lengths, mass, and water.

This module currently *scaffolds* the class (Task 3a); method bodies are
filled in by tasks 3b–3e. The composite architecture wrapper lands in
:mod:`v2ecoli.composites.kinetic_charging_baseline` (Task 3f). Behavior
tests in :mod:`tests.test_behavior_kinetic_charging` (also Task 3f) depend
on Task #5 (``library/sim_data.py`` plumbing) to actually populate the new
``config_schema`` keys from ``sim_data.relation``.

See also
--------
* :mod:`v2ecoli.processes.polypeptide.kinetic_charging_kernel` — the ported
  Cython kernel (Task #2 — fully complete).
* :mod:`v2ecoli.processes.polypeptide_elongation` — base classes
  (:class:`BasePolypeptideElongation`,
  :class:`TranslationSupplyPolypeptideElongation`,
  :class:`SteadyStatePolypeptideElongation`).
* :mod:`v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.relation` —
  source of truth for the kinetic-charging parameters
  (``trna_charging_kinetics``, ``codon_sequences``, etc.).
"""

from __future__ import annotations

import numpy as np

from v2ecoli.types.quantity import ureg as units
from v2ecoli.processes.polypeptide_elongation import (
    BasePolypeptideElongation,
    NAME,
    TOPOLOGY,
)


class KineticTrnaChargingPolypeptideElongation(BasePolypeptideElongation):
    """
    Polypeptide elongation with kinetic aminoacyl-tRNA-synthetase modeling.

    Peer of :class:`SteadyStatePolypeptideElongation` (both extend
    :class:`BasePolypeptideElongation`); selected at composite-build time
    via the ``kinetic_charging_baseline`` architecture.

    L-Selenocysteine is modeled with unlimited incorporation (high ``k_cat``,
    matching upstream's approach in :class:`TranslationSupplyElongationModel`).
    """

    description = (
        "Kinetic-Charging Polypeptide Elongation — codon-aware tRNA charging.\n\n"
        "    v_charge_a = k_cat * [synthetase_a] * sat_AA(K_M_aa) * sat_tRNA(K_M_t)\n"
        "  Codon reading uses the explicit tRNA-codon mapping; ribosome positions\n"
        "  are reconciled against kinetic-model predictions per tick.\n"
        "  v_charge_a = charging rate (aa/s); a indexes amino-acid species."
    )

    name = NAME
    topology = TOPOLOGY

    # Extra config knobs needed only by this elongation model. Merged onto
    # BasePolypeptideElongation.config_schema so the partitioner picks them
    # up at composite-build time. Defaults are empty / zero-shaped — Task #5
    # populates them from sim_data.relation when the composite is built.
    config_schema = {
        **BasePolypeptideElongation.config_schema,
        # ---- codon-sequence tables (from sim_data.relation) ----
        "codon_sequences": {
            "_type": "array[integer]",
            "_default": np.zeros((0, 0), dtype=np.int8),
        },
        "residue_weights_by_codon": {
            "_type": "array[float]",
            "_default": np.zeros(0, dtype=np.float64),
        },
        "n_codons": {"_type": "integer", "_default": 0},
        "i_start_codon": {"_type": "integer", "_default": 0},
        "is_map_substrate": {
            "_type": "array[integer]",
            "_default": np.zeros(0, dtype=bool),
        },
        # ---- tRNA <-> codon mapping (from sim_data.relation) ----
        "n_trna_codon_pairs": {"_type": "integer", "_default": 0},
        "trnas_to_codons": {
            "_type": "array[integer]",
            "_default": np.zeros((0, 0), dtype=np.int8),
        },
        "codons_to_amino_acids": {
            "_type": "array[integer]",
            "_default": np.zeros((0, 0), dtype=np.int8),
        },
        # ---- kinetic parameters (from sim_data.relation.trna_charging_kinetics) ----
        "k_cat__per_s": {
            "_type": "array[float]",
            "_default": np.zeros(0, dtype=np.float64),
        },
        "K_M_amino_acid__per_L": {
            "_type": "array[float]",
            "_default": np.zeros(0, dtype=np.float64),
        },
        "K_M_trna__per_L": {
            "_type": "array[float]",
            "_default": np.zeros(0, dtype=np.float64),
        },
        # ---- reconciliation ----
        "reconciliation_buffer": {"_type": "integer", "_default": 10},
    }

    # ---------- initialize ----------

    def initialize(self, config):
        """Unpack kinetic-charging params.

        Calls :meth:`BasePolypeptideElongation.initialize` to set up
        ``ribosomeElongationRate``, ``amino_acids``, ``uncharged_trna_names``,
        ``aa_from_trna``, ``random_state``, the bulk-index ``None`` markers,
        etc. Then unpacks the kinetic-charging-specific config keys (see
        :attr:`config_schema`) and derives the slice indexes for the molecules
        buffer used by :meth:`run_model`.

        Port of upstream ``KineticTrnaChargingModel.__init__`` (lines 2208–2282).
        Differences from upstream:

        * Upstream caches a reference to the parent ``PolypeptideElongation``
          process as ``self.process``; v2ecoli's class IS the process, so
          ``self.process.X`` references become ``self.X`` directly.
        * ``cellDensity`` matches the base config_schema key (upstream calls
          it ``cell_density``).
        * ``n_avogadro`` is set by base; we read it from the base attribute
          rather than re-fetching ``self.parameters``.
        """
        super().initialize(config)

        # ---- Constants ----
        self.cell_density = self.parameters["cellDensity"]
        # self.n_avogadro already set by BasePolypeptideElongation.initialize

        # ---- Codon sequences ----
        # These shadow base's amino-acid-sequence attrs (proteinSequences,
        # aaWeightsIncorporated) — the kinetic model walks codons, not AAs.
        self.protein_sequences = self.parameters["codon_sequences"]
        self.monomer_weights_incorporated = self.parameters[
            "residue_weights_by_codon"
        ]
        self.n_monomers = self.parameters["n_codons"]
        self.i_start_codon = self.parameters["i_start_codon"]
        self.is_map_substrate = self.parameters["is_map_substrate"]

        # ---- Tools for interacting with the kinetic model ----
        self.n_trnas = len(self.parameters["uncharged_trna_names"])
        self.n_codons = self.parameters["n_codons"]
        n_trna_codon_pairs = self.parameters["n_trna_codon_pairs"]

        # Layout of the flat molecules buffer that run_model returns/consumes.
        # The six segments are placed contiguously and accessed via Python
        # slice objects stored on self for cheap per-tick indexing.
        slice_lengths = [
            self.n_trnas,  # free_trnas
            self.n_trnas,  # charged_trnas
            len(self.parameters["amino_acids"]),  # amino_acids
            self.n_trnas,  # chargings (charging counter)
            self.n_trnas,  # reading counter
            n_trna_codon_pairs,  # codons_to_trnas_counter (flattened)
        ]
        self.molecules_input_size = sum(slice_lengths)

        slices = []
        previous = 0
        for length in slice_lengths:
            slices.append(slice(previous, previous + length))
            previous += length

        self.slice_free_trnas = slices[0]
        self.slice_charged_trnas = slices[1]
        self.slice_amino_acids = slices[2]
        self.slice_charging_counter = slices[3]
        self.slice_reading_counter = slices[4]
        self.slice_codons_to_trnas_counter = slices[5]

        # ---- Mapping arrays ----
        # aa_from_trna is set by base; cast and transpose for our local use.
        self.trnas_to_amino_acids = self.parameters["aa_from_trna"].astype(
            np.int64
        )
        self.amino_acids_to_trnas = self.parameters["aa_from_trna"].T
        self.trnas_to_codons = self.parameters["trnas_to_codons"]
        self.codons_to_trnas = self.parameters["trnas_to_codons"].T.astype(
            np.bool_
        )
        self.codons_to_amino_acids = self.parameters["codons_to_amino_acids"]

        # For each tRNA, record the amino acid it carries (single index).
        # The Cython kernel uses int8 for cheap memory footprint.
        self.trnas_to_amino_acid_indexes = np.zeros(self.n_trnas, dtype=np.int8)
        for i in range(self.trnas_to_amino_acids.shape[1]):
            j = np.where(self.trnas_to_amino_acids[:, i])[0][0]
            self.trnas_to_amino_acid_indexes[i] = j

        # Maximum reconciliation attempts handed to
        # kinetic_charging_kernel.reconcile_via_ribosome_positions.
        self.max_attempts = np.byte(4)

        # ---- Kinetic parameters ----
        # L-selenocysteine is modeled with a high k_cat to represent
        # unlimited incorporation, matching upstream's TranslationSupply
        # approach.
        self.k_cat__per_s = self.parameters["k_cat__per_s"]
        self.K_M_amino_acid__per_L = self.parameters["K_M_amino_acid__per_L"]
        self.K_M_trna__per_L = self.parameters["K_M_trna__per_L"]

        # ---- Reconciliation width buffer ----
        # The reconciliation step in :meth:`reconcile` uses the surrounding
        # codon sequence (towards both the N and C terminals) to fix up
        # disagreements between the kinetic-model predictions and the
        # ribosome-position model. ``buffer`` is the extra C-ward sequence
        # positions to view per tick.
        self.buffer = self.parameters["reconciliation_buffer"]

        # ---- Warm-start the next tick's binary search ----
        # First tick uses the basal elongation rate (~17.3 aa/s, set by
        # base from sim_data); subsequent ticks update from the realized
        # rate inside :meth:`elongation_rate`.
        self.previous_rate = int(
            self.ribosomeElongationRate * self.parameters["time_step"]
        )

    # ---------- request side ----------

    def elongation_rate(self, states):
        """Binary-search the kinetic-limited elongation rate.

        Uses :func:`kinetic_charging_kernel.get_elongation_rate`. Implemented
        in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.elongation_rate — Task 3c"
        )

    def request(self, states, aasInSequences):
        """Simulate the kinetic charging step + request resources.

        Runs :meth:`run_model` to estimate per-codon consumption, then requests
        amino acids, ATP, tRNAs, synthetases, and methionine aminopeptidase
        from the partitioner. Implemented in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.request — Task 3c"
        )

    # ---------- evolve side ----------

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        """Realized amino acids available after partition allocation.

        Sums total AA counts with charged-tRNA-bound amino acids.
        Implemented in Task 3d.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.final_amino_acids — Task 3d"
        )

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        """Apply elongation + tRNA-pool reconciliation; emit bulk deltas.

        Uses the kernel reconcile_via_* functions. Implemented in Task 3d.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.evolve — Task 3d"
        )

    # ---------- internal helpers ----------

    def run_model(self, codons, attr, states):
        """Simulate kinetic charging + codon reading; predict deltas.

        Called from both :meth:`request` (``attr="bulk_total"``) for resource
        sizing and :meth:`reconcile` (``attr="bulk"``) for the realized
        post-allocation run. Implemented in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.run_model — Task 3c"
        )

    def reconcile(self, states, result):
        """Reconcile partition allocation with kinetic predictions.

        Calls the kernel
        :func:`kinetic_charging_kernel.reconcile_via_ribosome_positions` and
        :func:`kinetic_charging_kernel.reconcile_via_trna_pools` helpers.
        Implemented in Task 3d.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.reconcile — Task 3d"
        )

    def protein_maturation(
        self, sequences, peptide_lengths, protein_indexes, water, gtp
    ):
        """N-terminal methionine cleavage by MAP.

        Implemented in Task 3d.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.protein_maturation — Task 3d"
        )

    def monomer_to_aa(self, monomer):
        """Map codon ID → amino acid ID.

        Implemented in Task 3e.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.monomer_to_aa — Task 3e"
        )

    def monomer_limit(self, states, monomer_count_in_sequence):
        """Per-codon usage cap from the partitioned allocation.

        Implemented in Task 3e.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.monomer_limit — Task 3e"
        )

    def codon_sequences_width(self, elongation_rates):
        """Sequence-table read-ahead width for the next tick.

        Implemented in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.codon_sequences_width — Task 3c"
        )

    def sequences(self, sequences):
        """Translate ribosome positions to per-ribosome codon arrays.

        Implemented in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.sequences — Task 3c"
        )

    def max_charging_rate(self, states, attr):
        """Max-charging-rate (kinetic ceiling) for the current allocation.

        Implemented in Task 3c.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.max_charging_rate — Task 3c"
        )

    def get_kinetic_constants(self, cell_mass):
        """Resolve mass-density-dependent kinetic constants.

        The kinetic Michaelis constants are stored per-litre
        (``K_M_amino_acid__per_L``, ``K_M_trna__per_L``) so they survive
        cell-volume changes; converts back to per-cell using the current
        cell mass and density. Returns ``(K_M_amino_acids, K_M_trnas)`` as
        ``pint.Quantity`` arrays.

        Port of upstream ``KineticTrnaChargingModel.get_kinetic_constants``
        (lines 2720–2725). Differences:

        * Upstream multiplies by ``cell_volume`` as a Unum scalar; we use
          pint Quantities throughout (via the unit_bridge).
        """
        cell_volume = cell_mass * units.fg / self.cell_density
        cell_volume = np.float64(cell_volume.to(units.L).magnitude)
        K_M_amino_acids = self.K_M_amino_acid__per_L * cell_volume
        K_M_trnas = self.K_M_trna__per_L * cell_volume
        return K_M_amino_acids, K_M_trnas
