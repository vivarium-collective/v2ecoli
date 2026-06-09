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
        """Unpack kinetic-charging params (constants, slice indexes, mapping
        arrays, kinetic constants, previous-rate seed).

        Implemented in Task 3b. Until then this raises immediately so any
        accidental composite-build catches the gap loud and early.
        """
        super().initialize(config)
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.initialize — Task 3b: "
            "unpack codon_sequences / k_cat / K_M_* / reconciliation_buffer "
            "etc. from self.parameters into the instance."
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

        Implemented in Task 3b.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.get_kinetic_constants — Task 3b"
        )
