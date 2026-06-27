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

Port status (as of Task 3c):

* 3a — scaffold + ``config_schema`` extensions.
* 3b — ``initialize`` + ``get_kinetic_constants``.
* 3c — ``elongation_rate``, ``request``, ``run_model``,
  ``codon_sequences_width``, ``sequences``, ``max_charging_rate``, plus the
  ``_init_bulk_indices`` override that adds ``atp_idx``, ``amp_idx``,
  ``ppi_idx``, ``met_idx``, ``map_idx`` to the base layout.
* 3d (pending) — ``evolve``, ``reconcile``, ``protein_maturation``,
  ``final_amino_acids``.
* 3e (pending) — ``monomer_to_aa``, ``monomer_limit``, listener emission.

The composite architecture wrapper lands in
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
from scipy.integrate import solve_ivp

from bigraph_schema import deep_merge

from v2ecoli.library.polymerize import buildSequences, computeMassIncrease, polymerize
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.processes.polypeptide import kinetic_charging_kernel as kernel
from v2ecoli.processes.polypeptide_elongation import (
    BasePolypeptideElongation,
    NAME,
    TOPOLOGY,
)
from v2ecoli.types.quantity import ureg as units
from wholecell.utils.random import stochasticRound


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
        # ---- consensus extension: opt-in AA-supply ODE merge (Phase 3) ----
        # When True, extends the kinetic ODE state vector with three
        # accumulator slices (total_synthesis, total_import, total_export)
        # and (Phase 3b) adds AA synthesis/import/export terms to the AA
        # balance in the RHS. Default False keeps the legacy 6-slice path
        # bit-identical. See workspace/investigations/consensus_elongation/
        # audit.md §2.
        "include_aa_supply": {"_type": "boolean", "_default": False},
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

        # Consensus opt-in: extend ODE state vector with AA-supply
        # accumulators. Phase 3a scaffold — slices allocated, no RHS writes
        # yet (those land in Phase 3b). When False the layout is
        # bit-identical to trna_charging_final@5ffb76de.
        self.include_aa_supply = self.parameters["include_aa_supply"]
        n_aas = len(self.parameters["amino_acids"])

        # Layout of the flat molecules buffer that run_model returns/consumes.
        # Six legacy segments + three accumulator segments (gated). All
        # placed contiguously and accessed via Python slice objects stored
        # on self for cheap per-tick indexing.
        slice_lengths = [
            self.n_trnas,  # free_trnas
            self.n_trnas,  # charged_trnas
            n_aas,  # amino_acids
            self.n_trnas,  # chargings (charging counter)
            self.n_trnas,  # reading counter
            n_trna_codon_pairs,  # codons_to_trnas_counter (flattened)
        ]
        if self.include_aa_supply:
            slice_lengths.extend([
                n_aas,  # total_synthesis accumulator
                n_aas,  # total_import accumulator
                n_aas,  # total_export accumulator
            ])
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
        if self.include_aa_supply:
            self.slice_total_synthesis = slices[6]
            self.slice_total_import = slices[7]
            self.slice_total_export = slices[8]
        else:
            self.slice_total_synthesis = None
            self.slice_total_import = None
            self.slice_total_export = None

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

    # ---------- bulk-index setup ----------

    def _init_bulk_indices(self, bulk_ids):
        """Add kinetic-charging-specific bulk indices to the base layout.

        Extends :meth:`BasePolypeptideElongation._init_bulk_indices` with the
        ATP/AMP/PPi/Met/MAP indices the kinetic model touches. Mirrors the
        upstream ``PolypeptideElongation.calculate_request`` block at lines
        534–538 of upstream's ``polypeptide_elongation.py``.
        """
        super()._init_bulk_indices(bulk_ids)
        self.atp_idx = bulk_name_to_idx(["ATP[c]"], bulk_ids)
        self.amp_idx = bulk_name_to_idx(["AMP[c]"], bulk_ids)
        self.ppi_idx = bulk_name_to_idx(["PPI[c]"], bulk_ids)
        self.met_idx = bulk_name_to_idx(["MET[c]"], bulk_ids)
        self.map_idx = bulk_name_to_idx(["EG10570-MONOMER[c]"], bulk_ids)

    # ---------- request side ----------

    def elongation_rate(self, states):
        """Binary-search the kinetic-limited elongation rate.

        Side-effects: sets ``self.sequences_width`` and
        ``self.longer_sequences`` (the codon-based sequence table for the
        coming tick) so :meth:`request` and :meth:`evolve` can consume them
        without re-deriving the build.

        Port of upstream ``KineticTrnaChargingModel.elongation_rate`` (lines
        2284–2309). Differences:

        * Upstream's signature is ``(states, protein_indexes, peptide_lengths)``;
          v2ecoli's contract is ``(states)``, so we re-derive the indexes from
          ``states["active_ribosome"]`` here.
        * Unum → pint: ``.asNumber(units.aa / units.s)`` →
          ``.to(units.aa / units.s).magnitude``.
        """
        protein_indexes, peptide_lengths = attrs(
            states["active_ribosome"], ["protein_index", "peptide_length"]
        )

        # Sequence-table read-ahead width for the next tick (1-element array
        # so buildSequences treats it as a per-ribosome constant).
        self.sequences_width = np.array(
            [
                np.ceil(
                    (self.basal_elongation_rate * states["timestep"]) + self.buffer
                ).astype(int)
            ]
        )

        self.longer_sequences = buildSequences(
            self.protein_sequences,
            protein_indexes,
            peptide_lengths,
            self.sequences_width,
        )

        target = (
            self.ribosomeElongationRateDict[states["environment"]["media_id"]]
        ).to(units.aa / units.s).magnitude

        rate = kernel.get_elongation_rate(
            self.longer_sequences,
            self.previous_rate,
            states["timestep"],
            target,
        )

        # Warm-start the next tick's binary search.
        self.previous_rate = int(rate * states["timestep"])
        return rate

    def request(self, states, aasInSequences):
        """Simulate kinetic charging + codon reading, then request resources.

        Runs :meth:`run_model` against the ``bulk_total`` pool to estimate
        what the cell will consume this tick (over-estimating slightly: the
        partitioner caps to actual availability anyway). Returns the bulk
        requests dict in the v2ecoli partitioner format.

        Note: ``aasInSequences`` is supplied by ``calculate_request`` but is
        amino-acid-based; the kinetic model walks codons, so we recompute
        ``monomers_in_sequences`` from ``self.longer_sequences`` (populated
        by :meth:`elongation_rate`).

        Port of upstream ``KineticTrnaChargingModel.request`` (lines
        2311–2411). Differences:

        * Upstream signature is ``(states, monomers_in_sequences,
          protein_indexes, peptide_lengths)``; v2ecoli's is ``(states,
          aasInSequences)``. Re-derive what we need here.
        * ``self.process.X_idx`` → ``self.X_idx`` (v2ecoli's class is the
          process).
        """
        protein_indexes, _ = attrs(
            states["active_ribosome"], ["protein_index", "peptide_length"]
        )

        # Recompute codon-domain monomers_in_sequences (ignoring
        # aasInSequences, which is in the amino-acid domain).
        sequences = self.longer_sequences
        monomers_in_sequences = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE],
            minlength=self.n_codons,
        )

        # Initiation requires one water per start codon.
        water_request = monomers_in_sequences[self.i_start_codon]

        # Simulate trna charging + codon reading against the total pool.
        (
            amino_acids_used,
            codons_read,
            free_trnas,
            charged_trnas,
            _chargings,
            _codons_to_trnas_matrix,
            listeners,
        ) = self.run_model(monomers_in_sequences, "bulk_total", states)

        # Cache the AA-used estimate (used as the "expected" delta in
        # :meth:`reconcile` — Task 3d) and the per-codon prediction (used
        # by :meth:`monomer_limit` — Task 3e).
        self.first = amino_acids_used
        self.codons_kinetics_model = codons_read

        # Request amino acids. +1 is a non-zero buffer; ceil(1.01 * x) over-
        # requests by ~1% to absorb discretization & reconciliation.
        requests = listeners
        requests["bulk"] = [
            (
                self.amino_acid_idx,
                np.ceil(1.01 * (amino_acids_used + 1)).astype(int),
            )
        ]

        # Request ATP. Upstream assumes all AAs go to charging (an upper
        # bound — the realised count is lower because some go straight to
        # translation), which over-requests but eases reconciliation.
        requests["bulk"].append(
            (self.atp_idx, amino_acids_used.sum().astype(int))
        )

        # Request all tRNAs (uncharged + charged).
        requests["bulk"].append(
            (self.uncharged_trna_idx, counts(states["bulk"], self.uncharged_trna_idx))
        )
        requests["bulk"].append(
            (self.charged_trna_idx, counts(states["bulk"], self.charged_trna_idx))
        )

        # Request all synthetase enzymes.
        requests["bulk"].append(
            (self.synthetase_idx, counts(states["bulk"], self.synthetase_idx))
        )

        # Request methionine aminopeptidase (consumed in
        # :meth:`protein_maturation` — Task 3d).
        requests["bulk"].append(
            (self.map_idx, counts(states["bulk"], self.map_idx))
        )

        # Termination water: any ribosome whose final codon in the
        # read-ahead window is a stop-padding slot may terminate this tick,
        # contributing one water per cleaved N-terminal methionine.
        may_terminate = self.longer_sequences[:, -1] == -1
        max_to_cleave = np.sum(
            np.bincount(
                protein_indexes[may_terminate],
                minlength=self.protein_sequences.shape[0],
            )[self.is_map_substrate]
        )
        water_request = water_request + max_to_cleave
        requests["bulk"].append((self.water_idx, water_request))

        # Fraction-charged is returned for the listener output in
        # ``calculate_request`` (per-amino-acid downstream — v2ecoli base
        # expects a length-n_aas vector). Aggregate the per-tRNA pools onto
        # amino acids via aa_from_trna before normalizing.
        charged_per_aa = self.aa_from_trna @ charged_trnas
        total_per_aa = self.aa_from_trna @ (free_trnas + charged_trnas)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction_charged = np.where(
                total_per_aa > 0, charged_per_aa / total_per_aa, 0.0
            )

        return fraction_charged, amino_acids_used.astype(float), requests

    # ---------- evolve side ----------

    def evolve_state(self, timestep, states):
        """Codon-aware kinetic evolve path; replaces the base's AA-only one.

        Base ``evolve_state`` polymerizes against the amino-acid pool via
        :meth:`final_amino_acids`. The kinetic model needs the codon-domain
        pipeline: build codon sequences from ``self.protein_sequences``,
        cap by the kinetic prediction (:meth:`monomer_limit`), run
        ``polymerize``, then run :meth:`reconcile` (which calls the kernel's
        reconcile_via_* pair), :meth:`protein_maturation` (MAP cleavage),
        and finally :meth:`evolve` to emit the bulk deltas.

        Port of upstream ``PolypeptideElongation.evolve_state`` (lines
        619–820), conditioned on the kinetic model. The non-kinetic branches
        in upstream are dropped — they're served by other v2ecoli classes.

        Differences from upstream:

        * ``self.elongation_model.X`` → ``self.X`` (v2ecoli's class IS the
          model and the process).
        * Bulk-index references are ``self.X_idx`` throughout.
        * Trip through pint Quantities for ``effective_elongation_rate``
          listener emission (the v2ecoli convention; upstream emits a bare
          float).
        """
        update = {
            "listeners": {
                "ribosome_data": {},
                "growth_limits": {},
                "trna_charging": {},
            },
            "polypeptide_elongation": {},
            "active_ribosome": {},
            "bulk": [],
        }

        # Pre-populate metabolism inputs in case of early return.
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = 0
        update["polypeptide_elongation"]["aa_count_diff"] = np.zeros(
            len(self.amino_acids), dtype=np.float64
        )

        n_active_ribosomes = states["active_ribosome"]["_entryState"].sum()
        update["listeners"]["growth_limits"]["active_ribosome_allocated"] = (
            n_active_ribosomes
        )
        update["listeners"]["growth_limits"]["aa_allocated"] = counts(
            states["bulk"], self.amino_acid_idx
        )
        if n_active_ribosomes == 0:
            return update

        # The kinetic pipeline mutates ``states["bulk"]`` in real time
        # (terminated polypeptides + freed ribosomal subunits), so make a
        # writeable copy.
        states["bulk"] = counts(states["bulk"], range(len(states["bulk"])))

        # ---- Build per-ribosome AA sequences (used by polymerize) ----
        protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
            states["active_ribosome"],
            ["protein_index", "peptide_length", "pos_on_mRNA"],
        )

        all_sequences = buildSequences(
            self.protein_sequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates + self.next_aa_pad,
        )
        sequences = all_sequences[:, : -self.next_aa_pad].copy()
        if sequences.size == 0:
            return update

        # ---- Build per-ribosome CODON sequences (used by reconcile) ----
        codon_sequences_width = self.codon_sequences_width(self.elongation_rates)
        # Note: we keep our cached longer_sequences (set by elongation_rate)
        # rather than rebuilding here; the kernel's reconcile_via_* helpers
        # need the matching layout.

        # Codon usage capacity from the kinetic prediction.
        monomer_count_in_sequence = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE],
            minlength=self.n_monomers,
        )
        monomer_count_in_sequence_in_aas = self.monomer_to_aa(
            monomer_count_in_sequence
        )
        allocated_aas = counts(states["bulk"], self.amino_acid_idx)

        # MODEL-SPECIFIC: codon-domain monomer limit.
        monomer_limit, monomer_limit_in_aas = self.monomer_limit(
            states, monomer_count_in_sequence_in_aas
        )

        # ---- Polymerize against the codon-limited pool ----
        result = polymerize(
            sequences,
            monomer_limit,
            10000000,  # ATP-limit is enforced elsewhere by Metabolism
            self.random_state,
            self.elongation_rates[protein_indexes],
            variable_elongation=self.variable_polymerize,
        )

        # MODEL-SPECIFIC: reconcile polymerize result with kinetic prediction.
        result, aas_used, net_charged, additional_listeners = self.reconcile(
            states, result
        )
        update = deep_merge(update, additional_listeners)

        sequence_elongations = result.sequenceElongation
        n_elongations = result.nReactions

        # Look-ahead AA count (base returns 0).
        next_amino_acid_count = self.next_amino_acids(
            all_sequences, sequence_elongations
        )

        # ---- Ribosome mass + position updates ----
        # Swap to the codon-based sequence table for mass accounting.
        sequences = self.sequences(sequences)
        added_protein_mass = computeMassIncrease(
            sequences,
            sequence_elongations,
            self.monomer_weights_incorporated,
        )

        updated_lengths = peptide_lengths + sequence_elongations
        updated_positions_on_mRNA = positions_on_mRNA + 3 * sequence_elongations

        did_initialize = (sequence_elongations > 0) & (peptide_lengths == 0)
        added_protein_mass[did_initialize] += self.endWeight

        # ---- Termination ----
        terminal_lengths = self.protein_lengths[protein_indexes]
        did_terminate = updated_lengths == terminal_lengths
        terminated_proteins = np.bincount(
            protein_indexes[did_terminate],
            minlength=self.protein_sequences.shape[0],
        )

        # MODEL-SPECIFIC: cleave N-terminal Met from MAP substrates that
        # actually have capacity this tick.
        (
            did_terminate,
            terminated_proteins,
            initial_methionines_cleaved,
            additional_listeners,
        ) = self.protein_maturation(
            states, did_terminate, terminated_proteins, protein_indexes
        )
        update = deep_merge(update, additional_listeners)

        # ---- Apply ribosome updates ----
        (protein_mass,) = attrs(states["active_ribosome"], ["massDiff_protein"])
        update["active_ribosome"].update(
            {
                "delete": np.where(did_terminate)[0],
                "set": {
                    "massDiff_protein": protein_mass + added_protein_mass,
                    "peptide_length": updated_lengths,
                    "pos_on_mRNA": updated_positions_on_mRNA,
                },
            }
        )

        update["bulk"].append((self.monomer_idx, terminated_proteins))
        states["bulk"][self.monomer_idx] += terminated_proteins

        n_terminated = int(did_terminate.sum())
        n_initialized = int(did_initialize.sum())

        update["bulk"].append((self.ribosome30S_idx, n_terminated))
        update["bulk"].append((self.ribosome50S_idx, n_terminated))
        states["bulk"][self.ribosome30S_idx] += n_terminated
        states["bulk"][self.ribosome50S_idx] += n_terminated

        # MODEL-SPECIFIC: emit charging + maturation bulk deltas.
        net_charged, aa_count_diff, evolve_update = self.evolve(
            states,
            allocated_aas,
            aas_used,
            next_amino_acid_count,
            n_elongations,
            n_initialized,
            net_charged,
            result.monomerUsages,
            initial_methionines_cleaved,
        )

        evolve_bulk_update = evolve_update.pop("bulk")
        update = deep_merge(update, evolve_update)
        update["bulk"].extend(evolve_bulk_update)

        update["polypeptide_elongation"]["aa_count_diff"] = aa_count_diff
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = (
            self.gtpPerElongation * n_elongations
        )

        # ---- Listener emission ----
        curr_elong_rate = (
            sequence_elongations.sum() / n_active_ribosomes
        ) / states["timestep"]

        update["listeners"]["growth_limits"]["net_charged"] = net_charged
        update["listeners"]["growth_limits"]["aas_used"] = aas_used
        update["listeners"]["growth_limits"]["aa_count_diff"] = aa_count_diff

        ribo = update["listeners"].setdefault("ribosome_data", {})
        ribo["effective_elongation_rate"] = (
            curr_elong_rate * units.amino_acid / units.s
        )
        ribo["aa_count_in_sequence"] = monomer_count_in_sequence_in_aas
        ribo["aa_counts"] = monomer_limit_in_aas
        ribo["actual_elongations"] = sequence_elongations.sum()
        ribo["actual_elongation_hist"] = np.histogram(
            sequence_elongations, bins=np.arange(0, 23)
        )[0]
        ribo["elongations_non_terminating_hist"] = np.histogram(
            sequence_elongations[~did_terminate], bins=np.arange(0, 23)
        )[0]
        ribo["did_terminate"] = int(did_terminate.sum())
        ribo["termination_loss"] = int(
            (terminal_lengths - peptide_lengths)[did_terminate].sum()
        )
        ribo["num_trpA_terminated"] = terminated_proteins[self.trpAIndex]
        ribo["process_elongation_rate"] = (
            self.ribosomeElongationRate / states["timestep"]
        )

        return update

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        """Not used by the kinetic model — see :meth:`evolve_state` override.

        Base's ``evolve_state`` calls this to decide what AAs are available
        for amino-acid-based polymerize. The kinetic model overrides
        ``evolve_state`` to use codon-based ``monomer_limit`` instead, so this
        method is never reached. Raising guarantees that any future refactor
        that accidentally re-routes through base's path surfaces the
        architectural mismatch loudly.
        """
        raise NotImplementedError(
            "KineticTrnaChargingPolypeptideElongation.final_amino_acids is not "
            "used — the kinetic model overrides evolve_state to use codon-based "
            "monomer_limit instead."
        )

    def evolve(
        self,
        states,
        total_aa_counts,
        amino_acids_used,
        next_amino_acid_count,
        n_elongations,
        n_initialized,
        net_charged,
        monomer_usages,
        initial_methionines_cleaved,
    ):
        """Apply bulk deltas from elongation + charging + maturation.

        Builds the ``update["bulk"]`` deltas:

        * Initialization water (n_initialized molecules of water consumed at
          translation start).
        * Net tRNA charging deltas (uncharged ↔ charged).
        * Amino acids used (total consumption from the pool).
        * ATP/AMP/PPi for each net charging event (one cycle each).
        * Proton release for each residue incorporated by a charged tRNA.
        * Water release for residues incorporated directly from the AA pool
          (the remaining elongations after subtracting charged-tRNA-mediated
          ones).
        * Water consumption + Met release for each cleaved initial Met.

        Port of upstream ``KineticTrnaChargingModel.evolve`` (lines 2860–2905).
        Signature differs from v2ecoli's base ``evolve`` — three extra args
        (``net_charged``, ``monomer_usages``, ``initial_methionines_cleaved``)
        come from :meth:`reconcile` and :meth:`protein_maturation` in the
        kinetic model's :meth:`evolve_state` override.
        """
        update = {"bulk": []}

        # Initialization water (one per newly-initialized polypeptide).
        update["bulk"].append((self.water_idx, -int(n_initialized)))

        # Net tRNA charging deltas.
        update["bulk"].append((self.uncharged_trna_idx, -net_charged))
        update["bulk"].append((self.charged_trna_idx, net_charged))

        # Amino acids used.
        update["bulk"].append((self.amino_acid_idx, -amino_acids_used))

        # Each NET (not absolute) charging event uses one ATP.
        atp_used = int(np.maximum(net_charged, 0).sum())
        update["bulk"].append((self.atp_idx, -atp_used))
        update["bulk"].append((self.amp_idx, atp_used))
        update["bulk"].append((self.ppi_idx, atp_used))

        # Each NET (not absolute) charged-tRNA-mediated incorporation
        # releases one proton.
        residues_incorporated = int(abs(np.minimum(net_charged, 0)).sum())
        update["bulk"].append((self.proton_idx, residues_incorporated))

        # Remaining elongation events (directly from AA pool) release one
        # water per peptide bond formed.
        update["bulk"].append(
            (self.water_idx, int(n_elongations - residues_incorporated))
        )

        # Initial-methionine cleavage by MAP: consumes one water, releases
        # one MET per cleavage.
        update["bulk"].append(
            (self.water_idx, -int(initial_methionines_cleaved))
        )
        update["bulk"].append(
            (self.met_idx, int(initial_methionines_cleaved))
        )

        return net_charged, {}, update

    # ---------- internal helpers ----------

    def run_model(self, codons, attr, states):
        """Simulate kinetic charging + codon reading; predict deltas.

        Drives a stiff RK45 IVP over the cell's molecules buffer (the
        ``slice_*`` segments set up in :meth:`initialize`). On each
        ``attr="bulk_total"`` call (during :meth:`request`) we also pre-compute
        the cell-volume-scaled Michaelis constants and the cell-AA saturation
        for the subsequent ``"bulk"`` call to reuse.

        Returns:

            (amino_acids_used, codons_read, free_trnas, charged_trnas,
             chargings, codons_to_trnas_matrix, listeners)

        all as ``np.int64`` ndarrays (post-discretization) except ``listeners``
        which is a nested dict of float arrays.

        Port of upstream ``KineticTrnaChargingModel.run_model`` (lines
        2413–2713). Differences:

        * ``self.process.X_idx`` → ``self.X_idx``.
        * Uses :mod:`scipy.integrate.solve_ivp` with RK45 (rtol=1e-4,
          atol=1e-7) matching upstream.
        """
        listeners: dict = {}

        # Free-variable closures into the inner ODE: we need
        # ``amino_acid_availability`` from outside. Defined before the inner
        # def so the closure captures it.
        free_trnas_input = counts(states[attr], self.uncharged_trna_idx)
        charged_trnas_input = counts(states[attr], self.charged_trna_idx)
        amino_acid_availability = counts(states[attr], self.amino_acid_idx)

        def ode_model(
            t,
            molecules,
            target_codon_rate,
            v_max,
            cell_amino_acid_saturation,
            K_M_amino_acids,
            K_M_trnas,
            amino_acid_limit,
        ):
            # Parse molecules buffer (the slice layout was set up in
            # initialize so we don't pay the dict-lookup cost per-step).
            free_trnas = molecules[self.slice_free_trnas]
            charged_trnas = molecules[self.slice_charged_trnas]
            amino_acids_remaining = molecules[self.slice_amino_acids]

            # Adjust target codon reading rate when charged fraction is low
            # — sin() roll-off prevents the rate from saturating at the
            # discontinuity at 0.
            fraction_charged = (
                self.trnas_to_codons @ charged_trnas
                / (
                    self.trnas_to_codons @ charged_trnas
                    + self.trnas_to_codons @ free_trnas
                )
            )
            needs_adjustment = fraction_charged < 0.05
            adjustment = np.ones_like(target_codon_rate)
            adjustment[needs_adjustment] = np.sin(
                10 * np.pi * fraction_charged[needs_adjustment]
            )
            adjusted_codon_rate = np.multiply(adjustment, target_codon_rate)

            # Adjust AA saturation when remaining pool is low (sin² roll-off
            # since the AA pool can hit exactly 0).
            mask = amino_acid_availability > 0
            fraction_remaining = np.zeros_like(amino_acids_remaining)
            fraction_remaining[mask] = (
                amino_acids_remaining[mask] / amino_acid_availability[mask]
            )
            needs_adjustment = fraction_remaining < 0.05
            adjustment = np.ones_like(cell_amino_acid_saturation)
            adjustment[needs_adjustment] = np.square(
                np.sin(10 * np.pi * fraction_remaining[needs_adjustment])
            )
            adjusted_amino_acid_saturation = np.multiply(
                adjustment, cell_amino_acid_saturation
            )

            # Charge tRNAs — Michaelis-Menten competitive inhibition.
            relative_trnas = free_trnas / K_M_trnas
            charging_rate = (
                self.amino_acids_to_trnas
                @ np.multiply(v_max, adjusted_amino_acid_saturation)
                * relative_trnas
                / (
                    1
                    + (
                        self.amino_acids_to_trnas
                        @ self.trnas_to_amino_acids
                        @ relative_trnas
                    )
                )
            )

            # Distribution of codons → tRNAs (columns sum to 1).
            charged_trnas_tile = np.tile(charged_trnas, (self.n_codons, 1)).T
            codons_to_trnas = np.where(self.codons_to_trnas, charged_trnas_tile, 0)
            denominator = codons_to_trnas.sum(axis=0)
            denominator[denominator == 0] = 1  # prevent divide-by-zero
            codons_to_trnas = codons_to_trnas / denominator

            # Read codons.
            reading_rate = codons_to_trnas @ adjusted_codon_rate

            # Assemble dx/dt.
            dx_dt = np.zeros_like(molecules)
            dx_dt[self.slice_free_trnas] = -charging_rate + reading_rate
            dx_dt[self.slice_charged_trnas] = charging_rate - reading_rate
            dx_dt[self.slice_amino_acids] = -(
                self.trnas_to_amino_acids @ charging_rate
            )
            dx_dt[self.slice_charging_counter] = charging_rate
            dx_dt[self.slice_reading_counter] = reading_rate
            dx_dt[self.slice_codons_to_trnas_counter] = np.multiply(
                codons_to_trnas, np.tile(adjusted_codon_rate, (self.n_trnas, 1))
            )[self.codons_to_trnas]

            return dx_dt

        # Pre-compute cell-volume-scaled K_M and AA saturation. Only on the
        # first (bulk_total) call per tick — the bulk call reuses them.
        if attr == "bulk_total":
            self.K_M_amino_acids, self.K_M_trnas = self.get_kinetic_constants(
                states["listeners"]["mass"]["cell_mass"]
            )
            cell_amino_acids = counts(states["bulk_total"], self.amino_acid_idx)
            self.cell_amino_acid_saturation = cell_amino_acids / (
                self.K_M_amino_acids + cell_amino_acids
            )

        # Pack inputs into the molecules buffer.
        molecules_input = np.zeros(self.molecules_input_size, dtype=np.int64)
        molecules_input[self.slice_free_trnas] = free_trnas_input
        molecules_input[self.slice_charged_trnas] = charged_trnas_input
        molecules_input[self.slice_amino_acids] = amino_acid_availability

        # Run ODE.
        ode_result = solve_ivp(
            ode_model,
            [0, states["timestep"]],
            molecules_input,
            args=(
                codons / states["timestep"],
                self.max_charging_rate(states, attr),
                self.cell_amino_acid_saturation,
                self.K_M_amino_acids,
                self.K_M_trnas,
                amino_acid_availability,
            ),
            method="RK45",
            rtol=1e-4,
            atol=1e-7,
        )

        # ---- Listener: tRNA saturation + turnover (bulk only) ----
        if attr == "bulk":
            delta_t = ode_result.t[1:] - ode_result.t[:-1]

            # Time-averaged tRNA saturation.
            relative_trnas = (
                ode_result.y[self.slice_free_trnas, :] / self.K_M_trnas[:, None]
            )
            trna_saturation = relative_trnas / (
                1
                + (
                    self.amino_acids_to_trnas
                    @ self.trnas_to_amino_acids
                    @ relative_trnas
                )
            )
            average_trna_saturation = (
                np.sum(np.multiply(trna_saturation[:, 1:], delta_t), axis=1)
                / states["timestep"]
            )

            trna_charging_listener = listeners.setdefault("trna_charging", {})
            trna_charging_listener["saturation_trna"] = average_trna_saturation

            # tRNA turnover (incorporation rate / charged tRNAs).
            turnovers = []
            previous_readings = np.zeros(self.n_trnas, dtype=np.int64)
            for i in range(ode_result.t.shape[0] - 1):
                codons_to_trnas_matrix = np.zeros(
                    (self.n_trnas, self.n_codons), dtype=np.int64
                )
                codons_to_trnas_matrix[self.codons_to_trnas] = ode_result.y[
                    self.slice_codons_to_trnas_counter, i
                ]
                readings = codons_to_trnas_matrix.sum(axis=1)
                delta_readings = readings - previous_readings

                incorporation = self.trnas_to_amino_acids @ delta_readings

                charged_trnas = (
                    self.trnas_to_amino_acids
                    @ ode_result.y[self.slice_charged_trnas, i]
                )

                turnovers.append(incorporation / delta_t[i] / charged_trnas)
                previous_readings = readings

            turnovers = np.array(turnovers)
            average_turnover = (
                np.sum(np.multiply(turnovers.T, delta_t), axis=1)
                / states["timestep"]
            )
            trna_charging_listener["turnover"] = average_turnover

        # ---- Parse ODE results ----
        molecules_output = ode_result.y[:, -1]
        raw_charging = molecules_output[self.slice_charging_counter]
        raw_codons_to_trnas = molecules_output[self.slice_codons_to_trnas_counter]

        # ---- Discretize charging events ----
        # Resource-sizing (bulk_total): round up so the request envelope
        # covers the realised consumption. Evolve (bulk): stochastic round
        # so cell-population averages stay correct.
        if attr == "bulk_total":
            chargings = np.ceil(raw_charging).astype(np.int64)
        else:
            chargings = stochasticRound(
                self.random_state, raw_charging
            ).astype(np.int64)

        # Cap at AA availability — undo charging events one-by-one if any
        # AA is over-spent.
        amino_acids_used = self.trnas_to_amino_acids @ chargings
        exceeds_availability = amino_acids_used > amino_acid_availability
        if np.any(exceeds_availability):
            for i in np.where(exceeds_availability)[0]:
                n_undo = amino_acids_used[i] - amino_acid_availability[i]
                trna_indexes = np.where(self.trnas_to_amino_acids[i])[0]
                for _ in range(n_undo):
                    i_undo = np.argsort(
                        (chargings - raw_charging)[trna_indexes]
                    )[-1]
                    chargings[trna_indexes[i_undo]] -= 1
            amino_acids_used = self.trnas_to_amino_acids @ chargings
            exceeds_availability = amino_acids_used > amino_acid_availability
            assert not np.any(exceeds_availability)
        assert np.all(chargings >= 0)

        # ---- Discretize reading events ----
        if attr == "bulk_total":
            codons_to_trnas = np.ceil(raw_codons_to_trnas).astype(np.int64)
        else:
            codons_to_trnas = stochasticRound(
                self.random_state, raw_codons_to_trnas
            ).astype(np.int64)

        # Reshape from the flat trna-codon-pairs vector into the (n_trnas,
        # n_codons) matrix.
        codons_to_trnas_matrix = np.zeros(
            (self.n_trnas, self.n_codons), dtype=np.int64
        )
        codons_to_trnas_matrix[self.codons_to_trnas] = codons_to_trnas

        readings = codons_to_trnas_matrix.sum(axis=1)
        assert np.all(readings >= 0)
        codons_read = codons_to_trnas_matrix.sum(axis=0)

        # ---- Reconcile tRNA pool over/underflow ----
        free_trnas = free_trnas_input - chargings + readings
        charged_trnas = charged_trnas_input + chargings - readings

        # Free-tRNA underflow → undo a charging event per missing tRNA.
        if np.any(free_trnas < 0):
            for i in np.where(free_trnas < 0)[0]:
                n_undo = abs(free_trnas[i])
                for _ in range(n_undo):
                    chargings[i] -= 1
            assert np.all(chargings >= 0)
            free_trnas = free_trnas_input - chargings + readings
            assert np.all(free_trnas >= 0)
            amino_acids_used = self.trnas_to_amino_acids @ chargings

        # Charged-tRNA underflow → undo a reading event per missing tRNA.
        if np.any(charged_trnas < 0):
            for i in np.where(charged_trnas < 0)[0]:
                n_undo = abs(charged_trnas[i])
                codon_indexes = np.where(codons_to_trnas_matrix[i])[0]
                for _ in range(n_undo):
                    i_undo = np.argsort(
                        codons_to_trnas_matrix[i, codon_indexes]
                    )[-1]
                    codons_to_trnas_matrix[i, codon_indexes[i_undo]] -= 1
            readings = codons_to_trnas_matrix.sum(axis=1)
            assert np.all(readings >= 0)
            charged_trnas = charged_trnas_input + chargings - readings
            assert np.all(charged_trnas >= 0)

        # Recompute the realised tRNA pools after both fix-ups.
        free_trnas = free_trnas_input - chargings + readings
        charged_trnas = charged_trnas_input + chargings - readings

        return (
            amino_acids_used,
            codons_read,
            free_trnas,
            charged_trnas,
            chargings,
            codons_to_trnas_matrix,
            listeners,
        )

    def reconcile(self, states, result):
        """Reconcile the polymerize result with the kinetic-model prediction.

        After ``polymerize`` runs on the allocated AA pool, the per-codon
        usage may not match what the kinetic model expected. This method:

        1. Runs :meth:`run_model` against the ``"bulk"`` pool (actual
           allocation) to get the kinetics' prediction of codons read.
        2. If predictions differ from ``polymerize``'s realized usage,
           seeds the kernel's RNG and calls
           :func:`kinetic_charging_kernel.reconcile_via_ribosome_positions`
           to adjust ribosome positions.
        3. If disagreements remain, also calls
           :func:`kinetic_charging_kernel.reconcile_via_trna_pools` to
           rebalance tRNA pools and charging counts.

        Returns ``(result, amino_acids_used, net_charged, listeners)``.

        Port of upstream ``KineticTrnaChargingModel.reconcile`` (lines
        2739–2796). Differences:

        * ``self.process.charged_trna_idx`` → ``self.charged_trna_idx``.
        * Calls go through the v2ecoli kernel module, not the upstream
          Cython names.
        * Seeds the kernel's RNG once per reconcile (from ``self.seed`` +
          tick salt) so the kernel calls inside ``reconcile_via_*`` are
          deterministic given the same seed.
        """
        # Simulate kinetic trna charging + codon reading against the
        # actual allocation.
        (
            amino_acids_used,
            codons_read,
            free_trnas,
            charged_trnas,
            chargings,
            codons_to_trnas_matrix,
            listeners,
        ) = self.run_model(result.monomerUsages, "bulk", states)

        # Initial disagreement listener — useful for diagnosing why the
        # kinetic model and polymerize diverged this tick.
        disagreements = codons_read - result.monomerUsages
        trna_charging_listener = listeners.setdefault("trna_charging", {})
        trna_charging_listener["initial_disagreements"] = disagreements

        if not np.all(result.monomerUsages == codons_read):
            # Seed the kernel's RNG; reuses the process's RandomState seed
            # so behavior is deterministic per (seed, tick).
            kernel.seed(int(self.random_state.randint(0, 2**31 - 1)))

            # First pass: adjust ribosome positions to match the kinetic
            # model's per-codon usage.
            kernel.reconcile_via_ribosome_positions(
                result.monomerUsages,
                result.sequenceElongation,
                codons_read,
                self.longer_sequences,
                int(self.max_attempts),
            )

            # If positions can't fully reconcile, rebalance the tRNA pools.
            if not np.all(result.monomerUsages == codons_read):
                kernel.reconcile_via_trna_pools(
                    result.monomerUsages,
                    codons_read,
                    free_trnas,
                    charged_trnas,
                    chargings,
                    amino_acids_used,
                    codons_to_trnas_matrix,
                    self.trnas_to_codons,
                    self.trnas_to_amino_acid_indexes,
                )

            result.nReactions = result.monomerUsages.sum()

        # Record final charging + reading events for the listener.
        trna_charging_listener["charging_events"] = chargings
        trna_charging_listener["reading_events"] = codons_to_trnas_matrix.sum(axis=1)
        trna_charging_listener["codons_to_trnas_counter"] = codons_to_trnas_matrix[
            self.codons_to_trnas
        ]

        # Net change in charged tRNAs vs allocation.
        net_charged = charged_trnas - counts(
            states["bulk"], self.charged_trna_idx
        )

        return result, amino_acids_used, net_charged, listeners

    def protein_maturation(
        self, states, did_terminate, terminated_proteins, protein_indexes
    ):
        """Cleave N-terminal Met from MAP-substrate proteins that just terminated.

        Methionine aminopeptidase has a kinetic capacity (``k_cat = 6 / s``,
        per-cell concentration) that may not cover every terminating MAP
        substrate this tick. If supply < demand, randomly defer termination
        for the excess by flipping ``did_terminate[i] = False`` for a
        multinomial sample of the candidates.

        Returns updated ``(did_terminate, terminated_proteins, cleaved,
        listeners)``.

        Port of upstream ``KineticTrnaChargingModel.protein_maturation``
        (lines 2801–2858). Differences:

        * ``self.process.X`` → ``self.X``.
        * Unum ``* units.s`` → pint ``* units.s``;
          ``.asNumber()`` → ``.to(units.dimensionless).magnitude``.
        """
        # How many MAP substrates terminated this tick.
        n_needs_cleaving = int(terminated_proteins[self.is_map_substrate].sum())

        # MAP kinetic capacity in this tick.
        cell_volume = states["listeners"]["mass"]["cell_mass"] / self.cell_density
        v_can_cleave = (
            (1 / units.s)
            * 6  # MAP k_cat
            / self.n_avogadro
            / cell_volume
            * counts(states["bulk"], self.map_idx)
        )
        n_can_cleave_q = (
            v_can_cleave
            * (units.s * states["timestep"])
            * cell_volume
            * self.n_avogadro
        )
        # Strip units to a plain scalar (dimensionless after the cancellations).
        # The pint .magnitude can be an ndarray shape (1,) if any input is a
        # bulk count array; coerce via float(np.asarray(...).sum()) to handle
        # both shapes.
        n_can_cleave_f = float(
            np.asarray(n_can_cleave_q.to(units.dimensionless).magnitude).sum()
        )
        n_can_cleave = stochasticRound(self.random_state, n_can_cleave_f)[0]

        # Decide how many actually terminate.
        if n_can_cleave >= n_needs_cleaving:
            cleaved = n_needs_cleaving
            not_cleaved = 0
        else:
            cleaved = int(n_can_cleave)
            not_cleaved = n_needs_cleaving - cleaved

            # Defer some terminations until MAP catches up next tick.
            candidates = np.logical_and(
                did_terminate,
                np.array([self.is_map_substrate[x] for x in protein_indexes]),
            )
            n_candidates = int(candidates.sum())
            if n_candidates > 0:
                i_cannot_cleave = self.random_state.multinomial(
                    not_cleaved, candidates / n_candidates
                ).astype(bool)
                did_terminate[i_cannot_cleave] = False
                terminated_proteins = np.bincount(
                    protein_indexes[did_terminate],
                    minlength=self.protein_sequences.shape[0],
                )

        listeners = {
            "trna_charging": {
                "cleaved": int(cleaved),
                "not_cleaved": int(not_cleaved),
            }
        }
        return did_terminate, terminated_proteins, cleaved, listeners

    def monomer_to_aa(self, monomer):
        """Aggregate a per-codon count vector into a per-AA count vector.

        ``codons_to_amino_acids`` is an ``(n_aas, n_codons)`` 0/1 matrix, so
        the matmul produces the per-AA usage sum.

        Port of upstream ``KineticTrnaChargingModel.monomer_to_aa``
        (lines 2727–2728). (Pulled forward from 3e into 3d because the
        :meth:`evolve_state` override consumes it.)
        """
        return self.codons_to_amino_acids @ monomer

    def monomer_limit(self, states, monomer_count_in_sequence):
        """Per-codon cap (and the AA-domain projection) for polymerize.

        Returns the codon-domain cap stored from the kinetic prediction
        in :meth:`request` (``self.codons_kinetics_model``), plus its
        AA-domain projection so the surrounding ``evolve_state`` can log
        the AA-equivalent.

        Port of upstream ``KineticTrnaChargingModel.monomer_limit``
        (lines 2730–2734). (Pulled forward from 3e into 3d because the
        :meth:`evolve_state` override consumes it.)
        """
        return (
            self.codons_kinetics_model,
            self.codons_to_amino_acids @ self.codons_kinetics_model,
        )

    def next_amino_acids(self, all_sequences, sequence_elongations):
        """Per-AA count of the next codon each ribosome would read.

        The base elongation models return 0 (no look-ahead); kinetic
        inherits that. Listener emission in :meth:`evolve_state`
        consumes the result.

        Mirrors upstream ``BaseElongationModel.next_amino_acids``
        (lines 943–944).
        """
        return 0

    def codon_sequences_width(self, elongation_rates):
        """Sequence-table read-ahead width for the next tick.

        Returns the per-tick width cached by :meth:`elongation_rate`. The
        ``elongation_rates`` arg is the base's choice (variable elongation
        for ppGpp models); ignored here since the kinetic model fixes the
        width at the basal rate + reconciliation buffer.

        Port of upstream ``KineticTrnaChargingModel.codon_sequences_width``
        (lines 2736–2737).
        """
        return self.sequences_width

    def sequences(self, sequences):
        """Return the codon-based sequence table built in :meth:`elongation_rate`.

        Called from ``evolve_state`` with the amino-acid-based ``sequences``
        argument from the base, which the kinetic model swaps out for its
        own codon-based ``longer_sequences``. The arg is intentionally
        unused (kept for API parity with upstream).

        Port of upstream ``KineticTrnaChargingModel.sequences`` (lines
        2798–2799).
        """
        return self.longer_sequences

    def max_charging_rate(self, states, attr):
        """Per-synthetase max charging rate at the current enzyme count.

        ``v_max = k_cat * [synthetase]`` for each amino acid.

        Port of upstream ``KineticTrnaChargingModel.max_charging_rate``
        (lines 2715–2718). Differences:

        * ``self.process.trna_synthetases_for_aas_idx`` →
          ``self.synthetase_idx``.
        """
        # v2ecoli's synthetase_idx covers the full enzyme list (~22 entries
        # incl. selenocysteine specials); k_cat__per_s is per-AA (21 entries).
        # Project all-enzyme counts onto AAs via aa_from_synthetase.
        n_synthetases_all = counts(states[attr], self.synthetase_idx)
        n_synthetases_per_aa = self.parameters["aa_from_synthetase"] @ n_synthetases_all
        v_max = self.k_cat__per_s * n_synthetases_per_aa
        return v_max

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
        # cell_mass may arrive as a plain fg float (upstream contract) or as
        # a pint Quantity (v2ecoli's listener convention). Normalize.
        if hasattr(cell_mass, "to"):
            cell_mass_q = cell_mass.to(units.fg)
        else:
            cell_mass_q = cell_mass * units.fg
        cell_volume = cell_mass_q / self.cell_density
        cell_volume = np.float64(cell_volume.to(units.L).magnitude)
        K_M_amino_acids = self.K_M_amino_acid__per_L * cell_volume
        K_M_trnas = self.K_M_trna__per_L * cell_volume
        return K_M_amino_acids, K_M_trnas
