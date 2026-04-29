"""
============================
DnaA-Gated Chromosome Replication
============================

Subclass of ``ChromosomeReplication`` that replaces the mass-threshold
initiation gate with a **DnaA-ATP fraction check**. The cell initiates
a round of replication when the DnaA-ATP fraction (DnaA-ATP / total
DnaA pool) crosses the pre-initiation peak threshold from the curated
literature (~85% per Kurokawa et al. 1999; ~70% used as a tunable
default to leave headroom for noise).

Why fraction, not absolute concentration? The literature describes a
*cell-cycle pattern* in which DnaA-ATP rises to a peak just before
initiation, RIDA drives it down, then DARS regenerates it. The
fraction-based gate naturally tracks that pattern because the
RIDA / DARS dynamics directly affect ATP / total. An absolute-
concentration gate sits above its threshold throughout the cycle and
fires almost continuously, producing runaway initiation.

The base class computes ``self.criticalMassPerOriC = cellMass /
n_oriC / criticalInitiationMass`` in ``_prepare`` and gates initiation
on ``criticalMassPerOriC >= 1.0`` in ``_evolve``. This subclass calls
``super()._prepare()`` to keep all the other request-side bookkeeping
(bulk-index caching, replisome subunit accounting, dNTP requests) and
then overwrites ``self.criticalMassPerOriC`` with a fraction-derived
ratio before ``_evolve`` consumes it. The ratio remains a dimensionless
pint Quantity so the existing listener emission
(``critical_mass_per_oriC``) keeps working — it now reads as
``observed_atp_fraction / threshold_atp_fraction``.

Phase 3 of the replication-initiation work — pairs with Phase 5 (RIDA)
and Phase 7 (DARS) which set up the cell-cycle DnaA-ATP dynamics that
drive this gate.

Reference (curated PDF):
    Speck C, Weigel C, Messer W. ATP- and ADP-DnaA protein, a molecular
    switch in gene regulation. EMBO J. 18(21):6169–6176 (1999).
"""

from __future__ import annotations

from v2ecoli.data.replication_initiation import (
    DNAA_ADP_BULK_ID, DNAA_APO_BULK_ID, DNAA_ATP_BULK_ID,
)
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.processes.chromosome_replication import ChromosomeReplication
from v2ecoli.types.quantity import ureg as units


# Default DnaA-ATP-per-oriC threshold for cooperative low-affinity
# oriC site assembly. With initial DnaA-ATP ≈ 100 and n_oriC = 2 at
# steady-state cell-cycle entry, ~50 DnaA-ATP per oriC is the rough
# crossover. After initiation (n_oriC doubles), the per-oriC value
# halves and the gate closes — exactly the self-limiting feedback
# the mass-per-oriC gate provided.
DEFAULT_DNAA_ATP_PER_ORIC_THRESHOLD: float = 60.0


class DnaAGatedChromosomeReplication(ChromosomeReplication):
    """ChromosomeReplication with a DnaA-ATP-per-oriC initiation gate.

    The gate is ``(DnaA-ATP count) / n_oriC >= threshold``, mirroring
    the structure of the baseline mass-per-oriC gate but driving
    initiation off the DnaA-ATP pool instead of cell mass. The per-
    oriC division gives the same self-limiting feedback as the mass
    gate: after initiation, oriC count doubles and the per-oriC value
    drops below threshold, closing the gate until the cell grows /
    DARS regenerates enough new DnaA-ATP to refill the per-oriC pool.
    """

    # Keep the baseline step name so the architecture's existing wiring
    # continues to work without any topology changes.
    name = 'ecoli-chromosome-replication'

    config_schema = {
        **ChromosomeReplication.config_schema,
        'dnaA_atp_per_oric_threshold': {
            '_type': 'float',
            '_default': DEFAULT_DNAA_ATP_PER_ORIC_THRESHOLD,
        },
        # Phase 4 — SeqA sequestration window. After an initiation
        # event, the gate is forced shut for this many seconds,
        # modeling SeqA binding to hemimethylated GATC sites at the
        # newly-replicated origin (~10 min in fast-growth E. coli per
        # the curated reference). Default 0 means no sequestration —
        # set to 600.0 by the architecture when enable_seqA_
        # sequestration=True.
        'seqA_sequestration_window_s': {
            '_type': 'float',
            '_default': 0.0,
        },
    }

    def initialize(self, config):
        super().initialize(config)
        self.dnaA_atp_per_oric_threshold = float(self.parameters.get(
            'dnaA_atp_per_oric_threshold',
            DEFAULT_DNAA_ATP_PER_ORIC_THRESHOLD))
        self.seqA_sequestration_window_s = float(self.parameters.get(
            'seqA_sequestration_window_s', 0.0))
        self._dnaA_atp_idx = None
        # Phase 4 state
        self._previous_n_oric: int | None = None
        self._last_initiation_time_s: float | None = None

    def update(self, states, interval=None):
        # Detect an initiation event from the previous tick: if oriC
        # count jumped between this tick's start and last tick's start,
        # someone fired the gate.
        n_oric_now = int(states['oriCs']['_entryState'].sum())
        if (self._previous_n_oric is not None
                and n_oric_now > self._previous_n_oric):
            self._last_initiation_time_s = float(states.get('global_time', 0))
        self._previous_n_oric = n_oric_now

        self._prepare(states)
        self.criticalMassPerOriC = self._compute_dnaA_gate(states)
        return self._evolve(states)

    def _seqA_gate_closed(self, states) -> bool:
        """SeqA sequestration check — block initiation for
        ``seqA_sequestration_window_s`` seconds after the previous
        initiation event. Models SeqA binding to hemimethylated GATC
        sites at the newly-replicated origin. The SeqA protein
        (``EG12197-MONOMER``) is already expressed via the standard
        transcription / translation pipeline; this gate is the
        downstream activity that uses it. A future refinement would
        consume SeqA stoichiometrically (one bound multimer per
        sequestered origin) so SeqA scarcity can shorten the window."""
        if self.seqA_sequestration_window_s <= 0:
            return False
        if self._last_initiation_time_s is None:
            return False
        current_time = float(states.get('global_time', 0))
        elapsed = current_time - self._last_initiation_time_s
        return 0.0 <= elapsed < self.seqA_sequestration_window_s

    def _compute_dnaA_gate(self, states):
        """Returns a dimensionless pint Quantity. Initiation fires when
        the value is >= 1.0 — i.e. when ``atp_per_oric >= threshold``.
        Importing ``DNAA_APO_BULK_ID`` / ``DNAA_ADP_BULK_ID`` is kept
        for future expansion of the gate logic."""
        if self._seqA_gate_closed(states):
            # SeqA-bound origin — DnaA can't reach oriC, gate is shut.
            return 0.0 * units.dimensionless
        if self._dnaA_atp_idx is None:
            self._dnaA_atp_idx = bulk_name_to_idx(
                DNAA_ATP_BULK_ID, states['bulk']['id'])
        atp_count = int(counts(states['bulk'], self._dnaA_atp_idx))
        n_oric = int(states['oriCs']['_entryState'].sum())
        if n_oric <= 0:
            return 0.0 * units.dimensionless
        atp_per_oric = atp_count / n_oric
        ratio = atp_per_oric / self.dnaA_atp_per_oric_threshold
        return ratio * units.dimensionless
