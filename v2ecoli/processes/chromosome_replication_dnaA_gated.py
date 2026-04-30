"""
============================
DnaA-Gated Chromosome Replication
============================

Subclass of ``ChromosomeReplication`` that replaces the mass-threshold
initiation gate with a **DnaA-occupancy gate at oriC**. The cell
initiates a round of replication when enough of the cooperative
low-affinity boxes at oriC are loaded with DnaA-ATP — the canonical
load-and-trigger criterion from the curated reference (Kasho, Ozaki,
Katayama 2023; Speck/Weigel/Messer 1999).

Why occupancy, not cytoplasmic concentration?
    The earlier version of this gate used a cytoplasmic
    DnaA-ATP-per-oriC ratio as a proxy. The proxy is biologically
    indirect: it fires whenever the *cell-wide* DnaA-ATP pool
    crosses a threshold per oriC, regardless of whether that DnaA-ATP
    has actually loaded onto the chromosome. That over-fired in
    tick 0 of the simulation, because the wcEcoli cache state
    arrives with a moderate DnaA-ATP pool and 2 oriC; the ratio sat
    just at the threshold and the gate fired, double-initiating the
    cell into a 4-oriC multifork state in tick 0.

    The new gate reads the per-tier oriC occupancy from the Phase 2
    binding listener: ``listeners.dnaA_binding.bound_oric_low``.
    Per the curated PDF, the high-affinity oriC boxes (R1, R2, R4)
    are bound essentially all the time, so the high tier doesn't
    distinguish "ready" from "resting". The cooperative low-affinity
    boxes (R5M, τ2, I1-I3, C1-C3 — 8 boxes total) are the actual
    initiation switch: they fill on the C1 → I3 → C2 → C3 ordering
    along the right arm of the DOR (anchored by R4-bound DnaA), with
    a second filament loading on the left arm (anchored by R1, with
    R2 helping). The DUE only unwinds once both filaments are
    assembled.

    The gate fires when ``bound_oric_low >= n_low_threshold``, with
    a default threshold of 4 of 8 (≈ "right-arm filament loaded").

Initial-state behavior
    process-bigraph merges step updates per tick and gives each step
    last-tick's view of the state at tick start. So at t=0 the gated
    step reads ``bound_oric_low = 0`` (the listener default — the
    Phase 2 binding step has not sampled yet). The gate ratio is 0
    in tick 0, no firing happens, and the cell starts in the
    cache's correct state of 1 chromosome / 2 oriC / 2 forks. From
    tick 1 onward the gate consumes the Phase 2 listener output.

Self-limiting feedback
    After initiation, oriC count doubles. The new origin is
    immediately within the SeqA refractory window (Phase 4) and
    cannot rebind DnaA, so the binding listener reports a
    short-term drop in bound_oric_low until SeqA releases. Combined
    with RIDA hydrolyzing DnaA-ATP through the ongoing replication,
    this closes the gate until the cell grows / DARS regenerates
    DnaA-ATP enough to refill the boxes again.

The base class computes ``self.criticalMassPerOriC = cellMass /
n_oriC / criticalInitiationMass`` in ``_prepare`` and gates initiation
on ``criticalMassPerOriC >= 1.0`` in ``_evolve``. This subclass calls
``super()._prepare()`` to keep all the other request-side bookkeeping
(bulk-index caching, replisome subunit accounting, dNTP requests) and
then overwrites ``self.criticalMassPerOriC`` with the occupancy-derived
ratio before ``_evolve`` consumes it. The ratio remains a dimensionless
pint Quantity so the existing listener emission
(``critical_mass_per_oriC``) keeps working — it now reads as
``bound_oric_low / n_low_threshold``.

Phase 3 of the replication-initiation work — pairs with Phase 2
(DnaA-box binding) which produces ``bound_oric_low``, Phase 4 (SeqA),
Phase 5 (RIDA), and Phase 7 (DARS).

References (curated PDF):
    Speck C, Weigel C, Messer W. ATP- and ADP-DnaA protein, a molecular
    switch in gene regulation. EMBO J. 18(21):6169–6176 (1999).
    Kasho K, Ozaki S, Katayama T. IHF and Fis as Escherichia coli cell
    cycle regulators. Int. J. Mol. Sci. 24(14):11572 (2023).
"""

from __future__ import annotations

from v2ecoli.processes.chromosome_replication import ChromosomeReplication
from v2ecoli.types.quantity import ureg as units


# Default low-affinity-box occupancy threshold. The curated PDF
# describes a right-arm filament along R4 → C1 → I3 → C2 → C3 (5 sites,
# of which R4 is high-affinity and always bound; the C1/I3/C2/C3
# quartet is the right-arm low-affinity filament) plus a left-arm
# filament anchored by R1. Setting the threshold at 4 of the 8
# low-affinity oriC boxes ≈ "right-arm filament loaded" — a
# conservative trigger that reproduces the load-and-trigger switch.
DEFAULT_BOUND_ORIC_LOW_THRESHOLD: int = 4


class DnaAGatedChromosomeReplication(ChromosomeReplication):
    """ChromosomeReplication with an oriC-occupancy initiation gate.

    The gate is ``bound_oric_low / n_low_threshold >= 1.0``, where
    ``bound_oric_low`` is the count of currently-bound low-affinity
    DnaA boxes at oriC reported by the Phase 2 binding listener. At
    typical fast-growth DnaA-ATP concentrations, the high-affinity
    boxes (R1/R2/R4) are saturated continuously and don't
    differentiate "ready" from "resting"; the cooperative low-affinity
    load is the actual trigger.
    """

    # Keep the baseline step name so the architecture's existing wiring
    # continues to work without any topology changes.
    name = 'ecoli-chromosome-replication'

    config_schema = {
        **ChromosomeReplication.config_schema,
        # Number of bound low-affinity oriC boxes required for the
        # gate to fire. Default 4 of 8 ≈ "right-arm filament loaded".
        'bound_oric_low_threshold': {
            '_type': 'integer',
            '_default': DEFAULT_BOUND_ORIC_LOW_THRESHOLD,
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
        self.bound_oric_low_threshold = max(1, int(self.parameters.get(
            'bound_oric_low_threshold',
            DEFAULT_BOUND_ORIC_LOW_THRESHOLD)))
        self.seqA_sequestration_window_s = float(self.parameters.get(
            'seqA_sequestration_window_s', 0.0))
        # Phase 4 state — track previous oriC count so we can detect
        # initiation events from the unique-store transition.
        self._previous_n_oric: int | None = None
        self._last_initiation_time_s: float | None = None

    def inputs(self):
        # Extend the base inputs with the Phase 2 binding listener
        # field that drives the gate. The default is 0 — exactly what
        # we want at t=0 before the binding step has sampled.
        base = super().inputs()
        listeners = dict(base.get('listeners', {}))
        listeners['dnaA_binding'] = {
            'bound_oric_low': {
                '_type': 'integer', '_default': 0},
        }
        base['listeners'] = listeners
        return base

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
        the value is >= 1.0 — i.e. when ``bound_oric_low >= threshold``.

        Reads ``listeners.dnaA_binding.bound_oric_low`` (emitted by the
        Phase 2 binding step) at the start of the tick. At t=0 this is
        the listener's default value of 0, so the gate does not fire
        on the cache's initial state — the cell stays at 1 chromosome
        / 2 oriC / 2 forks until the binding step has actually sampled
        the oriC occupancy at least once."""
        if self._seqA_gate_closed(states):
            # SeqA-bound origin — DnaA can't reach oriC, gate is shut.
            return 0.0 * units.dimensionless
        listeners = states.get('listeners', {}) or {}
        binding = listeners.get('dnaA_binding', {}) or {}
        bound_low = int(binding.get('bound_oric_low', 0) or 0)
        ratio = bound_low / float(self.bound_oric_low_threshold)
        return ratio * units.dimensionless
