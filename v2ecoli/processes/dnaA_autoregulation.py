"""
====================
DnaA Autoregulation
====================

dnaA promoter autoregulation — closes the regulatory loop on the DnaA
nucleotide-state cycle.

Mechanism per the curated PDF: DnaA binding to the DnaA boxes upstream
of its own gene (boxes 1, 2, 3, 4, a, b, c — together spanning the p1
and p2 promoters) represses transcription of *dnaA* itself. Both
DnaA-ATP and DnaA-ADP can bind the high-affinity boxes (box1, box2);
the low-affinity boxes are DnaA-ATP-preferential. The dominant
finding in the literature is repression from one's own promoter,
though one recent study reports both positive and negative regulation
at p2.

This first-cut implementation models the feedback as a multiplicative
scale on the dnaA TU's basal_prob inside the live
``TranscriptInitiation`` step::

    basal_prob[dnaA_TU_idx] = baseline * (1 - max_repression * f_bound)

where ``f_bound = bound_dnaA_promoter / n_total_dnaA_promoter_boxes``
and ``baseline`` is captured at first-tick read-out from the existing
basal_prob array (so the original Parca-fit value is preserved across
restarts and the feedback is fully reversible). The Step does *not*
distinguish p1 vs p2 — both promoters share the same upstream box
cluster and the listener-level binding signal we have today is per-region.

Listener:
    listeners.dnaA_autoregulation.bound_dnaA_promoter
    listeners.dnaA_autoregulation.fraction_bound
    listeners.dnaA_autoregulation.repression_factor   ∈ [1 - max_repression, 1]
    listeners.dnaA_autoregulation.dnaA_basal_prob     scaled value applied this tick
    listeners.dnaA_autoregulation.dnaA_basal_prob_baseline   pre-feedback baseline

Tuning:
    The default ``max_repression = 0.7`` yields up to ~3-fold repression
    at full occupancy (1 - 0.7 = 0.3 = ~3.3× down). Tighten or loosen
    to match measured fold-repression numbers in a follow-up.

References (in the curated PDF):
    Speck, Weigel, Messer 1999, EMBO J. 18(21):6169–6176.
    Saggioro, Olliver, Sclavi 2013, Biochem. J. 449(2):333–341.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.data.replication_initiation import DNAA_PROMOTER
from v2ecoli.library.ecoli_step import EcoliStep as Step


NAME = "dnaA_autoregulation"
TOPOLOGY = {
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


# EcoCyc gene id for dnaA; transcription units that include this gene
# share its promoter (the polycistronic dnaAN-recF operon TU00259 is
# the long one, TU593 is a shorter variant). Both are repressed by
# DnaA box occupancy.
DNAA_GENE_ID = "EG10235"

# Fallback: in unit-test stubs we may pass a TU array whose ids look
# like ``TU00259[c]`` rather than gene lists. We treat this as the
# one-and-only dnaA-containing TU and look it up by its bare TU id.
KNOWN_DNAA_TU_IDS: tuple[str, ...] = ("TU00259[c]", "TU593[c]")

# Total number of DnaA boxes in the dnaA promoter region (PDF figure).
# Used to convert the binding listener's ``bound_dnaA_promoter`` count
# into a fractional occupancy.
N_DNAA_PROMOTER_BOXES: int = len(DNAA_PROMOTER.dnaA_boxes)

# Default maximum fractional repression at full box occupancy. At
# f_bound = 1, basal_prob is multiplied by (1 - DEFAULT_MAX_REPRESSION).
DEFAULT_MAX_REPRESSION: float = 0.7


class DnaAAutoregulation(Step):
    """Autoregulatory feedback on dnaA transcription.

    Reads the per-tick ``listeners.dnaA_binding.bound_dnaA_promoter``
    occupancy (emitted by the Phase 2 ``DnaABoxBinding`` step) and
    rescales the dnaA TU's ``basal_prob`` in the live
    ``TranscriptInitiation`` instance. The scale factor is
    ``1 - max_repression * f_bound``; identity when nothing is bound,
    minimum ``1 - max_repression`` at full saturation.

    The original ``basal_prob[dnaA_idx]`` value is captured the first
    time the step runs, so the feedback is fully reversible and the
    Parca-fit baseline is preserved when the architecture is rebuilt.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'max_repression': f'float{{{DEFAULT_MAX_REPRESSION}}}',
        'n_total_boxes': f'integer{{{N_DNAA_PROMOTER_BOXES}}}',
        'emit_unique': 'boolean{false}',
    }

    def initialize(self, config):
        self.max_repression = float(self.parameters.get(
            'max_repression', DEFAULT_MAX_REPRESSION))
        self.n_total_boxes = int(self.parameters.get(
            'n_total_boxes', N_DNAA_PROMOTER_BOXES))
        # Resolved on first update — needs the live TranscriptInitiation
        # instance, which is inserted into the document after this Step
        # is constructed. ``_dnaA_TU_indices`` is a list because dnaA
        # appears on >1 TU (TU00259 + TU593 in some cache versions).
        self._txi_instance = None
        self._dnaA_TU_indices: list[int] = []
        self._baseline_basal_probs: list[float] = []

    def inputs(self):
        return {
            'listeners': {
                'dnaA_binding': {
                    'bound_dnaA_promoter': {
                        '_type': 'integer', '_default': 0},
                },
            },
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'dnaA_autoregulation': {
                    'bound_dnaA_promoter': {
                        '_type': 'overwrite[integer]', '_default': []},
                    'fraction_bound': {
                        '_type': 'overwrite[float]', '_default': []},
                    'repression_factor': {
                        '_type': 'overwrite[float]', '_default': []},
                    'dnaA_basal_prob': {
                        '_type': 'overwrite[float]', '_default': []},
                    'dnaA_basal_prob_baseline': {
                        '_type': 'overwrite[float]', '_default': []},
                },
            },
        }

    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def attach_transcript_initiation(self, txi_instance) -> None:
        """Wire the autoregulation step to the live TranscriptInitiation
        instance. Called from the architecture builder once both Steps
        are in the cell_state. We resolve the dnaA TU indices here so
        the first ``update`` doesn't pay the lookup cost.

        The id column on ``rna_data`` is a per-TU string like
        ``TU00259[c]``; the dnaA-containing TUs are the polycistronic
        dnaAN-recF operon and a shorter variant. We look up the
        canonical TU ids; if neither is present (heavily customized
        cache), we fall back to a no-op repression that still emits the
        listener."""
        self._txi_instance = txi_instance
        try:
            rna_ids = [str(x) for x in txi_instance.rna_data["id"]]
        except Exception:
            rna_ids = []
        self._dnaA_TU_indices = [
            i for i, x in enumerate(rna_ids) if x in KNOWN_DNAA_TU_IDS]
        if self._dnaA_TU_indices:
            self._baseline_basal_probs = [
                float(txi_instance.basal_prob[i])
                for i in self._dnaA_TU_indices
            ]
        else:
            self._baseline_basal_probs = []

    def update(self, states, interval=None):
        bound = int(states['listeners']['dnaA_binding'].get(
            'bound_dnaA_promoter', 0) or 0)
        n_total = max(1, int(self.n_total_boxes))
        f_bound = max(0.0, min(1.0, bound / n_total))
        repression = float(np.clip(
            1.0 - self.max_repression * f_bound, 0.0, 1.0))

        baseline = (self._baseline_basal_probs[0]
                    if self._baseline_basal_probs else 0.0)
        applied = baseline * repression
        if (self._txi_instance is not None
                and self._dnaA_TU_indices
                and self._baseline_basal_probs):
            for tu_idx, base in zip(self._dnaA_TU_indices,
                                     self._baseline_basal_probs):
                self._txi_instance.basal_prob[tu_idx] = base * repression

        return {
            'listeners': {
                'dnaA_autoregulation': {
                    'bound_dnaA_promoter': bound,
                    'fraction_bound': f_bound,
                    'repression_factor': repression,
                    'dnaA_basal_prob': float(applied),
                    'dnaA_basal_prob_baseline': float(baseline),
                },
            },
        }
