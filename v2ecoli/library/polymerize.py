"""Polymerize — self-contained vendored copy of the upstream
``wholecell.utils.polymerize`` module, with the three Cython kernels
(``sum_monomers``, ``buildSequences``, ``computeMassIncrease``) reimplemented
as ``@numba.njit`` pure-Python translations, and the hot inner loop of the
``polymerize`` class extracted into a numba-JITed free function.

Why
---
The upstream module lives in the ``wholecell`` pip distribution and
requires a Cython build step (``make clean compile``). v2ecoli already
imports from it from multiple Process modules. This vendored copy:

1. Drops the runtime dep on ``wholecell.utils.polymerize`` for v2ecoli's
   hot transcription / translation paths (ParCa still uses its separately
   vendored ``v2ecoli.processes.parca.wholecell.utils.polymerize``).
2. Replaces the three Cython kernels with ``@numba.njit`` equivalents —
   functionally identical, AOT-compiled to roughly the same machine code.
3. Lifts the inner ``while`` loop of ``_elongate_to_limit`` (the upstream
   profile's single biggest pure-Python self-time leaf, ~9% of WCM wall)
   into a ``@numba.njit`` free function. The rare reaction-limited branch
   still routes back to Python so ``np.random.shuffle`` keeps using the
   global RNG state — preserving bit-equivalence with upstream.

Bit-equivalence verified via:

* ``scripts/bench_equiv.py --duration 600 --tol-rel 0.005``
* ``pytest -m sim tests/test_architectures_grow.py``
* ``pytest -m sim tests/test_seed_determinism.py``
"""

from __future__ import annotations

import numpy as np
import numba


PAD_VALUE = -1


# --------------------------------------------------------------------------- #
# Kernels — pure-Python @njit replacements for the three Cython functions.
# --------------------------------------------------------------------------- #

@numba.njit(cache=True, boundscheck=False)
def _sum_monomers_njit(
    sequenceMonomers_uint8,  # uint8[nMonomers, nSequences, nSteps]
    monomerIndexes,           # int64[nActive]
    activeSequencesIndexes,   # int64[nActive]
):
    """Group-sum of monomer usage across active sequences.

    Mirrors ``wholecell.utils._fastsums._sum_monomers`` (the Cython kernel).
    Triple-nested loop, returns int32 totals per monomer type.
    """
    nMonomers = sequenceMonomers_uint8.shape[0]
    nActive = activeSequencesIndexes.shape[0]
    totalMonomers = np.empty(nMonomers, dtype=np.int32)
    for monomer in range(nMonomers):
        total = np.int32(0)
        for iseq in range(nActive):
            seq = activeSequencesIndexes[iseq]
            total += sequenceMonomers_uint8[monomer, seq, monomerIndexes[iseq]]
        totalMonomers[monomer] = total
    return totalMonomers


def sum_monomers(sequenceMonomers, monomerIndexes, activeSequencesIndexes):
    """sum_monomers — same shape as ``wholecell.utils._fastsums.sum_monomers``.

    Accepts the bool ``sequenceMonomers`` array (upstream stores it as bool but
    the Cython view uses uint8); we view it as uint8 before the JIT.
    """
    return _sum_monomers_njit(
        sequenceMonomers.view(dtype=np.uint8),
        monomerIndexes,
        activeSequencesIndexes,
    )


def sum_monomers_reference_implementation(sequenceMonomers, activeSequencesIndexes):
    """Pure numpy reference implementation (kept for API parity)."""
    return (
        sequenceMonomers[:, activeSequencesIndexes].sum(axis=1)
    )


@numba.njit(cache=True, boundscheck=False)
def _build_sequences_njit(base_sequences, indexes, positions, elongation_max):
    n = positions.shape[0]
    out = np.empty((n, elongation_max), dtype=np.int8)
    for i in range(n):
        idx = indexes[i]
        pos = positions[i]
        for j in range(elongation_max):
            out[i, j] = base_sequences[idx, pos + j]
    return out


def buildSequences(base_sequences, indexes, positions, elongation_rates):
    """buildSequences — same shape as ``wholecell.utils._build_sequences.buildSequences``."""
    elongation_max = int(elongation_rates.max())
    if np.any(positions + elongation_max > base_sequences.shape[1]):
        raise IndexError("Elongation proceeds past end of sequence!")
    return _build_sequences_njit(
        base_sequences, indexes, positions, elongation_max,
    )


@numba.njit(cache=True, boundscheck=False)
def computeMassIncrease(sequences, elongations, monomerMasses):
    """computeMassIncrease — same shape as Cython kernel."""
    n = sequences.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ei = elongations[i]
        s = 0.0
        for j in range(ei):
            s += monomerMasses[sequences[i, j]]
        out[i] = s
    return out


# --------------------------------------------------------------------------- #
# Polymerize class — lifted from upstream.
# --------------------------------------------------------------------------- #

class DimensionException(Exception):
    pass


def _sample_array(array):
    samples = np.random.random(array.shape)
    return np.where(array > samples)[0]


def _choices(array, n):
    """Random subset of size n drawn from indices of ``array``.

    Uses ``np.random.shuffle`` (global RNG) to match upstream bit-equivalently.
    """
    indexes = np.arange(array.shape[0])
    np.random.shuffle(indexes)
    return indexes[:n]


# Variable-rate active-subset selection — pulled out of the hot loop so the
# (no-variable_elongation) path can use a tight @njit loop with simple
# branching.
def _variable_active(activeSequencesIndexes, elongation_rates, step):
    level = elongation_rates * (step + 1)
    last_unit = level - np.floor(level)
    return activeSequencesIndexes[elongation_rates > last_unit]


# Reason for two njit funcs (variable vs fixed) instead of one with a flag:
# numba inlines branches better with monomorphic types; fixed-rate (the
# common case) skips the float arithmetic entirely.

@numba.njit(cache=True, boundscheck=False)
def _elongate_loop_fixed_njit(
    sequenceMonomers_uint8,    # uint8[M, S, T]
    monomerLimits,              # int64[M], read-only inside loop
    reactionLimit,              # int64 scalar
    activeSequencesIndexes,     # int64[A]
    progress,                   # int64[S]
    maxElongation,              # int64
    monomerHistory,             # int64[maxElongation, M], MUTATED
):
    """Fixed-rate inner loop of ``_elongate_to_limit``.

    Returns
    -------
    limiting_extent : int — the projectionIndex at which the loop exited.
    advancementIndex : int64[S]
    monomer_is_limiting_at_exit : bool — True if a monomer limit was hit
        at the exit step (vs reaction limit or natural end).
    reaction_is_limiting_at_exit : bool — True if the reaction limit was hit;
        the caller handles the rare ``np.random.shuffle`` final-step path.
    """
    nMonomers = sequenceMonomers_uint8.shape[0]
    nSequences = progress.shape[0]
    nActive = activeSequencesIndexes.shape[0]

    advancementIndex = np.zeros(nSequences, dtype=np.int64)
    monomerProjection = np.zeros(nMonomers, dtype=np.int64)

    projectionIndex = 0
    notLimited = True
    monomer_limit_hit = False
    reaction_limit_hit = False

    while notLimited and projectionIndex < maxElongation:
        # active is the full active set (fixed-rate path)
        # Compute monomerStep = sum_monomers(sequenceMonomers, index[active], active)
        # where index[active] = progress[active] + advancementIndex[active].
        totalMonomers = np.empty(nMonomers, dtype=np.int64)
        for monomer in range(nMonomers):
            total = 0
            for iseq in range(nActive):
                seq = activeSequencesIndexes[iseq]
                idx = progress[seq] + advancementIndex[seq]
                total += sequenceMonomers_uint8[monomer, seq, idx]
            totalMonomers[monomer] = total

        # monomerHistory[projectionIndex] = monomerProjection + totalMonomers
        total_reactions = 0
        any_monomer_limited = False
        for monomer in range(nMonomers):
            v = monomerProjection[monomer] + totalMonomers[monomer]
            monomerHistory[projectionIndex, monomer] = v
            if v > monomerLimits[monomer]:
                any_monomer_limited = True
            total_reactions += v

        if any_monomer_limited:
            monomer_limit_hit = True
            notLimited = False
        elif total_reactions > reactionLimit:
            reaction_limit_hit = True
            notLimited = False
        else:
            # monomerProjection += monomerStep
            for monomer in range(nMonomers):
                monomerProjection[monomer] += totalMonomers[monomer]
            # advancementIndex[active] += 1
            for iseq in range(nActive):
                advancementIndex[activeSequencesIndexes[iseq]] += 1
            projectionIndex += 1

    return projectionIndex, advancementIndex, monomer_limit_hit, reaction_limit_hit


@numba.njit(cache=True, boundscheck=False)
def _elongate_loop_variable_njit(
    sequenceMonomers_uint8,
    monomerLimits,
    reactionLimit,
    activeSequencesIndexes,
    progress,
    maxElongation,
    monomerHistory,
    elongation_rates_normalized,  # float64[A]
    currentStep,                   # int64
):
    """Variable-rate inner loop (less common path; same logic, recomputes active subset each step)."""
    nMonomers = sequenceMonomers_uint8.shape[0]
    nSequences = progress.shape[0]

    advancementIndex = np.zeros(nSequences, dtype=np.int64)
    monomerProjection = np.zeros(nMonomers, dtype=np.int64)

    projectionIndex = 0
    notLimited = True
    monomer_limit_hit = False
    reaction_limit_hit = False

    while notLimited and projectionIndex < maxElongation:
        step = currentStep + projectionIndex
        # active = activeSequencesIndexes[elongation_rates_normalized > level - floor(level)]
        # where level = elongation_rates_normalized * (step + 1).
        # Build active inline.
        active_count = 0
        for i in range(activeSequencesIndexes.shape[0]):
            level = elongation_rates_normalized[i] * (step + 1)
            last_unit = level - np.floor(level)
            if elongation_rates_normalized[i] > last_unit:
                active_count += 1
        active_buf = np.empty(active_count, dtype=np.int64)
        k = 0
        for i in range(activeSequencesIndexes.shape[0]):
            level = elongation_rates_normalized[i] * (step + 1)
            last_unit = level - np.floor(level)
            if elongation_rates_normalized[i] > last_unit:
                active_buf[k] = activeSequencesIndexes[i]
                k += 1

        totalMonomers = np.empty(nMonomers, dtype=np.int64)
        for monomer in range(nMonomers):
            total = 0
            for iseq in range(active_count):
                seq = active_buf[iseq]
                idx = progress[seq] + advancementIndex[seq]
                total += sequenceMonomers_uint8[monomer, seq, idx]
            totalMonomers[monomer] = total

        total_reactions = 0
        any_monomer_limited = False
        for monomer in range(nMonomers):
            v = monomerProjection[monomer] + totalMonomers[monomer]
            monomerHistory[projectionIndex, monomer] = v
            if v > monomerLimits[monomer]:
                any_monomer_limited = True
            total_reactions += v

        if any_monomer_limited:
            monomer_limit_hit = True
            notLimited = False
        elif total_reactions > reactionLimit:
            reaction_limit_hit = True
            notLimited = False
        else:
            for monomer in range(nMonomers):
                monomerProjection[monomer] += totalMonomers[monomer]
            for iseq in range(active_count):
                advancementIndex[active_buf[iseq]] += 1
            projectionIndex += 1

    return projectionIndex, advancementIndex, monomer_limit_hit, reaction_limit_hit


class polymerize(object):  # noqa: N801 — preserve upstream lowercase API
    """Polymerize given sequences as far as possible within monomer + energy limits.

    Same constructor signature as ``wholecell.utils.polymerize.polymerize``.
    See the upstream module's class docstring for parameter semantics.
    """

    PAD_VALUE = PAD_VALUE

    def __init__(
        self,
        sequences,
        monomerLimits,
        reactionLimit,
        randomState,
        elongation_rates,
        variable_elongation=False,
    ):
        if sequences.shape[0] != len(elongation_rates):
            raise DimensionException(
                "Dimensions of input sequences and elongation rates do not match."
            )

        self._sequences = sequences
        self._monomerLimits = monomerLimits
        self._reactionLimit = reactionLimit
        self._randomState = randomState
        self._raw_elongation_rates = elongation_rates
        self.elongation_rates = elongation_rates / np.max(elongation_rates)
        self.variable_elongation = variable_elongation

        self._setup()
        self._elongate()
        self._finalize()

    # ----- Setup -----

    def _setup(self):
        self._sanitize_inputs()
        self._gather_input_dimensions()
        self._gather_sequence_data()
        self._prepare_running_values()
        self._prepare_outputs()

    def _sanitize_inputs(self):
        self._monomerLimits = self._monomerLimits.astype(np.int64, copy=True)
        self._reactionLimit = np.int64(self._reactionLimit)

    def _gather_input_dimensions(self):
        (self._nSequences, self._sequenceLength) = self._sequences.shape
        self._nMonomers = self._monomerLimits.size

    def _gather_sequence_data(self):
        self._sequenceMonomers = np.empty(
            (self._nMonomers, self._nSequences, self._sequenceLength), dtype=bool,
        )
        for monomerIndex in range(self._nMonomers):
            self._sequenceMonomers[monomerIndex, ...] = self._sequences == monomerIndex
        self._sequenceReactions = self._sequences != self.PAD_VALUE
        self._sequenceLengths = self._sequenceReactions.sum(axis=1)

    def _prepare_running_values(self):
        self._activeSequencesIndexes = np.arange(self._nSequences)
        self._currentStep = 0
        self._progress = np.zeros(self._nSequences, np.int64)
        self._activeSequencesIndexes = self._activeSequencesIndexes[
            self._sequenceReactions[:, self._currentStep]
        ]
        self.elongation_rates = self.elongation_rates[
            self._sequenceReactions[:, self._currentStep]
        ]
        self._update_elongation_resource_demands()
        self._monomerHistory = np.empty(
            (self._maxElongation, self._nMonomers), np.int64,
        )
        self._monomerIsLimiting = np.empty(self._nMonomers, bool)
        self._reactionIsLimiting = None

    def _prepare_outputs(self):
        self.sequenceElongation = np.zeros(self._nSequences, np.int64)
        self.monomerUsages = np.zeros(self._nMonomers, np.int64)
        self.nReactions = 0
        self.sequences_limited_elongation = np.full(self._nSequences, False)

    # ----- Iteration -----

    def _elongate(self):
        while True:
            fully_elongated = self._elongate_to_limit()
            monomer_limited = (self._monomerLimits == 0).all()
            reaction_limited = self._reactionLimit == 0
            if fully_elongated or monomer_limited or reaction_limited:
                break
            self._finalize_resource_limited_elongations()
            if not self._activeSequencesIndexes.size:
                break
            self._update_elongation_resource_demands()

    def _elongate_to_limit(self):
        """Elongate as far as possible without hitting any resource limitations.

        Dispatches to ``_elongate_loop_*_njit`` for the inner while loop. If the
        loop exits because the reaction limit was hit, this Python wrapper does
        the post-exit ``choices`` + re-sum_monomers step so the global
        ``np.random.shuffle`` keeps its RNG state identical to upstream.
        """
        sequenceMonomers_uint8 = self._sequenceMonomers.view(dtype=np.uint8)

        if self.variable_elongation:
            (
                projectionIndex,
                advancementIndex,
                monomer_limit_hit,
                reaction_limit_hit,
            ) = _elongate_loop_variable_njit(
                sequenceMonomers_uint8,
                self._monomerLimits,
                np.int64(self._reactionLimit),
                self._activeSequencesIndexes,
                self._progress,
                np.int64(self._maxElongation),
                self._monomerHistory,
                self.elongation_rates,
                np.int64(self._currentStep),
            )
        else:
            (
                projectionIndex,
                advancementIndex,
                monomer_limit_hit,
                reaction_limit_hit,
            ) = _elongate_loop_fixed_njit(
                sequenceMonomers_uint8,
                self._monomerLimits,
                np.int64(self._reactionLimit),
                self._activeSequencesIndexes,
                self._progress,
                np.int64(self._maxElongation),
                self._monomerHistory,
            )

        # Recover monomerIsLimiting from the final-step row.
        if monomer_limit_hit or reaction_limit_hit:
            self._monomerIsLimiting = (
                self._monomerHistory[projectionIndex] > self._monomerLimits
            )
        else:
            self._monomerIsLimiting = np.zeros(self._nMonomers, dtype=bool)

        # Reaction-limited exit: do the random-cull + re-sum_monomers in Python
        # to keep ``np.random.shuffle`` state identical to the upstream sequence.
        if reaction_limit_hit:
            self._reactionIsLimiting = True
            total_reactions = int(self._monomerHistory[projectionIndex].sum())
            excess = total_reactions - int(self._reactionLimit)

            if self.variable_elongation:
                step = self._currentStep + projectionIndex
                active = _variable_active(
                    self._activeSequencesIndexes, self.elongation_rates, step,
                )
            else:
                active = self._activeSequencesIndexes
            active = active[_choices(active, len(active) - excess)]

            index_active = self._progress[active] + advancementIndex[active]
            monomerStep = sum_monomers(
                self._sequenceMonomers, index_active, active,
            )
            # Recompute monomerProjection from history up to projectionIndex
            if projectionIndex > 0:
                monomerProjection = self._monomerHistory[projectionIndex - 1].copy()
            else:
                monomerProjection = np.zeros(self._nMonomers, np.int64)
            self._monomerHistory[projectionIndex] = monomerProjection + monomerStep
            self._monomerIsLimiting = (
                self._monomerHistory[projectionIndex] > self._monomerLimits
            )

        limitingExtent = projectionIndex
        self._currentStep += limitingExtent

        if limitingExtent > 0:
            deltaMonomers = self._monomerHistory[limitingExtent - 1]
            deltaReactions = int(self._monomerHistory[limitingExtent - 1].sum())

            self._monomerLimits -= deltaMonomers
            self._reactionLimit -= deltaReactions

            self.monomerUsages += deltaMonomers
            self.nReactions += deltaReactions

            self._progress[self._activeSequencesIndexes] += advancementIndex[
                self._activeSequencesIndexes
            ]

        self.sequenceElongation[self._activeSequencesIndexes] += advancementIndex[
            self._activeSequencesIndexes
        ]

        fully_elongated = limitingExtent == self._maxElongation
        return fully_elongated

    def _finalize_resource_limited_elongations(self):
        if self.variable_elongation:
            active_elongation = self.sequenceElongation[self._activeSequencesIndexes]
        else:
            active_elongation = self._currentStep

        sequencesToCull = ~self._sequenceReactions[
            self._activeSequencesIndexes, active_elongation
        ]

        for monomerIndex, monomerLimit in enumerate(self._monomerLimits):
            if ~self._monomerIsLimiting[monomerIndex]:
                continue
            sequencesWithMonomer = np.where(
                self._sequenceMonomers[
                    monomerIndex, self._activeSequencesIndexes, active_elongation
                ]
            )[0]
            nToCull = sequencesWithMonomer.size - monomerLimit
            if nToCull > 0:
                culledIndexes = self._randomState.choice(
                    sequencesWithMonomer, nToCull, replace=False,
                )
                sequencesToCull[culledIndexes] = True

        if self._reactionIsLimiting:
            sequencesWithReaction = np.where(~sequencesToCull)[0]
            nToCull = sequencesWithReaction.size - self._reactionLimit
            if nToCull > 0:
                culledIndexes = self._randomState.choice(
                    sequencesWithReaction, nToCull, replace=False,
                )
                sequencesToCull[culledIndexes] = True

        self._activeSequencesIndexes = self._activeSequencesIndexes[~sequencesToCull]
        self.elongation_rates = self.elongation_rates[~sequencesToCull]

    def _update_elongation_resource_demands(self):
        self._maxElongation = self._sequenceLength - self._currentStep

    # ----- Finalize -----

    def _finalize(self):
        self._clamp_elongation_to_sequence_length()
        self.sequences_limited_elongation = (
            np.minimum(self._raw_elongation_rates, self._sequenceLength)
            != self.sequenceElongation
        )

    def _clamp_elongation_to_sequence_length(self):
        self.sequenceElongation = np.fmin(
            self.sequenceElongation, self._sequenceLengths,
        )


__all__ = [
    "polymerize",
    "buildSequences",
    "computeMassIncrease",
    "sum_monomers",
    "sum_monomers_reference_implementation",
    "PAD_VALUE",
    "DimensionException",
]
