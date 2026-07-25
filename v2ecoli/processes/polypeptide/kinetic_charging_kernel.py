"""
NumPy + numba port of the upstream Cython tRNA-charging kernel
(``wholecell/utils/_trna_charging.pyx`` on CovertLab/vEcoli@trna_charging_final).

Layout

    seed(seed)
        Reseeds the module's RNG. Upstream calls C ``srand(seed)`` — here we
        construct a fresh ``numpy.random.RandomState`` so the kernel is
        deterministic per process without leaking into ``numpy.random``'s
        global state.

    randint_below(n)
        Returns a non-negative ``int`` strictly less than ``n``. Upstream uses
        ``rand() % n`` with a libc ``rand()``. Our RNG is different, so output
        sequences are not byte-identical with upstream — but each call is
        deterministic given ``seed(...)``.

    get_initiations / get_codon_at / get_candidates_to_C / get_candidates_to_N /
    select_candidate / is_initial_state / get_codons_read
        Deterministic kernels, ported in Task 2b. All ``@njit``ed and verified
        bit-identical against the upstream Cython kernel.

    reconcile_via_ribosome_positions / reconcile_via_trna_pools
        Stochastic kernels, ported in Tasks 2c and 2d. Plain Python orchestration
        so they can call :func:`randint_below`; the hot inner loops dispatch
        through the 2b ``@njit`` helpers. Parity tested two ways: byte-identity
        vs a committed per-RNG golden
        (``tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz``),
        plus algorithmic-invariant checks vs the upstream libc-rand golden.

    get_elongation_rate
        Deterministic kernel, ported in Task 2e. ``@njit`` binary search.
        Verified bit-identical against upstream.

Parity tests live in ``tests/test_kinetic_charging_kernel.py`` and load
``tests/fixtures/trna_charging_kernel_golden.json.gz``. Deterministic
functions compare exactly against the upstream-captured golden output;
stochastic functions assert statistical equivalence against per-RNG goldens
captured at port time (see ``workspace/investigations/trna-charging-final/
capture_kernel_golden.py``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import njit


# ---------- module-private RNG ----------

_RNG: Optional[np.random.RandomState] = None


def seed(seed: int) -> None:
    """
    Reseed the kernel's RNG. Counterpart to upstream ``seed_rng(seed)``.

    Calls before this function returns ``None`` produce nondeterministic
    output, which we treat as a programmer error: the upstream tests always
    call ``seed_rng(0)`` in ``setUp``.
    """
    global _RNG
    _RNG = np.random.RandomState(int(seed))


def randint_below(n: int) -> int:
    """
    Return an int uniformly in ``[0, n)``. Counterpart to upstream
    ``rand() % n``.

    Caller is responsible for ``n >= 1`` (matches upstream contract — modulo
    by zero is UB in C; we'd raise ZeroDivisionError here, which is louder).
    """
    if _RNG is None:
        raise RuntimeError(
            "kinetic_charging_kernel.seed(...) must be called before any "
            "stochastic kernel function."
        )
    return int(_RNG.randint(0, n))


# ---------- kernel function stubs (filled in 2b-2e) ----------


@njit(error_model="numpy")
def get_initiations(
    elongations: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
    indexes: npt.NDArray[np.int64],
) -> int:
    """
    Count ribosomes that have just initiated this tick.

    A ribosome has initiated when it has been elongated by >= 1 codon
    (``elongations[i] > 0``) but the corresponding peptide has zero
    length on entry to the tick (``lengths[i] == 0``).

    ``indexes`` is unused; kept for upstream signature parity.
    """
    n_initiations = 0
    for i in range(elongations.shape[0]):
        if elongations[i] > 0 and lengths[i] == 0:
            n_initiations += 1
    return n_initiations


@njit(error_model="numpy")
def get_codon_at(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    ith_ribosome: int,
    relative_position: int,
    absolute_position: int = 0,
) -> int:
    """
    Return the codon at ``relative_position`` offsets from ribosome
    ``ith_ribosome``'s current C-terminal codon, or ``-1`` if the position
    falls outside the sequence's defined range.

    ``absolute_position`` is a scratch slot upstream uses to skip re-declaring
    a C local; we accept it for API parity but always recompute.
    """
    absolute_position = elongations[ith_ribosome] - 1 + relative_position
    if absolute_position < 0:
        return -1
    elif absolute_position >= sequences.shape[1]:
        return -1
    else:
        return sequences[ith_ribosome, absolute_position]


@njit(error_model="numpy")
def get_candidates_to_C(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    codon_id: int,
) -> tuple[int, int]:
    """
    Scan C-ward from each ribosome's current position until at least one
    candidate ribosome has ``codon_id`` at the same relative offset.

    Returns ``(candidates, relative_position)``. If no ribosome has the codon
    anywhere C-ward, ``candidates == 0`` and ``relative_position`` is the
    last offset tested (``sequences.shape[1]``).
    """
    candidates = 0
    relative_position = 0
    for relative_position in range(1, sequences.shape[1] + 1):
        for i in range(sequences.shape[0]):
            if (
                get_codon_at(sequences, elongations, i, relative_position, 0)
                == codon_id
            ):
                candidates += 1
        if candidates > 0:
            break
    return candidates, relative_position


@njit(error_model="numpy")
def get_candidates_to_N(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    codon_id: int,
) -> tuple[int, int]:
    """
    Mirror of ``get_candidates_to_C`` scanning N-ward. ``relative_position``
    walks 0, -1, -2, ... and the function returns at the first offset where
    one or more ribosomes carry ``codon_id``.
    """
    candidates = 0
    relative_position = 0
    for relative_position in range(0, -sequences.shape[1], -1):
        for i in range(sequences.shape[0]):
            if (
                get_codon_at(sequences, elongations, i, relative_position, 0)
                == codon_id
            ):
                candidates += 1
        if candidates > 0:
            break
    return candidates, relative_position


@njit(error_model="numpy")
def select_candidate(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    relative_position: int,
    codon_id: int,
    r: int,
) -> int:
    """
    Return the index of the ``r``-th ribosome (0-indexed) whose codon at
    ``relative_position`` equals ``codon_id``.

    Deterministic given ``r``: the upstream Cython draws ``r`` via
    ``rand() % candidates`` and passes it in; this function does not touch the
    RNG itself. If no match exists or ``r`` exceeds the number of matches,
    returns ``sequences.shape[0] - 1`` (matches upstream's loop-falls-through
    semantics; callers are responsible for ensuring ``0 <= r < candidates``).
    """
    j = -1
    i = 0
    for i in range(sequences.shape[0]):
        if (
            get_codon_at(sequences, elongations, i, relative_position, 0)
            == codon_id
        ):
            j += 1
            if j == r:
                break
    return i


@njit(error_model="numpy")
def is_initial_state(
    initial_state: npt.NDArray[np.int32],
    state: npt.NDArray[np.int32],
) -> bool:
    """
    Element-wise equality predicate over two int32 arrays.

    Upstream is flagged as dead code (the C compiler emits an unused-function
    warning at build) but we keep it for API symmetry with the upstream module
    so callers porting from vEcoli don't have to special-case its absence.
    """
    for i in range(initial_state.shape[0]):
        if initial_state[i] != state[i]:
            return False
    return True


@njit(error_model="numpy")
def get_codons_read(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    size: int,
) -> npt.NDArray[np.int64]:
    """
    Aggregate codon usage across all ribosomes through their current
    ``elongations[i]`` C-terminal extent.

    Returns a length-``size`` int64 histogram indexed by codon ID.
    """
    out = np.zeros(size, dtype=np.int64)
    for i in range(elongations.shape[0]):
        for j in range(elongations[i]):
            out[sequences[i, j]] += 1
    return out


def reconcile_via_ribosome_positions(
    sequence_codons: npt.NDArray[np.int64],
    elongations: npt.NDArray[np.int64],
    kinetics_codons: npt.NDArray[np.int64],
    sequences: npt.NDArray[np.int8],
    max_attempts: int,
) -> None:
    """
    Reconcile per-codon counts between the Sequence Model (``sequence_codons``,
    derived from where ribosomes are sitting) and the Kinetic Model
    (``kinetics_codons``, the per-codon target from the tRNA charging step).

    Two-phase per attempt, repeated up to ``max_attempts`` times:

    * **Forward phase** — where ``kinetics_codons[c] > sequence_codons[c]``,
      pick a codon weighted by its deficit, then a ribosome that could read
      that codon at some C-ward relative offset (``get_candidates_to_C``), and
      advance that ribosome to the offset (filling in the missing codons along
      the way). Codons with no C-ward candidates are marked ``exhausted`` for
      the remainder of this attempt.

    * **Backward phase** — where ``kinetics_codons[c] < sequence_codons[c]``,
      pick a codon weighted by its surplus, find a ribosome with that codon at
      some N-ward offset (``get_candidates_to_N``), and retract it. There's no
      ``exhausted`` array here: a surplus means the codon must exist among the
      ribosome positions, so candidates are guaranteed.

    Mutates ``sequence_codons`` and ``elongations`` in place. ``kinetics_codons``
    is read-only. The function early-exits when ``compromise == 0`` (i.e.,
    ``sequence_codons == kinetics_codons``).

    Notes
    -----
    * The ``disagreements_remaining`` flag is initialized to ``True`` once at
      function entry. Phase 1 sets it to ``False`` on completion and never
      resets it; phase 2 resets it to ``True`` before its own loop and sets it
      to ``False`` on completion. So on attempt 2+, the forward phase is
      skipped — only the backward phase runs. This is intentional in upstream
      and ``test_reconcile_attempts_threshold`` depends on it.
    * Stochastic. Caller must :func:`seed` the module RNG first.
    """
    if _RNG is None:
        raise RuntimeError(
            "kinetic_charging_kernel.seed(...) must be called before "
            "reconcile_via_ribosome_positions."
        )

    codons = sequence_codons.shape[0]
    exhausted = np.zeros(codons, dtype=np.int8)
    disagreements_remaining = True

    for _ in range(int(max_attempts)):
        exhausted.fill(0)

        # ---- Phase 1: forward steps ----
        # Note: on attempt 2+, disagreements_remaining is False from phase 2,
        # so this loop is skipped entirely (intentional, see docstring).
        while disagreements_remaining:
            # Disagreements = sum of unmet deficits across non-exhausted codons
            disagreements = 0
            for c in range(codons):
                if kinetics_codons[c] > sequence_codons[c] and exhausted[c] == 0:
                    disagreements += int(kinetics_codons[c] - sequence_codons[c])
            if disagreements == 0:
                disagreements_remaining = False
                continue

            # Pick a codon weighted by deficit. The cumulative-sum-with-break
            # mirrors upstream's loop exactly.
            r = randint_below(disagreements)
            running = -1
            codon = -1
            for c in range(codons):
                if kinetics_codons[c] > sequence_codons[c] and exhausted[c] == 0:
                    running += int(kinetics_codons[c] - sequence_codons[c])
                    if running >= r:
                        codon = c
                        break

            # Find candidate ribosomes that could read this codon C-ward
            candidates, relative_position = get_candidates_to_C(
                sequences, elongations, codon
            )
            if candidates == 0:
                exhausted[codon] = 1
                continue

            # Pick one of them
            r = randint_below(candidates)
            selected = select_candidate(
                sequences, elongations, relative_position, codon, r
            )

            # Advance the ribosome by relative_position codons, filling in
            # sequence_codons one codon at a time as we go.
            for _step in range(int(relative_position)):
                step_codon = get_codon_at(sequences, elongations, selected, 1, 0)
                elongations[selected] += 1
                sequence_codons[step_codon] += 1

        # ---- Phase 2: backward steps ----
        disagreements_remaining = True
        while disagreements_remaining:
            disagreements = 0
            for c in range(codons):
                if kinetics_codons[c] < sequence_codons[c]:
                    disagreements += int(sequence_codons[c] - kinetics_codons[c])
            if disagreements == 0:
                disagreements_remaining = False
                continue

            r = randint_below(disagreements)
            running = -1
            codon = -1
            for c in range(codons):
                if kinetics_codons[c] < sequence_codons[c]:
                    running += int(sequence_codons[c] - kinetics_codons[c])
                    if running >= r:
                        codon = c
                        break

            # candidates is guaranteed > 0 because surplus implies the codon
            # exists somewhere in the consumed-so-far range.
            candidates, relative_position = get_candidates_to_N(
                sequences, elongations, codon
            )

            r = randint_below(candidates)
            selected = select_candidate(
                sequences, elongations, relative_position, codon, r
            )

            # Retract the ribosome. range(1, relative_position, -1) yields
            # 1 - relative_position values for relative_position <= 0. At each
            # step, read the codon being undone (offset 0 = current C-terminal)
            # BEFORE decrementing, then update both arrays.
            for _step in range(1, int(relative_position), -1):
                step_codon = get_codon_at(sequences, elongations, selected, 0, 0)
                elongations[selected] -= 1
                sequence_codons[step_codon] -= 1

        # Accept the compromise if we've reached parity; otherwise retry
        # (but phase 1 won't run again — see disagreements_remaining note).
        compromise = int(np.abs(kinetics_codons - sequence_codons).sum())
        if compromise == 0:
            break


def reconcile_via_trna_pools(
    sequence_codons: npt.NDArray[np.int64],
    kinetics_codons: npt.NDArray[np.int64],
    free_trnas: npt.NDArray[np.int64],
    charged_trnas: npt.NDArray[np.int64],
    chargings: npt.NDArray[np.int64],
    amino_acids_used: npt.NDArray[np.int64],
    codons_to_trnas_counter: npt.NDArray[np.int64],
    trnas_to_codons: npt.NDArray[np.int8],
    trnas_to_amino_acid_indexes: npt.NDArray[np.int8],
) -> None:
    """
    Reconcile remaining kinetic-vs-sequence disagreements by walking tRNA pools.

    Called after :func:`reconcile_via_ribosome_positions` has done its best
    via ribosome movement. Any remaining ``kinetics_codons[c] > sequence_codons[c]``
    deficit means the Kinetic Model committed to codon reads the Sequence Model
    can't account for. We "pull back" the Kinetic Model by undoing one codon
    read at a time, until ``kinetics_codons[c] <= sequence_codons[c]`` for all
    ``c``.

    Per iteration, one codon disagreement is picked (weighted by deficit) and
    one of two branches runs:

    **Free-tRNA branch** — at least one free tRNA reads this codon. Pick one
    weighted by free count, return it to charged form. Net: one (codon read)
    event is undone, free → charged, ``codons_to_trnas_counter[i, codon] -= 1``,
    ``kinetics_codons[codon] -= 1``.

    **Charged-tRNA branch** (no free tRNAs available) — pick a tRNA weighted
    by charged count and undo both the most recent charging *and* the codon
    read. Net: ``chargings[i] -= 1``, ``amino_acids_used[aa] -= 1``,
    ``codons_to_trnas_counter[i, codon] -= 1``, ``kinetics_codons[codon] -= 1``.
    Free/charged abundances are unchanged because the tRNA that read the codon
    is the same one whose charging is being undone — it goes free → charged →
    free, ending where it started.

    Mutates every numpy buffer in place. ``kinetics_codons`` is mutated here
    (unlike in ``reconcile_via_ribosome_positions``). ``sequence_codons`` is
    read-only.

    Stochastic. Caller must :func:`seed` the module RNG first.
    """
    if _RNG is None:
        raise RuntimeError(
            "kinetic_charging_kernel.seed(...) must be called before "
            "reconcile_via_trna_pools."
        )

    codons = sequence_codons.shape[0]
    n_trnas = trnas_to_codons.shape[1]

    while True:
        # Sum of deficits across all codons
        disagreements = 0
        for c in range(codons):
            if kinetics_codons[c] > sequence_codons[c]:
                disagreements += int(kinetics_codons[c] - sequence_codons[c])
        if disagreements == 0:
            break

        # Pick a codon weighted by its deficit
        r = randint_below(disagreements)
        running = -1
        codon = -1
        for c in range(codons):
            if kinetics_codons[c] > sequence_codons[c]:
                running += int(kinetics_codons[c] - sequence_codons[c])
                if running >= r:
                    codon = c
                    break

        # Count free tRNAs that read this codon
        candidates = 0
        for i in range(n_trnas):
            if trnas_to_codons[codon, i] == 1:
                candidates += int(free_trnas[i])

        if candidates == 0:
            # Charged-tRNA branch: undo a charging event instead of a read.
            # Invariant: at least one charged tRNA exists for this codon
            # because tRNAs live in free OR charged form; if zero are free
            # and the kinetic model committed to reading this codon, then
            # at least one charged tRNA must have been the one consumed.
            for i in range(n_trnas):
                if trnas_to_codons[codon, i] == 1:
                    candidates += int(charged_trnas[i])

            r = randint_below(candidates)
            running = -1
            selected = -1
            for i in range(n_trnas):
                if trnas_to_codons[codon, i] == 1:
                    running += int(charged_trnas[i])
                    if running >= r:
                        selected = i
                        break

            chargings[selected] -= 1
            amino_acids_used[trnas_to_amino_acid_indexes[selected]] -= 1
            codons_to_trnas_counter[selected, codon] -= 1
        else:
            # Free-tRNA branch: return one free tRNA to its charged form.
            r = randint_below(candidates)
            running = -1
            selected = -1
            for i in range(n_trnas):
                if trnas_to_codons[codon, i] == 1:
                    running += int(free_trnas[i])
                    if running >= r:
                        selected = i
                        break

            free_trnas[selected] -= 1
            charged_trnas[selected] += 1
            codons_to_trnas_counter[selected, codon] -= 1

        # Pull the Kinetic Model back by one codon read
        kinetics_codons[codon] -= 1


@njit(error_model="numpy")
def get_elongation_rate(
    sequences: npt.NDArray[np.int8],
    col: int,
    time: float,
    target: float,
) -> int:
    """
    Binary-search over column count to find the number of timesteps whose
    measured elongation rate is closest to ``target``.

    The rate at a given ``col`` is ``(# of non-(-1) entries in sequences[:, :col])
    / n_ribosomes / time``. Each search step counts that subarray, compares to
    target, and bisects ``[lower, upper]`` accordingly. Termination: the search
    proposes a ``col`` already visited.

    The post-search "snap to time-step boundary" logic mirrors upstream
    behaviorally. Upstream is the source of truth — see
    ``wholecell/utils/_trna_charging.pyx::get_elongation_rate``. Notes on
    behaviors preserved verbatim:

    * The "is best_col even?" check (``best_col % 2 == 0``) is an upstream
      heuristic that's only strictly correct for ``time_int <= 2``. For
      higher integer time values it can return the wrong rounded result, but
      we match it for byte-identity.
    * When the binary search hits ``rate == target`` and breaks without
      incrementing ``index``, the post-loop "find best" sweep ``for i in
      range(index)`` skips the just-written entry. Upstream's Cython relies
      on ``cdef int i = 0`` initialization to make this case work via the
      degenerate ``range(0)``; we initialize ``best_i = 0`` to match.
    """
    n_ribosomes = sequences.shape[0]
    n_cols = sequences.shape[1]

    # Distinct cols visited during binary search ≤ n_cols + 1. Pre-allocate.
    max_size = n_cols + 1
    attempted_steps = np.empty(max_size, dtype=np.int64)
    resulting_rates = np.empty(max_size, dtype=np.float64)

    index = 0
    lower = 0
    upper = n_cols
    ribosomes_f = float(n_ribosomes)
    min_difference = 1000.0  # upstream's "a large number"
    time_int = int(time)
    search_complete = False

    while not search_complete:
        # Count non-(-1) entries in sequences[:, :col]
        elongations = 0
        for i in range(n_ribosomes):
            for j in range(col):
                if sequences[i, j] != -1:
                    elongations += 1
        rate = elongations / ribosomes_f / time

        attempted_steps[index] = col
        resulting_rates[index] = rate

        if abs(rate - target) < min_difference:
            min_difference = abs(rate - target)

        if rate > target:
            upper = col
            col = lower + (col - lower) // 2
        elif rate < target:
            lower = col
            col = col + (upper - col) // 2
        else:
            search_complete = True
            break

        index += 1
        # Check if `col` was already attempted (termination signal)
        for k in range(index):
            if attempted_steps[k] == col:
                search_complete = True
                break

    # Find best entry's index. Tracks the loop variable to mirror upstream's
    # implicit "i retains its last value after the for loop" semantics.
    best_i = 0
    for i in range(index):
        best_i = i
        if abs(resulting_rates[i] - target) == min_difference:
            break

    best_col = attempted_steps[best_i]

    if best_col % 2 == 0:
        result = best_col // time_int
    else:
        step_floor = int(np.floor(best_col / time))
        step_ceil = int(np.ceil(best_col / time))
        rate_floor = -1.0
        rate_ceil = -1.0
        for k in range(index):
            if attempted_steps[k] == step_floor * time_int:
                rate_floor = resulting_rates[k]
            if attempted_steps[k] == step_ceil * time_int:
                rate_ceil = resulting_rates[k]
        if rate_floor == -1.0:
            elongations = 0
            count_col = step_floor * time_int
            for i in range(n_ribosomes):
                for j in range(count_col):
                    if sequences[i, j] != -1:
                        elongations += 1
            rate_floor = elongations / ribosomes_f / time
        if rate_ceil == -1.0:
            elongations = 0
            count_col = step_ceil * time_int
            for i in range(n_ribosomes):
                for j in range(count_col):
                    if sequences[i, j] != -1:
                        elongations += 1
            rate_ceil = elongations / ribosomes_f / time
        if abs(rate_floor - target) < abs(rate_ceil - target):
            result = step_floor
        else:
            result = step_ceil

    return result
