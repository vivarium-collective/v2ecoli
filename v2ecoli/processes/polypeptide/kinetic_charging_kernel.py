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
        bit-identical against the upstream Cython kernel (see
        ``tests/test_kinetic_charging_kernel.py``).

    get_elongation_rate / reconcile_via_ribosome_positions /
    reconcile_via_trna_pools
        Stubs — implemented in tasks 2c–2e.

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
    """Port of upstream ``reconcile_via_ribosome_positions``. Implemented in Task 2c."""
    raise NotImplementedError("Task 2c")


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
    """Port of upstream ``reconcile_via_trna_pools``. Implemented in Task 2d."""
    raise NotImplementedError("Task 2d")


def get_elongation_rate(
    sequences: npt.NDArray[np.int8],
    col: int,
    time: float,
    target: float,
) -> int:
    """Port of upstream ``get_elongation_rate``. Implemented in Task 2e."""
    raise NotImplementedError("Task 2e")
