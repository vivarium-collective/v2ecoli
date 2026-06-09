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
    select_candidate / is_initial_state / get_codons_read / get_elongation_rate /
    reconcile_via_ribosome_positions / reconcile_via_trna_pools
        Stubs — implemented in tasks 2b–2e.

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


def get_initiations(
    elongations: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
    indexes: npt.NDArray[np.int64],
) -> int:
    """Port of upstream ``get_initiations``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def get_codon_at(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    ith_ribosome: int,
    relative_position: int,
    absolute_position: int = 0,
) -> int:
    """Port of upstream ``get_codon_at``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def get_candidates_to_C(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    codon_id: int,
) -> tuple[int, int]:
    """Port of upstream ``get_candidates_to_C``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def get_candidates_to_N(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    codon_id: int,
) -> tuple[int, int]:
    """Port of upstream ``get_candidates_to_N``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def select_candidate(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    relative_position: int,
    codon_id: int,
    r: int,
) -> int:
    """Port of upstream ``select_candidate``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def is_initial_state(
    initial_state: npt.NDArray[np.int32],
    state: npt.NDArray[np.int32],
) -> bool:
    """Port of upstream ``is_initial_state``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


def get_codons_read(
    sequences: npt.NDArray[np.int8],
    elongations: npt.NDArray[np.int64],
    size: int,
) -> npt.NDArray[np.int64]:
    """Port of upstream ``get_codons_read``. Implemented in Task 2b."""
    raise NotImplementedError("Task 2b")


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
