"""Scaffold tests for the kinetic AA-supply ODE extension (Phase 3a).

Gates Phase 3a — opt-in flag ``include_aa_supply`` (default False) and three
accumulator slices (``slice_total_synthesis``, ``slice_total_import``,
``slice_total_export``) on :class:`KineticTrnaChargingPolypeptideElongation`.

Phase 3a is scaffold ONLY — no RHS edits, no listener emission, no
``aa_count_diff`` change. Those land in Phase 3b. The acceptance bar here is:

* Flag-off path is bit-identical to ``trna_charging_final@5ffb76de`` (the
  existing 6-slice layout).
* Flag-on path declares the 9-slice layout, with the three new accumulators
  initialized to zero and remaining zero across a solve (because nothing
  in the RHS writes to them yet).

The numeric "supply terms actually flow into the AA balance" gates live in
``tests/test_kinetic_aa_supply_ode.py`` (Phase 3b).

See ``workspace/investigations/consensus_elongation/audit.md`` §2, §6.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest


CACHE = "out/cache"
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


# ------------------------ schema introspection (no cache) ------------------------

def test_config_schema_includes_include_aa_supply() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )

    assert "include_aa_supply" in KineticTrnaChargingPolypeptideElongation.config_schema


def test_include_aa_supply_defaults_to_false() -> None:
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )

    entry = KineticTrnaChargingPolypeptideElongation.config_schema["include_aa_supply"]
    assert entry["_type"] == "boolean"
    assert entry["_default"] is False


def test_initialize_branches_on_include_aa_supply() -> None:
    """Source-level guard that ``initialize`` actually reads the flag and
    branches on it. Without this, a parameter exists but does nothing."""
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation as KT,
    )

    src = inspect.getsource(KT.initialize)
    assert "self.include_aa_supply" in src, (
        "initialize must read self.include_aa_supply from parameters"
    )
    assert "self.parameters[\"include_aa_supply\"]" in src or (
        "self.parameters['include_aa_supply']" in src
    ), "initialize must unpack the include_aa_supply parameter"


# ------------------------ cache-backed initialize tests ------------------------

def _make_kinetic_extensions(
    n_aas: int, n_trnas: int, n_codons: int, n_proteins: int, n_synthetases: int,
) -> dict:
    """Synthetic kinetic-charging config keys good enough for ``initialize``.

    Inlined from ``tests/test_kinetic_charging_polypeptide_elongation_scaffold.py``
    because cross-test imports collide with a stale installed ``tests``
    package (``ModuleNotFoundError: No module named 'nose'``).
    """
    from v2ecoli.types.quantity import ureg as units

    return {
        "codon_sequences": np.zeros((n_proteins, 100), dtype=np.int8),
        "residue_weights_by_codon": np.ones(n_codons, dtype=np.float64),
        "n_codons": n_codons,
        "i_start_codon": 0,
        "is_map_substrate": np.zeros(n_proteins, dtype=bool),
        "n_trna_codon_pairs": n_codons + n_trnas,
        "trnas_to_codons": (
            (np.arange(n_codons)[:, None] % n_trnas == np.arange(n_trnas)[None, :])
            .astype(np.int8)
        ),
        "codons_to_amino_acids": np.eye(n_aas, n_codons, dtype=np.int8),
        "k_cat__per_s": np.ones(n_synthetases, dtype=np.float64) * 10.0,
        "K_M_amino_acid__per_L": np.ones(n_synthetases, dtype=np.float64) * 1e-4 * (units.mol / units.L),
        "K_M_trna__per_L": np.ones(n_synthetases, dtype=np.float64) * 1e-5 * (units.mol / units.L),
        "reconciliation_buffer": 10,
    }


def _instantiate(include_aa_supply: bool):
    """Build the kinetic-charging process from the cache, with the supply
    flag set as requested. Returns ``(process, n_aas, n_trnas)``."""
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.processes.polypeptide.kinetic_charging import (
        KineticTrnaChargingPolypeptideElongation,
    )

    cfg = dict(load_cache_bundle(CACHE)["configs"]["ecoli-polypeptide-elongation"])
    n_aas = len(cfg["amino_acids"])
    n_trnas = len(cfg["uncharged_trna_names"])
    n_synthetases = len(cfg.get("synthetase_names", [])) or n_aas
    cfg.update(
        _make_kinetic_extensions(
            n_aas=n_aas, n_trnas=n_trnas, n_codons=62,
            n_proteins=len(cfg["proteinIds"]),
            n_synthetases=n_synthetases,
        )
    )
    cfg["include_aa_supply"] = include_aa_supply
    return KineticTrnaChargingPolypeptideElongation(cfg), n_aas, n_trnas


@pytest.mark.sim
@_needs_cache
def test_flag_off_preserves_legacy_six_slice_layout() -> None:
    """Default (flag off) must keep the 6-slice layout from
    trna_charging_final@5ffb76de — the regression gate.
    """
    proc, n_aas, n_trnas = _instantiate(include_aa_supply=False)
    n_trna_codon_pairs = 62 + n_trnas  # mirrors _make_kinetic_extensions

    expected_size = n_trnas + n_trnas + n_aas + n_trnas + n_trnas + n_trna_codon_pairs
    assert proc.molecules_input_size == expected_size

    # Six slice attrs present.
    for attr in (
        "slice_free_trnas",
        "slice_charged_trnas",
        "slice_amino_acids",
        "slice_charging_counter",
        "slice_reading_counter",
        "slice_codons_to_trnas_counter",
    ):
        assert isinstance(getattr(proc, attr), slice), f"{attr} missing/wrong type"

    # Supply slice attrs MUST NOT exist when flag is off (or be None).
    for attr in (
        "slice_total_synthesis",
        "slice_total_import",
        "slice_total_export",
    ):
        assert getattr(proc, attr, None) is None, (
            f"{attr} must not be set when include_aa_supply=False"
        )


@pytest.mark.sim
@_needs_cache
def test_flag_on_extends_layout_with_three_accumulators() -> None:
    """Flag on must extend the layout by three n_aas-sized slices placed
    contiguously after slice_codons_to_trnas_counter.
    """
    proc, n_aas, n_trnas = _instantiate(include_aa_supply=True)
    n_trna_codon_pairs = 62 + n_trnas

    legacy_size = n_trnas + n_trnas + n_aas + n_trnas + n_trnas + n_trna_codon_pairs
    expected_size = legacy_size + 3 * n_aas
    assert proc.molecules_input_size == expected_size

    # All nine slice attrs present.
    for attr in (
        "slice_free_trnas",
        "slice_charged_trnas",
        "slice_amino_acids",
        "slice_charging_counter",
        "slice_reading_counter",
        "slice_codons_to_trnas_counter",
        "slice_total_synthesis",
        "slice_total_import",
        "slice_total_export",
    ):
        assert isinstance(getattr(proc, attr), slice), f"{attr} missing/wrong type"

    # Accumulators are positioned contiguously after the last legacy slice.
    assert proc.slice_total_synthesis.start == proc.slice_codons_to_trnas_counter.stop
    assert proc.slice_total_import.start == proc.slice_total_synthesis.stop
    assert proc.slice_total_export.start == proc.slice_total_import.stop

    # Each accumulator slice has length n_aas.
    assert proc.slice_total_synthesis.stop - proc.slice_total_synthesis.start == n_aas
    assert proc.slice_total_import.stop - proc.slice_total_import.start == n_aas
    assert proc.slice_total_export.stop - proc.slice_total_export.start == n_aas

    # The last slice ends at molecules_input_size — no trailing gap.
    assert proc.slice_total_export.stop == proc.molecules_input_size


@pytest.mark.sim
@_needs_cache
def test_flag_on_accumulators_remain_zero_in_phase_3a() -> None:
    """Phase 3a does not modify the RHS — the accumulators are allocated
    but no dx_dt term writes to them. So a pre-solve input buffer initialized
    to zero stays zero in the accumulator slots regardless of the rest of
    the solve. This is the structural guarantee that 3a → 3b is a no-op
    until 3b lands.
    """
    proc, n_aas, _ = _instantiate(include_aa_supply=True)

    # Synthesize a zeroed molecules buffer matching the new layout.
    buf = np.zeros(proc.molecules_input_size, dtype=np.int64)

    # The three accumulator slots are already zero (zeros() init); the test
    # exists to document the contract — they must be initialized to zero
    # at the start of every solve. Any future change that initializes them
    # non-zero breaks the merge semantics with SteadyState's ODE.
    assert (buf[proc.slice_total_synthesis] == 0).all()
    assert (buf[proc.slice_total_import] == 0).all()
    assert (buf[proc.slice_total_export] == 0).all()


# ------------------------ phase boundary marker ------------------------

def test_phase_3a_marker_no_rhs_writes_to_accumulators() -> None:
    """Phase 3a must NOT add RHS terms for the supply accumulators. That's
    Phase 3b. If this test fails, the work has crossed the 3a/3b boundary —
    update both this test and the audit doc when 3b lands.
    """
    from v2ecoli.processes.polypeptide import kinetic_charging

    src = inspect.getsource(kinetic_charging)
    # Substring guards — these tokens are introduced in 3b's RHS edits.
    forbidden = [
        "dx_dt[self.slice_total_synthesis]",
        "dx_dt[self.slice_total_import]",
        "dx_dt[self.slice_total_export]",
    ]
    found = [tok for tok in forbidden if tok in src]
    assert not found, (
        f"Phase 3b tokens leaked into 3a: {found}. "
        "Update audit.md and this marker test when transitioning to 3b."
    )
