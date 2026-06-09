"""
Tests for the Task #5 ``LoadSimData`` extensions that plumb the kinetic
tRNA-charging keys into ``ecoli-polypeptide-elongation``'s config dict.

Covers:

* ``_kinetic_charging_extensions`` returns an empty dict when ``relation``
  is ``None`` (the pre-port cache state) — guards the soft-fail path.
* ``_kinetic_charging_extensions`` returns an empty dict when ``relation``
  is the 145-line pre-port stub (no ``codon_sequences`` attr).
* With a mocked ``relation`` carrying all the post-port attrs, the method
  emits all 12 expected keys with the right shapes / dtypes / units.
* ``get_polypeptide_elongation_config`` splatters the kinetic dict at the
  expected spot in the final config (and the source documents this).

Runtime end-to-end (loading a real cache + building a composite) waits on
Task #8's ParCa rerun, which writes a sim_data with the new Relation
methods.
"""

from __future__ import annotations

import inspect
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_mock_load_sim_data() -> Any:
    """Build a stand-in ``LoadSimData`` instance with just enough surface for
    ``_kinetic_charging_extensions`` to run."""
    from v2ecoli.library.sim_data import LoadSimData

    instance = LoadSimData.__new__(LoadSimData)

    # Minimal sim_data fixtures. ``unum_to_pint`` wants pint-coercible inputs
    # for the kinetic constants; we feed Unum-shaped objects so the boundary
    # behaves as in production.
    from wholecell.utils import units as unum_units

    n_aas = 21
    n_trnas = 44

    sim_data = types.SimpleNamespace()
    sim_data.molecule_ids = types.SimpleNamespace(start_codon="AUG-start")
    sim_data.molecule_groups = types.SimpleNamespace(
        amino_acids=[f"AA-{i}" for i in range(n_aas)]
    )
    sim_data.constants = types.SimpleNamespace(
        n_avogadro=6.022e23 / unum_units.mol
    )
    # transcription.uncharged_trna_names — used to walk the K_M_trna map.
    sim_data.process = types.SimpleNamespace(
        transcription=types.SimpleNamespace(
            uncharged_trna_names=[f"trna-{i}" for i in range(n_trnas)]
        ),
        translation=types.SimpleNamespace(
            monomer_data={
                "cleavage_of_initial_methionine": np.zeros(7, dtype=bool)
            }
        ),
    )

    instance.sim_data = sim_data
    return instance


def _make_post_port_relation(
    n_aas: int = 21, n_trnas: int = 44, n_codons: int = 62
) -> Any:
    """Build a Relation-shaped namespace with all post-port attrs populated."""
    from wholecell.utils import units as unum_units

    amino_acids = [f"AA-{i}" for i in range(n_aas)]
    synthetases = [f"synth-{i}" for i in range(n_aas)]
    relation = types.SimpleNamespace(
        codon_sequences=np.zeros((7, 100), dtype=np.int8),
        residue_weights_by_codon=np.ones(n_codons, dtype=np.float64),
        codons=["AUG-start"] + [f"codon-{i}" for i in range(n_codons - 1)],
        reconciliation_buffer=10,
        trna_codon_pairs=[f"pair-{i}" for i in range(n_codons + n_trnas)],
        trnas_to_codons=np.zeros((n_codons, n_trnas), dtype=np.int8),
        codons_to_amino_acids=np.eye(n_aas, n_codons, dtype=np.int8),
        amino_acid_to_synthetase=dict(zip(amino_acids, synthetases)),
        # Per-synthetase k_cat: Unum quantity (1/s).
        synthetase_to_k_cat={s: (10.0 / unum_units.s) for s in synthetases},
        # Per-synthetase K_A: Unum quantity (umol/L).
        synthetase_to_K_A={
            s: (1.0e-4 * unum_units.umol / unum_units.L) for s in synthetases
        },
        # Per-tRNA K_T: Unum quantity (umol/L).
        trna_to_K_T={
            f"trna-{i}": (1.0e-5 * unum_units.umol / unum_units.L)
            for i in range(n_trnas)
        },
    )
    return relation


# ------------------------ unit tests (no cache) ------------------------

def test_returns_empty_when_relation_is_none() -> None:
    instance = _make_mock_load_sim_data()
    out = instance._kinetic_charging_extensions(relation=None)
    assert out == {}


def test_returns_empty_when_relation_is_pre_port_stub() -> None:
    """Defensive check: if Relation lacks ``codon_sequences`` (the marker
    attr added by Task #6's port), we soft-fail to empty dict so existing
    composites keep working."""
    instance = _make_mock_load_sim_data()
    stub = types.SimpleNamespace()  # no codon_sequences
    out = instance._kinetic_charging_extensions(relation=stub)
    assert out == {}


def test_emits_all_12_kinetic_keys_with_post_port_relation() -> None:
    """All 12 keys land with the right shapes and dtypes."""
    instance = _make_mock_load_sim_data()
    rel = _make_post_port_relation()
    out = instance._kinetic_charging_extensions(relation=rel)

    expected_keys = {
        "codon_sequences",
        "residue_weights_by_codon",
        "n_codons",
        "i_start_codon",
        "is_map_substrate",
        "n_trna_codon_pairs",
        "trnas_to_codons",
        "codons_to_amino_acids",
        "k_cat__per_s",
        "K_M_amino_acid__per_L",
        "K_M_trna__per_L",
        "reconciliation_buffer",
    }
    assert expected_keys.issubset(out.keys()), (
        f"missing keys: {expected_keys - out.keys()}"
    )

    # Spot-check the most non-trivial derivations.
    assert out["n_codons"] == 62
    # AUG-start is the first codon in our mock relation.codons.
    assert out["i_start_codon"] == 0
    assert out["reconciliation_buffer"] == 10
    assert out["n_trna_codon_pairs"] == 62 + 44


def test_kinetic_arrays_have_expected_dtypes() -> None:
    """k_cat__per_s, K_M_amino_acid__per_L, K_M_trna__per_L are float64
    numpy arrays (the kinetic process initialize() consumes them as plain
    numpy magnitudes, not pint Quantities, so the boundary strips here)."""
    instance = _make_mock_load_sim_data()
    rel = _make_post_port_relation()
    out = instance._kinetic_charging_extensions(relation=rel)

    for key in ("k_cat__per_s", "K_M_amino_acid__per_L", "K_M_trna__per_L"):
        arr = out[key]
        assert isinstance(arr, np.ndarray), f"{key} not an array"
        assert arr.dtype == np.float64, f"{key} dtype: {arr.dtype}"

    # Length contracts: per-AA for k_cat / K_M_amino_acid, per-tRNA for
    # K_M_trna.
    n_aas = len(instance.sim_data.molecule_groups.amino_acids)
    n_trnas = len(
        instance.sim_data.process.transcription.uncharged_trna_names
    )
    assert out["k_cat__per_s"].shape == (n_aas,)
    assert out["K_M_amino_acid__per_L"].shape == (n_aas,)
    assert out["K_M_trna__per_L"].shape == (n_trnas,)


def test_k_cat_scaled_per_second() -> None:
    """k_cat values come back in 1/s, not 1/(min·...) — the unit boundary
    is exercised."""
    instance = _make_mock_load_sim_data()
    rel = _make_post_port_relation()
    out = instance._kinetic_charging_extensions(relation=rel)

    # Mock has k_cat = 10/s for every synthetase.
    assert np.allclose(out["k_cat__per_s"], 10.0)


def test_K_M_amino_acid_includes_avogadro_scaling() -> None:
    """K_M values are stored per-litre in molecule counts (concentration ×
    Avogadro). Mock has K_A = 1e-4 µmol/L; after Avogadro multiplication we
    expect ~6.022e13 / L."""
    instance = _make_mock_load_sim_data()
    rel = _make_post_port_relation()
    out = instance._kinetic_charging_extensions(relation=rel)

    expected = 1e-4 * 1e-6 * 6.022e23  # umol/L → mol/L → count/L
    # Allow ~0.1% slop from the float64 round-trip through pint.
    assert np.allclose(out["K_M_amino_acid__per_L"], expected, rtol=1e-3)


# ------------------------ source-scan tests ------------------------

def test_get_polypeptide_elongation_config_splats_kinetic_extensions() -> None:
    """Source check: the kinetic extensions land inside the config dict via
    a ``**kinetic_charging_config`` splat, not silently dropped."""
    from v2ecoli.library.sim_data import LoadSimData

    src = inspect.getsource(LoadSimData.get_polypeptide_elongation_config)
    assert "self._kinetic_charging_extensions(relation)" in src
    assert "**kinetic_charging_config" in src


def test_extension_method_documents_unum_to_pint_boundary() -> None:
    """Sanity: the docstring documents the unit-bridge crossing so future
    refactors don't accidentally bypass it."""
    from v2ecoli.library.sim_data import LoadSimData

    docstring = LoadSimData._kinetic_charging_extensions.__doc__ or ""
    assert "Unum" in docstring
    assert "pint" in docstring or "unum_to_pint" in docstring
