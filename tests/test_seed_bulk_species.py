"""Equivalence + behavior tests for the drug-agnostic ``seed_bulk_species``
injection seam (HANDOFF_8 Workstream A).

``seed_bulk_species`` is the engine-neutral replacement for the drug-specific
``inject_antibiotic_bulk_species``: an injected subsystem declares the bulk
species + molar masses it needs seeded, so the engine holds no drug knowledge.
These tests assert the neutral path reproduces the legacy mecillinam injection
BYTE-FOR-BYTE on a synthetic columnar bulk store (cache-light — no ParCa fixture),
which is the load-bearing regression guarantee for retiring the drug flag.
"""
import numpy as np

from v2ecoli.library.parameters import param_store
from v2ecoli.library.sim_data import (
    inject_antibiotic_bulk_species,
    seed_bulk_species,
    MECILLINAM_SPECIES,
    PBP2_MONOMER_ID,
    _WATER_MOLAR_MASS,
)

_SUBMASS = ["rna_submass", "protein_submass", "metabolite_submass"]
_DTYPE = [("id", "U64"), ("count", "i8")] + [(c, "f8") for c in _SUBMASS]


def _synthetic_bulk():
    """A minimal columnar bulk store with the PBP2 target already present
    (nonzero protein submass) plus one unrelated molecule."""
    rows = [
        (PBP2_MONOMER_ID, 100, 0.0, 4.2e-8, 0.0),   # free PBP2 (protein mass)
        ("SOMEOTHER[c]", 10, 1.5e-9, 0.0, 3.1e-9),
    ]
    return np.array(rows, dtype=_DTYPE)


def _mecillinam_specs():
    """The neutral-seam declaration equivalent to LoadSimData(mecillinam=True),
    with masses read from the SAME param_store the legacy path uses."""
    mec = param_store.get(("mecillinam", "molar_mass")).magnitude
    return [
        {"id": "mecillinam[p]", "molar_mass_g_per_mol": mec},
        {"id": "mecillinam_hydrolyzed[p]", "molar_mass_g_per_mol": mec + _WATER_MOLAR_MASS},
        {"id": "mecillinam[p]-EG10606-MONOMER[i]",
         "molar_mass_g_per_mol": mec, "complex_with": PBP2_MONOMER_ID},
    ]


def test_seed_matches_legacy_mecillinam_injection_byte_for_byte():
    bulk = _synthetic_bulk()
    legacy = inject_antibiotic_bulk_species(bulk, mecillinam=True)
    neutral = seed_bulk_species(bulk, _mecillinam_specs())

    assert list(neutral["id"]) == list(legacy["id"])
    for name in MECILLINAM_SPECIES:
        assert name in list(neutral["id"])
    # Every column (count + all submasses) must match exactly.
    for col in ("count", *_SUBMASS):
        np.testing.assert_array_equal(neutral[col], legacy[col],
                                      err_msg=f"column {col} diverged")


def test_seed_is_idempotent_and_noop_on_empty():
    bulk = _synthetic_bulk()
    once = seed_bulk_species(bulk, _mecillinam_specs())
    twice = seed_bulk_species(once, _mecillinam_specs())
    assert list(twice["id"]) == list(once["id"])          # no duplication
    assert seed_bulk_species(bulk, []) is bulk            # empty = untouched
    assert seed_bulk_species(bulk, None) is bulk


def test_complex_mass_is_drug_plus_partner():
    bulk = _synthetic_bulk()
    out = seed_bulk_species(bulk, _mecillinam_specs())
    row = out[out["id"] == "mecillinam[p]-EG10606-MONOMER[i]"][0]
    pbp2 = bulk[bulk["id"] == PBP2_MONOMER_ID][0]
    mec = out[out["id"] == "mecillinam[p]"][0]
    # complex mass == free-drug mass + partner (PBP2) mass, per submass column
    for col in _SUBMASS:
        assert row[col] == float(mec[col]) + float(pbp2[col])


def test_missing_complex_partner_raises():
    import pytest
    bulk = _synthetic_bulk()
    with pytest.raises(ValueError, match="complex_with"):
        seed_bulk_species(bulk, [{"id": "X[c]", "molar_mass_g_per_mol": 1.0,
                                  "complex_with": "NOT-IN-BULK[c]"}])
