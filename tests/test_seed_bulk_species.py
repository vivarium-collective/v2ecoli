"""Tests for the drug-agnostic ``seed_bulk_species`` injection seam (HANDOFF_8).

``seed_bulk_species`` is the engine-neutral bulk-species seeder: an injected
subsystem declares the bulk species + molar masses it needs seeded (via
``injected_processes``), so the engine holds no drug knowledge. These tests
assert the fg-submass arithmetic on a synthetic columnar bulk store
(cache-light — no ParCa fixture), including the drug-target-complex case that
adds an existing species' mass. This replaces the retired mecillinam/amp_lysis
flag injection and its ``test_mecillinam_injection.py``.
"""
import numpy as np

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.sim_data import seed_bulk_species

_SUBMASS = ["rna_submass", "protein_submass", "metabolite_submass"]
_DTYPE = [("id", "U64"), ("count", "i8")] + [(c, "f8") for c in _SUBMASS]

# molar mass (g/mol) -> fg/molecule, the same conversion initialize_bulk_counts
# applies to every bulk molecule (mass.to(fg/mol) == *1e15; then / N_avogadro).
_N_A = (1 * units.avogadro_constant).to("1/mol").magnitude
_MOLAR_TO_FG = 1e15 / _N_A

_PBP2 = "EG10606-MONOMER[i]"
_PBP2_PROTEIN_FG = 4.2e-8  # arbitrary nonzero fixture mass for the target monomer


def _synthetic_bulk():
    rows = [
        (_PBP2, 100, 0.0, _PBP2_PROTEIN_FG, 0.0),   # free PBP2 (protein submass)
        ("SOMEOTHER[c]", 10, 1.5e-9, 0.0, 3.1e-9),
    ]
    return np.array(rows, dtype=_DTYPE)


def _mecillinam_specs():
    return [
        {"id": "mecillinam[p]", "molar_mass_g_per_mol": 325.426},
        {"id": "mecillinam_hydrolyzed[p]", "molar_mass_g_per_mol": 343.426},  # +H2O
        {"id": "mecillinam[p]-EG10606-MONOMER[i]",
         "molar_mass_g_per_mol": 325.426, "complex_with": _PBP2},
    ]


def test_seeds_species_at_count0_with_metabolite_submass():
    out = seed_bulk_species(_synthetic_bulk(), _mecillinam_specs())
    for name in ("mecillinam[p]", "mecillinam_hydrolyzed[p]",
                 "mecillinam[p]-EG10606-MONOMER[i]"):
        assert name in list(out["id"])
    free = out[out["id"] == "mecillinam[p]"][0]
    assert free["count"] == 0
    # the free-drug mass lands in the metabolite submass slot (fg), others zero
    assert free["metabolite_submass"] == 325.426 * _MOLAR_TO_FG
    assert free["rna_submass"] == 0.0 and free["protein_submass"] == 0.0
    hydro = out[out["id"] == "mecillinam_hydrolyzed[p]"][0]
    assert hydro["metabolite_submass"] == 343.426 * _MOLAR_TO_FG


def test_complex_mass_is_drug_plus_partner():
    bulk = _synthetic_bulk()
    out = seed_bulk_species(bulk, _mecillinam_specs())
    cplx = out[out["id"] == "mecillinam[p]-EG10606-MONOMER[i]"][0]
    free = out[out["id"] == "mecillinam[p]"][0]
    pbp2 = bulk[bulk["id"] == _PBP2][0]
    # complex mass == free-drug mass + partner (PBP2) mass, per submass column
    for col in _SUBMASS:
        assert cplx[col] == float(free[col]) + float(pbp2[col])
    # concretely: the complex carries the PBP2 protein submass it inherited
    assert cplx["protein_submass"] == _PBP2_PROTEIN_FG


def test_idempotent_and_noop_on_empty():
    bulk = _synthetic_bulk()
    once = seed_bulk_species(bulk, _mecillinam_specs())
    twice = seed_bulk_species(once, _mecillinam_specs())
    assert list(twice["id"]) == list(once["id"])          # no duplication
    assert list(once["id"]).count("mecillinam[p]") == 1
    assert seed_bulk_species(bulk, []) is bulk            # empty = untouched
    assert seed_bulk_species(bulk, None) is bulk


def test_missing_complex_partner_raises():
    import pytest
    bulk = _synthetic_bulk()
    with pytest.raises(ValueError, match="complex_with"):
        seed_bulk_species(bulk, [{"id": "X[c]", "molar_mass_g_per_mol": 1.0,
                                  "complex_with": "NOT-IN-BULK[c]"}])
