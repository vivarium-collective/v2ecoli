"""End-to-end test: RNAPs placed at real loci, zero protrusions.

Synthesises a minimal snapshot with 20 RNAPs, runs build_model with a small
top_n so the build is fast, then asserts:
  (a) the pack contains exactly 20 rna_polymerase placements, and
  (b) zero protrusions — every placement center is inside the cell envelope.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from v2ecoli.structural import build

# Reference cache (populated by earlier build runs — no network needed).
_STRUCT_CACHE = Path(
    "/Users/eranagmon/code/v2e-pdmp-refresh/out/ecoli3d_expanded/structures"
)

# Real DATA directory for shared reference files (uniprot_map.json, genome CSV).
_REAL_DATA = Path(__file__).resolve().parent.parent.parent / "v2ecoli" / "structural" / "data"


# ── helpers ─────────────────────────────────────────────────────────────────

def _dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance from point ``p`` to the line segment ``[a, b]`` (Å)."""
    ab = b - a
    ap = p - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def pack_count_of(pack: dict, name: str) -> int:
    """Count placements of ingredient ``name`` in the pack (array8 or object format)."""
    arr8 = pack.get("placement_format") == "array8"
    iid = next((ing["id"] for ing in pack["ingredients"] if ing["name"] == name), None)
    if iid is None:
        return 0
    if arr8:
        return sum(1 for p in pack["placements"] if p[0] == iid)
    return sum(1 for p in pack["placements"] if p.get("ingredient") == iid)


# NOTE: the synthetic snapshot used by this test contains NO flagella
# (CPLX0-7452), so pack_protrusions does NOT need — and must NOT add — a
# flagella-exclusion path. The real build injects flagellar whips deliberately
# outside the envelope post-pack; excluding CPLX0-7452 here would make the
# helper silently too lenient and let a genuine interior protrusion slip past.
def pack_protrusions(pack: dict, res: dict) -> int:
    """Count non-surface placements whose center lies outside the recipe capsule.

    Reads the capsule geometry from the recipe JSON
    (``composition.cell.compartment``) and the surface ingredient ids from
    ``composition.cell.regions.surface`` (surface/membrane molecules are
    placed ON the boundary by design and are excluded from the check).
    Returns 0 when all interior/fiber/chromosome placement centers are
    within the cell envelope.
    """
    recipe = json.loads(Path(res["recipe_path"]).read_text())
    comp = recipe["composition"]["cell"]["compartment"]
    a = np.array(comp["a"], dtype=float)
    b = np.array(comp["b"], dtype=float)
    r = float(comp["radius"])

    # Collect surface ingredient names so we can skip them.
    surface_objects = {
        d["object"]
        for d in recipe["composition"]["cell"]["regions"].get("surface", [])
    }
    # Build name → integer-id mapping from the pack ingredient list.
    name_to_id = {ing["name"]: ing["id"] for ing in pack["ingredients"]}
    surface_ids = {name_to_id[n] for n in surface_objects if n in name_to_id}

    arr8 = pack.get("placement_format") == "array8"
    outside = 0
    for p in pack["placements"]:
        if arr8:
            iid = p[0]
            x, y, z = float(p[1]), float(p[2]), float(p[3])
        else:
            iid = p["ingredient"]
            x, y, z = p["position"]
        if iid in surface_ids:
            continue  # membrane molecules are placed at the surface by design
        dist = _dist_to_segment(np.array([x, y, z], dtype=float), a, b)
        if dist > r:
            outside += 1
    return outside


# ── fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rnap_build_env(tmp_path, monkeypatch):
    """Set up the minimal environment for a small RNAP-placement build."""
    # Locate the parsimony binary. Respect an existing PARSIMONY_HOME (so this
    # test is portable to CI / other machines); the laptop path is only a fallback.
    monkeypatch.setenv(
        "PARSIMONY_HOME",
        os.environ.get("PARSIMONY_HOME", "/Users/eranagmon/code/parsimony"),
    )

    # Synthetic snapshot: 20 RNAPs spread across the genome, single domain,
    # all on the forward strand.
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]),
        counts=np.array([100]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        rnap_coordinates=(np.linspace(-2.2e6, 2.2e6, 20)).astype("i8"),
        rnap_domain_index=np.zeros(20, "i4"),
        rnap_is_forward=np.ones(20, bool),
    )

    # Reference files needed by build_model (uniprot_map + genome CSV).
    for fname in ("uniprot_map.json", "ecoli_k12_genes.csv"):
        src = _REAL_DATA / fname
        if src.exists():
            shutil.copy(src, tmp_path / fname)

    # Pre-seed the structures cache so no network downloads occur.
    out = tmp_path / "pack"
    struct_cache = out / "structures"
    struct_cache.mkdir(parents=True, exist_ok=True)
    for fname in (
        "rna_polymerase.pdb",
        "dna_segment.pdb",
        "replisome.pdb",
        "70s_ribosome.cif",
        "groel.pdb",
        "eg10367_monomer.pdb",   # GAPDH (EG10367-MONOMER in CURATED)
    ):
        src = _STRUCT_CACHE / fname
        if src.exists():
            shutil.copy(src, struct_cache / fname)

    monkeypatch.setattr(build, "DATA", tmp_path)
    return out


# ── test ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_build_places_rnaps_and_confines(rnap_build_env):
    """build_model places exactly 20 RNAPs and keeps all centers inside the cell."""
    out = rnap_build_env
    res = build.build_model(str(out), state_source="snapshot", top_n=5)

    pack = json.loads(Path(res["pack_path"]).read_text())

    n_rnap = pack_count_of(pack, "rna_polymerase")
    assert n_rnap == 20, (
        f"expected 20 rna_polymerase placements, got {n_rnap}. "
        f"Ingredient list: {[i['name'] for i in pack['ingredients']]}"
    )

    n_protrusions = pack_protrusions(pack, res)
    assert n_protrusions == 0, (
        f"{n_protrusions} placements protrude outside the cell envelope."
    )
