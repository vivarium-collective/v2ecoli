"""End-to-end test: nascent RNA strands rooted at their RNAPs.

Synthesises a minimal snapshot with 1 RNAP (unique_index=7) + 3 nascent RNAs
attached to that RNAP, runs build_model with a small top_n, then asserts:
  (a) rna_segment appears in the sidecar meta
  (b) the pack contains >0 rna_segment placements
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
_REAL_DATA = (
    Path(__file__).resolve().parent.parent.parent / "v2ecoli" / "structural" / "data"
)


# ── helpers ─────────────────────────────────────────────────────────────────

def pack_count_of(pack: dict, name: str) -> int:
    """Count placements of ingredient ``name`` in the pack (array8 or object format)."""
    arr8 = pack.get("placement_format") == "array8"
    iid = next((ing["id"] for ing in pack["ingredients"] if ing["name"] == name), None)
    if iid is None:
        return 0
    if arr8:
        return sum(1 for p in pack["placements"] if p[0] == iid)
    return sum(1 for p in pack["placements"] if p.get("ingredient") == iid)


# ── fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rna_build_env(tmp_path, monkeypatch):
    """Set up the minimal environment for a small RNA-placement build."""
    # Respect an existing PARSIMONY_HOME (portable to CI / other machines).
    monkeypatch.setenv(
        "PARSIMONY_HOME",
        os.environ.get("PARSIMONY_HOME", "/Users/eranagmon/code/parsimony"),
    )

    # Synthetic snapshot: 1 RNAP (unique_index=7) at coordinate 0, domain 0,
    # plus 3 nascent RNAs all attached to that RNAP with increasing lengths.
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]),
        counts=np.array([100]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        rnap_coordinates=np.array([0], dtype="i8"),
        rnap_domain_index=np.array([0], dtype="i4"),
        rnap_is_forward=np.array([True]),
        rnap_unique_index=np.array([7], dtype="i8"),
        rna_unique_index=np.array([20, 21, 22], dtype="i8"),
        rna_RNAP_index=np.array([7, 7, 7], dtype="i8"),
        rna_transcript_length=np.array([300, 900, 1500], dtype="i8"),
        rna_is_mRNA=np.array([True, True, True]),
        rna_is_full_transcript=np.array([False, False, False]),
        rna_TU_index=np.array([1, 2, 3], dtype="i8"),
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
    for src_name, dst_name in [
        ("rna_polymerase.pdb", "rna_polymerase.pdb"),
        ("dna_segment.pdb", "dna_segment.pdb"),
        ("dna_segment.pdb", "rna_segment.pdb"),  # RNA reuses the dsDNA 1BNA mesh
        ("replisome.pdb", "replisome.pdb"),
        ("70s_ribosome.cif", "70s_ribosome.cif"),
        ("groel.pdb", "groel.pdb"),
        ("eg10367_monomer.pdb", "eg10367_monomer.pdb"),
    ]:
        src = _STRUCT_CACHE / src_name
        if src.exists():
            shutil.copy(src, struct_cache / dst_name)

    monkeypatch.setattr(build, "DATA", tmp_path)
    return out


# ── test ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_build_renders_nascent_rna(rna_build_env):
    """build_model places >0 rna_segment placements for 3 nascent RNAs."""
    out = rna_build_env
    res = build.build_model(str(out), state_source="snapshot", top_n=5)

    pack = json.loads(Path(res["pack_path"]).read_text())
    meta = json.loads(Path(res["sidecar_path"]).read_text())["ingredients"]

    assert "rna_segment" in meta, (
        f"rna_segment not found in sidecar meta. Keys: {list(meta.keys())}"
    )
    n_rna = pack_count_of(pack, "rna_segment")
    assert n_rna > 0, (
        f"expected >0 rna_segment placements, got {n_rna}. "
        f"Ingredients: {[i['name'] for i in pack['ingredients']]}"
    )
