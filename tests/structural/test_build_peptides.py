"""TDD test: nascent peptide coil representation (C2-2).

Synthesises a snapshot with one mRNA (unique_index=20, is_mRNA=True, length=600)
and one ribosome (mRNA_index=20, pos=300, peptide_length=200), builds a small
model (top_n=5), then asserts:

  (a) ``peptide_segment`` placements > 0 — the nascent peptide coil trailing
      from the ribosome is tiled with segment instances.
  (b) ``peptide_segment`` appears in the sidecar meta — the ingredient is
      registered with display_name / category so the viewer can label it.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from v2ecoli.structural import build

# Pre-built structure cache from a previous full run (no network needed).
_STRUCT_CACHE = Path(
    "/Users/eranagmon/code/v2e-pdmp-refresh/out/ecoli3d_expanded/structures"
)

# Real DATA directory for shared reference files (uniprot_map, genome CSV).
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


def _seed_struct_cache(struct_cache: Path) -> None:
    """Copy meshable structures into ``struct_cache``.

    Includes ``peptide_segment.pdb`` (a copy of dna_segment.pdb) so the
    peptide_segment ingredient meshes successfully and appears in sidecar meta.
    """
    struct_cache.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("rna_polymerase.pdb", "rna_polymerase.pdb"),
        ("dna_segment.pdb", "dna_segment.pdb"),
        ("dna_segment.pdb", "rna_segment.pdb"),        # RNA reuses dsDNA 1BNA mesh
        ("dna_segment.pdb", "rna_segment_free.pdb"),   # free mRNA reuses same mesh
        ("dna_segment.pdb", "peptide_segment.pdb"),    # peptide reuses same mesh, distinct color
        ("replisome.pdb", "replisome.pdb"),
        ("70s_ribosome.cif", "70s_ribosome.cif"),      # 70S (ingredient id slug)
        ("groel.pdb", "groel.pdb"),
        ("eg10367_monomer.pdb", "eg10367_monomer.pdb"),
        # 30S/50S subunits: seed as dummy copies of an existing PDB.
        ("rna_polymerase.pdb", "30s_subunit.pdb"),
        ("rna_polymerase.pdb", "50s_subunit.pdb"),
    ]:
        src = _STRUCT_CACHE / src_name
        if src.exists():
            shutil.copy(src, struct_cache / dst_name)


# ── test ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_peptide_segment_placements(tmp_path, monkeypatch):
    """Peptide coil from ribosome with peptide_length=200 produces peptide_segment placements."""
    if not _STRUCT_CACHE.exists():
        pytest.skip(f"structure cache not available at {_STRUCT_CACHE}")

    monkeypatch.setenv(
        "PARSIMONY_HOME",
        os.environ.get("PARSIMONY_HOME", "/Users/eranagmon/code/parsimony"),
    )

    # Snapshot: 1 free mRNA (uid=20, is_mRNA, length=600) + 1 ribosome (peptide_length=200).
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]),
        counts=np.array([100]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        # RNAP fields (1 active RNAP, not connected to this mRNA)
        rnap_coordinates=np.array([0], dtype="i8"),
        rnap_domain_index=np.array([0], dtype="i4"),
        rnap_is_forward=np.array([True]),
        rnap_unique_index=np.array([7], dtype="i8"),
        # 1 free mRNA: unique_index=20, RNAP_index=-1 (released), length=600
        rna_unique_index=np.array([20], dtype="i8"),
        rna_RNAP_index=np.array([-1], dtype="i8"),
        rna_transcript_length=np.array([600], dtype="i8"),
        rna_is_mRNA=np.array([True]),
        rna_is_full_transcript=np.array([True]),
        rna_TU_index=np.array([1], dtype="i8"),
        # 1 ribosome on mRNA uid=20 at nt 300 with 200 aa peptide
        ribo_mRNA_index=np.array([20], dtype="i8"),
        ribo_pos_on_mRNA=np.array([300], dtype="i8"),
        ribo_peptide_length=np.array([200], dtype="i8"),
        ribo_protein_index=np.array([0], dtype="i8"),
    )

    # Reference files needed by build_model.
    for fname in ("uniprot_map.json", "ecoli_k12_genes.csv"):
        src = _REAL_DATA / fname
        if src.exists():
            shutil.copy(src, tmp_path / fname)

    out = tmp_path / "pack"
    _seed_struct_cache(out / "structures")
    monkeypatch.setattr(build, "DATA", tmp_path)

    res = build.build_model(str(out), state_source="snapshot", top_n=5)

    pack = json.loads(Path(res["pack_path"]).read_text())
    meta = json.loads(Path(res["sidecar_path"]).read_text())["ingredients"]

    # (a) peptide_segment placements > 0 — coil from ribosome with peptide_length=200.
    n_pep = pack_count_of(pack, "peptide_segment")
    assert n_pep > 0, (
        f"expected >0 peptide_segment placements (ribosome with peptide_length=200 "
        f"should grow a nascent peptide coil), got 0. "
        f"Ingredients in pack: {[i['name'] for i in pack['ingredients']]}"
    )

    # (b) peptide_segment in sidecar meta — ingredient registered with display/category.
    assert "peptide_segment" in meta, (
        f"peptide_segment missing from sidecar meta. "
        f"Keys present: {list(meta.keys())}"
    )
