"""TDD test: corrected ribosome representation (C1-4).

Synthesises a snapshot with one mRNA (unique_index=20, is_mRNA=True, length=600)
and one ribosome (mRNA_index=20, pos=300, peptide=0), builds a small model
(top_n=5), then asserts:

  (a) a ``70S_ribosome`` placement exists (>0) — the active ribosome is placed
      on the mRNA via the ribosome_marker mechanism, not random interior packing.
  (b) exactly 1 ``70S_ribosome`` is placed — NOT the fabricated 20000 from the
      old curated count; we have exactly 1 ribosome in the snapshot.
  (c) ``30S_subunit`` and ``50S_subunit`` appear in the sidecar meta — the free
      subunits are now registered as ingredients with real bulk counts.
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

    Includes dummy 30s_subunit.pdb and 50s_subunit.pdb (copies of an existing
    PDB) so the 30S/50S ingredients mesh successfully and appear in the sidecar
    meta — the test checks presence, not biological accuracy of the structure.
    """
    struct_cache.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("rna_polymerase.pdb", "rna_polymerase.pdb"),
        ("dna_segment.pdb", "dna_segment.pdb"),
        ("dna_segment.pdb", "rna_segment.pdb"),       # RNA reuses dsDNA 1BNA mesh
        ("replisome.pdb", "replisome.pdb"),
        ("70s_ribosome.cif", "70s_ribosome.cif"),      # 70S (ingredient id slug)
        ("groel.pdb", "groel.pdb"),
        ("eg10367_monomer.pdb", "eg10367_monomer.pdb"),
        # 30S/50S subunits: seed as dummy copies of an existing PDB so the mesh
        # step succeeds and the ingredients register in the sidecar.
        ("rna_polymerase.pdb", "30s_subunit.pdb"),
        ("rna_polymerase.pdb", "50s_subunit.pdb"),
    ]:
        src = _STRUCT_CACHE / src_name
        if src.exists():
            shutil.copy(src, struct_cache / dst_name)


# ── test ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_ribosome_wiring_and_subunits(tmp_path, monkeypatch):
    """Ribosome on mRNA via ribosome_marker; 30S/50S in sidecar; no 20000 fabrication."""
    if not _STRUCT_CACHE.exists():
        pytest.skip(f"structure cache not available at {_STRUCT_CACHE}")

    monkeypatch.setenv(
        "PARSIMONY_HOME",
        os.environ.get("PARSIMONY_HOME", "/Users/eranagmon/code/parsimony"),
    )

    # Snapshot: 1 RNAP + 1 free mRNA (uid=20) + 1 ribosome on that mRNA.
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]),
        counts=np.array([100]),
        volume=np.array(1.0),
        n_chromosomes=np.array(1),
        fork_fraction=np.array(0.0),
        division_progress=np.array(0.0),
        # RNAP fields (1 active RNAP)
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
        # 1 ribosome on mRNA uid=20 at nucleotide position 300
        ribo_mRNA_index=np.array([20], dtype="i8"),
        ribo_pos_on_mRNA=np.array([300], dtype="i8"),
        ribo_peptide_length=np.array([0], dtype="i8"),
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

    # (a) at least one 70S_ribosome placement — the active ribosome on the mRNA.
    n_70s = pack_count_of(pack, "70S_ribosome")
    assert n_70s > 0, (
        f"expected >0 70S_ribosome placements (ribosome on mRNA), got 0. "
        f"Ingredients in pack: {[i['name'] for i in pack['ingredients']]}"
    )

    # (b) exactly 1 placement — the 1 ribosome from the snapshot, NOT the old
    # fabricated 20000 randomly-packed copies.
    assert n_70s == 1, (
        f"expected exactly 1 70S_ribosome placement (one ribosome in snapshot), "
        f"got {n_70s}. The old count=20000 in CURATED produces many random placements; "
        f"after the fix count must be 0 (placed only via ribosome_marker)."
    )

    # (c) 30S and 50S subunits appear in the sidecar meta — they are now registered
    # as CURATED ingredients with real bulk counts (CPLX0-3953 / CPLX0-3962).
    assert "30S_subunit" in meta, (
        f"30S_subunit missing from sidecar meta. "
        f"Keys: {list(meta.keys())}"
    )
    assert "50S_subunit" in meta, (
        f"50S_subunit missing from sidecar meta. "
        f"Keys: {list(meta.keys())}"
    )
