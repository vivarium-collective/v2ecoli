"""TDD test: chromosome_index + is_daughter carried through build_model recipe.

Synthesises a minimal snapshot with:
  - 1 RNAP (unique_index=7, chromosome_index=1, is_daughter=True)
  - 1 nascent RNA attached to that RNAP  (RNAP_index=7)
  - 1 free RNA (RNAP_index=-1)

Asserts that build_model writes these fields into the recipe JSON's
chromosome.rnaps and chromosome.rnas entries, so parsimony can route
each molecule to the correct chromosome copy.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from v2ecoli.structural import build

# Reference structure cache (populated by earlier build runs — no network needed).
_STRUCT_CACHE = Path(
    "/Users/eranagmon/code/v2e-pdmp-refresh/out/ecoli3d_expanded/structures"
)

# Real DATA directory for shared reference files (uniprot_map.json, genome CSV).
_REAL_DATA = (
    Path(__file__).resolve().parent.parent.parent / "v2ecoli" / "structural" / "data"
)


def _seed_struct_cache(struct_cache: Path) -> None:
    """Copy the meshable structures the build needs into ``struct_cache``."""
    struct_cache.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("rna_polymerase.pdb", "rna_polymerase.pdb"),
        ("dna_segment.pdb", "dna_segment.pdb"),
        ("dna_segment.pdb", "rna_segment.pdb"),
        ("replisome.pdb", "replisome.pdb"),
        ("70s_ribosome.cif", "70s_ribosome.cif"),
        ("groel.pdb", "groel.pdb"),
        ("eg10367_monomer.pdb", "eg10367_monomer.pdb"),
    ]:
        src = _STRUCT_CACHE / src_name
        if src.exists():
            shutil.copy(src, struct_cache / dst_name)


@pytest.mark.slow
def test_build_carries_chromosome_fields_on_rnaps_and_rnas(tmp_path, monkeypatch):
    """build_model populates chromosome_index + is_daughter on RNAP and RNA recipe entries.

    With one RNAP on chromosome 1 (is_daughter=True) and one nascent RNA attached to
    it, the recipe JSON must carry:
      chromosome.rnaps[0].chromosome_index == 1
      chromosome.rnaps[0].is_daughter      == True
      chromosome.rnas[0].chromosome_index  == 1  (rooted at the RNAP)
      chromosome.rnas[0].is_daughter       == True

    The free RNA (RNAP_index==-1) must carry chromosome_index==0, is_daughter==False.
    """
    if not _STRUCT_CACHE.exists():
        pytest.skip(f"structure cache not available at {_STRUCT_CACHE}")

    monkeypatch.setenv(
        "PARSIMONY_HOME",
        os.environ.get("PARSIMONY_HOME", "/Users/eranagmon/code/parsimony"),
    )

    # Snapshot: 1 RNAP on chromosome 1 (daughter copy), 1 nascent RNA + 1 free RNA.
    np.savez(
        tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]),
        counts=np.array([100]),
        volume=np.array(1.0),
        n_chromosomes=np.array(2),
        fork_fraction=np.array(0.25),
        division_progress=np.array(0.0),
        # RNAP arrays
        rnap_coordinates=np.array([500_000], dtype="i8"),
        rnap_domain_index=np.array([1], dtype="i4"),
        rnap_is_forward=np.array([True]),
        rnap_unique_index=np.array([7], dtype="i8"),
        rnap_chromosome_index=np.array([1], dtype="i4"),
        rnap_is_daughter=np.array([True]),
        # RNA arrays: rna[0] = nascent (RNAP uid 7), rna[1] = free (uid -1)
        rna_unique_index=np.array([20, 21], dtype="i8"),
        rna_RNAP_index=np.array([7, -1], dtype="i8"),
        rna_transcript_length=np.array([600, 400], dtype="i8"),
        rna_is_mRNA=np.array([True, True]),
        rna_is_full_transcript=np.array([False, True]),
        rna_TU_index=np.array([1, 2], dtype="i8"),
    )

    for fname in ("uniprot_map.json", "ecoli_k12_genes.csv"):
        src = _REAL_DATA / fname
        if src.exists():
            shutil.copy(src, tmp_path / fname)

    out = tmp_path / "pack"
    _seed_struct_cache(out / "structures")
    monkeypatch.setattr(build, "DATA", tmp_path)

    res = build.build_model(str(out), state_source="snapshot", top_n=5)

    recipe = json.loads(Path(res["recipe_path"]).read_text())
    rnaps = recipe["chromosome"]["rnaps"]
    rnas = recipe["chromosome"]["rnas"]

    # --- RNAP assertions ---
    assert len(rnaps) == 1, f"expected 1 rnap entry, got {len(rnaps)}"
    assert rnaps[0].get("chromosome_index") == 1, (
        f"rnaps[0].chromosome_index should be 1, got {rnaps[0].get('chromosome_index')!r}"
    )
    assert rnaps[0].get("is_daughter") is True, (
        f"rnaps[0].is_daughter should be True, got {rnaps[0].get('is_daughter')!r}"
    )

    # --- RNA assertions ---
    assert len(rnas) == 2, f"expected 2 rna entries, got {len(rnas)}"

    # Nascent RNA: inherits chromosome_index/is_daughter from its RNAP.
    nascent = next(r for r in rnas if not r.get("is_free", False))
    assert nascent.get("chromosome_index") == 1, (
        f"nascent rna chromosome_index should be 1, got {nascent.get('chromosome_index')!r}"
    )
    assert nascent.get("is_daughter") is True, (
        f"nascent rna is_daughter should be True, got {nascent.get('is_daughter')!r}"
    )

    # Free RNA: cytoplasmic, no chromosome — chromosome_index=0, is_daughter=False.
    free = next(r for r in rnas if r.get("is_free", False))
    assert free.get("chromosome_index") == 0, (
        f"free rna chromosome_index should be 0, got {free.get('chromosome_index')!r}"
    )
    assert free.get("is_daughter") is False, (
        f"free rna is_daughter should be False, got {free.get('is_daughter')!r}"
    )
