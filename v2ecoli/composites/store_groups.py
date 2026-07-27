"""Biological grouping of the baseline store hierarchy — annotation ONLY.

This is the interpretable "reads as biology, not plumbing" grouping distilled
from the retired ``biological`` composite (see git history: composites/
biological.py + _remap.py, and docs/superpowers/specs/
2026-06-13-biological-composite-design.md). The ``biological`` composite applied
this map as a *path relabel*, which changed the emitted store paths and so broke
downstream analyses. This module keeps only the **grouping knowledge** as
metadata: baseline's emitted paths stay flat (``global_time``/``bulk``/
``listeners``/…), and display tools (Composite Explorer, loom) can render the
flat stores grouped biologically without moving any data.

``STORE_GROUPS`` maps each top-level baseline store to the biological group path
it belongs under. It is exposed as ``metadata['store_groups']`` on the baseline
generator; nothing in the simulation reads it.
"""
from __future__ import annotations

# top-level baseline store -> biological group path (compartment -> molecular class).
# Coordination/clock stores group under machinery/ and clock/ so the cell/ subtree
# reads as biology. This is the Phase-1 grouping: `bulk` and `unique` are grouped
# WHOLE (not split per-molecule) — see UNIQUE_GROUPS_PHASE2 for the deferred split.
STORE_GROUPS: dict[str, tuple[str, ...]] = {
    "bulk":                ("cell", "molecules"),
    "unique":              ("cell", "unique_molecules"),
    "listeners":           ("cell", "observables"),
    "ppgpp_state":         ("cell", "regulation", "ppgpp_state"),
    "attenuation_config":  ("cell", "regulation", "attenuation_config"),
    "boundary":            ("environment", "boundary"),
    "environment":         ("environment", "media"),
    "exchange":            ("environment", "exchange"),
    "process":             ("machinery", "process"),
    "allocator_rng":       ("machinery", "allocator_rng"),
    "process_state":       ("machinery", "process_state"),
    "next_update_time":    ("machinery", "next_update_time"),
    "request":             ("machinery", "request"),
    "allocate":            ("machinery", "allocate"),
    "pinned_flux_targets": ("machinery", "pinned_flux_targets"),
    "global_time":         ("clock", "global_time"),
    "timestep":            ("clock", "timestep"),
    "divide":              ("clock", "divide"),
    "division_threshold":  ("clock", "division_threshold"),
}

# Deferred Phase-2 refinement: split the monolithic `unique` store's molecules
# into chromosome / transcription / translation groups (with biological renames,
# e.g. active_RNAP -> rna_polymerases). Reference only — not applied. Realising
# this needs the per-molecule store split (same gate as splitting `bulk`), which
# is a genuine schema change, not a grouping annotation.
UNIQUE_GROUPS_PHASE2: dict[str, tuple[str, ...]] = {
    "full_chromosome":     ("cell", "chromosome", "full_chromosome"),
    "chromosome_domain":   ("cell", "chromosome", "chromosome_domain"),
    "oriC":                ("cell", "chromosome", "oriC"),
    "DnaA_box":            ("cell", "chromosome", "DnaA_box"),
    "chromosomal_segment": ("cell", "chromosome", "chromosomal_segment"),
    "gene":                ("cell", "chromosome", "gene"),
    "active_replisome":    ("cell", "chromosome", "active_replisome"),
    "active_RNAP":         ("cell", "transcription", "rna_polymerases"),
    "RNA":                 ("cell", "transcription", "transcripts"),
    "promoter":            ("cell", "transcription", "promoters"),
    "active_ribosome":     ("cell", "translation", "ribosomes"),
}


def group_of(store: str) -> tuple[str, ...] | None:
    """Biological group path for a top-level baseline store, or None if ungrouped."""
    return STORE_GROUPS.get(store)
