"""Pure path-remap: relabel the baseline store hierarchy into a biological
(compartment -> molecular class) one. See
docs/superpowers/specs/2026-06-13-biological-composite-design.md.

The transform is a relabel only — no store internals are split and no update
math changes — so a composite built through it is bit-identical to baseline.
"""
from __future__ import annotations

# Top-level data store -> new biological path. Coordination/clock stores move
# under machinery/ and clock/ so the cell/ subtree reads as biology, not plumbing.
#
# `unique` is RELOCATED WHOLE (like `bulk`), not split per-molecule. The spec's
# target hierarchy split unique molecules across cell/chromosome,
# cell/transcription and cell/translation on the premise that "processes target
# unique leaf paths." That premise is false in this codebase: the division
# process (port schema InPlaceDict) and both mass listeners (port schema
# map[node]) wire one port to the WHOLE `unique` store and read every molecule
# through it. Physically moving any molecule out of a shared `unique` store
# would hand those consumers a partial map and break bit-identity. So Phase 1
# keeps `unique` co-located under cell/unique_molecules (exactly the treatment
# `bulk` gets); the per-molecule biological split + renames are deferred to
# Phase 2 (same gate as splitting `bulk`). See UNIQUE_REMAP below for the
# intended Phase-2 targets, kept for reference.
REMAP: dict[str, tuple[str, ...]] = {
    'bulk':               ('cell', 'molecules'),
    'unique':             ('cell', 'unique_molecules'),
    'listeners':          ('cell', 'observables'),
    'ppgpp_state':        ('cell', 'regulation', 'ppgpp_state'),
    'attenuation_config': ('cell', 'regulation', 'attenuation_config'),
    'boundary':           ('environment', 'boundary'),
    'environment':        ('environment', 'media'),
    'exchange':           ('environment', 'exchange'),
    'process':            ('machinery', 'process'),
    'allocator_rng':      ('machinery', 'allocator_rng'),
    'process_state':      ('machinery', 'process_state'),
    'next_update_time':   ('machinery', 'next_update_time'),
    'request':            ('machinery', 'request'),
    'allocate':           ('machinery', 'allocate'),
    'pinned_flux_targets': ('machinery', 'pinned_flux_targets'),
    'global_time':        ('clock', 'global_time'),
    'timestep':           ('clock', 'timestep'),
    'divide':             ('clock', 'divide'),
    'division_threshold': ('clock', 'division_threshold'),
}

# Phase-1 location of every unique-molecule leaf: co-located WHOLE under
# cell/unique_molecules/<name> (see the REMAP['unique'] note). The equivalence
# test's _data_pairs helper reads this map to locate each molecule in the
# biological tree. The Phase-2 split/rename targets (chromosome/transcription/
# translation, e.g. active_RNAP -> rna_polymerases) are recorded in
# UNIQUE_REMAP_PHASE2 below but are NOT applied in Phase 1.
_UNIQUE_NAMES = (
    'full_chromosome', 'chromosome_domain', 'oriC', 'DnaA_box',
    'chromosomal_segment', 'gene', 'active_replisome', 'active_RNAP',
    'RNA', 'promoter', 'active_ribosome',
)
UNIQUE_REMAP: dict[str, tuple[str, ...]] = {
    name: ('cell', 'unique_molecules', name) for name in _UNIQUE_NAMES
}

# Deferred Phase-2 biological split/rename for unique molecules (reference only;
# not reachable as a pure relabel — see REMAP['unique']).
UNIQUE_REMAP_PHASE2: dict[str, tuple[str, ...]] = {
    'full_chromosome':     ('cell', 'chromosome', 'full_chromosome'),
    'chromosome_domain':   ('cell', 'chromosome', 'chromosome_domain'),
    'oriC':                ('cell', 'chromosome', 'oriC'),
    'DnaA_box':            ('cell', 'chromosome', 'DnaA_box'),
    'chromosomal_segment': ('cell', 'chromosome', 'chromosomal_segment'),
    'gene':                ('cell', 'chromosome', 'gene'),
    'active_replisome':    ('cell', 'chromosome', 'active_replisome'),
    'active_RNAP':         ('cell', 'transcription', 'rna_polymerases'),
    'RNA':                 ('cell', 'transcription', 'transcripts'),
    'promoter':            ('cell', 'transcription', 'promoters'),
    'active_ribosome':     ('cell', 'translation', 'ribosomes'),
}


def remap_path(path: list) -> list:
    """Rewrite one wire path (list of segments) into its biological location.

    The leading segment is rewritten through REMAP (``unique`` included — it is
    relocated whole, so both the bare ``['unique']`` port wire and any
    ``['unique', <molecule>, …]`` leaf wire repath consistently under
    ``cell/unique_molecules``); the tail is preserved. Unknown heads (flow
    tokens, 'agents', …) pass through unchanged.
    """
    if not path:
        return list(path)
    head = path[0]
    if head in REMAP:
        return list(REMAP[head]) + list(path[1:])
    return list(path)


_EDGE_TYPES = ('step', 'process')


def _is_edge(value) -> bool:
    return isinstance(value, dict) and value.get('_type') in _EDGE_TYPES


def _set_path(tree: dict, path: tuple, value) -> None:
    """Place value at nested path, creating intermediate dicts."""
    node = tree
    for seg in path[:-1]:
        node = node.setdefault(seg, {})
    node[path[-1]] = value


def _rewrite_wires(wires):
    """Rewrite a wire structure (dict of port -> path-list, possibly nested).

    Every store (``bulk``, ``unique``, ``listeners`` and the coordination/clock
    stores) relocates whole, so a port wired to a whole subtree (``['unique']``,
    ``['bulk']``, …) simply repaths to the new root and its sub-ports/children
    bind underneath exactly as before — a pure relabel, no process internals
    touched.
    """
    if isinstance(wires, list):
        return remap_path(wires)
    if isinstance(wires, dict):
        return {k: _rewrite_wires(v) for k, v in wires.items()}
    return wires


def remap_cell_state(cell_state: dict) -> dict:
    """Return a new cell-state tree with data stores relocated to biological
    paths and every edge's wires rewritten. Edges stay at the root. The input
    is not mutated.

    Unknown non-edge keys (not in REMAP) are carried over at the root unchanged
    so nothing is silently dropped. ``unique`` is in REMAP (relocated whole), so
    its molecules move together under cell/unique_molecules.
    """
    out: dict = {}
    for key, value in cell_state.items():
        if _is_edge(value):
            # SHALLOW copy the edge, then swap in freshly-rewritten wire dicts.
            # Never deep-copy: the edge holds a live process/step ``instance``
            # (e.g. ParquetEmitter, which owns a _queue.SimpleQueue) that is
            # unpicklable AND must stay the same shared object. _rewrite_wires
            # builds brand-new lists/dicts, so the original edge's inputs/outputs
            # are left untouched — the no-mutation contract still holds.
            edge = dict(value)
            if 'inputs' in edge:
                edge['inputs'] = _rewrite_wires(edge['inputs'])
            if 'outputs' in edge:
                edge['outputs'] = _rewrite_wires(edge['outputs'])
            out[key] = edge
        elif key in REMAP:
            _set_path(out, REMAP[key], value)
        else:
            out[key] = value
    return out
