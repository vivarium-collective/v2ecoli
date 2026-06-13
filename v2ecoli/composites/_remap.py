"""Pure path-remap: relabel the baseline store hierarchy into a biological
(compartment -> molecular class) one. See
docs/superpowers/specs/2026-06-13-biological-composite-design.md.

The transform is a relabel only — no store internals are split and no update
math changes — so a composite built through it is bit-identical to baseline.
"""
from __future__ import annotations

import copy

# Top-level data store -> new biological path. Coordination/clock stores move
# under machinery/ and clock/ so the cell/ subtree reads as biology, not plumbing.
REMAP: dict[str, tuple[str, ...]] = {
    'bulk':               ('cell', 'molecules'),
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
    'global_time':        ('clock', 'global_time'),
    'timestep':           ('clock', 'timestep'),
    'divide':             ('clock', 'divide'),
    'division_threshold': ('clock', 'division_threshold'),
}

# Each unique-molecule leaf -> its biological compartment/subsystem path.
UNIQUE_REMAP: dict[str, tuple[str, ...]] = {
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

    Leading segment(s) are rewritten through UNIQUE_REMAP (for 'unique/<x>')
    or REMAP; the tail is preserved. Unknown heads (flow tokens, 'agents', …)
    pass through unchanged.
    """
    if not path:
        return list(path)
    head = path[0]
    if head == 'unique' and len(path) >= 2 and path[1] in UNIQUE_REMAP:
        return list(UNIQUE_REMAP[path[1]]) + list(path[2:])
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
    """Rewrite a wire structure (dict of port -> path-list, possibly nested)."""
    if isinstance(wires, list):
        return remap_path(wires)
    if isinstance(wires, dict):
        return {k: _rewrite_wires(v) for k, v in wires.items()}
    return wires


def remap_cell_state(cell_state: dict) -> dict:
    """Return a new cell-state tree with data stores relocated to biological
    paths and every edge's wires rewritten. Edges stay at the root. The input
    is not mutated.

    Unknown non-edge keys (not in REMAP, not 'unique') are carried over at the
    root unchanged so nothing is silently dropped.
    """
    out: dict = {}
    for key, value in cell_state.items():
        if _is_edge(value):
            edge = copy.deepcopy(value)
            if 'inputs' in edge:
                edge['inputs'] = _rewrite_wires(edge['inputs'])
            if 'outputs' in edge:
                edge['outputs'] = _rewrite_wires(edge['outputs'])
            out[key] = edge
        elif key == 'unique':
            for uname, uval in value.items():
                target = UNIQUE_REMAP.get(uname)
                if target is None:
                    # Unmapped unique molecule: keep under cell/<name> rather
                    # than drop it, and make the omission visible.
                    target = ('cell', uname)
                _set_path(out, target, uval)
        elif key in REMAP:
            _set_path(out, REMAP[key], value)
        else:
            out[key] = value
    return out
