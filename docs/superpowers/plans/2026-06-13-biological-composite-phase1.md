# Biological Composite — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `biological()` composite for v2ecoli whose store hierarchy is organized by cellular compartment → molecular class, by post-processing the finished baseline document through a pure path-remap, and prove its trajectory is **bit-identical** to the baseline.

**Architecture:** `baseline()` already returns a complete process-bigraph document whose cell state lives at `state.agents.0`, with every process wired via `make_edge` topology tuples and all build-time seeding done on the original layout. Phase 1 adds (a) a reusable path-remap module that relocates data subtrees and rewrites edge wires, and (b) a `biological()` generator that calls `baseline()` then applies the remap. No process internals change, so the simulation is provably identical. A pytest gate runs both composites N steps and asserts every data leaf is `np.array_equal`.

**Tech Stack:** Python, numpy, process-bigraph (`Composite`), pbg-superpowers `@composite_generator`, pytest. Reuses `tests/_state_equal.py:deep_equal`.

---

## Execution environment

The worktree `/Users/eranagmon/code/v2e-biological` has **no `.venv`**. Run everything with the main checkout's interpreter and PYTHONPATH-shadow the worktree source:

```bash
export V2E_PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
export PYTHONPATH=/Users/eranagmon/code/v2e-biological
cd /Users/eranagmon/code/v2e-biological
```

- Tasks 1–2 are pure unit tests — **no ParCa cache needed.**
- Tasks 4–6 build the full composite — they need a built cache. Point at the main checkout's cache:
  ```bash
  export V2ECOLI_CACHE_DIR=/Users/eranagmon/code/v2ecoli/out/cache
  ```
  Confirm it exists first: `ls "$V2ECOLI_CACHE_DIR"/sim_data_cache.dill`. If absent, build with `$V2E_PY scripts/build_cache.py --mode full` (per memory: ~2.5 min on the mini; use `--mode full`, never `fast`, for simulation).

All pytest invocations below assume the three exports above are set.

---

## File structure

- Create `v2ecoli/composites/_remap.py` — the remap tables + transform (Tasks 1–2).
- Create `v2ecoli/composites/biological.py` — the `biological()` generator (Tasks 3–4).
- Create `tests/test_remap.py` — unit tests for the remap (Tasks 1–2).
- Create `tests/test_biological_equivalence.py` — the bit-identical gate (Task 5).
- Create `scripts/compare_biological.py` — biological-marker comparison + HTML (Task 6).

---

### Task 1: Remap tables + `remap_path`

**Files:**
- Create: `v2ecoli/composites/_remap.py`
- Test: `tests/test_remap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remap.py
from v2ecoli.composites._remap import remap_path


def test_bulk_relocates_to_cell_molecules():
    assert remap_path(['bulk']) == ['cell', 'molecules']

def test_bulk_subpath_preserves_tail():
    assert remap_path(['bulk', 'count']) == ['cell', 'molecules', 'count']

def test_unique_rnap_relocates_and_renames():
    assert remap_path(['unique', 'active_RNAP']) == ['cell', 'transcription', 'rna_polymerases']

def test_unique_chromosome_groups_under_chromosome():
    assert remap_path(['unique', 'full_chromosome']) == ['cell', 'chromosome', 'full_chromosome']

def test_listeners_subleaf_relocates():
    assert remap_path(['listeners', 'mass']) == ['cell', 'observables', 'mass']

def test_global_time_relocates_to_clock():
    assert remap_path(['global_time']) == ['clock', 'global_time']

def test_coordination_store_relocates_to_machinery():
    assert remap_path(['process_state', 'polypeptide_elongation']) == \
        ['machinery', 'process_state', 'polypeptide_elongation']

def test_flow_token_is_left_untouched():
    assert remap_path(['_layer_token_3']) == ['_layer_token_3']

def test_unknown_head_is_left_untouched():
    assert remap_path(['agents', '0']) == ['agents', '0']

def test_empty_path_is_noop():
    assert remap_path([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$V2E_PY -m pytest tests/test_remap.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.composites._remap'`

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/composites/_remap.py
"""Pure path-remap: relabel the baseline store hierarchy into a biological
(compartment -> molecular class) one. See
docs/superpowers/specs/2026-06-13-biological-composite-design.md.

The transform is a relabel only — no store internals are split and no update
math changes — so a composite built through it is bit-identical to baseline.
"""
from __future__ import annotations

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$V2E_PY -m pytest tests/test_remap.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/_remap.py tests/test_remap.py
git commit -m "feat(remap): biological path-remap tables + remap_path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Relocate data stores + rewrite edge wires (`remap_cell_state`)

**Files:**
- Modify: `v2ecoli/composites/_remap.py`
- Test: `tests/test_remap.py`

This is the structural transform applied to a finished `agents.0` cell state: data subtrees move to their biological paths; edges (`_type` in step/process) stay at the root but have every wire path rewritten.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_remap.py
import numpy as np
from v2ecoli.composites._remap import remap_cell_state


def _fake_edge():
    # Mimics make_edge output: wires are {port: [path, segments]} lists.
    return {
        '_type': 'step',
        'priority': 1.0,
        'instance': object(),
        '_inputs': {}, '_outputs': {},
        'inputs': {
            'bulk': ['bulk'],
            'active_RNAP': ['unique', 'active_RNAP'],
            'mass': ['listeners', 'mass'],
            'global_time': ['global_time'],
            '_layer_in_1': ['_layer_token_0'],
        },
        'outputs': {
            'bulk': ['bulk'],
            'next': ['next_update_time', 'metabolism'],
        },
    }


def _fake_cell_state():
    return {
        'bulk': np.array([1, 2, 3]),
        'unique': {'active_RNAP': np.array([10, 11]),
                   'full_chromosome': np.array([7])},
        'listeners': {'mass': {'cell_mass': 4.0}},
        'global_time': 0.0,
        'process_state': {'polypeptide_elongation': {'gtp_to_hydrolyze': 0}},
        'ecoli-metabolism': _fake_edge(),
    }


def test_data_stores_move_to_biological_paths():
    out = remap_cell_state(_fake_cell_state())
    assert np.array_equal(out['cell']['molecules'], np.array([1, 2, 3]))
    assert np.array_equal(out['cell']['transcription']['rna_polymerases'], np.array([10, 11]))
    assert np.array_equal(out['cell']['chromosome']['full_chromosome'], np.array([7]))
    assert out['cell']['observables']['mass'] == {'cell_mass': 4.0}
    assert out['clock']['global_time'] == 0.0
    assert out['machinery']['process_state'] == {'polypeptide_elongation': {'gtp_to_hydrolyze': 0}}


def test_old_top_level_keys_are_gone():
    out = remap_cell_state(_fake_cell_state())
    for old in ('bulk', 'unique', 'listeners', 'global_time', 'process_state'):
        assert old not in out


def test_edge_stays_at_root_and_wires_rewritten():
    out = remap_cell_state(_fake_cell_state())
    edge = out['ecoli-metabolism']
    assert edge['_type'] == 'step'
    assert edge['inputs']['bulk'] == ['cell', 'molecules']
    assert edge['inputs']['active_RNAP'] == ['cell', 'transcription', 'rna_polymerases']
    assert edge['inputs']['mass'] == ['cell', 'observables', 'mass']
    assert edge['inputs']['global_time'] == ['clock', 'global_time']
    assert edge['inputs']['_layer_in_1'] == ['_layer_token_0']      # untouched
    assert edge['outputs']['next'] == ['machinery', 'next_update_time', 'metabolism']


def test_input_is_not_mutated():
    src = _fake_cell_state()
    remap_cell_state(src)
    assert 'bulk' in src and 'cell' not in src    # original untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$V2E_PY -m pytest tests/test_remap.py -q`
Expected: FAIL with `ImportError: cannot import name 'remap_cell_state'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to v2ecoli/composites/_remap.py
import copy

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$V2E_PY -m pytest tests/test_remap.py -q`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/_remap.py tests/test_remap.py
git commit -m "feat(remap): relocate data stores + rewrite edge wires

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `biological()` generator (structural)

**Files:**
- Create: `v2ecoli/composites/biological.py`
- Test: `tests/test_remap.py` (structural assertion; no cache needed because we
  monkeypatch `baseline` with a tiny stub here)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_remap.py
def test_biological_wraps_baseline_and_remaps(monkeypatch):
    import v2ecoli.composites.biological as biomod

    def fake_baseline(**kwargs):
        return {
            'state': {
                'agents': {'0': {
                    'bulk': [1],
                    'unique': {'active_RNAP': [2]},
                    'listeners': {'mass': {}},
                    'global_time': 0.0,
                    'emitter': {'_type': 'step', 'inputs': {'b': ['bulk']},
                                'outputs': {}},
                }},
                'global_time': 0.0,
            },
            'skip_initial_steps': True,
            'sequential_steps': False,
            'flow_order': ['emitter'],
        }

    monkeypatch.setattr(biomod, 'baseline', fake_baseline)
    doc = biomod.biological(seed=0)
    agent = doc['state']['agents']['0']
    assert set(agent) >= {'cell', 'clock', 'emitter'}
    assert 'bulk' not in agent and 'unique' not in agent and 'listeners' not in agent
    assert agent['emitter']['inputs']['b'] == ['cell', 'molecules']
    # The outer document scaffolding is preserved verbatim.
    assert doc['skip_initial_steps'] is True
    assert doc['flow_order'] == ['emitter']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$V2E_PY -m pytest tests/test_remap.py::test_biological_wraps_baseline_and_remaps -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2ecoli.composites.biological'`

- [ ] **Step 3: Write minimal implementation**

```python
# v2ecoli/composites/biological.py
"""Biologically-organized E. coli whole-cell composite.

Identical simulation to ``baseline()`` — same processes, same update math —
but the store hierarchy is relabeled into cellular compartments / molecular
classes via a pure path-remap (see _remap.py and
docs/superpowers/specs/2026-06-13-biological-composite-design.md).

Phase 1: relabel only -> bit-identical to baseline (see
tests/test_biological_equivalence.py). Phase 2 (not built here) splits the
monolithic pools and adds unit-bearing schemas.
"""
from __future__ import annotations

from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.composites.baseline import baseline
from v2ecoli.composites._remap import remap_cell_state


@composite_generator(
    emitters=[
        {
            "address": "local:ParquetEmitter",
            "config": {},
            # Remapped emit paths (baseline used global_time/bulk/listeners).
            "paths": ["clock/global_time", "cell/molecules", "cell/observables"],
        },
    ],
)
def biological(core: Any = None, **kwargs) -> dict:
    """Build the biological composite document.

    All keyword arguments are forwarded verbatim to :func:`baseline`
    (seed, cache_dir, emitter, feature toggles, bundle, …). The finished
    baseline document is then relabeled in place at ``state.agents.0``.
    """
    doc = baseline(core=core, **kwargs)
    agents = doc['state']['agents']
    for agent_id, cell_state in list(agents.items()):
        agents[agent_id] = remap_cell_state(cell_state)
    return doc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$V2E_PY -m pytest tests/test_remap.py::test_biological_wraps_baseline_and_remaps -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/composites/biological.py tests/test_remap.py
git commit -m "feat(biological): biological() generator wrapping baseline + remap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Build-and-run smoke test (real cache)

Proves the remapped document actually constructs and ticks under the real
process-bigraph engine (catches path-resolution errors the unit stubs can't).

**Files:**
- Test: `tests/test_biological_equivalence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biological_equivalence.py
"""Phase-1 equivalence gate: the biological composite is bit-identical to
baseline. See docs/superpowers/specs/2026-06-13-biological-composite-design.md.
"""
from __future__ import annotations

import os

import pytest

import v2ecoli.library.unit_bridge  # noqa: F401  (registers pint units pre-cache)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache')),
        reason="ParCa cache not present; set V2ECOLI_CACHE_DIR or build it.",
    ),
]

CACHE = os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache')


def test_biological_builds_and_runs_one_step():
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.biological import biological

    core = build_core()
    doc = biological(core=core, seed=0, cache_dir=CACHE, emitter='null')
    composite = Composite(doc, core=core)
    composite.run(1)  # must not raise

    agent = composite.state['agents']['0']
    assert 'cell' in agent and 'molecules' in agent['cell']
    assert 'bulk' not in agent
```

- [ ] **Step 2: Run test to verify it fails (or errors meaningfully)**

Run: `$V2E_PY -m pytest tests/test_biological_equivalence.py::test_biological_builds_and_runs_one_step -q`
Expected: At this point the code from Tasks 1–3 exists, so this should PASS once
the cache is available. If it FAILS, the failure localizes a real wiring bug
(e.g. an edge whose wires were not rewritten, or a hard-coded path consumer).
Fix the remap until it passes — do **not** weaken the assertions.

- [ ] **Step 3: (only if Step 2 failed) diagnose & fix**

Likely culprits and fixes:
- A process reads a store by hard-coded absolute path bypassing `make_edge`.
  Grep for it: `grep -rn "'bulk'\|\"bulk\"\|'listeners'" v2ecoli/processes v2ecoli/steps | grep -i "state\[\|\.get("`. If found, that path must be added to the remap's awareness or the consumer left on the original key. Document the finding in the spec's "Open questions" section.
- An edge type not in `_EDGE_TYPES`. Inspect `composite.state['agents']['0']` keys for dict values with a `_type` not in `('step','process')` and extend `_EDGE_TYPES`.

- [ ] **Step 4: Run test to verify it passes**

Run: `$V2E_PY -m pytest tests/test_biological_equivalence.py::test_biological_builds_and_runs_one_step -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_biological_equivalence.py
git commit -m "test(biological): smoke build+run of the biological composite

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Bit-identical equivalence gate (the core proof)

**Files:**
- Modify: `tests/test_biological_equivalence.py`

Run baseline and biological from the **same cache bundle and seed**, step both
the same number of times with the null emitter, and assert every data leaf
matches after projecting biological paths back through the remap. Uses the
existing `tests/_state_equal.py:deep_equal`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_biological_equivalence.py
N_STEPS = 50  # enough to exercise transcription/translation/metabolism/replication


def _data_pairs(baseline_agent, bio_agent):
    """Yield (label, baseline_value, biological_value) for every data store we
    assert on: the bulk pool, all unique molecules, and the listeners tree."""
    from v2ecoli.composites._remap import REMAP, UNIQUE_REMAP

    def _dig(tree, path):
        node = tree
        for seg in path:
            node = node[seg]
        return node

    # bulk + listeners (relocated whole)
    for old_key in ('bulk', 'listeners'):
        yield (old_key, baseline_agent[old_key], _dig(bio_agent, REMAP[old_key]))
    # every unique molecule present in baseline
    for uname, val in baseline_agent['unique'].items():
        target = UNIQUE_REMAP.get(uname, ('cell', uname))
        yield (f'unique/{uname}', val, _dig(bio_agent, target))


def test_biological_is_bit_identical_to_baseline():
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline, load_cache_bundle
    from v2ecoli.composites.biological import biological
    from tests._state_equal import deep_equal

    bundle = load_cache_bundle(CACHE)  # one load -> identical initial state

    core_b = build_core()
    base = Composite(baseline(core=core_b, seed=0, bundle=bundle, emitter='null'),
                     core=core_b)
    core_x = build_core()
    bio = Composite(biological(core=core_x, seed=0, bundle=bundle, emitter='null'),
                    core=core_x)

    for step in range(N_STEPS):
        base.run(1)
        bio.run(1)
        ba = base.state['agents']['0']
        xa = bio.state['agents']['0']
        for label, bval, xval in _data_pairs(ba, xa):
            ok, reason = deep_equal(bval, xval, path=label)
            assert ok, f"divergence at step {step+1}, store {label}: {reason}"
```

- [ ] **Step 2: Run test to verify it fails (or passes — this is the gate)**

Run: `$V2E_PY -m pytest tests/test_biological_equivalence.py::test_biological_is_bit_identical_to_baseline -q`
Expected: PASS if the remap is faithful. A FAIL prints the exact step + store +
dotted leaf path that diverged — investigate with systematic-debugging; the
cause is always either an unrewritten wire or a hard-coded path consumer (see
Task 4 Step 3). Do not relax `deep_equal` or shrink `N_STEPS` to make it pass.

- [ ] **Step 3: (if needed) fix the remap, re-run until green**

- [ ] **Step 4: Run the full remap + equivalence suite**

Run: `$V2E_PY -m pytest tests/test_remap.py tests/test_biological_equivalence.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tests/test_biological_equivalence.py
git commit -m "test(biological): bit-identical equivalence gate vs baseline (50 steps)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Biological-marker comparison report

A side-by-side baseline-vs-biological run that also reports the biological
markers from the spec (growth rate, division-relevant mass, dry-mass fractions),
emitting a small self-contained HTML. For Phase 1 these markers are *identical*
(it's the same simulation); the script exists so Phase 2's tolerant comparison
has a home and so the equivalence is visible, not just asserted in CI.

**Files:**
- Create: `scripts/compare_biological.py`

- [ ] **Step 1: Write the script**

```python
# scripts/compare_biological.py
"""Run baseline vs biological for N steps and emit an HTML comparison of mass /
growth markers. Phase 1: the two are identical by construction; this makes that
visible and is the Phase-2 (tolerant) comparison's entry point.

Usage:
    python scripts/compare_biological.py --steps 100 \
        --cache "$V2ECOLI_CACHE_DIR" --out out/biological_comparison.html
"""
from __future__ import annotations

import argparse
import os

import v2ecoli.library.unit_bridge  # noqa: F401


def _mass(agent_listeners) -> dict:
    m = agent_listeners.get('mass', {})
    def _f(x):
        return float(getattr(x, 'magnitude', x))
    return {k: _f(m[k]) for k in ('cell_mass', 'dry_mass') if k in m}


def run(steps: int, cache: str):
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline, load_cache_bundle
    from v2ecoli.composites.biological import biological

    bundle = load_cache_bundle(cache)
    cb = build_core()
    base = Composite(baseline(core=cb, seed=0, bundle=bundle, emitter='null'), core=cb)
    cx = build_core()
    bio = Composite(biological(core=cx, seed=0, bundle=bundle, emitter='null'), core=cx)

    rows = []
    for i in range(steps):
        base.run(1); bio.run(1)
        bm = _mass(base.state['agents']['0']['listeners'])
        xm = _mass(bio.state['agents']['0']['cell']['observables'])
        rows.append((i + 1, bm, xm))
    return rows


def to_html(rows) -> str:
    head = ("<tr><th>step</th><th>baseline cell_mass</th>"
            "<th>biological cell_mass</th><th>Δ</th></tr>")
    body = []
    for step, bm, xm in rows:
        b = bm.get('cell_mass', float('nan'))
        x = xm.get('cell_mass', float('nan'))
        body.append(f"<tr><td>{step}</td><td>{b:.6g}</td>"
                    f"<td>{x:.6g}</td><td>{abs(b - x):.3g}</td></tr>")
    return ("<html><head><meta charset='utf-8'><title>baseline vs biological</title>"
            "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:4px 8px;font-family:monospace}</style></head><body>"
            "<h1>Baseline vs Biological — mass markers</h1>"
            "<p>Phase 1 is a pure relabel; Δ should be 0 at every step.</p>"
            f"<table>{head}{''.join(body)}</table></body></html>")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cache', default=os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache'))
    p.add_argument('--out', default='out/biological_comparison.html')
    args = p.parse_args(argv)

    rows = run(args.steps, args.cache)
    max_delta = max((abs(bm.get('cell_mass', 0) - xm.get('cell_mass', 0))
                     for _, bm, xm in rows), default=0.0)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(to_html(rows))
    print(f"wrote {args.out}; max cell_mass Δ over {args.steps} steps = {max_delta:.3g}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run the script**

Run: `$V2E_PY scripts/compare_biological.py --steps 50 --out out/biological_comparison.html`
Expected: prints `max cell_mass Δ over 50 steps = 0` (or `0.0`), and writes the HTML.

- [ ] **Step 3: Verify the delta is zero**

The printed `max cell_mass Δ` must be `0` — that's the visible confirmation of
bit-identity. A non-zero delta means Task 5 should have caught a divergence;
go back to Task 5 rather than shipping the script.

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_biological.py
git commit -m "feat(biological): baseline-vs-biological mass comparison report

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Definition of done (Phase 1)

- `tests/test_remap.py` and `tests/test_biological_equivalence.py` pass.
- `test_biological_is_bit_identical_to_baseline` is green for ≥50 steps with
  `deep_equal` (no tolerance).
- `scripts/compare_biological.py` reports `max cell_mass Δ == 0`.
- No process implementation was modified — `git diff --stat origin/main` touches
  only `composites/_remap.py`, `composites/biological.py`, the two test files,
  the script, and docs.

## Out of scope (deferred to Phase 2)

- Splitting `cell/molecules` into per-class sub-pools (needs an index-remap shim).
- Unit-bearing / `describe()`-backed leaf schemas.
- Distributing `cell/observables` into subsystem homes.
- Cross-generation / daughter-carry and the `EcoliWCM` bridge under the new layout.
- Remapping the dashboard catalog registration / explorer view.
