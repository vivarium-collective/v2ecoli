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
    from _state_equal import deep_equal  # tests/ is on sys.path under pytest

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
