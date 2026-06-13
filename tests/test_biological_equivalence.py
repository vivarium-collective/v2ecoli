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
