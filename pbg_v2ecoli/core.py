"""build_core() — bigraph-schema core with v2ecoli + multi_cell types registered.

Mirrors v2ecoli's own ``v2ecoli.core.build_core`` plus the colony composite's
multi_cell base. Required because dashboard composite runs go through the
run-runner with a non-None ``core`` argument, so each generator's own
``if core is None: register…`` branch is skipped — the core handed in must
already have everything the composites need.

Provides:
- multi_cell base types (notably ``pymunk_agent`` for colony physics).
- ``ECOLI_TYPES`` (16 v2ecoli-specific types).
- ``EcoliWCM`` link (the whole-cell bridge process).
"""
from multi_cell import core_import
from v2ecoli.bridge import EcoliWCM
from v2ecoli.types import ECOLI_TYPES


def build_core():
    core = core_import()
    core.register_types(ECOLI_TYPES)
    core.register_link("EcoliWCM", EcoliWCM)
    return core
