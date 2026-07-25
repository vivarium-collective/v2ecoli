"""Kinetic tRNA-charging baseline composite (55 processes, partitioned).

Sibling of :mod:`v2ecoli.composites.baseline` that swaps the polypeptide
elongation Process from :class:`SteadyStatePolypeptideElongation` to
:class:`KineticTrnaChargingPolypeptideElongation` (the per-ribosome
codon-tracking model ported from
``CovertLab/vEcoli@trna_charging_final::polypeptide_elongation.py:2198``).

Everything else — allocator wiring, listener emission, partition orchestration,
emitter declarations — matches ``baseline`` tick-for-tick.

Implementation note
-------------------

The orchestration body (``_get_step_config``, ``BASE_EXECUTION_LAYERS``,
emitter resolution) is identical to ``baseline``'s, so we delegate via a
scoped ``PARTITIONED_PROCESSES`` swap rather than copy ~600 LOC. The swap
is bounded by a context manager (:func:`_use_kinetic_partitioned_processes`)
that restores the original mapping after the build, so the dirty mutation
can't leak into a subsequent ``baseline`` build in the same Python process.

A clean alternative would refactor ``baseline.baseline`` to accept a
``partitioned_processes`` parameter. Deferred — too risky for this session's
scope; this monkey-patch is functionally equivalent and isolated.

Future work: factor ``baseline._build_baseline_doc(...)`` accepting a
process-overrides dict, then make this module a one-line delegation. Tracked
in audit.md "Task #3f progress log".
"""

from __future__ import annotations

import contextlib
from typing import Any

from pbg_superpowers.composite_generator import composite_generator, emitter_defaults

from v2ecoli.composites import _helpers
from v2ecoli.composites.baseline import baseline as _baseline
from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS
from v2ecoli.processes.polypeptide.kinetic_charging import (
    KineticTrnaChargingPolypeptideElongation,
)


@contextlib.contextmanager
def _use_kinetic_partitioned_processes():
    """Scope a ``PARTITIONED_PROCESSES`` swap to the body of the with-block.

    Mutates the module-level ``v2ecoli.composites._helpers.PARTITIONED_PROCESSES``
    dict to replace ``SteadyStatePolypeptideElongation`` with
    ``KineticTrnaChargingPolypeptideElongation`` for the
    ``ecoli-polypeptide-elongation`` slot. Restores the original entry on exit,
    even on exception, so a subsequent ``baseline()`` build in the same process
    is unaffected.
    """
    original = _helpers.PARTITIONED_PROCESSES["ecoli-polypeptide-elongation"]
    _helpers.PARTITIONED_PROCESSES["ecoli-polypeptide-elongation"] = (
        KineticTrnaChargingPolypeptideElongation
    )
    try:
        yield
    finally:
        _helpers.PARTITIONED_PROCESSES["ecoli-polypeptide-elongation"] = original


@composite_generator(
    name="kinetic_charging_baseline",
    description=(
        "55-process partitioned whole-cell E. coli model with the kinetic "
        "tRNA-charging elongation Process — peer of `baseline` that swaps "
        "SteadyStatePolypeptideElongation for "
        "KineticTrnaChargingPolypeptideElongation. Same allocator wiring, "
        "same listener emission, same emitter."
    ),
    parameters={
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to ParCa cache directory",
        },
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": "Declarative '<process>.<key>': value config overrides (variants)",
        },
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    emitters=[
        # Same default emitter declaration as baseline — parquet,
        # hive-partitioned, paths cover global_time + bulk + listeners.
        {
            "address": "local:ParquetEmitter",
            "config": {},
            "paths": ["global_time", "bulk", "listeners"],
        },
    ],
)
def kinetic_charging_baseline(
    core: Any = None,
    *,
    seed: int = 0,
    cache_dir: str = "out/cache",
    config_overrides: dict | None = None,
    bundle: dict | None = None,
) -> dict:
    """Build the process-bigraph state document for the kinetic_charging_baseline arch.

    Delegates to :func:`v2ecoli.composites.baseline.baseline` with a scoped
    ``PARTITIONED_PROCESSES`` swap so the polypeptide elongation slot resolves
    to :class:`KineticTrnaChargingPolypeptideElongation` instead of the
    steady-state default.

    Args:
        core: bigraph-schema core. Forwarded to ``baseline()``.
        seed: Master RNG seed.
        cache_dir: ParCa cache directory.
        config_overrides: Per-process config overrides (``"<process>.<key>": value``).
        bundle: Optional pre-loaded cache bundle.

    Returns:
        Process-bigraph document dict.
    """
    with _use_kinetic_partitioned_processes():
        return _baseline(
            core=core,
            seed=seed,
            cache_dir=cache_dir,
            config_overrides=config_overrides,
            bundle=bundle,
        )
