"""Preset configs that reproduce exact vEcoli behavior for the generic
ParquetEmitter and XArrayEmitter in v2ecoli.library.

This module imports neither polars nor xarray/zarr — dtype values are
referenced as Polars dtype strings (``"UInt16"``, ``"UInt32"``, ...) so the
module is importable in any v2ecoli install regardless of which optional
extras are selected.

.. warning::
    The ``xarray_vecoli`` preset's ``metadata_validators`` is intentionally
    empty. PR #414's hard-coded `expected` dict (``single_daughters: True``,
    ``save: False``, ``save_times: False``, ``emit_config: False``,
    ``emit_topology: False``, ``emit_processes: False``, ``emit_unique:
    False``) is vivarium-Engine-specific and no longer meaningful in a
    process-bigraph composite — reproduced as-is would be no-ops or false
    positives.
"""

from __future__ import annotations

from typing import Any
from urllib import parse

from ecoli.library.parquet_emitter import USE_UINT16, USE_UINT32


VECOLI_PARQUET_DTYPE_OVERRIDES: dict[str, str] = (
    {name: "UInt16" for name in USE_UINT16}
    | {name: "UInt32" for name in USE_UINT32}
)


VECOLI_XARRAY_METADATA_KEYS: list[str] = [
    "experiment_id", "description", "sim_data_path", "time",
    "suffix_time", "time_step", "initial_global_time",
    "max_duration", "fail_at_max_duration",
    "lineage_seed", "seed", "variants", "n_init_sims", "generations",
    "agent_id", "parallel",
    "skip_baseline", "log_updates",
    "single_daughters", "daughter_outdir",
    "fixed_media", "condition",
    "parca_options",
    "mar_regulon", "amp_lysis",
    "divide", "d_period", "division_threshold", "division_variable",
    "chromosome_path",
]


def parquet_vecoli(
    out_dir: str,
    *,
    experiment_id: str = "default",
    variant: int = 0,
    lineage_seed: int = 0,
    agent_id: str = "1",
    generation: int | None = None,
    batch_size: int = 400,
    threaded: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ParquetEmitter config dict that reproduces vEcoli behavior.

    The returned dict is suitable for passing as the ``config`` to
    ``v2ecoli.library.parquet_emitter.ParquetEmitter``.
    """
    quoted = parse.quote_plus(experiment_id)
    metadata: dict[str, Any] = {
        "experiment_id": quoted,
        "variant": variant,
        "lineage_seed": lineage_seed,
        "generation": generation if generation is not None else len(agent_id),
        "agent_id": agent_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "out_dir": out_dir,
        "batch_size": batch_size,
        "threaded": threaded,
        "flatten_separator": "__",
        "partitioning_keys": [
            "experiment_id", "variant", "lineage_seed", "generation", "agent_id",
        ],
        "dtype_overrides": dict(VECOLI_PARQUET_DTYPE_OVERRIDES),
        "metadata": metadata,
    }


def xarray_vecoli(
    out_uri: str,
    *,
    transducer: dict[str, Any],
    view: list[Any],
    writer: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    output_metadata: dict[str, Any] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Build an XArrayEmitter config dict that reproduces vEcoli behavior.

    ``transducer`` and ``view`` are not preset-able — they describe which
    variables the user wants emitted and at what shape, which is per-composite.
    The preset only fills in the vEcoli defaults around them.
    """
    if writer is None:
        writer = {"type": "async", "out_uri": out_uri}
    return {
        "out_uri": out_uri,
        "transducer": transducer,
        "view": view,
        "writer": writer,
        "metadata": metadata,
        "metadata_keys": list(VECOLI_XARRAY_METADATA_KEYS),
        "metadata_validators": {},
        "output_metadata": output_metadata,
        "debug": debug,
    }
