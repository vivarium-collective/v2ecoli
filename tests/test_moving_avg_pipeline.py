"""Integration smoke for the ported zarr map-reduce `moving_avg` pipeline.

Verifies the port wires up against the released viva-emitters engine (0.3.0+):
the module imports, its classes subclass the engine ABCs, and the shipped
config loads through v2ecoli's variant parser and validates. A full
``compute()`` run is intentionally out of scope here — it requires a real
multi-generation workflow store (the config's ``sim_data_path`` /
``out_dir`` are run-time inputs).
"""

from pathlib import Path

import pytest

pytest.importorskip("viva_emitters.xarray_emitter.zarr_mapreduce")
pytest.importorskip("altair")
pytest.importorskip("pandas")

from viva_emitters.xarray_emitter.storage import WorkflowConfig  # noqa: E402
from viva_emitters.xarray_emitter.zarr_mapreduce import (  # noqa: E402
    ZarrMapReduce,
    ZarrMapReduceConfig,
    ZarrMapReduceResult,
)

from v2ecoli.workflow.analyses import moving_avg  # noqa: E402
from v2ecoli.workflow.variants import parse_variant_params  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "v2ecoli" / "configs" / "moving_avg_analysis.json"


def test_pipeline_classes_subclass_engine_abcs():
    assert issubclass(moving_avg.MovingAvgPipeline, ZarrMapReduce)
    assert issubclass(moving_avg.MovingAvgConfig, ZarrMapReduceConfig)
    assert issubclass(moving_avg.MovingAvgResult, ZarrMapReduceResult)
    assert callable(moving_avg.main)


def test_shipped_config_loads_through_v2ecoli_variant_parser():
    # The injected parser is v2ecoli's parse_variant_params; the shipped config
    # is baseline-only (v2ecoli's variant grammar differs from vEcoli's).
    wc = WorkflowConfig.load(CONFIG, variant_parser=parse_variant_params)
    assert wc.is_uri is False
    assert wc.variants == []  # baseline-only template
    assert wc.sim["emitter"] == "xarray"
    # the analysis block the pipeline reads is present
    params = wc.sim["analysis_options"]["zarr_mapreduce"]["moving_avg"]
    assert params["variables"] == {"metabolic_fluxes": "FUM"}
    assert set(params["parameters"]) == {"window", "min_window"}
