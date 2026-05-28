"""Regenerate the xarray golden fixture.

Run this script when PR #414's transducer/writer behavior changes upstream.
The output Zarr store at tests/fixtures/xarray_golden/store.zarr is checked
into git and serves as the parity baseline for v2ecoli.library.xarray_emitter.

USAGE: cd v2ecoli && uv run python tests/fixtures/xarray_golden/regenerate.py
"""

import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
STORE = HERE / "store.zarr"


def _minimal_xarray_config(out_uri: str) -> dict:
    """Same shape as tests/conftest.py:minimal_xarray_config, hard-coded
    here so this script can run standalone."""
    return {
        "emit": {"global_time": "node"},
        "out_uri": out_uri,
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": [
            {
                "root": ("listeners",),
                "variables": {
                    "global_time": [{"path": "global_time", "dtype": "<f4"}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": out_uri,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "golden", "variant": 0,
            "lineage_seed": 0, "agent_id": "1",
            "time_step": 1.0, "max_duration": 4.0,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }


def main() -> None:
    from bigraph_schema import allocate_core

    from v2ecoli.library.xarray_emitter import XArrayEmitter

    if STORE.exists():
        shutil.rmtree(STORE)

    core = allocate_core()
    cfg = _minimal_xarray_config(str(STORE))

    emitter = XArrayEmitter(config=cfg, core=core)
    for t in range(4):
        # State must use agents/<agent_id> wrapping (vivarium store layout).
        emitter.update({
            "time": float(t),
            "agents": {"1": {"listeners": {"global_time": float(t)}}},
        })
    emitter.close(success=True)
    print(f"wrote golden fixture: {STORE}")


if __name__ == "__main__":
    main()
