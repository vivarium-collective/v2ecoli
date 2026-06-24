"""Drive the pristine-upstream pbg composite multi-generationally.

Builds the upstream-wrapped founder cell under ``agents/0`` + the harness
``UpstreamDivision`` step, then ticks it through successive cell cycles via
``run_multigen_xarray`` (the canonical multigen runner — division is followed by
agent-count growth in the colony map; per-generation XArray emitters are swapped
at each division). Proves real ``CovertLab/vEcoli`` runs multi-generationally on
process-bigraph with ZERO upstream edits.

Run from the harness root with the venv python and PYTHONHASHSEED=0:
    PYTHONHASHSEED=0 .venv/bin/python scripts/run_upstream_multigen.py
"""
import os
import time

os.environ.setdefault("PYTHONHASHSEED", "0")

from v2ecoli.library.upstream_division import build_upstream_agents_composite
from v2ecoli.library.xarray_run import run_multigen_xarray

MAX_STEPS = int(os.environ.get("MAX_STEPS", "45000"))
MAX_GENS = int(os.environ.get("MAX_GENS", "16"))
CHUNK = int(os.environ.get("CHUNK", "100"))
STORE = os.environ.get("STORE", "out/upstream_multigen/lineage.zarr")


def main():
    t0 = time.time()
    composite, info = build_upstream_agents_composite(
        seed=0, condition="basal",
        exclude_processes=["monomer_counts_listener"],
    )
    print(f"[driver] built founder in {time.time()-t0:.0f}s; "
          f"agents={list(composite.state['agents'].keys())}", flush=True)

    view = [{
        "root": ("listeners",),
        "variables": {
            "mass": {
                "dry_mass": [{"path": "dry_mass", "dtype": "<f8"}],
                "cell_mass": [{"path": "cell_mass", "dtype": "<f8"}],
            },
        },
    }]
    metadata_base = {
        "experiment_id": "upstream_pbg_multigen",
        "variant": 0,
        "lineage_seed": 0,
        "time_step": 1.0,
        "max_duration": float(MAX_STEPS),
    }

    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    result = run_multigen_xarray(
        composite,
        store_path=STORE,
        view=view,
        metadata_base=metadata_base,
        max_steps=MAX_STEPS,
        max_generations=MAX_GENS,
        chunk=CHUNK,
        initial_agent_id="0",
        overwrite=True,
    )
    print(f"[driver] DONE in {time.time()-t0:.0f}s: {result}", flush=True)
    print(f"[driver] generations sustained: {len(result['generations'])} "
          f"({result['generations']})", flush=True)


if __name__ == "__main__":
    main()
