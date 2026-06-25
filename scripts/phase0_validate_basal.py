"""Phase 0 gate: run both engines basal locally (shallow) and assert physical mass.

Exit 0 iff BOTH the upstream-wrapper and v2ecoli basal runs grow cell_mass ~2x
per generation and divide for the requested number of generations. This is the
dynamics validation PR #289 is missing; a green run here gates merging #289 + #147.

    PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/phase0_validate_basal.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts._compare.physical_validity import assess_physical, load_cell_mass

GENS = int(os.environ.get("PHASE0_GENS", "2"))
MAX_STEPS = int(os.environ.get("PHASE0_MAX_STEPS", "9000"))
ENGINES = {
    "v2ecoli": "out/cache",
    "vecoli": "out/compare_harness/vecoli_parca",
}


def _run(engine: str, cache_dir: str) -> str:
    out_root = f"out/phase0/{engine}"
    cmd = [
        ".venv/bin/python", "scripts/run_comparison_ensemble.py",
        "--composite", engine, "--condition", "basal", "--cache-dir", cache_dir,
        "--n-seeds", "1", "--max-generations", str(GENS),
        "--max-steps", str(MAX_STEPS), "--mode", "serial", "--out-root", out_root,
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": os.getcwd()}
    print(f"[phase0] running {engine}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    return f"{out_root}/{engine}_seed00.zarr"


def main() -> int:
    results = {}
    ok = True
    for engine, cache_dir in ENGINES.items():
        store = _run(engine, cache_dir)
        cm = load_cell_mass(store)
        v = assess_physical(cm, min_generations=GENS)
        results[engine] = {
            "store": store, "physical": v.physical,
            "generations_reached": v.generations_reached,
            "divisions_detected": v.divisions_detected,
            "per_gen_ratios": v.per_gen_ratios, "reasons": v.reasons,
        }
        ok = ok and v.physical
        print(f"[phase0] {engine}: physical={v.physical} ratios={v.per_gen_ratios} "
              f"reasons={v.reasons}", flush=True)
    Path("out/phase0").mkdir(parents=True, exist_ok=True)
    Path("out/phase0/verdict.json").write_text(json.dumps(results, indent=2))
    print(f"[phase0] GATE {'PASS' if ok else 'FAIL'} -> out/phase0/verdict.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
