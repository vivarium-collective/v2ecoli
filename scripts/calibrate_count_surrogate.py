"""Run the WCM ONCE at the reference transcription scale; measure per-seed
cumulative rna_init_event; fit the count surrogate; persist calibration JSON.
Usage: .venv/bin/python scripts/calibrate_count_surrogate.py [--n-seeds 8] [--n-steps 250]
Writes .pbg/runs/phase3-surrogate-calibration.json
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
from v2ecoli import build_composite
from v2ecoli.inference.count_surrogate import calibrate_from_counts

def _total(x):
    try: return float(np.nansum(np.asarray(x, dtype=float)))
    except Exception: return 0.0

def main(n_seeds, n_steps):
    cum = []
    for seed in range(n_seeds):
        c = build_composite("ecoli_baseline", cache_dir="out/cache", seed=seed)
        ci = 0.0
        for _ in range(n_steps):
            c.run(1)
            L = (((c.state.get("agents") or {}).get("0") or {}).get("listeners") or {})
            rie = (L.get("rnap_data") or {}).get("rna_init_event")
            if rie is not None: ci += _total(rie)
        cum.append(ci); print(f"seed {seed}: cum_rna_init={ci:.0f}", flush=True)
    calib = calibrate_from_counts(np.array(cum))
    out = Path(".pbg/runs/phase3-surrogate-calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**calib, "n_seeds": n_seeds, "n_steps": n_steps,
                               "cum_rna_init": cum}, indent=2))
    print("calibration:", calib, "->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=250)
    a = ap.parse_args(); main(a.n_seeds, a.n_steps)
