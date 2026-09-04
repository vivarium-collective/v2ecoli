"""Distil a completed study's parquet output to the quantities its analyses use.

Keeps, per (arm, seed, generation):
  * the MEAN over that generation's timesteps of the two per-TU probability
    vectors -- this is exactly what generational_decay.per_generation computes,
    so every deficit figure reproduces bit-for-bit;
  * mean ppGpp concentration and the timestep count (the n that gates stub
    generations);
and copies run.log / *_summary.json / run_identity.json verbatim.

Dropped: the per-timestep time series. That is what makes the saving ~1000x,
and it is also what a NEW time-resolved question would need -- re-run the sweep
if one arises. The manifest records exactly what was dropped.
"""
from __future__ import annotations
import glob, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, polars as pl

ACT = "listeners__rna_synth_prob__actual_rna_synth_prob"
TGT = "listeners__rna_synth_prob__target_rna_synth_prob"
PPGPP = "listeners__growth_limits__ppgpp_conc"
KEEP_VEC = (ACT, TGT)


def generations(run_dir: Path):
    return sorted({int(p.split("generation=")[1].split("/")[0])
                   for p in glob.glob(f"{run_dir}/**/generation=*", recursive=True)})


def distil_run(run_dir: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for f in list(run_dir.glob("*.json")) + list(run_dir.glob("*.log")):
        shutil.copy2(f, dest / f.name)
    out = {}
    for gen in generations(run_dir):
        files = sorted(glob.glob(f"{run_dir}/**/generation={gen}/**/*.pq", recursive=True))
        acc, n, ppg = {}, 0, []
        for f in files:
            names = pl.scan_parquet(f).collect_schema().names()
            want = [c for c in KEEP_VEC + (PPGPP,) if c in names]
            if not want:
                continue
            df = pl.scan_parquet(f).select(want).collect()
            for col in [c for c in KEEP_VEC if c in want]:
                for row in df[col].to_list():
                    v = np.asarray(row, dtype=float)
                    acc[col] = v.copy() if col not in acc else acc[col] + v
                    if col == ACT:
                        n += 1
            if PPGPP in want:
                ppg.append(float(df[PPGPP].mean()))
        if not n:
            continue
        rec = {"n": n, "ppgpp_conc": float(np.mean(ppg)) if ppg else float("nan")}
        for col, v in acc.items():
            rec[col] = (v / n).tolist()
        out[str(gen)] = rec
    (dest / "per_generation.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = {"source": str(src), "kept_vectors": list(KEEP_VEC),
                "kept_scalars": [PPGPP, "n"], "dropped": "per-timestep time series",
                "runs": {}}
    for run_dir in sorted(p.parent for p in src.glob("*/*/run.log")):
        rel = run_dir.relative_to(src)
        n_pq = len(glob.glob(f"{run_dir}/**/*.pq", recursive=True))
        size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        rec = distil_run(run_dir, dst / rel)
        manifest["runs"][str(rel)] = {
            "generations": sorted(int(g) for g in rec), "n_parquet_files": n_pq,
            "original_bytes": size}
        print(f"  {rel}: {len(rec)} gens, {n_pq} pq, {size/2**30:.2f} GB -> distilled")
    for f in src.glob("*.log"):
        shutil.copy2(f, dst / f.name)
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")
