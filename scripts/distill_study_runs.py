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
PROM = "listeners__rna_synth_prob__promoter_copy_number"
TGT = "listeners__rna_synth_prob__target_rna_synth_prob"
PPGPP = "listeners__growth_limits__ppgpp_conc"
KEEP_VEC = (ACT, TGT)


def generations(run_dir: Path):
    return sorted({int(p.split("generation=")[1].split("/")[0])
                   for p in glob.glob(f"{run_dir}/**/generation=*", recursive=True)})


def distil_run(run_dir: Path, dest: Path, vectors=KEEP_VEC, idx=None):
    """`idx` = (dnag, peers) enables the NON-LINEAR per-timestep statistics.

    A per-TU generation mean cannot reproduce a zero-fraction, nor a
    median-taken-per-timestep-then-averaged: median-then-mean != mean-then-median.
    Studies that use those must have them computed HERE, during the streaming
    pass, or distillation silently destroys the evidence behind their claims.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in list(run_dir.glob("*.json")) + list(run_dir.glob("*.log")):
        shutil.copy2(f, dest / f.name)
    out = {}
    for gen in generations(run_dir):
        files = sorted(glob.glob(f"{run_dir}/**/generation={gen}/**/*.pq", recursive=True))
        acc, n, ppg = {}, 0, []
        nz = {}          # per-vector count of timesteps where dnaG is zero
        pmed = {}        # per-vector running sum of the per-timestep peer median
        for f in files:
            names = pl.scan_parquet(f).collect_schema().names()
            want = [c for c in tuple(vectors) + (PPGPP,) if c in names]
            if not want:
                continue
            df = pl.scan_parquet(f).select(want).collect()
            for col in [c for c in vectors if c in want]:
                for row in df[col].to_list():
                    v = np.asarray(row, dtype=float)
                    acc[col] = v.copy() if col not in acc else acc[col] + v
                    if idx is not None:
                        d, peers = idx
                        if d < len(v):
                            nz[col] = nz.get(col, 0) + (1 if v[d] == 0 else 0)
                            pk = [v[i] for i in peers if i < len(v)]
                            if pk:
                                pmed[col] = pmed.get(col, 0.0) + float(np.median(pk))
                    if col == ACT:
                        n += 1
            if PPGPP in want:
                ppg.append(float(df[PPGPP].mean()))
        # timestep count for vectors other than ACT (a run may not emit ACT)
        n_ts = n or max((int(round(pmed.get(c, 0) and 0)) for c in acc), default=0)
        if not acc:
            continue
        rec = {"n": n, "ppgpp_conc": float(np.mean(ppg)) if ppg else float("nan")}
        for col, v in acc.items():
            rec[col] = (v / max(n, 1)).tolist()
        if idx is not None:
            for col in acc:
                if col in nz:
                    rec[f"{col}__dnag_zero_frac"] = nz[col] / max(n, 1)
                if col in pmed:
                    rec[f"{col}__peer_median_mean"] = pmed[col] / max(n, 1)
        out[str(gen)] = rec
    (dest / "per_generation.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    vectors = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else KEEP_VEC
    idx = None
    if len(sys.argv) > 4:                      # cache dir -> enable non-linear stats
        sys.path.insert(0, ".")
        from v2ecoli.library import generational_decay as gd
        d, peers, _ = gd.indices(sys.argv[4])
        idx = (d, peers)
    manifest = {"source": str(src), "kept_vectors": list(vectors),
                "nonlinear_stats": bool(idx),
                "kept_scalars": [PPGPP, "n"], "dropped": "per-timestep time series",
                "runs": {}}
    for run_dir in sorted(p.parent for p in src.glob("*/*/run.log")):
        rel = run_dir.relative_to(src)
        n_pq = len(glob.glob(f"{run_dir}/**/*.pq", recursive=True))
        size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        rec = distil_run(run_dir, dst / rel, vectors=vectors, idx=idx)
        manifest["runs"][str(rel)] = {
            "generations": sorted(int(g) for g in rec), "n_parquet_files": n_pq,
            "original_bytes": size}
        print(f"  {rel}: {len(rec)} gens, {n_pq} pq, {size/2**30:.2f} GB -> distilled")
    for f in src.glob("*.log"):
        shutil.copy2(f, dst / f.name)
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")
