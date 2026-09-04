"""Evaluate dnag-promoter-availability's five pre-registered axes.

ONE aggregation feeds every axis, so no axis re-derives its own numbers -- the
conflation that produced the misleading cumulative figures behind
dnag-generational-decay.
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np, polars as pl
from v2ecoli.library import generational_decay as gd

ROOT = "out/dnag-promoter-availability/mechanistic"
P = "listeners__rna_synth_prob__promoter_copy_number"
A = "listeners__rna_synth_prob__actual_rna_synth_prob"
BANDS = {"deriver-outputs-are-non-empty": (0.99, 1.0),
         "dnag-has-zero-promoters-at-generation-4": (0.5, 1.0),
         "peers-keep-their-promoters": (0.67, 1.5),
         "promoter-loss-explains-the-collapse": (0.67, 1.5)}


def per_seed(seeds=range(6)):
    dnag, peers, _ = gd.indices("out/cache")
    out = {}
    for s in seeds:
        d = f"{ROOT}/seed{s}"
        if not glob.glob(f"{d}/*_summary.json"):
            continue
        txt = open(f"{d}/run.log").read() if os.path.exists(f"{d}/run.log") else ""
        gens = {int(m[0]): (float(m[1]), float(m[2]), m[3]) for m in re.findall(
            r"gen (\d+) summary: tau=\s*([\d.]+) min\s+final_dry=\s*([\d.]+) fg\s+divided=(\w+)", txt)}
        rec = {}
        for g in sorted(set(list(gens) + list(range(1, 6)))):
            fs = sorted(glob.glob(f"{d}/**/history/**/generation={g}/**/*.pq", recursive=True))
            if not fs:
                continue
            df = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")
            rows = [np.asarray(r) for r in df[P].to_list()]
            full = [r for r in rows if r.size == 3277]
            act = np.vstack([np.asarray(r) for r in df[A].to_list()
                             if np.asarray(r).size == 3277])
            if not full:
                continue
            arr = np.vstack(full)
            rec[g] = {"n": len(rows), "full_frac": len(full) / len(rows),
                      "dnag_prom": float(arr[:, dnag].mean()),
                      "dnag_zero_frac": float((arr[:, dnag] == 0).mean()),
                      "peer_prom": float(np.median(arr[:, peers], axis=1).mean()),
                      "dnag_synth": float(act[:, dnag].mean()),
                      "tau": gens.get(g, (None,))[0],
                      "divided": gens.get(g, (None, None, None))[2]}
        out[s] = rec
    return out


def evaluate(pg):
    med = lambda v: float(np.median(v)) if v else None      # noqa: E731
    late = lambda r: max(r)                                  # noqa: E731
    a1 = med([pg[s][g]["full_frac"] for s in pg for g in pg[s]])
    a2 = med([pg[s][late(pg[s])]["dnag_zero_frac"] for s in pg])
    a3 = med([pg[s][late(pg[s])]["peer_prom"] / pg[s][1]["peer_prom"]
              for s in pg if 1 in pg[s] and pg[s][1]["peer_prom"]])
    ratios = []
    for s in pg:
        L = late(pg[s])
        if 1 not in pg[s]:
            continue
        pc = pg[s][L]["dnag_prom"] / pg[s][1]["dnag_prom"]
        sc = pg[s][L]["dnag_synth"] / pg[s][1]["dnag_synth"]
        if pc > 0 and 0 < sc < 1:
            ratios.append(abs(np.log(pc)) / abs(np.log(sc)))
    a4 = med(ratios)
    a5 = med([pg[s][1]["dnag_prom"] for s in pg if 1 in pg[s]])
    return {"deriver-outputs-are-non-empty": a1,
            "dnag-has-zero-promoters-at-generation-4": a2,
            "peers-keep-their-promoters": a3,
            "promoter-loss-explains-the-collapse": a4,
            "dnag-promoter-count-at-generation-1": a5}


if __name__ == "__main__":
    pg = per_seed()
    res = evaluate(pg)
    print(f"seeds evaluated: {sorted(pg)}")
    for k, v in res.items():
        if k in BANDS:
            lo, hi = BANDS[k]
            ok = v is not None and lo <= v <= hi
            print(f"  {'PASS' if ok else 'FAIL'}  {k:44} {v:.6g}   band [{lo}, {hi}]")
        else:
            print(f"  {'PASS' if v <= 8 else 'FAIL'}  {k:44} {v:.6g}   pin <= 8")
    json.dump({"per_seed": {str(s): {str(g): r for g, r in pg[s].items()} for s in pg},
               "axes": res}, open("/tmp/study12_eval.json", "w"), indent=1)
