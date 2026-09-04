"""Grade the LexA-repression account of dnaG's collapse.

Analysis only -- reuses dnag-promoter-availability's 6-seed sweep. Every term is
measured rather than assumed:
  b          per-timestep in-force ppGpp basal, reconstructed from the emitted
             ppgpp_conc (validated to 1.0003 against the per-generation mean)
  occupancy  from bound_TF_coordinates, restored by the deriver repair
  delta      the in-force relative delta built with ppgpp=True
  measured   target_rna_synth_prob / promoter_copy_number

Per timestep the code computes, for a promoter of TU i:
    p = b_i                       when the TF is unbound
    p = b_i * (1 + delta_i)       when it is bound
so the timestep-average prediction is b_i * (occ*(1+delta_i) + (1-occ)).
"""
from __future__ import annotations
import glob, json
import numpy as np, polars as pl

ROOT = "out/dnag-promoter-availability/mechanistic"
T = "listeners__rna_synth_prob__target_rna_synth_prob"
P = "listeners__rna_synth_prob__promoter_copy_number"
C = "listeners__growth_limits__ppgpp_conc"
BI = "listeners__rna_synth_prob__bound_TF_indexes"
BC = "listeners__rna_synth_prob__bound_TF_coordinates"
LEXA_TF = 14


def context():
    import dill, scipy.sparse as sp
    from v2ecoli.core import build_core
    from v2ecoli import build_composite
    from v2ecoli.library import generational_decay as gd
    build_core()
    sd = dill.load(open("out/cache/sim_data_cache.dill", "rb"))
    ti = sd["configs"]["ecoli-transcript-initiation"]
    ids = [str(x) for x in ti["rna_data"]["id"]]
    arr = ti["rna_data"].fullArray() if hasattr(ti["rna_data"], "fullArray") else ti["rna_data"]
    coords = np.asarray(arr["replication_coordinate"])
    dnag, peers, rprot = gd.indices("out/cache")
    comp = build_composite("ecoli_baseline", cache_dir="out/cache")
    step = comp.state["agents"]["0"]["ppgpp-initiation"]["instance"]
    tiP = comp.state["agents"]["0"]["ecoli-transcript-initiation"]["instance"]
    D = np.asarray(tiP.delta_prob_matrix)
    bp = np.asarray(ti["basal_prob"], dtype=float)
    lex_col = D[:, LEXA_TF]
    targets = [i for i in np.nonzero(lex_col)[0] if bp[i] > 0]
    cohort = [i for i in targets if 1 + lex_col[i] < 1e-6]
    unreg_peers = [i for i in peers if lex_col[i] == 0]
    return dict(ids=ids, coords=coords, dnag=dnag, peers=peers, cohort=cohort,
                unreg_peers=unreg_peers, D=D, step=step,
                delta_dnag=float(lex_col[dnag]),
                eg10620=ids.index("EG10620_RNA[c]") if "EG10620_RNA[c]" in ids else None)


def per_gen(ctx, seed, gen):
    from v2ecoli.processes.parca.wholecell.utils import units
    fs = sorted(glob.glob(f"{ROOT}/seed{seed}/**/history/**/generation={gen}/**/*.pq", recursive=True))
    if not fs:
        return None
    df = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")
    full = lambda col: [np.asarray(r) for r in df[col].to_list() if np.asarray(r).size == 3277]  # noqa: E731
    tgt, prm = full(T), full(P)
    if not tgt or not prm:
        return None
    tgt, prm = np.vstack(tgt), np.vstack(prm)
    conc = np.asarray(df[C].to_list(), dtype=float)
    # occupancy at dnaG's coordinate
    coord = ctx["coords"][ctx["dnag"]]
    occ, n = 0, 0
    for idxs, crds in zip(df[BI].to_list(), df[BC].to_list()):
        crds = np.asarray(crds)
        if crds.size == 0:
            continue
        n += 1
        m = np.abs(crds - coord) <= 1
        if m.any() and LEXA_TF in np.asarray(idxs)[m].tolist():
            occ += 1
    occupancy = occ / max(n, 1)
    # in-force basal for dnaG, averaged per timestep (subsampled for cost)
    sub = conc[:: max(1, len(conc) // 200)]
    bs = [float(ctx["step"].synth_prob(c * units.umol / units.L, ctx["step"].copy_number)[0][ctx["dnag"]])
          for c in sub]
    b = float(np.mean(bs))
    d = ctx["delta_dnag"]
    predicted = b * (occupancy * (1.0 + d) + (1.0 - occupancy))
    measured = float(tgt[:, ctx["dnag"]].mean() / max(prm[:, ctx["dnag"]].mean(), 1e-12))
    return {"occupancy": occupancy, "b": b, "predicted": predicted, "measured": measured,
            "cohort_silenced": int(sum(1 for i in ctx["cohort"] if tgt[:, i].mean() < 1e-9)),
            "cohort_n": len(ctx["cohort"]),
            "unreg_peer_ratio": float(np.median([tgt[:, i].mean() / max(prm[:, i].mean(), 1e-12) /
                                                 max(float(np.mean([
                                                     ctx["step"].synth_prob(c * units.umol / units.L,
                                                                            ctx["step"].copy_number)[0][i]
                                                     for c in conc[::max(1, len(conc)//20)]])), 1e-30)
                                                 for i in ctx["unreg_peers"][:15]])),
            "eg10620_min": float(tgt[:, ctx["eg10620"]].min()) if ctx["eg10620"] is not None else None}


if __name__ == "__main__":
    ctx = context()
    out = {}
    for s in range(6):
        for g in range(1, 6):
            r = per_gen(ctx, s, g)
            if r:
                out.setdefault(str(s), {})[str(g)] = r
    json.dump({"per_seed": out, "delta_dnag": ctx["delta_dnag"],
               "cohort_n": len(ctx["cohort"])}, open("/tmp/study14.json", "w"), default=float)
    print(json.dumps({"seeds": sorted(out), "delta": ctx["delta_dnag"],
                      "cohort_n": len(ctx["cohort"]),
                      "unreg_peers": len(ctx["unreg_peers"])}, indent=1))
