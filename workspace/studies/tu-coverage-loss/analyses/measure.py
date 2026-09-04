"""TU coverage loss: which genes lose their alternative promoters, and what happens.

Coverage side is static (transcription_units.tsv vs the model's rna_data).
Silencing side reuses dnag-promoter-availability's 6-seed sweep via
actual_rna_synth_prob_per_cistron -- emitted only because the rna_synth_prob
deriver repair landed before that sweep ran.
"""
from __future__ import annotations
import csv, glob, json
from collections import defaultdict
import numpy as np, polars as pl, scipy.sparse as sp

FLAT = ".venv/lib/python3.12/site-packages/reconstruction/ecoli/flat"
ROOT = "out/dnag-promoter-availability/mechanistic"
PC = "listeners__rna_synth_prob__actual_rna_synth_prob_per_cistron"
DNAG, RPOD, RPSU = "EG10239", "EG10896", "EG10920"
LEXA_TF = 14


def coverage():
    import dill
    from v2ecoli.core import build_core
    build_core()
    rows = list(csv.reader((l for l in open(f"{FLAT}/transcription_units.tsv")
                            if not l.startswith("#")), delimiter="\t"))
    hi = {h: i for i, h in enumerate(rows[0])}
    src = {r[hi["id"]]: json.loads(r[hi["genes"]]) for r in rows[1:] if len(r) > 2}
    sd = dill.load(open("out/cache/sim_data_cache.dill", "rb"))
    ti = sd["configs"]["ecoli-transcript-initiation"]
    tu_ids = [str(x) for x in ti["rna_data"]["id"]]
    in_model = {t[:-3] for t in tu_ids}
    rs = sd["configs"]["rna_synth_prob_listener"]
    cis = [str(x) for x in rs["cistron_ids"]]
    dp = ti["delta_prob"]
    D = sp.csr_matrix((dp["deltaV"], (dp["deltaI"], dp["deltaJ"])), shape=dp["shape"]).toarray()
    bp = np.asarray(ti["basal_prob"], dtype=float)
    tu_idx = {t: i for i, t in enumerate(tu_ids)}

    g_src, g_mod = defaultdict(set), defaultdict(set)
    for t, gs in src.items():
        for g in gs:
            g_src[g].add(t)
            if t in in_model:
                g_mod[g].add(t)
    # per gene: does EVERY surviving TU carry a repressing TF?
    def repressed_everywhere(g):
        tus = [f"{t}[c]" for t in g_mod[g] if f"{t}[c]" in tu_idx]
        if not tus:
            return None
        return all((D[tu_idx[t], :] < 0).any() for t in tus)
    fixed = set()
    for k in ("idx_rprotein", "idx_rnap", "idx_rRNA", "idx_tRNA"):
        v = ti.get(k)
        if v is not None:
            fixed |= set(np.asarray(v).ravel().tolist())
    ess_cistrons = set()
    for tu_i in fixed:
        pass
    return dict(src=src, g_src=g_src, g_mod=g_mod, cis=cis, D=D, bp=bp,
                tu_idx=tu_idx, fixed=fixed, repressed_everywhere=repressed_everywhere,
                C=rs["cistron_tu_mapping_matrix"])


def cistron_transcription(cis):
    """Mean per-cistron synthesis over the LAST generation, median across seeds."""
    out = []
    for s in range(6):
        gens = sorted({int(p.split("generation=")[1].split("/")[0])
                       for p in glob.glob(f"{ROOT}/seed{s}/**/history/**/generation=*/**/*.pq",
                                          recursive=True)})
        if not gens:
            continue
        g = max(gens)
        fs = sorted(glob.glob(f"{ROOT}/seed{s}/**/history/**/generation={g}/**/*.pq", recursive=True))
        df = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")
        if PC not in df.columns:
            continue
        rows = [np.asarray(r) for r in df[PC].to_list() if np.asarray(r).size == len(cis)]
        if rows:
            out.append(np.vstack(rows).mean(axis=0))
    return np.median(np.vstack(out), axis=0) if out else None
