"""Join delta_prob entries to their source fold changes and grade fidelity.

Reads two STATIC artefacts only -- the ParCa cache and reconstruction's
fold_changes.tsv. No simulation is involved; the result is a property of cache
fingerprint ea05ac9b23b1d843.

The join is (TF symbol, target gene symbol) -> (tf index, TU index):
  fold_changes.tsv   TF symbol, target symbol, log2 FC
  transcription_factors.tsv   TF symbol -> activeId (the id used in tf_ids)
  genes.tsv          symbol -> gene id
  cistron_tu_mapping_matrix   cistron -> TU(s)
A cistron in a polycistronic TU maps to several TUs; every such pair is kept and
flagged, because that case IS the finding (lexA regulates rpoD; dnaG is silenced
as its operon passenger).
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

FLAT = Path(".venv/lib/python3.12/site-packages/reconstruction/ecoli/flat")


def _rows(name):
    with open(FLAT / name) as fh:
        return list(csv.reader((l for l in fh if not l.startswith("#")), delimiter="\t"))


def load():
    import dill
    from v2ecoli.core import build_core
    build_core()
    sd = dill.load(open("out/cache/sim_data_cache.dill", "rb"))
    ti = sd["configs"]["ecoli-transcript-initiation"]
    rs = sd["configs"]["rna_synth_prob_listener"]
    tf_ids = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]

    g = _rows("genes.tsv")
    gi = {h: i for i, h in enumerate(g[0])}
    # csv.reader treats '"' as the quotechar, so headers/values arrive unquoted.
    sym2id = {r[gi["symbol"]]: r[gi["id"]] for r in g[1:] if len(r) > 2}

    t = _rows("transcription_factors.tsv")
    thi = {h: i for i, h in enumerate(t[0])}
    sym2active = {}
    for r in t[1:]:
        if len(r) <= thi["activeId"]:
            continue
        s = r[thi["TF"]]
        a = r[thi["activeId"]]
        if a:
            sym2active[s] = a

    cis = [str(x) for x in rs["cistron_ids"]]
    C = rs["cistron_tu_mapping_matrix"]
    C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)

    dp = ti["delta_prob"]
    M = sp.csr_matrix((dp["deltaV"], (dp["deltaI"], dp["deltaJ"])), shape=dp["shape"])
    bp = np.asarray(ti["basal_prob"], dtype=float)
    return dict(tf_ids=tf_ids, sym2id=sym2id, sym2active=sym2active,
                cis=cis, C=C, M=M, bp=bp,
                tu_ids=[str(x) for x in ti["rna_data"]["id"]])


def join(d):
    """One record per (fold change, TU) pair that resolves to a delta entry."""
    out, unjoined = [], {"tf": 0, "gene": 0, "cistron": 0, "no_delta": 0}
    tfidx = {t: i for i, t in enumerate(d["tf_ids"])}
    cidx = {c: i for i, c in enumerate(d["cis"])}
    for r in _rows("fold_changes.tsv")[1:]:
        if len(r) < 3:
            continue
        tf_sym, tgt_sym, fc = r[0], r[1], r[2]
        try:
            fc = float(fc)
        except ValueError:
            continue
        active = d["sym2active"].get(tf_sym)
        if active is None or active not in tfidx:
            unjoined["tf"] += 1; continue
        gid = d["sym2id"].get(tgt_sym)
        if gid is None:
            unjoined["gene"] += 1; continue
        ci = cidx.get(gid)
        if ci is None:
            unjoined["cistron"] += 1; continue
        j = tfidx[active]
        tus = d["C"][ci, :].nonzero()[1] if d["C"].shape[0] == len(d["cis"]) \
            else d["C"][:, ci].nonzero()[0]
        hit = False
        for tu in tus:
            v = d["M"][tu, j]
            v = float(v.toarray().ravel()[0]) if hasattr(v, "toarray") else float(v)
            if v == 0.0 or d["bp"][tu] <= 0:
                continue
            hit = True
            out.append({"tf": tf_sym, "target": tgt_sym, "log2fc": fc,
                        "tu": d["tu_ids"][tu], "n_tu_for_cistron": int(len(tus)),
                        "implied_retained": 1.0 + v / d["bp"][tu],
                        "expected_retained": 2.0 ** fc})
        if not hit:
            unjoined["no_delta"] += 1
    return out, unjoined


def grade(recs):
    ratio = lambda r: r["implied_retained"] / r["expected_retained"]  # noqa: E731
    med = lambda v: float(np.median(v)) if v else None                # noqa: E731
    rep = [r for r in recs if r["log2fc"] < 0]
    act = [r for r in recs if r["log2fc"] > 0]
    strong = [r for r in rep if r["log2fc"] <= -1]
    weak = [r for r in rep if -0.5 <= r["log2fc"] < 0]
    lg = lambda rs: [abs(np.log10(abs(ratio(r)))) for r in rs
                     if ratio(r) != 0 and np.isfinite(ratio(r))]   # noqa: E731
    ms, mw = med(lg(strong)), med(lg(weak))
    return {
        "n_joined": len(recs), "n_repressing": len(rep), "n_activating": len(act),
        "distortion-is-systematic": med([ratio(r) for r in recs]),
        "distortion-grows-with-repression-strength": (ms / mw) if (ms and mw) else None,
        "delta-respects-the-probability-bound": sum(1 for r in recs if r["implied_retained"] < 0),
        "activators-are-encoded-faithfully": med([ratio(r) for r in act]),
        "_median_ratio_repressors": med([ratio(r) for r in rep]),
        "_strong_median_absdex": ms, "_weak_median_absdex": mw,
    }


if __name__ == "__main__":
    d = load()
    recs, unjoined = join(d)
    res = grade(recs)
    res["_unjoined"] = unjoined
    # rpoD sits in TWO TUs, so pin the one that carries dnaG rather than
    # whichever the join happens to emit first.
    lex = [r for r in recs if r["tf"] == "lexA" and r["target"] == "rpoD"
           and r["tu"] == "TU00352[c]"]
    if lex:
        r = lex[0]
        res["lexa-rpod-discrepancy"] = r["expected_retained"] / r["implied_retained"]
    print(json.dumps(res, indent=1, default=float))
    json.dump({"records": recs, "result": res}, open("/tmp/study15.json", "w"), default=float)
