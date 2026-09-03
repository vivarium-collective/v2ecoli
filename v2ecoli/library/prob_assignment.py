"""Attribute the synthesis-probability shortfall to a step in the assignment path.

Science core for ``report_cards/probability_assignment_card.py``.

transcript_initiation.py builds the probability vector per PROMOTER and then
rescales whole CLASSES:

    promoter_init_probs = basal_prob[TU_index] + delta_prob . bound_TF
    promoter_init_probs /= sum()
    promoter_init_probs[is_mrna] *= mRna_fraction / sum(mrna)
    _rescale_initiation_probs(rprotein, rnap, fixed values)
    promoter_init_probs[~is_fixed] *= scaleTheRestBy

A class rescale multiplies every member of a class by one scalar and therefore
cannot distinguish transcripts WITHIN a class. So the comparison group is
restricted to mRNA here: comparing the cohort against a mixed-class group would
confound a per-TU effect with the class structure and could manufacture a
difference that is really just "mRNA versus tRNA".

The one per-TU term is the promoter indexing itself -- a TU's total probability
scales with how many promoter copies exist -- which is why promoter_copy_number
is measured alongside.
"""
from __future__ import annotations

import glob
from pathlib import Path

COHORT = ["TU00352[c]", "TU0-13121[c]", "TU00062[c]",
          "TU00216[c]", "TU0-14047[c]", "TU0-6686[c]"]
ACTUAL = "listeners__rna_synth_prob__actual_rna_synth_prob"
PCOPY = "listeners__rna_synth_prob__promoter_copy_number"
TOP_N = 300


def _plain(v) -> float:
    return float(v.asNumber() if hasattr(v, "asNumber") else v)


def measure(cache_dir, out_root, seeds=(0, 1, 2), generations=(1, 2, 3)) -> dict:
    import dill
    import numpy as np
    import polars as pl
    from v2ecoli.core import build_core
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        sd = dill.load(f)
    ti = sd["configs"]["ecoli-transcript-initiation"]
    ids = [str(x) for x in ti["rna_data"]["id"]]
    bp = np.asarray(ti["basal_prob"], dtype=float)
    arr = ti["rna_data"].fullArray() if hasattr(ti["rna_data"], "fullArray") else ti["rna_data"]
    is_mrna = np.asarray(arr["is_mRNA"], dtype=bool)
    length = np.asarray([_plain(x) for x in arr["length"]])
    coord = np.asarray([abs(_plain(x)) for x in arr["replication_coordinate"]])

    fixed = set()
    for k in ("idx_rprotein", "idx_rnap", "idx_rRNA", "idx_tRNA"):
        v = ti.get(k)
        if v is not None:
            fixed |= set(np.asarray(v).ravel().tolist())

    cidx = {t: ids.index(t) for t in COHORT if t in ids}
    top = list(np.argsort(bp)[::-1][:TOP_N])
    comp = [i for i in top if i not in set(cidx.values())]
    comp_mrna = [i for i in comp if is_mrna[i] and i not in fixed]
    comp_fixed = [i for i in comp if i in fixed]

    act = np.zeros(len(ids)); pcop = np.zeros(len(ids)); n = 0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            for col, acc in ((ACTUAL, act), (PCOPY, pcop)):
                if col in df.columns:
                    for row in df[col].to_list():
                        acc[: len(row)] += np.asarray(row, dtype=float)
            n += df.height
    if n == 0:
        return {"error": f"no data under {out_root}"}
    act /= n
    pcop /= n

    ratio = np.divide(act, bp, out=np.zeros_like(act), where=bp > 0)
    med = lambda a, idx: float(np.median([a[i] for i in idx]))  # noqa: E731
    ci = list(cidx.values())

    c_ratio, k_ratio = med(ratio, ci), med(ratio, comp_mrna)
    q = np.percentile([ratio[i] for i in comp_mrna], [25, 75])
    fixed_ratio = med(ratio, comp_fixed) if comp_fixed else None

    def corr(vec):
        x = np.array([vec[i] for i in top]); r = np.array([ratio[i] for i in top])
        ok = (r > 0) & (x > 0)
        return abs(float(np.corrcoef(np.log(x[ok]), np.log(r[ok]))[0, 1])) if ok.sum() > 2 else None

    return {
        "n_timesteps": n,
        "cohort_ratio_median": c_ratio,
        "comparison_mrna_ratio_median": k_ratio,
        "cohort_specificity": (c_ratio / k_ratio) if k_ratio > 0 else None,
        "cohort_promoter_copies": med(pcop, ci),
        "comparison_mrna_promoter_copies": med(pcop, comp_mrna),
        "promoter_copy_ratio": (med(pcop, ci) / med(pcop, comp_mrna)
                                if med(pcop, comp_mrna) > 0 else None),
        "comparison_mrna_iqr_over_median": (float((q[1] - q[0]) / k_ratio)
                                            if k_ratio > 0 else None),
        "fixed_class_ratio_median": fixed_ratio,
        "fixed_vs_mrna_difference": (abs(fixed_ratio - k_ratio) / k_ratio
                                     if (fixed_ratio and k_ratio > 0) else None),
        "length_corr": corr(length),
        "coord_corr": corr(coord),
        "n_comparison_mrna": len(comp_mrna),
        "n_comparison_fixed": len(comp_fixed),
        "per_tu": {t: {"actual": float(act[i]), "basal": float(bp[i]),
                       "ratio": float(ratio[i]), "promoter_copies": float(pcop[i])}
                   for t, i in cidx.items()},
    }
