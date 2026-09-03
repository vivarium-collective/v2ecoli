"""Measure the depressed-transcription cohort against its probability peers.

Science core for ``report_cards/silent_tus_card.py``. Measurement here, grading
in the card, rendering in the card's HTML.

The cohort is the six transcription units that received zero synthesis in the
single generation observed by ``dnag-production-deficit``, despite ranking in the
top 300 of 3277 by assigned probability. This module re-measures them across a
sweep, because a single generation cannot distinguish "never transcribed" from
"not transcribed that time" -- 43.3% of ALL transcripts get zero in one
generation simply because most are low-probability and a generation is a finite
number of draws.

Everything is measured as ``realized share / assigned share`` rather than as raw
counts. Raw counts confound the question with how much total transcription a
generation happens to contain; the ratio asks the sharper question of whether a
transcript is made as often as the model says it should be, and it makes the
comparison group a calibration (their ratio should sit near 1.0).

``basal_prob`` is used ONLY to define the peer ranking, never to claim what a
transcript's probability actually is: it is overridden for ``idx_rprotein`` TUs
(transcript_initiation.py:618-621). Any claim about realized behaviour comes from
``count_rna_synthesized`` in the elongation listener.
"""
from __future__ import annotations

import glob
from pathlib import Path

# The six top-300 TUs observed at zero synthesis in dnag-production-deficit.
COHORT = ["TU00352[c]", "TU0-13121[c]", "TU00062[c]",
          "TU00216[c]", "TU0-14047[c]", "TU0-6686[c]"]
DNAG_CISTRON = "EG10239"
SYN_COL = "listeners__transcript_elongation_listener__count_rna_synthesized"
TOP_N = 300


def _plain(v) -> float:
    """rna_data fields carry unum units; strip them before arithmetic."""
    return float(v.asNumber() if hasattr(v, "asNumber") else v)


def _cache(cache_dir):
    import dill
    from v2ecoli.core import build_core
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        return dill.load(f)


def measure(cache_dir, out_root, seeds=(0, 1, 2), generations=(1, 2, 3)) -> dict:
    import numpy as np
    import polars as pl

    sd = _cache(cache_dir)
    cf = sd["configs"]
    ti = cf["ecoli-transcript-initiation"]
    ids = [str(x) for x in ti["rna_data"]["id"]]
    bp = np.asarray(ti["basal_prob"], dtype=float)
    cohort_idx = {t: ids.index(t) for t in COHORT if t in ids}
    top = list(np.argsort(bp)[::-1][:TOP_N])
    comp = [i for i in top if i not in set(cohort_idx.values())]

    # Per (seed, generation) synthesis vectors -- kept separate so the null test
    # can ask whether ANY single observation is non-zero, which a pooled total
    # would hide.
    per_run: dict[tuple, "np.ndarray"] = {}
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            if SYN_COL not in df.columns:
                continue
            v = np.zeros(len(ids))
            for row in df[SYN_COL].to_list():
                v[: len(row)] += np.asarray(row, dtype=float)
            per_run[(seed, gen)] = v
    if not per_run:
        return {"error": f"no synthesis data under {out_root}"}

    tot = np.sum(list(per_run.values()), axis=0)
    realized = tot / max(tot.sum(), 1.0)
    assigned = bp / bp.sum()

    def ratio(i):
        return float(realized[i] / assigned[i]) if assigned[i] > 0 else None

    cohort = {}
    for t, i in cohort_idx.items():
        vals = [float(per_run[k][i]) for k in sorted(per_run)]
        cohort[t] = {"per_run": vals, "max": max(vals), "total": float(sum(vals)),
                     "realized_over_assigned": ratio(i),
                     "length": _plain(ti["rna_data"]["length"][i])}

    comp_ratios = [ratio(i) for i in comp if ratio(i) is not None]
    comp_counts = np.array([[per_run[k][i] for i in comp] for k in sorted(per_run)])

    # Cistrons whose EVERY transcript is in the cohort receive almost no mRNA.
    c2 = cf["rna_synth_prob_listener"]
    cis = [str(x) for x in c2["cistron_ids"]]
    M = c2["cistron_tu_mapping_matrix"]
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    cset = set(cohort_idx.values())
    stranded = [g for j, g in enumerate(cis)
                if len(np.nonzero(Md[j])[0])
                and set(np.nonzero(Md[j])[0].tolist()) <= cset]

    dnag_tus = np.nonzero(Md[cis.index(DNAG_CISTRON)])[0] if DNAG_CISTRON in cis else []
    return {
        "n_observations": len(per_run),
        "cohort": cohort,
        "cohort_max_any_run": max(v["max"] for v in cohort.values()),
        "cohort_total": float(sum(v["total"] for v in cohort.values())),
        "cohort_max_ratio": max(v["realized_over_assigned"] for v in cohort.values()),
        "comparison_n": len(comp),
        "comparison_median_count": float(np.median(comp_counts)),
        "comparison_median_ratio": float(np.median(comp_ratios)),
        "comparison_zero_fraction": float((comp_counts == 0).mean()),
        "depression_fold": (float(np.median(comp_ratios)
                                  / max(max(v["realized_over_assigned"]
                                            for v in cohort.values()), 1e-12))),
        "stranded_cistrons": stranded,
        "stranded_excluding_dnag": [g for g in stranded if g != DNAG_CISTRON],
        "dnag_total_synthesized": float(sum(tot[t] for t in dnag_tus)),
    }
