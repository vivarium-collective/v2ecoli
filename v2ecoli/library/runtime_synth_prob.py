"""Decompose the transcription deficit into assignment vs selection.

Science core for ``report_cards/runtime_synth_prob_card.py``.

Study 4 showed a six-TU cohort is realized ~51x below its assigned probability
and that no static ParCa attribute distinguishes it. Two explanations remain and
they need different repairs:

  assignment  the probability actually IN FORCE at runtime is depressed, so the
              transcripts are under-supplied by construction
  selection   the probability in force is normal and transcripts are simply not
              drawn from it, which would mean the process does not transcribe
              according to the probabilities it computes

The model emits the quantities that separate these, and the cache does not
expose them: ``actual_rna_synth_prob`` (in force), ``target_rna_synth_prob``
(before/after regulatory adjustment) and ``n_bound_TF_per_TU``.

Everything is compared cohort-vs-comparison rather than against an absolute
scale, because probabilities are normalised across all TUs and an absolute value
carries no meaning on its own. The comparison group doubles as the instrument
check: it is realized at 1.006x assigned, so its actual and target must agree, or
the columns do not mean what this module assumes.
"""
from __future__ import annotations

import glob
from pathlib import Path

COHORT = ["TU00352[c]", "TU0-13121[c]", "TU00062[c]",
          "TU00216[c]", "TU0-14047[c]", "TU0-6686[c]"]
ACTUAL = "listeners__rna_synth_prob__actual_rna_synth_prob"
TARGET = "listeners__rna_synth_prob__target_rna_synth_prob"
NTF = "listeners__rna_synth_prob__n_bound_TF_per_TU"
SYN = "listeners__transcript_elongation_listener__count_rna_synthesized"
TOP_N = 300


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
    cidx = {t: ids.index(t) for t in COHORT if t in ids}
    top = list(np.argsort(bp)[::-1][:TOP_N])
    comp = [i for i in top if i not in set(cidx.values())]

    act_sum = np.zeros(len(ids)); tgt_sum = np.zeros(len(ids))
    tf_sum = np.zeros(len(ids)); syn_tot = np.zeros(len(ids)); n_rows = 0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            for col, acc in ((ACTUAL, act_sum), (TARGET, tgt_sum), (NTF, tf_sum)):
                if col in df.columns:
                    for row in df[col].to_list():
                        acc[: len(row)] += np.asarray(row, dtype=float)
            if SYN in df.columns:
                for row in df[SYN].to_list():
                    syn_tot[: len(row)] += np.asarray(row, dtype=float)
            n_rows += df.height
    if n_rows == 0:
        return {"error": f"no data under {out_root}"}

    act = act_sum / n_rows      # mean probability in force per TU
    tgt = tgt_sum / n_rows
    tf = tf_sum / n_rows
    realized = syn_tot / max(syn_tot.sum(), 1.0)
    assigned = bp / bp.sum()

    ci = list(cidx.values())
    med = lambda a, idx: float(np.median([a[i] for i in idx]))  # noqa: E731

    c_act, k_act = med(act, ci), med(act, comp)
    c_tgt, k_tgt = med(tgt, ci), med(tgt, comp)
    # realized per unit of probability actually in force -- the selection fairness
    c_eff = med(realized, ci) / c_act if c_act > 0 else None
    k_eff = med(realized, comp) / k_act if k_act > 0 else None

    ratio_at = (c_act / c_tgt) if c_tgt > 0 else None
    k_ratio_at = (k_act / k_tgt) if k_tgt > 0 else None
    return {
        "n_timesteps": n_rows,
        "cohort_actual_median": c_act,
        "comparison_actual_median": k_act,
        "actual_depression": (c_act / k_act) if k_act > 0 else None,
        "cohort_target_median": c_tgt,
        "comparison_target_median": k_tgt,
        "cohort_actual_over_target": ratio_at,
        "comparison_actual_over_target": k_ratio_at,
        "comparison_control_fold": (max(k_ratio_at, 1 / k_ratio_at)
                                    if k_ratio_at else None),
        "cohort_realized_per_prob": c_eff,
        "comparison_realized_per_prob": k_eff,
        "selection_fairness": (c_eff / k_eff) if (c_eff and k_eff) else None,
        "cohort_tf_median": med(tf, ci),
        "comparison_tf_median": med(tf, comp),
        "cohort_max_realized_over_assigned": max(
            float(realized[i] / assigned[i]) for i in ci if assigned[i] > 0),
        "per_tu": {t: {"actual": float(act[i]), "target": float(tgt[i]),
                       "realized": float(realized[i]), "assigned": float(assigned[i]),
                       "n_tf": float(tf[i])} for t, i in cidx.items()},
    }
