"""Recover the synthesis-probability vector actually in force, and re-measure against it.

Science core for ``report_cards/inforce_reference_card.py``.

Three studies measured the transcription deficit as a ratio against the cache's
``basal_prob``. With ``ppgpp_regulation`` on -- the default -- TranscriptInitiation
REPLACES that vector with the one PpgppInitiation writes
(transcript_initiation.py:547-556), so those ratios divided by a vector the model
does not consult.

The replacement is not emitted. It is recovered here by building the composite
and reading the RESOLVED callable off the constructed step:

    comp = build_composite("ecoli_baseline", cache_dir=...)
    step = comp.state["agents"]["0"]["ppgpp-initiation"]["instance"]
    basal_ppgpp, _ = step.synth_prob(ppgpp_conc, step.copy_number)

That indirection is necessary, not incidental. The cache stores synth_prob as a
``{_function, _data}`` registry wrapper; ``initialize()`` only assigns it, and the
wrapper is resolved to a callable by the SCHEMA deserializer during composite
construction. Instantiating the step directly leaves it a dict. Editing the
emitter to record the vector instead is not an option: the emit schema lives in
_helpers.py, which is in the cache's hashed input set, so changing it invalidates
the cache and breaks comparability with every prior study.

Because this RECOMPUTES rather than reads, it carries a validation the study
grades explicitly: for the comparison transcripts -- which are realized at 1.006x
their assigned probability and therefore behave -- the reconstructed vector must
predict the observed actual_rna_synth_prob. If it fails there, the reconstruction
is wrong and nothing downstream of it may be trusted.
"""
from __future__ import annotations

import glob
from pathlib import Path

COHORT = ["TU00352[c]", "TU0-13121[c]", "TU00062[c]",
          "TU00216[c]", "TU0-14047[c]", "TU0-6686[c]"]
ACTUAL = "listeners__rna_synth_prob__actual_rna_synth_prob"
PPGPP_CONC = "listeners__growth_limits__ppgpp_conc"
TOP_N = 300


def inforce_vector(cache_dir, ppgpp_conc_umol):
    """The ppGpp-adjusted basal vector at a given ppGpp concentration."""
    import numpy as np
    from v2ecoli import build_composite
    from v2ecoli.processes.parca.wholecell.utils import units
    comp = build_composite("ecoli_baseline", cache_dir=str(cache_dir))
    step = comp.state["agents"]["0"]["ppgpp-initiation"]["instance"]
    out = step.synth_prob(ppgpp_conc_umol * units.umol / units.L, step.copy_number)
    v = out[0] if isinstance(out, tuple) else out
    return np.asarray(v, dtype=float)


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
    fixed = set()
    for k in ("idx_rprotein", "idx_rnap", "idx_rRNA", "idx_tRNA"):
        v = ti.get(k)
        if v is not None:
            fixed |= set(np.asarray(v).ravel().tolist())

    cidx = {t: ids.index(t) for t in COHORT if t in ids}
    top = list(np.argsort(bp)[::-1][:TOP_N])
    comp_idx = [i for i in top if i not in set(cidx.values())]
    comp_mrna = [i for i in comp_idx if is_mrna[i] and i not in fixed]

    act = np.zeros(len(ids)); conc_sum = 0.0; n = 0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            if ACTUAL in df.columns:
                for row in df[ACTUAL].to_list():
                    act[: len(row)] += np.asarray(row, dtype=float)
            if PPGPP_CONC in df.columns:
                conc_sum += float(df[PPGPP_CONC].sum())
            n += df.height
    if n == 0:
        return {"error": f"no data under {out_root}"}
    act /= n
    mean_conc = conc_sum / n

    ref = inforce_vector(cache_dir, mean_conc)

    med = lambda a, idx: float(np.median([a[i] for i in idx]))  # noqa: E731
    ci = list(cidx.values())

    # Reconstruction validation: does the recovered vector predict what the
    # well-behaved transcripts actually did?
    ref_share = ref / ref.sum()
    act_share = act / act.sum()
    pred = np.divide(act_share, ref_share, out=np.zeros_like(act),
                     where=ref_share > 0)
    comp_pred_median = med(pred, comp_mrna)

    ratio_ref = np.divide(act, ref, out=np.zeros_like(act), where=ref > 0)
    ratio_bp = np.divide(act, bp, out=np.zeros_like(act), where=bp > 0)

    # Additive-regulator prediction: what basal + delta_prob would give if the
    # configured regulators were applied additively. Under ppGpp the TF effect is
    # MULTIPLICATIVE, so this is the additive null and is reported as such.
    import scipy.sparse as sp_
    dp = ti["delta_prob"]
    M = sp_.csr_matrix((dp["deltaV"], (dp["deltaI"], dp["deltaJ"])), shape=dp["shape"])
    delta_sum = np.asarray(M.sum(axis=1)).ravel()
    floor = 1e-12
    pred_add = np.maximum(bp + delta_sum, floor)
    add_mismatch = float(np.median([abs(act[i] / pred_add[i] - 1.0) for i in ci]))

    # No TF bound (the additive path cannot fire) -- recomputed here as the pin.
    logdiff = np.abs(np.log((ref[top] + 1e-30) / (bp[top] + 1e-30)))
    return {
        "n_timesteps": n,
        "mean_ppgpp_conc_umol": mean_conc,
        "reference_vs_basal_median_logratio": float(np.median(logdiff)),
        "reference_sum": float(ref.sum()),
        "basal_sum": float(bp.sum()),
        "reconstruction_validation": comp_pred_median,
        "cohort_ratio_vs_reference": med(ratio_ref, ci),
        "comparison_ratio_vs_reference": med(ratio_ref, comp_mrna),
        "deficit_vs_reference": (med(ratio_ref, ci) / med(ratio_ref, comp_mrna)
                                 if med(ratio_ref, comp_mrna) > 0 else None),
        "cohort_ratio_vs_basal": med(ratio_bp, ci),
        "comparison_ratio_vs_basal": med(ratio_bp, comp_mrna),
        "deficit_vs_basal": (med(ratio_bp, ci) / med(ratio_bp, comp_mrna)
                             if med(ratio_bp, comp_mrna) > 0 else None),
        "additive_regulator_mismatch": add_mismatch,
        "delta_sum_over_basal": {t: float(delta_sum[i] / bp[i]) if bp[i] > 0 else None
                                 for t, i in cidx.items()},
        "n_comparison_mrna": len(comp_mrna),
        "per_tu": {t: {"actual": float(act[i]), "reference": float(ref[i]),
                       "basal": float(bp[i]),
                       "ratio_ref": float(ratio_ref[i]),
                       "ratio_basal": float(ratio_bp[i])} for t, i in cidx.items()},
    }
