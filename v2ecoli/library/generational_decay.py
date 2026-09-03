"""Per-generation transcription aggregates for the dnag-generational-decay study.

Every other helper in this library pools timesteps across a WINDOW of
generations and returns one average. That pooling is what produced the
cumulative figures (0.045 / 0.023 / 0.016) which motivated this study and which
cannot show the shape of a decay -- a cumulative mean falls whenever later
generations are lower, no matter how. This module keeps generations separate.

Comparison group is ``idx_rprotein`` MINUS dnaG itself, not the top-300 mRNA
group used elsewhere in this investigation: TU00352[c] is in idx_rprotein, and
that group explicitly removes the fixed-allocation classes, so it excludes
dnaG's own class-mates. See project-dnag-comparison-group-flaw.
"""
from __future__ import annotations

import glob
from pathlib import Path

DNAG_TU = "TU00352[c]"
ACTUAL = "listeners__rna_synth_prob__actual_rna_synth_prob"
PPGPP_CONC = "listeners__growth_limits__ppgpp_conc"


def indices(cache_dir):
    """(dnaG index, peer indices, all idx_rprotein indices) from the cache."""
    import dill, numpy as np
    from v2ecoli.core import build_core
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        sd = dill.load(f)
    ti = sd["configs"]["ecoli-transcript-initiation"]
    ids = [str(x) for x in ti["rna_data"]["id"]]
    dnag = ids.index(DNAG_TU)
    rprot = sorted(set(np.asarray(ti["idx_rprotein"]).ravel().tolist()))
    return dnag, [i for i in rprot if i != dnag], rprot


def per_generation(out_root, seeds=(0, 1, 2), generations=(1, 2, 3, 4),
                   cache_dir="out/cache") -> dict:
    """``{seed: {generation: {dnag, peer_median, class_total, ppgpp_conc, n}}}``.

    Each value is a mean over the timesteps of THAT generation only. A
    generation with no parquet is omitted rather than zero-filled, so a missing
    generation can never be read as a collapse to zero.
    """
    import numpy as np, polars as pl
    dnag, peers, rprot = indices(cache_dir)
    out: dict = {}
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            if ACTUAL not in df.columns:
                continue
            acc = None
            n = 0
            for row in df[ACTUAL].to_list():
                v = np.asarray(row, dtype=float)
                acc = v.copy() if acc is None else acc + v
                n += 1
            if not n:
                continue
            act = acc / n
            conc = (float(df[PPGPP_CONC].mean())
                    if PPGPP_CONC in df.columns else float("nan"))
            out.setdefault(seed, {})[gen] = {
                "dnag": float(act[dnag]),
                "peer_median": float(np.median([act[i] for i in peers])),
                "class_total": float(sum(act[i] for i in rprot)),
                "ppgpp_conc": conc,
                "n": n,
            }
    return out


def ratio(pg: dict, key: str, late: int, early: int):
    """Median across seeds of ``key`` at generation `late` over generation `early`.

    Seeds missing either generation are skipped; returns None if none remain.
    """
    import numpy as np
    vals = [pg[s][late][key] / pg[s][early][key]
            for s in pg
            if late in pg[s] and early in pg[s] and pg[s][early][key]]
    return float(np.median(vals)) if vals else None
