"""Measure where DnaG production is lost, and what it costs replication.

Science core for ``report_cards/dnag_deficit_card.py``. Same three-way split as
``replisome_arrest.py`` and ``gate_sufficiency.py``: measurement here, grading in
the card, rendering in the card's HTML.

The question. Study 2 established that DnaG is the resource limiting replication
initiation: it is the only pool short at the stall in all 8 seeds, and removing
it from the gate restores full survival. This asks the upstream question -- why
is there so little of it -- by walking the production chain and locating the step
where the shortfall appears.

Four quantities, deliberately kept separate because they answer different things:

``literature``
    Four datasets (Schmidt 2016, Soufi/Maeck 2015, Mori 2021, Li 2014) read from
    the user's local proteomics archive. The median matters more than any single
    set: Schmidt reads DnaG far lower than the other three, so a one-dataset
    comparison is misleading in a specific, reproducible direction.

``parca_expected``
    What this branch's own ParCa fixture expects at basal. This is the model's
    internal target, NOT a literature value, and it is the comparison that
    separates "miscalibrated against experiment" from "not reproducing its own
    parameterisation".

``simulated``
    Per-timestep counts from a real multi-generation lineage. The gate reads THIS
    number, not the fixture's. Reported per generation so a decline across the
    lineage is visible rather than averaged away.

``chain``
    Transcription probability, translation efficiency and degradation rate, each
    as a percentile against every other gene, so the step that is anomalous is
    identifiable rather than asserted. Percentiles, not raw values, because the
    raw units differ per step and cannot be compared to one another.

The dedup hypothesis this was expected to confirm -- that collapsing the rpsU
promoters into TU00352 starves dnaG of transcription -- is testable here and can
fail: if TU00352's basal probability is high, transcription is not the lossy step
and the hypothesis is wrong regardless of how appealing it is.
"""
from __future__ import annotations

import os
from pathlib import Path

DNAG_CISTRON = "EG10239"
DNAG_MONOMER = "EG10239-MONOMER[c]"
DNAG_TU = "TU00352[c]"
# The gate's demand on DnaG: 2 copies per origin (monomer group).
MONOMER_MULT = 2


def _cache(cache_dir):
    import dill
    from v2ecoli.core import build_core
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        return dill.load(f)


def production_chain(cache_dir) -> dict:
    """Where dnaG sits in each production step, as a percentile across all genes."""
    import numpy as np
    sd = _cache(cache_dir)
    cfgs = sd["configs"]
    out: dict = {}

    ti = cfgs["ecoli-transcript-initiation"]
    ids = list(ti["rna_data"]["id"])
    bp = np.asarray(ti["basal_prob"], dtype=float)
    i = ids.index(DNAG_TU)
    out["transcription"] = {
        "step": "transcription initiation",
        "quantity": "basal synthesis probability",
        "value": float(bp[i]),
        "percentile": float(100 * (bp < bp[i]).mean()),
        "median_all": float(np.median(bp)),
        "n": int(bp.size),
    }

    pi = cfgs["ecoli-polypeptide-initiation"]
    te = np.asarray(pi["translation_efficiencies"], dtype=float)
    mids = [str(m) for m in pi["monomer_ids"]]
    j = next(k for k, m in enumerate(mids) if DNAG_CISTRON in m)
    out["translation"] = {
        "step": "translation initiation",
        "quantity": "translation efficiency",
        "value": float(te[j]),
        "percentile": float(100 * (te < te[j]).mean()),
        "median_all": float(np.median(te)),
        "n": int(te.size),
    }

    pdeg = cfgs.get("ecoli-protein-degradation", {})
    for key, v in pdeg.items():
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.size == len(mids) and arr.dtype.kind in "fi":
            a = arr.astype(float)
            out["degradation"] = {
                "step": "protein degradation",
                "quantity": key,
                "value": float(a[j]),
                "percentile": float(100 * (a < a[j]).mean()),
                "median_all": float(np.median(a)),
                "n": int(a.size),
            }
            break

    # Operon context: which cistrons share dnaG's only transcript.
    c = cfgs["rna_synth_prob_listener"]
    cis = [str(x) for x in c["cistron_ids"]]
    M = c["cistron_tu_mapping_matrix"]
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    ci = cis.index(DNAG_CISTRON)
    tus = np.nonzero(Md[ci])[0]
    out["operon"] = {
        "n_transcripts": int(len(tus)),
        "transcript_ids": [str(ids[t]) for t in tus],
        "cistrons_sharing": [cis[j2] for t in tus for j2 in np.nonzero(Md[:, t])[0]],
    }
    return out


# NOTE: the cache's ``initial_state`` is empty on this branch, so the model's
# own target is taken from the ParCa FIXTURE via proteome_compare's model column
# (its "model" figure is ParCa's fitted per-monomer count, pre-complexation).
# That is the number the simulation should reproduce and visibly does not.


def simulated(bundle_glob) -> dict:
    """Per-generation DnaG counts from distilled evidence bundles."""
    import glob
    import numpy as np
    import polars as pl
    files = sorted(glob.glob(str(bundle_glob)))
    if not files:
        return {}
    per_gen: dict[int, list[float]] = {}
    pooled: list[float] = []
    for f in files:
        df = pl.read_parquet(f)
        if "DnaG" not in df.columns:
            continue
        pooled += df["DnaG"].to_list()
        for g, sub in df.group_by("generation"):
            gi = int(g[0] if isinstance(g, tuple) else g)
            per_gen.setdefault(gi, []).extend(sub["DnaG"].to_list())
    if not pooled:
        return {}
    p = np.asarray(pooled, dtype=float)
    return {
        "n_files": len(files),
        "n_timesteps": int(p.size),
        "median": float(np.median(p)),
        "mean": float(p.mean()),
        "max": float(p.max()),
        "frac_zero": float((p == 0).mean()),
        "per_generation": {
            g: {"median": float(np.median(v)), "mean": float(np.mean(v)),
                "frac_zero": float(np.mean(np.asarray(v) == 0)), "n": len(v)}
            for g, v in sorted(per_gen.items())
        },
    }


def literature(script_path, data_root=None, schmidt=None,
               genes=("dnaG", "rpsU", "rpoD"), fixture=None) -> dict:
    """Per-dataset DnaG counts via the workspace's existing proteome_compare.py.

    Reuses that script rather than re-implementing four xlsx readers; it already
    knows each dataset's sheet and column conventions.
    """
    import re
    import subprocess
    import sys
    script_path = Path(script_path)
    if not script_path.is_file():
        return {"error": f"proteome_compare.py not found at {script_path}"}
    cmd = [sys.executable, str(script_path), *genes]
    if fixture:
        cmd += ["--fixture", str(fixture)]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    if schmidt:
        cmd += ["--schmidt", str(schmidt)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=script_path.parent.parent)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-500:]}
    def _num(x):
        return float(x.replace(",", ""))

    out: dict = {"per_gene": {}}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] in genes:
            try:
                out["per_gene"][parts[0]] = {
                    "model_fixture": _num(parts[1]), "Schmidt": _num(parts[2]),
                    "Soufi": _num(parts[3]), "Mori": _num(parts[4]),
                    "Li": _num(parts[5]), "median": _num(parts[6]),
                    "raw": line.strip(),
                }
            except (IndexError, ValueError):
                continue
    if "dnaG" not in out["per_gene"]:
        return {"error": "no dnaG row in proteome_compare output"}
    d = out["per_gene"]["dnaG"]
    out.update({k: d[k] for k in ("model_fixture", "Schmidt", "Soufi", "Mori",
                                  "Li", "median", "raw")})
    return out


def measure(cache_dir, bundle_glob, proteome_script,
            data_root=None, schmidt=None, fixture=None) -> dict:
    """Everything the card grades."""
    chain = production_chain(cache_dir)
    sim = simulated(bundle_glob)
    lit = literature(proteome_script, data_root, schmidt, fixture=fixture)
    exp = ({"count": lit.get("model_fixture"), "source": "ParCa fixture (proteome_compare model column)"}
           if lit.get("model_fixture") is not None else None)

    lit_med = lit.get("median")
    sim_med = sim.get("median")
    sim_mean = sim.get("mean")

    # Which step is the outlier? Lowest percentile across the chain steps.
    steps = {k: v for k, v in chain.items() if k in ("transcription", "translation",
                                                     "degradation")}
    bottleneck = min(steps.items(), key=lambda kv: kv[1]["percentile"])[0] if steps else None

    # The gate's demand vs what the lineage actually holds.
    demand = MONOMER_MULT  # per oriC; oriC >= 1 always
    return {
        "literature": lit,
        "parca_expected": exp,
        "simulated": sim,
        "chain": chain,
        "bottleneck_step": bottleneck,
        "bottleneck_percentile": steps[bottleneck]["percentile"] if bottleneck else None,
        "transcription_percentile": chain["transcription"]["percentile"],
        "translation_percentile": chain["translation"]["percentile"],
        "sim_vs_lit": (sim_mean / lit_med) if (sim_mean is not None and lit_med) else None,
        "sim_vs_parca": ((sim_mean / exp["count"]) if (sim_mean is not None and exp
                         and exp.get("count")) else None),
        "parca_vs_lit": ((exp["count"] / lit_med) if (exp and exp.get("count") and lit_med)
                         else None),
        "operon_ratios": {g: (v["model_fixture"] / v["median"] if v["median"] else None)
                          for g, v in (lit.get("per_gene") or {}).items()},
        "gate_demand_per_oric": demand,
        "frac_below_gate_demand": sim.get("frac_zero"),
    }
