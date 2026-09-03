"""Is the ParCa-vs-runtime transcription disagreement dnaG-specific or global?

Science core for ``report_cards/fit_runtime_card.py``.

dnag-protein-mass-balance found ParCa's fitted DnaG implies 6.85x more
transcription than the run delivers. This asks whether that is peculiar to dnaG
or a property of the whole transcriptome, and whether it equals the ratio between
the two probability vectors -- basal_prob, which the fit uses, and the
ppGpp-adjusted replacement, which the runtime uses.

The implied rate comes from inverting the SAME mass balance that
dnag-protein-mass-balance validated at 74.5% within a factor of 2 on 200 held-out
controls, with its one free constant re-fitted here. Every quantity below depends
on that balance, so the control is re-measured rather than assumed: in the
predecessor an earlier, dimensionally-wrong balance scored 0% on it, which is the
only reason the error surfaced.
"""
from __future__ import annotations

import glob
from pathlib import Path

DNAG_MONOMER = "EG10239-MONOMER[c]"
MONOMER_COUNTS = "listeners__monomer_counts"
SYN = "listeners__transcript_elongation_listener__count_rna_synthesized"
PPGPP_CONC = "listeners__growth_limits__ppgpp_conc"
LN2 = 0.6931471805599453


def measure(cache_dir, out_root, seeds=(0, 1, 2), generations=(1, 2, 3),
            n_controls: int = 200,
            fixture: str = "models/parca/parca_state.pkl.gz",
            proteome_script: str = ("/Users/rashmidissasekara/Documents/code/"
                                    "v2ecoli/scripts/proteome_compare.py")) -> dict:
    import dill
    import json
    import numpy as np
    import polars as pl
    from v2ecoli.core import build_core
    from v2ecoli.library.inforce_reference import inforce_vector
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        sd = dill.load(f)
    cf = sd["configs"]
    ti = cf["ecoli-transcript-initiation"]
    pi = cf["ecoli-polypeptide-initiation"]
    rna_ids = [str(x) for x in ti["rna_data"]["id"]]
    bp = np.asarray(ti["basal_prob"], dtype=float)
    mono_ids = [str(x) for x in pi["monomer_ids"]]
    te = np.asarray(pi["translation_efficiencies"], dtype=float)
    m2c = pi.get("monomer_index_to_cistron_index") or {}
    deg = None
    for k, v in (cf.get("ecoli-protein-degradation") or {}).items():
        a = np.asarray(v)
        if a.ndim == 1 and a.size == len(mono_ids) and a.dtype.kind in "fi":
            deg = a.astype(float); break
    c2 = cf["rna_synth_prob_listener"]
    cis = [str(x) for x in c2["cistron_ids"]]
    M = c2["cistron_tu_mapping_matrix"]
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M)

    syn = np.zeros(len(rna_ids)); prot = np.zeros(len(mono_ids))
    n = 0; secs = 0.0; conc_sum = 0.0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            for row in df[SYN].to_list():
                syn[: len(row)] += np.asarray(row, dtype=float)
            for row in df[MONOMER_COUNTS].to_list():
                prot[: len(row)] += np.asarray(row, dtype=float)
            if PPGPP_CONC in df.columns:
                conc_sum += float(df[PPGPP_CONC].sum())
            secs += float(df["global_time"].max() - df["global_time"].min())
            n += df.height
    prot /= n
    rate = syn / max(secs, 1.0)
    ref = inforce_vector(cache_dir, conc_sum / n)

    taus = []
    for seed in seeds:
        for p in Path(out_root).glob(f"seed{seed}/*_summary.json"):
            for g in json.loads(p.read_text()).get("gens", []):
                if g.get("duration_min"):
                    taus.append(float(g["duration_min"]) * 60.0)
    dilution = LN2 / (float(np.median(taus)) if taus else 2400.0)

    def tus_of(mi):
        ci = m2c.get(mi) if isinstance(m2c, dict) else None
        return np.nonzero(Md[int(ci)])[0] if ci is not None else []

    def unscaled(mi):
        t = tus_of(mi)
        if len(t) == 0:
            return None
        loss = deg[mi] + dilution
        return (float(sum(rate[x] for x in t)) * float(te[mi]) / loss) if loss > 0 else None

    j = mono_ids.index(DNAG_MONOMER)
    order = np.argsort(prot)[::-1]
    ctrl, raw = [], []
    for i in order:
        if i == j or prot[i] <= 0:
            continue
        u = unscaled(int(i))
        if u is None or u <= 0:
            continue
        ctrl.append(int(i)); raw.append(u)
        if len(ctrl) >= n_controls:
            break
    raw = np.asarray(raw); obs = np.asarray([prot[i] for i in ctrl])
    scale = float(np.median(obs / raw))
    ratios = (raw * scale) / obs
    within2 = float(np.mean((ratios < 2.0) & (ratios > 0.5)))

    # ParCa's FITTED monomer counts. These are NOT stored on monomer_data (which
    # carries only id/cistron_id/deg_rate/length/aa_counts/mw); ParCa derives them
    # from expression fractions and total protein mass. proteome_compare.load_model
    # already implements that derivation and is reused rather than reimplemented.
    #
    # Using the simulation's observed protein here instead would merely re-derive
    # the mass balance's own residual (for dnaG, 1/0.694 = 1.44) rather than the
    # fit-runtime disagreement -- the bug the first version of this module had.
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_pc", str(proteome_script))
    _pc = _ilu.module_from_spec(spec); spec.loader.exec_module(_pc)
    counts_by_mono, _mw, _sym, _tot = _pc.load_model(str(fixture))
    fitted = np.array([float(counts_by_mono.get(mid, 0.0)) for mid in mono_ids])

    def implied_rate(mi):
        """Transcription rate ParCa's FITTED protein count implies, via the balance."""
        if fitted[mi] <= 0:
            return None
        loss = deg[mi] + dilution
        return (fitted[mi] * loss) / (te[mi] * scale) if (te[mi] > 0 and scale) else None

    # Population: implied vs delivered, and the vector ratio, per monomer.
    disc, vecr, keep = [], [], []
    for i in range(len(mono_ids)):
        t = tus_of(i)
        if len(t) == 0 or prot[i] <= 0 or te[i] <= 0:
            continue
        d = float(sum(rate[x] for x in t))
        b = float(sum(bp[x] for x in t)); r = float(sum(ref[x] for x in t))
        im = implied_rate(i)
        if d <= 0 or im is None or im <= 0 or b <= 0 or r <= 0:
            continue
        disc.append(im / d); vecr.append(b / r); keep.append(i)
    disc = np.asarray(disc); vecr = np.asarray(vecr)
    med = float(np.median(disc))

    jd = keep.index(j) if j in keep else None
    dnag_disc = float(disc[jd]) if jd is not None else None
    dnag_vecr = float(vecr[jd]) if jd is not None else None
    corr = float(np.corrcoef(np.log(disc), np.log(vecr))[0, 1]) if disc.size > 2 else None
    sub = (dnag_disc / dnag_vecr) if (dnag_disc and dnag_vecr) else None
    return {
        "n_timesteps": n, "n_population": int(disc.size),
        "control_within_2x": within2, "control_n": len(ctrl),
        "dnag_discrepancy": dnag_disc,
        "population_median_discrepancy": med,
        "dnag_over_population": (dnag_disc / med) if (dnag_disc and med) else None,
        "dnag_vector_ratio_basal_over_ppgpp": dnag_vecr,
        "substitution_agreement": (max(sub, 1 / sub) if sub else None),
        "population_correlation": corr,
    }
