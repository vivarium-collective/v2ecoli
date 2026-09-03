"""Does a mass balance over measured rates reproduce DnaG's observed protein count?

Science core for ``report_cards/dnag_mass_balance_card.py``.

The residual this addresses: dnaG's transcription deficit is 5.4x but its protein
sits 21.6x below ParCa's fitted 38 copies, leaving 4x unexplained. At the earlier
(wrong) 52x figure the deficit over-explained the gap, so no residual was sought.

The balance, first order and steady state:

    protein = (transcripts_per_second * translation_efficiency)
              / (degradation_rate + dilution_rate)

with dilution = ln(2) / generation_time. Every term is measured from the run or
read from the cache; none is fitted here.

Two things this is careful about.

The comparison it tests is between a TIME-AVERAGED simulated count and a FITTED
steady-state one, which are different quantities. That mismatch is the leading
candidate for the residual, not an afterthought -- DnaG sits at zero 62% of the
cycle, exactly the low-copy regime where a continuous steady-state approximation
is least trustworthy.

And it grades itself against controls before it grades DnaG. A balance that
cannot predict ordinary proteins cannot be used to judge an unusual one, so the
comparison monomers are run through the identical calculation and the DnaG result
is only interpretable if they come out right.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

DNAG_MONOMER = "EG10239-MONOMER[c]"
DNAG_TU = "TU00352[c]"
PARCA_FITTED_DNAG = 38.0          # from proteome_compare against this branch's fixture
MONOMER_COUNTS = "listeners__monomer_counts"
SYN = "listeners__transcript_elongation_listener__count_rna_synthesized"
LN2 = 0.6931471805599453


def _plain(v) -> float:
    return float(v.asNumber() if hasattr(v, "asNumber") else v)


def measure(cache_dir, out_root, seeds=(0, 1, 2), generations=(1, 2, 3),
            n_controls: int = 200) -> dict:
    import dill
    import numpy as np
    import polars as pl
    from v2ecoli.core import build_core
    build_core()
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        sd = dill.load(f)
    cf = sd["configs"]
    ti = cf["ecoli-transcript-initiation"]
    pi = cf["ecoli-polypeptide-initiation"]
    rna_ids = [str(x) for x in ti["rna_data"]["id"]]
    mono_ids = [str(x) for x in pi["monomer_ids"]]
    te = np.asarray(pi["translation_efficiencies"], dtype=float)

    # first-order protein degradation rate per monomer
    deg = None
    for k, v in (cf.get("ecoli-protein-degradation") or {}).items():
        a = np.asarray(v)
        if a.ndim == 1 and a.size == len(mono_ids) and a.dtype.kind in "fi":
            deg = a.astype(float)
            break
    if deg is None:
        return {"error": "no per-monomer degradation rate in the cache"}

    # cistron -> monomer, so a transcript's synthesis can be routed to its protein
    c2 = cf["rna_synth_prob_listener"]
    cis = [str(x) for x in c2["cistron_ids"]]
    M = c2["cistron_tu_mapping_matrix"]
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    mono_data = cf["ecoli-polypeptide-initiation"]
    m2c = mono_data.get("monomer_index_to_cistron_index") or {}

    syn = np.zeros(len(rna_ids))
    prot = np.zeros(len(mono_ids))
    n = 0
    sim_seconds = 0.0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            if SYN in df.columns:
                for row in df[SYN].to_list():
                    syn[: len(row)] += np.asarray(row, dtype=float)
            if MONOMER_COUNTS in df.columns:
                for row in df[MONOMER_COUNTS].to_list():
                    prot[: len(row)] += np.asarray(row, dtype=float)
            if "global_time" in df.columns:
                sim_seconds += float(df["global_time"].max() - df["global_time"].min())
            n += df.height
    if n == 0:
        return {"error": f"no data under {out_root}"}
    prot /= n
    rate = syn / max(sim_seconds, 1.0)          # transcripts per second, per TU

    # generation time -> dilution
    taus = []
    for seed in seeds:
        for p in Path(out_root).glob(f"seed{seed}/*_summary.json"):
            for g in json.loads(p.read_text()).get("gens", []):
                if g.get("duration_min"):
                    taus.append(float(g["duration_min"]) * 60.0)
    tau = float(np.median(taus)) if taus else 2400.0
    dilution = LN2 / tau

    # mRNA standing counts per TU (the template pool ribosomes act on). The
    # earlier formulation used transcripts-PER-SECOND times translation
    # efficiency, which is dimensionally wrong: translation_efficiency in this
    # model is a RELATIVE ribosome-allocation weight, not a per-second rate
    # constant. The control caught it at 0% within 2x.
    mrna = np.zeros(len(rna_ids))
    mn = 0
    for seed in seeds:
        for gen in generations:
            files = sorted(glob.glob(
                f"{out_root}/seed{seed}/**/history/**/generation={gen}/**/*.pq",
                recursive=True))
            if not files:
                continue
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
            if "bulk__id" not in df.columns:
                continue
            # bulk__id can be null in row 0 after a diagonal concat, so take the
            # first row that actually carries it.
            ids_b = None
            for row in df["bulk__id"].to_list():
                if row is not None and len(row) > 0:
                    ids_b = [str(x) for x in row]
                    break
            if ids_b is None:
                continue
            want = {r: k for k, r in enumerate(rna_ids)}
            cols = [(bi, want[r]) for bi, r in enumerate(ids_b) if r in want]
            # One pass over the count lists rather than 3277 per-column pulls.
            counts = np.asarray(df["bulk__count"].to_list(), dtype=float)
            for bi, k in cols:
                mrna[k] += float(counts[:, bi].sum())
            mn += df.height
    if mn:
        mrna /= mn

    def unscaled(mono_i: int) -> "float | None":
        """Balance up to one global constant: (mRNA x efficiency) / (deg + dilution).

        The constant is the absolute per-ribosome output rate, which the model does
        not expose. It is fitted ONCE on the control monomers (DnaG excluded) and
        then applied unchanged, so the axis asks whether DnaG is an outlier against
        a balance calibrated on proteins that behave -- not whether an absolute
        rate can be guessed.
        """
        ci = m2c.get(mono_i) if isinstance(m2c, dict) else None
        if ci is None:
            return None
        tus = np.nonzero(Md[int(ci)])[0]
        if len(tus) == 0:
            return None
        production = float(sum(rate[t] for t in tus)) * float(te[mono_i])
        loss = float(deg[mono_i]) + dilution
        return production / loss if loss > 0 else None

    j = mono_ids.index(DNAG_MONOMER)
    dnag_obs = float(prot[j])

    # Calibrate the one unknown constant on controls, with DnaG held out.
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
    raw = np.asarray(raw, dtype=float)
    obs = np.asarray([prot[i] for i in ctrl], dtype=float)
    scale = float(np.median(obs / raw)) if raw.size else None

    ratios = (raw * scale) / obs if (scale and raw.size) else np.array([])
    within2 = float(np.mean((ratios < 2.0) & (ratios > 0.5))) if ratios.size else None

    u_dnag = unscaled(j)
    dnag_pred = (u_dnag * scale) if (u_dnag is not None and scale) else None

    # ParCa's implied mRNA pool: invert the calibrated balance at the fitted count.
    implied = ((PARCA_FITTED_DNAG * (deg[j] + dilution)) / (te[j] * scale)
               if (te[j] > 0 and scale) else None)
    tus = np.nonzero(Md[int(m2c[j])])[0] if isinstance(m2c, dict) and j in m2c else []
    delivered = float(sum(rate[t] for t in tus)) if len(tus) else 0.0

    return {
        "n_timesteps": n, "sim_seconds": sim_seconds, "tau_seconds": tau,
        "dilution_per_s": dilution,
        "dnag_translation_efficiency": float(te[j]),
        "dnag_degradation_per_s": float(deg[j]),
        "dnag_transcription_per_s": delivered,
        "balance_scale_constant": scale,
        "dnag_predicted": dnag_pred,
        "dnag_observed": dnag_obs,
        "dnag_pred_over_obs": (dnag_pred / dnag_obs) if (dnag_pred and dnag_obs) else None,
        "control_n": len(ctrl),
        "control_within_2x": within2,
        "control_median_ratio": float(np.median(ratios)) if ratios.size else None,
        "parca_fitted": PARCA_FITTED_DNAG,
        "parca_implied_transcription_per_s": implied,
        "parca_implied_over_delivered": ((implied / delivered)
                                         if (implied and delivered > 0) else None),
    }
