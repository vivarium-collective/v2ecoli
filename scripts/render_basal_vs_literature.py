"""Render the basal report card's ``vs_literature`` reference mode.

A third reference mode for ``population_phenotype_basal`` (alongside self-pin
*drift* and ``vs_vecoli`` *equivalence*): grade the blessed v2ecoli baseline
against **curated experimental + theoretical reference values** from the
ecoli-sources ``validation_data`` bundle — the same reference-agnostic grader
(PR #134) pointed at literature instead of a pinned run.

Physiology-first scope (μ, q_glc, Yxs):

  * ``physiology.biomass_yield``  — the DIRECT mass-balance yield
    Yxs = ΔDW / ∫(q_glc·DW)dt (g dry weight made / g glucose consumed), per cell.
    Graded against the measured band AND the ``theoretical_max`` ceiling: a model
    value above the stoichiometric ceiling is a *differentiated first-principles
    failure*.
  * ``physiology.growth_rate``    — μ (1/h) vs the measured band.
  * ``physiology.glucose_uptake`` — q_glc (mmol/gDW/h) vs the measured band.

The model (measured-side) per-cell values are computed by DIRECT mass balance
from the blessed-baseline sweep parquet (``--from-sweep``) and **baked into a
committed ``model_physiology.json``** so the card + tests stay independent of the
(gitignored) sweep. The literature (reference-side) values come from
``ecoli_sources.VALIDATION_BUNDLE_PATH`` read directly (the validation bundle has
none of the ParCa required keys, so the ParCa ``SourceBundle`` contract is bypassed).

Run:
    python scripts/render_basal_vs_literature.py                 # re-render from baked json
    python scripts/render_basal_vs_literature.py --from-sweep out/population_phenotype_basal
    # -> docs/report_cards/population_phenotype_basal/vs_literature/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from v2ecoli.library.report_card import grade_card, render_html, verdict_json  # noqa: E402

M_GLC = 0.180156  # g/mmol glucose
M_C = 12.011      # g/mol carbon

OUT = REPO / "docs/report_cards/population_phenotype_basal/vs_literature"
_MODEL_JSON = OUT / "model_physiology.json"        # baked per-cell model values (committed)
_MET_JSON = OUT / "model_metabolism.json"          # baked by scripts/bake_model_metabolism.py
_PRO_JSON = OUT / "model_proteome.json"
_SWEEP = REPO / "out/population_phenotype_basal"   # blessed ensemble (gitignored; --from-sweep)

# Metabolism exchange axes (scalar, graded like physiology against measured bands).
_EXCH_AXES = [
    ("metabolism.o2_uptake", "o2_uptake", "O₂ uptake (OUR)", "mmol/gDW/h"),
    ("metabolism.co2_evolution", "co2_evolution", "CO₂ evolution (CER)", "mmol/gDW/h"),
    ("metabolism.acetate_secretion", "acetate_secretion", "Acetate secretion", "mmol/gDW/h"),
]
# Crown 2015 G6P-fate branches -> their flux ids in metabolic_fluxes.tsv.
_G6P_CROWN = {"EMP": "crown_f2", "oxPPP": "crown_f10", "ED": "crown_f18"}
# Central-carbon flux scatter (model vs Crown 2015). The reaction set, per-reaction
# direction (signed, aligned to Crown's substrate->product), and annotation flags
# are resolved in the bake (model_metabolism.json["central_carbon"]["reactions"]),
# built by the stoichiometric reaction-set matcher; here we just pair each row to
# its Crown flux by crown_fid. Glucose entry (crown_f1) is intentionally omitted —
# PTS vs the model's glucokinase+PYK entry is ¹³C-non-identifiable, not a claim.
_INCLUDE_CC_SCATTER = True
# Per-flag glyph appended to the point label on the scatter (see the `how` text).
_CC_FLAG_GLYPH = {"pts_coupled": " †", "aldolase_bypass": " ‡",
                  "reductive_reverse": " ⤺"}

# (card path, bundle observable key, label, units, has-theoretical-max, model derivation)
_AXES = [
    ("physiology.growth_rate", "growth_rate", "Growth rate (μ)", "1/h", False,
     "Model: ensemble growth rate from per-cell doubling times (μ = ln 2 / doubling_time) "
     "of the blessed baseline ensemble (post-burn-in cells)."),
    ("physiology.biomass_yield", "biomass_yield", "Biomass yield (Yxs)", "gDW/g glucose", True,
     "Model: DIRECT mass-balance yield per cell = ΔDW / ∫(q_glc·DW)dt — grams dry weight "
     "made per gram glucose consumed (a mass ratio, robust to the steady-state assumption; "
     "the μ/(q_glc·M_glc) shortcut agrees at balanced growth but runs ~7% higher and noisier)."),
    ("physiology.glucose_uptake", "glucose_uptake", "Glucose uptake (q_glc)", "mmol/gDW/h", False,
     "Model: ensemble-mean specific glucose uptake = |GLC[p] exchange flux| over the "
     "post-burn-in cells."),
]


_GLC_IDX, _CO2_IDX, _ACET_IDX = 37, 11, 3  # 1-indexed positions in the 87-flux array


def physiology_from_sweep(sweep_dir: Path = _SWEEP, gen_lb: int = 3) -> dict:
    """Per-cell physiology by DIRECT mass balance from the sweep parquet.

    For each post-burn-in cell, integrated over its cycle from the per-timestep
    dry mass + specific exchange fluxes:
      * growth rate    μ      = ln 2 / cell-cycle time
      * glucose uptake q_glc  = time-mean |GLC[p] exchange|
      * biomass yield  Yxs    = ΔDW / (∫ q_glc·DW dt · M_glc)  — a direct mass
        ratio (g DW made / g glucose eaten), NOT the steady-state μ/(q·M) shortcut.
    Also reports the carbon implied into biomass per gDW
    (= (glucose-C − CO₂-C − acetate-C) / ΔDW) — a conservation sanity check;
    a physically plausible ~0.45–0.50 gC/gDW means carbon is conserved."""
    import duckdb
    files = glob.glob(os.path.join(str(sweep_dir), "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        raise SystemExit(f"no parquet under {sweep_dir} (need a blessed-baseline sweep)")
    flist = "[" + ",".join("'" + f + "'" for f in files) + "]"
    rows = duckdb.sql(f"""
      SELECT variant, lineage_seed, generation, agent_id, global_time,
        listeners__mass__dry_mass AS dw,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_GLC_IDX})  AS glc,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_CO2_IDX})  AS co2,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_ACET_IDX}) AS acet
      FROM read_parquet({flist}, hive_partitioning=true)
      WHERE generation >= {gen_lb}
      ORDER BY variant, lineage_seed, generation, agent_id, global_time""").fetchall()
    cells: dict = {}
    for r in rows:
        cells.setdefault(r[:4], []).append(r[4:])

    mu_c, q_c, y_c, impliedC = [], [], [], []
    for rs in cells.values():
        if len(rs) < 3:
            continue
        t = [x[0] / 3600.0 for x in rs]            # h
        dw = [x[1] * 1e-15 for x in rs]            # g (dry_mass is fg)
        glc = [abs(float(x[1 + 1] or 0)) for x in rs]   # |GLC| specific (mmol/gDW/h)
        co2 = [float(x[1 + 2] or 0) for x in rs]
        acet = [float(x[1 + 3] or 0) for x in rs]
        dts = [t[i + 1] - t[i] for i in range(len(t) - 1)]

        def integ(spec):  # trapezoid of (specific flux · DW) -> absolute mmol over the cycle
            rate = [spec[i] * dw[i] for i in range(len(spec))]
            return sum(0.5 * (rate[i] + rate[i + 1]) * dts[i] for i in range(len(dts)))

        glc_mmol, ddw = integ(glc), dw[-1] - dw[0]
        if glc_mmol <= 0 or ddw <= 0:
            continue
        mu_c.append(math.log(2) / (t[-1] - t[0]))
        q_c.append(sum(glc) / len(glc))
        y_c.append(ddw / (glc_mmol * M_GLC))
        c_into_biomass_mmol = glc_mmol * 6 - integ(co2) - 2 * integ(acet)  # mmol C
        impliedC.append(c_into_biomass_mmol * M_C / 1000.0 / ddw)          # gC/gDW

    n = len(y_c)
    return {
        "biomass_yield": sum(y_c) / n, "growth_rate": sum(mu_c) / n,
        "glucose_uptake": sum(q_c) / n,
        "biomass_yield_cells": y_c, "growth_rate_cells": mu_c, "glucose_uptake_cells": q_c,
        "implied_biomass_C_gC_per_gDW": sum(impliedC) / n,
        "method": "direct mass balance (ΔDW / ∫q_glc·DW dt)",
        "ensemble": f"blessed baseline, gen≥{gen_lb}", "n_cells": n,
    }


def model_physiology(model_json: Path = _MODEL_JSON) -> dict:
    """Load the baked per-cell model physiology (committed; regenerate with
    ``--from-sweep``). Keeps the card + tests independent of the gitignored sweep."""
    return json.load(open(model_json, encoding="utf-8"))


def literature(bundle_path: Path | None = None) -> dict:
    """Per-observable measured values + theoretical_max from the validation bundle.

    Read via the validation manifest (NOT the ParCa SourceBundle contract);
    returns ``{observable: {measured: [...], theoretical_max: x|None, sources:
    [...], strains: [...]}}``."""
    if bundle_path is None:
        from ecoli_sources import VALIDATION_BUNDLE_PATH
        bundle_path = VALIDATION_BUNDLE_PATH
    bundle_path = Path(bundle_path)
    manifest = pd.read_csv(bundle_path, sep="\t", comment="#")
    root = bundle_path.parent
    out: dict[str, dict] = {}
    for _, row in manifest.iterrows():
        key = str(row["canonical_key"])
        if not key.startswith("basal__"):
            continue
        obs = key.split("__", 1)[1]
        df = pd.read_csv(root / str(row["source_path"]), sep="\t", comment="#")
        meas = df[df["kind"] == "measured"]
        tmax = df[df["kind"] == "theoretical_max"]
        tmax_row = tmax.loc[tmax["value"].idxmin()] if len(tmax) else None
        out[obs] = {
            "measured": [float(v) for v in meas["value"]],
            "measured_unc": [(float(u) if pd.notna(u) else None)
                             for u in meas.get("uncertainty", [None] * len(meas))],
            "theoretical_max": (float(tmax_row["value"]) if tmax_row is not None else None),
            "theoretical_source": (str(tmax_row["source_id"]) if tmax_row is not None else None),
            "sources": [str(s) for s in meas["source_id"]],
            "strains": [str(s) for s in meas.get("strain", [])],
        }
    return out


def build_reference(lit: dict, model: dict, *, tol_rel: float = 0.10) -> dict:
    """A vs_literature reference: one ``literature``-criterion axis per observable."""
    axes: dict[str, dict] = {}
    for path, obs, label, units, _has_max, deriv in _AXES:
        spec = lit.get(obs)
        if not spec or not spec["measured"]:
            continue
        lo, hi = min(spec["measured"]), max(spec["measured"])
        how = (f"{deriv} Graded against curated experimental measurements "
               f"(band {lo:.3g}–{hi:.3g} {units}; sources shown in the plot).")
        if spec["theoretical_max"] is not None:
            how += (f" The red dashed line is the theoretical-max stoichiometric ceiling "
                    f"({spec['theoretical_max']:.3g} {units}) — a model value above it is a "
                    "first-principles violation.")
        axes[path] = {
            "group": "Physiology",
            "label": label, "units": units, "how": how, "plot": "literature",
            "criterion": {
                "type": "literature",
                "measured": [round(v, 6) for v in spec["measured"]],
                "measured_unc": spec.get("measured_unc"),
                "theoretical_max": spec["theoretical_max"],
                "theoretical_source": spec.get("theoretical_source"),
                "tol_rel": tol_rel,
                "sources": spec["sources"],
            },
        }
    return axes


def build_card(model: dict) -> dict:
    """Measured (model-side) card: scalar means for grading + per-cell value lists
    for the sim violin. The direct mass-balance method gives yield a per-cell
    distribution too (the ratio-of-means did not)."""
    return {"physiology": {
        "biomass_yield": {"mean": model["biomass_yield"],
                          "values": model.get("biomass_yield_cells")},
        "growth_rate": {"mean": model["growth_rate"],
                        "values": model.get("growth_rate_cells")},
        "glucose_uptake": {"mean": model["glucose_uptake"],
                           "values": model.get("glucose_uptake_cells")},
    }}


def _bundle_path(bundle_path: Path | None) -> Path:
    if bundle_path is None:
        from ecoli_sources import VALIDATION_BUNDLE_PATH
        bundle_path = VALIDATION_BUNDLE_PATH
    return Path(bundle_path)


def crown_g6p_composition(bundle_path: Path | None = None) -> dict:
    """Crown 2015 G6P-fate composition (fractions of glucose uptake) from the
    metabolic_fluxes bundle — the reference for the glycolysis-split bars."""
    root = _bundle_path(bundle_path).parent
    df = pd.read_csv(root / "data/basal/metabolic_fluxes.tsv", sep="\t", comment="#")
    cr = df[df["source_id"] == "crown_2015"].set_index("reaction_id")
    return {k: float(cr.loc[fid, "value_relative_pct"]) for k, fid in _G6P_CROWN.items()}


def crown_fate_nodes(bundle_path: Path | None = None) -> dict:
    """Crown 2015 reference compositions for the isocitrate + AcCoA fate nodes.
    Isocitrate: ICDH (oxidative, f23) vs ICL (glyoxylate, f29). AcCoA: citrate
    synthase (TCA, f21) vs acetate overflow (f35) vs biosynthesis (the balance of
    PDH production f20 not going to TCA or acetate)."""
    root = _bundle_path(bundle_path).parent
    df = pd.read_csv(root / "data/basal/metabolic_fluxes.tsv", sep="\t", comment="#")
    f = df[df["source_id"] == "crown_2015"].set_index("reaction_id")["value_relative_pct"]
    return {
        "isocitrate": {"oxidative_TCA": float(f["crown_f23"]),
                       "glyoxylate": float(f["crown_f29"])},
        "accoa": {"TCA": float(f["crown_f21"]), "acetate": float(f["crown_f35"]),
                  "biosynthesis": float(f["crown_f20"] - f["crown_f21"] - f["crown_f35"])},
    }


def proteome_reference(bundle_path: Path | None = None) -> dict:
    """Schmidt MG1655 proteome as ``{gene: copies/cell}`` (the basal__proteome slot)."""
    root = _bundle_path(bundle_path).parent
    df = pd.read_csv(root / "data/basal/proteome.tsv", sep="\t", comment="#")
    return {str(r.gene): float(r.value) for r in df.itertuples()}


def build_metabolism(lit: dict, met: dict, bundle_path: Path | None = None) -> tuple[dict, dict]:
    """(axes, card_node) for the Metabolism section: O₂/CO₂/acetate exchanges, the
    central-carbon flux scatter, and the branch-point fate splits (glycolysis /
    isocitrate / acetyl-CoA), all as stacked model-vs-Crown composition bars."""
    g6p = met["nodes"]["g6p"]
    exch = met["exchanges"]
    cmol = exch["cmol_pct"]
    axes: dict[str, dict] = {}
    card: dict[str, dict] = {}

    # Section order (Chris): exchanges (the headline O₂/CO₂/acetate deficits) →
    # the full central-carbon scatter → the intracellular branch-point splits.

    # Exchange axes (absolute rates vs measured bands; C-mol context in `how`).
    cmol_note = {
        "o2_uptake": f"Per glucose-C the model oxidizes only ~{cmol['co2']:.0f}% to CO₂.",
        "co2_evolution": (f"C-mol balance: {cmol['biomass']:.0f}% of glucose carbon → biomass, "
                          f"{cmol['co2']:.0f}% → CO₂, {cmol['acetate']:.0f}% → acetate "
                          f"(a real cell on glucose ~50% biomass). RQ = CER/OUR = "
                          f"{(exch['rq'] or 0):.2f} (full oxidation ≈ 1)."),
        "acetate_secretion": "The overflow axis — measured rates are direct (HPLC), not GUR-derived.",
    }
    for path, obs, label, units in _EXCH_AXES:
        spec = lit.get(obs)
        if not spec or not spec["measured"]:
            continue
        lo, hi = min(spec["measured"]), max(spec["measured"])
        axes[path] = {
            "group": "Metabolism", "label": label, "units": units, "plot": "literature",
            "how": (f"Model: ensemble-mean specific {label.split('(')[0].strip().lower()} "
                    f"over post-burn-in cells. Graded vs curated measurements "
                    f"(band {lo:.3g}–{hi:.3g} {units}). " + cmol_note.get(obs, "")),
            "criterion": {
                "type": "literature",
                "measured": [round(v, 6) for v in spec["measured"]],
                "measured_unc": spec.get("measured_unc"),
                "theoretical_max": None, "tol_rel": 0.10,
                "sources": spec["sources"],
            },
        }
        key = obs  # card path tail
        card[key] = {"mean": exch["absolute"][{"o2_uptake": "o2", "co2_evolution": "co2",
                                               "acetate_secretion": "acetate"}[obs]],
                     "values": exch["per_cell"][{"o2_uptake": "o2", "co2_evolution": "co2",
                                                 "acetate_secretion": "acetate"}[obs]]}

    # Central-carbon flux scatter: model vs Crown 2015, SIGNED and normalized to
    # glucose = 100. Each row is a base reaction mapped by the stoichiometric
    # reaction-set matcher and aligned to Crown's substrate->product direction, so
    # a genuine reverse-runner plots in the wrong quadrant rather than being hidden
    # by a magnitude. Crown's fluxes are positive (their written direction).
    cc = met.get("central_carbon", {})
    if cc.get("reactions") and _INCLUDE_CC_SCATTER:
        root = _bundle_path(bundle_path).parent
        crdf = pd.read_csv(root / "data/basal/metabolic_fluxes.tsv", sep="\t", comment="#")
        cr = crdf[crdf["source_id"] == "crown_2015"].set_index("reaction_id")
        crv, cru = cr["value_relative_pct"], cr["uncertainty"]
        ids, mvec, rvec, mstd, rstd = [], [], [], [], []
        for r in cc["reactions"]:
            fid = r["crown_fid"]
            if fid not in crv.index:
                continue
            ids.append(r["label"] + _CC_FLAG_GLYPH.get(r["flag"], ""))
            mvec.append(float(r["model"]))            # signed
            mstd.append(float(r.get("model_std", 0.0)))            # cell-to-cell
            rvec.append(float(crv.loc[fid]))           # Crown, positive
            rstd.append(float(cru.loc[fid]) if pd.notna(cru.loc[fid]) else 0.0)  # Crown CI stdev
        axes["metabolism.central_carbon_flux"] = {
            "group": "Metabolism",
            "label": "Central-carbon fluxes (vs Crown 2015)", "units": "% of glucose uptake",
            "how": ("Model: ensemble central-carbon fluxes (signed, reaction-set-summed, "
                    "aligned to the reference direction), normalized to glucose uptake = 100. "
                    "Graded vs Crown 2015 COMPLETE-MFA (MG1655, ¹³C PLE). Glycolysis and "
                    "oxidative-PPP sit on the identity line — carbon ROUTING into central "
                    "metabolism is right. The divergences are the finding: PDH, αKG "
                    "dehydrogenase and malate synthase collapse to ~0 and acetate overflow "
                    "is absent (the oxidative TCA isn't turning — the reaction-level face of "
                    "the under-respiration), while the lower TCA runs REDUCTIVELY (Fum, MDH "
                    "negative — OAC→Mal→Fum) ⤺. Glucose entry (Crown's PTS) is omitted: the "
                    "model enters glucose mostly via glucokinase + pyruvate kinase, which is "
                    "¹³C-indistinguishable from PTS — not a gradeable difference. † Pyk: its "
                    "flux is coupled to that entry choice, shown but not independently "
                    "resolvable. ‡ Pfk/Fba: the model routes most hexose→triose through "
                    "fructose-6-P / sedoheptulose-bisP aldolases (an FBA carbon-rearrangement "
                    "deviation — node balances are unaffected), so these read low."),
            "plot": "flux_scatter",
            # qualitative=False: these are internal fluxes — a reaction at ~0 is a
            # low value, not a categorical on/off, so drop the appeared/lost
            # callouts (which read as a missing value and aren't more telling than
            # the sign flips). Graded on identity-R² over the whole vector.
            "criterion": {"type": "flux_scatter", "ref_vector": rvec, "ref_std": rstd,
                          "flux_ids": ids, "active_eps": 1e-6, "qualitative": False,
                          "r2_min": 0.8, "r2_drift": 0.5},
        }
        card["central_carbon_flux"] = {"vector": mvec, "std": mstd,
                                       "n_cells": cc.get("n_cells") or met.get("n_cells")}

    # Intracellular branch-point splits (last). Glycolysis-split composition,
    # graded by total-variation distance vs Crown.
    crown = crown_g6p_composition(bundle_path)
    axes["metabolism.glycolysis_split"] = {
        "group": "Metabolism",
        "label": "Glycolysis split (EMP / oxPPP / ED)", "units": "",
        "how": ("Model: G6P-fate composition from the ensemble base-reaction fluxes "
                "(EMP=phosphoglucose isomerase, oxPPP=6-phosphogluconate dehydrogenase, "
                "ED=KDPG aldolase), as fractions of glucose uptake. Graded by total-"
                "variation distance vs the Crown 2015 ¹³C-MFA composition (the routing "
                "of carbon through central metabolism, independent of uptake rate); the "
                "hatched residual is G6P unaccounted by the three branches — a small "
                "biomass drain, expected < 5%. The model routes carbon CORRECTLY here "
                "(low TV) — the defects are downstream (respiration / overflow)."),
        "plot": "split",
        "criterion": {
            "type": "composition", "ref_fractions": crown, "ref_label": "Crown 2015",
            "tv_good": 0.05, "tv_warn": 0.15, "residual_max": 0.05, "residual_warn": 0.10,
        },
    }
    card["glycolysis_split"] = {"branches": g6p["model_flux"], "influx": g6p["influx"]}

    # TCA branch-point fate nodes (isocitrate, AcCoA), graded as compositions vs
    # Crown — shown as stacked model-vs-Crown bars (2- and 3-way, not ternary).
    if "isocitrate" in met.get("nodes", {}) and "accoa" in met["nodes"]:
        cf = crown_fate_nodes(bundle_path)
        axes["metabolism.isocitrate_split"] = {
            "group": "Metabolism",
            "label": "Isocitrate fate (oxidative TCA / glyoxylate)", "units": "",
            "how": ("Model: the isocitrate branch point — oxidative decarboxylation "
                    "(isocitrate dehydrogenase → α-ketoglutarate) vs the glyoxylate "
                    "shunt (isocitrate lyase). Graded by total-variation distance vs "
                    "Crown 2015. The model routes ~96% oxidative (ICDH makes α-KG for "
                    "biosynthesis — note α-KG is NOT dehydrogenated onward, αKGDH ≈ 0)."),
            "plot": "split",
            "criterion": {"type": "composition", "ref_fractions": cf["isocitrate"],
                          "ref_label": "Crown 2015", "tv_good": 0.05, "tv_warn": 0.15,
                          "residual_max": 0.02, "residual_warn": 0.05},
        }
        card["isocitrate_split"] = {"branches": met["nodes"]["isocitrate"]["model_flux"],
                                    "influx": met["nodes"]["isocitrate"]["influx"]}
        axes["metabolism.accoa_split"] = {
            "group": "Metabolism",
            "label": "Acetyl-CoA fate (TCA / acetate / biosynthesis)", "units": "",
            "how": ("Model: where acetyl-CoA goes — citrate synthase (TCA), acetate "
                    "overflow (Pta/AckA), or biosynthesis (fatty acids + amino acids). "
                    "Graded by total-variation distance vs Crown 2015. The contrast is "
                    "the overflow defect: the model dumps ~79% of acetyl-CoA into "
                    "biosynthesis with **no acetate overflow**, where Crown overflows "
                    "~59% to acetate — the AcCoA-node face of the missing overflow."),
            "plot": "split",
            "criterion": {"type": "composition", "ref_fractions": cf["accoa"],
                          "ref_label": "Crown 2015", "tv_good": 0.05, "tv_warn": 0.15,
                          "residual_max": 0.05, "residual_warn": 0.10},
        }
        card["accoa_split"] = {"branches": met["nodes"]["accoa"]["model_flux"],
                               "influx": met["nodes"]["accoa"]["influx"]}
    return axes, card


def build_proteome(pro: dict, bundle_path: Path | None = None) -> tuple[dict, dict]:
    """(axes, card_node) for the Proteome section: model copies/cell vs Schmidt
    MG1655, graded by log-log Pearson r (concordance, scale-offset-robust)."""
    ref = proteome_reference(bundle_path)
    model = pro["by_symbol"]
    genes = sorted(g for g in ref if g in model)
    cand_vec = [model[g] for g in genes]
    ref_vec = [ref[g] for g in genes]
    axes = {"proteome.abundance": {
        "group": "Proteome",
        "label": "Protein abundance (copies/cell)", "units": "copies/cell",
        "how": ("Model: ensemble-mean protein copies/cell per gene (time-mean within "
                "cell, then across post-burn-in cells). Graded vs Schmidt 2016 MG1655 "
                "glucose-minimal proteome by log-log Pearson r over shared genes — the "
                "literature convention, robust to a systematic scale offset (unlike "
                "identity-R²)."),
        "plot": "loglog",
        "criterion": {"type": "pearson", "ref_vector": ref_vec,
                      "r_min": 0.9, "r_drift": 0.7},
    }}
    card = {"abundance": {"vector": cand_vec}}
    return axes, card


def build(model_json: Path = _MODEL_JSON, bundle_path: Path | None = None) -> tuple[dict, dict, dict]:
    """Return (card, reference, model) — the gradeable inputs (importable for tests)."""
    model = model_physiology(model_json)
    lit = literature(bundle_path)
    iC = model.get("implied_biomass_C_gC_per_gDW")
    reference = {
        "title": "Basal-condition physiology — v2ecoli vs experimental literature",
        "status": "populated",
        "stimulus": {
            "reference_model": "experimental literature (ecoli-sources validation_data)",
            "measured_model": "v2ecoli baseline",
            "ensemble": model["ensemble"],
        },
        "findings": [
            "vs_literature reference mode: the v2ecoli baseline graded against "
            "curated experimental + theoretical values, not a pinned run.",
            "Biomass yield is computed by DIRECT mass balance (ΔDW / ∫q_glc·DW dt) "
            "per cell — a true g-DW-made / g-glucose-eaten ratio, robust to the "
            "steady-state assumption (the μ/(q·M) shortcut ran ~7% higher).",
            "The yield exceeds the theoretical_max stoichiometric ceiling: an "
            "ENERGETIC first-principles violation (the model gets ATP without "
            f"respiring — implied biomass carbon ≈ {iC:.2f} gC/gDW is physically "
            "plausible, so carbon IS conserved; the model under-respires rather "
            "than creating mass)." if iC is not None else
            "The yield exceeds the theoretical_max stoichiometric ceiling — a "
            "first-principles violation.",
        ],
        "footer": "Behavioral report card (PR #134 grader) · vs_literature mode · "
                  "physiology · metabolism · proteome.",
        "axes": build_reference(lit, model),
    }
    card = build_card(model)

    # Metabolism + Proteome sections (present only if their fixtures are baked).
    if _MET_JSON.exists():
        met = json.load(open(_MET_JSON, encoding="utf-8"))
        ax, cnode = build_metabolism(lit, met, bundle_path)
        reference["axes"].update(ax)
        card["metabolism"] = cnode
        e = met["exchanges"]
        reference["findings"].append(
            "Metabolism splits into two independent reads: the model routes central "
            f"carbon CORRECTLY (G6P split ≈ Crown's EMP/oxPPP/ED) but UNDER-RESPIRES "
            f"(O₂ {e['absolute']['o2']:.2g} vs ~11–15, RQ {(e['rq'] or 0):.1f} vs ≈1.1) "
            f"with no overflow — {e['cmol_pct']['biomass']:.0f}% of glucose carbon is "
            "retained as biomass (vs ~50% in a real aerobic cell), the C-mol root of the "
            "inflated yield. The defect is energetic (respiration), not carbon routing.")
    if _PRO_JSON.exists():
        pro = json.load(open(_PRO_JSON, encoding="utf-8"))
        ax, cnode = build_proteome(pro, bundle_path)
        reference["axes"].update(ax)
        card["proteome"] = cnode

    return card, reference, model


def main(from_sweep: str | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if from_sweep:
        model = physiology_from_sweep(Path(from_sweep))
        with open(_MODEL_JSON, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        print(f"baked {_MODEL_JSON.name}: {model['n_cells']} cells · "
              f"Yxs {model['biomass_yield']:.3f} · μ {model['growth_rate']:.3f} · "
              f"q_glc {model['glucose_uptake']:.3f} · implied biomass C "
              f"{model['implied_biomass_C_gC_per_gDW']:.3f} gC/gDW")
    card, reference, model = build()
    generated = time.strftime("%Y-%m-%d %H:%M")
    model_ref = f"v2ecoli baseline ({model['ensemble']})"
    report = grade_card(card, reference)

    (OUT / "report_card.html").write_text(
        render_html(card, reference, model_ref=model_ref, generated=generated))
    with open(OUT / "literature_reference.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)
    with open(OUT / "report_card_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json(report, model_ref=model_ref,
                               reference_model=reference["stimulus"]["reference_model"],
                               generated=generated), f, indent=2, ensure_ascii=False)

    print(f"overall: {report['overall']}")
    for path, a in report["axes"].items():
        fp = " [first-principles violation]" if a.get("detail", {}).get(
            "first_principles_violation") else ""
        print(f"  {path:28} {a['verdict']:11} {a.get('meter','')}{fp}")
    print(f"\nwrote {OUT}/ (report_card.html + literature_reference.json + "
          "report_card_verdict.json)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-sweep", metavar="DIR", default=None,
                    help="recompute the baked model_physiology.json from a sweep parquet dir")
    a = ap.parse_args()
    main(from_sweep=a.from_sweep)
