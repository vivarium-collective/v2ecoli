"""Basal ``vs_literature`` report-card logic (reference-building + grading inputs).

The reusable core of the basal-condition ``vs_literature`` reference mode: grade
the blessed v2ecoli baseline against **curated experimental + theoretical
reference values** from the ecoli-sources ``validation_data`` bundle — the same
reference-agnostic grader (PR #134) pointed at literature instead of a pinned run.

This module holds everything the card + tests need — it is import-safe (no sweep,
no duckdb, no CLI): the measured (model-side) per-cell values are read from the
committed ``tests/fixtures/population_phenotype_basal/model_*.json`` fixtures, and
the reference (literature) side is read live from
``ecoli_sources.VALIDATION_BUNDLE_PATH``. The (heavy) bake path that recomputes
``model_physiology.json`` from a sweep parquet lives in
``scripts/render_basal_vs_literature.py`` (the thin CLI wrapper), which re-exports
these names for back-compat.

Physiology-first scope (μ, q_glc, Yxs), with Metabolism / Proteome / Composition /
Metabolite-pool sections layered on when their baked fixtures exist:

  * ``physiology.biomass_yield``  — the DIRECT mass-balance yield
    Yxs = ΔDW / ∫(q_glc·DW)dt (g dry weight made / g glucose consumed), per cell.
    Graded against the measured band AND the ``theoretical_max`` ceiling: a model
    value above the stoichiometric ceiling is a *differentiated first-principles
    failure*.
  * ``physiology.growth_rate``    — μ (1/h) vs the measured band.
  * ``physiology.glucose_uptake`` — q_glc (mmol/gDW/h) vs the measured band.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

M_GLC = 0.180156  # g/mmol glucose
M_C = 12.011      # g/mol carbon

REPO = Path(__file__).resolve().parents[2]
# Golden model fixtures: committed per-cell aggregates baked from a blessed sweep
# (organized sim output, not reference data) — carry a `provenance` block; see
# scripts/_provenance.py. Read here, re-baked by the bake paths in the CLI wrapper.
FIXTURES = REPO / "tests/fixtures/population_phenotype_basal"
_MODEL_JSON = FIXTURES / "model_physiology.json"   # baked per-cell model values (render --from-sweep)
_MET_JSON = FIXTURES / "model_metabolism.json"     # baked by scripts/bake_model_metabolism.py
_PRO_JSON = FIXTURES / "model_proteome.json"
_COMP_JSON = FIXTURES / "model_composition.json"
_POOLS_JSON = FIXTURES / "model_metabolite_pools.json"

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


def model_physiology(model_json: Path = _MODEL_JSON) -> dict:
    """Load the baked per-cell model physiology (committed; regenerate with
    ``render_basal_vs_literature.py --from-sweep``). Keeps the card + tests
    independent of the gitignored sweep."""
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
        "o2_uptake": f"Per glucose-C, ~{cmol['co2']:.0f}% is recovered as CO₂.",
        "co2_evolution": (f"C-mol balance: {cmol['biomass']:.0f}% of glucose carbon → biomass, "
                          f"{cmol['co2']:.0f}% → CO₂, {cmol['acetate']:.0f}% → acetate "
                          f"(reference aerobic cell on glucose ~50% biomass). RQ = CER/OUR = "
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
                    "Graded vs Crown 2015 COMPLETE-MFA (MG1655, ¹³C PLE) by identity-R² over "
                    "the reaction vector; per-reaction model−reference values are listed in "
                    "the companion table (with sign flips ⤺ where the model runs a reaction "
                    "reverse to Crown's direction). Methodological caveats: glucose entry "
                    "(Crown's PTS) is omitted — the model enters glucose mostly via "
                    "glucokinase + pyruvate kinase, ¹³C-indistinguishable from PTS, so not a "
                    "gradeable difference. † Pyk: its flux is coupled to that entry choice, "
                    "shown but not independently resolvable. ‡ Pfk/Fba: the model routes most "
                    "hexose→triose through fructose-6-P / sedoheptulose-bisP aldolases (an FBA "
                    "carbon-rearrangement degeneracy — node balances are unaffected), so the "
                    "single-reaction Pfk/Fba flux is not a glycolytic-throughput proxy."),
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
                "biomass drain, expected < 5%."),
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
                    "shunt (isocitrate lyase), as fractions of isocitrate flux. Graded "
                    "by total-variation distance vs Crown 2015."),
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
                    "overflow (Pta/AckA), or biosynthesis (fatty acids + amino acids), "
                    "as fractions of acetyl-CoA flux. Graded by total-variation distance "
                    "vs Crown 2015."),
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


def composition_reference(model_td_min: float, bundle_path: Path | None = None) -> dict:
    """Bremer & Dennis protein/RNA/DNA dry-mass fractions interpolated to the
    model's doubling time, plus the implied ``other`` (= 1 − Σ). The reference is
    a growth-rate series; we grade against the matched growth rate."""
    root = _bundle_path(bundle_path).parent
    df = pd.read_csv(root / "data/basal/macromolecular_composition.tsv", sep="\t", comment="#")
    bd = df[df["source_id"] == "bremer_dennis_2008"]

    def _interp(td: list[float], val: list[float], x: float) -> float:
        pts = sorted(zip(td, val))
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= x <= t1:
                return v0 + (v1 - v0) * (x - t0) / (t1 - t0)
        return pts[-1][1]

    out = {}
    for comp in ("protein", "rna", "dna"):
        sub = bd[bd["component"] == comp]
        out[comp] = _interp(list(sub["doubling_time_min"]), list(sub["value"]), model_td_min)
    out["other"] = 1.0 - out["protein"] - out["rna"] - out["dna"]
    return out


def build_composition(comp: dict, bundle_path: Path | None = None) -> tuple[dict, dict]:
    """(axes, card_node) for the Composition section: model macromolecular
    dry-mass fractions (protein/RNA/DNA/other) vs the Bremer & Dennis growth-rate
    composition at the matched doubling time — an INDEPENDENT reference (the model
    is not fit to B&D), graded as a composition (total-variation distance)."""
    td = comp["doubling_time_min"]
    ref = composition_reference(td, bundle_path)
    fr = comp["fractions"]
    axes = {"composition.macromolecular": {
        "group": "Composition",
        "label": "Macromolecular composition (% dry weight)", "units": "",
        "how": (f"Model: protein / RNA (total RNA: rRNA+tRNA+mRNA) / DNA as fractions "
                f"of dry weight (time-mean within cell, then across post-burn-in cells); "
                f"'other' is the dry-weight remainder. Graded by total-variation distance "
                f"vs Bremer &amp; Dennis 2008 (E. coli B/r) at the model's doubling time "
                f"({td:.0f} min, interpolated on their growth-rate curve — composition is "
                f"growth-rate-dependent). 'other' is each side's remainder over DIFFERENT "
                f"unmeasured components (B&amp;D: lipid/murein/LPS/etc, which B&amp;D did "
                f"not itemize; the model: its lumped small-molecule mass), so the 'other' "
                f"segments are not a like-for-like species comparison. Strain caveat: "
                f"B&amp;D is B/r, the model MG1655."),
        "plot": "split",
        "criterion": {"type": "composition", "ref_fractions": ref,
                      "ref_label": "Bremer & Dennis 2008", "xlabel": "fraction of dry weight",
                      "tv_good": 0.05, "tv_warn": 0.15, "residual_max": 0.02,
                      "residual_warn": 0.05},
    }}
    card = {"macromolecular": {"branches": {b: fr[b] for b in comp["branches"]},
                               "influx": 1.0}}
    return axes, card


def bennett_pools(bundle_path: Path | None = None) -> dict:
    """The Bennett 2009 glucose per-metabolite concentrations (mol/L) + 95% CI
    bounds, keyed by the model EcoCyc id carried in the slot's ``ecocyc_id``
    column (populated for the 93 metabolites that have a single model id)."""
    root = _bundle_path(bundle_path).parent
    df = pd.read_csv(root / "data/basal/metabolite_pools.tsv", sep="\t", comment="#")
    df = df[(df["condition"] == "glucose") & df["ecocyc_id"].notna()]
    return {str(r.ecocyc_id): {"value": float(r.value), "name": str(r.metabolite),
                               "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)}
            for r in df.itertuples()}


def build_metabolite_pools(pools: dict, bundle_path: Path | None = None) -> tuple[dict, dict]:
    """(axes, card_node) for the Metabolite pools section: the per-metabolite
    scatter (model realized concentration vs the genuine Bennett 2009 value + 95%
    CI, log-log). The model is calibrated to these per-metabolite targets, so this
    is a fit-to consistency check, not independent validation."""
    axes: dict = {}
    card: dict = {}

    # Per-metabolite scatter: model realized concentration vs the genuine Bennett
    # 2009 published value + 95% CI (joined by the slot's ecocyc_id), log-log,
    # graded by Pearson r (like the proteome axis).
    per = pools.get("per_metabolite", {})
    ref = bennett_pools(bundle_path)
    ids = sorted(set(per) & set(ref))
    if ids:
        names = [ref[i]["name"] for i in ids]
        model_vec = [per[i]["model"] for i in ids]
        ref_vec = [ref[i]["value"] for i in ids]
        ref_err = [[max(ref[i]["value"] - ref[i]["ci_low"], 0.0) for i in ids],
                   [max(ref[i]["ci_high"] - ref[i]["value"], 0.0) for i in ids]]
        axes["pools.per_metabolite"] = {
            "group": "Metabolite pools",
            "label": "Per-metabolite concentration (mol/L)", "units": "mol/L",
            "plot": "loglog",
            "how": (f"Model: ensemble realized intracellular concentration per "
                    f"metabolite (bulk count / cell volume / Nₐ; per-cell time-mean "
                    f"then ensemble). Graded vs the Bennett 2009 absolute "
                    f"concentrations (with 95% CI as horizontal error bars) over the "
                    f"{len(ids)} metabolites mapped to a single model id, by log-log "
                    f"Pearson r. The model is FIT to these targets — a fit-to "
                    f"consistency check, NOT independent validation; the companion "
                    f"table lists the metabolites that diverge most. Strain: Bennett "
                    f"K-12 NCM3722, glucose minimal, filter culture (td 77 min)."),
            "criterion": {"type": "pearson", "ref_vector": ref_vec, "ref_err": ref_err,
                          "r_min": 0.9, "r_drift": 0.7,
                          "ids": names, "entity_label": "metabolite",
                          "count_label": "concentration", "min_count": 0.0,
                          "outlier_log2fc": 2.0,
                          "up_label": "higher in", "down_label": "lower in"},
        }
        card["per_metabolite"] = {"vector": model_vec}
    return axes, card


# The basal behavioral spec — the (deliberately partial) set of basal-condition
# behaviors a glucose-minimal MG1655 cell exhibits that a faithful whole-cell
# model should reproduce. This card POPULATES the `covered` rows; the `planned`
# rows are named-but-not-yet-built — the spec is meant to grow, one report-card-
# gated axis at a time. Adding a category is a row here + its axis; the panel is
# the standing invitation ("what else would convince you it's being E. coli?").
# Kept intentionally lean: one worked axis per covered category, not exhaustive.
#
# Two KINDS of reference, worth keeping distinct (the "kind" field):
#   "independent" — the model is NOT fit to it; matching is genuine validation
#                   (Crown fluxes, Schmidt proteome, the ¹³C yields).
#   "fit-to"      — a ParCa INPUT the model is calibrated to; matching is a
#                   round-trip consistency check, not independent validation.
# Each row: (category, status, kind, detail). status ∈ covered|planned.
BASAL_SPEC = [
    ("Growth & physiology", "covered", "independent",
     "μ · biomass yield · glucose uptake — vs Bremer-Dennis + ¹³C-MFA studies"),
    ("Metabolic fluxes & exchanges", "covered", "independent",
     "central-carbon flux map · O₂/CO₂/acetate · branch-point splits — vs Crown 2015 / Toya 2010"),
    ("Proteome census", "covered", "independent",
     "protein copies/cell across the proteome — vs Schmidt 2016 MG1655"),
    ("Macromolecular composition", "covered", "independent",
     "protein / RNA / DNA fractions of dry weight — vs Bremer &amp; Dennis 2008 "
     "(EcoSal Plus, doi:10.1128/ecosal.5.2.3), at the model's growth rate"),
    ("Cell size, mass & geometry", "planned", "independent",
     "cell mass · volume · length · surface-to-volume — vs BioNumbers / Volkmer-Heinemann 2011"),
    ("Metabolite pools", "covered", "fit-to",
     "absolute intracellular metabolite concentrations — vs Bennett 2009 (per-metabolite "
     "scatter; the model is calibrated to these targets — a fit-to consistency check)."),
]


# Physically plausible range for E. coli biomass carbon content (gC/gDW). A cell's
# dry-weight carbon fraction sits ~0.45–0.50; the ~0.40–0.55 band is a generous
# sanity window. The "carbon is conserved" narrative is only asserted when the
# implied biomass carbon actually falls inside this window (issue #422: the claim
# used to be emitted with no comparison against any range).
_PLAUSIBLE_BIOMASS_C = (0.40, 0.55)


def _biomass_yield_finding(model: dict, axis: dict | None, card_node: dict | None) -> str:
    """Narrate the biomass-yield finding from the ACTUAL graded verdict, not fixed
    prose. Grades the same axis the report card grades (``grade_axis`` on the
    literature criterion) so the sentence a human reads follows the machine
    verdict: only a genuine ceiling crossing is called a "first-principles
    violation"; a within-band / under-ceiling model is narrated as compliant.

    Fixes issue #422, where the finding unconditionally asserted a violation and
    an untested "carbon is conserved" claim regardless of the yield."""
    from v2ecoli.library.card_criteria import grade_axis

    iC = model.get("implied_biomass_C_gC_per_gDW")
    if axis is None or card_node is None:
        # No gradeable axis (e.g. missing literature spec) — state only what we know.
        return ("Biomass yield could not be graded against the literature band "
                "(no reference criterion available).")

    crit = axis.get("criterion", {})
    grade = grade_axis(card_node, crit)
    verdict = grade.get("verdict", "ungraded")
    detail = grade.get("detail", {})
    violates = bool(detail.get("first_principles_violation"))
    got = model.get("biomass_yield")
    meas = crit.get("measured") or []
    lo, hi = (min(meas), max(meas)) if meas else (None, None)
    tmax = crit.get("theoretical_max")

    # The implied-biomass-carbon / conservation clause: assert conservation ONLY
    # when the implied carbon is inside the physically plausible window.
    carbon = ""
    if iC is not None:
        plo, phi = _PLAUSIBLE_BIOMASS_C
        if plo <= iC <= phi:
            carbon = (f" The implied biomass carbon (≈ {iC:.2f} gC/gDW) sits within "
                      f"the physically plausible range ({plo:.2f}–{phi:.2f} gC/gDW), "
                      "consistent with carbon conservation.")
        else:
            carbon = (f" The implied biomass carbon (≈ {iC:.2f} gC/gDW) falls OUTSIDE "
                      f"the physically plausible range ({plo:.2f}–{phi:.2f} gC/gDW) — "
                      "carbon is not conserved.")

    got_s = f"{got:.3g}" if got is not None else "n/a"
    max_s = f"{tmax:.3g}" if tmax is not None else "n/a"
    band_s = f"{lo:.3g}–{hi:.3g}" if lo is not None else "the measured band"

    if violates:
        return ("The yield "
                f"({got_s}) EXCEEDS the theoretical_max stoichiometric ceiling "
                f"({max_s}) — a first-principles violation (no carbon- and "
                "energy-balanced cell can exceed it)." + carbon)
    if verdict == "within_tol":
        return ("The yield "
                f"({got_s}) sits within the measured band ({band_s}) and below the "
                f"theoretical_max stoichiometric ceiling ({max_s}) — no "
                "first-principles violation." + carbon)
    # Under the ceiling but off the measured band (drift / mismatch).
    off = "drifts from" if verdict == "drift" else "falls outside"
    return ("The yield "
            f"({got_s}) is below the theoretical_max stoichiometric ceiling "
            f"({max_s}) — no first-principles violation — but {off} the measured "
            f"band ({band_s}).") + carbon


def build(model_json: Path = _MODEL_JSON, bundle_path: Path | None = None,
          fixtures_dir: Path | None = None) -> tuple[dict, dict, dict]:
    """Return (card, reference, model) — the gradeable inputs (importable for tests).

    All five model fixtures are read from ``fixtures_dir``, which defaults to
    ``model_json``'s own directory. Previously only ``model_json`` (physiology)
    was redirectable and the other four were read from the module-level
    ``FIXTURES`` constant, so pointing a caller at another directory silently
    mixed one redirected fixture with four repo copies."""
    fdir = Path(fixtures_dir) if fixtures_dir is not None else Path(model_json).parent
    met_json, pro_json = fdir / _MET_JSON.name, fdir / _PRO_JSON.name
    comp_json, pools_json = fdir / _COMP_JSON.name, fdir / _POOLS_JSON.name
    model = model_physiology(model_json)
    lit = literature(bundle_path)
    # Build the gradeable inputs FIRST so the findings narrative can follow the
    # actual per-axis verdict (issue #422) instead of hardcoded prose.
    axes = build_reference(lit, model)
    card = build_card(model)
    biomass_finding = _biomass_yield_finding(
        model, axes.get("physiology.biomass_yield"),
        card.get("physiology", {}).get("biomass_yield"))
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
            biomass_finding,
        ],
        "footer": "Behavioral report card (PR #134 grader) · vs_literature mode · "
                  "physiology · metabolism · proteome.",
        "coverage": {
            "lede": ("A <b>basal behavioral spec</b> for glucose-minimal MG1655 — "
                     "the behaviors a faithful whole-cell model should reproduce at "
                     "rest. This is a <b>starting point, not a complete card</b>: it "
                     "populates the categories below as they are built, one "
                     "report-card-gated axis at a time."),
            "invite": ("What else would convince you the model is being E. coli? "
                       "Missing categories are an open invitation — propose one "
                       "(a behavior + a reference) and it becomes the next row."),
            "rows": BASAL_SPEC,
        },
        "axes": axes,
    }

    # Metabolism + Proteome sections (present only if their fixtures are baked).
    if met_json.exists():
        met = json.load(open(met_json, encoding="utf-8"))
        ax, cnode = build_metabolism(lit, met, bundle_path)
        reference["axes"].update(ax)
        card["metabolism"] = cnode
        e = met["exchanges"]
        reference["findings"].append(
            "Metabolism axes (verdicts): the G6P split matches Crown's EMP/oxPPP/ED "
            f"composition (within_tol); the O₂ uptake ({e['absolute']['o2']:.2g} vs "
            f"measured ~11–15), CO₂ evolution, and acetate secretion exchanges fall "
            f"below the measured bands (mismatch). RQ = CER/OUR = {(e['rq'] or 0):.1f} "
            f"(full oxidation ≈ 1); C-mol balance {e['cmol_pct']['biomass']:.0f}% of "
            "glucose carbon → biomass (reference aerobic cell ~50%).")
    if pro_json.exists():
        pro = json.load(open(pro_json, encoding="utf-8"))
        ax, cnode = build_proteome(pro, bundle_path)
        reference["axes"].update(ax)
        card["proteome"] = cnode
    if comp_json.exists():
        comp = json.load(open(comp_json, encoding="utf-8"))
        ax, cnode = build_composition(comp, bundle_path)
        reference["axes"].update(ax)
        card["composition"] = cnode
    if pools_json.exists():
        pools = json.load(open(pools_json, encoding="utf-8"))
        ax, cnode = build_metabolite_pools(pools, bundle_path)
        reference["axes"].update(ax)
        card["pools"] = cnode

    return card, reference, model
