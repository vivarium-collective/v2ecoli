"""Aerobic acetate-overflow report card — the science core.

Grades v2ecoli's overflow response — acetate-carbon yield vs growth rate —
against curated chemostat/batch references, using the reference-agnostic
``threshold_linear`` criterion pointed at a swept response.

Graded as a **dimensionless acetate-carbon yield** ``Y_ac = (2·acetate)/(6·glucose)``
vs **growth rate** (the common, unit-matched x for model + all references — per-OD
vs per-gDW cancels in a yield). Shape-first: the rise from ~0 to ~linear is the
graded behavior; the threshold position is reported but secondary (it shifts
strain-to-strain).

References (ecoli-sources ``perturbation__overflow__acetate_vs_growth`` slot):
  * **Vemuri 2006** (MG1655 chemostat) — PRIMARY; carries glucose uptake
    co-measured, so ``Y_ac`` is raw per point. Used as the graded reference curve.
  * **Basan 2015** (NCM3722 batch ptsG titration) — CONTEXT; ``Y_ac`` via the
    Supplementary-Note-1 closed form (see below). Overlaid, not graded.
  * **MG1655 ¹³C anchor** — Long/Crown/LaCroix basal acetate ÷ glucose uptake, a
    single independent yield point at the basal growth rate. Overlaid.

The model arm comes from the GUR-titration runner
(``workspace/studies/overflow-acetate-vs-growth/sims/run_gur_titration.py``): per
cap, the emergent growth rate + acetate + glucose exchange, baked to a committed
fixture so the card stays sweep-independent and grades run-free.

This module is import-safe — no CLI, no duckdb, no sweep access. The CLI wrapper
is ``scripts/render_overflow_card.py``; the Gen-2 Step is
``v2ecoli/workflow/report_cards/acetate_overflow_card.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests/fixtures/overflow_acetate_vs_growth"
# The baked model arm (the GUR-titration runner's output). Committed so the card
# + tests stay independent of the (gitignored) sweep.
_MODEL_JSON = FIXTURES / "model_overflow_baseline.json"

# Basan 2015 Supplementary Note 1 carbon-partition parameters (for the CONTEXT
# curve only; Vemuri carries glucose uptake so needs none). J_E = σ·λ,
# J_C,BM = β·λ, J_E,f = ef·J_C,f, Sac = 1/3 (ferment C in = 3·Jac). Then
# J_C,in = (β + σ/er)·λ + (3 − 3·ef/er)·Jac ; Y_ac = 2·Jac / J_C,in.
_BASAN = {"beta": 28.5, "sigma": 45.7, "ef": 2.0, "er": 4.4}


def _yc(acetate: float, glucose_c_per_h: float) -> float | None:
    """Acetate-carbon yield = 2·acetate / glucose-carbon. ``glucose_c_per_h`` is
    the glucose CARBON influx (= 6·glucose-uptake)."""
    return (2.0 * acetate / glucose_c_per_h) if glucose_c_per_h else None


def _bundle_path(bundle_path: Path | None) -> Path:
    if bundle_path is None:
        from ecoli_sources import VALIDATION_BUNDLE_PATH
        bundle_path = VALIDATION_BUNDLE_PATH
    return Path(bundle_path)


def _overflow_slot(bundle_path: Path | None) -> pd.DataFrame:
    root = _bundle_path(bundle_path).parent
    return pd.read_csv(root / "data/perturbation/overflow/acetate_vs_growth.tsv",
                       sep="\t", comment="#")


def vemuri_curve(bundle_path: Path | None = None) -> dict:
    """PRIMARY graded reference: (growth rate, Y_ac) for the Vemuri chemostat
    series, Y_ac raw from co-measured glucose uptake. Sorted by growth rate;
    carries asymmetric Y_ac error bars from the slot's acetate CIs."""
    df = _overflow_slot(bundle_path)
    v = df[df["source_id"] == "vemuri_2006"].sort_values("growth_rate_per_h")
    x, y, elo, ehi = [], [], [], []
    for r in v.itertuples():
        gc = 6.0 * float(r.glucose_uptake)
        yac = _yc(float(r.value), gc)
        if yac is None:
            continue
        x.append(float(r.growth_rate_per_h)); y.append(yac)
        elo.append(max(yac - _yc(float(r.ci_low), gc), 0.0) if pd.notna(r.ci_low) else 0.0)
        ehi.append(max(_yc(float(r.ci_high), gc) - yac, 0.0) if pd.notna(r.ci_high) else 0.0)
    return {"x": x, "y": y, "err": [elo, ehi], "label": "Vemuri 2006 (MG1655 chemostat)"}


def basan_curve(bundle_path: Path | None = None) -> dict:
    """CONTEXT: (growth rate, Y_ac) for the Basan ptsG glucose titration, Y_ac via
    the SI carbon-partition closed form (no co-measured uptake)."""
    df = _overflow_slot(bundle_path)
    b = df[(df["source_id"] == "basan_2015") & (df["series"] == "ptsG_glucose")]
    b = b.sort_values("growth_rate_per_h")
    p = _BASAN
    a = p["beta"] + p["sigma"] / p["er"]              # λ coefficient (≈38.9)
    c = 3.0 - 3.0 * p["ef"] / p["er"]                 # Jac coefficient (≈1.64)
    x, y = [], []
    for r in b.itertuples():
        lam, jac = float(r.growth_rate_per_h), float(r.value)
        jcin = a * lam + c * jac
        if jcin > 0:
            x.append(lam); y.append(2.0 * jac / jcin)
    return {"x": x, "y": y, "label": "Basan 2015 (NCM3722 batch, SI-derived)"}


def mg1655_anchor(bundle_path: Path | None = None) -> dict:
    """CONTEXT: independent MG1655 ¹³C yield point(s) — basal acetate ÷ glucose
    uptake at the basal growth rate (a single raw yield anchor per source)."""
    root = _bundle_path(bundle_path).parent
    ac = pd.read_csv(root / "data/basal/acetate_secretion.tsv", sep="\t", comment="#")
    gl = pd.read_csv(root / "data/basal/glucose_uptake.tsv", sep="\t", comment="#")
    gr = pd.read_csv(root / "data/basal/growth_rate.tsv", sep="\t", comment="#")
    glm = {r.source_id: float(r.value) for r in gl.itertuples()}
    grm = {r.source_id: float(r.value) for r in gr.itertuples()}
    x, y, labels = [], [], []
    for r in ac.itertuples():
        s = r.source_id
        if s in glm and s in grm:
            yac = _yc(float(r.value), 6.0 * glm[s])
            if yac is not None:
                x.append(grm[s]); y.append(yac); labels.append(s)
    return {"x": x, "y": y, "labels": labels, "label": "MG1655 ¹³C anchor"}


def model_curve(model: dict) -> dict:
    """Model overflow curve from the baked GUR-titration arm: (emergent growth
    rate, Y_ac) per cap. ``glucose`` is signed (neg = uptake)."""
    x, y = [], []
    for r in model.get("rows", []):
        mu, ac, glc = r.get("growth_rate_per_h"), r.get("acetate"), r.get("glucose")
        if mu is None or ac is None or not glc:
            continue
        yac = _yc(float(ac), 6.0 * abs(float(glc)))
        if yac is not None:
            x.append(float(mu)); y.append(yac)
    order = sorted(range(len(x)), key=lambda i: x[i])
    return {"x": [x[i] for i in order], "y": [y[i] for i in order]}


def build_overflow(model: dict, bundle_path: Path | None = None) -> tuple[dict, dict]:
    """(axes, card_node) for the overflow axis: model Y_ac-vs-growth graded vs the
    Vemuri curve (threshold_linear), with Basan + the MG1655 anchor as context."""
    ref = vemuri_curve(bundle_path)
    ctx = basan_curve(bundle_path)
    anchor = mg1655_anchor(bundle_path)
    mc = model_curve(model)
    axes = {"metabolism.acetate_overflow": {
        "group": "Overflow",
        "label": "Acetate overflow (carbon yield vs growth rate)", "units": "",
        "plot": "overflow_curve",
        "how": ("Model: acetate-carbon yield Y_ac = (2·acetate)/(6·glucose) at the "
                "emergent growth rate of each glucose-uptake-cap setting (GUR "
                "titration). Graded vs Vemuri 2006 (MG1655 glucose-limited "
                "chemostat; Y_ac raw from co-measured glucose uptake) by the "
                "threshold_linear criterion: the response shape (Y_ac rising ~0 → "
                "linear above a threshold) and the yield slope; the threshold is "
                "reported but secondary (it shifts strain-to-strain). Basan 2015 "
                "(NCM3722 batch, SI-derived yield) and an independent MG1655 ¹³C "
                "yield point are overlaid as context."),
        "criterion": {
            "type": "threshold_linear", "ref_x": ref["x"], "ref_y": ref["y"],
            "active_eps": 0.005, "lin_good": 0.90, "lin_warn": 0.70,
            "slope_tol": 0.30, "slope_warn": 0.60, "onset_tol": 0.10, "onset_warn": 0.20,
            # plot-only context (not part of grading):
            "ref_err": ref["err"], "ref_label": ref["label"],
            "context_curves": [ctx], "anchor": anchor,
        },
    }}
    card = {"acetate_overflow": {"x": mc["x"], "y": mc["y"]}}
    return axes, card


def build(model_json: Path = _MODEL_JSON,
          bundle_path: Path | None = None) -> tuple[dict, dict, dict]:
    """Return (card, reference, model) — the gradeable inputs (importable for tests)."""
    model = json.load(open(model_json, encoding="utf-8")) if Path(model_json).exists() else {"rows": []}
    ax, cnode = build_overflow(model, bundle_path)
    reference = {
        "title": "Aerobic acetate overflow — acetate-carbon yield vs growth rate",
        "status": "populated",
        "stimulus": {
            "reference_model": "experimental literature (Vemuri 2006 chemostat; Basan 2015 context)",
            "measured_model": f"v2ecoli baseline FBA · GUR titration ({model.get('knob', '')})",
            "ensemble": f"{model.get('n_seeds', '?')}×{model.get('generations', '?')} per cap",
        },
        "findings": [
            "The model's acetate-overflow response is graded as a dimensionless "
            "acetate-carbon yield vs growth rate (per-OD vs per-gDW cancels), "
            "shape-first — the 0→linear rise and its slope, threshold secondary.",
            "Primary reference: Vemuri 2006 MG1655 glucose-limited chemostat (Y_ac raw "
            "from co-measured glucose uptake). Basan 2015 + an MG1655 ¹³C point are context.",
        ],
        "footer": "Behavioral report card (threshold_linear grader) · aerobic acetate overflow.",
        "axes": ax,
    }
    return cnode_to_card(cnode), reference, model


def cnode_to_card(cnode: dict) -> dict:
    """Wrap the overflow card node under its axis-path prefix (metabolism.*)."""
    return {"metabolism": cnode}
