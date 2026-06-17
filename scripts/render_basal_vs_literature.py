"""Render the basal report card's ``vs_literature`` reference mode.

A third reference mode for ``population_phenotype_basal`` (alongside self-pin
*drift* and ``vs_vecoli`` *equivalence*): grade the blessed v2ecoli baseline
against **curated experimental + theoretical reference values** from the
ecoli-sources ``validation_data`` bundle — the same reference-agnostic grader
(PR #134) pointed at literature instead of a pinned run.

Physiology-first scope (μ, q_glc, Yxs):

  * ``physiology.biomass_yield``  — Yxs = μ / (q_glc · M_glc). Graded against the
    measured band AND the ``theoretical_max`` ceiling: a model value above the
    stoichiometric ceiling is a *differentiated first-principles failure*.
  * ``physiology.growth_rate``    — μ (1/h) vs the measured band.
  * ``physiology.glucose_uptake`` — q_glc (mmol/gDW/h) vs the measured band.

The model (measured-side) values come from the blessed self-pin reference
fixture — the same baseline ensemble graded by the #134 cards — so this mode
needs no new run. The literature (reference-side) values come from
``ecoli_sources.VALIDATION_BUNDLE_PATH`` via the ``validate=False`` path (the
validation bundle has none of the ParCa required keys; a missing slot ->
``ungraded``).

Run:
    python scripts/render_basal_vs_literature.py
    # -> docs/report_cards/population_phenotype_basal/vs_literature/
"""
from __future__ import annotations

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

_FIXTURE = REPO / "tests/fixtures/population_phenotype_basal_reference.json"
OUT = REPO / "docs/report_cards/population_phenotype_basal/vs_literature"

# (card path, bundle observable key, label, units, has-theoretical-max, model derivation)
_AXES = [
    ("physiology.growth_rate", "growth_rate", "Growth rate (μ)", "1/h", False,
     "Model: ensemble growth rate from per-cell doubling times (μ = ln 2 / doubling_time) "
     "of the blessed baseline ensemble (post-burn-in cells)."),
    ("physiology.biomass_yield", "biomass_yield", "Biomass yield (Yxs)", "gDW/g glucose", True,
     "Model: Yxs = μ / (q_glc · M_glc), M_glc = 0.180156 g/mmol — the ensemble ratio of "
     "growth rate to specific glucose uptake (a ratio of means, so no per-cell spread)."),
    ("physiology.glucose_uptake", "glucose_uptake", "Glucose uptake (q_glc)", "mmol/gDW/h", False,
     "Model: ensemble-mean specific glucose uptake = |GLC[p] exchange flux| over the "
     "post-burn-in cells."),
]


def model_physiology(fixture_path: Path = _FIXTURE) -> dict:
    """Blessed-baseline μ, q_glc, Yxs from the self-pin reference fixture.

    μ from the doubling-time per-cell reference values (seconds); q_glc from the
    pinned exchange-flux vector at GLC[p] (uptake is negative -> magnitude).
    Yxs = μ / (q_glc · M_glc) — the ensemble ratio, matching how the references
    are defined and how the finding was derived."""
    fx = json.load(open(fixture_path, encoding="utf-8"))
    ax = fx["axes"]
    # per-cell growth rate from per-cell doubling times (seconds)
    dt = ax["physiology.doubling_time"]["criterion"]["ref_values"]
    mu_cells = [math.log(2) / (t / 3600.0) for t in dt]
    mu = sum(mu_cells) / len(mu_cells)
    # per-cell glucose uptake magnitude (the KPI axis carries per-cell flux;
    # fall back to the ensemble exchange vector if absent)
    gcrit = ax.get("fluxes.glucose", {}).get("criterion", {})
    if gcrit.get("ref_values"):
        q_cells = [abs(v) for v in gcrit["ref_values"]]
    else:
        fc = ax["fluxes.exchange"]["criterion"]
        q_cells = [abs(fc["ref_vector"][fc["flux_ids"].index("GLC[p]")])]
    q_glc = sum(q_cells) / len(q_cells)
    yxs = mu / (q_glc * M_GLC)
    return {
        "biomass_yield": yxs, "growth_rate": mu, "glucose_uptake": q_glc,
        "growth_rate_cells": mu_cells, "glucose_uptake_cells": q_cells,
        "ensemble": fx.get("stimulus", {}).get("ensemble", ""),
        "n_cells": len(dt),
    }


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
    """Measured (model-side) card: the blessed baseline's physiology — scalar
    means for grading + per-cell value lists for the sim violin (yield is a
    ratio of ensemble means, so it has no per-cell distribution)."""
    return {"physiology": {
        "biomass_yield": {"mean": model["biomass_yield"]},
        "growth_rate": {"mean": model["growth_rate"],
                        "values": model.get("growth_rate_cells")},
        "glucose_uptake": {"mean": model["glucose_uptake"],
                           "values": model.get("glucose_uptake_cells")},
    }}


def build(fixture_path: Path = _FIXTURE, bundle_path: Path | None = None) -> tuple[dict, dict, dict]:
    """Return (card, reference, model) — the gradeable inputs (importable for tests)."""
    model = model_physiology(fixture_path)
    lit = literature(bundle_path)
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
            "Biomass yield carries a theoretical_max ceiling — exceeding it is a "
            "differentiated first-principles failure, not merely a deviation.",
            "Model μ, q_glc, Yxs read from the blessed self-pin baseline fixture; "
            "references from ecoli_sources.VALIDATION_BUNDLE_PATH.",
        ],
        "footer": "Behavioral report card (PR #134 grader) · vs_literature mode · "
                  "physiology (μ, q_glc, Yxs).",
        "axes": build_reference(lit, model),
    }
    return build_card(model), reference, model


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
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
    main()
