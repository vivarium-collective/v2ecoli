"""`metabolism` card — growth-law + biomass comparison for THIS study's
condition: v2ecoli-vs-vEcoli growth-rate trace (Plotly overlay) plus a
biomass (cell/dry mass, final-value) grouped-bar. as_step Step; the harness
invokes it via core.link_registry['metabolism_report_card'].

Scope note: there is no flux/exchange observable in the matched-trajectory
data this comparison harness reads (compare_matched_trajectories.OBSERVABLES
carries only cell_mass/dry_mass/protein_mass/rna_mass/
instantaneous_growth_rate — verified against the redux_cards fixture, see
tests/fixtures/redux_cards/README.md). So this card is the "growth law"
proxy for the condition — growth rate vs this condition's nutrient regime —
not an exchange-flux comparison. Wiring a per-condition flux scatter (the
``flux_scatter``/``composition`` criteria already exist in
v2ecoli.library.card_criteria) is a documented follow-up once a flux/
exchange leaf is emitted into the matched zarr.

Reads the zarr stores directly off state["v2_dir"]/state["ve_dir"] via
``read_pbg_local`` — same source as the trajectory/distribution cards.
"""
from __future__ import annotations

import os

from process_bigraph.composite import as_step

from scripts._compare.plotly_helpers import grouped_bar_html, overlay_html
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.verdict import worst
from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

GROWTH_OBS = "instantaneous_growth_rate"
BIOMASS_OBS = ["cell_mass", "dry_mass"]
GROWTH_TOL = {"type": "rel_tol", "tol_rel": 0.05}


def _read_seed(dir_path: str, prefix: str, seed: int, observables) -> dict:
    path = os.path.join(dir_path, f"{prefix}_seed{seed:02d}.zarr")
    if not os.path.isdir(path):
        return {}
    return read_pbg_local(path, observables)


def _collect(state: dict) -> dict:
    """{obs: {"v2": [(t,v),...], "ve": [(t,v),...]}} across all seeds, for
    the growth + biomass observables only."""
    obs_set = [GROWTH_OBS] + BIOMASS_OBS
    n_seeds = max(int(state.get("seeds") or 1), 1)
    per_obs = {obs: {"v2": [], "ve": []} for obs in obs_set}
    for seed in range(n_seeds):
        v2 = _read_seed(state["v2_dir"], "v2ecoli", seed, OBSERVABLES)
        ve = _read_seed(state["ve_dir"], "vecoli", seed, OBSERVABLES)
        for obs in obs_set:
            if obs in v2:
                per_obs[obs]["v2"].append(v2[obs])
            if obs in ve:
                per_obs[obs]["ve"].append(ve[obs])
    return per_obs


def _finals(traces: list) -> list:
    """Last emitted value of each (times, values) trace."""
    return [float(v[-1]) for _, v in traces if len(v)]


def _grade_growth(per_obs: dict) -> dict:
    from v2ecoli.library.card_criteria import grade_axis

    v2_final = _finals(per_obs[GROWTH_OBS]["v2"])
    ve_final = _finals(per_obs[GROWTH_OBS]["ve"])
    got = sum(v2_final) / len(v2_final) if v2_final else None
    ref = sum(ve_final) / len(ve_final) if ve_final else None
    criterion = dict(GROWTH_TOL, reference=ref)
    graded = grade_axis(got, criterion)
    return {
        "id": "metabolism.growth_rate_final", "label": "Growth rate (final)",
        "verdict": graded["verdict"], "value": graded.get("value"),
        "meter": graded.get("meter", ""),
        "detail": dict(graded.get("detail") or {}, v2_final=v2_final, ve_final=ve_final),
    }


def _biomass_bar_html(per_obs: dict, title: str, first: bool = False) -> str:
    ve_vals = [
        (sum(f) / len(f)) if (f := _finals(per_obs[obs]["ve"])) else None
        for obs in BIOMASS_OBS
    ]
    v2_vals = [
        (sum(f) / len(f)) if (f := _finals(per_obs[obs]["v2"])) else None
        for obs in BIOMASS_OBS
    ]
    if not any(v is not None for v in ve_vals + v2_vals):
        return ""
    return grouped_bar_html(BIOMASS_OBS, ve_vals, v2_vals,
                            title=f"{title} — biomass (final)".strip(" —"),
                            yaxis_title="mass (fg)", first=first)


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="metabolism_report_card",
         aliases=["metabolism"])
def update_metabolism_report_card(state):
    per_obs = _collect(state)
    title = state.get("name", "")

    growth_html = overlay_html({GROWTH_OBS: per_obs[GROWTH_OBS]}, title=title)
    biomass_html = _biomass_bar_html(per_obs, title, first=not growth_html)
    html = growth_html + biomass_html
    if not html:
        html = '<p style="color:#6b7280">no matched growth/biomass data available</p>'

    axis = _grade_growth(per_obs)
    return {"card_html": html, "verdict": worst([axis["verdict"]]), "axes": [axis]}


REPORT_CARD_STEPS["metabolism_report_card"] = update_metabolism_report_card
