"""`distribution` card — per-cell violin/strip of v2ecoli vs vEcoli values,
pooled across seeds x generations, graded by a Welch t-test per observable
(``v2ecoli.library.card_criteria.grade_axis``, ``type: "ttest"``). as_step
Step; the harness invokes it via core.link_registry['distribution_report_card'].

Reads the zarr stores directly off state["v2_dir"]/state["ve_dir"] via
``read_pbg_local`` — same source as the trajectory card. Per generation
present in a seed's trajectory, one "cell" value is the mean of that
generation's observable window; those per-cell values are pooled across all
seeds/generations available on disk.

grade_axis's ttest branch already returns "ungraded" when either side has
fewer than 2 values (the redux_cards fixture is 1 seed x 1 generation ->
n=1 per engine, so every axis is ungraded there by construction) — no extra
guard is needed here, only that we never crash building the pooled lists.
"""
from __future__ import annotations

import os

import numpy as np
from process_bigraph.composite import as_step

from scripts._compare.plotly_helpers import violin_html
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.verdict import worst
from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

TTEST_CRITERION = {"type": "ttest", "within_pct": 0.05, "mismatch_pct": 0.10, "p_min": 0.05}


def _per_cell_values(dir_path: str, prefix: str, seed: int, observables) -> dict:
    """{obs: [per-generation mean, ...]} for one seed's zarr; one value per
    generation present in the trajectory (the "cell" unit for the t-test)."""
    path = os.path.join(dir_path, f"{prefix}_seed{seed:02d}.zarr")
    if not os.path.isdir(path):
        return {}
    data = read_pbg_local(path, observables)
    gen_traj = data.get("_generation")
    out: dict = {}
    for obs in observables:
        if obs not in data:
            continue
        _, v = data[obs]
        v = np.asarray(v, dtype=float)
        if gen_traj is not None:
            _, g = gen_traj
            g = np.asarray(g)
            out[obs] = [float(np.mean(v[g == gnum])) for gnum in sorted(set(g.tolist()))
                        if np.any(g == gnum)]
        elif v.size:
            out[obs] = [float(np.mean(v))]
    return out


def pooled_per_cell(state: dict) -> dict:
    """{obs: {"v2": [...], "ve": [...]}} pooled over all seeds in state["seeds"]."""
    n_seeds = max(int(state.get("seeds") or 1), 1)
    pooled = {obs: {"v2": [], "ve": []} for obs in OBSERVABLES}
    for seed in range(n_seeds):
        v2 = _per_cell_values(state["v2_dir"], "v2ecoli", seed, OBSERVABLES)
        ve = _per_cell_values(state["ve_dir"], "vecoli", seed, OBSERVABLES)
        for obs in OBSERVABLES:
            pooled[obs]["v2"].extend(v2.get(obs, []))
            pooled[obs]["ve"].extend(ve.get(obs, []))
    return pooled


def _grade(pooled: dict) -> list:
    from v2ecoli.library.card_criteria import grade_axis

    axes = []
    for obs, vals in pooled.items():
        v2_vals, ve_vals = vals["v2"], vals["ve"]
        criterion = dict(TTEST_CRITERION, ref_values=ve_vals)
        graded = grade_axis({"values": v2_vals}, criterion)
        axes.append({
            "id": f"distribution.{obs}", "label": obs, "verdict": graded["verdict"],
            "value": graded.get("value"), "meter": graded.get("meter", ""),
            "detail": dict(graded.get("detail") or {}, v2_values=v2_vals, ve_values=ve_vals),
        })
    return axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="distribution_report_card",
         aliases=["distribution"])
def update_distribution_report_card(state):
    pooled = pooled_per_cell(state)
    axes = _grade(pooled)
    axis_records = [
        {"label": ax["label"], "v2_values": pooled[ax["label"]]["v2"],
         "ve_values": pooled[ax["label"]]["ve"], "meter": ax["meter"]}
        for ax in axes
    ]
    html = violin_html(axis_records, title=state.get("name", ""))
    if not html:
        html = '<p style="color:#6b7280">no per-cell values available</p>'
    return {"card_html": html, "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["distribution_report_card"] = update_distribution_report_card
