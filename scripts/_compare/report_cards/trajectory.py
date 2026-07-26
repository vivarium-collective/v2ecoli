"""`trajectory` card — interactive Plotly value-vs-time overlay of v2ecoli vs
vEcoli, matched on simulation time. as_step Step; the harness invokes it via
core.link_registry['trajectory_report_card'].

Reads the zarr stores directly off state["v2_dir"]/state["ve_dir"] via
``read_pbg_local`` (the local-path reader) — NOT state["observables"]/
state["plot_trajs"], which the older (standard/statistical/parca) cards
consume. See tests/fixtures/redux_cards/README.md.
"""
from __future__ import annotations

import os

from process_bigraph.composite import as_step

from scripts._compare.plotly_helpers import overlay_html
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local


def _read_seed(dir_path: str, prefix: str, seed: int, observables) -> dict:
    path = os.path.join(dir_path, f"{prefix}_seed{seed:02d}.zarr")
    if not os.path.isdir(path):
        return {}
    return read_pbg_local(path, observables)


def _gen_bounds(gen_traj) -> list:
    """Sim-time boundaries where the generation label changes, for vlines."""
    if gen_traj is None:
        return []
    t, g = gen_traj
    return [float(t[i]) for i in range(1, len(g)) if g[i] != g[i - 1]]


def build_per_obs(state: dict) -> dict:
    """{obs: {"v2": [(t,v),...], "ve": [(t,v),...], "gen_bounds": [...]}} across
    all seeds in state["seeds"], skipping any seed whose zarr is missing."""
    n_seeds = max(int(state.get("seeds") or 1), 1)
    per_obs = {obs: {"v2": [], "ve": [], "gen_bounds": []} for obs in OBSERVABLES}
    for seed in range(n_seeds):
        v2 = _read_seed(state["v2_dir"], "v2ecoli", seed, OBSERVABLES)
        ve = _read_seed(state["ve_dir"], "vecoli", seed, OBSERVABLES)
        for obs in OBSERVABLES:
            if obs in v2:
                per_obs[obs]["v2"].append(v2[obs])
                if not per_obs[obs]["gen_bounds"]:
                    per_obs[obs]["gen_bounds"] = _gen_bounds(v2.get("_generation"))
            if obs in ve:
                per_obs[obs]["ve"].append(ve[obs])
    return {obs: d for obs, d in per_obs.items() if d["v2"] or d["ve"]}


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="trajectory_report_card",
         aliases=["trajectory"])
def update_trajectory_report_card(state):
    per_obs = build_per_obs(state)
    html = overlay_html(per_obs, title=state.get("name", ""))
    if not html:
        html = '<p style="color:#6b7280">no matched trajectories available</p>'
    # Purely descriptive overlay — no scalar comparison here (that's the
    # standard/statistical/distribution cards' job), so always ungraded.
    return {"card_html": html, "verdict": "ungraded", "axes": []}


REPORT_CARD_STEPS["trajectory_report_card"] = update_trajectory_report_card
