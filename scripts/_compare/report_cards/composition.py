"""`composition` card — proteome/RNA/"other" mass-fraction comparison
(v2ecoli vs vEcoli) plus a per-generation mass-growth readout and an
optional perf note. as_step Step; the harness invokes it via
core.link_registry['composition_report_card'].

Mass fractions (protein_mass/dry_mass, rna_mass/dry_mass, 1 - the two) are
computed from each engine's FINAL emitted values and graded independently
via rel_tol (v2ecoli.library.card_criteria.grade_axis), vEcoli as the
reference — within 5%, drift to 10%, else mismatch.

The redux_cards fixture is 1 seed x 1 generation (no division events), so
there is no doubling time to measure; the readout instead reports the
final-vs-initial cell_mass ratio over the available window per engine — a
single-generation growth-factor stat, descriptive only (not graded).

An optional steps/s perf line is read from ``<v2_dir>/run_summary.json`` /
``<ve_dir>/run_summary.json`` if present (skipped gracefully otherwise — the
fixture has no such file).

Reads the zarr stores directly off state["v2_dir"]/state["ve_dir"] via
``read_pbg_local`` — same source as the trajectory/distribution/metabolism
cards.
"""
from __future__ import annotations

import html as _html
import json
import os

from process_bigraph.composite import as_step

from scripts._compare.plotly_helpers import grouped_bar_html
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.verdict import worst
from scripts.compare_matched_trajectories import OBSERVABLES, read_pbg_local

MASS_OBS = ["dry_mass", "protein_mass", "rna_mass", "cell_mass"]
FRACTION_TOL = {"type": "rel_tol", "tol_rel": 0.05}


def _read_seed(dir_path: str, prefix: str, seed: int, observables) -> dict:
    path = os.path.join(dir_path, f"{prefix}_seed{seed:02d}.zarr")
    if not os.path.isdir(path):
        return {}
    return read_pbg_local(path, observables)


def _collect(state: dict) -> dict:
    """{obs: {"v2": [(t,v),...], "ve": [(t,v),...]}} across all seeds, for
    the mass observables only."""
    n_seeds = max(int(state.get("seeds") or 1), 1)
    per_obs = {obs: {"v2": [], "ve": []} for obs in MASS_OBS}
    for seed in range(n_seeds):
        v2 = _read_seed(state["v2_dir"], "v2ecoli", seed, OBSERVABLES)
        ve = _read_seed(state["ve_dir"], "vecoli", seed, OBSERVABLES)
        for obs in MASS_OBS:
            if obs in v2:
                per_obs[obs]["v2"].append(v2[obs])
            if obs in ve:
                per_obs[obs]["ve"].append(ve[obs])
    return per_obs


def _final(traces: list) -> float | None:
    finals = [float(v[-1]) for _, v in traces if len(v)]
    return (sum(finals) / len(finals)) if finals else None


def _initial(traces: list) -> float | None:
    initials = [float(v[0]) for _, v in traces if len(v)]
    return (sum(initials) / len(initials)) if initials else None


def _fractions(per_obs: dict, engine: str) -> dict | None:
    """{"protein": frac, "rna": frac, "other": frac} of dry_mass for one
    engine, or None if dry_mass is missing/zero."""
    dry = _final(per_obs["dry_mass"][engine])
    protein = _final(per_obs["protein_mass"][engine])
    rna = _final(per_obs["rna_mass"][engine])
    if not dry:
        return None
    protein_frac = (protein / dry) if protein is not None else None
    rna_frac = (rna / dry) if rna is not None else None
    other_frac = (1.0 - (protein_frac or 0.0) - (rna_frac or 0.0)
                 if protein_frac is not None and rna_frac is not None else None)
    return {"protein": protein_frac, "rna": rna_frac, "other": other_frac}


def _grade_fractions(ve_frac: dict | None, v2_frac: dict | None) -> list:
    from v2ecoli.library.card_criteria import grade_axis

    axes = []
    for key, label in (("protein", "protein fraction"), ("rna", "rna fraction"),
                       ("other", "other fraction")):
        ref = (ve_frac or {}).get(key)
        got = (v2_frac or {}).get(key)
        criterion = dict(FRACTION_TOL, reference=ref)
        graded = grade_axis(got, criterion)
        axes.append({
            "id": f"composition.{key}_fraction", "label": label,
            "verdict": graded["verdict"], "value": graded.get("value"),
            "meter": graded.get("meter", ""),
            "detail": dict(graded.get("detail") or {}, ve_fraction=ref, v2_fraction=got),
        })
    return axes


def _fraction_bar_html(ve_frac: dict | None, v2_frac: dict | None, title: str) -> str:
    if not ve_frac and not v2_frac:
        return ""
    cats = ["protein", "rna", "other"]
    ve_vals = [(ve_frac or {}).get(c) for c in cats]
    v2_vals = [(v2_frac or {}).get(c) for c in cats]
    return grouped_bar_html(cats, ve_vals, v2_vals,
                            title=f"{title} — mass fraction".strip(" —"),
                            yaxis_title="fraction of dry mass", first=True)


def _growth_readout_html(per_obs: dict) -> str:
    """Descriptive final-vs-initial cell_mass ratio per engine (no division
    events in this 1-generation fixture, so no doubling time)."""
    ve_i, ve_f = _initial(per_obs["cell_mass"]["ve"]), _final(per_obs["cell_mass"]["ve"])
    v2_i, v2_f = _initial(per_obs["cell_mass"]["v2"]), _final(per_obs["cell_mass"]["v2"])
    rows = []
    if ve_i and ve_f is not None:
        rows.append(f"<li>vEcoli: {ve_f / ve_i:.3f}× over the window "
                    f"({ve_i:.4g} → {ve_f:.4g} fg)</li>")
    if v2_i and v2_f is not None:
        rows.append(f"<li>v2ecoli: {v2_f / v2_i:.3f}× over the window "
                    f"({v2_i:.4g} → {v2_f:.4g} fg)</li>")
    if not rows:
        return ""
    return ('<div class="growth-readout"><p style="color:#6b7280;font-size:12px">'
            "final-vs-initial cell mass (1 generation, no division events — "
            "not a doubling time)</p><ul style=\"font-size:13px\">"
            + "".join(rows) + "</ul></div>")


def _perf_note_html(state: dict) -> str:
    """Optional steps/s line from <v2_dir>/run_summary.json or
    <ve_dir>/run_summary.json; empty string if neither exists or carries a
    recognizable perf key."""
    for dir_key, engine in (("v2_dir", "v2ecoli"), ("ve_dir", "vEcoli")):
        path = os.path.join(state.get(dir_key) or "", "run_summary.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        steps_per_sec = summary.get("steps_per_sec")
        if steps_per_sec is None:
            n_steps, wall_time = summary.get("n_steps"), summary.get("wall_time_s")
            if n_steps and wall_time:
                steps_per_sec = n_steps / wall_time
        if steps_per_sec is not None:
            return (f'<p style="color:#6b7280;font-size:12px">{_html.escape(engine)} '
                    f"perf: {steps_per_sec:.3g} steps/s</p>")
    return ""


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="composition_report_card",
         aliases=["composition"])
def update_composition_report_card(state):
    per_obs = _collect(state)
    ve_frac = _fractions(per_obs, "ve")
    v2_frac = _fractions(per_obs, "v2")
    title = state.get("name", "")

    bar_html = _fraction_bar_html(ve_frac, v2_frac, title)
    growth_html = _growth_readout_html(per_obs)
    perf_html = _perf_note_html(state)
    html = bar_html + growth_html + perf_html
    if not html:
        html = '<p style="color:#6b7280">no matched composition data available</p>'

    axes = _grade_fractions(ve_frac, v2_frac)
    return {"card_html": html, "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["composition_report_card"] = update_composition_report_card
