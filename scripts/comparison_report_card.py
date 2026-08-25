"""scripts/comparison_report_card.py — the STANDARD multi-section v2ecoli<->vEcoli
comparison report, reproducible from the GovCloud S3 data.

One deterministic generator that, given each engine's per-condition S3 experiment
IDs, assembles the canonical multi-section report via ``report.render_report``:

  1. Overview            — verdict counts + one-line-per-condition summary.
  2. ParCa / initial     — per-condition t~0 (first matched emit) cell/dry/protein
     state               /rna masses for BOTH engines. Both sims start from their
                           ParCa fit, so matched initial masses ARE the "same
                           initial state" evidence (robust where cross-env
                           SimulationData pickle loads are not — v2ecoli's dill
                           needs v2ecoli's wholecell, vEcoli's cPickle needs
                           vEcoli's; neither loads in the harness venv).
  3. Report card         — report_card.grade_card -> verdict_json over the 7
                           standard axes (4 masses + growth_rate + active_RNAP +
                           active_ribosome), graded at MATCHED gen-1 timepoints
                           (vEcoli = reference), 5%/10% bands.
  4. Matched trajectory  — per condition, per observable rows + inline-SVG overlay
     sections             images (reuses compare_matched_trajectories).

Nothing here is ad-hoc: it reuses the existing S3 readers
(compare_matched_trajectories, vecoli_parquet_reader), the report-card library
(report_card_section.build_report_card -> v2ecoli.library.report_card), and the
multi-section page renderer (report.render_report).

Why matched-timepoint, not end-of-run snapshot: v2ecoli is a faithful PORT of
vEcoli (same ParCa, same processes, same initial state). At MATCHED simulation
times basal tracks to ~1%. An end-of-run SNAPSHOT compares cells at different
cell-cycle phases -> +/-100% artifacts (invalid). ``total_rna_init`` is a unit
mismatch and is excluded.

Credentials (aiobotocore SSO refresh is unreliable; env creds required)::

    export AWS_PROFILE=stanford-sso AWS_DEFAULT_REGION=us-gov-west-1
    eval "$(aws configure export-credentials --format env)"
    .venv/bin/python scripts/comparison_report_card.py --only basal

Condition -> (v2ecoli dir, vEcoli dir) selection precedence (first that applies):

  1. ``--conditions-json '{"basal":["<v2_dir>","<ve_dir>"], ...}'`` (explicit override)
  2. ``<out>/experiments.json`` written by ``scripts/comparison_harness.sh launch``
     — so launch -> report is one reproducible chain with no hand-editing.
     Schema: ``{"conditions": {cond: [v2_dir, ve_dir]}, ...}`` (or a bare
     ``{cond: [v2_dir, ve_dir]}`` map).
  3. the ``CONDITIONS`` default baked in below.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._compare import report
from scripts._compare.report_card_section import build_report_card
from v2ecoli.core import build_core
from scripts._compare.vecoli_parquet_reader import read_vecoli_trajectory
from scripts.compare_matched_trajectories import (
    BUCKET, PREFIX, REGION, OBS_LABEL, _legend_html, overlay_svg_multi,
    read_v2ecoli_trajectory, read_pbg_local, read_vecoli_pbg_trajectory,
    _gen1_window)

# condition -> (v2ecoli zarr experiment dir, vEcoli parquet experiment dir).
# v2: sim48-v2-full-* is basal-everywhere for the NON-basal dirs (the old bug);
# only BASAL is a valid v2<->vE comparison until sim59-v2fix-* (per-condition
# initial state, ParCa commit 872bfdf2) lands — then replace the v2 dirs here.
CONDITIONS: dict[str, tuple[str, str]] = {
    "basal":     ("sim48-v2-full-basal-d52e",     "sim47-ve-full-basal-b498"),
    "with_aa":   ("sim48-v2-full-with_aa-0f7a",   "sim47-ve-full-with_aa-5261"),
    "succinate": ("sim48-v2-full-succinate-9493", "sim47-ve-full-succinate-e2cf"),
    "no_oxygen": ("sim48-v2-full-no_oxygen-2371", "sim47-ve-full-no_oxygen-9784"),
    "acetate":   ("sim48-v2-full-acetate-d575",   "sim47-ve-full-acetate-7042"),
}

# The 7 standard axes. observable-leaf -> report-card key (growth uses the short
# name CARD_AXES expects). active_RNAP/active_ribosome are graded once a compact
# emitter exports them (absent from sim48 -> ungraded, shown honestly).
OBSERVABLES = ["cell_mass", "dry_mass", "protein_mass", "rna_mass",
               "instantaneous_growth_rate", "active_RNAP", "active_ribosome"]
CARD_KEY = {"instantaneous_growth_rate": "growth_rate"}
MASS_OBS = ["cell_mass", "dry_mass", "protein_mass", "rna_mass"]
SEEDS = [0, 1, 2, 3]
TOL, TOL2 = 0.05, 0.10  # within_tol / drift bands

# extra report-card axes beyond CARD_AXES (cell_mass + growth_rate).
EXTRA_AXES = [
    {"group": "Physiology", "path": "physiology.dry_mass",
     "label": "Dry mass", "key": "dry_mass", "plot": "violin"},
    {"group": "Physiology", "path": "physiology.protein_mass",
     "label": "Protein mass", "key": "protein_mass", "plot": "violin"},
    {"group": "Physiology", "path": "physiology.rna_mass",
     "label": "RNA mass", "key": "rna_mass", "plot": "violin"},
    {"group": "Ribosomes", "path": "ribosomes.active_ribosome",
     "label": "Active ribosome", "key": "active_ribosome", "plot": "violin"},
    {"group": "Transcription", "path": "transcription.active_RNAP",
     "label": "Active RNAP", "key": "active_RNAP", "plot": "violin"},
]


def _grade(med_abs_rel: float) -> str:
    if med_abs_rel <= TOL:
        return "within_tol"
    return "drift" if med_abs_rel <= TOL2 else "mismatch"


def _matched(v2g1, veg1) -> dict | None:
    """Matched gen-1 stats for one observable: vEcoli (dense) interpolated onto
    v2ecoli's emit grid. Returns init values (first matched point), gen-1 means,
    and median |relative Δ| over the window (vEcoli = reference)."""
    v2_t, v2_v = v2g1
    ve_t, ve_v = veg1
    if v2_t.size == 0 or ve_t.size == 0:
        return None
    grid = v2_t[(v2_t >= ve_t.min()) & (v2_t <= ve_t.max())]
    if grid.size == 0:
        return None
    ve_on = np.interp(grid, ve_t, ve_v)
    v2_on = np.interp(grid, v2_t, v2_v)
    denom = np.where(np.abs(ve_on) > 1e-30, ve_on, np.nan)
    rel = ((v2_on - ve_on) / denom)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return None
    return {
        "init_t": float(grid[0]), "init_v2": float(v2_on[0]),
        "init_ve": float(ve_on[0]),
        "init_rel": float((v2_on[0] - ve_on[0]) / ve_on[0]) if ve_on[0] else float("nan"),
        "v2_mean": float(np.mean(v2_on)), "ve_mean": float(np.mean(ve_on)),
        "median_rel": float(np.median(np.abs(rel))),
        "max_rel": float(np.max(np.abs(rel))), "n": int(rel.size),
    }


# --------------------------------------------------------------------------- #
# read every condition's per-seed matched stats + a seed's trajectories to plot
# --------------------------------------------------------------------------- #
def _gen_bounds(gen_traj) -> list[float]:
    """Times at which the generation label increments (gen boundaries)."""
    if not gen_traj:
        return []
    gt, gv = gen_traj
    return [float(gt[i]) for i in range(1, len(gv)) if gv[i] != gv[i - 1]]


def build(conditions: dict[str, tuple[str, str]], *,
          read_v2=None, read_ve=None, seeds=None) -> dict:
    """Read each condition's per-seed matched stats + trajectories.

    Default readers compare S3 v2ecoli-zarr vs vEcoli-Nextflow-parquet. Pass
    ``read_v2`` / ``read_ve`` ``(dir_or_path, seed) -> {obs: (t, v)}`` to drive a
    different pairing — e.g. pbg-vs-pbg, where BOTH engines read v2ecoli-format
    zarr via ``read_pbg_local`` — and ``seeds`` to override the seed list.
    """
    if read_v2 is None:
        read_v2 = lambda d, s: read_v2ecoli_trajectory(d, s, OBSERVABLES)
    if read_ve is None:
        read_ve = lambda d, s: read_vecoli_trajectory(
            f"{PREFIX}/{d}", BUCKET, OBSERVABLES, lineage_seed=s, region=REGION)
    seeds = SEEDS if seeds is None else seeds
    cond_data: dict[str, tuple] = {}
    for cond, (v2_dir, ve_dir) in conditions.items():
        print(f"=== {cond} ===  v2={v2_dir}  ve={ve_dir}")
        per_obs = {obs: [] for obs in OBSERVABLES}
        # all-seed full trajectories per observable, for the overlay plots.
        plot_trajs = {obs: {"v2": [], "ve": []} for obs in OBSERVABLES}
        v2_bounds: list[float] = []
        for seed in seeds:
            try:
                v2 = read_v2(v2_dir, seed)
            except Exception as e:  # noqa: BLE001
                print(f"  seed{seed}: v2 read failed: {type(e).__name__} {str(e)[:60]}")
                continue
            if not v2:
                continue
            try:
                ve = read_ve(ve_dir, seed)
            except Exception as e:  # noqa: BLE001
                print(f"  seed{seed}: vE read failed: {type(e).__name__} {str(e)[:60]}")
                continue
            v2_gen, ve_gen = v2.get("_generation"), ve.get("_generation")
            if not v2_bounds:
                v2_bounds = _gen_bounds(v2_gen)
            for obs in OBSERVABLES:
                if obs not in v2 or obs not in ve:
                    continue
                v2g1 = _gen1_window(v2[obs], v2_gen) if v2_gen else v2[obs]
                veg1 = _gen1_window(ve[obs], ve_gen) if ve_gen else ve[obs]
                st = _matched(v2g1, veg1)
                if st:
                    st["seed"] = seed
                    per_obs[obs].append(st)
                # keep EVERY seed's full trajectory for the overlay plot.
                plot_trajs[obs]["v2"].append(v2[obs])
                plot_trajs[obs]["ve"].append(ve[obs])
        cond_data[cond] = (per_obs, plot_trajs, v2_bounds)
        for obs in OBSERVABLES:
            sts = per_obs[obs]
            if sts:
                med = float(np.median([s["median_rel"] for s in sts]))
                print(f"  {obs:26s} n_seed={len(sts)} median|Δ|gen1={med*100:6.2f}%")
    return cond_data


# --------------------------------------------------------------------------- #
# section builders
# --------------------------------------------------------------------------- #
def _med(sts, key):
    return float(np.median([s[key] for s in sts])) if sts else float("nan")


def overview_section(cond_data: dict) -> dict:
    """High-level SUMMARY MATRIX: rows = conditions, columns = observables.

    Each observable cell = the per-condition matched gen-1 |Δ| (median over seeds
    of median_rel, abs) as a percent, colored by _grade; a leading seeds column
    and a trailing worst-axis condition verdict. Self-contained inline HTML.
    """
    from scripts._compare.report import _e, _verdict_chip, _slug
    # (data-key, display label) for the matrix columns, in render order.
    cols = [("cell_mass", "cell mass"), ("dry_mass", "dry mass"),
            ("protein_mass", "protein mass"), ("rna_mass", "rna mass"),
            ("instantaneous_growth_rate", "growth rate")]
    tok = {"within_tol": "green", "drift": "amber",
           "mismatch": "red", "not_compared": "grey"}
    rank = {"within_tol": 0, "drift": 1, "mismatch": 2}

    def worst(vs: list[str]) -> str:
        g = [v for v in vs if v in rank]
        return max(g, key=lambda v: rank[v]) if g else "not_compared"

    body_rows, cond_verdicts, seed_counts = [], [], set()
    for cond, (per_obs, *_rest) in cond_data.items():
        n_seed = max((len(per_obs.get(o, [])) for o, _ in cols), default=0)
        seed_counts.add(n_seed)
        cells, axis_v = [], []
        for okey, _lbl in cols:
            sts = per_obs.get(okey, [])
            if not sts:
                cells.append('<td style="padding:6px 11px;text-align:center;'
                             'color:var(--grey)">—</td>')
                continue
            rel = abs(_med(sts, "median_rel"))
            g = _grade(rel)
            axis_v.append(g)
            t = tok[g]
            cells.append(
                f'<td style="padding:6px 11px;text-align:right;font-weight:600;'
                f'color:var(--{t});box-shadow:inset 3px 0 0 var(--{t})">'
                f'{rel*100:.1f}%</td>')
        cv = worst(axis_v)
        cond_verdicts.append(cv)
        # Link the row to that condition's "simulation runs" sub-section — it is
        # rendered in BOTH modes (the "config (vEcoli)" panel is SKIPPED in
        # local-pbg / pbg-vs-pbg mode, so anchoring there left the row pointing at
        # a non-existent id and the click did nothing). Use the SAME report._slug on
        # the SAME title string the section is rendered with so the anchor resolves
        # exactly. Whole row clickable (onclick) + the condition cell is a real
        # <a> for an accessible, hover-cued link target.
        anchor = _slug(f"{cond} — simulation runs")
        body_rows.append(
            f'<tr onclick="location.hash=\'#{anchor}\'" style="cursor:pointer">'
            f'<td style="padding:6px 11px;font-weight:650">'
            f'<a href="#{anchor}" style="color:var(--ink);text-decoration:none;'
            f'border-bottom:1px dotted var(--muted)" '
            f'title="jump to {_e(cond)} block">{_e(cond)} &rarr;</a></td>'
            f'<td style="padding:6px 11px;text-align:center;color:var(--muted)">'
            f'{n_seed}</td>{"".join(cells)}'
            f'<td style="padding:6px 11px">{_verdict_chip(cv)}</td></tr>')

    overall = worst(cond_verdicts)
    n_cond = len(cond_data)
    seeds_desc = (f"{next(iter(seed_counts))} seeds × gen-1 each"
                  if len(seed_counts) == 1 and seed_counts != {0}
                  else "per-config seeds (see column)")
    th = ('padding:6px 11px;text-align:right;font-size:11px;'
          'text-transform:uppercase;letter-spacing:.5px;color:var(--muted)')
    head = (f'<th style="{th};text-align:left">config</th>'
            f'<th style="{th};text-align:center">seeds</th>'
            + "".join(f'<th style="{th}">{_e(lbl)}</th>' for _, lbl in cols)
            + f'<th style="{th};text-align:left">config verdict</th>')
    hdr = (f'<p style="margin:0 0 12px;font-size:14px">'
           f'<b>Overall:</b> {_verdict_chip(overall)} '
           f'&middot; {n_cond} config{"s" if n_cond != 1 else ""} '
           f'&middot; {seeds_desc}</p>')
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        f'<thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>')
    return {"title": "Overview — results by config", "kind": "content",
            "desc": "Summary matrix — every config (row) × every measurement "
                    "(column). Each cell is the matched gen-1 |Δ| (median over "
                    "seeds), colored 5% within / 10% drift / else mismatch; "
                    "the private reference repository = baseline. Trailing column "
                    "= that config's worst-axis verdict.",
            "html": hdr + table}


def parca_section(cond_data: dict) -> dict:
    """Per-condition t~0 initial masses for both engines = same-initial-state
    evidence (each sim starts from its engine's ParCa fit)."""
    rows = []
    for cond, (per_obs, *_rest) in cond_data.items():
        rows.append({"label": f"── {cond} ──", "left": "vEcoli init",
                     "right": "v2ecoli init", "verdict": "not_compared",
                     "group": cond, "reason": "first matched emit (t~0), median over seeds"})
        for obs in MASS_OBS:
            sts = per_obs.get(obs, [])
            if not sts:
                rows.append({"label": OBS_LABEL.get(obs, obs), "left": "—",
                             "right": "—", "verdict": "not_compared",
                             "group": cond, "reason": "no paired trajectory"})
                continue
            ive, iv2 = _med(sts, "init_ve"), _med(sts, "init_v2")
            rel = abs((iv2 - ive) / ive) if ive else float("nan")
            rows.append({
                "label": OBS_LABEL.get(obs, obs), "group": cond,
                "left": f"{ive:.4g}", "right": f"{iv2:.4g}",
                "verdict": _grade(rel), "median_rel": rel,
                "reason": f"t~{sts[0]['init_t']:.0f}s · init Δ={(iv2-ive)/ive*100:+.2f}% "
                          f"(median of {len(sts)} seed(s))",
            })
    return {"title": "ParCa / initial-state match (per condition)",
            "desc": "Both engines build their initial cell from their ParCa fit, so "
                    "the t~0 emitted masses ARE the same-initial-state evidence. "
                    "Matched per condition; graded 5%/10%, vEcoli = reference.",
            "rows": rows}


def report_card(cond_data: dict, *, conditions) -> tuple[dict, str, dict]:
    """Grade the 7 standard axes at matched gen-1 timepoints via the canonical
    report_card library. Returns (verdict_json, standalone-card-html, section).

    Per seed, each axis value = the gen-1 matched-grid MEAN; vEcoli list is the
    reference, v2ecoli the measured. build_report_card -> grade_card -> verdict_json
    with 5%/10% ttest bands.
    """
    # pool seeds across the graded conditions (basal-only by default).
    left_by_cell: dict[str, list] = {}   # vEcoli (reference)
    right_by_cell: dict[str, list] = {}  # v2ecoli (measured)
    disp: dict[str, tuple] = {}          # card-key -> (ref_mean, meas_mean) for cols
    for obs in OBSERVABLES:
        ck = CARD_KEY.get(obs, obs)
        ve_vals, v2_vals = [], []
        for cond in conditions:
            for s in cond_data.get(cond, ({},))[0].get(obs, []):
                ve_vals.append(s["ve_mean"])
                v2_vals.append(s["v2_mean"])
        left_by_cell[ck], right_by_cell[ck] = ve_vals, v2_vals
        disp[ck] = (float(np.mean(ve_vals)) if ve_vals else None,
                    float(np.mean(v2_vals)) if v2_vals else None)

    vjson, html = build_report_card(
        left_by_cell, right_by_cell, extra_axes=EXTRA_AXES,
        model_ref="v2ecoli@sim48/59", tol_rel=TOL)

    # build report-section rows straight from the canonical verdict_json.
    _norm = {"ungraded": "not_compared"}
    rows = []
    for gslug, g in (vjson.get("groups") or {}).items():
        for ax in g.get("axes", []):
            # disp is keyed by card-key = leaf of the dotted axis path.
            ref_mean, meas_mean = disp.get(ax["id"].split(".")[-1], (None, None))
            v = _norm.get(ax["verdict"], ax["verdict"])
            rows.append({
                "label": ax["label"], "group": gslug.replace("_", " "),
                "left": "—" if ref_mean is None else f"{ref_mean:.4g}",
                "right": "—" if meas_mean is None else f"{meas_mean:.4g}",
                "verdict": v, "reason": ax.get("meter", ""),
            })
    section = {
        "title": "Statistical-equivalence report card",
        "desc": "Canonical report_card.grade_card verdict over the 7 standard axes "
                "(4 masses + growth rate + active RNAP + active ribosome), graded at "
                "matched gen-1 timepoints (per-seed gen-1 mean; vEcoli = reference; "
                "5% within / 10% drift). Full rendered card saved alongside as "
                "report_card.html; machine-readable verdict in verdict.json.",
        "rows": rows}
    return vjson, html, section


# --------------------------------------------------------------------------- #
# per-condition CONFIG PANELS (restored feature)
#
# For each condition (and variant — these runs are variant=0 only, so one panel
# per condition; the structure carries a variant so multi-variant runs would each
# get a panel) we surface BOTH engines' resolved run configuration as a
# report.config_panel_section content block:
#
#   * vEcoli  — the REAL fully-resolved workflow_config.json that workflow.py ran
#               (s3://.../<ve_dir>/cond_<c>/nextflow/workflow_config.json).
#   * v2ecoli — the compact emitter did NOT persist the full build config, so we
#               fall back to the v2ecoli RUN config recoverable from the zarr
#               lineage-node attrs (condition/seed/time_step/max_duration/...) plus
#               the experiment dir, clearly labelled as the run config (not the
#               full ParCa build config). Nothing is fabricated.
# --------------------------------------------------------------------------- #
def _read_vecoli_config(ve_dir: str, cond: str) -> dict | None:
    """The real resolved workflow_config.json vEcoli's workflow.py ran, from S3.

    Returns None (graceful) if absent. ``workflow_config_stripped.json`` keys are
    surfaced as ``_dropped_vecoli_keys`` (the resolved-only superset shown in a
    collapsible details), so the config panel can show what the stripped form omits.
    """
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    base = f"{PREFIX}/{ve_dir}/cond_{cond}/nextflow"

    def _get(key):
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001  (missing / access — handled by caller)
            return None

    full = _get(f"{base}/workflow_config.json")
    if full is None:
        return None
    stripped = _get(f"{base}/workflow_config_stripped.json") or {}
    extra = {k: full[k] for k in full
             if k not in stripped and not k.startswith("_")}
    if extra:
        full = dict(full)
        full["_dropped_vecoli_keys"] = extra
    return full


def _read_v2ecoli_config(v2_dir: str, seed: int = 0) -> dict | None:
    """v2ecoli RUN config recovered from the compact zarr's lineage-node attrs.

    The compact emitter did not persist the full v2ecoli build/ParCa config, so
    this is the run config (condition, seed, time_step, max_duration, variant,
    generation, experiment_id) read straight from the emitted store — labelled as
    such by the caller. Returns None if the store/attrs are unreadable.
    """
    import xarray as xr
    uri = f"s3://{BUCKET}/{PREFIX}/{v2_dir}/v2ecoli_seed{seed:02d}.zarr"
    try:
        tree = xr.open_datatree(uri, engine="zarr", consolidated=False,
                                storage_options={"anon": False,
                                                 "client_kwargs": {"region_name": REGION}})
    except Exception:  # noqa: BLE001
        return None
    lpaths = [p for p in tree.groups
              if p.split("/")[-1].startswith("lineage_seed=")]
    if not lpaths:
        return None
    attrs = dict(tree[lpaths[0]].attrs)
    # the run-config dict lives under the per-generation gen=<g> attr key.
    cfg = {}
    for k, v in attrs.items():
        if k.startswith("gen=") and isinstance(v, dict):
            cfg.update(v)
    if not cfg:
        return None
    cfg["zarr_store"] = f"{PREFIX}/{v2_dir}/v2ecoli_seed{seed:02d}.zarr"
    return cfg


def _read_v2ecoli_build_config(v2_dir: str) -> dict | None:
    """The resolved v2ecoli build config sidecar emitted by the run.

    ``run_comparison_ensemble.py`` writes ``<out_root>/v2ecoli_build_config.json``
    (process set, per-process config keys, topology, time_step, condition, seed,
    options) straight from the built Composite document. Returns None if absent
    (older runs that predate the sidecar — the caller falls back to zarr attrs).
    """
    # Local run dir first (pbg-vs-pbg / local-pbg modes write the sidecar
    # straight to <v2_dir>); fall back to S3 for cloud runs.
    local = os.path.join(v2_dir, "v2ecoli_build_config.json")
    if os.path.exists(local):
        try:
            with open(local) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    key = f"{PREFIX}/{v2_dir}/v2ecoli_build_config.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:  # noqa: BLE001  (missing / access — handled by caller)
        return None


def _fmt_cfg_val(v) -> str:
    """Compact, HTML-safe one-liner for a config value (big containers summarised)."""
    from scripts._compare.report import _e
    if isinstance(v, dict):
        return _e(f"dict[{len(v)} keys]") if len(v) > 6 else _e(json.dumps(v))
    if isinstance(v, list):
        return _e(f"list[{len(v)}]") if len(v) > 6 else _e(json.dumps(v))
    s = json.dumps(v)
    return _e(s if len(s) <= 90 else s[:87] + "…")


def config_diff_section(cond: str, ve_cfg: dict | None, v2_cfg: dict | None,
                        v2_source: str) -> dict:
    """Per-condition CONFIG DIFF (vEcoli resolved vs v2ecoli build config).

    Renders ``config_adapter.schema_diff`` as a verdict-styled table: shared keys
    that DIFFER (red — the candidates that could explain the ~8% growth
    divergence), shared keys that MATCH (green), and keys present in only one
    engine (grey). The dominant signal is usually the disjoint key SETS: the two
    engines are configured from largely independent namespaces (vEcoli workflow
    knobs vs the v2ecoli composite-build defaults), which is exactly what the
    auto-translation goal (config_adapter.translate_vecoli_config) closes.
    """
    from scripts._compare.config_adapter import schema_diff
    title = f"Config diff — {cond}"
    if not ve_cfg or not v2_cfg:
        missing = "vEcoli" if not ve_cfg else "v2ecoli"
        return {"title": title, "kind": "content",
                "desc": f"Config diff for '{cond}'.",
                "html": f"<p style='color:#6b7280'>diff unavailable — {missing} "
                        f"config not found.</p>"}
    d = schema_diff(ve_cfg, v2_cfg)
    different = d["different"]
    shared = set(ve_cfg) & set(v2_cfg)
    matched = sorted(shared - set(different))

    def _row(tok, key, left, right, note=""):
        return (f'<tr style="box-shadow:inset 3px 0 0 var(--{tok})">'
                f'<td style="padding:4px 10px;font-weight:600">{key}</td>'
                f'<td style="padding:4px 10px;color:#334155">{left}</td>'
                f'<td style="padding:4px 10px;color:#334155">{right}</td>'
                f'<td style="padding:4px 10px;color:#6b7280;font-size:12px">{note}</td>'
                f'</tr>')

    rows = []
    for k in sorted(different):
        vv, v2v = different[k]
        structural = isinstance(vv, (dict, list)) or isinstance(v2v, (dict, list))
        note = "structural — compared in the config panels above" if structural else \
               "DIFFERS — candidate divergence driver"
        rows.append(_row("red", k, _fmt_cfg_val(vv), _fmt_cfg_val(v2v), note))
    for k in matched:
        rows.append(_row("green", k, _fmt_cfg_val(ve_cfg[k]), _fmt_cfg_val(v2_cfg[k]),
                         "match"))
    for k in d["only_in_vecoli"]:
        rows.append(_row("grey", k, _fmt_cfg_val(ve_cfg[k]), "—",
                         "vEcoli-only (v2ecoli sets this internally / via the ParCa cache)"))
    for k in d["only_in_v2"]:
        rows.append(_row("grey", k, "—", _fmt_cfg_val(v2_cfg[k]), "v2ecoli-only"))

    n_scalar_diff = sum(1 for k in different
                        if not isinstance(different[k][0], (dict, list))
                        and not isinstance(different[k][1], (dict, list)))
    summary = (f'<p style="margin:0 0 10px">'
               f'<b>{len(matched)}</b> shared keys match, '
               f'<b>{len(different)}</b> shared keys differ '
               f'(<b>{n_scalar_diff}</b> scalar — the divergence candidates), '
               f'<b>{len(d["only_in_vecoli"])}</b> only in vEcoli, '
               f'<b>{len(d["only_in_v2"])}</b> only in v2ecoli.</p>')
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        '<thead><tr style="text-align:left;color:#6b7280">'
        '<th style="padding:4px 10px">key</th>'
        '<th style="padding:4px 10px">vEcoli</th>'
        '<th style="padding:4px 10px">v2ecoli</th>'
        '<th style="padding:4px 10px">note</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>')
    return {
        "title": title, "kind": "content",
        "desc": f"config_adapter.schema_diff of vEcoli's resolved workflow_config "
                f"vs the v2ecoli build config ({v2_source}) for '{cond}'. Red = "
                f"shared-but-different (divergence candidates); green = match; grey "
                f"= present in only one engine. Large structural values (process "
                f"set, topology) are summarised — see the config panels above.",
        "html": summary + table,
    }


def config_sections_for(cond: str, v2_dir: str, ve_dir: str) -> list[dict]:
    """The vEcoli + v2ecoli config panels + the config-diff panel for ONE
    condition (variant 0), in render order — dropped straight into that
    condition's block (config -> runs -> evaluation)."""
    sections: list[dict] = []

    ve_cfg = _read_vecoli_config(ve_dir, cond)
    if ve_cfg is not None:
        sec = report.config_panel_section(ve_cfg)
        sec["title"] = f"{cond} — config (vEcoli)"
        sec["desc"] = (
            f"vEcoli resolved workflow_config.json (the full config workflow.py "
            f"ran) for condition '{cond}', from s3://.../{ve_dir}/cond_{cond}/"
            f"nextflow/. Keys present only in the resolved (non-stripped) form "
            f"are tucked into the collapsible details.")
        sections.append(sec)
    else:
        sections.append({
            "title": f"{cond} — config (vEcoli)", "kind": "content",
            "desc": f"vEcoli config for condition '{cond}'.",
            "html": f"<p style='color:#6b7280'>config not found for {cond} "
                    f"(no workflow_config.json under {ve_dir}/cond_{cond}).</p>"})

    # The v2ecoli config panel itself is rendered by v2_config_section (every
    # mode); here we only resolve v2_cfg/source for the config-DIFF below.
    v2_build = _read_v2ecoli_build_config(v2_dir)
    if v2_build is not None:
        v2_cfg, v2_source = v2_build, "build-config sidecar"
    else:
        v2_cfg = _read_v2ecoli_config(v2_dir)
        v2_source = "zarr-attrs run config (sidecar absent)"

    # CONFIG DIFF panel — vEcoli resolved vs v2ecoli build config.
    diff = config_diff_section(cond, ve_cfg, v2_cfg, v2_source)
    diff["title"] = f"{cond} — config diff"
    sections.append(diff)
    return sections


def _process_interfaces(v2_build: dict | None) -> dict[str, dict]:
    """Map process name AND address → its resolved interface ({inputs,outputs})
    from the build-config sidecar, so a converted process can be looked up by
    whichever identifier the injection spec used."""
    out: dict[str, dict] = {}
    for p in (v2_build or {}).get("processes") or []:
        iface = p.get("interface")
        if not iface:
            continue
        for k in (p.get("name"), p.get("address")):
            if k:
                out[k] = iface
    return out


def _schema_table(iface: dict | None) -> str:
    """Render a converted process's RESULTING process-bigraph schema: each
    input/output port → its resolved type. Returns '' when unavailable."""
    from scripts._compare.report import _e
    if not iface:
        return ("<div style='color:#6b7280;font-size:12px;padding:2px 0 6px'>"
                "resulting schema unavailable (process instance not captured).</div>")

    def _rows(ports: dict, kind: str) -> str:
        if not ports:
            return (f"<tr><td style='color:#6b7280'>{kind}</td>"
                    f"<td style='color:#6b7280' colspan='2'>(none)</td></tr>")
        rs = []
        for i, (port, typ) in enumerate(ports.items()):
            tstr = typ if isinstance(typ, str) else json.dumps(typ, default=str)
            rs.append(f"<tr><td style='color:#6b7280'>{kind if i == 0 else ''}</td>"
                      f"<td style='padding:2px 10px'><code>{_e(port)}</code></td>"
                      f"<td style='padding:2px 10px;color:#334155'>"
                      f"<code>{_e(tstr if len(tstr) <= 80 else tstr[:77] + '…')}</code>"
                      f"</td></tr>")
        return "".join(rs)
    return ("<table style='border-collapse:collapse;font-size:12px;margin:2px 0 8px'>"
            "<tbody>"
            + _rows(iface.get("inputs") or {}, "inputs")
            + _rows(iface.get("outputs") or {}, "outputs")
            + "</tbody></table>")


def _transferred_source(fork_repo: str, proc_name: str):
    """``(rel_path, source_code)`` for a transferred fork process, read DIRECTLY
    from the fork checkout WITHOUT importing it.

    Importing the fork's process stack (to resolve the class) pulls in jax /
    gillespy2 / ray, which (a) leaves non-daemon threads that wedge the report
    process at exit and (b) resolves the INSTALLED ecoli's copy (shadow), not the
    fork's. Instead, map the registry name to a source file heuristically and read
    the fork file — so the embedded code is the EXACT vEcoli-private code that was
    transferred. Best-effort: ``(None, None)`` if no matching module is found."""
    try:
        import glob as _glob
        fork_abs = os.path.abspath(os.path.expanduser(fork_repo))
        # registry name -> module stem: "ecoli-metabolism-redux" -> "metabolism_redux"
        stem = proc_name.split("/")[-1]
        for pre in ("ecoli-", "ecoli_"):
            if stem.startswith(pre):
                stem = stem[len(pre):]
        stem = stem.replace("-", "_")
        cands = _glob.glob(os.path.join(fork_abs, "ecoli", "**", stem + ".py"),
                           recursive=True)
        for src in cands:
            code_txt = Path(src).read_text(encoding="utf-8")
            # confirm it defines a process/step class (avoid an unrelated match)
            if "class " in code_txt and any(
                    k in code_txt for k in ("Process", "Step", "ports_schema",
                                            "next_update", "calculate_request")):
                return os.path.relpath(src, fork_abs), code_txt
        return None, None
    except Exception:  # noqa: BLE001 — never break the report over a source read
        return None, None


def _source_block(name: str, fork: str, rel: str, code: str) -> str:
    """A labeled, scrollable code block embedding a transferred process's full source."""
    import html as _html
    lines = code.count("\n") + 1
    return (
        f"<div style='margin-top:12px'>"
        f"<div style='font-size:12px;color:#374151;margin:4px 0'>"
        f"<b>{_html.escape(name)}</b> — full transferred source "
        f"(<code>{_html.escape(rel)}</code>, {lines} lines, from fork "
        f"<code>{_html.escape(fork)}</code>):</div>"
        f"<pre style='max-height:560px;overflow:auto;background:#0d1117;color:#e6edf3;"
        f"padding:12px;border-radius:6px;font-size:12px;line-height:1.45;margin:0'>"
        f"<code>{_html.escape(code)}</code></pre></div>")


def _git_provenance(path: str | None) -> dict | None:
    """{name, lineage-agnostic git facts} for a repo checkout, or None. ``name``
    prefers the origin remote's repo slug (e.g. ``sms-ecoli``, ``vEcoli-private``)
    and falls back to the directory basename."""
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return None
    import subprocess

    def _g(*args: str) -> str:
        try:
            r = subprocess.run(["git", "-C", path, *args],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    remote = _g("config", "--get", "remote.origin.url")
    slug = ""
    if remote:
        slug = remote.rstrip("/").split("/")[-1]
        if slug.endswith(".git"):
            slug = slug[:-4]
    return {
        "name": slug or os.path.basename(path),
        "branch": _g("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _g("rev-parse", "--short", "HEAD"),
        "remote": remote,
        "path": path,
    }


def repositories_section(candidate: dict | None, reference: dict | None) -> dict:
    """A top-of-report panel naming the two ACTUAL repositories being compared,
    with commits — so the reader sees `sms-ecoli` / `vEcoli-private`, not just the
    generic `v2ecoli` / `vEcoli` lineage labels."""
    def _row(role: str, lineage: str, prov: dict | None) -> str:
        if not prov:
            return (f"<tr><td>{role}</td><td colspan='4'>"
                    f"<em>unknown (no checkout resolved)</em></td></tr>")
        rurl = prov.get("remote") or ""
        rcell = (f"<a href='{report._e(rurl)}'>{report._e(rurl)}</a>"
                 if rurl.startswith("http") else report._e(rurl or "—"))
        return (f"<tr><td>{role}</td>"
                f"<td><b>{report._e(prov['name'])}</b> "
                f"<span style='color:#6b7280'>({lineage})</span></td>"
                f"<td><code>{report._e(prov.get('branch') or '?')}</code></td>"
                f"<td><code>{report._e(prov.get('commit') or '?')}</code></td>"
                f"<td style='font-size:12px'>{rcell}</td></tr>")
    table = ("<table><thead><tr><th>role</th><th>repository</th><th>branch</th>"
             "<th>commit</th><th>remote</th></tr></thead><tbody>"
             + _row("Candidate (measured)", "v2ecoli", candidate)
             + _row("Reference (baseline)", "vEcoli", reference)
             + "</tbody></table>")
    cand_n = (candidate or {}).get("name", "the candidate")
    ref_n = (reference or {}).get("name", "the reference")
    # The section `desc` is HTML-escaped by the renderer, so it must be plain text;
    # the rich prose (bold repo names, &Delta;) goes in `html`, which renders raw.
    intro = (
        f"<p style='margin:0 0 12px;line-height:1.5'>This report compares "
        f"<b>{report._e(cand_n)}</b> (the v2ecoli-lineage <em>candidate</em>) against "
        f"<b>{report._e(ref_n)}</b> (the vEcoli-lineage <em>reference</em>). Every "
        f"&Delta; below is the candidate measured against the reference at matched "
        f"gen-1 timepoints. Both engines run as process-bigraph composites.</p>")
    return {
        "title": "Repositories compared",
        "kind": "content",
        "nav_group": "Overall",
        "desc": (f"Comparing {cand_n} (candidate, v2ecoli lineage) against "
                 f"{ref_n} (reference, vEcoli lineage); each delta is candidate vs "
                 f"reference at matched gen-1 timepoints."),
        "html": intro + table,
    }


def how_to_read_section(candidate: dict | None, reference: dict | None) -> dict:
    """Plain-language orientation for reviewers unfamiliar with the codebase
    (security / compliance audience). States the report's two purposes, what it
    is NOT (a full repository diff), and how to read each config block. All rich
    text lives in ``html`` (the renderer escapes ``desc``)."""
    cand_n = (candidate or {}).get("name", "this repository")
    ref_n = (reference or {}).get("name", "the private reference repository")
    body = (
        f"<p style='margin:0 0 14px;line-height:1.55;font-size:14px'>"
        f"This report has <strong>two purposes</strong>:</p>"
        f"<ol style='margin:0 0 16px;padding-left:22px;line-height:1.55;font-size:14px'>"
        f"<li style='margin-bottom:8px'><strong>Show the code that was transferred</strong> "
        f"from the private repository (<b>{report._e(ref_n)}</b>) into "
        f"<b>{report._e(cand_n)}</b>. Each transferred process is shown with its "
        f"<strong>full source code</strong>, embedded inline in that config's section below.</li>"
        f"<li style='margin-bottom:0'><strong>Demonstrate the behavioral impact</strong> "
        f"of those transfers, by running both repositories in simulation and comparing "
        f"their outputs through <strong>report cards</strong> (pass / drift / mismatch "
        f"verdicts on standard whole-cell measurements).</li></ol>"
        f"<div style='background:var(--card,#f8fafc);border-left:4px solid #f59e0b;"
        f"padding:12px 16px;margin:0 0 16px;border-radius:4px;font-size:13.5px;line-height:1.55'>"
        f"<strong>What this report is NOT:</strong> it is <strong>not a full repository "
        f"diff</strong>. It does not enumerate every changed file. It focuses on the "
        f"transferred simulation <em>processes</em> (the biological model code) and the "
        f"measurable effect of running them.</div>"
        f"<p style='margin:0 0 8px;line-height:1.55;font-size:14px'>"
        f"<strong>How to read it:</strong></p>"
        f"<ul style='margin:0;padding-left:22px;line-height:1.55;font-size:13.5px'>"
        f"<li style='margin-bottom:6px'>Each section below is one <strong>config</strong> "
        f"(a configuration file that defines a simulation scenario). "
        f"<code>baseline</code> / <code>basal</code> is the plain wild-type cell with no "
        f"transferred code &mdash; the control.</li>"
        f"<li style='margin-bottom:6px'>A config either <strong>transfers a process</strong> "
        f"(its source is embedded in that section) or only <strong>changes settings / the "
        f"genome</strong> (no new code) &mdash; each section states which.</li>"
        f"<li style='margin-bottom:6px'>Within a config: the <strong>transferred source "
        f"code</strong> comes first, then the <strong>simulation runs</strong>, then the "
        f"<strong>report-card evaluation</strong> comparing the two repositories.</li>"
        f"<li style='margin-bottom:0'>Verdict colors: "
        f"<span style='color:#16a34a;font-weight:700'>&#9679; within tolerance</span> &middot; "
        f"<span style='color:#d97706;font-weight:700'>&#9679; drift</span> &middot; "
        f"<span style='color:#dc2626;font-weight:700'>&#9679; mismatch</span>.</li></ul>")
    return {
        "title": "Overview — how to read this report",
        "kind": "content",
        "nav_group": "Overall",
        "desc": ("Start here. This report shows (1) the simulation code transferred from "
                 "the private repository into this one, and (2) the behavioral impact of "
                 "those transfers via simulation report cards. It is not a full repository diff."),
        "html": body,
    }


def converted_processes_section(cond: str, v2_build: dict | None) -> dict | None:
    """Surface the fork processes converted + injected into the v2ecoli composite,
    with each one's RESULTING process-bigraph schema (resolved port types) AND the
    full transferred source code, read from the fork checkout.

    Sourced from the build-config sidecar's ``options.overrides.injected_processes``
    (written by run_comparison_ensemble): the add_processes + swap_processes that
    the bridge auto-converted from vivarium-1.0 to process-bigraph and wired into
    baseline. The resulting schema comes from each process's captured
    ``interface`` (inputs()/outputs()) in the sidecar's ``processes`` list.
    Returns None when the run injected nothing.
    """
    inj = (((v2_build or {}).get("options") or {}).get("overrides") or {}).get(
        "injected_processes")
    if not inj:
        return None
    adds = list(inj.get("add_processes") or [])
    swaps = dict(inj.get("swap_processes") or {})
    if not adds and not swaps:
        return None
    fork = inj.get("fork_repo", "?")
    ifaces = _process_interfaces(v2_build)
    rows = []

    def _card(name: str, mode: str, note: str) -> str:
        return (f"<tr><td><code>{name}</code></td><td>{mode}</td><td>{note}</td>"
                f"<td>vivarium-1.0 → process-bigraph</td></tr>"
                f"<tr><td colspan='4' style='padding:0 0 6px 18px'>"
                f"<div style='font-size:12px;color:#6b7280;margin:2px 0'>"
                f"resulting schema:</div>{_schema_table(ifaces.get(name))}</td></tr>")
    for name in adds:
        rows.append(_card(name, "add", "—"))
    for old, new in swaps.items():
        rows.append(_card(new, "swap", f"replaces <code>{old}</code>"))
    table = ("<table><thead><tr><th>process</th><th>mode</th><th>note</th>"
             "<th>conversion</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")
    # Embed the FULL transferred source for every converted process, read from the
    # fork checkout — so the report always carries the exact code that ran. Dedupe
    # by source file (co-located processes share a module).
    src_blocks: list[str] = []
    seen_files: set[str] = set()
    for name in list(adds) + list(swaps.values()):
        rel, code = _transferred_source(fork, name)
        if code and rel not in seen_files:
            seen_files.add(rel)
            src_blocks.append(_source_block(name, fork, rel, code))
    return {
        "title": f"{cond} — transferred process code (from vEcoli-private)",
        "kind": "content",
        "nav_group": "Config",
        "desc": (f"Fork (vEcoli) processes auto-converted by the v2ecoli bridge "
                 f"(<code>wrap_vivarium_process</code>) and injected into the "
                 f"v2ecoli composite for '{cond}', from fork <code>{fork}</code>. "
                 f"Each row shows the conversion plus the RESULTING process-bigraph "
                 f"schema (resolved input/output port types). Each ran in the "
                 f"v2ecoli engine of this comparison. The FULL transferred source "
                 f"code of each converted process is embedded below, read from the "
                 f"fork checkout."),
        "html": table + "".join(src_blocks),
    }


def v2_config_section(cond: str, v2_build: dict | None) -> dict | None:
    """The FULL resolved v2ecoli build config for ONE condition, rendered from a
    build-config dict — process set, per-process config keys, topology wiring,
    time_step, and the build options. Renders in EVERY mode (it needs no
    Nextflow config), so a run that injected processes shows its real config
    instead of reading as a bare condition label. Returns None without a build."""
    if not v2_build:
        return None
    sec = report.config_panel_section(v2_build)
    sec["title"] = f"{cond} — config (v2ecoli)"
    sec["nav_group"] = cond
    procs = v2_build.get("processes") or []
    inj = (((v2_build.get("options") or {}).get("overrides") or {})
           .get("injected_processes") or {})
    injected = list(inj.get("add_processes") or []) + \
        list((inj.get("swap_processes") or {}).values())
    note = (f" Includes {len(injected)} injected process"
            f"{'es' if len(injected) != 1 else ''}: "
            f"{', '.join(injected)}." if injected else "")
    sec["desc"] = (
        f"v2ecoli RESOLVED build config for '{cond}' — {len(procs)} processes "
        f"actually built into the composite (with per-process config keys + "
        f"topology), time_step {v2_build.get('time_step')}, from the "
        f"v2ecoli_build_config.json sidecar emitted off the built Composite.{note}")
    return sec


def vecoli_config_section(cond: str, ve_build: dict | None) -> dict | None:
    """The FULL resolved vEcoli config for ONE condition, rendered from the
    vecoli_build_config.json sidecar (the EcoliSim.config the genuine-vEcoli pbg
    run actually used — its real process set, exclude list, media, time_step).
    Renders in EVERY mode, so the report shows what vEcoli truly ran (e.g. its
    default baseline, when the v2 side injected a v2-only process). Returns None
    without a build."""
    if not ve_build:
        return None
    sec = report.config_panel_section(ve_build)
    sec["title"] = f"{cond} — config (vEcoli)"
    sec["nav_group"] = cond
    n = len(ve_build.get("processes") or []) + len(ve_build.get("steps") or [])
    sec["desc"] = (
        f"vEcoli RESOLVED config for '{cond}' — {n} processes+steps the "
        f"genuine-vEcoli engine actually ran (from EcoliSim.config; vEcoli runs "
        f"most mechanism as partitioned requester/evolver steps), time_step "
        f"{ve_build.get('time_step')}, media {ve_build.get('media_id')}, captured "
        f"into the vecoli_build_config.json sidecar. This is vEcoli's OWN config; "
        f"compare it against the v2ecoli config panel to see any divergence.")
    return sec


def _read_vecoli_build_config(ve_dir: str) -> dict | None:
    """The resolved vEcoli config sidecar (vecoli_build_config.json) the
    genuine-vEcoli pbg run emits next to its zarr. Local dir first, then S3.
    Returns None when absent (older runs that predate the sidecar)."""
    local = os.path.join(ve_dir, "vecoli_build_config.json")
    if os.path.exists(local):
        try:
            with open(local) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    key = f"{PREFIX}/{ve_dir}/vecoli_build_config.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:  # noqa: BLE001
        return None


def runs_section(cond: str, per_obs: dict, plot_trajs: dict,
                 v2_bounds: list[float]) -> dict:
    """Simulation-runs block for ONE condition: all-replicate trajectory
    overlays (every seed of BOTH engines), full trajectories, gen boundaries."""
    n_seeds = max((len(per_obs.get(o, [])) for o in OBSERVABLES), default=0)
    plots = [_legend_html(n_seeds)]
    for obs in OBSERVABLES:
        pt = plot_trajs.get(obs, {"v2": [], "ve": []})
        if not pt["v2"] or not pt["ve"]:
            continue
        svg = overlay_svg_multi(pt["v2"], pt["ve"], obs, vlines=v2_bounds)
        plots.append(
            f'<div style="display:inline-block;width:48%;vertical-align:top;'
            f'margin:0 1% 8px 0"><div style="font-size:12px;font-weight:600;'
            f'color:#334155">{OBS_LABEL.get(obs, obs)}</div>{svg}</div>')
    return {
        "title": f"{cond} — simulation runs", "kind": "content",
        "desc": f"All replicate trajectories for '{cond}' — every seed of both "
                f"engines overlaid (vEcoli indigo, v2ecoli amber), full trajectory "
                f"across all generations on a shared seconds axis; dashed verticals "
                f"mark generation boundaries.",
        "html": '<div style="padding:8px 18px 14px">' + "".join(plots) + "</div>"}


def eval_section(cond: str, per_obs: dict) -> dict:
    """Comparison-evaluation block for ONE condition: per-observable matched
    verdict rows (median |Δ| over the matched gen-1 window, ALL seeds)."""
    rows = []
    for obs in OBSERVABLES:
        sts = per_obs.get(obs, [])
        if not sts:
            rows.append({"label": OBS_LABEL.get(obs, obs), "left": "—",
                         "right": "—", "verdict": "not_compared",
                         "reason": "no paired trajectory (observable absent in "
                                   "one engine's emitter)"})
            continue
        med = _med(sts, "median_rel")
        ive, iv2 = _med(sts, "init_ve"), _med(sts, "init_v2")
        irel = (iv2 - ive) / ive if ive else float("nan")
        rows.append({
            "label": OBS_LABEL.get(obs, obs), "left": f"{ive:.4g}",
            "right": f"{iv2:.4g}", "verdict": _grade(med),
            "median_rel": med, "max_rel": _med(sts, "max_rel"),
            "reason": f"t~{sts[0]['init_t']:.0f}s init Δ={irel*100:+.2f}%; gen-1 "
                      f"matched median |Δ|={med*100:.2f}% (median of {len(sts)} seed)",
        })
    return {
        "title": f"{cond} — evaluation",
        "desc": "vEcoli (left) vs v2ecoli (right) initial values; verdict from the "
                "gen-1 matched-time median |Δ| over ALL seeds. Graded 5% within / "
                "10% drift; vEcoli = reference.",
        "rows": rows}


def _find_new_genes(obj):
    """Best-effort recursive search for a ``new_genes`` value in a fork config."""
    if isinstance(obj, dict):
        if obj.get("new_genes"):
            return obj["new_genes"]
        for v in obj.values():
            r = _find_new_genes(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_new_genes(v)
            if r:
                return r
    return None


def _condition_summary(spec, v2_build):
    """What a condition's config changes vs the plain baseline: transferred
    process swaps/adds (real code transfer) and any genome variant (new_genes).
    ``transfers`` is True only when a PROCESS was injected."""
    cfg = str(getattr(spec, "config", "") or "")
    inj = (((v2_build or {}).get("options") or {}).get("overrides") or {}).get(
        "injected_processes") or {}
    swaps = dict(inj.get("swap_processes") or {})
    adds = list(inj.get("add_processes") or [])
    new_genes = None
    fork = os.environ.get("V2E_VECOLI_DIR")
    if fork and cfg.endswith(".json"):
        try:
            new_genes = _find_new_genes(json.load(
                open(os.path.join(os.path.expanduser(fork), cfg))))
        except Exception:  # noqa: BLE001
            pass
    return {"config": cfg or "baseline (built-in condition)", "swaps": swaps,
            "adds": adds, "new_genes": new_genes, "transfers": bool(swaps or adds)}


def _transfer_phrase(summary):
    """One-line HTML describing what a condition transfers/changes."""
    if summary["transfers"]:
        parts = [f"<code>{report._e(o)}</code>&rarr;<code>{report._e(n)}</code>"
                 for o, n in summary["swaps"].items()]
        parts += [f"add <code>{report._e(a)}</code>" for a in summary["adds"]]
        txt = "<b>transfers process:</b> " + ", ".join(parts)
    else:
        txt = "<span style='color:#6b7280'>no process transferred</span>"
    if summary["new_genes"]:
        txt += f" · genome variant <code>{report._e(summary['new_genes'])}</code>"
    return txt


def conditions_map_section(summaries):
    """Overview table: each CONDITION == a CONFIG, and what it transfers/changes.
    Makes the condition↔config equivalence explicit up front."""
    rows = []
    for name, s in summaries.items():
        rows.append(
            f"<tr><td><b>{report._e(name)}</b></td>"
            f"<td><code>{report._e(s['config'])}</code></td>"
            f"<td>{_transfer_phrase(s)}</td></tr>")
    table = ("<table><thead><tr><th>config</th><th>config file</th>"
             "<th>what it transfers / changes vs baseline</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")
    return {
        "title": "Configs in this report",
        "kind": "content",
        "nav_group": "Overall",
        "desc": ("Each row is one config (a configuration file defining a simulation "
                 "scenario), run on both repositories. A config either transfers a "
                 "process (its full source is embedded in that config's section below) "
                 "or only changes the genome / settings. 'baseline' is the plain "
                 "wild-type cell with no transferred code — the control."),
        "html": table,
    }


def _condition_banner(name, summary):
    """A prominent divider that starts each condition's section, so the boundary
    between conditions is obvious while scrolling."""
    html = (
        f"<div style='border-top:4px solid #2563eb;margin-top:8px;padding:14px 0 2px'>"
        f"<div style='font-size:12px;letter-spacing:.08em;color:#2563eb;"
        f"font-weight:700'>CONFIG</div>"
        f"<div style='font-size:22px;font-weight:800;margin:2px 0'>"
        f"{report._e(name)}</div>"
        f"<div style='font-size:13px'>config file: <code>{report._e(summary['config'])}"
        f"</code> &nbsp;·&nbsp; {_transfer_phrase(summary)}</div></div>")
    return {"title": f"▸ {name}", "kind": "content", "nav_group": name,
            "desc": "", "html": html}


def assemble_from_studies(specs, cond_data, conds, verdict_root=None,
                          studies_root=None):
    """Overview + per-study assigned-card sections, driven by study specs (the
    study-YAML-only model — no manifest). Each StudySpec carries name (store key),
    condition, seeds/gens and cards; the `config` card renders the study's run
    spec. Writes one report_card_verdict.json per study (each card a group). When
    verdict_root is None it is derived from the studies' investigation. Per-study
    cards are written under the top-level study registry (``studies_root/name``);
    when studies_root is None it defaults to ``STUDIES_ROOT``
    (``workspace/studies``) — every reader (aggregate.py, study.yaml
    report_cards, dashboard embeds) resolves cards from that top-level dir, not
    the nested per-investigation layout."""
    from scripts._compare.study_spec import STUDIES_ROOT
    from scripts._compare.verdict import write_condition_verdict
    from scripts._compare.viz_cards import write_report_cards
    if studies_root is None:
        studies_root = STUDIES_ROOT
    core = build_core()
    if verdict_root is None and specs:
        verdict_root = f"docs/report_cards/{specs[0].invest_name}"
    overview = overview_section(cond_data); overview["nav_group"] = "Overall"
    sections = [overview]
    # Per-condition summaries (each condition IS a config) + the condition↔config map.
    summaries = {}
    for spec in specs:
        if spec.name in cond_data:
            v2d = conds.get(spec.name, ("", ""))[0]
            summaries[spec.name] = _condition_summary(
                spec, _read_v2ecoli_build_config(v2d))
    if summaries:
        sections.append(conditions_map_section(summaries))
    for spec in specs:
        name = spec.name
        if name not in cond_data:
            print(f"[assemble] skip study {name!r}: no store under --out", flush=True)
            continue
        per_obs, plot_trajs, v2_bounds = cond_data[name]
        v2_dir, ve_dir = conds.get(name, ("", ""))
        # Prominent divider so each condition's block is obvious while scrolling.
        sections.append(_condition_banner(name, summaries[name]))
        state = {"name": name, "condition": spec.condition, "seeds": spec.seeds,
                 "generations": spec.gens, "variant": 0, "observables": per_obs,
                 "plot_trajs": plot_trajs, "v2_bounds": v2_bounds,
                 # config-specific bulk KPIs the study declared (for the bulk_kpi card)
                 "observable_bulk_ids": list(getattr(spec, "observable_bulk_ids", []) or []),
                 "config": {"condition": spec.condition, "seeds": spec.seeds,
                            "generations": spec.gens, "cards": spec.cards,
                            "observable_bulk_ids": list(getattr(spec, "observable_bulk_ids", []) or []),
                            # WHICH QUANTITY the exchange_flux leaves carry. A card
                            # that normalises a flux by dry mass MUST know this: on
                            # "gdcw" the leaf is already mmol/gDCW/h and normalising
                            # again divides by dry mass twice, which grades cleanly
                            # and is wrong. Threaded from the spec so the study's
                            # single declaration reaches the grading layer too.
                            "exchange_flux_basis": str(
                                getattr(spec, "exchange_flux_basis", None) or "counts")},
                 "v2_dir": v2_dir, "ve_dir": ve_dir}
        card_verdicts, viz = {}, []
        for card in spec.cards:
            step = core.link_registry[f"{card}_report_card"](config={}, core=core)
            out = step.update(state)
            sections.append({"title": f"{name} — {card}", "kind": "content",
                             "html": out["card_html"], "nav_group": name})
            card_verdicts[card] = {"verdict": out.get("verdict", "ungraded"),
                                   "axes": out.get("axes") or []}
            viz.append({"name": card, "html": out["card_html"],
                        "verdict": card_verdicts[card]["verdict"],
                        "axes": card_verdicts[card]["axes"]})
        # Converted-processes panel — for any study whose candidate injected fork
        # processes (e.g. metabolism_redux), surface the conversion AND embed the
        # FULL transferred source read from the fork checkout. No-op otherwise.
        conv = converted_processes_section(name, _read_v2ecoli_build_config(v2_dir))
        if conv is not None:
            conv["nav_group"] = name
            sections.append(conv)
        if verdict_root:
            write_condition_verdict(verdict_root, name, card_verdicts)
        if studies_root:
            write_report_cards(Path(studies_root) / name, viz)
    return sections


def _load_experiments(out_dir: str) -> dict | None:
    """Read condition -> [v2_dir, ve_dir] written by comparison_harness.sh launch.

    Returns None if no out/<...>/experiments.json exists. Accepts either the
    ``{"conditions": {...}, ...}`` wrapper the harness writes or a bare map.
    """
    p = Path(out_dir) / "experiments.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("conditions", data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="basal",
                   help="comma-separated conditions to include, or 'all' "
                        "(default basal — the only valid v2<->vE comparison until "
                        "sim59-v2fix-* lands)")
    p.add_argument("--conditions-json", default=None,
                   help='override condition dirs as JSON: '
                        '{"basal":["<v2_dir>","<ve_dir>"], ...}')
    p.add_argument("--local-pbg", default=None,
                   help='pbg-vs-pbg mode: JSON {cond:[v2_zarr_path, ve_zarr_path]} '
                        'of LOCAL v2ecoli-format zarr stores (BOTH engines as '
                        'process-bigraph composites). Reads both via read_pbg_local '
                        'and emits the overview + report-card + per-condition '
                        'trajectory/eval sections (skips the S3/Nextflow-only panels).')
    p.add_argument("--local-pbg-dir", default=None,
                   help='MULTI-SEED local pbg-vs-pbg mode: JSON {cond:[v2_dir, ve_dir]} '
                        'of LOCAL directories each holding per-seed v2ecoli-format zarr '
                        '(v2ecoli_seed<NN>.zarr / vecoli_seed<NN>.zarr). Reads seeds '
                        '0..--local-pbg-seeds-1 via read_pbg_local. Use for local 4x4x5.')
    p.add_argument("--local-pbg-seeds", type=int, default=4,
                   help="number of seeds (0..N-1) for --local-pbg-dir (default 4).")
    p.add_argument("--pbg-vs-pbg", action="store_true",
                   help='S3 pbg-vs-pbg mode (the default upstream-wrapper route): '
                        'BOTH engines emit v2ecoli-format zarr to S3 — v2 under '
                        'v2ecoli_seed*.zarr, vecoli under vecoli_seed*.zarr in its '
                        'own experiment dir. Reads both via the zarr reader from S3 '
                        '(condition dirs from experiments.json / --conditions-json) '
                        'and skips the Nextflow-only config panels. Use this to '
                        'render the standardized report for any pbg-vs-pbg run.')
    p.add_argument("-o", "--out", default="out/full_compare",
                   help="output directory (also read for experiments.json)")
    p.add_argument("--investigation", default=None,
                   help="path or name of an investigation (study-YAML-only mode): "
                        "renders each study's assigned cards and writes one "
                        "report_card_verdict.json per study.")
    p.add_argument("--study", default=None,
                   help="with --investigation, render only this study (by name).")
    args = p.parse_args(argv)

    local_pbg = bool(args.local_pbg)
    local_pbg_dir = bool(args.local_pbg_dir)
    pbg_vs_pbg = bool(args.pbg_vs_pbg)
    investigation_mode = bool(args.investigation)
    # these modes drop the S3/Nextflow-only panels (vecoli side is a pbg zarr, not
    # a Nextflow workflow_config.json); local reads from disk, pbg_vs_pbg from S3.
    skip_nextflow = (local_pbg or local_pbg_dir or pbg_vs_pbg
                     or investigation_mode)
    if investigation_mode:                            # -2. study-YAML-only
        from scripts._compare.study_spec import load_investigation as _load_inv
        _ctx, _specs = _load_inv(args.investigation)
        if args.study:
            _specs = [s for s in _specs if s.name == args.study]
            if not _specs:
                raise SystemExit(f"study {args.study!r} not in investigation")
        conds = {s.name: (f"{args.out}/{s.name}", f"{args.out}/{s.name}")
                 for s in _specs}
    elif local_pbg_dir:                               # 0b. multi-seed local dirs
        conds = json.loads(args.local_pbg_dir)
    elif local_pbg:                                   # 0. pbg-vs-pbg local zarr
        conds = json.loads(args.local_pbg)
    elif args.conditions_json:                        # 1. explicit override
        conds = json.loads(args.conditions_json)
    else:                                             # 2. launch's experiments.json
        conds = _load_experiments(args.out)
        if conds is not None:
            print(f"Using launched experiment ids from {args.out}/experiments.json")
        else:                                         # 3. baked-in CONDITIONS default
            conds = {k: list(v) for k, v in CONDITIONS.items()}
    conds = {k: tuple(v) for k, v in conds.items()}
    # In manifest mode the manifest's `configs` ARE the selection — don't let
    # --only (default 'basal') silently drop the other configs.
    if not investigation_mode and args.only.strip().lower() != "all":
        want = [c.strip() for c in args.only.split(",") if c.strip()]
        conds = {k: conds[k] for k in want if k in conds}
    if not conds:
        raise SystemExit("no conditions selected")
    print(f"Conditions: {list(conds)}\n")

    if local_pbg_dir or investigation_mode:
        # MULTI-SEED local: each cond maps to [v2_dir, ve_dir]; read per-seed
        # v2ecoli-format zarr (v2ecoli_seed<NN>.zarr / vecoli_seed<NN>.zarr) for
        # seeds 0..N-1 through the same local reader. Enables the local 4x4x5
        # and manifest-driven runs whose per-config dirs live under args.out.
        cond_data = build(
            conds, seeds=list(range(args.local_pbg_seeds)),
            read_v2=lambda d, s: read_pbg_local(f"{d}/v2ecoli_seed{s:02d}.zarr", OBSERVABLES),
            read_ve=lambda d, s: read_pbg_local(f"{d}/vecoli_seed{s:02d}.zarr", OBSERVABLES))
    elif local_pbg:
        # Both engines as process-bigraph composites → both read v2ecoli-format
        # zarr from a LOCAL path; the json maps each cond to a single seed's
        # stores, so the per-seed loop runs once.
        cond_data = build(
            conds, seeds=[0],
            read_v2=lambda d, s: read_pbg_local(d, OBSERVABLES),
            read_ve=lambda d, s: read_pbg_local(d, OBSERVABLES))
    elif pbg_vs_pbg:
        # The default upstream-wrapper route: BOTH engines emit v2ecoli-format
        # zarr to S3 (v2 -> v2ecoli_seed*.zarr, vecoli -> vecoli_seed*.zarr),
        # each in its own experiment dir. Read both via the zarr reader from S3.
        cond_data = build(
            conds,
            read_v2=lambda d, s: read_v2ecoli_trajectory(d, s, OBSERVABLES),
            read_ve=lambda d, s: read_vecoli_pbg_trajectory(d, s, OBSERVABLES))
    else:
        cond_data = build(conds)
    graded = list(conds)  # conditions whose data feeds the report card

    # 1. OVERALL RESULTS first — the canonical report-card render + the
    #    per-condition verdict-count overview. Both collapse into ONE "Overall"
    #    nav entry (nav_group), so the sticky nav stays at ~7 top-level buttons.
    # report_card() still emits the canonical verdict.json + standalone
    # report_card.html artifacts, but its rendered SECTION is intentionally NOT
    # shown at the top: it grades only the graded subset (≈basal) on the gen-1
    # MEAN value — a narrower scope + different metric than the Overview matrix
    # (all conditions × observables, matched |Δ|), so showing both reads as
    # contradictory. The Overview matrix is the high-level view; verdict.json
    # stays as the machine-readable artifact for tooling/dashboard.
    vjson, card_html, _card_section = report_card(cond_data, conditions=graded)
    if investigation_mode:
        # Study-YAML-only: per-study card assignment via assemble_from_studies.
        sections = assemble_from_studies(_specs, cond_data, conds)
    else:
        overview = overview_section(cond_data)
        overview["nav_group"] = "Overall"
        sections = [overview]
        # 2. ParCa / initial-state comparison — its own nav entry. Renders for ALL
        #    modes (incl. pbg-vs-pbg): it reads the t~0 emitted masses straight from
        #    the loaded trajectories (cond_data), NOT any Nextflow config — and in the
        #    matched-initial-state comparison it IS the headline evidence (both engines
        #    start from identical molecule counts → t~0 masses agree).
        parca = parca_section(cond_data)
        parca["nav_group"] = "ParCa comparison"
        sections.append(parca)
        # 3. One block PER CONDITION, fixed order, each reading top-to-bottom as
        #    config -> simulation runs -> evaluation. All of a condition's sections
        #    share nav_group=<cond>, so they fold into ONE nav button per condition.
        #    The config panels read S3/Nextflow workflow_config.json, so they are
        #    omitted in local-pbg mode (both engines are local pbg composites).
        for cond in conds:
            v2_dir, ve_dir = conds[cond]
            per_obs, plot_trajs, v2_bounds = cond_data[cond]
            v2_build = _read_v2ecoli_build_config(v2_dir)
            cond_sections = [runs_section(cond, per_obs, plot_trajs, v2_bounds),
                             eval_section(cond, per_obs)]
            # Converted-processes panel (with each converted process's RESULTING
            # schema) AND the full v2ecoli config panel render in EVERY mode —
            # both read the v2ecoli build-config sidecar, not any Nextflow config.
            # So a 'basal' run that injected processes shows its real process set
            # and the conversions, not just the bare condition label.
            conv = converted_processes_section(cond, v2_build)
            if conv is not None:
                cond_sections = [conv] + cond_sections
            v2cfg = v2_config_section(cond, v2_build)
            if v2cfg is not None:
                cond_sections = [v2cfg] + cond_sections
            # vEcoli config panel — every mode, from the vecoli sidecar (what
            # genuine vEcoli actually ran). So both engines show full config.
            ve_build = _read_vecoli_build_config(ve_dir)
            vecfg = vecoli_config_section(cond, ve_build)
            if vecfg is not None:
                cond_sections = [vecfg] + cond_sections
            # vEcoli Nextflow config panel + config diff — only when the
            # workflow_config.json is available (genuine S3/Nextflow runs).
            if not skip_nextflow:
                cond_sections = (config_sections_for(cond, v2_dir, ve_dir)
                                 + cond_sections)
            for s in cond_sections:
                s["nav_group"] = cond
            sections += cond_sections

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Name the ACTUAL repositories (e.g. sms-ecoli / vEcoli-private), not just the
    # generic v2ecoli / vEcoli lineage labels. Candidate = the v2ecoli-lineage
    # checkout this script runs from; reference = the vEcoli fork at V2E_VECOLI_DIR.
    cand = _git_provenance(str(Path(__file__).resolve().parents[1]))
    ref = _git_provenance(os.environ.get("V2E_VECOLI_DIR"))
    # Reviewer orientation FIRST (plain-language purpose + how-to-read), then the
    # repositories panel, then the results. Prevents the report being misread as a
    # full repository diff.
    sections = [how_to_read_section(cand, ref),
                repositories_section(cand, ref)] + sections
    cand_lbl = f"{cand['name']} (v2ecoli)" if cand else "v2ecoli"
    ref_lbl = f"{ref['name']} (vEcoli)" if ref else "vEcoli"
    title = f"{cand_lbl} ↔ {ref_lbl} — whole-cell model comparison ({gen})"
    html = report.render_report(sections, title=title)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep_path = out / "standardized_comparison_report.html"
    rep_path.write_text(html, encoding="utf-8")
    (out / "report_card.html").write_text(card_html, encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(vjson, indent=2), encoding="utf-8")
    dl = Path.home() / "Downloads" / "v2ecoli_vecoli_standard_report.html"
    shutil.copy(rep_path, dl)

    print(f"\nReport card overall: {vjson['overall']}")
    print("Saved:")
    print(f"  {rep_path}")
    print(f"  {out / 'report_card.html'}")
    print(f"  {out / 'verdict.json'}")
    print(f"  {dl}")


if __name__ == "__main__":
    main()
    # Force a clean exit: embedding the transferred fork source imports the fork's
    # process stack (jax / gillespy2 / ray), which can leave non-daemon threads
    # that keep the process alive after the report is written — which then blocks
    # the study→study orchestration. The report is fully flushed above, so exit now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
