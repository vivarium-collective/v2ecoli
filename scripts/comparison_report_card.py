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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._compare import report
from scripts._compare.report_card_section import build_report_card
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
     "label": "Dry mass", "key": "dry_mass"},
    {"group": "Physiology", "path": "physiology.protein_mass",
     "label": "Protein mass", "key": "protein_mass"},
    {"group": "Physiology", "path": "physiology.rna_mass",
     "label": "RNA mass", "key": "rna_mass"},
    {"group": "Ribosomes", "path": "ribosomes.active_ribosome",
     "label": "Active ribosome", "key": "active_ribosome"},
    {"group": "Transcription", "path": "transcription.active_RNAP",
     "label": "Active RNAP", "key": "active_RNAP"},
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
                  else "per-condition seeds (see column)")
    th = ('padding:6px 11px;text-align:right;font-size:11px;'
          'text-transform:uppercase;letter-spacing:.5px;color:var(--muted)')
    head = (f'<th style="{th};text-align:left">condition</th>'
            f'<th style="{th};text-align:center">seeds</th>'
            + "".join(f'<th style="{th}">{_e(lbl)}</th>' for _, lbl in cols)
            + f'<th style="{th};text-align:left">condition verdict</th>')
    hdr = (f'<p style="margin:0 0 12px;font-size:14px">'
           f'<b>Overall:</b> {_verdict_chip(overall)} '
           f'&middot; {n_cond} condition{"s" if n_cond != 1 else ""} '
           f'&middot; {seeds_desc}</p>')
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        f'<thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>')
    return {"title": "Overview — all conditions", "kind": "content",
            "desc": "Summary matrix — every condition (row) × every observable "
                    "(column). Each cell is the matched gen-1 |Δ| (median over "
                    "seeds), colored 5% within / 10% drift / else mismatch; "
                    "vEcoli = reference. Trailing column = the condition's "
                    "worst-axis verdict.",
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

    # Prefer the new build-config sidecar (full resolved process set +
    # topology straight from the Composite); fall back to the zarr-attrs run
    # config for older runs that predate it. Label which one is shown.
    v2_build = _read_v2ecoli_build_config(v2_dir)
    if v2_build is not None:
        v2_cfg, v2_source = v2_build, "build-config sidecar"
        sec = report.config_panel_section(v2_cfg)
        sec["title"] = f"{cond} — config (v2ecoli)"
        sec["desc"] = (
            f"v2ecoli RESOLVED build config for condition '{cond}', from the "
            f"v2ecoli_build_config.json sidecar the run emitted straight off the "
            f"built Composite (process set, per-process config keys, topology, "
            f"time_step, condition, seed, options).")
        sections.append(sec)
    else:
        v2_cfg = _read_v2ecoli_config(v2_dir)
        v2_source = "zarr-attrs run config (sidecar absent)"
        if v2_cfg is not None:
            sec = report.config_panel_section(v2_cfg)
            sec["title"] = f"{cond} — config (v2ecoli)"
            sec["desc"] = (
                f"v2ecoli RUN config for condition '{cond}', recovered from the "
                f"compact zarr's lineage-node attrs (the build-config sidecar was "
                f"absent — this is the run config: condition, seed, time_step, "
                f"max_duration, variant, generation).")
            sections.append(sec)
        else:
            v2_cfg = None
            sections.append({
                "title": f"{cond} — config (v2ecoli)", "kind": "content",
                "desc": f"v2ecoli run config for condition '{cond}'.",
                "html": f"<p style='color:#6b7280'>config not found for {cond} "
                        f"(no sidecar and no recoverable run config in "
                        f"{v2_dir} zarr attrs).</p>"})

    # CONFIG DIFF panel — vEcoli resolved vs v2ecoli build config.
    diff = config_diff_section(cond, ve_cfg, v2_cfg, v2_source)
    diff["title"] = f"{cond} — config diff"
    sections.append(diff)
    return sections


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


def assemble_from_manifest(manifest, cond_data, conds, config_names,
                           verdict_root="docs/report_cards/v2ecoli-vecoli-comparison"):
    """Overview + per-config assigned-card sections, mirroring the manifest.

    config_names maps a manifest config path -> the condition key used in
    cond_data/conds (the runner names stores by condition). When verdict_root is
    set, also writes one report_card_verdict.json per rendered condition (each
    report card becomes a group), so the comparison can gate via report_card_axis.
    """
    from scripts._compare import report_cards as rc
    from scripts._compare.verdict import write_condition_verdict
    default_cards = (manifest.get("defaults", {}) or {}).get("cards") or ["standard"]
    import json as _json
    import os as _os
    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def _load_cfg(p):
        # Read the manifest's config file deterministically (repo-relative or
        # absolute) so the `config` card is ALWAYS populated. _source records
        # which path resolved.
        for cand in (p, _os.path.join(_repo, p)):
            try:
                with open(cand, encoding="utf-8") as _fh:
                    d = _json.load(_fh)
                d["_source"] = cand
                return d
            except (OSError, ValueError):
                continue
        return {}

    overview = overview_section(cond_data); overview["nav_group"] = "Overall"
    sections = [overview]
    for entry in manifest.get("configs", []):
        name = config_names[entry["config"]]
        if name not in cond_data:
            continue
        per_obs, plot_trajs, v2_bounds = cond_data[name]
        v2_dir, ve_dir = conds.get(name, ("", ""))
        _cfg = _load_cfg(entry["config"])
        ctx = rc.CardContext(config_name=name, variant=0, v2_dir=v2_dir,
                             ve_dir=ve_dir, seeds=int(_cfg.get("n_init_sims") or 0),
                             gens=int(_cfg.get("generations") or 0), per_obs=per_obs,
                             plot_trajs=plot_trajs, v2_bounds=v2_bounds, config=_cfg)
        card_verdicts = {}
        for card in (entry.get("cards") or default_cards):
            cardv = None
            for sec in rc.render(card, ctx):
                sec["nav_group"] = name
                sections.append(sec)
                if "verdict_axes" in sec:
                    # last-graded-section-wins; relies on the one-graded-section-per-card invariant
                    cardv = {"verdict": sec.get("verdict", "ungraded"),
                             "axes": sec["verdict_axes"]}
            card_verdicts[card] = cardv or {"verdict": "ungraded", "axes": []}
        if verdict_root:
            write_condition_verdict(verdict_root, name, card_verdicts)
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
    p.add_argument("--manifest", default=None,
                   help="path to a comparison manifest JSON; when set, loads the "
                        "manifest and routes assembly through assemble_from_manifest "
                        "(per-config card assignment). --only still filters configs.")
    args = p.parse_args(argv)

    local_pbg = bool(args.local_pbg)
    local_pbg_dir = bool(args.local_pbg_dir)
    pbg_vs_pbg = bool(args.pbg_vs_pbg)
    manifest_mode = bool(args.manifest)
    # both modes drop the S3/Nextflow-only panels (vecoli side is a pbg zarr, not
    # a Nextflow workflow_config.json); local reads from disk, pbg_vs_pbg from S3.
    skip_nextflow = local_pbg or local_pbg_dir or pbg_vs_pbg or manifest_mode
    if manifest_mode:                                 # -1. manifest-driven
        import os as _os
        from scripts._compare.config_adapter import store_key
        manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        _fork = _os.environ.get("V2E_VECOLI_DIR", "")

        # Section/store key: uses canonical store_key (honors explicit manifest `name`,
        # then fork-resolved condition, then filename stem) so runner/renderer/scaffold
        # always agree.
        _config_names = {_entry["config"]: store_key(_entry, _fork)
                         for _entry in manifest_data.get("configs", [])}
        # In manifest mode both engines' stores (v2ecoli_seed*.zarr and
        # vecoli_seed*.zarr) live in the SAME per-condition dir under args.out —
        # the same layout as --local-pbg-dir's 4x4x5 run.
        conds = {_n: (f"{args.out}/{_n}", f"{args.out}/{_n}")
                 for _n in _config_names.values()}
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
    if not manifest_mode and args.only.strip().lower() != "all":
        want = [c.strip() for c in args.only.split(",") if c.strip()]
        conds = {k: conds[k] for k in want if k in conds}
    if not conds:
        raise SystemExit("no conditions selected")
    print(f"Conditions: {list(conds)}\n")

    if local_pbg_dir or manifest_mode:
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
    if manifest_mode:
        # Manifest-driven: per-config card assignment via assemble_from_manifest.
        sections = assemble_from_manifest(manifest_data, cond_data, conds,
                                          _config_names)
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
            cond_sections = [runs_section(cond, per_obs, plot_trajs, v2_bounds),
                             eval_section(cond, per_obs)]
            if not skip_nextflow:
                cond_sections = (config_sections_for(cond, v2_dir, ve_dir)
                                 + cond_sections)
            for s in cond_sections:
                s["nav_group"] = cond
            sections += cond_sections

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    title = (f"vEcoli-pbg ↔ v2ecoli-pbg — process-bigraph comparison ({gen})"
             if skip_nextflow
             else f"v2ecoli ↔ vEcoli — standardized comparison ({gen})")
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
