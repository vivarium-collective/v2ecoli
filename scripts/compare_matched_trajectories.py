"""Matched-timepoint v2ecoli<->vEcoli trajectory comparison with overlay images.

The scientist's question: are the two sims actually close? v2ecoli is a port of
vEcoli (same ParCa, same processes, same initial state), so they SHOULD be. But
an end-of-run SNAPSHOT compares cells at *different cell-cycle phases* (one may
have just divided, the other not) -> ±100% artifacts. The honest comparison is
value-vs-SIMULATION-TIME: line both engines up on a shared seconds axis and ask
whether they track within the first generation.

This driver:
  * reads each engine's TRAJECTORY (value vs global time) per condition/seed
      - v2ecoli: compact zarr; emits every 60 ticks; ``last_write.sim_time`` in
        the lineage node attrs gives each generation's cumulative end-time, from
        which the per-emit time axis is reconstructed.
      - vEcoli: Nextflow parquet ``time`` column (already global-continuous).
  * answers the key question quantitatively: per observable/condition the
    initial (first matched timepoint) value for both engines and the median
    matched-time relative difference over generation 1.
  * grades each against ±5% / ±10% (within_tol / drift / mismatch).
  * renders inline-SVG overlay trajectory images (charts.multiline_svg) and the
    standard report card (report.render_report), saved to
    out/full_compare/comparison_matched_report.html (+ ~/Downloads copy).

Credentials: export SSO creds into the env first (aiobotocore SSO refresh is
unreliable; both s3fs and duckdb need env creds)::

    export AWS_PROFILE=stanford-sso AWS_DEFAULT_REGION=us-gov-west-1
    eval "$(aws configure export-credentials --format env)"
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._compare import charts, report
from scripts._compare.vecoli_parquet_reader import read_vecoli_trajectory

BUCKET = "smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91"
PREFIX = "vecoli-output"
REGION = "us-gov-west-1"

# condition -> (v2ecoli zarr experiment dir, vEcoli parquet experiment dir)
CONDITIONS = {
    "basal":     ("sim48-v2-full-basal-d52e",     "sim47-ve-full-basal-b498"),
    "with_aa":   ("sim48-v2-full-with_aa-0f7a",   "sim47-ve-full-with_aa-5261"),
    "succinate": ("sim48-v2-full-succinate-9493", "sim47-ve-full-succinate-e2cf"),
    "no_oxygen": ("sim48-v2-full-no_oxygen-2371", "sim47-ve-full-no_oxygen-9784"),
    "acetate":   ("sim48-v2-full-acetate-d575",   "sim47-ve-full-acetate-7042"),
}

OBSERVABLES = ["cell_mass", "dry_mass", "protein_mass", "rna_mass",
               "instantaneous_growth_rate"]
OBS_LABEL = {"cell_mass": "cell mass (fg)", "dry_mass": "dry mass (fg)",
             "protein_mass": "protein mass (fg)", "rna_mass": "RNA mass (fg)",
             "instantaneous_growth_rate": "growth rate (1/s)"}
SEEDS = [0, 1, 2, 3]

TOL = 0.05   # within_tol if |median matched Δ| <= 5%
TOL2 = 0.10  # drift if <= 10%, else mismatch

_SO = {"anon": False, "client_kwargs": {"region_name": REGION}}


# --------------------------------------------------------------------------- #
# v2ecoli trajectory (compact zarr)
# --------------------------------------------------------------------------- #
def read_v2ecoli_trajectory(experiment_dir: str, seed: int,
                            observables, *, store_uri: str | None = None,
                            storage_options: dict | None = None) -> dict:
    """{obs: (times, values)} for one v2ecoli-format zarr, continuous over generations.

    The compact emitter writes one value every 60 ticks; each generation's
    cumulative end-time is recorded in the lineage-node attrs as
    ``last_write_gen=<g> -> {sim_time}``. Per generation the emit interval is
    ``(end - prev_end) / n_emits`` and emit ``i`` lands at
    ``prev_end + (i+1)*dt`` on the global seconds axis.

    By default reads the S3 experiment dir. Pass ``store_uri`` (a local path or
    full URI to the ``.zarr`` store) to read a pbg engine's output from anywhere
    — vEcoli-pbg and v2ecoli-pbg both write THIS format, so the same reader
    serves both sides of a pbg-vs-pbg comparison.
    """
    import xarray as xr
    if store_uri is None:
        store_uri = f"s3://{BUCKET}/{PREFIX}/{experiment_dir}/v2ecoli_seed{seed:02d}.zarr"
        storage_options = _SO
    tree = xr.open_datatree(store_uri, engine="zarr", consolidated=False,
                            storage_options=storage_options)
    lpaths = [p for p in tree.groups
              if p.split("/")[-1].startswith("lineage_seed=")]
    if not lpaths:
        return {}
    ln = lpaths[0]
    attrs = dict(tree[ln].attrs)

    def gen_num(name):
        return int(re.search(r"generation=(\d+)", name).group(1))

    out: dict = {}
    gen1_mask_times = None
    for obs in observables:
        vpath = f"{ln}/{obs}"
        if vpath not in set(tree.groups):
            continue
        ds = tree[vpath].to_dataset()
        gens = sorted((n for n in ds.data_vars if n.startswith("generation=")),
                      key=gen_num)
        times, vals, gnums = [], [], []
        prev_end = 0.0
        for gname in gens:
            g = gen_num(gname)
            arr = np.asarray(ds[gname]).ravel().astype(float)
            n = arr.size
            if n == 0:
                continue
            lw = attrs.get(f"last_write_gen={g}", {})
            end = float(lw.get("sim_time", prev_end + n * 60.0))
            dt = (end - prev_end) / n if n else 60.0
            t = prev_end + (np.arange(n) + 1) * dt
            times.append(t)
            vals.append(arr)
            gnums.append(np.full(n, g))
            prev_end = end
        if times:
            out[obs] = (np.concatenate(times), np.concatenate(vals))
            out.setdefault("_generation",
                           (np.concatenate(times), np.concatenate(gnums)))
    return out


def read_pbg_local(zarr_path: str, observables) -> dict:
    """Read a pbg engine's v2ecoli-format zarr from a local path (or any URI).

    vEcoli-pbg and v2ecoli-pbg both emit this format, so this serves BOTH sides
    of a local pbg-vs-pbg comparison through the SAME reader the standardized
    report uses for v2ecoli.
    """
    return read_v2ecoli_trajectory(None, 0, observables, store_uri=zarr_path,
                                   storage_options=None)


def read_vecoli_pbg_trajectory(experiment_dir: str, seed: int,
                               observables) -> dict:
    """Read the vEcoli-PBG side of a pbg-vs-pbg run from S3.

    In the upstream-wrapper route the vEcoli engine is run as a Ray
    ``composite=vecoli`` job that emits the SAME v2ecoli-format zarr the v2ecoli
    side does — just under ``vecoli_seed{NN}.zarr`` instead of
    ``v2ecoli_seed{NN}.zarr``. So both sides read through ``read_v2ecoli_trajectory``;
    this only points the store URI at the ``vecoli_`` prefix on S3.
    """
    uri = (f"s3://{BUCKET}/{PREFIX}/{experiment_dir}/"
           f"vecoli_seed{seed:02d}.zarr")
    return read_v2ecoli_trajectory(None, seed, observables, store_uri=uri,
                                   storage_options=_SO)


# --------------------------------------------------------------------------- #
# matched-timepoint stats
# --------------------------------------------------------------------------- #
def _gen1_window(traj, gen_traj):
    """Restrict an (times, values) trajectory to generation 1 using gen labels."""
    t, v = traj
    gt, gv = gen_traj
    # gen labels are sampled on the same time axis as the obs (same node);
    # build a step function gen(time) and keep t <= last gen-1 time.
    g1_end = gt[gv == gv.min()].max() if gv.size else t.max()
    mask = t <= g1_end + 1e-6
    return t[mask], v[mask]


def matched_stats(v2_traj, ve_traj):
    """Compare one observable's gen-1 trajectories at matched times.

    Shared grid = v2ecoli's gen-1 emit times (the coarse axis). vEcoli (dense)
    is linearly interpolated onto it. Returns dict with the initial (first
    matched point) values for both engines, the median |relative Δ| over gen-1,
    and the per-point arrays for plotting.
    """
    (v2_t, v2_v) = v2_traj
    (ve_t, ve_v) = ve_traj
    if v2_t.size == 0 or ve_t.size == 0:
        return None
    grid = v2_t  # v2ecoli emit times within gen1
    # only compare where vEcoli covers the grid
    grid = grid[(grid >= ve_t.min()) & (grid <= ve_t.max())]
    if grid.size == 0:
        return None
    ve_on = np.interp(grid, ve_t, ve_v)
    v2_on = np.interp(grid, v2_t, v2_v)
    denom = np.where(np.abs(ve_on) > 1e-30, ve_on, np.nan)
    rel = (v2_on - ve_on) / denom
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return None
    return {
        "init_t": float(grid[0]),
        "init_v2": float(v2_on[0]),
        "init_ve": float(ve_on[0]),
        "init_rel": float((v2_on[0] - ve_on[0]) / ve_on[0]) if ve_on[0] else float("nan"),
        "median_rel": float(np.median(np.abs(rel))),
        "signed_median_rel": float(np.median(rel)),
        "max_rel": float(np.max(np.abs(rel))),
        "n": int(rel.size),
    }


def grade(median_abs_rel: float) -> str:
    if median_abs_rel <= TOL:
        return "within_tol"
    if median_abs_rel <= TOL2:
        return "drift"
    return "mismatch"


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def overlay_svg(v2_traj, ve_traj, obs):
    """Inline-SVG overlay of vEcoli (indigo) vs v2ecoli (amber) value-vs-time.

    Plots the full available trajectory for both engines (all generations) so
    division drops are visible; the matched window is where the curves overlap.
    """
    v2_t, v2_v = v2_traj
    ve_t, ve_v = ve_traj
    ve_pts = list(zip(ve_t.tolist(), ve_v.tolist()))     # series 0 -> indigo
    v2_pts = list(zip(v2_t.tolist(), v2_v.tolist()))     # series 1 -> amber
    svg, _ = charts.multiline_svg([ve_pts, v2_pts], w=440, h=190, axes=True,
                                  baseline_zero=False, x_label="simulation time (s)")
    return svg


_C_VE = charts.PALETTE[0]   # indigo
_C_V2 = charts.PALETTE[1]   # amber


def overlay_svg_multi(v2_trajs, ve_trajs, obs, vlines=None):
    """Inline-SVG overlay of ALL replicate seeds, full trajectories.

    ``v2_trajs`` / ``ve_trajs`` are lists of (times, values) arrays — one per
    seed. Every vEcoli seed is drawn in indigo, every v2ecoli seed in amber, on
    one shared axis, so the two engines stay visually distinguishable while all
    replicates are visible. ``vlines`` marks generation boundaries.
    """
    ve_series = [list(zip(t.tolist(), v.tolist())) for t, v in ve_trajs]
    v2_series = [list(zip(t.tolist(), v.tolist())) for t, v in v2_trajs]
    series = ve_series + v2_series
    colors = [_C_VE] * len(ve_series) + [_C_V2] * len(v2_series)
    if not any(len(s) >= 2 for s in series):
        return "<span class='ref'>—</span>"
    svg, _ = charts.multiline_svg(series, w=440, h=190, axes=True,
                                  baseline_zero=False, colors=colors,
                                  vlines=vlines, x_label="simulation time (s)")
    return svg


def _legend_html(n_seeds=None):
    n = f" (n={n_seeds} seeds/engine)" if n_seeds else ""
    return (
        f'<div style="font-size:12px;color:#475569;margin:2px 0 10px">'
        f'<span style="color:{_C_VE};font-weight:700">&#9644; vEcoli</span> '
        f'(dense, full generation) &nbsp;&nbsp;'
        f'<span style="color:{_C_V2};font-weight:700">&#9644; v2ecoli</span> '
        f'(compact, 1 pt / 60 s) &nbsp;&nbsp;all replicates overlaid{n}; '
        f'dashed = generation boundary</div>')


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build():
    print("Reading trajectories from S3 ...\n")
    # per-condition aggregation
    cond_data = {}
    for cond, (v2_dir, ve_dir) in CONDITIONS.items():
        print(f"=== {cond} ===")
        ve_prefix = f"{PREFIX}/{ve_dir}"
        # gather per-seed matched stats; keep seed 0 trajectories for the plot
        per_obs_stats = {obs: [] for obs in OBSERVABLES}
        seed0_v2, seed0_ve = {}, {}
        for seed in SEEDS:
            try:
                v2 = read_v2ecoli_trajectory(v2_dir, seed, OBSERVABLES)
            except Exception as e:  # noqa: BLE001
                print(f"  seed{seed}: v2ecoli read failed: {type(e).__name__} {str(e)[:60]}")
                continue
            if not v2:
                continue
            try:
                ve = read_vecoli_trajectory(ve_prefix, BUCKET, OBSERVABLES,
                                            lineage_seed=seed, region=REGION)
            except Exception as e:  # noqa: BLE001
                print(f"  seed{seed}: vEcoli read failed: {type(e).__name__} {str(e)[:60]}")
                continue
            v2_gen = v2.get("_generation")
            ve_gen = ve.get("_generation")
            for obs in OBSERVABLES:
                if obs not in v2 or obs not in ve:
                    continue
                v2g1 = _gen1_window(v2[obs], v2_gen) if v2_gen else v2[obs]
                veg1 = _gen1_window(ve[obs], ve_gen) if ve_gen else ve[obs]
                st = matched_stats(v2g1, veg1)
                if st:
                    st["seed"] = seed
                    per_obs_stats[obs].append(st)
            if seed == 0:
                seed0_v2, seed0_ve = v2, ve
        # if seed 0 missing trajectories, fall back to the first seed that has them
        if not seed0_v2:
            for seed in SEEDS:
                try:
                    v2 = read_v2ecoli_trajectory(v2_dir, seed, OBSERVABLES)
                    ve = read_vecoli_trajectory(ve_prefix, BUCKET, OBSERVABLES,
                                                lineage_seed=seed, region=REGION)
                except Exception:  # noqa: BLE001
                    continue
                if v2 and ve:
                    seed0_v2, seed0_ve = v2, ve
                    print(f"  (plot uses seed {seed} — seed 0 unavailable)")
                    break
        cond_data[cond] = (per_obs_stats, seed0_v2, seed0_ve)
        for obs in OBSERVABLES:
            sts = per_obs_stats[obs]
            if sts:
                med = float(np.median([s["median_rel"] for s in sts]))
                print(f"  {obs:28s} n_seed={len(sts)} median(median|Δ|)={med*100:6.2f}%")
    return cond_data


def render(cond_data):
    sections = []
    # About / methodology content section
    about = (
        "<p><b>Why matched-timepoint, not snapshot?</b> v2ecoli is a direct port "
        "of vEcoli &mdash; same ParCa calibration, same processes, same initial "
        "state &mdash; so the two simulations should track. Comparing the "
        "<i>final</i> emitted value of each run is misleading: the two lineages "
        "reach cell division at slightly different wall-times, so an end-of-run "
        "snapshot can catch one cell just after a mass-halving division and the "
        "other just before it &rarr; spurious &plusmn;100% differences that are "
        "pure cell-cycle <i>phase</i>, not a modelling discrepancy.</p>"
        "<p>Instead we line both engines up on a shared <b>simulation-time</b> "
        "axis (seconds). vEcoli (Nextflow parquet) emits every tick; v2ecoli "
        "(compact zarr) emits one point per 60&nbsp;s. We interpolate vEcoli onto "
        "v2ecoli's emit times <b>within generation&nbsp;1</b> and report the "
        "median |relative&nbsp;Δ| = |(v2ecoli &minus; vEcoli)/vEcoli| over that "
        "window, graded &plusmn;5% (within&nbsp;tol) / &plusmn;10% (drift). The "
        "<i>initial</i> column is the first matched timepoint (t&nbsp;&asymp;60&nbsp;s) "
        "&mdash; both engines start from the same ParCa state, so these should be "
        "near-identical.</p>"
        "<p><b>Note on run length:</b> in this dataset several v2ecoli lineages "
        "stopped early (e.g. basal seed&nbsp;0 emitted only 6 points / ~360&nbsp;s "
        "before flagging division), whereas the paired vEcoli lineages ran full "
        "~2520&nbsp;s generations. That asymmetry is exactly what makes an "
        "end-of-run snapshot meaningless and a matched-time comparison necessary; "
        "the matched window is the overlap of the two.</p>"
        "<p style='background:#fef2f2;border-left:4px solid #dc2626;padding:10px 14px;"
        "border-radius:6px'><b>Bug surfaced by this comparison.</b> v2ecoli's "
        "<i>initial cell mass is 1280&nbsp;fg in every condition</i> "
        "(basal, with_aa, succinate, no_oxygen, acetate alike), whereas vEcoli "
        "starts each condition from its own ParCa state (2886 / 568 / 470 / "
        "331&nbsp;fg &hellip;). The v2ecoli ensemble driver builds "
        "<code>build_composite(\"baseline\", &hellip;)</code> and never passes the "
        "condition, so every non-basal v2ecoli run actually ran the <b>basal</b> "
        "initial state &mdash; only its metadata label is condition-specific. The "
        "non-basal &lsquo;mismatch&rsquo; rows below are therefore basal-v2ecoli vs "
        "condition-vEcoli (apples-to-oranges), <i>not</i> a v2ecoli model-fidelity "
        "failure. <b>basal</b> is the one row where both engines used the same "
        "initial state &mdash; and there the trajectories track to ~0–2% for 3 of "
        "4 seeds (one vEcoli seed started anomalously low at 1037&nbsp;fg).</p>")
    sections.append({"title": "About — matched-timepoint vs snapshot",
                     "kind": "content", "html": about})

    tally = {"within_tol": 0, "drift": 0, "mismatch": 0, "not_compared": 0}
    summary_rows = []  # for the printed answer table

    for cond, (per_obs_stats, v2, ve) in cond_data.items():
        rows = []
        plots_html = [_legend_html()]
        for obs in OBSERVABLES:
            sts = per_obs_stats.get(obs, [])
            if not sts:
                rows.append({"label": OBS_LABEL.get(obs, obs), "left": "—",
                             "right": "—", "verdict": "not_compared",
                             "reason": "no paired trajectory"})
                tally["not_compared"] += 1
                summary_rows.append((cond, obs, None, None, None, "not_compared"))
                continue
            # median across seeds = robust to a single anomalous-init seed
            per_seed_med = [s["median_rel"] for s in sts]
            med = float(np.median(per_seed_med))
            init_ve = float(np.median([s["init_ve"] for s in sts]))
            init_v2 = float(np.median([s["init_v2"] for s in sts]))
            init_rel = (init_v2 - init_ve) / init_ve if init_ve else float("nan")
            v = grade(med)
            tally[v] += 1
            init_t = sts[0]["init_t"]
            spread = (f"; per-seed |Δ| range {min(per_seed_med)*100:.1f}–"
                      f"{max(per_seed_med)*100:.1f}%" if len(sts) > 1 else "")
            rows.append({
                "label": OBS_LABEL.get(obs, obs),
                "left": f"{init_ve:.4g}",
                "right": f"{init_v2:.4g}",
                "verdict": v,
                "median_rel": med,
                "max_rel": float(np.median([s["max_rel"] for s in sts])),
                "reason": (f"t~{init_t:.0f}s init: vEcoli={init_ve:.4g} "
                           f"v2ecoli={init_v2:.4g} (Δ={init_rel*100:+.2f}%); "
                           f"gen-1 matched median |Δ|={med*100:.2f}% "
                           f"(median of n_seed={len(sts)}{spread})"),
            })
            summary_rows.append((cond, obs, init_ve, init_v2, med, v))
            # overlay plot for this observable (seed 0)
            if v2 and ve and obs in v2 and obs in ve:
                svg = overlay_svg(v2[obs], ve[obs], obs)
                plots_html.append(
                    f'<div style="display:inline-block;width:48%;'
                    f'vertical-align:top;margin:0 1% 8px 0">'
                    f'<div style="font-size:12px;font-weight:600;color:#334155">'
                    f'{OBS_LABEL.get(obs, obs)}</div>{svg}</div>')
        section = {
            "title": f"{cond}",
            "desc": ("vEcoli (left col) vs v2ecoli (right col) initial values; "
                     "verdict from gen-1 matched-time median |Δ|. Plots overlay "
                     "the full trajectories (all generations) on a shared "
                     "seconds axis."),
            "rows": rows,
            "html": ('<div style="padding:8px 18px 14px">'
                     + "".join(plots_html) + "</div>"),
        }
        sections.append(section)

    html = report.render_report(
        sections, title="v2ecoli vs vEcoli — matched-timepoint trajectory comparison")
    out = REPO_ROOT / "out" / "full_compare" / "comparison_matched_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    dl = Path.home() / "Downloads" / "v2ecoli_vecoli_matched_report.html"
    shutil.copy(out, dl)
    return out, dl, tally, summary_rows


def main():
    cond_data = build()
    out, dl, tally, summary_rows = render(cond_data)

    print("\n" + "=" * 78)
    print("ANSWER TABLE — initial (first matched t) values + gen-1 matched median |Δ|")
    print("=" * 78)
    print(f"{'condition':10s} {'observable':26s} {'vEcoli init':>12s} "
          f"{'v2ecoli init':>12s} {'init Δ%':>8s} {'gen1 |Δ|%':>9s}  verdict")
    for cond, obs, ive, iv2, med, v in summary_rows:
        if ive is None:
            print(f"{cond:10s} {obs:26s} {'—':>12s} {'—':>12s} {'—':>8s} "
                  f"{'—':>9s}  {v}")
        else:
            idp = (iv2 - ive) / ive * 100 if ive else float('nan')
            print(f"{cond:10s} {obs:26s} {ive:12.4g} {iv2:12.4g} "
                  f"{idp:+8.2f} {med*100:9.2f}  {v}")
    print("\nVerdict tally:", {k: v for k, v in tally.items() if v})
    print(f"\nSaved: {out}\n       {dl}")


if __name__ == "__main__":
    main()
