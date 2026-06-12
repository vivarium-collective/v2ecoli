"""Render driver for the cross-variant COMPARISON analyses (showcase-4).

Iterates the eight variant-comparison analyses
(``v2ecoli/workflow/analyses/{growth_overlay,doubling_time_grouped,
mass_fraction_grouped,replication_overlay,proteome_delta,fba_flux_overlay,
regulation_overlay,scorecard}.py``) over a multivariant sweep, passing a real
``variant_metadata`` int->display-name map, captures each Altair chart (via
chart_to_html) or matplotlib Figure (via fig_to_html), and renders PNG + SVG.

These analyses overlay / group / delta baseline + variants, so they are only
meaningful on a MULTI-VARIANT sweep (the real 5-variant sweep).  For local
read-path testing against the single-variant showcase-2 clean run, pass
``--synthesize-variants`` to duplicate ``variant=0`` into synthetic
``variant=1..N`` partitions (the duplicate is perturbed slightly so the
overlay/delta machinery has non-degenerate data to plot — NOT scientifically
meaningful, only a code exercise).

Usage
-----
    # real 5-variant sweep:
    python scripts/render_variant_comparison.py \
        --sweep <sweep_dir> --sim-data out/workflow/simData.cPickle \
        --out workspace/studies/showcase-4-variant-comparison/charts

    # local read-path smoke test on showcase-2 clean data:
    python scripts/render_variant_comparison.py --synthesize-variants \
        --out /tmp/variant_comparison_smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import warnings

import duckdb

ROOT = os.environ.get("V2ECOLI_ROOT", "/Users/eranagmon/code/v2ecoli")
DEFAULT_SWEEP = os.path.join(
    ROOT, ".pbg/runs/showcase2-baseline-clean/sweep/showcase2_baseline")
DEFAULT_SIM_DATA = os.path.join(ROOT, "out/workflow/simData.cPickle")

# The runner's variant_metadata int->display-name map for the 5-variant sweep.
VARIANT_METADATA = {
    0: "baseline",
    1: "ppgpp-off",
    2: "dnaA-2x",
    3: "elong-down",
    4: "media",
}

# (analysis name, scale) — all are multivariant comparison analyses.
COMPARISON_ANALYSES = [
    ("growth_overlay", "multivariant"),
    ("doubling_time_grouped", "multivariant"),
    ("mass_fraction_grouped", "multivariant"),
    ("replication_overlay", "multivariant"),
    ("proteome_delta", "multivariant"),
    ("fba_flux_overlay", "multivariant"),
    ("regulation_overlay", "multivariant"),
    ("scorecard", "multivariant"),
]

sys.path.insert(0, ROOT)

import v2ecoli.workflow.analyses._helpers as helpers  # noqa: E402
import v2ecoli.workflow.render as rendermod  # noqa: E402

# ---- capture hooks (mirror render_showcase2.py) ------------------------------
_CAP = {"chart": None, "fig": None}
_orig_chart_to_html = helpers.chart_to_html
_orig_fig_to_html = rendermod.fig_to_html


def _cap_chart(chart, title=""):
    _CAP["chart"] = chart
    return _orig_chart_to_html(chart, title)


def _cap_fig(fig, title=""):
    _CAP["fig"] = fig
    return _orig_fig_to_html(fig, title)


helpers.chart_to_html = _cap_chart
rendermod.fig_to_html = _cap_fig

# Importing the analyses package registers all analyses AND binds the (now
# patched) chart_to_html / fig_to_html names into each analysis module.
import v2ecoli.workflow.analyses as _analyses_pkg  # noqa: E402,F401
from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY  # noqa: E402

# Re-point the names already imported into each comparison module to the
# capture hooks (they did `from _helpers import chart_to_html`).
for _modname in (
    "growth_overlay", "doubling_time_grouped", "mass_fraction_grouped",
    "replication_overlay", "proteome_delta", "fba_flux_overlay",
    "regulation_overlay", "scorecard",
):
    _mod = getattr(_analyses_pkg, _modname, None)
    if _mod is not None and hasattr(_mod, "chart_to_html"):
        _mod.chart_to_html = _cap_chart
    if _mod is not None and hasattr(_mod, "fig_to_html"):
        _mod.fig_to_html = _cap_fig

from bigraph_schema import allocate_core  # noqa: E402
from v2ecoli.workflow.analysis_runner import (  # noqa: E402
    _history_from_clause, resolve_validation_data,
)
from v2ecoli.library.sim_data import LoadSimData  # noqa: E402


def synthesize_variants(src_sweep: str, n_extra: int, dst: str) -> str:
    """Duplicate ``variant=0`` history into synthetic ``variant=1..n_extra``.

    Copies the parquet tree, rewriting the ``variant=0`` path segment to
    ``variant=k`` for each synthetic variant, so the multivariant read-path sees
    several variants.  The duplicates are NOT scientifically distinct (they are
    copies); this only exercises the color-/group-/delta-by-variant code.
    """
    import polars as pl

    hist_src = glob.glob(os.path.join(src_sweep, "**", "history"),
                         recursive=True)
    if not hist_src:
        raise FileNotFoundError(f"no history/ dir under {src_sweep!r}")
    # Mirror the whole sweep (summary.json etc.) then add synthetic variants.
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_sweep, dst)

    files = glob.glob(os.path.join(dst, "**", "history", "**", "variant=0",
                                   "**", "*.pq"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"no variant=0 parquet under copied sweep {dst!r}")

    for k in range(1, n_extra + 1):
        for f in files:
            new = f.replace(os.sep + "variant=0" + os.sep,
                            os.sep + f"variant={k}" + os.sep)
            os.makedirs(os.path.dirname(new), exist_ok=True)
            # Perturb a few scalar columns so overlays/deltas are non-degenerate
            # (purely so the rendering exercises non-zero deltas; not science).
            # NOTE: ``variant`` is a HIVE-PARTITION column (from the directory
            # path), not stored in the parquet — the path rewrite above sets it,
            # so we only perturb the in-file data columns here.
            df = pl.read_parquet(f)
            scale = 1.0 + 0.08 * k
            for col, fac in (
                ("listeners__mass__instantaneous_growth_rate", 1.0 / scale),
                ("listeners__mass__cell_mass", scale),
                ("listeners__mass__protein_mass", scale),
                ("listeners__growth_limits__ppgpp_conc",
                 (0.0 if k == 1 else 1.0 / scale)),  # variant 1 = ppGpp-off
            ):
                if col in df.columns:
                    df = df.with_columns((pl.col(col) * fac).alias(col))
            df.write_parquet(new)
    return dst


def _save_chart_png_svg(chart, base):
    import vl_convert as vlc
    spec = json.dumps(chart.to_dict())
    with open(base + ".png", "wb") as f:
        f.write(vlc.vegalite_to_png(spec, scale=2))
    try:
        with open(base + ".svg", "w", encoding="utf-8") as f:
            f.write(vlc.vegalite_to_svg(spec))
    except Exception as e:  # noqa: BLE001
        print(f"  (svg failed for {os.path.basename(base)}: {e})")


def _save_fig_png_svg(fig, base):
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--sim-data", default=DEFAULT_SIM_DATA)
    ap.add_argument("--out", default="/tmp/variant_comparison_charts")
    ap.add_argument("--synthesize-variants", type=int, nargs="?", const=4,
                    default=0, metavar="N",
                    help="duplicate variant=0 into N synthetic variants "
                         "(read-path smoke test only); default 4 when bare")
    ap.add_argument("--render-png", action="store_true",
                    help="also render PNG/SVG via vl_convert (needs vl_convert)")
    args = ap.parse_args()

    sweep = args.sweep
    if args.synthesize_variants:
        tmp = os.path.join(args.out, "_synth_sweep")
        sweep = synthesize_variants(sweep, args.synthesize_variants, tmp)
        print(f"synthesized {args.synthesize_variants} extra variants under "
              f"{sweep}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    print("loading sim_data ...", flush=True)
    sim_data = LoadSimData(sim_data_path=args.sim_data).sim_data
    validation_data = resolve_validation_data(sim_data)
    print("validation_data:",
          "present" if validation_data is not None else "None",
          "| reactionFlux:",
          "yes" if getattr(validation_data, "reactionFlux", None) is not None
          else "no", flush=True)

    conn = duckdb.connect()
    from_clause = _history_from_clause(sweep)
    # multivariant history_sql = the whole sweep (no WHERE filter).
    history_sql = f"SELECT * FROM {from_clause} ORDER BY global_time"
    variants = [r[0] for r in conn.sql(
        f"SELECT DISTINCT variant FROM {from_clause} ORDER BY 1").fetchall()]
    print(f"variants in sweep: {variants}", flush=True)
    core = allocate_core()

    results = {}
    for name, scale in COMPARISON_ANALYSES:
        print(f"=== {name} ({scale}) ===", flush=True)
        _CAP["chart"] = None
        _CAP["fig"] = None
        try:
            step = ANALYSIS_REGISTRY[name]({}, core=core)
            out = step.update({
                "conn": conn, "history_sql": history_sql,
                "config_sql": "", "success_sql": "",
                "sim_data": sim_data, "validation_data": validation_data,
                "variant_metadata": VARIANT_METADATA,
            })
            view = out.get("view") or ""
            nonempty_view = bool(view) and "<svg" in view or "vega" in view
            base = os.path.join(args.out, name)
            kind = None
            if args.render_png:
                if _CAP["chart"] is not None:
                    _save_chart_png_svg(_CAP["chart"], base)
                    kind = "altair-png"
                elif _CAP["fig"] is not None:
                    _save_fig_png_svg(_CAP["fig"], base)
                    kind = "matplotlib-png"
            # Always also write the HTML view (proof of non-empty render).
            if view:
                with open(base + ".html", "w", encoding="utf-8") as f:
                    f.write(view)
            results[name] = {
                "status": "OK" if nonempty_view else "EMPTY_VIEW",
                "kind": kind or ("altair" if _CAP["chart"] is not None
                                 else "matplotlib" if _CAP["fig"] is not None
                                 else "none"),
                "view_bytes": len(view),
                "data_keys": sorted((out.get("data") or {}).keys()),
            }
            print(f"  {results[name]}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = {"status": f"ERROR: {type(e).__name__}: {e}"}

    print("\n=== SUMMARY ===")
    ok = 0
    for k, v in results.items():
        print(f"  {k}: {v.get('status')}  "
              f"({v.get('kind','-')}, {v.get('view_bytes','-')} bytes)")
        if v.get("status") == "OK":
            ok += 1
    with open(os.path.join(args.out, "render_results.json"), "w") as f:
        json.dump({"variant_metadata": VARIANT_METADATA,
                   "sweep": sweep, "variants": variants,
                   "rendered_at": time.time(), "results": results}, f, indent=2)
    print(f"\n{ok}/{len(COMPARISON_ANALYSES)} analyses OK")
    return 0 if ok == len(COMPARISON_ANALYSES) else 1


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
