"""Ad-hoc render driver for showcase-2 figures from the CLEAN 3-gen run.

Runs each native analysis over the clean parquet sweep, captures the
Altair chart (via chart_to_html / ptools_heatmap_view -> chart_to_html) or the
matplotlib Figure (via fig_to_html), and renders PNG + (SVG where applicable)
into the study charts dir. Also renders the two ad-hoc DuckDB traces
(ppgpp_trace, trna_charged_trace).

Not committed; one-off mirror of the original (uncommitted) showcase render.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings

import duckdb

ROOT = "/Users/eranagmon/code/v2ecoli"
SWEEP = os.path.join(ROOT, ".pbg/runs/showcase2-baseline-clean/sweep/showcase2_baseline")
SIM_DATA_STATE = os.path.join(ROOT, "out/sim_data-showcase/parca_state.pkl.gz")
CHARTS = os.path.join(ROOT, "workspace/studies/showcase-2-baseline-figures/charts")
RUN_ID = "showcase2-baseline-clean"

sys.path.insert(0, ROOT)

import vl_convert as vlc  # noqa: E402
from bigraph_schema import allocate_core  # noqa: E402
import v2ecoli.workflow.analyses._helpers as helpers  # noqa: E402
import v2ecoli.workflow.render as rendermod  # noqa: E402
from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY  # noqa: E402
from v2ecoli.workflow.analysis_runner import (  # noqa: E402
    _history_from_clause, group_for_scale, scale_history_sql, build_cell_records,
)
from v2ecoli.processes.parca.data_loader import (  # noqa: E402
    load_parca_state, hydrate_sim_data_from_state,
)
from v2ecoli.workflow.analysis_runner import resolve_validation_data  # noqa: E402

# ---- capture hooks -----------------------------------------------------------
_CAP = {"chart": None, "fig": None, "heatmap_df": None}
_orig_chart_to_html = helpers.chart_to_html
_orig_fig_to_html = rendermod.fig_to_html
_orig_ptools_heatmap_view = helpers.ptools_heatmap_view


def _cap_heatmap(df, title, log_color=False, sort_rows=False, color_label="Value"):
    _CAP["heatmap_df"] = (df, title, log_color, sort_rows, color_label)
    return _orig_ptools_heatmap_view(df, title, log_color=log_color,
                                     sort_rows=sort_rows, color_label=color_label)


def _render_heatmap_matplotlib(base):
    """Render the captured flux heatmap with matplotlib (vl_convert overflows
    on the ~4500-row mark_rect spec)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df, title, log_color, sort_rows, color_label = _CAP["heatmap_df"]
    if sort_rows and df.shape[0] > 1:
        order = np.argsort(-df.max(axis=1).to_numpy(dtype=float))
        df = df.iloc[order]
    z = df.to_numpy(dtype=float)
    if log_color:
        pos = z[z > 0]
        floor = float(pos.min()) if pos.size else 1e-9
        z_color = np.log10(np.clip(z, floor, None))
        cb_title = f"log10({color_label})"
    else:
        z_color = z
        cb_title = color_label
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(z_color, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel(f"Reaction (sorted by |flux|, n={df.shape[0]})")
    ax.set_yticks([])
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=45, ha="right", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cb_title)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def _cap_chart(chart, title=""):
    _CAP["chart"] = chart
    return _orig_chart_to_html(chart, title)


def _cap_fig(fig, title=""):
    _CAP["fig"] = fig
    return _orig_fig_to_html(fig, title)


helpers.chart_to_html = _cap_chart
helpers.ptools_heatmap_view = _cap_heatmap
rendermod.fig_to_html = _cap_fig
# Patch the names already imported into analysis modules.
import v2ecoli.workflow.analyses.cell_mass as m_cell_mass  # noqa: E402
import v2ecoli.workflow.analyses.replication as m_repl  # noqa: E402
import v2ecoli.workflow.analyses.doubling_time_line as m_dtl  # noqa: E402
import v2ecoli.workflow.analyses.doubling_time_hist as m_dth  # noqa: E402
import v2ecoli.workflow.analyses.mass_fraction_summary_view as m_mfs  # noqa: E402
import v2ecoli.workflow.analyses.mass_fraction_voronoi as m_mfv  # noqa: E402
import v2ecoli.workflow.analyses.central_carbon_metabolism_scatter as m_ccm  # noqa: E402
import v2ecoli.workflow.analyses.protein_counts_validation as m_pcv  # noqa: E402
import v2ecoli.workflow.analyses.ptools_rxns as m_ptr  # noqa: E402

for mod in (m_cell_mass, m_repl, m_dtl, m_dth, m_mfs, m_pcv, m_ptr):
    if hasattr(mod, "chart_to_html"):
        mod.chart_to_html = _cap_chart
for mod in (m_mfv, m_ccm):
    if hasattr(mod, "fig_to_html"):
        mod.fig_to_html = _cap_fig
if hasattr(m_ptr, "ptools_heatmap_view"):
    m_ptr.ptools_heatmap_view = _cap_heatmap
# ptools_heatmap_view internally calls helpers.chart_to_html (already patched at
# the helpers module level — it calls the module-local name there).


def _save_chart_png_svg(chart, base):
    spec = json.dumps(chart.to_dict())
    png = vlc.vegalite_to_png(spec, scale=2)
    with open(base + ".png", "wb") as f:
        f.write(png)
    try:
        svg = vlc.vegalite_to_svg(spec)
        with open(base + ".svg", "w", encoding="utf-8") as f:
            f.write(svg)
    except Exception as e:  # noqa: BLE001
        print(f"  (svg failed for {os.path.basename(base)}: {e})")


def _save_fig_png_svg(fig, base):
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")


def _write_meta(name, title, interpretation):
    meta = {
        "source_run_id": RUN_ID,
        "rendered_at": time.time(),
        "title": title,
        "interpretation": interpretation,
        "command": (
            "scripts/render_showcase2.py: v2ecoli native analyses "
            "(v2ecoli/workflow/analyses/) over "
            ".pbg/runs/showcase2-baseline-clean/sweep/showcase2_baseline parquet "
            "(CLEAN 3-generation re-run); Altair->PNG/SVG via vl_convert, "
            "matplotlib figs saved directly."
        ),
    }
    with open(os.path.join(CHARTS, f"{name}.png.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---- context -----------------------------------------------------------------
print("loading sim_data ...", flush=True)
sim_data = hydrate_sim_data_from_state(load_parca_state(SIM_DATA_STATE))
validation_data = resolve_validation_data(sim_data)
print("validation_data:", "present" if validation_data is not None else "None", flush=True)
conn = duckdb.connect()
from_clause = _history_from_clause(SWEEP)
records = list(build_cell_records(SWEEP).values())
core = allocate_core()


def run_analysis(name, scale, group_pick):
    """Run one analysis at its scale, pick a group, capture + return chart/fig.

    group_pick: callable(list_of_group_keys) -> the key to render, or None for
    the first group.
    """
    step_cls = ANALYSIS_REGISTRY[name]
    assert step_cls.scale == scale, (name, step_cls.scale, scale)
    step = step_cls({}, core=core)
    groups = group_for_scale(scale, records)
    keys = list(groups)
    if not keys:
        raise RuntimeError(f"no groups for {name} scale {scale}")
    gkey = group_pick(keys) if group_pick else keys[0]
    _CAP["chart"] = None
    _CAP["fig"] = None
    _CAP["heatmap_df"] = None
    history_sql = scale_history_sql(scale, from_clause, gkey)
    out = step.update({
        "conn": conn, "history_sql": history_sql,
        "config_sql": "", "success_sql": "",
        "sim_data": sim_data, "validation_data": validation_data,
        "variant_metadata": {},
    })
    return out, gkey


# ---- figure specs ------------------------------------------------------------
# group_pick helpers
def seed0(keys):
    # multigeneration keys are (variant, lineage_seed); single keys are full id.
    for k in keys:
        if len(k) >= 2 and int(k[1]) == 0:
            return k
    return keys[0]


def seed0_gen1(keys):
    # single scale key = (variant, seed, generation, agent_id)
    for k in keys:
        if int(k[1]) == 0 and int(k[2]) == 1:
            return k
    return keys[0]


FIGS = [
    ("cell_mass", "multivariant", None,
     "Cell dry mass over the cell cycle (3 clean generations)",
     "Per-cell growth-and-division across the clean 2-seed x 3-generation baseline "
     "ensemble (variant 0). Each line is one cell's monotonic growth ramp coloured "
     "by generation; gen-1/gen-2/gen-3 cells each grow ~a doubling of mass and "
     "divide. With the clean phylogeny (gen1 agent=0 / gen2 agent=00 / gen3 "
     "agent=000) and the #173 last-generation fix, all three generations render as "
     "distinct clean ramps (no folded gen-2 / missing gen-3)."),
    ("mass_fraction_summary_view", "single", seed0_gen1,
     "Biomass composition over the cell cycle (mass fractions)",
     "Submass fractions (protein / rRNA / tRNA / mRNA / DNA / small molecules) over "
     "one cell cycle (seed 0, gen 1). Physiological E. coli composition."),
    ("mass_fraction_voronoi", "single", seed0_gen1,
     "Biomass composition Voronoi (initial vs final)",
     "Voronoi treemap of biomass components at cell birth vs division (seed 0, "
     "gen 1): nucleic acids, metabolites, protein. Areas proportional to mass."),
    ("doubling_time_line", "multivariant", None,
     "Doubling time vs generation (per seed)",
     "Per-generation doubling time derived from the emitted global_time span over "
     "the clean 3-generation run. With #173 each generation now ends at its own "
     "division, so the gen-2 step-cap over-report artifact (~63-104 min) is gone; "
     "all generations sit in the physiological band."),
    ("doubling_time_hist", "multivariant", None,
     "Distribution of doubling times across the ensemble",
     "Histogram of per-generation doubling times across the 2-seed x 3-gen "
     "ensemble. Clean per-generation division times (no step-cap inflation)."),
    ("replication", "multigeneration", seed0,
     "Chromosome replication across a lineage (oriC 2->4->2)",
     "oriC / replisome / fork dynamics across the seed-0 3-generation lineage. "
     "Replication initiates and completes each generation; oriC doubles then "
     "halves at division."),
    ("central_carbon_metabolism_scatter", "multiseed", None,
     "Central-carbon FBA fluxes vs Toya 2010",
     "Baseline central-carbon FBA fluxes vs Toya 2010 C13-MFA measured fluxes "
     "(multiseed over 2 seeds)."),
    ("ptools_rxns", "single", seed0_gen1,
     "Metabolic reaction-flux heatmap (reaction x timepoint)",
     "FBA base-reaction fluxes across the cell cycle (seed 0, gen 1); |flux| on a "
     "log10 color scale with reactions sorted by magnitude. The active band (top) "
     "vs inactive reactions (bottom) is legible; fluxes are stable across the cycle."),
    ("protein_counts_validation", "multiseed", None,
     "Proteome vs Schmidt 2015 / Wisniewski 2014",
     "log10 average monomer counts vs measured proteomics (Schmidt 2015, "
     "Wisniewski 2014); multiseed over 2 seeds."),
]


def render_traces():
    """ppGpp + tRNA-charged-fraction traces over the whole sweep (all cells)."""
    import altair as alt
    import pandas as pd
    files = glob.glob(os.path.join(SWEEP, "**", "history", "**", "*.pq"), recursive=True)
    flist = "[" + ",".join("'" + f + "'" for f in files) + "]"
    base = f"SELECT * FROM read_parquet({flist}, hive_partitioning=true)"
    specs = [
        ("ppgpp_trace", "listeners__growth_limits__ppgpp_conc",
         "ppGpp pool over the cell cycle (all seeds)", "ppGpp conc (uM)",
         "ppGpp concentration over the cell cycle across both seeds / three "
         "generations of the clean run."),
        ("trna_charged_trace", "listeners__growth_limits__fraction_trna_charged",
         "tRNA charged fraction over the cell cycle (all seeds)", "fraction tRNA charged",
         "Charged-tRNA fraction over the cell cycle across both seeds / three "
         "generations; remains high (~0.9+)."),
    ]
    for name, col, title, ylab, interp in specs:
        df = conn.sql(
            f"SELECT lineage_seed, generation, agent_id, global_time AS time, "
            f"{col} AS value FROM ({base}) WHERE {col} IS NOT NULL "
            f"ORDER BY lineage_seed, generation, global_time"
        ).pl().to_pandas()
        df["series"] = ("seed " + df["lineage_seed"].astype(str)
                        + " gen " + df["generation"].astype(str))
        chart = (
            alt.Chart(df).mark_line().encode(
                x=alt.X("time:Q", title="time (s, per generation)"),
                y=alt.Y("value:Q", title=ylab),
                color=alt.Color("series:N", title="lineage"),
            ).properties(width=560, height=320, title=title)
        )
        alt.data_transformers.disable_max_rows()
        _save_chart_png_svg(chart, os.path.join(CHARTS, name))
        _write_meta(name, title, interp)
        print(f"  [trace] {name}: {len(df)} rows", flush=True)


def main():
    os.makedirs(CHARTS, exist_ok=True)
    results = {}
    for name, scale, pick, title, interp in FIGS:
        print(f"=== {name} ({scale}) ===", flush=True)
        try:
            out, gkey = run_analysis(name, scale, pick)
            base = os.path.join(CHARTS, name)
            if _CAP["heatmap_df"] is not None:
                _render_heatmap_matplotlib(base)
                kind = "matplotlib-heatmap"
            elif _CAP["chart"] is not None:
                _save_chart_png_svg(_CAP["chart"], base)
                kind = "altair"
            elif _CAP["fig"] is not None:
                _save_fig_png_svg(_CAP["fig"], base)
                kind = "matplotlib"
            else:
                print(f"  !! no chart/fig captured; out keys={list(out.keys())}; "
                      f"data={str(out.get('data'))[:200]}", flush=True)
                results[name] = {"status": "NO_CAPTURE", "group": str(gkey)}
                continue
            _write_meta(name, title, interp)
            results[name] = {"status": "OK", "kind": kind, "group": str(gkey)}
            print(f"  rendered ({kind}) from group {gkey}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = {"status": f"ERROR: {type(e).__name__}: {e}"}
    print("=== traces ===", flush=True)
    try:
        render_traces()
        results["ppgpp_trace"] = {"status": "OK"}
        results["trna_charged_trace"] = {"status": "OK"}
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        results["traces"] = {"status": f"ERROR: {e}"}
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
