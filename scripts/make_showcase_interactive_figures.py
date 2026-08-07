#!/usr/bin/env python
"""Interactive, unit-labelled Plotly figures for the v2ecoli-baseline-showcase.

Companion to the static Altair/matplotlib figures (scripts/render_showcase2.py):
this adds self-contained interactive Plotly charts — one HTML file per figure,
written to ``reports/figures/<study>/<name>.html`` so the dashboard
auto-discovers them as embedded visualizations (no study.yaml edits required;
curated captions in study.yaml override the generic ones).

Every axis and hover read carries its **declared schema unit** (fg, uM, 1/s, …)
resolved from the live port schemas via ``v2ecoli.library.units_resolver`` — the
same source the Units Atlas uses. A small verified supplement covers the
mass-deriver observables (volume=fL, growth rate=1/s, the *_mass components=fg)
whose units live on a deriver the index walker doesn't enumerate; the values are
read straight from ``v2ecoli/steps/derivers/mass_deriver.py``.

The Units Atlas itself is folded in here as an interactive table figure
(``units_atlas`` in showcase-2), replacing the standalone units-atlas
investigation.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/make_showcase_interactive_figures.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import plotly.graph_objects as go

from v2ecoli.library.units_resolver import (
    build_units_index,
    resolve_unit,
    format_axis_label,
)
from v2ecoli.library.units_atlas import build_atlas

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = REPO_ROOT / "reports" / "figures"
SHOWCASE2_RUN = (
    REPO_ROOT
    / ".pbg/runs/showcase2-baseline-full/sweep/showcase2_baseline"
)

# ---------------------------------------------------------------------------
# Units — schema index + a verified supplement for mass-deriver observables.
# The supplement values are read from v2ecoli/steps/derivers/mass_deriver.py
# (overwrite[quantity[float,fL]] for volume, 1/s for instantaneous growth rate,
# fg for every dry-mass component). Schema-derived units always win.
# ---------------------------------------------------------------------------
_UNITS_SUPPLEMENT = {
    "listeners.mass.dna_mass": "fg",
    "listeners.mass.mRna_mass": "fg",
    "listeners.mass.rRna_mass": "fg",
    "listeners.mass.tRna_mass": "fg",
    "listeners.mass.rna_mass": "fg",
    "listeners.mass.smallMolecule_mass": "fg",
    "listeners.mass.water_mass": "fg",
    "listeners.mass.growth": "fg",
    "listeners.mass.volume": "fL",
    "listeners.mass.instantaneous_growth_rate": "1/s",
}

_UNITS_INDEX = build_units_index()


def unit_for(path: str) -> str | None:
    """Declared unit for a dotted observable path (schema first, supplement)."""
    return resolve_unit(_UNITS_INDEX, path) or _UNITS_SUPPLEMENT.get(path)


def col_to_path(col: str) -> str:
    """history column 'listeners__mass__cell_mass' -> 'listeners.mass.cell_mass'."""
    return col.replace("__", ".")


def axis(base_label: str, path: str) -> str:
    """Axis title 'Dry mass (fg)' from a base label + an observable path."""
    return format_axis_label(base_label, unit_for(path))


# ---------------------------------------------------------------------------
# HTML shell — one self-contained file per figure (Plotly from CDN).
# ---------------------------------------------------------------------------
TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, Helvetica, Arial, sans-serif; color: #1f2933; background: #fff; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 18px 20px 28px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  p.cap {{ font-size: 13px; color: #52606d; margin: 0 0 14px; line-height: 1.5; }}
  .plot {{ width: 100%; }}
  .foot {{ font-size: 11px; color: #9aa5b1; margin-top: 10px; }}
</style></head>
<body><div class="wrap">
  <h1>{title}</h1>
  <p class="cap">{caption}</p>
  <div class="plot">{plot_div}</div>
  <p class="foot">Interactive · units from declared v2ecoli port schemas · {source}</p>
</div></body></html>
"""

PALETTE = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c",
           "#319795", "#b7791f", "#718096"]


def write_fig(study_dir: str, name: str, fig: go.Figure, title: str,
              caption: str, source: str, height: int = 560) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=28, t=30, b=56),
        template="plotly_white",
        font=dict(size=12, color="#1f2933"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False)
    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False,
                           config={"responsive": True, "displaylogo": False})
    html = TEMPLATE.format(title=title, caption=caption, plot_div=plot_div,
                           source=source)
    out_dir = FIG_ROOT / study_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out.relative_to(REPO_ROOT)}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# showcase-2 data — read the hive-partitioned history into per-cell ramps.
# ---------------------------------------------------------------------------
def load_showcase2():
    """Return a list of per-cell dicts: {seed, gen, label, t_min, cols{...}}.

    Splits each (seed, generation, agent_id) partition into real cells using the
    dry_mass_fold_change reset (the showcase-2 baseline packs a gen-3 daughter
    into the gen-2 partition — see analyses/cell_mass.py).
    """
    import duckdb

    files = glob.glob(str(SHOWCASE2_RUN / "history/**/*.pq"), recursive=True)
    if not files:
        return []
    con = duckdb.connect()
    rel = con.execute(
        f"select * from read_parquet('{SHOWCASE2_RUN}/history/**/*.pq', "
        "hive_partitioning=true) "
        "order by lineage_seed, generation, agent_id, global_time"
    ).pl()
    cols = [c for c in rel.columns if c.startswith("listeners__")]
    cells = []
    for (seed, gen, aid), grp in rel.group_by(
        ["lineage_seed", "generation", "agent_id"], maintain_order=True
    ):
        grp = grp.sort("global_time")
        fc = grp["listeners__mass__dry_mass_fold_change"].to_list()
        # sub-cell index increments whenever fold-change resets toward 1.0
        sub, prev = 0, None
        sub_idx = []
        for v in fc:
            if prev is not None and v is not None and prev is not None \
                    and v < prev * 0.7:
                sub += 1
            sub_idx.append(sub)
            prev = v
        for s in sorted(set(sub_idx)):
            mask = [i for i, x in enumerate(sub_idx) if x == s]
            t = [grp["global_time"][i] / 60.0 for i in mask]
            data = {c: [grp[c][i] for i in mask] for c in cols}
            real_gen = int(gen) + s
            cells.append({
                "seed": int(seed), "gen": real_gen,
                "label": f"seed {int(seed)} · gen {real_gen}",
                "t_min": t, "cols": data,
            })
    con.close()
    return cells


def _series(cells, col):
    """Yield (label, gen, t_min, y) for a single history column across cells."""
    for c in cells:
        y = c["cols"].get(col)
        if y is None:
            continue
        yield c["label"], c["gen"], c["t_min"], y


def fig_dry_mass(cells):
    path = "listeners.mass.dry_mass"
    u = unit_for(path) or "fg"
    fig = go.Figure()
    for i, (label, gen, t, y) in enumerate(_series(cells, "listeners__mass__dry_mass")):
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=label,
            line=dict(width=2),
            hovertemplate=f"t=%{{x:.1f}} min<br>dry mass=%{{y:.0f}} {u}<extra>{label}</extra>",
        ))
    fig.update_layout(xaxis_title="Time (min)", yaxis_title=axis("Dry mass", path))
    return fig


def fig_mass_composition(cells):
    comps = [
        ("listeners__mass__protein_mass", "Protein"),
        ("listeners__mass__rna_mass", "RNA"),
        ("listeners__mass__dna_mass", "DNA"),
        ("listeners__mass__smallMolecule_mass", "Small molecules"),
    ]
    # one representative cell (first seed-0 gen-1) for a clean composition view
    cell = next((c for c in cells if c["seed"] == 0 and c["gen"] == 1), cells[0])
    u = "fg"
    fig = go.Figure()
    for col, label in comps:
        y = cell["cols"].get(col)
        if y is None:
            continue
        path = col_to_path(col)
        uu = unit_for(path) or u
        fig.add_trace(go.Scatter(
            x=cell["t_min"], y=y, mode="lines", name=label, stackgroup="one",
            hovertemplate=f"t=%{{x:.1f}} min<br>{label}=%{{y:.1f}} {uu}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Time (min)",
                      yaxis_title=f"Mass ({u})")
    return fig


def fig_growth_rate(cells):
    path = "listeners.mass.instantaneous_growth_rate"
    u = unit_for(path) or "1/s"
    fig = go.Figure()
    for label, gen, t, y in _series(cells, "listeners__mass__instantaneous_growth_rate"):
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=label, line=dict(width=2),
            hovertemplate=f"t=%{{x:.1f}} min<br>μ=%{{y:.2e}} {u}<extra>{label}</extra>",
        ))
    fig.update_layout(xaxis_title="Time (min)",
                      yaxis_title=axis("Instantaneous growth rate", path))
    return fig


def fig_regulation(cells):
    """ppGpp concentration (uM) + tRNA charged fraction on a shared time axis."""
    pp_path = "listeners.growth_limits.ppgpp_conc"
    pp_u = unit_for(pp_path) or "uM"
    fig = go.Figure()
    for label, gen, t, y in _series(cells, "listeners__growth_limits__ppgpp_conc"):
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=f"ppGpp · {label}",
            line=dict(width=2),
            hovertemplate=f"t=%{{x:.1f}} min<br>ppGpp=%{{y:.1f}} {pp_u}<extra>{label}</extra>",
        ))
    for label, gen, t, y in _series(cells, "listeners__growth_limits__fraction_trna_charged"):
        # fraction_trna_charged is a per-amino-acid array each step; collapse to
        # the mean charged fraction so the trace is a scalar time series.
        y_mean = [
            (sum(v) / len(v)) if isinstance(v, (list, tuple)) and v else
            (v if isinstance(v, (int, float)) else None)
            for v in y
        ]
        fig.add_trace(go.Scatter(
            x=t, y=y_mean, mode="lines", name=f"tRNA charged (mean) · {label}",
            line=dict(width=1.5, dash="dot"), yaxis="y2",
            hovertemplate=f"t=%{{x:.1f}} min<br>mean charged=%{{y:.2f}}<extra>{label}</extra>",
        ))
    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis=dict(title=axis("ppGpp", pp_path)),
        yaxis2=dict(title="tRNA charged fraction (dimensionless)",
                    overlaying="y", side="right", showgrid=False,
                    range=[0, 1]),
    )
    return fig


# ---------------------------------------------------------------------------
# Units Atlas — folded in as an interactive table grouped by dimension.
# ---------------------------------------------------------------------------
_DIM_COLOR = {
    "mass": "#2b6cb0", "concentration": "#38a169", "rate": "#dd6b20",
    "count": "#805ad5", "time": "#319795", "other": "#718096",
}


def fig_units_atlas():
    atlas = build_atlas(str(SHOWCASE2_RUN))
    order = ["mass", "concentration", "rate", "count", "time", "other"]
    headers = ["Observable path", "Unit", "Example", "Min", "Max"]
    cells_path, cells_unit, cells_ex, cells_min, cells_max, fill = (
        [], [], [], [], [], [])

    def fmt(x):
        if x is None:
            return "—"
        ax = abs(x)
        if ax != 0 and (ax < 1e-3 or ax >= 1e5):
            return f"{x:.2e}"
        return f"{x:.4g}"

    for dim in order:
        rows = atlas.get(dim) or []
        for r in rows:
            cells_path.append(r["path"])
            cells_unit.append(r["unit"])
            cells_ex.append(fmt(r.get("example")))
            cells_min.append(fmt(r.get("min")))
            cells_max.append(fmt(r.get("max")))
            fill.append(_DIM_COLOR.get(dim, "#718096"))
    fig = go.Figure(go.Table(
        columnwidth=[3.2, 0.9, 1.1, 1.1, 1.1],
        header=dict(values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#1f2933", font=dict(color="white", size=12),
                    align="left", height=30),
        cells=dict(
            values=[cells_path, cells_unit, cells_ex, cells_min, cells_max],
            align=["left", "left", "right", "right", "right"],
            fill_color=[["#f7fafc"] * len(cells_path)],
            font=dict(size=11, color="#1f2933"),
            height=24,
            line_color="#e4e7eb",
        ),
    ))
    n = len(cells_path)
    fig.update_layout(height=min(900, 80 + 26 * n))
    return fig, n


# ---------------------------------------------------------------------------
# showcase-1 — interactive sim_data summary (counts) with units where known.
# ---------------------------------------------------------------------------
def fig_simdata_summary():
    counts_json = (REPO_ROOT
                   / "workspace/studies/showcase-1-parca/charts"
                   / "showcase1_simdata_summary.counts.json")
    if not counts_json.is_file():
        return None
    data = json.loads(counts_json.read_text())
    if not isinstance(data, dict) or not data:
        return None
    labels = list(data.keys())
    values = [data[k] for k in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color="#2b6cb0",
        hovertemplate="%{y}: %{x:,} (count)<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Count (molecules / reactions)",
                      yaxis_title="ParCa reconstruction entity",
                      yaxis=dict(autorange="reversed"))
    return fig


def main():
    if not SHOWCASE2_RUN.exists():
        print(f"!! showcase-2 run data not found at {SHOWCASE2_RUN}", file=sys.stderr)
    cells = load_showcase2()
    print(f"loaded {len(cells)} showcase-2 cells")

    S2 = "showcase-2-baseline-figures"
    src2 = "2-seed × multi-gen wild-type baseline (showcase2-baseline-full)"
    if cells:
        write_fig(S2, "cell_mass_interactive", fig_dry_mass(cells),
                  "Dry mass over the cell cycle",
                  "Single-cell exponential growth and division — each line is one "
                  "cell's ramp; mass halves at division. Hover for unit-tagged reads.",
                  src2)
        write_fig(S2, "mass_composition_interactive", fig_mass_composition(cells),
                  "Dry-mass composition over one cell cycle",
                  "Stacked protein / RNA / DNA / small-molecule mass for a "
                  "representative cell (seed 0, gen 1).", src2)
        write_fig(S2, "growth_rate_interactive", fig_growth_rate(cells),
                  "Instantaneous specific growth rate",
                  "μ = d(ln mass)/dt per cell across the ensemble.", src2)
        write_fig(S2, "regulation_interactive", fig_regulation(cells),
                  "ppGpp and tRNA charging over the cell cycle",
                  "Stringent-response signal (ppGpp, left axis) against the "
                  "charged-tRNA fraction (right axis).", src2)

    # Units Atlas — folded in from the retired units-atlas investigation and
    # surfaced in EVERY baseline-showcase study so units are visible across the
    # whole investigation, not just the figures study.
    atlas_fig, n = fig_units_atlas()
    atlas_caption = (
        f"{n} declared unit-bearing observables grouped by physical dimension, "
        "with example magnitude + range sampled from the baseline run. Folds in "
        "the former units-atlas investigation so units stay attached to the "
        "readouts they describe."
    )
    ALL_STUDIES = [
        "showcase-1-parca",
        "showcase-2-baseline-figures",
        "showcase-3-variant-decide",
        "showcase-4-variant-comparison",
        "showcase-5-next-direction-decide",
        "showcase-6-equivalence-large",
    ]
    for study in ALL_STUDIES:
        write_fig(study, "units_atlas", atlas_fig,
                  "Units Atlas — every unit-bearing baseline readout",
                  atlas_caption, src2, height=min(900, 80 + 26 * n))

    sd = fig_simdata_summary()
    if sd is not None:
        write_fig("showcase-1-parca", "simdata_summary_interactive", sd,
                  "ParCa reconstruction summary",
                  "Molecule and reaction counts in the fitted sim_data bundle.",
                  "showcase-1 ParCa cache")
    print("done.")


if __name__ == "__main__":
    main()
