"""DnaA concentration viz — count / cell mass over time.

Haochen 2026-05-25 Round-3 ask: "We need to add a plot to show the
concentration of DnaA, namely, number of DnaA molecules / cell mass."

For a study's latest completed run, reads sqlite history rows and plots:
  Panel 1 — DnaA monomer count over time (raw count)
  Panel 2 — DnaA concentration = count / cell_mass (units: count/fg)

Vertical markers at division events (agent_id rollover, e.g. '0' → '00').

Usage:
    python scripts/render_dnaa_concentration.py --study dnaa-00-parameter-foundation

Writes ``studies/<study>/viz/dnaa_concentration.html``. Requires the
study's emit_paths to include ``listeners.monomer_counts`` and
``listeners.mass.cell_mass`` (sqlite history rows must contain both).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

# MONOMER0-160 (DnaA) is at index 3861 in the v2ecoli monomer_counts vector;
# documented in studies/dnaa-00-parameter-foundation/study.yaml's readouts.
DNAA_MONOMER_INDEX = 3861


def _latest_run_id(db: Path, sim_name: str) -> str | None:
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT run_id FROM runs_meta WHERE sim_name=? AND status='completed' "
            "ORDER BY started_at DESC LIMIT 1",
            (sim_name,),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _extract_series(db: Path, sim_id: str):
    """Return (times, dnaA_counts, cell_masses, division_times)."""
    conn = sqlite3.connect(str(db))
    times: list[float] = []
    counts: list[float | None] = []
    masses: list[float | None] = []
    division_times: list[float] = []
    prev_agents: set[str] | None = None
    try:
        rows = conn.execute(
            "SELECT step, global_time, state FROM history WHERE simulation_id=? "
            "ORDER BY step",
            (sim_id,),
        ).fetchall()
    finally:
        conn.close()
    for _, t, raw in rows:
        state = json.loads(raw) if isinstance(raw, str) else raw
        agents = state.get("agents", {})
        agent_ids = set(agents.keys())
        if prev_agents is not None and agent_ids != prev_agents:
            division_times.append(float(t))
        prev_agents = agent_ids
        if not agents:
            continue
        # Pick first agent's data (single-daughter runs).
        first_id = sorted(agents.keys())[0]
        a = agents[first_id]
        listeners = a.get("listeners", {}) if isinstance(a, dict) else {}
        mc = listeners.get("monomer_counts")
        mass_obj = listeners.get("mass", {}) if isinstance(listeners, dict) else {}
        cm = (mass_obj or {}).get("cell_mass") if isinstance(mass_obj, dict) else None
        dnaA = None
        if isinstance(mc, list) and len(mc) > DNAA_MONOMER_INDEX:
            dnaA = mc[DNAA_MONOMER_INDEX]
        times.append(float(t))
        counts.append(dnaA)
        masses.append(cm)
    return times, counts, masses, division_times


def _build_html(study_dir: Path, sim_id: str, times, counts, masses, divs) -> Path:
    # Concentration in count / fg
    concs = [
        (c / m) if (c is not None and m not in (None, 0)) else None
        for c, m in zip(counts, masses)
    ]

    division_shapes = [
        {
            "type": "line", "xref": "x", "yref": "paper",
            "x0": t, "x1": t, "y0": 0, "y1": 1,
            "line": {"color": "#dc2626", "width": 1.6, "dash": "dash"},
            "opacity": 0.7,
        }
        for t in divs
    ]
    division_annotations = [
        {
            "x": t, "y": 1.02, "xref": "x", "yref": "paper",
            "text": f"division @ {t:.0f}s ({t/60:.1f} min)",
            "showarrow": False,
            "font": {"color": "#dc2626", "size": 11},
        }
        for t in divs
    ]

    count_trace = {
        "type": "scatter", "mode": "lines+markers",
        "x": times, "y": counts,
        "name": "DnaA monomer count",
        "line": {"color": "#1f77b4", "width": 2},
        "marker": {"size": 4},
        "hovertemplate": "t=%{x:.0f}s<br>DnaA=%{y:.0f}<extra></extra>",
    }
    count_layout = {
        "title": {"text": "DnaA monomer count over time",
                  "x": 0.5, "xanchor": "center"},
        "xaxis": {"title": "time (s)", "showgrid": True, "gridcolor": "#e5e7eb"},
        "yaxis": {"title": "DnaA molecules / cell"},
        "shapes": division_shapes,
        "annotations": division_annotations,
        "plot_bgcolor": "#fafafa", "paper_bgcolor": "#fff",
        "margin": {"t": 80, "r": 30, "b": 60, "l": 80},
        "height": 360,
    }

    conc_trace = {
        "type": "scatter", "mode": "lines+markers",
        "x": times, "y": concs,
        "name": "DnaA / cell mass",
        "line": {"color": "#16a34a", "width": 2},
        "marker": {"size": 4},
        "hovertemplate": "t=%{x:.0f}s<br>conc=%{y:.4f} count/fg<extra></extra>",
    }
    conc_layout = {
        "title": {"text": "DnaA concentration = count / cell_mass",
                  "x": 0.5, "xanchor": "center"},
        "xaxis": {"title": "time (s)", "showgrid": True, "gridcolor": "#e5e7eb"},
        "yaxis": {"title": "DnaA molecules / fg cell mass"},
        "shapes": division_shapes,
        "annotations": division_annotations,
        "plot_bgcolor": "#fafafa", "paper_bgcolor": "#fff",
        "margin": {"t": 80, "r": 30, "b": 60, "l": 80},
        "height": 360,
    }

    subtitle = (
        f"{len(times)} sample(s) · {len(divs)} division(s) · "
        f"sim run id …{sim_id[-12:]} · DnaA monomer index {DNAA_MONOMER_INDEX}"
    )

    out = study_dir / "viz" / "dnaa_concentration.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>DnaA concentration</title>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'
        '<style>html,body{margin:0;padding:18px 22px;background:#fff;color:#1f2937;'
        'font-family:-apple-system,"Segoe UI",sans-serif;overflow:hidden}'
        'h1{font-size:1.15em;margin:0 0 4px 0}'
        '.subtitle{color:#6b7280;font-size:0.85em;margin-bottom:14px}'
        '.panel{margin-bottom:14px}</style>'
        '</head><body>'
        f'<h1>DnaA concentration — {study_dir.name}</h1>'
        f'<div class="subtitle">{subtitle}</div>'
        '<div class="panel"><div id="chart-count"></div></div>'
        '<div class="panel"><div id="chart-conc"></div></div>'
        '<script>'
        f'Plotly.newPlot("chart-count", [{json.dumps(count_trace)}], '
        f'{json.dumps(count_layout)}, '
        '{responsive: true, displayModeBar: false});'
        f'Plotly.newPlot("chart-conc", [{json.dumps(conc_trace)}], '
        f'{json.dumps(conc_layout)}, '
        '{responsive: true, displayModeBar: false});'
        '</script></body></html>'
    )
    out.write_text(html)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True,
                   help="study slug (e.g. dnaa-00-parameter-foundation)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    study_dir = repo_root / "studies" / args.study
    db = study_dir / "runs.db"
    if not db.is_file():
        raise SystemExit(f"no runs.db at {db}")

    sim_id = _latest_run_id(db, args.study)
    if not sim_id:
        raise SystemExit(f"no completed runs for sim_name={args.study}")
    times, counts, masses, divs = _extract_series(db, sim_id)
    if not times:
        raise SystemExit(f"no history rows for run {sim_id}")
    n_with_mass = sum(1 for m in masses if m is not None)
    n_with_count = sum(1 for c in counts if c is not None)
    if n_with_mass == 0:
        raise SystemExit(
            "no cell_mass values in history — add "
            "agents.0.listeners.mass.cell_mass to study readouts + re-run")
    out = _build_html(study_dir, sim_id, times, counts, masses, divs)
    sz = max(1, out.stat().st_size // 1024)
    print(
        f"[dnaa_concentration] wrote {out} ({sz} KB; {n_with_count} count "
        f"samples, {n_with_mass} mass samples, {len(divs)} division(s))"
    )


if __name__ == "__main__":
    main()
