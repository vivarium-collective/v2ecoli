"""Multi-gen initiation-timeline viz.

For a study's latest run, read sqlite history rows, detect initiation
events (number_of_oric step-ups, e.g. 2→4), partition by agent_id
(= generation), and render an annotated timeline showing when each
initiation fired in which generation.

This is the "one-initiation-per-generation" panel — the visible payoff
of the multi-gen runner (sqlite history rows now span multiple
generations under the same simulation_id, with the agent_id rolling
over at division).

Usage:
    python scripts/render_initiation_timeline.py --study dnaa-04-initiation-mechanism

Writes ``studies/<study>/viz/initiation_timeline.html``.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


def _extract_per_gen_events(db_path: Path, sim_id: str) -> dict[str, list[tuple[float, int, int]]]:
    """Walk history rows; per agent_id (= generation), return the list of
    initiation events as ``(t, prev_oric, new_oric)`` tuples.

    Detection: oriC count step-up — `number_of_oric` strictly increased
    relative to the previous row of the same agent. v2ecoli's
    chromosome_replication step doubles oriC on initiation (2→4 or 4→8).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        per_agent_last: dict[str, int | None] = {}
        per_agent_events: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for t, agents_json in conn.execute(
            "SELECT global_time, json_extract(state,'$.agents') FROM history "
            "WHERE simulation_id=? ORDER BY global_time", (sim_id,)
        ):
            try:
                agents = json.loads(agents_json or "{}")
            except Exception:
                continue
            for aid, astate in agents.items():
                oric = ((astate or {}).get("listeners", {})
                                       .get("replication_data", {})
                                       .get("number_of_oric"))
                if oric is None:
                    continue
                try:
                    oric = int(oric)
                except (TypeError, ValueError):
                    continue
                prev = per_agent_last.get(aid)
                if prev is not None and oric > prev:
                    per_agent_events[aid].append((float(t), prev, oric))
                per_agent_last[aid] = oric
    finally:
        conn.close()
    return per_agent_events


def _agent_time_ranges(db_path: Path, sim_id: str) -> dict[str, tuple[float, float]]:
    """For each agent_id, find (first_seen_time, last_seen_time)."""
    conn = sqlite3.connect(str(db_path))
    try:
        out: dict[str, list[float]] = {}
        for t, agents_json in conn.execute(
            "SELECT global_time, json_extract(state,'$.agents') FROM history "
            "WHERE simulation_id=? ORDER BY global_time", (sim_id,)
        ):
            try:
                agents = json.loads(agents_json or "{}")
            except Exception:
                continue
            for aid in agents:
                lst = out.setdefault(aid, [float(t), float(t)])
                lst[1] = float(t)
        return {aid: (lst[0], lst[1]) for aid, lst in out.items()}
    finally:
        conn.close()


_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]


def render(study_dir: Path) -> Path:
    db = study_dir / "runs.db"
    if not db.is_file():
        raise SystemExit(f"no runs.db at {db}")
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT run_id, sim_name FROM runs_meta WHERE status='completed' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit(f"no completed run in {db}")
    sim_id, sim_name = row

    per_agent_events = _extract_per_gen_events(db, sim_id)
    agent_ranges = _agent_time_ranges(db, sim_id)

    # Sort agents by first-seen time (gives generation order)
    agents_sorted = sorted(agent_ranges.keys(), key=lambda a: agent_ranges[a][0])

    # Build Plotly traces: one horizontal bar per agent (gen lifetime),
    # plus markers at each initiation event time.
    bar_traces = []
    init_traces = []
    annotations = []
    for i, aid in enumerate(agents_sorted):
        t0, t1 = agent_ranges[aid]
        color = _PALETTE[i % len(_PALETTE)]
        bar_traces.append({
            "type": "bar", "orientation": "h",
            "x": [t1 - t0], "y": [f"gen {i + 1} (agent {aid!r})"],
            "base": [t0],
            "marker": {"color": color, "opacity": 0.35},
            "name": f"gen {i + 1}",
            "hovertemplate": f"agent {aid!r}<br>t=[{{base}}, {{x:.0f}}+base]s<extra></extra>",
        })
        ev = per_agent_events.get(aid, [])
        if ev:
            init_traces.append({
                "type": "scatter", "mode": "markers+text",
                "x": [e[0] for e in ev],
                "y": [f"gen {i + 1} (agent {aid!r})"] * len(ev),
                "text": [f"{p}→{n}" for _, p, n in ev],
                "textposition": "top center",
                "marker": {"color": color, "size": 14, "symbol": "diamond",
                           "line": {"width": 1, "color": "#1f2937"}},
                "name": f"gen {i + 1} init",
                "hovertemplate": "init @ t=%{x:.0f}s<br>oriC %{text}<extra></extra>",
            })

    n_init_total = sum(len(ev) for ev in per_agent_events.values())
    n_gen = len(agents_sorted)
    subtitle = (f"{n_init_total} initiation event(s) across {n_gen} generation(s) — "
                f"sim {sim_name!r}, run id …{sim_id[-12:]}")

    plot_data = bar_traces + init_traces
    plot_layout = {
        "title": {"text": f"Initiation timeline — {study_dir.name}", "x": 0.5, "xanchor": "center"},
        "xaxis": {"title": "time (s)", "showgrid": True, "gridcolor": "#e5e7eb"},
        "yaxis": {"title": "", "autorange": "reversed", "showgrid": False},
        "annotations": annotations,
        "showlegend": False,
        "barmode": "overlay",
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "margin": {"t": 80, "r": 30, "b": 60, "l": 200},
        "height": max(180, 60 + 50 * n_gen),
    }

    out = study_dir / "viz" / "initiation_timeline.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Initiation timeline</title>'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'
        '<style>html,body{margin:0;padding:18px 22px;background:#fff;color:#1f2937;'
        'font-family:-apple-system,"Segoe UI",sans-serif}'
        'h1{font-size:1.15em;margin:0 0 4px 0}'
        '.subtitle{color:#6b7280;font-size:0.85em;margin-bottom:14px}</style>'
        '</head><body>'
        f'<h1>Initiation timeline — {study_dir.name}</h1>'
        f'<div class="subtitle">{subtitle}</div>'
        '<div id="chart"></div>'
        '<script>Plotly.newPlot("chart", '
        f'{json.dumps(plot_data)}, {json.dumps(plot_layout)}, '
        '{responsive: true, displayModeBar: false});</script>'
        '</body></html>'
    )
    out.write_text(html)
    size_kb = max(1, out.stat().st_size // 1024)
    print(f"[initiation_timeline] wrote {out} ({size_kb} KB; {n_init_total} events across {n_gen} gens)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True,
                   help="study slug (e.g. dnaa-04-initiation-mechanism)")
    args = p.parse_args()
    workspace = Path(__file__).resolve().parent.parent
    study_dir = workspace / "studies" / args.study
    render(study_dir)


if __name__ == "__main__":
    main()
