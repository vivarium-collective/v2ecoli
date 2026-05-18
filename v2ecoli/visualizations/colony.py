"""ColonyVisualization — multi-cell pymunk colony report.

Migrated rendering from reports/colony_report.py. The Step takes a single
trajectory grouped by ``agent_id`` and renders colony snapshots, per-cell
mass trajectories, and phylogeny coloring. The wrapper at
reports/colony_report.py handles the EcoliWCM + pymunk physics +
surrogate-cell simulation orchestration (and GIF generation).

GIF placement
-------------
GIF generation (colony spatial animation, chromosome state animation) is
performed **in the wrapper** (``reports/colony_report.py``) because it
requires the full pymunk simulation results and PIL frame assembly.  The
wrapper encodes the resulting GIFs as base64 strings and passes them via
the ``metadata`` dict::

    metadata["colony_gif_b64"]     — colony spatial animation (optional)
    metadata["chrom_gif_b64"]      — chromosome state animation (optional)

The Step embeds whichever base64 strings are present as inline ``<img>``
data URIs so the HTML is self-contained.  If neither key is present the
report still renders with the summary tables.
"""

from __future__ import annotations

import base64
import time
from html import escape
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import (
    render_document,
    group_by_agent,
)


class ColonyVisualization(Visualization):
    """Render multi-cell colony report from a single trajectory.

    Inputs
    ------
    history:  list of flat snapshot dicts, each carrying at minimum
              ``agent_id`` (str), ``time`` (float), ``x`` (float),
              ``y`` (float), ``length`` (float), and ``mass`` (float).
    metadata: run metadata dict.  Recognised optional keys:

              * ``colony_size``     — int, initial colony size
              * ``n_adder``         — int, number of surrogate cells
              * ``n_initial``       — int, initial cell count
              * ``n_final``         — int, final cell count
              * ``duration_min``    — float, simulated minutes
              * ``env_size``        — float, environment side length (µm)
              * ``build_time``      — float, composite build time (s)
              * ``wall_time``       — float, wall-clock sim time (s)
              * ``seed``            — int, random seed
              * ``repro``           — dict from _get_reproducibility_info()
              * ``colony_gif_b64``  — base64 PNG/GIF of spatial animation
              * ``chrom_gif_b64``   — base64 PNG/GIF of chromosome animation
              * ``n_emitter_frames``— int, number of emitter frames

    The Step groups rows by ``agent_id``, builds a per-agent mass trajectory
    table, then embeds any supplied GIF animations as data URIs.
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[node]]",
            "metadata": "map[node]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta    = state.get("metadata") or {}
        title   = self.config.get("title") or "v2ecoli colony"

        by_agent = group_by_agent(history)
        body     = self._render_body(history, by_agent, meta, title)
        html     = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    # ------------------------------------------------------------------
    # Private rendering (migrated from reports/colony_report.py)
    # ------------------------------------------------------------------

    def _render_body(
        self,
        history: list[dict],
        by_agent: dict[str, list[dict]],
        meta: dict,
        title: str,
    ) -> str:
        """Build the HTML body with colony GIF, chromosome GIF, summary tables.

        GIF animations are accepted as base64 strings in ``meta`` (see module
        docstring).  Matplotlib is **not** used here because image production
        happens in the wrapper; this method only assembles HTML.
        """
        import numpy as np  # lazy — used for simple per-agent stats

        # ---- unpack metadata -------------------------------------------
        n_initial       = meta.get("n_initial", meta.get("colony_size", "?"))
        n_final         = meta.get("n_final", "?")
        duration_min    = meta.get("duration_min", "?")
        env_size        = meta.get("env_size", "?")
        n_adder         = meta.get("n_adder", "?")
        build_time      = meta.get("build_time")
        wall_time       = meta.get("wall_time")
        seed            = meta.get("seed", "?")
        n_frames        = meta.get("n_emitter_frames", len(history))
        repro           = meta.get("repro") or {}

        colony_gif_b64  = meta.get("colony_gif_b64")
        chrom_gif_b64   = meta.get("chrom_gif_b64")

        # ---- per-agent mass summary ------------------------------------
        agent_rows_html = ""
        for aid in sorted(by_agent.keys()):
            rows = by_agent[aid]
            masses = [float(r.get("mass", 0)) for r in rows if r.get("mass") is not None]
            init_m = masses[0]  if masses else 0.0
            final_m= masses[-1] if masses else 0.0
            n_snaps= len(rows)
            agent_rows_html += (
                f"<tr><td><code>{escape(aid)}</code></td>"
                f"<td>{n_snaps}</td>"
                f"<td>{init_m:.1f}</td>"
                f"<td>{final_m:.1f}</td></tr>\n"
            )

        n_agents = len(by_agent)

        # ---- assemble HTML body ----------------------------------------
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Colony spatial GIF (inline data URI)
        colony_gif_html = ""
        if colony_gif_b64:
            colony_gif_html = f"""
<div class="media">
  <img src="data:image/gif;base64,{colony_gif_b64}"
       alt="Colony simulation — spatial view">
  <div class="media-label">
    Colony spatial view: {n_initial} initial cells &rarr; {n_final} final cells
    over {duration_min} min.
    Colored cell = whole-cell <em>E. coli</em>; grey = surrogates.
  </div>
</div>
"""

        # Chromosome state GIF (inline data URI)
        chrom_gif_html = ""
        if chrom_gif_b64:
            chrom_gif_html = f"""
<h2>Chromosome State</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#10b981; border-radius:50%"></div> OriC (origin of replication)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ef4444"></div> Ter (terminus)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6; border-radius:50%"></div> RNA polymerase</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f59e0b"></div> Replication fork</div>
</div>
<p>The circular chromosome of each whole-cell <em>E. coli</em>, synchronized with the
colony animation above. Chromosome replication initiates around ~23 min, producing 2
chromosomes visible as separate circles. Each frame shows the current number of
chromosomes, replication forks, and active RNA polymerases.</p>
<div class="media">
  <img src="data:image/gif;base64,{chrom_gif_b64}"
       alt="Chromosome state over time">
  <div class="media-label">
    Chromosome state: replication forks traverse the circular genome;
    RNA polymerases transcribe genes along the chromosome.
  </div>
</div>
"""

        # Simulation-parameter table
        param_rows = ""
        if duration_min != "?":
            duration_s = float(duration_min) * 60 if isinstance(duration_min, (int, float)) else "?"
            param_rows += f"<tr><td>Duration</td><td>{duration_min} min ({duration_s}s)</td></tr>\n"
        param_rows += "<tr><td>Whole-cell <em>E. coli</em></td><td>1 cell (EcoliWCM bridge, 55 steps)</td></tr>\n"
        if n_adder != "?":
            param_rows += f"<tr><td>Surrogate cells</td><td>{n_adder} (AdderGrowDivide)</td></tr>\n"
        if env_size != "?":
            param_rows += f"<tr><td>Environment</td><td>{env_size} &times; {env_size} &micro;m</td></tr>\n"
        param_rows += "<tr><td>Physics interval</td><td>10s</td></tr>\n"
        param_rows += "<tr><td>WCM update interval</td><td>60s</td></tr>\n"

        # Results table
        result_rows = ""
        if build_time is not None:
            result_rows += f"<tr><td>Build time</td><td>{float(build_time):.1f}s</td></tr>\n"
        if wall_time is not None:
            wt = float(wall_time)
            result_rows += f"<tr><td>Wall time</td><td>{wt:.0f}s ({wt/60:.1f} min)</td></tr>\n"
            if isinstance(duration_min, (int, float)):
                speed = float(duration_min) * 60 / max(wt, 1e-9)
                result_rows += f"<tr><td>Speed</td><td>{speed:.1f}&times; realtime</td></tr>\n"
        result_rows += f"<tr><td>Initial cells</td><td>{n_initial}</td></tr>\n"
        result_rows += f"<tr><td>Final cells</td><td>{n_final}</td></tr>\n"
        result_rows += f"<tr><td>Emitter frames</td><td>{n_frames}</td></tr>\n"

        # Reproducibility table
        repro_rows = ""
        if repro:
            for field, value in repro.items():
                repro_rows += f"<tr><td>{escape(str(field))}</td><td><code>{escape(str(value))}</code></td></tr>\n"
        repro_rows += f"<tr><td>Seed</td><td>{escape(str(seed))}</td></tr>\n"

        body = f"""
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
    background: #f8fafc;
    color: #1e293b;
  }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #16a34a; padding-bottom: 8px; }}
  h2 {{ color: #166534; margin-top: 2em; }}
  h3 {{ color: #334155; }}
  p {{ line-height: 1.6; }}
  table {{ border-collapse: collapse; margin: 1em 0; width: 100%; }}
  th, td {{ padding: 6px 16px; border: 1px solid #e2e8f0; text-align: left; }}
  th {{ background: #f1f5f9; }}
  .media {{ text-align: center; margin: 1.5em 0; }}
  .media img {{ max-width: 100%; border: 2px solid #e2e8f0; border-radius: 8px; }}
  .media-label {{ font-size: 0.85em; color: #64748b; margin-top: 0.5em; }}
  .legend {{ display: flex; gap: 2em; justify-content: center; margin: 1em 0; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.5em; }}
  .legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; border: 1px solid #ccc; }}
  .section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5em; margin: 1em 0; }}
  .metric-row {{ display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }}
  .metric {{ background: #f1f5f9; padding: 10px 16px; border-radius: 6px; min-width: 120px; }}
  .metric .label {{ font-size: 0.82em; color: #64748b; }}
  .metric .value {{ font-size: 1.3em; font-weight: 600; }}
</style>

<h1>{escape(title)}</h1>

<div class="section">
<p>This simulation places a <strong>whole-cell <em>E. coli</em> model</strong> &mdash; with 55
biological processes including metabolism, transcription, translation, DNA replication, and
chromosome segregation &mdash; inside a <strong>2D colony</strong> alongside simpler surrogate
cells.</p>

<p>Each whole-cell <em>E. coli</em> is implemented as an <code>EcoliWCM</code> process &mdash; a
process-bigraph <code>Process</code> that holds an internal <code>Composite</code> connected via a
bridge. The bridge maps external colony ports (mass, length, volume) to internal whole-cell stores.
The whole-cell model (v2ecoli) runs the full mechanistic simulation of intracellular biology, while
the colony framework (pymunk-process) handles spatial physics: cell body collisions,
growth-driven elongation, and division mechanics.</p>

<p>The <strong style="color:rgb(51,191,77);">green cell</strong> is the whole-cell
<em>E. coli</em> &mdash; its length and mass are driven by the internal biological simulation. The
<span style="color:#999;">grey cells</span> are surrogate cells using a simple adder growth model.
When the whole-cell model reaches its division threshold (~702 fg dry mass, ~42 min), the bridge
removes the mother cell and adds two daughter cells, each with a fresh copy of the whole-cell model.
Daughters are shown with color-shifted variants of the mother&rsquo;s green.</p>
</div>

<div class="metric-row">
  <div class="metric">
    <div class="label">Initial cells</div>
    <div class="value">{n_initial}</div>
  </div>
  <div class="metric">
    <div class="label">Final cells</div>
    <div class="value">{n_final}</div>
  </div>
  <div class="metric">
    <div class="label">Tracked agents</div>
    <div class="value">{n_agents}</div>
  </div>
  <div class="metric">
    <div class="label">Duration</div>
    <div class="value">{duration_min} min</div>
  </div>
</div>

<h2>Colony Dynamics</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:rgb(51,191,77)"></div>
    Whole-cell <em>E. coli</em> (v2ecoli, 55 processes)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#b0b0b0"></div>
    Surrogate cell (adder growth/division)</div>
</div>

{colony_gif_html}

{chrom_gif_html}

<h2>Per-Agent Mass Summary</h2>
<table>
  <thead>
    <tr>
      <th>Agent ID</th>
      <th>Snapshots</th>
      <th>Initial mass (fg)</th>
      <th>Final mass (fg)</th>
    </tr>
  </thead>
  <tbody>
    {agent_rows_html}
  </tbody>
</table>

<h2>Simulation Parameters</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  {param_rows}
</table>

<h2>Results</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {result_rows}
</table>

<h2>Reproducibility</h2>
<table style="font-size:0.85em;">
  <tr><th>Field</th><th>Value</th></tr>
  {repro_rows}
</table>

<footer style="margin-top:2em; padding-top:1em; border-top:1px solid #e2e8f0;
               color:#94a3b8; font-size:0.85em;">
  v2ecoli colony &middot; pure process-bigraph &middot;
  <a href="https://github.com/vivarium-collective/v2ecoli">github</a>
  &middot; rendered by <code>ColonyVisualization</code> at {escape(now_str)}.
</footer>
"""
        return body
