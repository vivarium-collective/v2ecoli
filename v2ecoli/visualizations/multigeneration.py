"""MultigenerationVisualization — N-generation lineage with mass trajectories.

Migrated rendering from reports/multigeneration_report.py. The Step takes a
single trajectory tagged by ``generation`` and renders end-to-end mass
trajectory plots across all generations + summary statistics. The wrapper
at reports/multigeneration_report.py handles the multi-generation simulation
loop (build composite → run to division → keep daughter → repeat).
"""

from __future__ import annotations

import base64
import io
import time
from html import escape
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import (
    render_document,
    group_by_generation,
)

# Mass component keys (key, display label, hex color) — migrated from legacy.
_MASS_KEYS = [
    ("dry_mass",           "Dry Mass",  "k"),
    ("protein_mass",       "Protein",   "#22c55e"),
    ("dna_mass",           "DNA",       "#8b5cf6"),
    ("rRna_mass",          "rRNA",      "#3b82f6"),
    ("tRna_mass",          "tRNA",      "#06b6d4"),
    ("mRna_mass",          "mRNA",      "#f97316"),
    ("smallMolecule_mass", "Small mol", "#f59e0b"),
]


def _fig_to_b64(fig) -> str:
    """Save a matplotlib figure to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt  # lazy — only imported when rendering
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class MultigenerationVisualization(Visualization):
    """Render N-generation lineage report from a single trajectory.

    Inputs
    ------
    history:  list of flat snapshot dicts, each with at least a ``generation``
              integer (1-based) and a ``time`` float. May also carry mass keys:
              ``dry_mass``, ``protein_mass``, ``dna_mass``, ``rRna_mass``,
              ``tRna_mass``, ``mRna_mass``, ``smallMolecule_mass``.
    metadata: run metadata dict (``n_generations``, ``seed``, etc.).

    The Step groups rows by ``generation``, then plots mass trajectories on
    a concatenated time axis with dashed generation boundaries, plus a
    per-generation summary table.
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history":  "list[map[any]]",
            "metadata": "map[any]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        history = state.get("history") or []
        meta    = state.get("metadata") or {}
        title   = self.config.get("title") or "v2ecoli multigeneration"

        gens  = group_by_generation(history)
        body  = self._render_body(history, gens, meta, title)
        html  = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    # ------------------------------------------------------------------
    # Private rendering (migrated from reports/multigeneration_report.py)
    # ------------------------------------------------------------------

    def _render_body(
        self,
        history: list[dict],
        gens: list[list[dict]],
        meta: dict,
        title: str,
    ) -> str:
        """Build the HTML body with mass plot + per-generation summary table.

        Lazy-imports matplotlib inside this method (headless Agg backend) so
        that importing the Step module never triggers a matplotlib cold-start.
        """
        import matplotlib  # lazy — only when rendering
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # ---- mass plot -------------------------------------------------
        fig, (ax_abs, ax_fold) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        n_gens = len(gens)
        fig.suptitle(
            f"Multigeneration lineage — {n_gens} generation(s)",
            fontsize=13,
        )

        cumulative_t = 0.0
        gen_boundaries: list[float] = []

        for gen_rows in gens:
            if not gen_rows:
                continue
            times = np.array([r.get("time", 0) for r in gen_rows], dtype=float)
            # Normalise times to start from 0 within each generation for
            # fold-change calculation; use the first snapshot's time as origin.
            t_start = times[0] if times.size > 0 else 0.0
            plot_times = (times - t_start + cumulative_t) / 60.0  # minutes

            gen_idx = int(gen_rows[0].get("generation", 0))
            is_first = gen_idx == min(int(r.get("generation", 0)) for r in history)

            for key, label, color in _MASS_KEYS:
                vals = np.array([r.get(key, 0) for r in gen_rows], dtype=float)
                if vals.size == 0 or vals[0] <= 0:
                    continue
                legend_label = label if is_first else None
                ax_abs.plot(plot_times, vals, color=color, lw=1.4, label=legend_label)
                ax_fold.plot(
                    plot_times,
                    vals / vals[0],
                    color=color,
                    lw=1.4,
                    label=legend_label,
                )

            gen_duration = float(times[-1] - t_start) if times.size > 1 else 0.0
            cumulative_t += gen_duration
            gen_boundaries.append(cumulative_t)

        for ax, ylabel, ax_title in (
            (ax_abs,  "Mass (fg)",                         "Absolute mass"),
            (ax_fold, "Fold change (within each generation)", "Per-generation fold change"),
        ):
            for b in gen_boundaries[:-1]:
                ax.axvline(b / 60.0, ls="--", color="#64748b", alpha=0.4, lw=1)
            ax.set_title(ax_title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.15)
            if ax is ax_abs:
                ax.legend(fontsize=8, ncol=4, loc="upper left")

        ax_fold.set_xlabel("Cumulative time (min)")
        fig.tight_layout()
        plot_b64 = _fig_to_b64(fig)

        # ---- per-generation summary table ------------------------------
        # Build synthetic GenerationResult-like summaries from the grouped rows.
        total_sim_time = 0.0
        gens_rows_html = ""
        for gen_rows in gens:
            if not gen_rows:
                continue
            gen_idx   = int(gen_rows[0].get("generation", 0))
            times_g   = [float(r.get("time", 0)) for r in gen_rows]
            duration  = times_g[-1] - times_g[0] if len(times_g) > 1 else 0.0
            total_sim_time += duration

            # Collect the best mass key available: prefer dry_mass, then mass.
            def _mass(row: dict) -> float:
                for k in ("dry_mass", "mass"):
                    v = row.get(k, 0)
                    if v:
                        return float(v)
                return 0.0

            initial_mass = _mass(gen_rows[0])
            final_mass   = _mass(gen_rows[-1])
            growth_ratio = final_mass / max(initial_mass, 1e-9) if initial_mass > 0 else 0.0

            gens_rows_html += (
                "<tr>"
                f"<td>{gen_idx}</td>"
                f"<td>{duration:.0f}</td>"
                f"<td>{initial_mass:.1f}</td>"
                f"<td>{final_mass:.1f}</td>"
                f"<td>{growth_ratio:.2f}×</td>"
                "</tr>"
            )

        n_snaps_total = sum(len(g) for g in gens)

        # ---- assemble HTML body ----------------------------------------
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 20px;
    color: #1e293b;
  }}
  h1 {{ margin-bottom: 0; }}
  h2 {{ margin-top: 32px; color: #0f172a; }}
  .metric-row {{
    display: flex;
    gap: 16px;
    margin: 16px 0;
    flex-wrap: wrap;
  }}
  .metric {{
    background: #f1f5f9;
    padding: 10px 16px;
    border-radius: 6px;
    min-width: 140px;
  }}
  .metric .label {{ font-size: 0.82em; color: #64748b; }}
  .metric .value {{ font-size: 1.3em; font-weight: 600; }}
  table {{
    border-collapse: collapse;
    margin: 12px 0;
    width: 100%;
  }}
  th, td {{
    padding: 6px 14px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
  }}
  th {{ background: #f8fafc; }}
  .plot img {{ max-width: 100%; }}
  p.intro {{ color: #475569; }}
</style>

<h1>{escape(title)}</h1>
<p class="intro">
  Single-lineage simulation: start from one newborn cell, run to division,
  keep exactly one daughter, repeat. The plot below shows mass over time
  across all generations, with dashed lines at generation boundaries.
</p>

<div class="metric-row">
  <div class="metric">
    <div class="label">Generations</div>
    <div class="value">{n_gens}</div>
  </div>
  <div class="metric">
    <div class="label">Total simulated time</div>
    <div class="value">{total_sim_time / 60:.1f} min</div>
  </div>
  <div class="metric">
    <div class="label">Total snapshots</div>
    <div class="value">{n_snaps_total}</div>
  </div>
</div>

<h2>Per-generation summary</h2>
<table>
  <thead>
    <tr>
      <th>Gen</th>
      <th>Sim time (s)</th>
      <th>Initial mass (fg)</th>
      <th>Final mass (fg)</th>
      <th>Growth</th>
    </tr>
  </thead>
  <tbody>
    {gens_rows_html}
  </tbody>
</table>

<h2>Mass across generations</h2>
<div class="plot">
  <img src="data:image/png;base64,{plot_b64}" alt="multigeneration mass plot">
</div>

<p style="color:#94a3b8; font-size:0.85em; margin-top:48px;">
  Generated by <code>MultigenerationVisualization</code> at {escape(now_str)}.
  Each generation is seeded from the previous cell's divide-time state via
  <code>v2ecoli.library.division.divide_cell</code> (daughter 1 kept).
</p>
"""
        return body
