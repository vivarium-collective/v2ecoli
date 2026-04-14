"""Generate HTML report from plasmid simulation timeseries.

Reads out/plasmid/timeseries.json (produced by run_plasmid_experiment.py)
and writes reports/plasmid_replication_report.html with: a reproducibility
banner, a navigation menu, six plots inspired by the vEcoli_Plasmid PR
visualizations, and a bottom section with the plasmid process docstring,
math, and a static topology diagram.
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.library.repro_banner import banner_html
from v2ecoli.processes.plasmid_replication import (
    NAME as PLASMID_NAME,
    TOPOLOGY as PLASMID_TOPOLOGY,
    PlasmidReplication,
)

IN = "out/plasmid/timeseries.json"
OUT = "reports/plasmid_replication_report.html"


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def topology_diagram():
    """Static SVG showing the plasmid process and its stores."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Central process node
    proc = mpatches.FancyBboxPatch(
        (4.0, 2.5), 2.0, 1.0, boxstyle="round,pad=0.1",
        fc="#F4A7A1", ec="#8B2E27", lw=1.5,
    )
    ax.add_patch(proc)
    ax.text(5.0, 3.0, "ecoli-plasmid-\nreplication",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Ports: store positions
    stores = [
        # (x, y, name, path, color)
        (0.5, 5.0, "bulk",                       "bulk",                    "#FDE68A"),
        (0.5, 3.8, "listeners",                  "listeners",               "#D5D5D5"),
        (0.5, 2.6, "environment",                "environment",             "#BFDBFE"),
        (0.5, 1.4, "timestep / global_time",     "timestep/global_time",    "#E5E7EB"),
        (8.8, 5.2, "full_plasmids",              "unique/full_plasmid",     "#A4D4A4"),
        (8.8, 4.2, "oriVs",                      "unique/oriV",             "#A4D4A4"),
        (8.8, 3.2, "plasmid_domains",            "unique/plasmid_domain",   "#A4D4A4"),
        (8.8, 2.2, "plasmid_active_replisomes",  "unique/plasmid_active_replisome", "#A4D4A4"),
        (8.8, 0.9, "plasmid_rna_control",        "process_state/plasmid_rna_control", "#C084FC"),
    ]
    for x, y, port, path, color in stores:
        box = mpatches.FancyBboxPatch(
            (x - 0.1, y - 0.25), 1.9, 0.6, boxstyle="round,pad=0.05",
            fc=color, ec="#334155", lw=1.0,
        )
        ax.add_patch(box)
        ax.text(x + 0.85, y + 0.10, port, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text(x + 0.85, y - 0.08, path, ha="center", va="center",
                fontsize=7, color="#475569", style="italic")
        # edge to process
        is_left = x < 5
        if is_left:
            ax.annotate("", xy=(4.0, 3.0), xytext=(x + 1.8, y),
                        arrowprops=dict(arrowstyle="-|>", color="#334155",
                                        lw=1.0, shrinkA=2, shrinkB=2))
        else:
            ax.annotate("", xy=(6.0, 3.0), xytext=(x + 0.0, y),
                        arrowprops=dict(arrowstyle="-|>", color="#334155",
                                        lw=1.0, shrinkA=2, shrinkB=2))

    ax.text(5.0, 5.5, "Plasmid replication topology (ports → stores)",
            ha="center", fontsize=12, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#F4A7A1", label="process"),
        mpatches.Patch(color="#A4D4A4", label="unique molecule store"),
        mpatches.Patch(color="#FDE68A", label="bulk store"),
        mpatches.Patch(color="#C084FC", label="process_state (RNA I/II)"),
        mpatches.Patch(color="#BFDBFE", label="environment"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=8, ncol=3,
              frameon=False)
    return svg(fig)


def extract_math_blocks(docstring):
    """Render key math equations as MathJax blocks."""
    return [
        ("RNA I / II / hybrid ODE (Ataai-Shuler 1986)", r"""
\[
\begin{aligned}
\frac{d[\text{RNA}_{\text{I}}]}{dt}  &= K_T^{\text{I}} \cdot N_{\text{plasmids}} - k_h \cdot [\text{RNA}_{\text{I}}][\text{RNA}_{\text{II}}] - \gamma_I\,[\text{RNA}_{\text{I}}] \\
\frac{d[\text{RNA}_{\text{II}}]}{dt} &= K_T^{\text{II}} \cdot N_{\text{plasmids}} - k_h \cdot [\text{RNA}_{\text{I}}][\text{RNA}_{\text{II}}] - \gamma_{II}\,[\text{RNA}_{\text{II}}] \\
\frac{d[\text{Hybrid}]}{dt}          &= k_h \cdot [\text{RNA}_{\text{I}}][\text{RNA}_{\text{II}}] - \gamma_H\,[\text{Hybrid}]
\end{aligned}
\]"""),
        ("Initiation criterion", r"""
\[
\Delta\,\text{PL}_{\text{fractional}} \;=\; N_{\text{plasmids}} \cdot f \cdot \exp(-k_h \cdot [\text{RNA}_{\text{I}}] \cdot t_{\text{tx}})
\]
<p>Every <code>1 / K_T^II</code> ≈ 360&nbsp;s, each plasmid fires one RNA&nbsp;II initiation attempt. The surviving fraction (not bound by RNA&nbsp;I during transcription time <code>t_tx</code>) is multiplied by primer efficiency <code>f</code> and added to the fractional accumulator. When the accumulator crosses&nbsp;1.0, one integer initiation fires.</p>
"""),
        ("Elongation (polymerize algorithm)", r"""
\[
\text{sequences} = \text{buildSequences}(\text{template}, \; \text{fork}_{\text{pos}}, \; v \cdot \Delta t)
\quad\Rightarrow\quad
\text{result} = \text{polymerize}(\text{sequences}, \; [\text{dNTP}], \; R_{\text{limit}})
\]
<p>Unidirectional (one replichore, 2 sequence rows). Mass increase per fork: <code>Δm = Σ n<sub>nt</sub> · w<sub>nt</sub></code> (fg). One PPi released per dNTP polymerized.</p>
"""),
        ("Termination", r"""
\[
\text{coord} \ge L_{\text{replichore}} \;\Rightarrow\; \text{delete replisome, create new full\_plasmid}
\]
"""),
    ]


def main():
    with open(IN) as f:
        data = json.load(f)

    snaps = data["snapshots"]
    t = np.array([s["time"] for s in snaps])
    get = lambda k: np.array([s.get(k, 0) for s in snaps])

    plots = {}

    # 1. Plasmid copy number
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("n_full_plasmids"), "-o", color="tab:green", label="full_plasmid", lw=2, ms=5)
    ax.plot(t, get("n_oriV"), "-s", color="tab:blue", label="oriV", lw=1.5, ms=4, alpha=0.7)
    ax.plot(t, get("n_plasmid_domains"), "-^", color="tab:orange",
            label="plasmid_domain", lw=1.5, ms=4, alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("count")
    ax.set_title("Plasmid unique molecule counts")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    plots["copy_number"] = svg(fig)

    # 2. Active replisomes
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("n_plasmid_active_replisomes"), "-o", color="tab:purple",
            label="plasmid_active_replisome", lw=2, ms=5)
    ax.plot(t, get("n_active_replisomes"), "-", color="tab:red",
            label="chromosome active_replisome", lw=1.5, alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("count")
    ax.set_title("Active replisomes: plasmid vs chromosome")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    plots["replisomes"] = svg(fig)

    # 3. RNA I/II/hybrid
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("rna_I"), "-", color="#e67e22", label="RNA I (inhibitor)", lw=2)
    ax.plot(t, get("rna_II"), "-", color="#27ae60", label="RNA II (primer)", lw=2)
    ax.plot(t, get("hybrid"), "-", color="#8e44ad", label="RNA I:II hybrid", lw=2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("count per cell")
    ax.set_title("RNA I/II copy number control (Ataai-Shuler 1986)")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    plots["rna_control"] = svg(fig)

    # 4. PL_fractional + initiations
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(t, get("PL_fractional"), "-", color="tab:cyan", lw=2)
    axes[0].axhline(1.0, ls="--", color="grey", alpha=0.5, label="initiation threshold")
    axes[0].set_ylabel("PL_fractional")
    axes[0].set_title("Replication initiation accumulator")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, get("n_rna_initiations"), "-o", color="tab:red", ms=4)
    axes[1].plot(t, get("time_since_rna_II"), "-", color="grey", lw=1, alpha=0.5,
                 label="time_since_rna_II (s)")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("value")
    axes[1].set_title("Initiation events fired / RNA II interval countdown")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plots["initiation"] = svg(fig)

    # 5. Replisome subunit pool
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("replisome_trimer_min"), "-o", color="tab:blue",
            label="trimer subunits (sum)", lw=1.5, ms=4)
    ax.plot(t, get("replisome_monomer_min"), "-s", color="tab:red",
            label="monomer subunits (sum)", lw=1.5, ms=4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("count")
    ax.set_title("Replisome subunit pool")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    plots["subunits"] = svg(fig)

    # 6. Cell mass
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("cell_mass"), "-", color="tab:green", label="cell_mass", lw=2)
    ax2 = ax.twinx()
    ax2.plot(t, get("dna_mass"), "-", color="tab:purple", label="dna_mass", lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cell_mass (fg)", color="tab:green")
    ax2.set_ylabel("dna_mass (fg)", color="tab:purple")
    ax.set_title("Cell growth"); ax.grid(True, alpha=0.3)
    plots["mass"] = svg(fig)

    duration = data["duration"]
    interval = data["interval"]
    wall = data["wall_time"]
    n_final = snaps[-1]["n_full_plasmids"]
    n_init = snaps[0]["n_full_plasmids"]
    rna_i_final = snaps[-1]["rna_I"]
    rna_ii_final = snaps[-1]["rna_II"]

    topo_svg = topology_diagram()
    math_blocks = extract_math_blocks(PlasmidReplication.__doc__ or "")
    docstring = (PlasmidReplication.__doc__ or "").strip()
    # escape < > in docstring so it renders as text
    import html as _html
    docstring_html = _html.escape(docstring)

    # Topology table
    topo_rows = "\n".join(
        f"<tr><td><code>{port}</code></td><td><code>{' / '.join(path)}</code></td></tr>"
        for port, path in PLASMID_TOPOLOGY.items()
    )

    toc = """
<nav class="toc">
  <strong>Jump to:</strong>
  <a href="#summary">Summary</a>
  <a href="#copy-number">1. Plasmid counts</a>
  <a href="#replisomes">2. Active replisomes</a>
  <a href="#rna">3. RNA I/II</a>
  <a href="#initiation">4. Initiation</a>
  <a href="#subunits">5. Subunits</a>
  <a href="#mass">6. Mass</a>
  <a href="#topology">Topology</a>
  <a href="#math">Math</a>
  <a href="#docstring">Docstring</a>
</nav>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Plasmid Replication Report — v2ecoli</title>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px;
          margin: 0 auto; padding: 1em 1.5em 3em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #27ae60; padding-bottom: 0.3em; margin-top: 0.3em; }}
  h2 {{ color: #2c3e50; margin-top: 2em; padding-top: 0.4em;
        border-top: 1px solid #e5e7eb; }}
  .toc {{ position: sticky; top: 0; background: #fff; z-index: 10;
          padding: 0.5em 0.8em; margin: 0 -0.3em 1.5em; border-bottom: 1px solid #e5e7eb;
          font-size: 0.9em; }}
  .toc strong {{ color: #0f172a; margin-right: 0.5em; }}
  .toc a {{ color: #2563eb; text-decoration: none; margin-right: 1em; }}
  .toc a:hover {{ text-decoration: underline; }}
  .meta {{ background: #f6f8fa; border-left: 4px solid #27ae60;
           padding: 0.8em 1.2em; border-radius: 4px; font-size: 0.95em; }}
  .plot {{ margin: 1em 0; }}
  .plot svg {{ max-width: 100%; height: auto; }}
  .caveat {{ background: #fff7ed; border-left: 4px solid #f59e0b;
             padding: 0.5em 0.9em; border-radius: 4px; font-size: 0.9em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; }}
  th, td {{ padding: 0.3em 0.8em; border: 1px solid #ddd; text-align: left;
            font-size: 0.9em; }}
  th {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: 1px 5px; border-radius: 3px;
          border: 1px solid #e5e7eb; font-size: 0.88em; }}
  pre.docstring {{ background: #f6f8fa; border: 1px solid #e5e7eb;
                   border-radius: 4px; padding: 1em; font-size: 0.82em;
                   line-height: 1.4; overflow-x: auto; white-space: pre-wrap;
                   font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .mathbox {{ background: #fafbfc; border: 1px solid #e5e7eb;
              border-radius: 6px; padding: 0.8em 1.2em; margin: 1em 0; }}
  .mathbox h3 {{ margin-top: 0; color: #0f172a; font-size: 1em; }}
</style>
</head>
<body>

{banner_html()}

<h1>Plasmid Replication Report</h1>
{toc}

<div class="meta" id="summary">
<strong>ColE1 / pBR322 plasmid replication with RNA I/II copy-number control</strong>
(<a href="https://doi.org/10.1002/bit.260280616">Ataai &amp; Shuler 1986</a>).
<br><br>
Duration: <code>{duration}s</code> &middot;
Emission interval: <code>{interval}s</code> &middot;
Wall time: <code>{wall:.1f}s</code> ({wall/max(duration,1):.2f}× realtime) &middot;
Snapshots: <code>{len(snaps)}</code>
<br>
Branch: <code>plasmids</code> in v2ecoli &middot; sim_data: <code>v2parca#plasmid</code>
</div>

<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Initial</th><th>Final</th></tr>
<tr><td>full_plasmid count</td><td>{n_init}</td><td>{n_final}</td></tr>
<tr><td>RNA I (inhibitor)</td><td>3.00</td><td>{rna_i_final:.3f}</td></tr>
<tr><td>RNA II (primer)</td><td>0.00</td><td>{rna_ii_final:.3f}</td></tr>
<tr><td>plasmid_active_replisomes</td><td>{snaps[0]['n_plasmid_active_replisomes']}</td>
  <td>{snaps[-1]['n_plasmid_active_replisomes']}</td></tr>
</table>

<p class="caveat">
RNA&nbsp;II initiation interval is ~360&nbsp;s. Runs &lt; 360&nbsp;s generally
don't show replication events; this report confirms the ODE plumbing but
longer runs are needed for steady-state copy number (~20 at 60&nbsp;min doubling).
</p>

<h2 id="copy-number">1. Plasmid unique molecule counts</h2>
<div class="plot">{plots['copy_number']}</div>

<h2 id="replisomes">2. Active replisomes</h2>
<div class="plot">{plots['replisomes']}</div>

<h2 id="rna">3. RNA I / II / hybrid dynamics</h2>
<div class="plot">{plots['rna_control']}</div>

<h2 id="initiation">4. Replication initiation</h2>
<div class="plot">{plots['initiation']}</div>

<h2 id="subunits">5. Replisome subunit pool</h2>
<div class="plot">{plots['subunits']}</div>

<h2 id="mass">6. Cell mass</h2>
<div class="plot">{plots['mass']}</div>

<h2 id="topology">Plasmid process topology</h2>
<p>Ports and the store paths they bind to. Process runs as a plain
<code>EcoliStep</code> in the standalone-replication execution layer,
alongside <code>ecoli-chromosome-replication</code>.</p>
<div class="plot">{topo_svg}</div>

<table>
<tr><th>Port</th><th>Store path</th></tr>
{topo_rows}
</table>

<h2 id="math">Mathematical model</h2>
{''.join(f'<div class="mathbox"><h3>{title}</h3>{body}</div>' for title, body in math_blocks)}

<h2 id="docstring">Process docstring</h2>
<p>From <code>v2ecoli/processes/plasmid_replication.py</code>:</p>
<pre class="docstring">{docstring_html}</pre>

</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
