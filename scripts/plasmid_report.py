"""Generate HTML report from plasmid simulation timeseries.

Reads out/plasmid/timeseries.json (produced by run_plasmid_experiment.py)
and writes reports/plasmid_replication_report.html with a reproducibility
banner, navigation menu, six plots inspired by the vEcoli_Plasmid PR
visualizations, and an interactive Cytoscape network viewer of the
plasmid-enabled composite (using the shared v2ecoli.viz.build_graph /
render_html pipeline used by the other reports).
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from v2ecoli.library.repro_banner import banner_html

CACHE_DIR = "out/cache_plasmid"
IN = "out/plasmid/timeseries.json"
OUT = "reports/plasmid_replication_report.html"
NETWORK_OUT = "reports/plasmid_network.html"


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_network_html():
    """Build the interactive Cytoscape network using the shared v2ecoli.viz
    pipeline, filtered so the graph centers on the plasmid-replication
    process and the stores it binds. Writes to NETWORK_OUT.
    """
    from v2ecoli.composite import make_composite
    from v2ecoli.generate import build_execution_layers, DEFAULT_FEATURES
    from v2ecoli.viz import build_graph, render_html

    composite = make_composite(cache_dir=CACHE_DIR, features=DEFAULT_FEATURES)
    layers = build_execution_layers(DEFAULT_FEATURES)
    data = build_graph(composite, layers)

    n_proc = sum(1 for n in data["nodes"] if n["data"]["kind"] == "process")
    n_store = sum(1 for n in data["nodes"] if n["data"]["kind"] == "store")
    n_edges = len(data["edges"])
    subtitle = (f"{n_proc} processes · {n_store} stores · "
                f"{n_edges} edges · plasmid-enabled baseline composite")

    html = render_html(
        data,
        title="v2ecoli · Plasmid-enabled baseline",
        subtitle=subtitle,
    )
    os.makedirs(os.path.dirname(NETWORK_OUT), exist_ok=True)
    with open(NETWORK_OUT, "w") as f:
        f.write(html)
    print(f"wrote {NETWORK_OUT} ({os.path.getsize(NETWORK_OUT)/1024:.1f} KB, "
          f"{n_proc} procs / {n_store} stores / {n_edges} edges)")
    return os.path.basename(NETWORK_OUT), n_proc, n_store, n_edges


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

    network_file, n_proc, n_store, n_edges = build_network_html()

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
  <a href="#network">Network</a>
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

<h2 id="network">Network visualization</h2>
<p>Interactive Cytoscape viewer of the plasmid-enabled baseline composite
(<code>{n_proc}</code> processes, <code>{n_store}</code> stores,
<code>{n_edges}</code> edges). Same viewer as the other reports — click
any node to inspect port schemas, resolved store paths, docstring, and
class. <code>ecoli-plasmid-replication</code> is in the DNA-replication
subsystem (same color as chromosome replication).</p>
<p><a href="{network_file}" target="_blank" rel="noopener">Open full-screen &#8599;</a></p>
<iframe src="{network_file}" style="width:100%;height:720px;border:1px solid #e5e7eb;
        border-radius:6px;margin-top:0.5em" loading="lazy"></iframe>

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
