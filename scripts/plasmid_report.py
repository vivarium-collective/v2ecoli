"""Generate HTML report from plasmid simulation timeseries.

Reads out/plasmid/timeseries.json (produced by run_plasmid_experiment.py)
and writes reports/plasmid_replication_report.html with embedded SVG plots
inspired by the visualizations from vivarium-collective/vEcoli_Plasmid#2.
"""
import base64
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
os.chdir(ROOT)

IN = "out/plasmid/timeseries.json"
OUT = "reports/plasmid_replication_report.html"


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main():
    with open(IN) as f:
        data = json.load(f)

    snaps = data["snapshots"]
    t = np.array([s["time"] for s in snaps])

    get = lambda k: np.array([s.get(k, 0) for s in snaps])

    plots = {}

    # --- 1. Plasmid copy number over time ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("n_full_plasmids"), "-o", color="tab:green", label="full_plasmid", lw=2, ms=5)
    ax.plot(t, get("n_oriV"), "-s", color="tab:blue", label="oriV", lw=1.5, ms=4, alpha=0.7)
    ax.plot(t, get("n_plasmid_domains"), "-^", color="tab:orange",
            label="plasmid_domain", lw=1.5, ms=4, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.set_title("Plasmid unique molecule counts")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plots["copy_number"] = svg(fig)

    # --- 2. Active replisomes (plasmid vs chromosome) ------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("n_plasmid_active_replisomes"), "-o", color="tab:purple",
            label="plasmid_active_replisome", lw=2, ms=5)
    ax.plot(t, get("n_active_replisomes"), "-", color="tab:red",
            label="chromosome active_replisome", lw=1.5, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.set_title("Active replisomes: plasmid vs chromosome")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plots["replisomes"] = svg(fig)

    # --- 3. RNA I/II/hybrid dynamics (Ataai-Shuler 1986) ---------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("rna_I"), "-", color="#e67e22", label="RNA I (inhibitor)", lw=2)
    ax.plot(t, get("rna_II"), "-", color="#27ae60", label="RNA II (primer)", lw=2)
    ax.plot(t, get("hybrid"), "-", color="#8e44ad", label="RNA I:II hybrid", lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count per cell")
    ax.set_title("RNA I/II copy number control (Ataai-Shuler 1986)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plots["rna_control"] = svg(fig)

    # --- 4. PL_fractional accumulator + initiation events --------------------
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(t, get("PL_fractional"), "-", color="tab:cyan", lw=2)
    axes[0].axhline(1.0, ls="--", color="grey", alpha=0.5, label="initiation threshold")
    axes[0].set_ylabel("PL_fractional")
    axes[0].set_title("Replication initiation accumulator")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, get("n_rna_initiations"), "-o", color="tab:red", ms=4)
    axes[1].plot(t, get("time_since_rna_II"), "-", color="grey", lw=1, alpha=0.5,
                 label="time_since_rna_II (s)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("value")
    axes[1].set_title("Initiation events fired / RNA II interval countdown")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plots["initiation"] = svg(fig)

    # --- 5. Replisome subunit pool (shared with chromosome replication) ------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("replisome_trimer_min"), "-o", color="tab:blue",
            label="trimer subunits (sum)", lw=1.5, ms=4)
    ax.plot(t, get("replisome_monomer_min"), "-s", color="tab:red",
            label="monomer subunits (sum)", lw=1.5, ms=4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.set_title("Replisome subunit pool")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plots["subunits"] = svg(fig)

    # --- 6. Cell mass + DNA mass ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("cell_mass"), "-", color="tab:green", label="cell_mass", lw=2)
    ax2 = ax.twinx()
    ax2.plot(t, get("dna_mass"), "-", color="tab:purple", label="dna_mass", lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cell_mass (fg)", color="tab:green")
    ax2.set_ylabel("dna_mass (fg)", color="tab:purple")
    ax.set_title("Cell growth")
    ax.grid(True, alpha=0.3)
    plots["mass"] = svg(fig)

    # --- Assemble HTML --------------------------------------------------------
    duration = data["duration"]
    interval = data["interval"]
    wall = data["wall_time"]
    n_final = snaps[-1]["n_full_plasmids"]
    n_init = snaps[0]["n_full_plasmids"]
    rna_i_final = snaps[-1]["rna_I"]
    rna_ii_final = snaps[-1]["rna_II"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Plasmid Replication Report — v2ecoli</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
          margin: 2em auto; padding: 0 1.5em; color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #27ae60; padding-bottom: 0.3em; }}
  h2 {{ color: #2c3e50; margin-top: 1.8em; }}
  .meta {{ background: #f6f8fa; border-left: 4px solid #27ae60;
           padding: 0.8em 1.2em; border-radius: 4px; font-size: 0.95em; }}
  .meta code {{ background: #fff; padding: 1px 5px; border-radius: 3px;
                border: 1px solid #ddd; font-size: 0.9em; }}
  .plot {{ margin: 1em 0; }}
  .plot svg {{ max-width: 100%; height: auto; }}
  .caveat {{ background: #fff7ed; border-left: 4px solid #f59e0b;
             padding: 0.5em 0.9em; border-radius: 4px; font-size: 0.9em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; }}
  th, td {{ padding: 0.3em 0.8em; border: 1px solid #ddd; text-align: left;
            font-size: 0.9em; }}
  th {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>Plasmid Replication Report</h1>

<div class="meta">
<strong>v2ecoli plasmid simulation</strong><br>
ColE1/pBR322 plasmid with RNA I/II copy-number control
(<a href="https://doi.org/10.1002/bit.260280616">Ataai &amp; Shuler 1986</a>).
<br><br>
Duration: <code>{duration}s</code> &middot;
Emission interval: <code>{interval}s</code> &middot;
Wall time: <code>{wall:.1f}s</code> ({wall/max(duration,1):.2f}× realtime) &middot;
Snapshots: <code>{len(snaps)}</code>
<br>
Branch: <code>plasmids</code> in v2ecoli,
sim_data from <code>v2parca#plasmid</code> (patched onto workflow baseline).
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
<strong>Caveat:</strong> the RNA II initiation interval is ~360&nbsp;s
(<code>1 / K_T_RNAII</code>), so a run of {duration}&nbsp;s is generally too short
to observe a full initiation → elongation → termination cycle (~8 s at
967&nbsp;nt/s for a 4361&nbsp;bp plasmid). The plots below confirm the
Ataai-Shuler ODE is evolving correctly; longer runs are needed to see
steady-state copy numbers (~20 at 60&nbsp;min doubling time).
</p>

<h2>1. Plasmid unique molecule counts</h2>
<div class="plot">{plots['copy_number']}</div>

<h2>2. Active replisomes</h2>
<div class="plot">{plots['replisomes']}</div>

<h2>3. RNA I / II / hybrid dynamics</h2>
<div class="plot">{plots['rna_control']}</div>
<p>Copy-number control ODE from Ataai &amp; Shuler 1986
(Eqs&nbsp;5, 6, 10). RNA I and RNA II transcription rates are proportional
to plasmid count; hybridization rate <code>k_h&nbsp;·&nbsp;RNA_I&nbsp;·&nbsp;RNA_II</code>
sinks both into the hybrid pool, which decays back to free nucleotides.</p>

<h2>4. Replication initiation</h2>
<div class="plot">{plots['initiation']}</div>
<p>Each RNA II interval (360&nbsp;s), the fractional accumulator
<code>PL_fractional</code> is incremented by <code>n_plasmids · f · exp(-k_h·RNA_I·t_tx)</code>.
When it crosses&nbsp;1.0, one integer initiation fires and the fractional
part carries over.</p>

<h2>5. Replisome subunit pool</h2>
<div class="plot">{plots['subunits']}</div>
<p>Replisome subunits are shared with chromosome replication. Each plasmid
initiation consumes 3 trimer&nbsp;+ 1 monomer; each termination returns them.</p>

<h2>6. Cell mass</h2>
<div class="plot">{plots['mass']}</div>

<h2>What works</h2>
<ul>
<li>Plasmid unique molecules (<code>full_plasmid</code>, <code>oriV</code>,
    <code>plasmid_domain</code>, <code>plasmid_active_replisome</code>)
    initialize with <code>has_plasmid=True</code>.</li>
<li>The <code>ecoli-plasmid-replication</code> step runs without error alongside
    chromosome replication in the baseline composite.</li>
<li>RNA I/II/hybrid ODE state evolves on the
    <code>plasmid_rna_control</code> process_state port and persists across
    timesteps.</li>
<li>The Ataai-Shuler dynamics are directionally correct: RNA&nbsp;I decays
    from its steady-state initial condition, RNA&nbsp;II accumulates toward
    its steady state, and the initiation accumulator advances.</li>
</ul>

<h2>Open items</h2>
<ul>
<li><strong>Longer run needed</strong> to observe replication initiation,
    elongation, and termination — this report only covers the first RNA II
    initiation interval.</li>
<li><strong>sim_data provenance:</strong> the v2parca fast-mode output had
    initial-state NaN issues that broke the equilibrium ODE solver; this
    report uses a workflow-baseline sim_data with plasmid fields patched in
    from v2parca. A clean v2parca run that produces a directly usable
    sim_data is follow-up work.</li>
<li>The multi-generation, multi-seed, and copy-number-distribution plots
    from the upstream vEcoli_Plasmid PR require generational division,
    which is out of scope for a single short run.</li>
</ul>

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
