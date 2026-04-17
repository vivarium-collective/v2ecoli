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
MULTIGEN_IN = "out/plasmid/multigen_timeseries.json"
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

    # 7. DnaG depletion trajectory (parity with vEcoli plasmid_replication_report
    # Fig. 10). Fields are produced by scripts/run_plasmid_experiment.snapshot()
    # via the DNAG_INVESTIGATION_IDS dict.  If the fields are absent (older
    # timeseries), skip the section gracefully.
    dnag_fields = ("dnaG", "pol_core", "beta_clamp", "dnaB", "holA",
                   "lexA_mon", "lexA_dimer")
    has_dnag_data = all(f in snaps[0] for f in dnag_fields)
    if has_dnag_data:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 3.8))

        # Left: DnaG vs mechanistic-gate minimum (2 per oriC).
        dnag = get("dnaG")
        axA.plot(t, dnag, "-o", color="#dc2626", lw=2, ms=3.5,
                 label="DnaG (EG10239-MONOMER)")
        # Per-oriC minimum is 2; with 2 oriC (post-replication) the pool-level
        # requirement is 4.  Draw both as horizontal dashed lines.
        axA.axhline(2, ls="--", color="#64748b", lw=1, alpha=0.7,
                    label="min per oriC = 2")
        axA.axhline(4, ls=":",  color="#64748b", lw=1, alpha=0.7,
                    label="min for 2 oriC = 4")
        axA.set_xlabel("time (s)"); axA.set_ylabel("count")
        axA.set_title("DnaG (DNA primase) — mechanistic-gate subunit")
        axA.legend(loc="best", fontsize=8); axA.grid(True, alpha=0.3)

        # Right: other gate subunits for context.
        axB.plot(t, get("pol_core"),   "-", color="#2563eb", lw=1.3,
                 label="pol III core")
        axB.plot(t, get("beta_clamp"), "-", color="#0891b2", lw=1.3,
                 label="β clamp")
        axB.plot(t, get("dnaB"),       "-", color="#16a34a", lw=1.3,
                 label="DnaB helicase")
        axB.plot(t, get("holA"),       "-", color="#ea580c", lw=1.3,
                 label="HolA (δ)")
        axB.plot(t, get("dnaG"),       "-", color="#dc2626", lw=2.0,
                 label="DnaG (for reference)")
        axB.set_xlabel("time (s)"); axB.set_ylabel("count")
        axB.set_title("Replisome subunit pool — context")
        axB.legend(loc="best", fontsize=8, ncol=2); axB.grid(True, alpha=0.3)
        fig.tight_layout()
        plots["dnag"] = svg(fig)

        # LexA (regulator of dnaG via lexA→dnaG = -1.71 log₂) separate figure.
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(t, get("lexA_mon"),   "-", color="#6366f1", lw=1.5,
                label="LexA monomer (EG10534-MONOMER)")
        ax.plot(t, get("lexA_dimer"), "-", color="#a855f7", lw=1.8,
                label="LexA dimer (PC00010, active TF)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("count")
        ax.set_title("LexA regulator (0CS — always-active on dnaG)")
        ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
        plots["lexa"] = svg(fig)

        dnag_initial = int(dnag[0])
        dnag_final   = int(dnag[-1])
        dnag_min     = int(dnag.min())

    # 8. Multigeneration chromosome dynamics (if a multigen run is on disk).
    # Produced by scripts/run_plasmid_multigen.py; renders alongside the
    # single-generation plasmid plots for direct comparison.
    multigen = None
    if os.path.exists(MULTIGEN_IN):
        with open(MULTIGEN_IN) as f:
            multigen = json.load(f)

    if multigen is not None:
        gens = multigen["generations"]

        # Build a concatenated time axis (cumulative across generations).
        cum_t, cum_off, boundaries = [], 0.0, []
        per_gen_snaps = []
        for g in gens:
            gsnaps = g["snapshots"]
            for s in gsnaps:
                cum_t.append(s["time"] + cum_off)
            per_gen_snaps.append(gsnaps)
            cum_off += g["duration"]
            boundaries.append(cum_off)
        cum_t = np.array(cum_t)
        flat = [s for g in gens for s in g["snapshots"]]
        mg = lambda k: np.array([s.get(k, 0) for s in flat], dtype=float)

        # (a) DnaG + mechanistic-gate subunits across the whole lineage.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8))
        ax1.plot(cum_t / 60, mg("dnaG"), "-", color="#dc2626", lw=1.6,
                 label="DnaG (primase)")
        ax1.axhline(2, ls="--", color="#64748b", lw=1, alpha=0.7,
                    label="min per oriC = 2")
        ax1.axhline(4, ls=":",  color="#64748b", lw=1, alpha=0.7,
                    label="min for 2 oriC")
        for b in boundaries[:-1]:
            ax1.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
        ax1.set_xlabel("cumulative time (min)")
        ax1.set_ylabel("DnaG count")
        ax1.set_title("DnaG across generations "
                      f"(n={len(gens)} gens)")
        ax1.legend(fontsize=8, loc="best"); ax1.grid(True, alpha=0.3)

        ax2.plot(cum_t / 60, mg("pol_core"),   "-", color="#2563eb",
                 lw=1.0, label="pol III core")
        ax2.plot(cum_t / 60, mg("beta_clamp"), "-", color="#0891b2",
                 lw=1.0, label="β clamp")
        ax2.plot(cum_t / 60, mg("dnaB"),       "-", color="#16a34a",
                 lw=1.0, label="DnaB helicase")
        ax2.plot(cum_t / 60, mg("holA"),       "-", color="#ea580c",
                 lw=1.0, label="HolA (δ)")
        ax2.plot(cum_t / 60, mg("dnaG"),       "-", color="#dc2626",
                 lw=1.8, label="DnaG")
        for b in boundaries[:-1]:
            ax2.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
        ax2.set_xlabel("cumulative time (min)")
        ax2.set_ylabel("count")
        ax2.set_title("Replisome subunit pool")
        ax2.legend(fontsize=8, loc="best", ncol=2); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        plots["multigen_dnag"] = svg(fig)

        # (b) Dry mass across generations — classic "chromosome dynamics" view.
        fig, (axm, axfc) = plt.subplots(1, 2, figsize=(13, 3.8))
        axm.plot(cum_t / 60, mg("dry_mass"), "-", color="#0f172a", lw=1.4,
                 label="dry_mass")
        axm.plot(cum_t / 60, mg("protein_mass"), "-", color="#22c55e",
                 lw=1.0, alpha=0.8, label="protein")
        axm.plot(cum_t / 60, mg("dna_mass"),     "-", color="#8b5cf6",
                 lw=1.0, alpha=0.8, label="DNA")
        for b in boundaries[:-1]:
            axm.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
        axm.set_xlabel("cumulative time (min)")
        axm.set_ylabel("mass (fg)")
        axm.set_title("Mass across generations")
        axm.legend(fontsize=8, loc="best"); axm.grid(True, alpha=0.3)

        # Per-gen dry-mass fold change normalized to each gen's start.
        for g in gens:
            gs = g["snapshots"]
            gt = np.array([s["time"] for s in gs])
            dm = np.array([s["dry_mass"] for s in gs])
            if dm.size == 0 or dm[0] == 0:
                continue
            axfc.plot(gt / 60, dm / dm[0], lw=1.1,
                      label=f"gen {g['index']}")
        axfc.axhline(2.0, ls="--", color="#64748b", alpha=0.7,
                     label="2× (division)")
        axfc.set_xlabel("time within generation (min)")
        axfc.set_ylabel("dry-mass fold change")
        axfc.set_title("Per-generation fold change")
        axfc.legend(fontsize=7, loc="best", ncol=2)
        axfc.grid(True, alpha=0.3)
        fig.tight_layout()
        plots["multigen_mass"] = svg(fig)

        # (c) Chromosome + plasmid unique-molecule counts across the lineage.
        fig, (axc, axp) = plt.subplots(1, 2, figsize=(13, 3.8))
        axc.plot(cum_t / 60, mg("n_full_chromosomes"),  "-",
                 color="#0ea5e9", lw=1.1, label="full_chromosome")
        axc.plot(cum_t / 60, mg("n_active_replisomes"), "-",
                 color="#dc2626", lw=1.1, label="active_replisome")
        for b in boundaries[:-1]:
            axc.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
        axc.set_xlabel("cumulative time (min)"); axc.set_ylabel("count")
        axc.set_title("Chromosome state")
        axc.legend(fontsize=8, loc="best"); axc.grid(True, alpha=0.3)

        axp.plot(cum_t / 60, mg("n_full_plasmids"),         "-o",
                 color="#22c55e", lw=1.1, ms=1.5,
                 label="full_plasmid")
        axp.plot(cum_t / 60, mg("n_oriV"),                   "-",
                 color="#f59e0b", lw=1.0, alpha=0.8,
                 label="oriV")
        axp.plot(cum_t / 60, mg("n_plasmid_active_replisomes"), "-",
                 color="#8b5cf6", lw=1.0, alpha=0.8,
                 label="plasmid_active_replisome")
        for b in boundaries[:-1]:
            axp.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
        axp.set_xlabel("cumulative time (min)"); axp.set_ylabel("count")
        axp.set_title("Plasmid state")
        axp.legend(fontsize=8, loc="best"); axp.grid(True, alpha=0.3)
        fig.tight_layout()
        plots["multigen_copies"] = svg(fig)

        # Per-gen summary table rows.
        mg_rows = ""
        for g in gens:
            gs = g["snapshots"]
            dm0 = gs[0]["dry_mass"]
            dmf = gs[-1]["dry_mass"]
            dg0 = gs[0]["dnaG"]
            dgf = gs[-1]["dnaG"]
            dgmin = min(s["dnaG"] for s in gs)
            pf = gs[-1]["n_full_plasmids"]
            mg_rows += (
                f"<tr><td>{g['index']}</td>"
                f"<td>{g['duration']:.0f}</td>"
                f"<td>{g['wall_time']:.0f}</td>"
                f"<td>{dm0:.0f} → {dmf:.0f}</td>"
                f"<td>{dg0} → {dgf} (min {dgmin})</td>"
                f"<td>{pf}</td>"
                f"<td class=\"{'green' if g['divided'] else 'red'}\">"
                f"{'divided' if g['divided'] else 'stalled'}</td></tr>"
            )
        multigen_wall = multigen["pipeline_wall_time"]
        n_gens_done = len(gens)
        n_gens_req = multigen["n_generations_requested"]

    duration = data["duration"]
    interval = data["interval"]
    wall = data["wall_time"]
    n_final = snaps[-1]["n_full_plasmids"]
    n_init = snaps[0]["n_full_plasmids"]
    rna_i_final = snaps[-1]["rna_I"]
    rna_ii_final = snaps[-1]["rna_II"]

    network_file, n_proc, n_store, n_edges = build_network_html()

    dnag_toc_link = (
        '  <a href="#dnag">7. DnaG</a>\n' if has_dnag_data else ""
    )
    multigen_toc_link = (
        '  <a href="#multigen">8. Multigeneration</a>\n'
        if multigen is not None else ""
    )
    toc = f"""
<nav class="toc">
  <strong>Jump to:</strong>
  <a href="#summary">Summary</a>
  <a href="#copy-number">1. Plasmid counts</a>
  <a href="#replisomes">2. Active replisomes</a>
  <a href="#rna">3. RNA I/II</a>
  <a href="#initiation">4. Initiation</a>
  <a href="#subunits">5. Subunits</a>
  <a href="#mass">6. Mass</a>
{dnag_toc_link}{multigen_toc_link}  <a href="#network">Network</a>
</nav>
"""

    dnag_section = ""
    if has_dnag_data:
        dnag_section = f"""
<h2 id="dnag">7. DnaG depletion — single-generation trajectory</h2>
<p>
  Parity check with the vEcoli plasmid investigation
  (<code>vEcoli/reports/plasmid_replication_report.html</code>, Fig. 10).
  The vEcoli report identified <strong>four compounding regulatory-pipeline
  errors</strong> that cause DnaG to drift from ~6 molecules/cell at generation 1
  down to 0–1 by generation 5–6. All four have been
  <a href="plasmid_dnag_investigation_v2ecoli.html">confirmed in v2ecoli</a>:
  the inflated <code>lexA→dnaG = −1.71 log₂</code> NCA fold change; LexA
  modelled as <code>0CS</code> (always active); cistron-level FC application;
  and the <code>_build_rna_sequences</code> dedup that drops TU00434 &amp; TU00435.
  The plots below are the <em>within-generation</em> trajectory — a
  single-generation baseline that the multigeneration depletion compounds on top of.
</p>
<div class="plot">{plots['dnag']}</div>
<table>
<tr><th>Metric</th><th>Value (single gen)</th></tr>
<tr><td>DnaG at t=0</td><td>{dnag_initial}</td></tr>
<tr><td>DnaG at division</td><td>{dnag_final}</td></tr>
<tr><td>DnaG minimum within the generation</td><td>{dnag_min}</td></tr>
<tr><td>Mechanistic-gate minimum per oriC</td><td>2</td></tr>
</table>
<p class="caveat">
  Generation 1 starts well above the gate threshold (<code>DnaG</code>
  initialises at ~6–10 per the parca). Multigenerational depletion is not yet
  directly observable in v2ecoli because <code>reports/multigeneration_report.py</code>
  rebuilds each daughter as a fresh <code>Composite</code> (a workaround for
  the <code>listeners/growth_limits</code> realize-Array bug that currently
  blocks in-place structural division). Once that workaround is wired into the
  workflow, the gen-over-gen trajectory analogous to vEcoli Fig. 10 can be
  drawn here.
</p>

<h3>LexA regulator (parity reference)</h3>
<p>
  LexA is classified as <code>0CS</code> (zero-component system) in
  <code>v2ecoli/processes/parca/reconstruction/ecoli/flat/condition/tf_condition.tsv:17</code>
  and has no DNA-damage activation path in v2ecoli — so it sits near-saturated
  on all 53 of its targets (including the <code>EG10239_RNA</code>/dnaG
  cistron) for the entire run.
</p>
<div class="plot">{plots['lexa']}</div>
"""

    multigen_section = ""
    if multigen is not None:
        multigen_section = f"""
<h2 id="multigen">8. Chromosome dynamics — multigeneration lineage</h2>
<p>
  Single-lineage run through a plasmid-enabled cache: start from the
  baseline newborn, run to division, keep daughter 1, rebuild a fresh
  <code>Composite</code>, repeat. Ran {n_gens_done}/{n_gens_req}
  generations in {multigen_wall:.0f}s wall. Snapshot interval:
  <code>{multigen['snapshot_interval']}s</code>. All figures below are
  on one <strong>cumulative time axis</strong> — dashed vertical lines are
  generation boundaries.
</p>

<h3>DnaG across generations</h3>
<p>
  The vEcoli plasmid investigation showed DnaG drifting from ~6/cell at
  gen 1 to 0–1/cell by gen 5–6 — the pattern that triggers the mechanistic
  <code>TimeLimitError</code>. The corresponding trajectory in v2ecoli is
  plotted below. Horizontal dashed lines: DnaG minima required by the
  replisome gate (2 per oriC; 4 for a post-replication 2-oriC cell).
</p>
<div class="plot">{plots['multigen_dnag']}</div>

<h3>Mass across generations</h3>
<div class="plot">{plots['multigen_mass']}</div>

<h3>Chromosome &amp; plasmid unique-molecule counts</h3>
<div class="plot">{plots['multigen_copies']}</div>

<h3>Per-generation summary</h3>
<table>
<tr><th>Gen</th><th>Sim time (s)</th><th>Wall (s)</th>
  <th>Dry mass (fg)</th><th>DnaG</th>
  <th>Plasmid count (end)</th><th>Status</th></tr>
{mg_rows}
</table>
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
{dnag_section}
{multigen_section}

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
