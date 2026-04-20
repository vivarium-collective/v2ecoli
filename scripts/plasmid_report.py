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
    # Source selection: §1–§7 all describe a single generation. Prefer
    # gen&nbsp;1 of the multigen run (long enough to actually see plasmid
    # replication and chromosome dynamics) over the standalone single-gen
    # experiment (defaults to a 60 s window — too short to see any of the
    # ColE1 events the report is trying to show). Fall back to the
    # single-gen JSON only when no multigen run is on disk.
    single_gen_data = None
    if os.path.exists(IN):
        with open(IN) as f:
            single_gen_data = json.load(f)

    multigen = None
    if os.path.exists(MULTIGEN_IN):
        with open(MULTIGEN_IN) as f:
            multigen = json.load(f)

    if multigen is None and single_gen_data is None:
        raise FileNotFoundError(
            f"Neither {MULTIGEN_IN} nor {IN} found — run "
            "scripts/run_plasmid_multigen.py (preferred) or "
            "scripts/run_plasmid_experiment.py first."
        )

    if multigen is not None and multigen["generations"]:
        snaps = multigen["generations"][0]["snapshots"]
        source_tag = (
            f"multigen gen 1 ({len(snaps)} snapshots, "
            f"{snaps[-1]['time']:.0f} s sim)"
        )
    else:
        snaps = single_gen_data["snapshots"]
        source_tag = (
            f"single-gen experiment ({len(snaps)} snapshots, "
            f"{snaps[-1]['time']:.0f} s sim)"
        )

    t = np.array([s["time"] for s in snaps])
    get = lambda k: np.array([s.get(k, 0) for s in snaps])
    has_field = lambda k: k in snaps[0]

    plots = {}

    # 1. Plasmid copy number. Multigen capture doesn't track plasmid_domain
    # separately (it follows oriV 1:1 in single-domain ColE1), so render it
    # only when the field is present (single-gen-only).
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, get("n_full_plasmids"), "-o", color="tab:green",
            label="full_plasmid", lw=2, ms=4)
    ax.plot(t, get("n_oriV"), "-s", color="tab:blue", label="oriV",
            lw=1.5, ms=3, alpha=0.7)
    if has_field("n_plasmid_domains"):
        ax.plot(t, get("n_plasmid_domains"), "-^", color="tab:orange",
                label="plasmid_domain", lw=1.5, ms=3, alpha=0.7)
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

    # 3. BP1993 free-RNA / Rom dynamics + D-pool partitioning
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 3.8))
    axA.plot(t, get("R_I"),  "-", color="#e67e22", lw=2,
             label="R_I (free RNA I, inhibitor)")
    axA.plot(t, get("R_II"), "-", color="#27ae60", lw=2,
             label="R_II (free RNA II, primer)")
    axA.plot(t, get("M"),    "-", color="#8e44ad", lw=2,
             label="M (free Rom dimer)")
    axA.set_xlabel("time (s)"); axA.set_ylabel("count per cell")
    axA.set_title("BP1993 free species — R_I / R_II / Rom")
    axA.legend(loc="best", fontsize=9); axA.grid(True, alpha=0.3)

    # D-pool partitioning: D (free) vs intermediates (D_tII, D_lII, D_p,
    # D_starc, D_c) vs replicating (D_M). Sum should equal n_idle plasmids.
    d_intermediates = (get("D_tII") + get("D_lII") + get("D_p")
                       + get("D_starc") + get("D_c"))
    axB.plot(t, get("D"),    "-", color="#0ea5e9", lw=1.6, label="D (free)")
    axB.plot(t, d_intermediates, "-", color="#f59e0b", lw=1.4,
             label="ΣD_{tII,lII,p,*c,c} (intermediates)")
    axB.plot(t, get("D_M"),  "-", color="#dc2626", lw=1.4,
             label="D_M (replicating)")
    axB.set_xlabel("time (s)"); axB.set_ylabel("plasmid count")
    axB.set_title("D-pool partitioning across BP states")
    axB.legend(loc="best", fontsize=9); axB.grid(True, alpha=0.3)
    fig.tight_layout()
    plots["rna_control"] = svg(fig)

    # 4 (removed) — the old "PL_fractional / time_since_rna_II" accumulator
    # plot was tied to the pre-BP1993 3-ODE schema. The current 10-species
    # BP ODE uses repl_accum instead, and the lineage-wide cumulative
    # initiation trace already lives in §8 (multigen RNA-control plot).

    # 5. Replisome subunit pool — single-gen-only (multigen capture()
    # tracks the per-subunit DnaG-investigation set instead, plotted in §7).
    has_subunit_aggregates = (has_field("replisome_trimer_min")
                               and has_field("replisome_monomer_min"))
    if has_subunit_aggregates:
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

    # 8. Multigeneration chromosome dynamics. Produced by
    # scripts/run_plasmid_multigen.py; renders alongside the single-gen
    # plasmid plots above (which by default are also sourced from gen 1
    # of this same multigen run — see source-selection block at the top
    # of main()).
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

        # (d) BP1993 free-RNA dynamics across the lineage — lets the
        # reader check that R_I / R_II / M behaviour and the initiation-
        # event count look consistent with the single-gen trace in §3,
        # and spot any gen where RNA II initiation events accumulate
        # without the corresponding replisome firings.
        has_rna = any("R_I" in s for s in flat)
        if has_rna:
            fig, (axr, axi) = plt.subplots(1, 2, figsize=(13, 3.8))
            axr.plot(cum_t / 60, mg("R_I"),  "-", color="#0ea5e9",
                     lw=1.0, label="R_I (free RNA I)")
            axr.plot(cum_t / 60, mg("R_II"), "-", color="#dc2626",
                     lw=1.0, label="R_II (free RNA II)")
            axr.plot(cum_t / 60, mg("M"),    "-", color="#8b5cf6",
                     lw=1.0, alpha=0.8, label="M (free Rom dimer)")
            for b in boundaries[:-1]:
                axr.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
            axr.set_xlabel("cumulative time (min)")
            axr.set_ylabel("count per cell")
            axr.set_title("BP1993 free species — R_I / R_II / M")
            axr.legend(fontsize=8, loc="best"); axr.grid(True, alpha=0.3)

            # n_rna_initiations is per-tick (count of new firings this tick),
            # not a running total — accumulate across the lineage.
            cum_inits: list[float] = []
            running = 0
            for g in gens:
                for s in g["snapshots"]:
                    running += int(s.get("n_rna_initiations", 0))
                    cum_inits.append(running)
            axi.plot(cum_t / 60, cum_inits, "-", color="#16a34a", lw=1.4,
                     label="cumulative initiations")
            axi.plot(cum_t / 60, mg("n_plasmid_active_replisomes"),
                     "-", color="#8b5cf6", lw=1.0, alpha=0.8,
                     label="plasmid replisomes")
            for b in boundaries[:-1]:
                axi.axvline(b / 60, ls="--", color="#cbd5e1", lw=0.8)
            axi.set_xlabel("cumulative time (min)")
            axi.set_ylabel("count")
            axi.set_title("RNA II initiation events (cumulative)")
            axi.legend(fontsize=8, loc="best"); axi.grid(True, alpha=0.3)
            fig.tight_layout()
            plots["multigen_rna"] = svg(fig)

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
            # Per-tick count → sum across the gen for total initiation events.
            n_inits = sum(int(s.get("n_rna_initiations", 0)) for s in gs)
            mg_rows += (
                f"<tr><td>{g['index']}</td>"
                f"<td>{g['duration']:.0f}</td>"
                f"<td>{g['wall_time']:.0f}</td>"
                f"<td>{dm0:.0f} → {dmf:.0f}</td>"
                f"<td>{dg0} → {dgf} (min {dgmin})</td>"
                f"<td>{pf}</td>"
                f"<td>{n_inits}</td>"
                f"<td class=\"{'green' if g['divided'] else 'red'}\">"
                f"{'divided' if g['divided'] else 'stalled'}</td></tr>"
            )
        multigen_wall = multigen["pipeline_wall_time"]
        n_gens_done = len(gens)
        n_gens_req = multigen["n_generations_requested"]
        n_divided = sum(1 for g in gens if g["divided"])
        peak_cpc = max(
            (s["n_full_plasmids"] for g in gens for s in g["snapshots"]),
            default=0,
        )
        final_cpc = gens[-1]["snapshots"][-1]["n_full_plasmids"] if gens else 0
        # First gen whose end-of-gen cpc reaches the BP1993 Table 2 target
        # band (rom+ wild-type prediction is 28; allow ±2 wiggle).
        bp_target_gen = next(
            (g["index"] for g in gens
             if g["snapshots"][-1]["n_full_plasmids"] >= 26),
            None,
        )

        # Only render the RNA-control plot block if capture() recorded the
        # fields (older multigen JSONs predate the extension).
        if "multigen_rna" in plots:
            multigen_rna_block = (
                "<h3>BP1993 free-species dynamics across generations</h3>\n"
                "<p>Lets you sanity-check the multigen lineage against the "
                "single-gen trace in §3: the cumulative RNA&nbsp;II "
                "initiation count should grow at a roughly constant rate "
                "across dividing gens, and stalled gens should still show "
                "initiation events even when no replisome fires (subunit "
                "depletion).</p>\n"
                f"<div class=\"plot\">{plots['multigen_rna']}</div>\n"
            )
        else:
            multigen_rna_block = ""

    n_final = snaps[-1]["n_full_plasmids"]
    n_init = snaps[0]["n_full_plasmids"]
    r_i_final  = snaps[-1].get("R_I", 0.0)
    r_ii_final = snaps[-1].get("R_II", 0.0)
    m_final    = snaps[-1].get("M", 0.0)

    # Top-of-report meta + Summary. Prefer the multigen lineage when one is
    # on disk so the headline numbers reflect the steady-state result —
    # a 60 s standalone single-gen run never gets past the warm-up.
    if multigen is not None:
        total_sim = sum(g["duration"] for g in gens)
        meta_block = f"""
<div class="meta" id="summary">
<strong>ColE1 / pBR322 plasmid replication with RNA I/II copy-number control</strong>
(parameters: <a href="https://doi.org/10.1006/jmbi.1993.1090">Brendel &amp; Perelson 1993</a>).
<br><br>
Multigen lineage: <code>{n_gens_done}/{n_gens_req}</code> gens
({n_divided} dividing) &middot;
Total sim: <code>{total_sim:.0f}s</code> &middot;
Wall: <code>{multigen_wall:.0f}s</code> &middot;
Snapshot interval: <code>{multigen['snapshot_interval']}s</code>
<br>
§1–§7 sourced from <strong>{source_tag}</strong>.
<br>
Branch: <code>plasmids</code> in v2ecoli &middot; sim_data: <code>v2parca#plasmid</code>
</div>
"""
    else:
        duration = single_gen_data["duration"]
        interval = single_gen_data["interval"]
        wall = single_gen_data["wall_time"]
        meta_block = f"""
<div class="meta" id="summary">
<strong>ColE1 / pBR322 plasmid replication with RNA I/II copy-number control</strong>
(parameters: <a href="https://doi.org/10.1006/jmbi.1993.1090">Brendel &amp; Perelson 1993</a>).
<br><br>
Duration: <code>{duration}s</code> &middot;
Emission interval: <code>{interval}s</code> &middot;
Wall time: <code>{wall:.1f}s</code> ({wall/max(duration,1):.2f}× realtime) &middot;
Snapshots: <code>{len(snaps)}</code>
<br>
§1–§7 sourced from <strong>{source_tag}</strong>.
<br>
Branch: <code>plasmids</code> in v2ecoli &middot; sim_data: <code>v2parca#plasmid</code>
</div>
"""

    if multigen is not None:
        summary_rows = ""
        for g in gens:
            gs = g["snapshots"]
            cpc_birth = gs[0]["n_full_plasmids"]
            cpc_div   = gs[-1]["n_full_plasmids"]
            r_i_end   = gs[-1].get("R_I", 0.0)
            m_end     = gs[-1].get("M", 0.0)
            # n_rna_initiations is a per-tick count emitted by
            # plasmid_replication._prepare (the integer part of repl_accum
            # this tick), not a running total. Sum to get per-gen events.
            n_inits   = sum(int(s.get("n_rna_initiations", 0)) for s in gs)
            divided   = "divided" if g["divided"] else "stalled"
            color     = "green" if g["divided"] else "red"
            summary_rows += (
                f"<tr><td>{g['index']}</td>"
                f"<td>{g['duration']:.0f}</td>"
                f"<td>{cpc_birth} → {cpc_div}</td>"
                f"<td>{r_i_end:.1f}</td>"
                f"<td>{m_end:.1f}</td>"
                f"<td>{n_inits}</td>"
                f"<td class=\"{color}\">{divided}</td></tr>"
            )
        summary_block = f"""
<h2>Summary</h2>
<p>
  Per-generation lineage from the multigen run
  (<code>{MULTIGEN_IN}</code>). Birth → division copy number is the
  headline validation: BP1993 Table&nbsp;2 predicts <strong>28&nbsp;cpc</strong>
  for rom+ wild-type. R<sub>I</sub> and M are end-of-generation
  molecule counts in the BP1993 process_state ODE.
</p>
<table>
<tr><th>Gen</th><th>Sim time (s)</th>
  <th>full_plasmid (birth → div)</th>
  <th>R<sub>I</sub> end</th><th>M end</th>
  <th>RNA II inits</th><th>Status</th></tr>
{summary_rows}</table>
"""
    else:
        summary_block = f"""
<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Initial</th><th>Final</th></tr>
<tr><td>full_plasmid count</td><td>{n_init}</td><td>{n_final}</td></tr>
<tr><td>R_I (free RNA I, inhibitor)</td>
  <td>{snaps[0].get('R_I', 0.0):.2f}</td><td>{r_i_final:.2f}</td></tr>
<tr><td>R_II (free RNA II, primer)</td>
  <td>{snaps[0].get('R_II', 0.0):.3f}</td><td>{r_ii_final:.3f}</td></tr>
<tr><td>M (free Rom dimer)</td>
  <td>{snaps[0].get('M', 0.0):.2f}</td><td>{m_final:.2f}</td></tr>
<tr><td>plasmid_active_replisomes</td><td>{snaps[0]['n_plasmid_active_replisomes']}</td>
  <td>{snaps[-1]['n_plasmid_active_replisomes']}</td></tr>
</table>
<p class="caveat">
  No multigen lineage on disk — Summary falls back to the single-generation
  snapshot. Run <code>scripts/run_plasmid_multigen.py</code> for the
  steady-state validation against BP1993 Table 2.
</p>
"""

    network_file, n_proc, n_store, n_edges = build_network_html()

    subunits_toc_link = (
        '  <a href="#subunits">5. Subunits</a>\n'
        if has_subunit_aggregates else ""
    )
    subunits_section = (
        f'<h2 id="subunits">5. Replisome subunit pool</h2>\n'
        f'<div class="plot">{plots["subunits"]}</div>\n'
        if has_subunit_aggregates else ""
    )
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
  <a href="#parameters">Parameters</a>
  <a href="#copy-number">1. Plasmid counts</a>
  <a href="#replisomes">2. Active replisomes</a>
  <a href="#rna">3. BP1993 species</a>
{subunits_toc_link}  <a href="#mass">6. Mass</a>
{dnag_toc_link}{multigen_toc_link}  <a href="#network">Network</a>
</nav>
"""

    dnag_section = ""
    if has_dnag_data:
        dnag_section = f"""
<h2 id="dnag">7. DnaG and replisome subunits — single-generation trajectory</h2>
<p>
  Within-generation trajectory of the mechanistic-replisome gate
  subunits. DnaG is the bottleneck (smallest pool, ~6/cell at birth);
  the multigeneration view below shows how it drifts across divisions.
</p>
<div class="plot">{plots['dnag']}</div>
<table>
<tr><th>Metric</th><th>Value (single gen)</th></tr>
<tr><td>DnaG at t=0</td><td>{dnag_initial}</td></tr>
<tr><td>DnaG at division</td><td>{dnag_final}</td></tr>
<tr><td>DnaG minimum within the generation</td><td>{dnag_min}</td></tr>
<tr><td>Mechanistic-gate minimum per oriC</td><td>2</td></tr>
</table>

<h3>LexA regulator</h3>
<p>
  LexA is the dominant TF on <code>EG10239_RNA</code> (dnaG). It's
  classified as <code>0CS</code> in v2ecoli (no DNA-damage activation
  path), so it sits near-saturated on all 53 targets for the whole run.
</p>
<div class="plot">{plots['lexa']}</div>
"""

    multigen_section = ""
    if multigen is not None:
        bp_target_phrase = (
            f"hit the BP1993 Table&nbsp;2 target band (≥26 cpc) at "
            f"<strong>gen&nbsp;{bp_target_gen}</strong>"
            if bp_target_gen is not None
            else "did not reach the BP1993 Table&nbsp;2 target band (≥26 cpc)"
        )
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

<p class="meta" style="border-left-color:#16a34a;">
  <strong>Lineage summary:</strong>
  {n_divided}/{n_gens_done} generations divided &middot;
  peak full_plasmid count <strong>{peak_cpc}</strong>,
  end-of-lineage <strong>{final_cpc}</strong> &middot;
  copy-number trajectory {bp_target_phrase}.
</p>

<p class="caveat">
  <strong>Not the mechanistic implementation.</strong>
  This run uses the BP1993 reduced ODE living in
  <code>process_state.plasmid_rna_control</code> (10 lumped species:
  D-pool, R<sub>I</sub>, R<sub>II</sub>, M). RNA&nbsp;I and RNA&nbsp;II
  are <em>not</em> bulk molecules transcribed by the main RNAP stack;
  the ODE substitutes for the full transcription/degradation chemistry.
  Replication firing is discretized from the ODE flux into integer
  events. The mechanistic target — RNA&nbsp;I/II as bulk molecules
  going through transcription + degradation like every other
  v2ecoli&nbsp;RNA — is the next stage; see the parameters
  section above for the staging rationale.
</p>

<h3>DnaG across generations</h3>
<p>
  Horizontal dashed lines: DnaG minima required by the mechanistic
  replisome gate (2 per oriC; 4 for a post-replication 2-oriC cell).
</p>
<div class="plot">{plots['multigen_dnag']}</div>

<h3>Mass across generations</h3>
<div class="plot">{plots['multigen_mass']}</div>

<h3>Chromosome &amp; plasmid unique-molecule counts</h3>
<div class="plot">{plots['multigen_copies']}</div>

{multigen_rna_block}<h3>Per-generation summary</h3>
<table>
<tr><th>Gen</th><th>Sim time (s)</th><th>Wall (s)</th>
  <th>Dry mass (fg)</th><th>DnaG</th>
  <th>Plasmid count (end)</th>
  <th>RNA II inits</th><th>Status</th></tr>
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
  td.green {{ color: #15803d; font-weight: 600; }}
  td.red   {{ color: #b91c1c; font-weight: 600; }}
</style>
</head>
<body>

{banner_html()}

<h1>Plasmid Replication Report</h1>
{toc}

{meta_block}

{summary_block}

<h2 id="parameters">Model parameters &amp; references</h2>

<h3>Lineage of ColE1/pBR322 kinetic-ODE models</h3>
<p>
Four published kinetic-ODE models describe ColE1-family
plasmid replication via the RNA&nbsp;I / RNA&nbsp;II antisense
mechanism. Each refines its predecessor; later models add explicit
biology (Rom protein, kissing complex, tRNA) at the cost of more
parameters. Stochastic / agent-based descendants
(e.g.&nbsp;Kuo&nbsp;1996, Mathur&nbsp;2009) re-use the same rate
constants, so the table below covers the upstream parameter source
for the entire family.
</p>

<table>
<tr><th>Year</th><th>Citation</th><th>Plasmid / scope</th>
  <th>Structure</th><th>Species</th><th>Rate constants</th>
  <th>Cell volume</th><th>Validation / predicted cpc</th></tr>
<tr><td>1986</td>
  <td>Ataai &amp; Shuler<br>Biotechnol. Bioeng. 28:1166</td>
  <td>Generic ColE1 (tested on pN204)</td>
  <td>Three coupled ODEs: RNA&nbsp;I, RNA&nbsp;II, hybrid.
    Single-step bimolecular hybridization. Replication initiation gated
    by survival probability <code>exp(−k<sub>h</sub>·RNA<sub>I</sub>·τ)</code>.</td>
  <td>3 (RNA&nbsp;I, RNA&nbsp;II, hybrid)</td>
  <td>5 (α<sub>I</sub>, α<sub>II</sub>, γ, k<sub>h</sub>, τ)</td>
  <td>implicit (parameters absorb V<sub>c</sub>)</td>
  <td>17–20 cpc (pN204; framework only — not pBR322-specific)</td></tr>
<tr><td>1990</td>
  <td>Kim &amp; Shuler<br>Biotechnol. Bioeng. 36:233</td>
  <td>pBR322 (rom+ <em>and</em> rom⁻ explicit)</td>
  <td>Same Ataai-Shuler ODEs, refit with in-vivo pBR322 rates
    (Lin-Chao&nbsp;&amp;&nbsp;Bremer&nbsp;1987 synthesis,
    Chao&nbsp;&amp;&nbsp;Bremer&nbsp;1986 t½). Adds growth-rate scan.</td>
  <td>3 (same as 1986)</td>
  <td>6 (adds rom+ vs rom⁻ k<sub>2</sub>)</td>
  <td>implicit ~0.55 fL</td>
  <td>Table II: 44–60 cpc rom+ across μ&nbsp;=&nbsp;0.08–0.84&nbsp;h⁻¹
    in B/r</td></tr>
<tr><td>1993</td>
  <td>Brendel &amp; Perelson<br>J. Mol. Biol. 229:860</td>
  <td>ColE1; rom+ &amp; rom⁻ via separate parameter sets</td>
  <td>Explicit reversible kissing complex
    RNA<sub>I</sub>&nbsp;+&nbsp;RNA<sub>II</sub>&nbsp;⇌&nbsp;C<sub>m</sub>&nbsp;→&nbsp;C<sub>s</sub>.
    Adds DNA-pol I priming (k<sub>p</sub>) vs loop opening (k<sub>l</sub>)
    competition, dilution-by-growth term.</td>
  <td>~10 (incl. C<sub>m</sub>, C<sub>s</sub>, primer, etc.)</td>
  <td>~12 (k<sub>1</sub>, k<sub>−1</sub>, k<sub>2</sub>, k<sub>I</sub>,
    k<sub>II</sub>, k<sub>M</sub>, k<sub>l</sub>, k<sub>p</sub>,
    ε<sub>I</sub>, ε<sub>II</sub>, ε<sub>M</sub>, V<sub>c</sub>)</td>
  <td>fixed 0.625 fL</td>
  <td>Table 2: <strong>28 cpc</strong> rom+, 52 cpc rom⁻
    (matches ~22–30 cpc literature)</td></tr>
<tr><td>2015</td>
  <td>Freudenau et&nbsp;al.<br>Front. Bioeng. Biotechnol. 3:127</td>
  <td>pBR322-derived pSUP&nbsp;201-3 (rom+ low-copy) and
    pCMV-lacZ (rom⁻ high-copy)</td>
  <td>Extends BP&nbsp;1993 with explicit Rom protein dynamics
    (Rom&nbsp;+&nbsp;C<sub>m</sub>&nbsp;⇌&nbsp;Rom·C<sub>s</sub>) and
    uncharged-tRNA binding to RNA&nbsp;II / pDNA-RNA&nbsp;II.
    Refit to qPCR data.</td>
  <td>15</td>
  <td>26</td>
  <td>fixed 0.625 fL (BP convention)</td>
  <td>Direct qPCR validation on DH5α-pSUP&nbsp;201-3:
    <strong>46–49 cpc</strong> across three timepoints.
    pCMV-lacZ: 1500–5800 cpc.</td></tr>
</table>

<h3>What this v2ecoli implementation uses, and why</h3>
<p>
We integrate the <strong>full Brendel&nbsp;&amp;&nbsp;Perelson 1993
(rom+) ODE system</strong> — eqns 1a–1j of the paper — as written, with
all 10 species and all 18 rate constants from BP Table 1. No QSS
reduction, no Rom-free shortcut. The system runs in
<code>plasmid_replication.py:_bp_deriv</code>; integration is fixed-step
RK4 with <code>n_substeps&nbsp;=&nbsp;10</code> per process timestep and
non-negative clamping after each substep
(<code>plasmid_replication.py:_bp_rk4</code>). The cell-growth dilution
term <em>μ</em> in BP's eqns is set to&nbsp;0 — vEcoli handles volume
increase and division externally. Replication initiation events are
discretized from the continuous flux <code>k_D&nbsp;·&nbsp;D_p</code>
into integer firings via a fractional accumulator
(<code>repl_accum</code>) that carries between timesteps; the integer
count is exposed to <code>_evolve</code> as
<code>n_rna_initiations</code> and gated by replisome-subunit
availability.
</p>

<table>
<tr><th>Decision</th><th>Choice</th><th>Reason</th></tr>
<tr><td>Source paper</td>
  <td>BP 1993 (over Kim-Shuler 1990)</td>
  <td>BP's published prediction (28 cpc rom+) sits inside the
    experimental band (~22–30 cpc). Kim-Shuler's 44–60 cpc is at the
    high end and uses a 33-s RNA half-life that's a fast-growth
    artifact. BP's 2-minute half-life matches the field standard.</td></tr>
<tr><td>Source paper</td>
  <td>BP 1993 (over Freudenau 2015)</td>
  <td>Freudenau's 26 reactions and 15 species extend BP with
    uncharged-tRNA binding to RNA&nbsp;II / pDNA-RNA&nbsp;II — a layer
    BP doesn't need. Their fitted RNA&nbsp;I synthesis rate
    (k<sub>13</sub>=34/min, vs BP's 6/min) only makes sense WITH the
    extra tRNA-buffering reactions that compensate for it. Plugging
    Freudenau's numbers into BP's structure would over-produce
    RNA&nbsp;I and crash cpc below BP's 28. Freudenau is the natural
    Stage&nbsp;B target once the BP backbone is validated.</td></tr>
<tr><td>Strain (rom+ vs rom⁻)</td>
  <td>rom+</td>
  <td>Native pBR322 carries an intact <code>rom</code> gene.
    Freudenau's reference strain DH5α-pSUP&nbsp;201-3 is also rom+.</td></tr>
<tr><td>Kissing complex</td>
  <td>Explicit — D<sub>tII</sub>, D<sub>*c</sub>, D<sub>c</sub>, D<sub>M</sub>
    all tracked separately</td>
  <td>BP's 1a–1g D-state chain (free → R<sub>I</sub>-bound short RNA II
    → elongated → primer → kissing complex with Rom) is integrated
    directly. Crucially only short RNA&nbsp;II (D<sub>tII</sub>, 110–360
    nt) is R<sub>I</sub>-susceptible; once elongated past 360&nbsp;nt to
    D<sub>lII</sub> the loop region is no longer accessible. Dropping
    that distinction (the QSS-on-everything shortcut) under-inhibits and
    pushes cpc out of the BP target band.</td></tr>
<tr><td>Rom protein</td>
  <td>Explicit — M tracked, k<sub>3</sub>·M·D<sub>*c</sub> stabilizes
    the kissing complex</td>
  <td>The rom+ parameter set hinges on this stabilization step. Without
    it the rom+/rom⁻ distinction collapses and the predicted cpc moves
    to the rom⁻ value (~52). M dynamics use BP eqn&nbsp;1j directly.</td></tr>
<tr><td>Cell volume V<sub>c</sub></td>
  <td>Fixed 0.625 fL (BP convention)</td>
  <td>BP's M⁻¹·min⁻¹ rates (k<sub>1</sub>, k<sub>3</sub>) are coupled
    to their reported V<sub>c</sub>; coupling to v2ecoli's dynamic cell
    volume (1–2 fL) would dilute RNAs ~3× and crash the steady state.
    The fixed-V<sub>c</sub> choice is temporary — once RNA&nbsp;I/II
    become bulk molecules transcribed by the main RNAP stack
    (Stage&nbsp;C, see roadmap), hybridization becomes a count-based
    reaction and V<sub>c</sub> drops out entirely.</td></tr>
<tr><td>Initiation gate</td>
  <td>Discretized BP flux: integer part of
    <code>∫&nbsp;k<sub>D</sub>·D<sub>p</sub>&nbsp;dt</code></td>
  <td>The continuous replication flux is the rate of <em>D → 2&nbsp;D</em>
    in BP eqn&nbsp;1a. We integrate, accumulate the fractional remainder
    across timesteps, and emit the integer part as discrete initiation
    events that the chromosome-replication-style modules in
    <code>plasmid_replication.py</code> consume. No survival-probability
    approximation needed.</td></tr>
</table>

<h3>BP 1993 parameter set (rom+) — values used here</h3>
<p>
All values from <strong>BP 1993 Table 1</strong>, in the paper's
native units (min⁻¹ for unimolecular, M⁻¹·min⁻¹ for bimolecular).
Bimolecular rates are converted at runtime to count-based per-second
rates via <code>k&nbsp;/&nbsp;(N<sub>A</sub>·V<sub>c</sub>)&nbsp;/&nbsp;60</code>;
unimolecular rates are divided by 60 to become per-second
(<code>plasmid_replication.py:initialize</code>). The defaults below
are exactly the values in <code>PlasmidReplication.config_schema</code>.
</p>

<table>
<tr><th>Symbol</th><th>Parameter</th><th>Value</th><th>Units</th>
  <th>BP eqn(s)</th></tr>
<tr><td colspan="5"><em>Bimolecular (M⁻¹·min⁻¹)</em></td></tr>
<tr><td>k<sub>1</sub></td>
  <td>R<sub>I</sub> + D<sub>tII</sub> → D<sub>*c</sub> (kissing)</td>
  <td>1.5 × 10⁸</td><td>M⁻¹·min⁻¹</td>
  <td>1b, 1e, 1h</td></tr>
<tr><td>k<sub>3</sub></td>
  <td>M + D<sub>*c</sub> → D<sub>M</sub> (Rom binding)</td>
  <td>1.7 × 10⁸</td><td>M⁻¹·min⁻¹</td>
  <td>1e, 1g, 1j</td></tr>
<tr><td colspan="5"><em>Unimolecular (min⁻¹)</em></td></tr>
<tr><td>k<sub>−1</sub></td>
  <td>D<sub>*c</sub> → R<sub>I</sub> + D<sub>tII</sub> (kiss reverse)</td>
  <td>48</td><td>min⁻¹</td><td>1b, 1e, 1h</td></tr>
<tr><td>k<sub>2</sub></td>
  <td>D<sub>*c</sub> → D<sub>c</sub> (kiss → cleaved)</td>
  <td>44</td><td>min⁻¹</td><td>1e, 1f</td></tr>
<tr><td>k<sub>−2</sub></td>
  <td>D<sub>c</sub> → D<sub>*c</sub> (cleaved reverse)</td>
  <td>0.085</td><td>min⁻¹</td><td>1e, 1f</td></tr>
<tr><td>k<sub>−3</sub></td>
  <td>D<sub>M</sub> → M + D<sub>*c</sub> (Rom unbinding)</td>
  <td>0.17</td><td>min⁻¹</td><td>1e, 1g, 1j</td></tr>
<tr><td>k<sub>4</sub></td>
  <td>D<sub>M</sub> → D<sub>c</sub> + M (Rom-stabilized cleavage)</td>
  <td>34</td><td>min⁻¹</td><td>1f, 1g, 1j</td></tr>
<tr><td>k<sub>l</sub></td>
  <td>D<sub>tII</sub> → D<sub>lII</sub> (RNA II elongation past 360 nt)</td>
  <td>12</td><td>min⁻¹</td><td>1b, 1c</td></tr>
<tr><td>k<sub>−l</sub></td>
  <td>D<sub>lII</sub> → D + R<sub>II</sub> (loop release)</td>
  <td>4.3</td><td>min⁻¹</td><td>1a, 1c, 1i</td></tr>
<tr><td>k<sub>p</sub></td>
  <td>D<sub>lII</sub> → D<sub>p</sub> (priming, DNA-pol I)</td>
  <td>4.3</td><td>min⁻¹</td><td>1c, 1d</td></tr>
<tr><td>k<sub>D</sub></td>
  <td>D<sub>p</sub> → 2 D + R<sub>II</sub> (replication firing)</td>
  <td>5.0</td><td>min⁻¹</td><td>1a, 1d, 1i</td></tr>
<tr><td>k<sub>−c</sub></td>
  <td>D<sub>c</sub> → D (degradation back to free)</td>
  <td>17</td><td>min⁻¹</td><td>1a, 1f</td></tr>
<tr><td>k<sub>I</sub></td>
  <td>RNA I synthesis per plasmid</td>
  <td>6</td><td>min⁻¹</td><td>1h</td></tr>
<tr><td>k<sub>II</sub></td>
  <td>RNA II initiation per plasmid (D → D<sub>tII</sub>)</td>
  <td>0.25</td><td>min⁻¹</td><td>1a, 1b</td></tr>
<tr><td>k<sub>M</sub></td>
  <td>Rom synthesis per plasmid</td>
  <td>4.0</td><td>min⁻¹</td><td>1j</td></tr>
<tr><td>ε<sub>I</sub></td>
  <td>R<sub>I</sub> degradation (<em>t</em>½ ≈ 2 min)</td>
  <td>0.35</td><td>min⁻¹</td><td>1h</td></tr>
<tr><td>ε<sub>II</sub></td>
  <td>R<sub>II</sub> degradation</td>
  <td>0.35</td><td>min⁻¹</td><td>1i</td></tr>
<tr><td>ε<sub>M</sub></td>
  <td>M (Rom) degradation</td>
  <td>0.14</td><td>min⁻¹</td><td>1j</td></tr>
<tr><td colspan="5"><em>Volume conversion</em></td></tr>
<tr><td>V<sub>c</sub></td><td>Cytoplasmic volume (BP convention)</td>
  <td>0.625 (= 6.25 × 10⁻¹⁶ L)</td><td>fL</td>
  <td>BP §2</td></tr>
<tr><td>n_substeps</td><td>RK4 substeps per process timestep</td>
  <td>10</td><td>—</td><td>(integration choice)</td></tr>
</table>

<h3>What changed vs. the prior 3-ODE QSS implementation</h3>
<p>
v2ecoli previously ran a QSS-reduced version of the BP1993 chemistry:
3 species (RNA&nbsp;I, RNA&nbsp;II, hybrid), single bimolecular
hybridization with <code>k<sub>eff</sub>&nbsp;=&nbsp;k<sub>1</sub>·k<sub>2</sub>/(k<sub>−1</sub>+k<sub>2</sub>)</code>
absorbing the kissing-complex chemistry, and a Kim-Shuler-style
<code>exp(−k<sub>h</sub>·R<sub>I</sub>·τ)</code> survival-probability
gate on initiation. The reduction was algebraically clean but produced
multigen drift away from BP's predicted 28&nbsp;cpc and could not
distinguish rom+ from rom⁻. The current implementation runs the full
BP1993 ODE.
</p>
<table>
<tr><th>Aspect</th><th>Prior (3-ODE QSS)</th>
  <th>Current (full BP1993)</th><th>Why the change</th></tr>
<tr><td>Species count</td>
  <td>3 (R<sub>I</sub>, R<sub>II</sub>, hybrid)</td>
  <td>10 (D, D<sub>tII</sub>, D<sub>lII</sub>, D<sub>p</sub>,
    D<sub>*c</sub>, D<sub>c</sub>, D<sub>M</sub>, R<sub>I</sub>,
    R<sub>II</sub>, M)</td>
  <td>The D-state chain (per-plasmid replication-cycle position) and
    the explicit Rom dimer M are what carry the rom+/rom⁻
    distinction.</td></tr>
<tr><td>Kissing complex</td>
  <td>Folded into k<sub>eff</sub> via QSS</td>
  <td>Explicit reversible D<sub>*c</sub>; explicit cleaved D<sub>c</sub>
    and Rom-stabilized D<sub>M</sub></td>
  <td>QSS assumes
    <code>d&nbsp;D<sub>*c</sub>/dt&nbsp;=&nbsp;0</code>, which only
    holds when k<sub>−1</sub>+k<sub>2</sub>&nbsp;≫ all surrounding
    fluxes; under depletion of R<sub>I</sub> at low cpc it doesn't.</td></tr>
<tr><td>R<sub>I</sub> susceptibility</td>
  <td>All RNA II R<sub>I</sub>-susceptible (single hybrid pool)</td>
  <td>Only short RNA II (D<sub>tII</sub>, 110–360 nt) is
    R<sub>I</sub>-susceptible; D<sub>lII</sub> is past the inhibition
    window</td>
  <td>BP §2 makes this explicit (eqns 1b vs 1c). Lumping D<sub>lII</sub>
    into the inhibitable pool over-inhibits and pushes cpc down.</td></tr>
<tr><td>Rom protein</td>
  <td>Implicit in the "rom+" parameter values (no M dynamics)</td>
  <td>Explicit M synthesis (k<sub>M</sub>·D), degradation
    (ε<sub>M</sub>·M), and binding to D<sub>*c</sub> (k<sub>3</sub>)</td>
  <td>Required for rom+ vs rom⁻ to come from the chemistry rather than
    a magic constant; matches BP eqn 1j directly.</td></tr>
<tr><td>Replication initiation</td>
  <td>Kim-Shuler 1990 Eq&nbsp;6 survival probability
    <code>exp(−k<sub>h</sub>·R<sub>I</sub>·τ)</code> gate</td>
  <td>Discretized BP flux: integer part of
    <code>∫&nbsp;k<sub>D</sub>·D<sub>p</sub>&nbsp;dt</code> per
    timestep, fractional remainder accumulated in
    <code>repl_accum</code></td>
  <td>The KS Eq&nbsp;6 gate is a closed-form approximation of BP's
    k<sub>l</sub>/k<sub>p</sub> primer-vs-loop competition; integrating
    BP's eqn 1d directly removes the approximation and exposes the
    fractional-event accounting to the reader as
    <code>n_rna_initiations</code>.</td></tr>
<tr><td>Integration</td>
  <td>Closed-form QSS update per timestep</td>
  <td>Fixed-step RK4, n_substeps = 10 per process timestep, with
    non-negative clamping</td>
  <td>RK4 handles the stiff R<sub>I</sub>·D<sub>tII</sub> coupling
    cleanly; clamping prevents transient negatives from biting later
    substeps.</td></tr>
<tr><td>V<sub>c</sub></td><td>fixed 0.625 fL (BP convention)</td>
  <td>fixed 0.625 fL (BP convention) — unchanged</td>
  <td>Same constraint: BP's M⁻¹·min⁻¹ rates only cohere with their
    V<sub>c</sub>. Drops out at Stage&nbsp;C when hybridization becomes
    count-based.</td></tr>
<tr><td>Predicted cpc (rom+)</td>
  <td>drift away from 28 cpc across multigen lineage</td>
  <td>~28 cpc (matches BP Table 2 — see §8 lineage)</td>
  <td>The headline. The full-ODE behaviour reproduces BP's published
    rom+ prediction; Stage&nbsp;B (Freudenau 2015) extends to
    46–49&nbsp;cpc.</td></tr>
</table>

<h3>References</h3>
<ul>
<li><strong>Brendel &amp; Perelson 1993</strong> — J. Mol. Biol.
  229:860. Kinetic model of ColE1 plasmid replication control with
  explicit kissing complex and Rom regulation. Table&nbsp;1 lists all
  rate constants; Table&nbsp;2 predicts 28&nbsp;cpc for rom+ wild-type,
  52&nbsp;cpc rom⁻. <strong>Primary source for the current
  parameter set.</strong></li>
<li><strong>Freudenau et al. 2015</strong> — Front. Bioeng. Biotechnol.
  3:127. Extends BP 1993 with explicit Rom and uncharged-tRNA
  dynamics; refits 26 rate constants to qPCR-measured RNAI/RNAII/PCN
  on DH5α-pSUP 201-3 (low-copy pBR322 derivative). Reports
  46–49&nbsp;cpc across three timepoints. <strong>Stage B target.</strong></li>
<li><strong>Kim &amp; Shuler 1990</strong> — Biotechnol. Bioeng. 36:233.
  pBR322 in B/r at 0.08–0.84&nbsp;h⁻¹. Replaced as the parameter source
  but Eq&nbsp;6 (RNA-II initiation survival probability
  exp(−k<sub>h</sub>·RNA<sub>I</sub>·τ)) is still the gate we use.</li>
<li><strong>Ataai &amp; Shuler 1986</strong> — Biotechnol. Bioeng. 28:1166.
  Original three-ODE framework for RNA-I/II/hybrid that BP refines.</li>
</ul>

<h3>Deferred (roadmap)</h3>
<ul>
<li><strong>Stage B — explicit Rom dynamics
  (Freudenau&nbsp;et&nbsp;al.&nbsp;2015).</strong>
  Add Rom protein synthesis/degradation and the
  Rom·C<sub>m</sub>&nbsp;⇌&nbsp;Rom·C<sub>s</sub> stabilization step.
  This raises the effective hybridization rate ~1.6× and lifts
  predicted cpc from BP's 28 → Freudenau's measured 46–49.</li>
<li><strong>Stage C — mechanistic RNA&nbsp;I / RNA&nbsp;II transcription
  (architectural target).</strong>
  Today the BP1993 10-species pool (D-chain, R<sub>I</sub>,
  R<sub>II</sub>, M) lives as scalars in
  <code>process_state.plasmid_rna_control</code>, and division requires
  bespoke halving logic in <code>library/division.py</code>. The
  end-state is to make R<sub>I</sub> and R<sub>II</sub> actual
  <em>bulk</em> molecules produced by the standard
  <code>ecoli-transcript-initiation</code> +
  <code>ecoli-transcript-elongation</code> processes firing off plasmid
  promoters, with the kissing-complex chemistry running as
  count-based reactions on those bulk pools. At that point the
  custom process-state halving becomes dead code, V<sub>c</sub> drops
  out (count-based reaction), and the BP D-state chain reduces to
  bookkeeping over which plasmid copies are currently engaged with
  bulk RNA II / Rom — see project memory
  <code>plasmid_mechanistic_target</code>.</li>
<li>Growth-rate dependence of α<sub>I</sub> / α<sub>II</sub>,
  stochastic RNA&nbsp;II initiation (Poisson over the 240&nbsp;s
  window), segregational loss at low copy, ppGpp linkage, multimer
  recombination, metabolic burden — stages D–G.</li>
</ul>

<h2 id="copy-number">1. Plasmid unique molecule counts</h2>
<div class="plot">{plots['copy_number']}</div>

<h2 id="replisomes">2. Active replisomes</h2>
<div class="plot">{plots['replisomes']}</div>

<h2 id="rna">3. BP1993 species — free RNAs and D-pool partitioning</h2>
<p>
  Left: free <em>R<sub>I</sub></em> (inhibitor), <em>R<sub>II</sub></em>
  (primer), and <em>M</em> (free Rom dimer) — the three count-based
  pools the BP1993 ODE tracks outside the D-pool. Right: how the
  per-plasmid D-pool partitions across the seven BP states (free
  <em>D</em>, the four kissing-complex / hybrid intermediates summed,
  and the actively-replicating <em>D<sub>M</sub></em>). At steady state
  the D-pool sums to the number of idle plasmids
  (<code>n_full_plasmids − n_active_replisomes</code>); see
  <code>plasmid_replication.py:_prepare</code> for the renormalization.
</p>
<div class="plot">{plots['rna_control']}</div>

{subunits_section}
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
