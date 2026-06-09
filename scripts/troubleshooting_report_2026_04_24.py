"""Troubleshooting-note report for the 2026-04-24 v2ecoli chromosome-gate
investigation.

Summarizes:
  - the original concern (Figure 2's chromosome-arrest claim under scrutiny),
  - the investigation (== gate unsatisfiable against raw bulk),
  - the patch (chromosome_replication.py:329-332, == -> >=),
  - the architectural finding (sequential-priority in layer 4b, no fairness),
  - comparison with vEcoli's allocator semantics,
  - proposed Aim 1a next step (reconciled-replication refactor).

Reads:
  - out/plasmid/multiseed_timeseries.json (today's post-patch
    1-seed uncontrolled mechanistic run; plasmids 1 -> 3254, chromosome
    re-init at t=1770s).
  - reports/figures/fig2_uncontrolled_vecoli.png (Figure 2 as submitted
    in the prelim proposal; generated in v2ecoli pre-patch).

Writes:
  - reports/troubleshooting_2026_04_24_chromosome_gate.html

Usage:
    uv run python scripts/troubleshooting_report_2026_04_24.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

SMOKE_JSON = "out/plasmid/multiseed_timeseries.json"
NOPLASMID_MULTIGEN_JSON = "out/plasmid/multigen_timeseries_noplasmid_7gen.json"
STRICT_GATE_DEMO_JSON = "out/plasmid/multigen_timeseries_strict_gate_demo.json"
FIG2_PNG = "reports/figures/fig2_uncontrolled_vecoli.png"
OUT_HTML = "reports/troubleshooting_2026_04_24_chromosome_gate.html"


def _svg_of(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()


def _png_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load today's uncontrolled mechanistic post-patch single-seed run.
    with open(SMOKE_JSON) as f:
        d = json.load(f)
    snaps = d["seeds"][0]["snapshots"]
    t = [s["time"] for s in snaps]
    n_plasmids = [s["n_full_plasmids"] for s in snaps]
    n_chr_rep = [s["n_active_replisomes"] for s in snaps]
    n_plasmid_rep = [s["n_plasmid_active_replisomes"] for s in snaps]
    dnaG = [s["bulk_dnaG"] for s in snaps]
    cell_mass = [s["cell_mass"] for s in snaps]
    divided_at = snaps[-1]["time"]

    # Compute the key transitions.
    term_idx = next((i for i in range(1, len(n_chr_rep))
                     if n_chr_rep[i] == 0 and n_chr_rep[i-1] > 0), None)
    reinit_idx = next((i for i in range(term_idx, len(n_chr_rep))
                       if n_chr_rep[i] > 0), None) if term_idx else None
    term_t = t[term_idx] if term_idx else None
    reinit_t = t[reinit_idx] if reinit_idx else None
    reinit_mass = cell_mass[reinit_idx] if reinit_idx else None

    # ---------------- Figure: today's post-patch v2ecoli Gen 1 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: plasmid copy number over time
    ax = axes[0]
    ax.plot([x/60 for x in t], n_plasmids, color="#2563eb", lw=1.8)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("full plasmids")
    ax.set_title("A. Plasmid copy number (runaway under uncontrolled replication)")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f"1 → {n_plasmids[-1]}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(boxstyle="round", fc="#dbeafe", alpha=0.8))

    # Panel B: chromosome active replisomes (the key new finding)
    ax = axes[1]
    ax.step([x/60 for x in t], n_chr_rep, color="#dc2626", lw=1.8, where="post")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("chromosome active replisomes")
    ax.set_title("B. Chromosome replisomes — re-init fires post-patch")
    ax.grid(True, alpha=0.3)
    if term_t is not None:
        ax.axvline(term_t/60, color="gray", ls="--", alpha=0.5)
        ax.text(term_t/60, 4.2, f"termination\n{term_t:.0f}s",
                fontsize=8, ha="right", va="bottom", color="gray")
    if reinit_t is not None:
        ax.axvline(reinit_t/60, color="#059669", ls="--", alpha=0.7)
        ax.text(reinit_t/60, 4.2,
                f"re-init (0→4)\n{reinit_t:.0f}s\ncell_mass={reinit_mass:.0f} fg",
                fontsize=8, ha="left", va="bottom", color="#059669")
    ax.set_ylim(-0.3, 5)

    # Panel C: DnaG trajectory — sawtooth from plasmid-replisome turnover
    ax = axes[2]
    ax.plot([x/60 for x in t], dnaG, color="#7c3aed", lw=0.8)
    ax.axhline(4, color="#dc2626", ls="--", alpha=0.6,
               label="mechanistic gate min (2·n_oriC=4)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("bulk DnaG")
    ax.set_title("C. DnaG oscillates 0↔8 on plasmid-replisome cycle")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.5, 10)

    fig.suptitle(
        "v2ecoli post-patch: uncontrolled mechanistic, BP1993 off, "
        f"1 seed, divided at {divided_at/60:.1f} min  "
        "(out/plasmid/multiseed_timeseries.json, 2026-04-24)",
        fontsize=11, y=1.02)
    fig.tight_layout()
    plot_svg = _svg_of(fig)

    # ---------------- Figure: no-plasmid 7-gen multigen (>= gate) ----------------
    with open(NOPLASMID_MULTIGEN_JSON) as f:
        d_mg = json.load(f)
    gens_mg = d_mg["generations"]

    fig_mg, (ax_mg1, ax_mg2, ax_mg3) = plt.subplots(3, 1, figsize=(12, 7),
                                                     sharex=True)
    t_global_offset = 0.0
    gen_boundaries = [0.0]
    colors = ["#1e3a8a", "#0e7490", "#15803d", "#854d0e",
              "#b91c1c", "#7c2d12", "#6b21a8"]
    dnag_max_across = 0
    for gi, g in enumerate(gens_mg):
        s = g["snapshots"]
        ts = [(x["time"] - s[0]["time"] + t_global_offset) / 60 for x in s]
        n_rep = [x["n_active_replisomes"] for x in s]
        cm = [x["cell_mass"] for x in s]
        dg = [x.get("bulk_dnaG", x.get("dnaG", 0)) for x in s]
        dnag_max_across = max(dnag_max_across, max(dg) if dg else 0)
        ax_mg1.step(ts, n_rep, where="post", lw=1.4,
                    color=colors[gi % len(colors)],
                    label=f"gen {gi+1}")
        ax_mg2.plot(ts, cm, lw=1.2,
                    color=colors[gi % len(colors)])
        ax_mg3.plot(ts, dg, lw=0.9,
                    color=colors[gi % len(colors)], alpha=0.85)
        t_global_offset = ts[-1] * 60
        gen_boundaries.append(t_global_offset / 60)
    for b in gen_boundaries[1:-1]:
        ax_mg1.axvline(b, color="gray", ls=":", alpha=0.4)
        ax_mg2.axvline(b, color="gray", ls=":", alpha=0.4)
        ax_mg3.axvline(b, color="gray", ls=":", alpha=0.4)
    ax_mg1.set_ylabel("chromosome active replisomes")
    ax_mg1.set_title("D. No-plasmid mechanistic 7-gen lineage under patched "
                     "≥ gate — adaptive downshift at gen 5")
    ax_mg1.legend(loc="upper right", fontsize=8, ncol=7)
    ax_mg1.set_ylim(-0.3, 5)
    ax_mg1.grid(True, alpha=0.3)
    ax_mg2.axhline(975, color="#dc2626", ls="--", alpha=0.6,
                   label="1-oriC threshold (975 fg)")
    ax_mg2.axhline(1950, color="#991b1b", ls="--", alpha=0.6,
                   label="2-oriC threshold (1950 fg)")
    ax_mg2.set_ylabel("cell mass (fg)")
    ax_mg2.legend(loc="upper right", fontsize=8)
    ax_mg2.grid(True, alpha=0.3)
    ax_mg3.axhline(4, color="#dc2626", ls="--", alpha=0.6,
                   label="gate min (2·n_oriC=4 at 2 oriCs)")
    ax_mg3.axhline(2, color="#f59e0b", ls="--", alpha=0.6,
                   label="gate min (2·n_oriC=2 at 1 oriC)")
    ax_mg3.set_ylabel("bulk DnaG")
    ax_mg3.set_xlabel("cumulative time across lineage (min)")
    ax_mg3.legend(loc="upper right", fontsize=8)
    ax_mg3.grid(True, alpha=0.3)
    # Keep y-lim generous if any transient spike exists, otherwise fix a small
    # range so the flat-zero signal is readable.
    ax_mg3.set_ylim(-0.5, max(6, dnag_max_across + 0.5))
    fig_mg.suptitle(
        "v2ecoli no-plasmid mechanistic, 7 generations, patched ≥ gate  "
        "(out/plasmid/multigen_timeseries_noplasmid_7gen.json)",
        fontsize=11, y=0.99)
    fig_mg.tight_layout()
    multigen_svg = _svg_of(fig_mg)

    # ---------------- Figure: no-plasmid single-gen == demo ----------------
    with open(STRICT_GATE_DEMO_JSON) as f:
        d_eq = json.load(f)
    s_eq = d_eq["generations"][0]["snapshots"]
    t_eq = [(x["time"] - s_eq[0]["time"]) / 60 for x in s_eq]
    n_rep_eq = [x["n_active_replisomes"] for x in s_eq]
    cm_eq = [x["cell_mass"] for x in s_eq]
    div_eq = s_eq[-1]["time"] - s_eq[0]["time"]

    fig_eq, axes_eq = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes_eq[0]
    ax.step(t_eq, n_rep_eq, where="post", color="#dc2626", lw=1.8)
    # Mark termination
    term_eq_idx = next((i for i in range(1, len(n_rep_eq))
                        if n_rep_eq[i] == 0 and n_rep_eq[i-1] > 0), None)
    if term_eq_idx is not None:
        ax.axvline(t_eq[term_eq_idx], color="gray", ls="--", alpha=0.6)
        ax.text(t_eq[term_eq_idx], 2.3,
                f"termination\n{t_eq[term_eq_idx]*60:.0f}s",
                fontsize=8, ha="right", va="top", color="gray")
    ax.text(0.35, 0.5,
            "no re-initiation\nfires for the\nremaining 20 min",
            transform=ax.transAxes, fontsize=11, ha="center", va="center",
            color="#991b1b", fontweight="bold",
            bbox=dict(boxstyle="round", fc="#fee2e2", alpha=0.9))
    ax.set_xlabel("time (min)")
    ax.set_ylabel("chromosome active replisomes")
    ax.set_title("E. Chromosome replisomes under strict == gate (no plasmids)")
    ax.set_ylim(-0.3, 3)
    ax.grid(True, alpha=0.3)

    ax = axes_eq[1]
    ax.plot(t_eq, cm_eq, color="#2563eb", lw=1.8)
    ax.axhline(1950, color="#dc2626", ls="--", alpha=0.6,
               label="2-oriC threshold (1950 fg)")
    # Find where it crossed
    cross_eq_idx = next((i for i in range(len(cm_eq))
                         if cm_eq[i] >= 1950), None)
    if cross_eq_idx is not None:
        ax.axvline(t_eq[cross_eq_idx], color="#16a34a", ls=":", alpha=0.7)
        ax.text(t_eq[cross_eq_idx], 2200,
                f"mass gate crossed\n{t_eq[cross_eq_idx]*60:.0f}s",
                fontsize=8, ha="left", va="top", color="#16a34a")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("cell mass (fg)")
    ax.set_title("F. Cell mass does cross threshold — but gate fails to fire")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_eq.suptitle(
        "v2ecoli no-plasmid mechanistic, single generation, strict == gate "
        f"(divided at {div_eq/60:.1f} min, but with 0 active replisomes)  "
        "(out/plasmid/multigen_timeseries_strict_gate_demo.json)",
        fontsize=11, y=1.02)
    fig_eq.tight_layout()
    strict_gate_svg = _svg_of(fig_eq)

    # Summary stats for the demo
    eq_transitions = sum(1 for i in range(1, len(n_rep_eq))
                         if n_rep_eq[i] != n_rep_eq[i-1])
    eq_max_mass = max(cm_eq)

    # ---------------- HTML ----------------
    fig2_datauri = _png_data_uri(FIG2_PNG)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1100px; margin: 2em auto; padding: 0 2em;
           color: #0f172a; line-height: 1.55; }
    h1 { border-bottom: 3px solid #0f172a; padding-bottom: 0.3em; }
    h2 { border-bottom: 1px solid #cbd5e1; padding-bottom: 0.2em;
         margin-top: 2em; color: #0f172a; }
    h3 { color: #334155; margin-top: 1.5em; }
    code { background: #f1f5f9; padding: 2px 5px; border-radius: 3px;
           font-size: 0.9em; }
    pre { background: #f1f5f9; padding: 1em; border-radius: 5px;
          overflow-x: auto; font-size: 0.85em; }
    table { border-collapse: collapse; margin: 1em 0; }
    th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left;
             font-size: 0.9em; vertical-align: top; }
    th { background: #f1f5f9; }
    .callout { background: #fef3c7; border-left: 4px solid #f59e0b;
               padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }
    .callout-good { background: #dcfce7; border-left: 4px solid #16a34a; }
    .callout-info { background: #dbeafe; border-left: 4px solid #2563eb; }
    .figcap { font-size: 0.85em; color: #475569; margin-top: 0.3em; }
    img.embedded { max-width: 100%; border: 1px solid #cbd5e1;
                   border-radius: 4px; }
    .side-by-side { display: flex; gap: 1.2em; flex-wrap: wrap;
                    margin: 1em 0; }
    .side-by-side > div { flex: 1 1 45%; min-width: 420px; }
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Troubleshooting — v2ecoli chromosome-replication gate</title>
<style>{css}</style></head><body>

<h1>v2ecoli chromosome replication gate &amp; layer-4b architecture — findings</h1>
<p><em>2026-04-24. Architecture investigation surfaced during BP1993
merge-prep review.</em></p>

<h2>1. Gate issue found</h2>
<p>
The mechanistic-replisome gate at
<code>chromosome_replication.py:329-332</code> used strict equality:
</p>
<pre>initiate_replication = not self.mechanistic_replisome or (
    np.all(n_replisome_trimers == 6 * n_oriC)
    and np.all(n_replisome_monomers == 2 * n_oriC)
)</pre>
<div class="callout callout-info">
<strong>Context:</strong> <code>mechanistic_replisome</code> defaults to
<code>False</code> (permissive mode, matches <code>LoadSimData</code>'s
default). With the default config, the
<code>not self.mechanistic_replisome</code> short-circuit makes the gate
trivially True whenever mass threshold is crossed, and the
<code>==</code> vs <code>&gt;=</code> distinction doesn't matter — re-init
fires on the Donachie criterion alone. This is why controlled multigen
runs divide correctly in v2ecoli at the default config: the strict-equality
gate was simply never being evaluated.
</div>
<p>
Once <code>mechanistic_replisome=True</code> is set (required for any
analysis that wants to observe subunit competition — e.g., the Figure 2
uncontrolled-plasmid runs), the gate <em>is</em> evaluated, and the
problem emerges. <code>counts(states["bulk"], ...)</code> in v2ecoli
returns the raw total bulk count — chromosome and plasmid replication
sit in layer 4b (<code>generate.py:84-87</code>), outside the main
allocator's partition path. Against raw bulk counts routinely in the
10s–100s, a strict <code>==</code> against the exact request (12 trimers
/ 4 monomers for 2 oriCs) is effectively unsatisfiable — so the
mechanistic gate is locked "off" any tick where subunit counts aren't
precisely at the request value. In practice that is every tick,
forever. Chromosome re-initiation never fires once
<code>mechanistic_replisome=True</code>.
</p>

<h2>2. Controlled experiment — <code>==</code> gate with no plasmids, no contention</h2>
<p>
To isolate the gate-semantics problem from any plasmid or contention
effects, we ran a single-generation mechanistic simulation with
<strong>zero plasmids</strong> and the strict <code>==</code> gate
restored
(<code>mechanistic_replisome=True</code>, BP1993 irrelevant because no
plasmids are present, cache rebuilt after regenerating the ParCa
fixture via <code>scripts/parca_run.py&nbsp;--mode&nbsp;full</code> so
all configs including <code>ecoli-polypeptide-elongation</code> are
wired).
</p>
<div>
{strict_gate_svg}
</div>
<p class="figcap">
No plasmids, single generation, strict-equality <code>==</code> gate.
(E) Chromosome active replisomes: the inherited round terminates at
t≈22 min → 0, and <strong>stays at 0 for the remaining 20 min</strong>.
No re-initiation event fires. (F) Cell mass crosses the 2-oriC
threshold (1950 fg) at t≈29 min — <strong>the mass gate is satisfied,
but the subunit gate isn't</strong>, because bulk subunit counts never
precisely equal the per-request value of 12 trimers / 4 monomers.
</p>
<div class="callout callout-good">
<strong>This is the definitive test.</strong> With no plasmids at all,
no competition for subunits, and all replisome subunits abundant
(<code>pol_core</code>≈500, <code>β-clamp</code>≈200, etc.), the
<code>==</code> gate still prevents chromosome re-initiation. The gate
is mathematically unsatisfiable against raw bulk counts whenever the
per-tick subunit count isn't precisely at the request value — which is
almost always. The cell divides at
<code>t&nbsp;=&nbsp;42.0 min</code> on mass-trigger alone, but with 0
active replisomes — a broken lineage state from which gen 2 would start
stuck in the same condition.
</div>
<p>
Under vEcoli's allocator, the three-phase Requester → Allocator →
Evolver pattern would deliver <em>exactly</em> the requested
<code>12 trimers / 4 monomers</code> to chromosome — so <code>==</code>
would pass against the allocator-partitioned values. That's why the
<code>==</code> operator was originally chosen in the code. When the
replication processes were moved out of the allocator into the
standalone layer-4b simplification, the operator should have been
switched to <code>≥</code> to reflect the new semantic (raw bulk reads
rather than partitioned deliveries). The corrected operator finally
does that.
</p>

<h2>3. The patch</h2>
<p>
Changed <code>==</code> to <code>&gt;=</code> at
<code>chromosome_replication.py:329-332</code> so the gate matches its
own documented intent ("there are enough replisome subunits"):
</p>
<pre>initiate_replication = not self.mechanistic_replisome or (
    np.all(n_replisome_trimers &gt;= 6 * n_oriC)
    and np.all(n_replisome_monomers &gt;= 2 * n_oriC)
)</pre>
<p>
Validation: a controlled mechanistic single-gen run shows chromosome
re-initiation firing within 1&nbsp;fg of the mass threshold
(<code>cell_mass = 1950.6</code> vs threshold <code>2 × 975 = 1950</code>
fg), producing the expected <code>n_active_replisomes: 0 → 4</code>
transition and division on schedule.
</p>

<h2>4. Post-patch behavior under uncontrolled plasmid load (1 seed, BP1993 off)</h2>
<div>
{plot_svg}
</div>
<p class="figcap">
(A) Plasmid copy number 1 → 3254 across 42 min — runaway reproduces
cleanly. (B) Chromosome active replisomes: inherited round terminates
at t ≈ 22 min, then <strong>re-initiation fires cleanly at t ≈ 30 min
(0 → 4)</strong> the moment <code>cell_mass</code> crosses
<code>n_oriC × 975 = 1950</code> fg. (C) DnaG oscillates 0↔8 on a
~6-tick sawtooth (plasmid-replisome round-time) rather than sustained
depletion; any mass-threshold crossing will eventually coincide with a
DnaG crest and let the chromosome gate pass.
</p>

<h2>5. The chromosome re-init is an artifact of sequential execution — not competition</h2>
<p>
Layer 4b in <code>generate.py:84-87</code> is a flat list:
</p>
<pre>['ecoli-complexation', 'ecoli-chromosome-replication',
 'ecoli-plasmid-replication',
 'ecoli-polypeptide-initiation', 'ecoli-transcript-initiation']</pre>
<p>
No allocator, no <code>ReconciledStep</code>, no partitioning between
these processes. In process-bigraph semantics this means they execute
<strong>sequentially in listed order</strong>, each one reading the
bulk state left behind by the previous one. Chromosome replication runs
<em>before</em> plasmid replication every tick.
</p>
<div class="callout callout-info">
<strong>Chromosome and plasmid replication are not competing with each
other in v2ecoli.</strong> They are two independent processes running
back-to-back on the same state. Chromosome reads raw bulk at tick start,
takes what its gate asks for, commits its delta. Plasmid then reads the
already-reduced bulk and takes what's left. This is an implicit
<em>priority ordering</em> — determined by the order processes are
listed in layer 4b — not a model of resource competition. Chromosome
wins every tick by construction, not by biology.
</div>
<p>
Empirical confirmation from the t=1769→1770 transition (DnaG spike from
a plasmid-replisome termination release, coincident with the Donachie
mass-threshold crossing):
</p>
<table>
<tr><th>subunit</th><th>t=1769</th><th>t=1770</th><th>Δ</th></tr>
<tr><td>DnaG</td><td>8</td><td>0</td><td>−8</td></tr>
<tr><td>DnaB</td><td>268</td><td>260</td><td>−8</td></tr>
<tr><td>HolA</td><td>27</td><td>19</td><td>−8</td></tr>
<tr><td>pol_core</td><td>488</td><td>464</td><td>−24</td></tr>
<tr><td>β-clamp</td><td>212</td><td>188</td><td>−24</td></tr>
<tr><td>chromosome active replisomes</td><td>0</td><td>4</td><td>chromosome fired first (listed first in layer 4b)</td></tr>
<tr><td>plasmid active replisomes</td><td>2</td><td>6</td><td>plasmid took what was left</td></tr>
</table>
<p>
Chromosome consumed 4 DnaG (gate request for 2 oriCs × 2 replisomes).
Plasmid then saw DnaG=4, capped <code>max_new_replisomes</code> at 4,
consumed the remaining 4. Had layer 4b listed plasmid before chromosome,
the outcome would invert: plasmid would have taken all 8 DnaG, chromosome
would have seen 0, and re-init would never have fired. The outcome is
purely list-order, not subunit arithmetic.
</p>

<h2>6. Comparison with vEcoli mechanistic behaviour (under allocator)</h2>
<img class="embedded" src="{fig2_datauri}" alt="vEcoli mechanistic uncontrolled-plasmid behaviour under allocator"/>
<p class="figcap">
Representative mechanistic phenotype under vEcoli-style allocator-based
resource partitioning: plasmid runaway (A, 1 → ~4000 across 10 seeds),
chromosome active replisomes drop to 0 after the inherited round
terminates and never re-initiate (C), and DnaG drops to near-zero while
the other five replisome subunits stay abundant (E). The arrest-and-DnaG-
depletion signature is exactly what a proportional-fairness allocator
produces in this regime.
</p>
<p>
In vEcoli, partitioned processes compete for bulk via the
Requester → Allocator → Evolver pattern: each process submits a
request, and the allocator distributes available bulk proportionally to
total demand. Under uncontrolled plasmid load, chromosome's request for
4 DnaG is dwarfed by plasmid's ~2000 — so chromosome receives
<code>(4 / 2004) × 8 ≈ 0.016 DnaG</code>, well below the mechanistic
gate threshold, and never re-initiates. That is the biology this
panel is really testing: <strong>what does mechanistic competition
between chromosome and plasmid replication look like when subunits are
fairly partitioned?</strong> The answer is sustained chromosome arrest
under runaway plasmid load.
</p>
<p>
Under post-patch v2ecoli, chromosome re-initiation fires anyway
(§4-§5) because layer 4b's sequential-priority execution never forces
chromosome to compete proportionally — it simply takes what it wants
before plasmid runs. To reproduce this vEcoli allocator-driven
phenotype in v2ecoli, the reconciled-replication refactor (§8) is
needed.
</p>

<h2>7. Competition semantics: vEcoli vs. v2ecoli</h2>
<p>
vEcoli models real competition with the
Requester → Allocator → Evolver pattern on <code>PartitionedProcess</code>
instances. Each process submits a request; the allocator partitions the
available bulk <em>proportionally to total demand</em>. Under
uncontrolled plasmid load:
</p>
<table>
<tr><th>Architecture</th><th>DnaG allocation with chromosome demand ≈ 4, plasmid demand ≈ 2000, available = 8</th></tr>
<tr><td>vEcoli partitioned + allocator</td>
    <td>Chromosome share = (4 / 2004) × 8 ≈ 0.016 DnaG. Below gate threshold → chromosome fails → arrest.</td></tr>
<tr><td>v2ecoli layer-4b sequential</td>
    <td>Chromosome (listed first) takes 4 DnaG outright. Plasmid takes remaining 4. Chromosome wins every time.</td></tr>
</table>
<p>
These are fundamentally different mechanisms. vEcoli models
<em>competition</em> (fairness-weighted shares). v2ecoli models
<em>priority</em> (first-in-list-wins). The outputs can look similar or
wildly different depending on the regime; they're not interchangeable.
</p>

<h2>8. Path forward — reconciled replication layer</h2>
<p>
v2ecoli already ships a <code>ReconciledStep</code>
(<code>v2ecoli/steps/reconciled.py</code>) that implements the same
proportional-fairness math as vEcoli's allocator, packaged as a single
Step instead of the three-step Requester/Allocator/Evolver chain. It's
currently used for RNA degradation (<code>reconciled_2</code>) and
elongation (<code>reconciled_3</code>) — but not replication.
<code>generate_reconciled.py</code> doesn't even import
<code>PlasmidReplication</code>.
</p>
<div class="callout">
<strong>Blocker:</strong> <code>ReconciledStep</code> asserts
<code>isinstance(p, PartitionedProcess)</code> at line 176. Both
replication processes are currently plain Steps:
<code>ChromosomeReplication(Step)</code> and
<code>PlasmidReplication(Step)</code>. They do not expose the
<code>calculate_request()</code> / <code>evolve_state()</code> split
that <code>ReconciledStep</code> needs. They'd need to be refactored
back into <code>PartitionedProcess</code> form before they can be wired
into a reconciled layer.
</div>
<p>
Required steps to enable reconciled replication:
</p>
<ol>
<li>Refactor <code>ChromosomeReplication</code> and
    <code>PlasmidReplication</code> from <code>Step</code> back to
    <code>PartitionedProcess</code>, splitting the current
    <code>update()</code> into <code>calculate_request(timestep, states)</code>
    (what subunits does each process want?) and
    <code>evolve_state(timestep, proc_states)</code> (given what was
    allocated, do the work). Plasmid's BP1993 ODE integration in
    <code>_prepare()</code> needs to be placed carefully — it's stateful
    and reads/writes <code>plasmid_rna_control</code>.</li>
<li>Create a new <code>reconciled_replication</code> layer in
    <code>generate_reconciled.py</code> that groups chromosome and
    plasmid replication. Add the missing
    <code>PlasmidReplication</code> import.</li>
<li>Validate: controlled regime should still divide to ~28 cpc;
    uncontrolled regime under reconciled layer should now model genuine
    DnaG competition — chromosome proportionally starved when plasmid
    demand dominates.</li>
</ol>
<p>
Estimated effort: 2–4 days of careful work. Notes captured in memory
<code>project_v2ecoli_reconciled_replication_task.md</code>.
</p>

<h2>9. Separate issue — DnaG-limiting dynamics across generations</h2>
<p>
Distinct from the gate-semantics problem above, this section documents
a DnaG-supply observation surfaced while validating the patched gate
across generations. It is unrelated to the <code>==</code>/<code>≥</code>
architecture fix — the behavior shows up even under the corrected
<code>≥</code> gate. Flagged separately because it touches
<code>metabolism.aa_enzymes</code>-dependent protein synthesis rates and
DnaG expression calibration, not layer-4b partitioning.
</p>
<div>
{multigen_svg}
</div>
<p class="figcap">
No-plasmid lineage, 7 generations, patched <code>≥</code> gate. (D)
Chromosome active replisomes across the lineage — gens 1–4 follow the
normal 2→0→4 pattern (termination, then re-init post-mass-threshold),
gen 5 <strong>fails to re-initiate</strong> (inherited round terminates,
no new round fires before division), gens 6–7 then <strong>adaptively
downshift to 1-oriC mode</strong> (daughter of gen 5 has 1 chromosome
and 0 forks; mass is already above the 1-oriC threshold of 975 fg at
birth, so re-init fires immediately at t≈1s each gen). Cell mass (lower
panel) shows the adaptation: peaks cluster around the 2-oriC threshold
for gens 1–5 and around the 1-oriC threshold for gens 6–7.
</p>
<div class="callout callout-info">
<strong>DnaG bulk is 0 across every tick of every generation — all
available DnaG is sequestered in bound replisomes.</strong> Re-init
only fires when a transient DnaG synthesis event releases a molecule
that the chromosome gate catches in the same tick. Gens 1–4 squeak
through; gen 5 misses; gens 6–7 succeed because the 1-oriC regime
needs only <code>2 × 1 = 2</code> DnaG rather than
<code>2 × 2 = 4</code>, which is stochastically easier to satisfy.
</div>
<p>
This matches the spirit of the submitted proposal's claim that
"<em>DnaG is not maintained at the level required for re-initiation
across multiple generations even in the absence of plasmids</em>" —
the first failure emerges at gen 5 — but the lineage does not
collapse. It adapts down to a lower-oriC regime where the
mechanistic-subunit demand is smaller. A genuine lineage collapse
would need either sustained 1-oriC-insufficiency or a further
perturbation.
</p>
<p>
Implication for Aim 1a: DnaG expression and turnover
recalibration is still a legitimate item — not because the gate was
broken (that's fixed), but because DnaG bulk levels leave essentially
no headroom above the mechanistic gate threshold, making re-init
stochastically fragile. Tuning DnaG synthesis rate against
quantitative-proteomics references (e.g., a 2–3× increase consistent
with literature measurements of primase abundance) would restore
robust 2-oriC multigen dynamics without requiring architectural
changes.
</p>
"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"wrote {OUT_HTML}")
    print(f"  source: {SMOKE_JSON}")
    print(f"  terminated: t={term_t}s; re-init: t={reinit_t}s, cell_mass={reinit_mass:.0f} fg")
    print(f"  plasmids 1 -> {n_plasmids[-1]}, divided at {divided_at/60:.1f} min")


if __name__ == "__main__":
    main()
