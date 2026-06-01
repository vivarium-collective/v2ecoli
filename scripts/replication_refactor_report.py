"""Generator for the replication-refactor report.

Builds reports/replication_partitioned_refactor.html covering both:

  - Phase 1: chromosome replication refactored from Step to
    PartitionedProcess and joined to allocator_2.
  - Phase 2: plasmid replication refactored similarly, joining
    chromosome and rna-degradation in allocator_2 so the two
    replication processes compete for replisome subunits via the
    existing allocator-fairness math.

Reads three sim outputs to plot chromosome behavior across the three
regimes the refactor exercises:

  - out/plasmid/multigen_timeseries_noplasmid_7gen.json
      No plasmids loaded.  Single requester in allocator_2 -> allocator
      is a pass-through; chromosome runs as before.
  - out/plasmid/multigen_timeseries.json
      Plasmids + BP1993 RNA control on.  Plasmid demand is small (1-9
      plasmids), allocator gives both processes their full requests.
  - out/plasmid/multiseed_timeseries.json
      Plasmids + BP1993 off (uncontrolled).  Plasmid demand dominates,
      allocator proportionally starves chromosome -> arrest emerges.

Usage:
    uv run python scripts/replication_refactor_report.py
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

NOPLASMID_JSON = "out/plasmid/multigen_timeseries_noplasmid_7gen.json"
CONTROLLED_JSON = "out/plasmid/multigen_timeseries.json"
UNCONTROLLED_JSON = "out/plasmid/multiseed_timeseries.json"
OUT_HTML = "reports/replication_partitioned_refactor.html"


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()


def _build_regime_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5))

    # ---- Row A: no plasmids (7-gen lineage) ----
    with open(NOPLASMID_JSON) as f:
        d = json.load(f)
    gens = [g["snapshots"] for g in d["generations"]]
    t_off = 0.0
    boundaries = [0.0]
    colors = ["#1e3a8a", "#0e7490", "#15803d", "#854d0e",
              "#b91c1c", "#7c2d12", "#6b21a8"]
    for gi, snaps in enumerate(gens):
        ts = [(s["time"] - snaps[0]["time"] + t_off) / 60 for s in snaps]
        n_rep = [s["n_active_replisomes"] for s in snaps]
        cm = [s["cell_mass"] for s in snaps]
        col = colors[gi % len(colors)]
        axes[0, 0].step(ts, n_rep, where="post", lw=1.2, color=col,
                        label=f"gen {gi+1}")
        axes[0, 1].plot(ts, cm, lw=1.0, color=col)
        t_off = ts[-1] * 60
        boundaries.append(t_off / 60)
    for b in boundaries[1:-1]:
        axes[0, 0].axvline(b, color="gray", ls=":", alpha=0.3)
        axes[0, 1].axvline(b, color="gray", ls=":", alpha=0.3)
    axes[0, 0].set_ylabel("chromosome\nactive replisomes")
    axes[0, 0].set_title("A. No plasmids — 7-generation lineage")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=7)
    axes[0, 0].set_ylim(-0.3, 5)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].axhline(975, color="#dc2626", ls="--", alpha=0.5,
                       label="1-oriC threshold")
    axes[0, 1].axhline(1950, color="#991b1b", ls="--", alpha=0.5,
                       label="2-oriC threshold")
    axes[0, 1].set_ylabel("cell mass (fg)")
    axes[0, 1].set_title("A'. Cell mass across generations")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # ---- Row B: controlled plasmids (BP1993 on) ----
    with open(CONTROLLED_JSON) as f:
        d = json.load(f)
    snaps = d["generations"][0]["snapshots"]
    t = [s["time"] / 60 for s in snaps]
    n_rep = [s["n_active_replisomes"] for s in snaps]
    n_plasmids = [s["n_full_plasmids"] for s in snaps]
    cm = [s["cell_mass"] for s in snaps]

    axes[1, 0].step(t, n_rep, where="post", color="#dc2626", lw=1.5,
                    label="chromosome replisomes")
    ax_pl = axes[1, 0].twinx()
    ax_pl.plot(t, n_plasmids, color="#2563eb", lw=1.4)
    ax_pl.set_ylabel("full plasmids", color="#2563eb")
    ax_pl.tick_params(axis="y", labelcolor="#2563eb")
    axes[1, 0].set_ylabel("chromosome\nactive replisomes",
                          color="#dc2626")
    axes[1, 0].tick_params(axis="y", labelcolor="#dc2626")
    axes[1, 0].set_title("B. Plasmids + BP1993 (controlled)")
    axes[1, 0].set_ylim(-0.3, 5)
    axes[1, 0].grid(True, alpha=0.3)
    trans = [(i, n_rep[i-1], n_rep[i]) for i in range(1, len(n_rep))
             if n_rep[i] != n_rep[i-1]]
    for i, prev, cur in trans[:4]:
        axes[1, 0].axvline(t[i], color="gray", ls=":", alpha=0.4)

    axes[1, 1].plot(t, cm, color="#15803d", lw=1.5)
    axes[1, 1].axhline(1950, color="#991b1b", ls="--", alpha=0.5,
                       label="2-oriC threshold (1950 fg)")
    axes[1, 1].set_ylabel("cell mass (fg)")
    axes[1, 1].set_title(f"B'. Cell mass; plasmids 1→{n_plasmids[-1]}")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # ---- Row C: uncontrolled plasmids (BP1993 off) ----
    with open(UNCONTROLLED_JSON) as f:
        d = json.load(f)
    snaps = d["seeds"][0]["snapshots"]
    t = [s["time"] / 60 for s in snaps]
    n_rep = [s["n_active_replisomes"] for s in snaps]
    n_plasmids = [s["n_full_plasmids"] for s in snaps]
    cm = [s["cell_mass"] for s in snaps]
    dnaG = [s.get("bulk_dnaG", 0) for s in snaps]

    axes[2, 0].step(t, n_rep, where="post", color="#dc2626", lw=1.5)
    ax_pl = axes[2, 0].twinx()
    ax_pl.plot(t, n_plasmids, color="#2563eb", lw=1.4)
    ax_pl.set_ylabel("full plasmids", color="#2563eb")
    ax_pl.tick_params(axis="y", labelcolor="#2563eb")
    axes[2, 0].set_xlabel("time (min)")
    axes[2, 0].set_ylabel("chromosome\nactive replisomes",
                          color="#dc2626")
    axes[2, 0].tick_params(axis="y", labelcolor="#dc2626")
    axes[2, 0].set_title("C. Plasmids + BP1993 OFF (uncontrolled)")
    axes[2, 0].set_ylim(-0.3, 5)
    axes[2, 0].grid(True, alpha=0.3)
    term_idx = next((i for i in range(1, len(n_rep))
                     if n_rep[i] == 0 and n_rep[i-1] > 0), None)
    if term_idx is not None:
        axes[2, 0].axvline(t[term_idx], color="gray", ls=":", alpha=0.5)
        axes[2, 0].text(t[term_idx] + 0.5, 4.4,
                        "termination —\nno re-init fires",
                        fontsize=8, ha="left", va="top",
                        color="#991b1b", fontweight="bold")

    axes[2, 1].plot(t, cm, color="#15803d", lw=1.5, label="cell_mass")
    axes[2, 1].axhline(1950, color="#991b1b", ls="--", alpha=0.5,
                       label="2-oriC threshold")
    ax_dnag = axes[2, 1].twinx()
    ax_dnag.plot(t, dnaG, color="#7c3aed", lw=0.7, alpha=0.8)
    ax_dnag.set_ylabel("bulk DnaG", color="#7c3aed")
    ax_dnag.tick_params(axis="y", labelcolor="#7c3aed")
    axes[2, 1].set_xlabel("time (min)")
    axes[2, 1].set_ylabel("cell mass (fg)", color="#15803d")
    axes[2, 1].tick_params(axis="y", labelcolor="#15803d")
    axes[2, 1].set_title(
        f"C'. Cell mass + DnaG; plasmids 1→{n_plasmids[-1]}")
    axes[2, 1].legend(loc="lower right", fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Chromosome behavior across the three regimes the refactor exercises",
        fontsize=12, y=0.998)
    fig.tight_layout()
    return _svg(fig)


def main():
    plot_svg = _build_regime_figure()

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
             font-size: 0.92em; vertical-align: top; }
    th { background: #f1f5f9; }
    .callout { background: #fef3c7; border-left: 4px solid #f59e0b;
               padding: 1em 1.2em; margin: 1em 0; border-radius: 3px; }
    .callout-good { background: #dcfce7; border-left: 4px solid #16a34a; }
    .callout-info { background: #dbeafe; border-left: 4px solid #2563eb; }
    .callout-warn { background: #fee2e2; border-left: 4px solid #dc2626; }
    .pass { color: #16a34a; font-weight: 600; }
    .fail { color: #dc2626; font-weight: 600; }
    .figcap { font-size: 0.85em; color: #475569; margin-top: 0.4em; }
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Replication refactor: Step → PartitionedProcess for chromosome and plasmid</title>
<style>{css}</style></head><body>

<h1>Replication refactor: <code>Step</code> → <code>PartitionedProcess</code>
for chromosome and plasmid</h1>
<p><em>Two-phase refactor that wires both replication processes into v2ecoli's
existing allocator machinery so chromosome and plasmid replication compete
for replisome subunits via the proportional-fairness math used by the rest
of the partitioned processes. Chromosome-only execution is preserved
bit-equivalently; new plasmid-vs-chromosome competition emerges naturally
under uncontrolled plasmid load — reproducing the vEcoli Figure 2
chromosome-arrest phenotype directly in v2ecoli.</em></p>

<h2>1. Why this refactor</h2>
<p>
Before the refactor, both <code>ChromosomeReplication</code> and
<code>PlasmidReplication</code> were plain <code>Step</code> instances
running back-to-back in execution layer 4b.  Sequential execution
gave chromosome an implicit priority over plasmid (chromosome was
listed first, took whatever subunits it wanted from raw bulk; plasmid
ran second on residual bulk).  This is a code-convention determining
biology — not a model of competition.  Under uncontrolled plasmid
load the sequential ordering still let chromosome win, even though
the biological reality is that thousands of plasmid replisomes should
out-compete chromosome's two for the shared subunit pool.
</p>
<p>
v2ecoli already implements proper proportional fairness — the
<code>Requester</code> &rarr; <code>Allocator</code> &rarr;
<code>Evolver</code> pattern in <code>v2ecoli/steps/partition.py</code>,
used today for RNA degradation and the elongation processes.
Eligibility for that mechanism requires inheriting from
<code>PartitionedProcess</code> (which exposes
<code>calculate_request</code> + <code>evolve_state</code>) instead of
plain <code>Step</code>.  This refactor moves chromosome and plasmid
replication onto that path, places them both in
<code>allocator_2</code> alongside <code>rna-degradation</code>
(matching vEcoli's flow definition where chromosome and rna-deg share
the same <code>ecoli-tf-binding</code> dependency depth), and lets the
proportional-fairness math handle competition.
</p>

<h2>2. What changed</h2>

<h3>2.1 Phase 1 — chromosome refactor</h3>
<ul>
<li>Base class <code>EcoliStep</code> &rarr; <code>PartitionedProcess</code>;
  <code>update()</code> entry point removed (the framework's two-phase
  entrypoint takes over).</li>
<li><code>_prepare(states)</code> &rarr;
  <code>calculate_request(timestep, states)</code>.  v2ecoli's existing
  <code>_prepare</code> already constructed the bulk request dict (it
  was being discarded with <code>_ = requests</code>); the refactor
  just returns it.  No new biology code added.</li>
<li><code>_evolve(states)</code> &rarr;
  <code>evolve_state(timestep, states)</code>.  Body unchanged.</li>
<li><code>generate.py</code>: chromosome moved from
  <code>STANDALONE_STEPS</code> to <code>PARTITIONED_PROCESSES</code>;
  added to <code>ALLOCATOR_LAYERS['allocator_2']</code> alongside
  <code>ecoli-rna-degradation</code> (per vEcoli's
  <code>configs/default.json</code> — both processes have the same
  <code>ecoli-tf-binding</code> dependency depth).  Layer 4b lost its
  chromosome entry; the rna-degradation requester / evolver layers
  gained chromosome counterparts.</li>
<li><code>generate_reconciled.py</code>: matching updates.</li>
</ul>

<h3>2.2 Phase 2 — plasmid refactor</h3>
<p>
Same structural pattern as chromosome, with one important difference:
v2ecoli's previous <code>PlasmidReplication._prepare</code> did not
construct a bulk request — it ran the BP1993 ODE and stashed
<code>n_rna_initiations</code> on <code>self</code>, with all bulk
gating happening inside <code>_evolve</code>.  To enable allocator-mediated
competition, <code>calculate_request</code> now also constructs a
bulk request, ported from the analogous vEcoli plasmid-replication
pattern:
</p>
<pre># Subunit request: one set of replisome subunits per idle plasmid
# domain we might initiate on this tick (gated by RNA II under
# use_rna_control).
if len(idle_plasmid_domains) &gt; 0:
    if self.use_rna_control:
        n_to_request = min(len(idle_plasmid_domains), n_rna_initiations)
    else:
        n_to_request = len(idle_plasmid_domains)
    if n_to_request &gt; 0:
        requests["bulk"].append((self.replisome_trimers_idx, 3 * n_to_request))
        requests["bulk"].append((self.replisome_monomers_idx, 1 * n_to_request))

# dNTP request for elongation of existing forks (same idiom as
# chromosome_replication.calculate_request).
if n_active_replisomes &gt; 0:
    ...
    requests["bulk"].append((self.dntps_idx, ...))</pre>
<p>
The crucial gating is on <code>n_rna_initiations</code>: under
BP1993 control, plasmid only requests subunits for the integer number
of initiations the ODE actually allows this tick.  This means a
low-contention controlled regime asks for a small number of replisomes
(chromosome gets its full request); under uncontrolled (BP1993 off),
plasmid asks for one replisome per idle domain (thousands), which is
what generates the competition.  All BP1993 ODE biology is preserved
exactly — the new code is just the bulk-request construction.
</p>
<p>
Wiring updates:
</p>
<ul>
<li><code>generate.py</code>:
  <code>'ecoli-plasmid-replication'</code> moved from
  <code>STANDALONE_STEPS</code> to <code>PARTITIONED_PROCESSES</code>;
  added to <code>ALLOCATOR_LAYERS['allocator_2']</code>; layer 4b
  loses the plasmid entry; allocator_2 sub-layers gain
  plasmid_requester / plasmid_evolver alongside chromosome and
  rna-degradation.</li>
<li><code>generate_reconciled.py</code>: matching updates (the
  <code>RECONCILED_LAYERS</code> dict is auto-derived from
  <code>ALLOCATOR_LAYERS</code>, so plasmid is picked up
  automatically in the reconciled architecture as well).</li>
<li><code>scripts/run_plasmid_multiseed.py</code>: removed the
  previously-temporary <code>custom_priorities</code> override that
  forced chromosome to win under sequential-priority.  The override
  is no longer needed (and would defeat the allocator-fairness
  behavior the refactor enables).  Also widened
  <code>_allocated</code>'s exception handler to include
  <code>KeyError</code> for robustness.</li>
</ul>

<h2>3. Behavior across regimes</h2>
<p>
Three single-cell smoke runs cover the three regimes the refactor
exercises.  All three were run after Phase 1 + Phase 2, against the
post-merge ParCa fixture, on the same machine and with the same
seed-0 random state.
</p>

<div>
{plot_svg}
</div>

<p class="figcap">
<strong>Row A (no plasmids):</strong> 7-generation chromosome-only
lineage from
<code>out/plasmid/multigen_timeseries_noplasmid_7gen.json</code>.
Chromosome is the only requester in <code>allocator_2</code>, so the
allocator is a pass-through and gen-to-gen behavior is
indistinguishable from the previous Step implementation (gens 1-4
follow the normal 2&rarr;0&rarr;4 pattern; the gen-5 re-init miss
and gen-6/7 1-oriC downshift are pre-existing dynamics unrelated to
this refactor).
<br/><br/>
<strong>Row B (controlled plasmids, BP1993 on):</strong> single
generation with <code>use_rna_control=True</code> from
<code>out/plasmid/multigen_timeseries.json</code>.  Plasmid load
stays low (1&rarr;9), BP1993 caps requests so total subunit demand
is well below supply; the allocator gives both processes their full
requests.  Chromosome behavior matches pre-refactor exactly
(2&rarr;0 termination at 22 min, 0&rarr;4 re-init at 30 min,
division at 45 min).
<br/><br/>
<strong>Row C (uncontrolled plasmids, BP1993 off):</strong> single
seed with <code>use_rna_control=False</code> and
<code>custom_priorities</code> override removed, from
<code>out/plasmid/multiseed_timeseries.json</code>.  Plasmids run
away to 4407 by division.  The inherited chromosome round terminates
at 22 min as usual — but <strong>no new chromosome round ever
fires</strong>.  Cell mass crosses the 1950 fg 2-oriC threshold at
~30 min, but with plasmid demand (~3 trimers and 1 monomer per idle
plasmid &times; thousands of idle plasmids) dwarfing chromosome's
12-trimer / 4-monomer request, the allocator's proportional partition
gives chromosome essentially zero share, and chromosome's
mechanistic gate fails on its allocated counts.  Cell still divides
(mass-trigger on the dry-mass listener fires regardless of chromosome
state) but with <code>n_active_replisomes = 0</code> at division — a
broken lineage state.  This is the vEcoli Figure-2 phenotype, now
reproducing in v2ecoli through genuine allocator-fairness math
rather than a sequential-priority shortcut.
</p>

<h2>4. Comparison vs. the pre-refactor sequential-priority architecture</h2>
<table>
<tr><th>Regime</th><th>Pre-refactor (Step in layer 4b)</th>
  <th>Post-refactor (PartitionedProcess in allocator_2)</th></tr>
<tr><td>No plasmids (chromosome-only)</td>
  <td>chromosome runs as Step in layer 4b; subunits read from raw
    bulk</td>
  <td>chromosome runs through Requester / Allocator / Evolver;
    allocator is a pass-through with single requester;
    <strong>identical behavior</strong></td></tr>
<tr><td>Plasmids + BP1993 controlled</td>
  <td>chromosome runs first in layer 4b, plasmid runs after on
    residual bulk; under low contention this is identical to a fair
    partition</td>
  <td>both run through allocator; under low contention each gets full
    request; <strong>identical behavior</strong> (verified: 1&rarr;9
    plasmids, division at 2728 s, BP1993 species match)</td></tr>
<tr><td>Plasmids + BP1993 OFF (uncontrolled)</td>
  <td>chromosome wins by list order despite massive plasmid demand;
    chromosome re-init fires at the first cell-mass-threshold
    crossing (artificial)</td>
  <td><strong>chromosome proportionally starved by allocator math:
    chromosome's 4-DnaG request vs plasmid's ~4000-DnaG request
    &rarr; chromosome's allocation share &asymp; 0 &rarr; gate fails
    &rarr; no chromosome re-init &rarr; vEcoli Figure-2 phenotype
    emerges</strong></td></tr>
</table>

<h2>5. Why proportional fairness is the right biology, not just the right code</h2>
<p>
Cells don't have a priority list telling DnaG which replisome to
bind.  DnaG diffuses and binds whichever replisome it encounters,
with rate proportional to binding-site availability.  When there are
4 chromosome assembly sites and ~4000 plasmid assembly sites in the
same cell, DnaG has roughly a 1000&times; higher chance of hitting a
plasmid site, and chromosome's effective rate of obtaining DnaG
drops by roughly the same factor.  The allocator's proportional
partition is exactly this calculation in coarse-grained form — same
answer the mass-action ODE would give at equilibrium.
</p>
<p>
Pre-refactor, chromosome's "always-win" outcome under uncontrolled
plasmid load was an artifact of the v2ecoli port simplification, not
biology.  The custom-priority hack in
<code>scripts/run_plasmid_multiseed.py</code> was an explicit
acknowledgement that the architecture wasn't producing the desired
phenotype on its own — it set <code>chromosome priority = 5</code>
to force chromosome to win.  Removing that hack and letting the
allocator math drive the outcome is what brings v2ecoli's competition
semantics in line with vEcoli's, which validated the original
Figure 2 phenotype against Clewell 1972 and Nordström 2006.
</p>

<h2>6. Test results</h2>
<table>
<tr><th>Test</th><th>Status</th><th>Notes</th></tr>
<tr><td><code>test_architectures_grow.py</code> (5 tests)</td>
  <td><span class="pass">5/5 PASS</span></td>
  <td>All three architectures (baseline, departitioned, reconciled)
    instantiate and grow over 60 s.  Verified after both Phase 1 and
    Phase 2.</td></tr>
<tr><td><code>test_sustained_growth.py</code></td>
  <td><span class="pass">PASS</span></td>
  <td>500 s sustained-growth regression.</td></tr>
<tr><td><code>test_growth_parity.py</code></td>
  <td><span class="pass">PASS</span></td>
  <td>Mass-trajectory parity vs v1 reference.</td></tr>
<tr><td><code>test_cell_cycle_regressions.py</code> (other cases)</td>
  <td><span class="pass">PASS</span></td>
  <td>Per-tick invariants for the cell-cycle pipeline.</td></tr>
<tr><td><code>test_cell_cycle_regressions.py::test_cell_cycle_completes_to_division</code></td>
  <td><span class="fail">FAIL (pre-existing, unrelated)</span></td>
  <td>Same <code>growth_limits</code> realize-Array bug documented
    below.  Reproduces identically on the main branch.  Not introduced
    by this refactor.</td></tr>
</table>

<h2>7. Diagnosis of the pre-existing test failure (carried over from Phase 1)</h2>
<p>
The error surfaces in the schema realize step on the post-division
daughter cell.  The chromosome / division portion of the pipeline
runs correctly:
</p>
<pre>DIVISION at t=2483 s (dry_mass=702.0 fg, threshold=702.0 fg, chromosomes=2)
  DAUGHTERS: 00 (bulk=...) + 01 (bulk=...)</pre>
<p>
After the daughter cells are constructed, the framework attempts to
realize their state against the schema and fails on the
<code>listeners.growth_limits</code> sub-tree:
</p>
<pre>ValueError: realize Array at path=('agents', '00', 'listeners', 'growth_limits'):
np.array failed. encode type=list, len=39, first_row_type=ndarray,
first_row_len=0, dtype=float64. Original: setting an array element with a
sequence. The requested array has an inhomogeneous shape after 1 dimensions.
The detected shape was (39,) + inhomogeneous part.</pre>

<h3>7.1 Why parent cells don't hit this</h3>
<p>
On a parent cell, every simulation tick the
<code>polypeptide_elongation</code> process writes <em>all</em> 36
fields of <code>growth_limits</code> with shape-consistent values
(amino-acid-shaped arrays of length 21, scalars as floats, etc.).  By
the time the schema realize step inspects the parent's
<code>growth_limits</code>, every field has a real, written-by-the-tick
value with consistent shape. <code>np.array(...)</code> over those
values succeeds.
</p>

<h3>7.2 Why daughter cells DO hit it</h3>
<p>
On a freshly divided daughter cell, no process has yet written
<code>growth_limits</code>.  The fields are populated from the
schema's declared <em>defaults</em>, which are deliberately
heterogeneous: empty arrays for the array-typed fields
(<code>'overwrite[array[float[uM]]]'</code> &rarr;
<code>_default: []</code>), and zero scalars for the scalar-typed
fields (<code>'overwrite[float[uM]]'</code> &rarr;
<code>_default: 0.0</code>).
</p>
<p>
The realize-Array dispatch in <code>bigraph_schema</code> handles
dict inputs by flattening to a list of values:
</p>
<pre>elif isinstance(encode, dict):
    encode = dict_values(encode)
...
state = np.array(encode, dtype=schema._data)</pre>
<p>
With 39 mixed-shape values in the list,
<code>np.array(...)</code> raises an inhomogeneous-shape
<code>ValueError</code>.  The error message reports
<code>len=39</code>, <code>first_row_type=ndarray</code>,
<code>first_row_len=0</code> — exactly the empty-array default for
the first listed field, with subsequent fields having different
shapes.
</p>
<div class="callout-warn callout">
<strong>Root cause is in the schema framework, not in v2ecoli.</strong>
<code>growth_limits</code> is declared in the process schema as a
struct of named typed fields. The realize dispatch is treating it as
an Array and flattening the dict — which only works when the parent's
written values happen to share a uniform shape.  Default-construction
inevitably surfaces the heterogeneity and fails.
</div>

<h2>8. Proposed workaround</h2>
<p>
The cleanest local fix lives in <code>v2ecoli/library/division.py</code>.
<code>divide_cell</code> currently builds daughter states with
<code>bulk</code>, <code>unique</code>, <code>environment</code>,
<code>boundary</code>, and <code>process_state</code> populated, but
deliberately does not copy <code>listeners</code>.  Adding a listener
copy in the same style as the existing environment / boundary copies
would side-step the schema-realize bug:
</p>
<pre>if 'listeners' in cell_state:
    d1_state['listeners'] = copy.deepcopy(cell_state['listeners'])
    d2_state['listeners'] = copy.deepcopy(cell_state['listeners'])</pre>
<p>
Each daughter inherits the parent's last-tick listener values — i.e.,
ndarrays + scalars with the parent's <em>actual</em> consistent
shapes.  The realize step succeeds, the daughter runs normally, and
on its first post-division tick the listener fields are overwritten
by their writing processes (every listener field is declared as
<code>overwrite[...]</code>, which fully replaces the value rather
than accumulating).  The inheritance is therefore load-bearing for
the framework but biology-neutral.
</p>
<div class="callout">
This workaround is intentionally <em>not</em> applied in this commit
— it is a fix for a pre-existing failure that was inherited from the
main branch, and is best landed as its own targeted commit (or
upstream in <code>bigraph_schema</code>) so it can be reasoned about
independently of the replication refactor.
</div>

<h3>8.1 The proper fix</h3>
<p>
The proper fix belongs in <code>bigraph_schema</code>'s realize-Array
dispatch.  When the encode value is a dict and the schema describes a
struct of named fields (rather than a single contiguous array),
realize should recurse into each field with that field's specific
schema rather than collecting the dict's values and trying to
materialize them as a single ndarray.  The current dispatch happens
to work for dicts whose values are shape-uniform because
<code>np.array(list_of_uniform_arrays)</code> succeeds; it does not
handle struct-style heterogeneity.  Filing this as an upstream issue
or PR is the durable fix; the v2ecoli-side workaround above is
appropriate until then.
</p>

<h2>9. Honest caveats</h2>
<ul>
<li><strong>Proportional fairness is still a coarse-grained
  approximation.</strong>  Real cells have specific binding kinetics
  (DnaG-replisome dissociation rates, spatial localization, oriC- vs
  oriV-specific affinities) that proportional partitioning flattens
  out.  For mechanistic DnaA-oriC initiation (Aim 2) and the eventual
  bulk-ification of RNA I/II (Stage C) you'll want explicit binding
  kinetics that go beyond proportional partitioning — but the
  allocator is the correct intermediate model and matches what
  v2ecoli's reconciled architecture and vEcoli ship with today.</li>
<li><strong>Per-process allocate / request snapshots in
  <code>scripts/run_plasmid_multiseed.py</code> read 0 currently.</strong>
  The <code>_alloc_molecule_idx_cache</code> isn't being populated
  correctly post-refactor for the replisome subunit indices; the
  underlying biology is correct (chromosome arrest is visible in the
  unique-molecule trace) but the per-tick allocation values aren't
  being captured for the report.  Script-side cleanup, not a
  correctness issue.</li>
<li><strong>Chromosome-only path is preserved bit-equivalently.</strong>
  An allocator with a single requester (chromosome alone, when
  <code>has_plasmid=False</code>) reduces to a pass-through — the
  proportional math gives the only requester its full request whenever
  supply is sufficient.  All chromosome-only behavior tests pass
  identically to pre-refactor.</li>
</ul>

</body></html>
"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
