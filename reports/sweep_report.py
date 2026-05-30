"""Self-contained HTML report for a v2ecoli workflow sweep.

Reads the hive-partitioned parquet a sweep emits, embeds matplotlib plots as
base64 PNGs, prepends a git/run provenance banner, and appends how-to-run +
framework docs. Output is one standalone ``.html`` (portable — no external
assets, so it survives even if the parquet is later deleted).

    python reports/sweep_report.py <sweep_dir>           # dir holding parquet/ (+ sweep.pbg)
    python reports/sweep_report.py <sweep_dir> --open

``<sweep_dir>`` is the run directory: it contains ``parquet/`` (hive layout
``experiment_id=…/variant=…/lineage_seed=…/generation=…/agent_id=…``) and,
optionally, ``sweep.pbg``. Output defaults to ``<sweep_dir>/report.html``.

Requires the ``[parquet]`` extra (duckdb) and matplotlib.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import glob
import io
import os
import platform
import re
import subprocess
import sys

PR = "#95"

# Evergreen documentation embedded into every report ------------------------
HOWTO_HTML = """
<h2>How to run your own sweep</h2>
<p>Install the console script once, then drive a sweep from a vEcoli-style JSON config:</p>
<pre>uv pip install --python .venv/bin/python -e .          # registers v2ecoli-workflow
v2ecoli-workflow --config v2ecoli/configs/two_generations.json --out out/myrun
v2ecoli-workflow --config &lt;cfg&gt; --build-only            # write sweep.pbg without running
v2ecoli-workflow --config &lt;cfg&gt; --max-sim-time 20000   # sim-time cap (seconds)</pre>
<p>…or programmatically:</p>
<pre>from v2ecoli.workflow.run import run_workflow
run_workflow(config, max_sim_time=20000, pbg_out="out/myrun/sweep.pbg")</pre>

<h3>Config knobs</h3>
<table class="doc">
<tr><th>key</th><th>meaning</th></tr>
<tr><td><code>n_init_sims</code></td><td>number of <b>seeds</b> — independent replicate lineages</td></tr>
<tr><td><code>generations</code></td><td>cell divisions per lineage</td></tr>
<tr><td><code>lineage_seed</code></td><td>first seed; seeds = <code>lineage_seed … lineage_seed+n_init_sims-1</code></td></tr>
<tr><td><code>single_daughters</code></td><td><code>true</code>: keep one daughter per division. <code>false</code> (binary tree) is not yet implemented — raises a clear error.</td></tr>
<tr><td><code>variants</code></td><td>declarative parameter sweep (see below)</td></tr>
<tr><td><code>different_seeds_per_variant</code></td><td>give each variant a non-overlapping seed range</td></tr>
<tr><td><code>skip_baseline</code></td><td>omit the unmodified baseline variant from the grid</td></tr>
<tr><td><code>cache_dir</code> / <code>out_dir</code></td><td>ParCa cache in; parquet + <code>sweep.pbg</code> out</td></tr>
<tr><td><code>time_step</code></td><td>composite tick cadence (s)</td></tr>
<tr><td><code>max_duration_per_gen</code></td><td>per-generation safety cap (s)</td></tr>
<tr><td><code>inherit_from</code></td><td>list of parent configs to merge (priority: current &gt; first &gt; …)</td></tr>
<tr><td><code>analysis_options</code></td><td>declared analyses; <code>mass_fraction_summary</code> implemented, cross-cell scales pending</td></tr>
</table>

<h3>Changing seeds &amp; generations</h3>
<p>The sweep grid is <b>variants × seeds</b>, and each branch walks <code>generations</code> divisions. To run 4 seeds × 2 generations:</p>
<pre>{ "inherit_from": ["default.json"], "experiment_id": "4seed_2gen",
  "n_init_sims": 4, "generations": 2 }</pre>

<h3>Variants (parameter sweep)</h3>
<p>A variant overrides a process <code>config_schema</code> value by dotted path. Single parameter:</p>
<pre>"variants": { "kcat": { "target": "ecoli-metabolism.kcat", "value": [1, 2, 3] } }</pre>
<p>Multiple parameters combine via <code>op</code> (<code>prod</code> | <code>zip</code> | <code>add</code>); <code>linspace</code> (or any numpy generator) is also accepted:</p>
<pre>"variants": { "sweep": {
    "a": { "target": "p.a", "value": [1, 2] },
    "b": { "target": "p.b", "linspace": { "start": 0.0, "stop": 1.0, "num": 3 } },
    "op": "prod" } }</pre>
<p>This yields 2&times;3 = 6 variant points; crossed with <code>n_init_sims</code> seeds it produces one
<b>branch</b> per (variant, seed). Each branch emits hive-partitioned parquet keyed by
<code>experiment_id / variant / lineage_seed / generation / agent_id</code>.
<span class="meta">(Deferred: <code>nested</code> grammar and <code>sim_data</code>-recomputing variants such as gene knockouts.)</span></p>
"""

PR_HTML = """
<h2>What this framework adds (PR {pr})</h2>
<p>Brings vEcoli's multiseed / multigeneration / variant workflow framework into v2ecoli as
<b>pure process-bigraph</b> — the whole sweep is one inspectable meta-composite document, replacing NextFlow.</p>
<ul>
 <li><b>Config loader</b> (<code>workflow/config.py</code>) — vEcoli <code>inherit_from</code> chains, list-merge, cycle detection.</li>
 <li><b>Variant grammar</b> (<code>workflow/variants.py</code>) — <code>value</code>/<code>linspace</code> + <code>op: prod|zip|add</code> → declarative process-config overrides → branch grid.</li>
 <li><b>Declarative variants</b> (<code>composites/baseline.py</code>) — <code>config_overrides</code> patch process config at build time without mutating the shared ParCa cache.</li>
 <li><b>LineageProcess</b> (<code>workflow/lineage.py</code>) — one (variant, seed) lineage as an embeddable Process; single-daughter generation walk with per-generation parquet metadata.</li>
 <li><b>Meta-composite + registration</b> (<code>workflow/meta_composite.py</code>) — one <code>LineageProcess</code> branch per (variant, seed); <code>local:</code> address resolution.</li>
 <li><b>Driver + CLI</b> (<code>workflow/run.py</code>) — <code>v2ecoli-workflow</code>; ticks until all branches complete, saves <code>sweep.pbg</code>.</li>
 <li><b>Analyses as Steps</b> (<code>workflow/analysis.py</code>) — <code>AnalysisStep</code> base + five-scale registry + the <code>MassFractionSummary</code> example.</li>
 <li><b>Ported configs</b> — <code>configs/default.json</code>, <code>configs/two_generations.json</code>.</li>
 <li><b>Parquet emits bulk molecules</b> — the parquet path captures the ~16k-molecule <code>bulk__id</code>/<code>bulk__count</code> arrays (RAM-only before).</li>
 <li><b>Fix: double-divide</b> — daughters were halved twice (started ~¼ mother mass, grew too slowly to divide). Now the inner Division's daughter is carried directly; generations divide normally.</li>
</ul>
<p class="meta">Remaining follow-ups: cross-cell analysis Steps (multiseed / multigeneration / multivariant), <code>single_daughters=false</code> binary-tree lineages, resume, and parallel execution.</p>
""".replace("{pr}", PR)

MASS_COLS = ["global_time", "listeners__mass__dry_mass", "listeners__mass__protein_mass",
             "listeners__mass__rRna_mass", "listeners__mass__dna_mass"]

_SEED_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
_VARIANT_STYLES = ["-", "--", ":", "-."]


def _git(*args, cwd, default=""):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def load_cells(sweep_dir):
    """Return ``{(variant, seed, gen): numpy array of MASS_COLS}`` from parquet."""
    import duckdb
    import numpy as np

    pattern = os.path.join(sweep_dir, "**", "history", "**", "*.pq")
    by_cell = {}
    for f in glob.glob(pattern, recursive=True):
        m = re.search(r"variant=([^/]+)/lineage_seed=(\d+)/generation=(\d+)", f)
        if not m:
            # variant may be absent in some layouts — fall back to seed/gen only
            m2 = re.search(r"lineage_seed=(\d+)/generation=(\d+)", f)
            if not m2:
                continue
            key = (0, int(m2.group(1)), int(m2.group(2)))
        else:
            v = m.group(1)
            key = (int(v) if v.isdigit() else v, int(m.group(2)), int(m.group(3)))
        by_cell.setdefault(key, []).append(f)

    cells = {}
    for key, files in by_cell.items():
        union = " UNION ALL ".join(
            f"SELECT {','.join(MASS_COLS)} FROM read_parquet('{f}')" for f in files)
        recs = duckdb.sql(f"SELECT * FROM ({union}) t ORDER BY global_time").fetchall()
        cells[key] = np.array(recs, dtype=float)
    return cells


def _b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _plots(cells):
    """Return (trajectory_png_b64, fractions_png_b64, fractions_dict, div_rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = sorted({v for v, s, g in cells})
    seeds = sorted({s for v, s, g in cells})
    gens = sorted({g for v, s, g in cells})

    # mass fractions per cell
    frac, div_rows = {}, []
    for (v, s, g), a in cells.items():
        t, dry, prot, rrna, dna = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        frac[(v, s, g)] = {"protein": float(np.mean(prot / dry)),
                           "rRNA": float(np.mean(rrna / dry)),
                           "DNA": float(np.mean(dna / dry))}
        div_rows.append((v, s, g, dry[0], dry[-1], t[-1]))

    # Plot 1: saw-tooth dry-mass trajectory per (variant, seed) lineage
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (v, s) in sorted({(v, s) for v, s, g in cells}):
        offset, first = 0.0, True
        for g in gens:
            a = cells.get((v, s, g))
            if a is None:
                continue
            t, dry = a[:, 0], a[:, 1]
            label = (f"seed {s}" + (f" / v{v}" if len(variants) > 1 else "")) if first else None
            ax.plot((t + offset) / 60.0, dry,
                    color=_SEED_COLORS[seeds.index(s) % len(_SEED_COLORS)],
                    ls=_VARIANT_STYLES[variants.index(v) % len(_VARIANT_STYLES)],
                    lw=1.4, label=label)
            ax.axvline(offset / 60.0, color="#9ca3af", ls=":", lw=0.7)
            offset += t[-1]
            first = False
    ax.set_xlabel("time across lineage (min)")
    ax.set_ylabel("dry mass (fg)")
    ax.set_title("Multigeneration dry-mass trajectory (grow → divide)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    p1 = _b64(fig)

    # Plot 2: protein fraction vs generation + grouped fraction bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for (v, s) in sorted({(v, s) for v, s, g in cells}):
        ys = [frac[(v, s, g)]["protein"] for g in gens if (v, s, g) in frac]
        xs = [g for g in gens if (v, s, g) in frac]
        axes[0].plot(xs, ys, marker="o",
                     color=_SEED_COLORS[seeds.index(s) % len(_SEED_COLORS)],
                     ls=_VARIANT_STYLES[variants.index(v) % len(_VARIANT_STYLES)])
    axes[0].set_title("Protein mass fraction vs generation")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("protein / dry mass")
    axes[0].set_xticks(gens)
    axes[0].grid(alpha=0.25)
    width = 0.25
    for i, comp in enumerate(["protein", "rRNA", "DNA"]):
        vals = [np.mean([frac[k][comp] for k in frac if k[2] == g]) for g in gens]
        axes[1].bar(np.array(gens) + (i - 1) * width, vals, width, label=comp)
    axes[1].set_title("Mean mass fractions (all cells)")
    axes[1].set_xlabel("generation")
    axes[1].set_xticks(gens)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, which="both")
    p2 = _b64(fig)
    return p1, p2, frac, sorted(div_rows)


def _provenance(sweep_dir, cells, repo):
    variants = sorted({v for v, s, g in cells})
    seeds = sorted({s for v, s, g in cells})
    gens = sorted({g for v, s, g in cells})
    return {
        "branch": _git("branch", "--show-current", cwd=repo, default="?"),
        "commit": _git("rev-parse", "--short", "HEAD", cwd=repo, default="?"),
        "subject": _git("log", "-1", "--pretty=%s", cwd=repo, default="?"),
        "date": _git("log", "-1", "--pretty=%cd", "--date=short", cwd=repo, default="?"),
        "dirty": bool(_git("status", "--porcelain", cwd=repo)),
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "sweep_dir": sweep_dir,
        "n_variants": len(variants), "n_seeds": len(seeds),
        "n_gens": len(gens), "n_cells": len(cells),
        "seeds": seeds, "gens": gens,
    }


def render_html(provenance, plot1, plot2, frac, div_rows):
    """Assemble the full standalone HTML document (pure: no IO)."""
    dirty = " <span class='warn'>+ uncommitted changes</span>" if provenance["dirty"] else ""
    has_variant = provenance["n_variants"] > 1
    prov = f"""<div class="prov">
  <div class="prov-title">Provenance</div>
  <table class="kv">
   <tr><td>repo / branch</td><td><code>v2ecoli</code> @ <code>{provenance['branch']}</code> &middot; PR <code>{PR}</code></td></tr>
   <tr><td>commit</td><td><code>{provenance['commit']}</code> — {provenance['subject']} ({provenance['date']}){dirty}</td></tr>
   <tr><td>generated</td><td>{provenance['generated']} on <code>{provenance['host']}</code> &middot; Python {provenance['python']}</td></tr>
   <tr><td>sweep</td><td>{provenance['n_variants']} variant(s) &times; {provenance['n_seeds']} seed(s) &times; {provenance['n_gens']} generation(s) = {provenance['n_cells']} cells</td></tr>
   <tr><td>source</td><td><code>{provenance['sweep_dir']}</code></td></tr>
  </table>
</div>"""

    rows = ""
    for (v, s, g, m0, m1, t1) in div_rows:
        fr = frac[(v, s, g)]
        vcell = f"<td>{v}</td>" if has_variant else ""
        rows += (f"<tr>{vcell}<td>{s}</td><td>{g}</td><td>{m0:.0f}</td><td>{m1:.0f}</td>"
                 f"<td>{t1:.0f}</td><td>{fr['protein']:.3f}</td>"
                 f"<td>{fr['rRNA']:.3f}</td><td>{fr['DNA']:.3f}</td></tr>")
    vhdr = "<th>variant</th>" if has_variant else ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>v2ecoli workflow sweep report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem;color:#1f2937;line-height:1.55}}
 h1{{font-size:1.5rem;margin-bottom:.2rem}} h2{{font-size:1.15rem;margin-top:2.2rem;border-bottom:1px solid #e5e7eb;padding-bottom:.3rem}}
 h3{{font-size:1rem;margin-top:1.4rem}}
 .meta{{color:#6b7280;font-size:.9rem}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem;margin-top:.5rem}}
 th,td{{border:1px solid #e5e7eb;padding:.35rem .6rem;text-align:right;vertical-align:top}} th{{background:#f9fafb}}
 table.doc td,table.doc th{{text-align:left}} .doc tr td:first-child{{width:230px;white-space:nowrap}}
 td:first-child{{text-align:center}}
 pre{{background:#0f172a;color:#e2e8f0;padding:.7rem .9rem;border-radius:6px;overflow-x:auto;font-size:.82rem;line-height:1.4}}
 code{{background:#f3f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}} pre code{{background:none;padding:0}}
 .prov{{background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #2563eb;border-radius:6px;padding:.6rem 1rem;margin:1rem 0}}
 .prov-title{{font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:#2563eb;margin-bottom:.3rem}}
 .prov table.kv{{width:100%}} .prov td{{border:none;padding:.2rem .5rem .2rem 0;text-align:left;font-size:.86rem}}
 .prov td:first-child{{color:#6b7280;white-space:nowrap;width:120px}} .warn{{color:#b45309;font-weight:600}}
 .note{{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:.6rem .9rem;font-size:.9rem;margin-top:1rem}}
 ul{{font-size:.93rem}} li{{margin:.25rem 0}}
</style></head><body>
<h1>v2ecoli workflow sweep report</h1>
<p class="meta">Pure process-bigraph meta-composite sweep</p>
{prov}
<h2>Multigeneration mass trajectory</h2>
<img src="data:image/png;base64,{plot1}">
<p class="meta">Each lineage grows from a newborn to its division threshold, divides, and one daughter
is carried forward (dotted lines = generation boundaries).</p>
<h2>Mass-fraction analysis (<code>mass_fraction_summary</code>, single scale)</h2>
<img src="data:image/png;base64,{plot2}">
<h2>Per-cell summary</h2>
<table>
<tr>{vhdr}<th>seed</th><th>gen</th><th>newborn (fg)</th><th>division (fg)</th><th>cycle (s)</th>
<th>protein</th><th>rRNA</th><th>DNA</th></tr>
{rows}
</table>
{HOWTO_HTML}
{PR_HTML}
<div class="note"><b>Scope:</b> <code>mass_fraction_summary</code> (single scale) is the one implemented
analysis Step. Raw bulk molecule counts are in the emitted parquet (<code>bulk__id</code> / <code>bulk__count</code>).</div>
</body></html>"""


def build_report(sweep_dir, out=None, repo=None):
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cells = load_cells(sweep_dir)
    if not cells:
        raise SystemExit(f"no sweep parquet found under {sweep_dir!r} "
                         f"(expected …/history/…/*.pq)")
    plot1, plot2, frac, div_rows = _plots(cells)
    prov = _provenance(sweep_dir, cells, repo)
    html = render_html(prov, plot1, plot2, frac, div_rows)
    out = out or os.path.join(sweep_dir, "report.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", help="run dir holding parquet/ (+ sweep.pbg)")
    p.add_argument("--out", default=None, help="output html (default <sweep_dir>/report.html)")
    p.add_argument("--open", action="store_true", help="open the report when done")
    args = p.parse_args()
    out = build_report(args.sweep_dir, out=args.out)
    print(f"wrote {out}")
    if args.open:
        import subprocess as sp
        sp.run(["open", out], check=False)


if __name__ == "__main__":
    main()
