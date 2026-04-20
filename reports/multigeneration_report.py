"""v2ecoli multigeneration report.

Runs a single cell to division, keeps exactly one daughter, runs that to
division, keeps one of its daughters, etc. — for a configurable number
of generations. Plots mass trajectories end-to-end across all
generations and writes an HTML report (same provenance banner as the
rest of the reports).

    python reports/multigeneration_report.py --generations 3

Output: out/multigeneration/multigeneration_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.composite import make_composite, _build_core
from v2ecoli.library.division import divide_cell
from v2ecoli.generate import build_document
from process_bigraph import Composite


OUTPUT_DIR = "out/multigeneration"
CACHE_DIR = "out/cache" if os.path.isdir("out/cache") else "out/workflow/cache"
SNAPSHOT_INTERVAL = 50  # seconds between mass captures
MAX_GENERATION_DURATION = 3600  # safety cap per generation

# Keys copied when carrying a divided daughter state forward.
_CELL_DATA_KEYS = {
    "bulk",
    "unique",
    "listeners",
    "environment",
    "boundary",
    "global_time",
    "timestep",
    "divide",
    "division_threshold",
    "process_state",
    "allocator_rng",
}


@dataclass
class GenerationResult:
    index: int
    duration: float
    wall_time: float
    divided: bool
    initial_dry_mass: float
    final_dry_mass: float
    snapshots: list[dict] = field(default_factory=list)
    cell_data_after: dict[str, Any] | None = None


def _get_emitter_instance(composite):
    """Return the emitter instance for agent 0 (while the agent still exists)."""
    cell = composite.state["agents"].get("0")
    if not cell:
        return None
    emitter_edge = cell.get("emitter", {})
    if isinstance(emitter_edge, dict):
        return emitter_edge.get("instance")
    return None


def _snapshots_from_history(history) -> list[dict]:
    """Turn an emitter history list into mass snapshots."""
    snaps = []
    for snap in history:
        t = snap.get("global_time", 0)
        if int(t) % SNAPSHOT_INTERVAL != 0 and t != 1:
            continue
        mass = (
            snap.get("listeners", {}).get("mass", {})
            if isinstance(snap.get("listeners"), dict)
            else {}
        )
        snaps.append(
            {
                "time": float(t),
                "dry_mass": float(mass.get("dry_mass", 0)),
                "cell_mass": float(mass.get("cell_mass", 0)),
                "protein_mass": float(mass.get("protein_mass", 0)),
                "dna_mass": float(mass.get("dna_mass", 0)),
                "rRna_mass": float(mass.get("rRna_mass", 0)),
                "tRna_mass": float(mass.get("tRna_mass", 0)),
                "mRna_mass": float(mass.get("mRna_mass", 0)),
                "smallMolecule_mass": float(mass.get("smallMolecule_mass", 0)),
            }
        )
    return snaps


def _extract_cell_data(cell: dict) -> dict[str, Any]:
    """Copy the state keys needed to seed the next generation."""
    return {
        k: v
        for k, v in cell.items()
        if k in _CELL_DATA_KEYS
        or k.startswith("request_")
        or k.startswith("allocate_")
    }


def _run_generation(
    composite: Composite,
    gen_idx: int,
    max_duration: float,
) -> GenerationResult:
    """Run the composite forward in SNAPSHOT_INTERVAL chunks until division
    or the per-generation duration cap. Returns a GenerationResult plus
    the last observed cell_data snapshot (for carrying into the next
    generation)."""
    cell = composite.state["agents"]["0"]
    initial_dry = float(cell["listeners"]["mass"].get("dry_mass", 0))

    # Grab the emitter instance NOW — the agent node (and our edge handle)
    # gets detached from composite.state once the Division step fires.
    emitter_instance = _get_emitter_instance(composite)

    t_wall0 = time.time()
    total_run = 0.0
    divided = False
    last_cell_data: dict[str, Any] | None = None

    while total_run < max_duration:
        chunk = min(SNAPSHOT_INTERVAL, max_duration - total_run)
        try:
            composite.run(chunk)
        except Exception as e:
            err_str = str(e)
            if (
                "divide" in err_str.lower()
                or "_add" in err_str
                or "_remove" in err_str
            ):
                total_run += chunk
                divided = True
                break
            # Post-add realize errors (e.g. growth_limits Array) fire after
            # the structural divide has already happened. If the mother agent
            # is gone, treat as divided — don't keep running the broken tree.
            if composite.state.get("agents", {}).get("0") is None:
                divided = True
                break
            raise
        total_run += chunk

        cur_cell = composite.state.get("agents", {}).get("0")
        if cur_cell is None:
            divided = True
            break
        last_cell_data = _extract_cell_data(cur_cell)

    wall_time = time.time() - t_wall0
    history = emitter_instance.history if emitter_instance is not None else []
    snaps = _snapshots_from_history(history)
    # Drop post-division snapshots where the listener has been reset
    # (agent removed) — we want the mass at the moment of division.
    snaps = [s for s in snaps if s.get("dry_mass", 0) > 0]
    final_dry = (
        snaps[-1]["dry_mass"]
        if snaps
        else float(
            composite.state.get("agents", {})
            .get("0", {})
            .get("listeners", {})
            .get("mass", {})
            .get("dry_mass", 0)
        )
    )

    return GenerationResult(
        index=gen_idx,
        duration=total_run,
        wall_time=wall_time,
        divided=divided,
        initial_dry_mass=initial_dry,
        final_dry_mass=final_dry,
        snapshots=snaps,
        cell_data_after=last_cell_data,
    )


def run_multigeneration(
    n_generations: int,
    max_duration_per_gen: float,
) -> list[GenerationResult]:
    """Run n_generations of single-lineage cells, carrying one daughter
    forward across each division."""
    # Load cache configs for rebuilding daughter composites.
    with open(os.path.join(CACHE_DIR, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    # On the pint-migrated branches, cache contains pint Quantities whose
    # registry identity is lost across dill; rebind them to the shared
    # app-registry. No-op on main (cache is pure-Unum there).
    try:
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache)
    except ImportError:
        pass
    configs = cache.get("configs", {})
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    results: list[GenerationResult] = []

    # Generation 1 — start from the fresh initial state the workflow/canary use.
    print(f"  Gen 1: building from cache {CACHE_DIR}")
    composite = make_composite(cache_dir=CACHE_DIR)
    cell0 = composite.state["agents"]["0"]
    print(
        f"    initial dry_mass={cell0['listeners']['mass'].get('dry_mass',0):.1f} fg"
    )

    gen = _run_generation(composite, 1, max_duration_per_gen)
    print(
        f"    gen 1: {gen.wall_time:.0f}s wall, "
        f"sim {gen.duration:.0f}s, "
        f"dry_mass {gen.initial_dry_mass:.0f}→{gen.final_dry_mass:.0f}, "
        f"divided={gen.divided}"
    )
    results.append(gen)

    prev_cell_data = gen.cell_data_after

    # Generations 2..N — divide, keep daughter 1, build a fresh composite.
    for gen_idx in range(2, n_generations + 1):
        if prev_cell_data is None or "bulk" not in prev_cell_data:
            print(f"    gen {gen_idx}: no prior cell state — stopping")
            break
        print(f"  Gen {gen_idx}: dividing previous cell, keeping daughter 1")
        d1_state, _d2_state = divide_cell(prev_cell_data)

        t_build0 = time.time()
        doc = build_document(
            d1_state,
            configs,
            unique_names,
            dry_mass_inc_dict=dry_mass_inc,
            seed=gen_idx,
        )
        composite = Composite(doc, core=_build_core())
        build_time = time.time() - t_build0
        print(f"    built daughter composite in {build_time:.1f}s")

        gen = _run_generation(composite, gen_idx, max_duration_per_gen)
        print(
            f"    gen {gen_idx}: {gen.wall_time:.0f}s wall, "
            f"sim {gen.duration:.0f}s, "
            f"dry_mass {gen.initial_dry_mass:.0f}→{gen.final_dry_mass:.0f}, "
            f"divided={gen.divided}"
        )
        results.append(gen)
        prev_cell_data = gen.cell_data_after

    return results


MASS_KEYS = [
    ("dry_mass", "Dry Mass", "k"),
    ("protein_mass", "Protein", "#22c55e"),
    ("dna_mass", "DNA", "#8b5cf6"),
    ("rRna_mass", "rRNA", "#3b82f6"),
    ("tRna_mass", "tRNA", "#06b6d4"),
    ("mRna_mass", "mRNA", "#f97316"),
    ("smallMolecule_mass", "Small mol", "#f59e0b"),
]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_multigeneration_mass(results: list[GenerationResult]) -> str:
    """Plot mass fold change across all generations on one concatenated
    time axis. Each generation re-normalizes fold change to 1.0 at its
    own start; generation boundaries are drawn as vertical dashed lines."""
    fig, (ax_abs, ax_fold) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True
    )
    fig.suptitle(
        f"Multigeneration lineage — {len(results)} generation(s)",
        fontsize=13,
    )

    cumulative_t = 0.0
    gen_boundaries: list[float] = []

    for gen in results:
        if not gen.snapshots:
            continue
        times = np.array([s["time"] for s in gen.snapshots])
        t_offset = cumulative_t
        plot_times = (times + t_offset) / 60.0  # minutes

        for key, label, color in MASS_KEYS:
            vals = np.array([s.get(key, 0) for s in gen.snapshots], dtype=float)
            if vals.size == 0 or vals[0] <= 0:
                continue
            # Only label on the first generation to keep the legend tidy
            legend_label = label if gen.index == 1 else None
            ax_abs.plot(plot_times, vals, color=color, lw=1.4, label=legend_label)
            ax_fold.plot(
                plot_times,
                vals / vals[0],
                color=color,
                lw=1.4,
                label=legend_label,
            )

        cumulative_t += gen.duration
        gen_boundaries.append(cumulative_t)

    for ax, ylabel, title in (
        (ax_abs, "Mass (fg)", "Absolute mass"),
        (ax_fold, "Fold change (within each generation)", "Per-generation fold change"),
    ):
        for b in gen_boundaries[:-1]:
            ax.axvline(b / 60.0, ls="--", color="#64748b", alpha=0.4, lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)
        if ax is ax_abs:
            ax.legend(fontsize=8, ncol=4, loc="upper left")

    ax_fold.set_xlabel("Cumulative time (min)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_html_report(
    results: list[GenerationResult],
    report_plot_b64: str,
    output_path: str,
    pipeline_wall_time: float,
) -> None:
    try:
        from v2ecoli.library.repro_banner import banner_html

        repro_banner = banner_html()
    except Exception:
        repro_banner = ""

    gens_rows = ""
    total_sim_time = 0.0
    for gen in results:
        total_sim_time += gen.duration
        gens_rows += (
            "<tr>"
            f"<td>{gen.index}</td>"
            f"<td>{gen.duration:.0f}</td>"
            f"<td>{gen.wall_time:.0f}</td>"
            f"<td>{gen.initial_dry_mass:.1f}</td>"
            f"<td>{gen.final_dry_mass:.1f}</td>"
            f"<td>{gen.final_dry_mass/max(gen.initial_dry_mass,1e-9):.2f}×</td>"
            f"<td class=\"{'green' if gen.divided else 'red'}\">"
            f"{'divided' if gen.divided else 'no division'}</td>"
            "</tr>"
        )

    n_snaps_total = sum(len(g.snapshots) for g in results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(
            f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>v2ecoli — multigeneration report</title>
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
    min-width: 160px;
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
  .green {{ color: #15803d; }}
  .red {{ color: #b91c1c; }}
  .plot img {{ max-width: 100%; }}
  p.intro {{ color: #475569; }}
</style>
</head>
<body>
{repro_banner}

<h1>Multigeneration lineage</h1>
<p class="intro">
  Single-lineage simulation: start from one newborn cell, run to division,
  keep exactly one daughter, repeat. The plot below shows mass over time
  across all generations, with dashed lines at generation boundaries.
</p>

<div class="metric-row">
  <div class="metric">
    <div class="label">Generations</div>
    <div class="value">{len(results)}</div>
  </div>
  <div class="metric">
    <div class="label">Total simulated time</div>
    <div class="value">{total_sim_time/60:.0f} min</div>
  </div>
  <div class="metric">
    <div class="label">Total wall time</div>
    <div class="value">{pipeline_wall_time:.0f} s</div>
  </div>
  <div class="metric">
    <div class="label">Snapshots captured</div>
    <div class="value">{n_snaps_total}</div>
  </div>
</div>

<h2>Per-generation summary</h2>
<table>
  <thead>
    <tr>
      <th>Gen</th>
      <th>Sim time (s)</th>
      <th>Wall (s)</th>
      <th>Initial dry mass (fg)</th>
      <th>Final dry mass (fg)</th>
      <th>Growth</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    {gens_rows}
  </tbody>
</table>

<h2>Mass across generations</h2>
<div class="plot">
  <img src="data:image/png;base64,{report_plot_b64}" alt="multigeneration mass plot">
</div>

<p style="color:#94a3b8; font-size:0.85em; margin-top:48px;">
  Generated by <code>reports/multigeneration_report.py</code> at {time.strftime('%Y-%m-%d %H:%M:%S')}.
  Each generation is seeded from the previous cell's divide-time state via
  <code>v2ecoli.library.division.divide_cell</code> (daughter 1 kept).
</p>
</body>
</html>"""
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations to run (default: 3).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_GENERATION_DURATION,
        help="Safety cap (seconds) per generation (default: 3600).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory (e.g. out/cache_plasmid for a "
             "plasmid-enabled run).",
    )
    args = parser.parse_args()

    if args.cache_dir is not None:
        global CACHE_DIR
        CACHE_DIR = args.cache_dir

    print("=" * 60)
    print(f"v2ecoli multigeneration report — {args.generations} generation(s)")
    print("=" * 60)

    t_pipeline = time.time()
    results = run_multigeneration(args.generations, args.max_duration)
    pipeline_wall = time.time() - t_pipeline

    print("  Generating plot…")
    plot_b64 = plot_multigeneration_mass(results)

    print("  Writing HTML report…")
    report_path = os.path.join(OUTPUT_DIR, "multigeneration_report.html")
    generate_html_report(results, plot_b64, report_path, pipeline_wall)

    print("=" * 60)
    print(f"Pipeline wall time: {pipeline_wall:.0f} s")
    print(f"Report: {report_path}")

    try:
        import subprocess

        subprocess.run(["open", report_path], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
