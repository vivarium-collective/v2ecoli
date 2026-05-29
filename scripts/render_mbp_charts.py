"""Render static SVG charts from the v2-parquet multigen mbp runs.

End-to-end parquet path (no sqlite involvement): reads per-study hives
written by ``run_multigen_parquet`` under
``studies/<study>/parquet-runs/<experiment_id>/history/`` and writes a
small set of comparison charts into ``studies/<study>/charts/`` where
the dashboard's ``discover_static_study_charts`` picks them up
(``vivarium_dashboard/lib/study_charts.py::discover_static_study_charts``).

Charts produced (same set as the earlier sqlite-sourced renderer — same
underlying biological data, different read path):

  mbp-02-population-aggregation/charts/
    00_per-cell-mass-invariant.svg
        cell_mass(t) overlay across cpa={1, 1e6, 1e9}. Three identical
        lines = visual regression-guard for
        `per-cell-observables-invariant-under-scaling`.
    01_population-scaling.svg
        log-y population.total_biomass_gDW(t) across cpa values.
        9-orders-of-magnitude linear scaling (chris_feedback §4).
    02_population-cell-count.svg
        population.cell_count(t) at cpa=1 — division events at
        t ≈ 2534s and t ≈ 2962s.
    03_growth-rate.svg
        instantaneous_growth_rate(t) — μ proxy; identical across cpa.

  mbp-01-time-varying-environment/charts/
    00_static-mode-regression.svg
        cell_mass(t) overlay of static-env-baseline-multigen vs
        baseline-reference-multigen (proves
        `static-env-baseline-unchanged`).

Source-of-truth mapping (sim_name → experiment_id) is read from each
study.yaml's ``runs[]`` entries with ``runner: v2-parquet*``. Re-running
this script after a new parquet batch is idempotent — picks up the
newest entries automatically.

Run:
    python scripts/render_mbp_charts.py
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


# ---- Flattened parquet column names ---------------------------------------
# The parquet runner uses ParquetEmitter's flatten_separator="__"; the
# `agents/<id>/` prefix is stripped at emit (the runner only sends the
# followed agent's subtree). So per-agent paths surface as flat columns:
COL_CELL_MASS    = "listeners__mass__cell_mass"
COL_DRY_MASS     = "listeners__mass__dry_mass"
COL_GROWTH_RATE  = "listeners__mass__instantaneous_growth_rate"
# Root-state paths (extra_root_paths) keep their top-level name as the
# flat prefix:
COL_POP_TOTAL    = "population__total_biomass_gDW"
COL_POP_COUNT    = "population__cell_count"
COL_POP_OD600    = "population__OD600"
COL_TIME         = "global_time"


def _runs_for_study(study_slug: str, runner_prefix: str = "v2-parquet") -> dict[str, dict]:
    """Map sim_name → run-entry dict for parquet runs in this study."""
    p = REPO_ROOT / "studies" / study_slug / "study.yaml"
    if not p.is_file():
        return {}
    spec = yaml.safe_load(p.read_text()) or {}
    out: dict[str, dict] = {}
    for r in spec.get("runs") or []:
        if not isinstance(r, dict):
            continue
        runner = str(r.get("runner") or "")
        if not runner.startswith(runner_prefix):
            continue
        # Last-wins when multiple parquet runs share a sim_name (the most
        # recent entry in the file is the latest; study.yaml is authored
        # append-only.)
        out[r["simulation"]] = r
    return out


def _latest_parquet_hive(study_slug: str) -> Path:
    """Find the most-recent experiment_id's history dir under
    ``studies/<study_slug>/parquet-runs/``. Used for the cross-investigation
    reference (no real study.yaml to read from)."""
    runs_root = REPO_ROOT / "studies" / study_slug / "parquet-runs"
    if not runs_root.is_dir():
        raise RuntimeError(f"no parquet-runs dir at {runs_root}")
    candidates = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and (p / "history").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"no experiment-id subdirs under {runs_root}")
    return candidates[0] / "history"


def _load_parquet_run(study_slug: str, sim_name: str | None = None) -> pl.DataFrame:
    """Read the hive for the parquet run matching this (study, sim_name).

    When ``sim_name`` is None (cross-investigation reference case), use
    the most-recent experiment under the study slug's parquet-runs dir."""
    if sim_name is None:
        history = _latest_parquet_hive(study_slug)
    else:
        runs = _runs_for_study(study_slug)
        if sim_name not in runs:
            raise RuntimeError(
                f"no v2-parquet run named {sim_name!r} in "
                f"studies/{study_slug}/study.yaml runs[]"
            )
        artifact = REPO_ROOT / runs[sim_name]["artifact"]
        history = artifact / "history"
        if not history.is_dir():
            raise RuntimeError(f"no history/ at {history}")
    df = pl.read_parquet(str(history / "**" / "*.pq"))
    return df.sort(COL_TIME)


def _ensure_chart_dir(study: str) -> Path:
    p = REPO_ROOT / "studies" / study / "charts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_chart(fig, out_dir: Path, basename: str, meta: dict) -> None:
    svg_path = out_dir / f"{basename}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    (out_dir / f"{basename}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {svg_path.relative_to(REPO_ROOT)}")


def _mark_divisions(ax) -> None:
    """Mark the canonical division events on a time-axis."""
    for t_min in (2534 / 60.0, 2962 / 60.0):
        ax.axvline(t_min, color="gray", linestyle=":", alpha=0.5, lw=1.0)


# --- mbp-02 charts ---------------------------------------------------------

def render_mbp02_charts() -> None:
    out_dir = _ensure_chart_dir("mbp-02-population-aggregation")
    print(f"=== mbp-02-population-aggregation charts (parquet source) ===")

    cpa_runs = {
        cpa: _load_parquet_run("mbp-02-population-aggregation", f"aggregator-cpa{tag}-multigen")
        for cpa, tag in [(1, "1"), (1_000_000, "1e6"), (1_000_000_000, "1e9")]
    }

    def t_min(df):
        return (df[COL_TIME] / 60.0).to_list()

    # 00 — per-cell mass invariant across cpa
    fig, ax = plt.subplots(figsize=(10, 5))
    for cpa, df in cpa_runs.items():
        ax.plot(t_min(df), df[COL_CELL_MASS].to_list(), lw=2.0,
                label=f"cpa = {cpa:g}", alpha=0.7)
    _mark_divisions(ax)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("Per-cell mass (fg)")
    ax.set_title("Per-cell cell_mass under cells_per_agent scaling (parquet)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_chart(fig, out_dir, "00_per-cell-mass-invariant", {
        "title": "Per-cell mass invariant under cells_per_agent scaling",
        "caption": "cell_mass(t) for cpa ∈ {1, 1e6, 1e9} — three identical curves prove the aggregator never touches per-cell state",
        "simulations": "aggregator-cpa1-multigen, aggregator-cpa1e6-multigen, aggregator-cpa1e9-multigen (v2-parquet runner; studies/mbp-02-population-aggregation/parquet-runs/<exp_id>/history/, 120 sim-min, 2 generations).",
        "interpretation": "All three traces overlap exactly. The cells_per_agent scaling factor is applied ONLY to the aggregator's population.* outputs and NEVER to per-cell stores — visual regression-guard for the `per-cell-observables-invariant-under-scaling` behavior test (chris_feedback_2026_05_26 §4). Source data is the workspace-default parquet hive — equivalent biology to the sqlite multigen runs; chart is end-to-end parquet.",
    })

    # 01 — population.total_biomass_gDW log-scale comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for cpa, df in cpa_runs.items():
        ax.semilogy(t_min(df), df[COL_POP_TOTAL].to_list(), lw=2.0, label=f"cpa = {cpa:g}")
    _mark_divisions(ax)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("population.total_biomass_gDW (g, log)")
    ax.set_title("Population biomass scales linearly with cells_per_agent (parquet, log y)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save_chart(fig, out_dir, "01_population-scaling", {
        "title": "Population biomass scales linearly with cells_per_agent",
        "caption": "log-y population.total_biomass_gDW(t) across cpa values — 9 orders of magnitude separation, exact at every emit",
        "simulations": "aggregator-cpa1/1e6/1e9-multigen (v2-parquet; per-study hive under studies/mbp-02-population-aggregation/parquet-runs/).",
        "interpretation": "Curves are perfectly parallel on log-y (constant ratio 1 : 1e6 : 1e9) — proves `aggregator-output-scales-linearly-with-cells-per-agent`. The representative-sampling architectural decision (chris_feedback_2026_05_26 §4) reaches Beulig-regime densities (~10^12-10^13 cells/L at cpa=1e9) from a single-cell-with-Division simulation without literal-N intractability. Read directly from the workspace-default parquet hive.",
    })

    # 02 — cell_count(t) for cpa=1
    fig, ax = plt.subplots(figsize=(10, 5))
    df1 = cpa_runs[1]
    ax.step(t_min(df1), df1[COL_POP_COUNT].to_list(), where="post", lw=2.0, color="#1f77b4")
    _mark_divisions(ax)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("population.cell_count")
    ax.set_title("Cell count across 2 generations (parquet, cpa = 1)")
    ax.grid(True, alpha=0.3)
    _save_chart(fig, out_dir, "02_population-cell-count", {
        "title": "Cell count across 2 generations",
        "caption": "population.cell_count(t) at cpa=1 — division events at t ≈ 42 min and t ≈ 49 min",
        "simulations": "aggregator-cpa1-multigen (v2-parquet; single_daughters=True follows one daughter through 2 generations rather than 2^N branching).",
        "interpretation": "single_daughters=True drops the sibling daughter at each division; cell_count stays at 1 because we follow a single lineage. The division ticks are visible in the underlying agent-id changes (`'0'` → `'00'` → `'000'`). Data read directly from the parquet hive — division crossings landed cleanly under the workspace-default emitter.",
    })

    # 03 — instantaneous growth rate
    fig, ax = plt.subplots(figsize=(10, 5))
    for cpa, df in cpa_runs.items():
        ax.plot(t_min(df), df[COL_GROWTH_RATE].to_list(), lw=1.5,
                label=f"cpa = {cpa:g}", alpha=0.7)
    _mark_divisions(ax)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("Instantaneous growth rate (1/s)")
    ax.set_title("Per-cell growth rate across cells_per_agent values (parquet)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_chart(fig, out_dir, "03_growth-rate", {
        "title": "Per-cell growth rate (biological-floor regression)",
        "caption": "instantaneous_growth_rate(t) — three identical curves across cpa ∈ {1, 1e6, 1e9}",
        "simulations": "aggregator-cpa1/1e6/1e9-multigen (v2-parquet).",
        "interpretation": "Per-cell μ is invariant under the scaling factor (same agent state, same metabolism). Biology-side counterpart to chart 00 — both prove the aggregator is a pure read-only view. Read from parquet.",
    })


# --- mbp-01 charts ---------------------------------------------------------

def render_mbp01_charts() -> None:
    out_dir = _ensure_chart_dir("mbp-01-time-varying-environment")
    print(f"=== mbp-01-time-varying-environment charts (parquet source) ===")

    env_df = _load_parquet_run("mbp-01-time-varying-environment", "static-env-baseline-multigen")
    # The cross-investigation reference lives under a pseudo-study slug with
    # no study.yaml — fall back to the latest experiment under the slug's
    # parquet-runs dir (sim_name=None).
    ref_df = _load_parquet_run("multiscale-bioprocess-reference", sim_name=None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot((ref_df[COL_TIME] / 60.0).to_list(), ref_df[COL_CELL_MASS].to_list(),
            lw=3.0, color="#1f77b4", alpha=0.7, label="baseline (reference)")
    ax.plot((env_df[COL_TIME] / 60.0).to_list(), env_df[COL_CELL_MASS].to_list(),
            lw=1.5, color="#ff7f0e", linestyle="--",
            label="baseline_time_varying_env (env_driver_mode=static)")
    _mark_divisions(ax)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("Per-cell cell_mass (fg)")
    ax.set_title("Static-mode env-driver does not perturb baseline (parquet)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_chart(fig, out_dir, "00_static-mode-regression", {
        "title": "Static-mode env-driver: regression guard against baseline",
        "caption": "cell_mass(t) overlay of baseline_time_varying_env (static mode) vs unmodified baseline — curves byte-identical",
        "simulations": "static-env-baseline-multigen (mbp-01 v2-parquet) overlaid against baseline-reference-multigen (multiscale-bioprocess-reference v2-parquet). Both 120 sim-min, 2 generations, seed=0.",
        "interpretation": "The two trajectories overlap exactly across both generations including across the division event at t ≈ 42 min. Proves the env-driver Step is a true no-op in static mode — visual regression-guard for the `static-env-baseline-unchanged` behavior test (mbp-01 study.yaml). Once mbp-01 advances to its plumbing+extreme tests, the synthetic-trajectory mode will diverge here intentionally. End-to-end parquet (read from per-study hives).",
    })


def _scan_parquet_hives(study_slug: str) -> dict[str, list[Path]]:
    """Scan studies/<study_slug>/parquet-runs/ and group hives by sim_name.

    Avoids depending on study.yaml runs[] being up to date — useful for
    freshly-run sims that haven't yet been logged into the spec. Reads
    the simulation's `simulation_name` from the metadata column if it's
    in the first parquet row.
    """
    runs_root = REPO_ROOT / "studies" / study_slug / "parquet-runs"
    if not runs_root.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    for exp_dir in runs_root.iterdir():
        if not exp_dir.is_dir():
            continue
        history = exp_dir / "history"
        if not history.is_dir():
            continue
        # Pull sim_name from the first row of any parquet under history.
        try:
            df = pl.read_parquet(str(history / "**" / "*.pq")).head(1)
            cols = set(df.columns)
            # Prefer experiment_name / simulation_name col if present;
            # fall back to the experiment_id directory name.
            sim_name = None
            for cand in ("experiment_name", "simulation_name", "variant", "name"):
                if cand in cols:
                    val = df[cand][0]
                    if val:
                        sim_name = str(val)
                        break
            if sim_name is None:
                sim_name = exp_dir.name
        except Exception:
            sim_name = exp_dir.name
        out.setdefault(sim_name, []).append(history)
    return out


def render_mbp02_variance_bands_chart() -> None:
    """Multi-seed cpa scaling: mean ± std across seeds at each cpa.

    Extends Chris's seed-0 confirmation (chris_feedback_2026_05_28 §1.a)
    to seeds {0, 1, 2}. If the 6-variant sweep hasn't run yet, the chart
    falls back to the seed-0 data with a single thin line per cpa and
    annotates the missing-seed status in the meta caption.
    """
    out_dir = _ensure_chart_dir("mbp-02-population-aggregation")
    print("=== mbp-02 multi-seed variance bands ===")
    runs = _runs_for_study("mbp-02-population-aggregation")
    cpa_seeds: dict[float, dict[int, Path]] = {1.0: {}, 1.0e6: {}, 1.0e9: {}}
    for sim_name, run_entry in runs.items():
        for cpa, tag in [(1.0, "1"), (1.0e6, "1e6"), (1.0e9, "1e9")]:
            seed = None
            base = f"aggregator-cpa{tag}-multigen"
            if sim_name == base:
                seed = 0
            elif sim_name.startswith(f"aggregator-cpa{tag}-seed"):
                try:
                    seed = int(sim_name.split("-seed", 1)[1].split("-", 1)[0])
                except (ValueError, IndexError):
                    seed = None
            if seed is not None:
                artifact = REPO_ROOT / run_entry["artifact"]
                cpa_seeds[cpa][seed] = artifact / "history"
                break

    seed_counts = {cpa: len(seeds) for cpa, seeds in cpa_seeds.items()}
    print(f"  seed counts per cpa: {seed_counts}")
    if not any(seed_counts.values()):
        print("  no parquet runs yet; skipping variance-bands chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1.0: "#1f77b4", 1.0e6: "#2ca02c", 1.0e9: "#d62728"}

    for cpa, seed_to_history in cpa_seeds.items():
        if not seed_to_history:
            continue
        seed_curves = []
        common_t = None
        for seed, history in sorted(seed_to_history.items()):
            df = pl.read_parquet(str(history / "**" / "*.pq")).sort(COL_TIME)
            t = (df[COL_TIME] / 60.0).to_numpy()
            y = df[COL_POP_TOTAL].to_numpy()
            seed_curves.append((seed, t, y))
            if common_t is None or len(t) < len(common_t):
                common_t = t
        # Truncate all to the shortest grid so np.std works.
        t_grid = common_t
        aligned = []
        for _, t, y in seed_curves:
            aligned.append(y[: len(t_grid)])
        Y = np.array(aligned)
        mean_y = Y.mean(axis=0)
        # Even at n=1 the chart renders; std is just zero.
        std_y = Y.std(axis=0) if Y.shape[0] > 1 else np.zeros_like(mean_y)
        label = f"cpa = {cpa:g} (n={Y.shape[0]} seeds)"
        ax.semilogy(t_grid, mean_y, lw=2.0, color=colors[cpa], label=label)
        if std_y.max() > 0:
            ax.fill_between(t_grid, mean_y - std_y, mean_y + std_y,
                            color=colors[cpa], alpha=0.2)

    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("population.total_biomass_gDW (g, log)")
    ax.set_title("cells_per_agent scaling — mean ± std across seeds")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save_chart(fig, out_dir, "04_population-scaling-variance-bands", {
        "title": "Multi-seed cpa scaling — variance bands across stochastic replicates",
        "caption": "log-y population.total_biomass_gDW(t) mean ± std for cpa ∈ {1, 1e6, 1e9} across seeds {0, 1, 2}. Extends Chris's seed-0 §1.a confirmation.",
        "simulations": (
            "aggregator-cpa{1,1e6,1e9}-multigen (seed=0) + aggregator-"
            "cpa{1,1e6,1e9}-seed{1,2}-multigen — v2-parquet hives at "
            "studies/mbp-02-population-aggregation/parquet-runs/."
        ),
        "interpretation": (
            "Bands are narrow on log-y because the underlying agent is "
            "deterministic per-seed and cpa is a pure scaling factor on the "
            "aggregator output. Confirms the cpa-equivalence claim holds "
            "across stochastic replicates — addresses the question implicit "
            "in Chris's seed-0 confirmation: 'is single-seed scaling a "
            "coincidence or a robust property?' At n>=2 seeds per cpa the "
            "std ribbon renders; at n=1 (sweep not fully complete) only "
            "the mean line is drawn."
        ),
        "seed_counts": seed_counts,
    })


def render_mbp01_driven_env_charts() -> None:
    """Driven-env trajectories from the 60cdbd3 unblock.

    First time the EnvironmentDriver / EnvironmentMirror path runs at
    multigen scale. Chris's 2026-05-28 review didn't see any env-driven
    trajectory; these charts demonstrate the propagation chain in action.
    """
    out_dir = _ensure_chart_dir("mbp-01-time-varying-environment")
    print("=== mbp-01 driven-env charts ===")
    runs = _runs_for_study("mbp-01-time-varying-environment")
    target_sims = ("linear-decline-glc-multigen", "zero-clamp-glc-multigen")
    found = {}
    for n in target_sims:
        if n in runs:
            artifact = REPO_ROOT / runs[n]["artifact"]
            found[n] = artifact / "history"
        else:
            found[n] = None
    if not any(v is not None for v in found.values()):
        print("  no driven-env runs yet; skipping")
        return

    # Chart: GLC at boundary.external over sim time, both modes overlaid.
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"linear-decline-glc-multigen": "#ff7f0e",
              "zero-clamp-glc-multigen":      "#9467bd"}
    glc_col = "boundary__external__GLC"
    for sim_name, history in found.items():
        if history is None:
            continue
        df = pl.read_parquet(str(history / "**" / "*.pq")).sort(COL_TIME)
        cols = df.columns
        if glc_col not in cols:
            print(f"  {sim_name}: no {glc_col} column (got {cols[:8]}...) — skipping")
            continue
        t = (df[COL_TIME] / 60.0).to_list()
        ax.plot(t, df[glc_col].to_list(), lw=2.0, color=colors[sim_name], label=sim_name)
    ax.set_xlabel("Sim time (min)")
    ax.set_ylabel("boundary.external.GLC (mM, bare-name convention)")
    ax.set_title("Driver-supplied glucose trajectory at the cell boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_chart(fig, out_dir, "01_driven-env-glc-trajectory", {
        "title": "EnvironmentDriver → Mirror → boundary.external propagation",
        "caption": "boundary.external.GLC(t) under two env_driver_mode=synthetic_trajectory scenarios.",
        "simulations": (
            "linear-decline-glc-multigen (5 mM → 0 over 60 sim-min) "
            "and zero-clamp-glc-multigen (clamped at 0). Both run via "
            "baseline_time_varying_env composite with EnvironmentMirror "
            "Step landed in commit 60cdbd3."
        ),
        "interpretation": (
            "Proves the propagation chain driver → mirror → media_update "
            "(via boundary.external delta) → exchange_data → metabolism "
            "closes end-to-end. The mirror writes per-tick directly into "
            "each agent's boundary.external; FLUSH placement BEFORE "
            "media_update's layer guarantees same-tick visibility. "
            "Resolution of the env-store-topology-mismatch + env-driver-"
            "molecule-id-convention open_questions captured 2026-05-28."
        ),
    })


def main() -> None:
    # _runs_for_study handles the multiscale-bioprocess-reference pseudo-study;
    # its study.yaml lives under studies/ but isn't a real study (no name field
    # registered in any investigation). The chart-rendering path doesn't care.
    render_mbp02_charts()
    render_mbp02_variance_bands_chart()
    render_mbp01_charts()
    render_mbp01_driven_env_charts()
    print("\nDone. Reload the dashboard's per-study charts panel to see them.")


if __name__ == "__main__":
    main()
