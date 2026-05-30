"""Render the dnaa-0 / dnaa-1 visualization HTML from a parquet run.

Bypasses the generic parquet_viz inputs_map resolver because the dnaa
viz needs an INDEX into listeners.monomer_counts (the dnaA monomer at
index 3861) and the mRNA / init counts also live inside vector
listeners. This script does the column extraction inline and feeds the
viz directly.

Usage::

    # dnaa-0 (3-panel: oriC, cell_mass, DnaA monomer):
    python scripts/render_dnaa_runs.py dnaa-0-parameter-foundation \\
        studies/dnaa-0-parameter-foundation/parquet-runs/dnaa0-6gen-2026-05-29

    # dnaa-1 (4-panel: DnaA monomer, concentration, mRNA, init events):
    python scripts/render_dnaa_runs.py dnaa-1-expression-dynamics \\
        studies/dnaa-1-expression-dynamics/parquet-runs/<run_id>
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import polars as pl
import yaml
import dill


# dnaA monomer index in the monomer_counts listener vector. Stable
# across ParCa fixtures (monomer_ids[3861] == 'PD03831[c]' verified on
# out/cache-succinate and shipped main fixture).
DNAA_MONOMER_INDEX = 3861


def _load_history(run_dir: Path) -> tuple[pl.DataFrame, list[float], int]:
    """Load the run's history with generations laid out SEQUENTIALLY in time.

    ``global_time`` resets to 0 at the start of each generation (every gen is
    a freshly-built composite resuming from the prior daughter), so a naive
    concat-and-sort would stack all generations on the same 0–τ range and the
    lineage would look like a single overlapping band. Instead, offset each
    generation's ``global_time`` by the cumulative duration of the prior
    generations, producing a monotonic lineage clock.

    Matches history parquet at ANY depth — run_condition_multigen_parquet
    writes ``<out_dir>/<experiment_id>/history/...``, so when ``run_dir`` is
    the out_dir the data is one level deeper than ``<run_dir>/history``.

    Returns ``(df, boundaries, n_gens)``: ``df`` carries the cumulative
    ``global_time``; ``boundaries`` are the cumulative generation-end times
    (s), one per generation except the last — the viz draws the vertical
    separator lines + generation-number labels from them (Rashmi's 2026-05-30
    chart feedback).
    """
    files = [f for f in glob.glob(str(run_dir / "**" / "*.pq"), recursive=True)
             if f"{os.sep}history{os.sep}" in f]
    if not files:
        raise FileNotFoundError(f"no history pq files under {run_dir}")
    by_gen: dict[int, list[str]] = defaultdict(list)
    for f in files:
        m = re.search(r"generation=(\d+)", f)
        if m:
            by_gen[int(m.group(1))].append(f)
    gens = sorted(by_gen)
    parts: list[pl.DataFrame] = []
    boundaries: list[float] = []
    offset = 0.0
    for i, g in enumerate(gens):
        gfiles = sorted(by_gen[g],
                        key=lambda p: int(p.rsplit("/", 1)[1].split(".")[0]))
        d = pl.concat([pl.read_parquet(f) for f in gfiles],
                      how="diagonal_relaxed")
        gmax = 0.0
        if "global_time" in d.columns and d.height:
            d = d.sort("global_time")
            gmax = float(d["global_time"].max())
            d = d.with_columns(
                (pl.col("global_time") + offset).alias("global_time"))
        parts.append(d)
        offset += gmax
        if i < len(gens) - 1:
            boundaries.append(offset)  # cumulative end of this gen = boundary
    df = pl.concat(parts, how="diagonal_relaxed")
    return df, boundaries, len(gens)


def _resolve_dnaa_indices(cache_dir: str) -> dict:
    """Resolve dnaA-related listener indices from the cache."""
    with open(Path(cache_dir) / "sim_data_cache.dill", "rb") as f:
        cache = dill.load(f)
    mcl = cache["configs"]["monomer_counts_listener"]
    monomer_ids = list(mcl["monomer_ids"])
    if monomer_ids[DNAA_MONOMER_INDEX] != "PD03831[c]":
        # Fallback search.
        idx = monomer_ids.index("PD03831[c]")
    else:
        idx = DNAA_MONOMER_INDEX
    out = {"dnaa_monomer_idx": idx}
    # mRNA index — TU index for dnaA in transcription
    tic = cache["configs"]["ecoli-transcript-initiation"]
    rna_data = tic.get("rna_data")
    if rna_data is not None:
        try:
            tu_ids = list(rna_data["id"])
            for i, tid in enumerate(tu_ids):
                if "TU00259" in str(tid):
                    out["dnaa_mrna_tu_idx"] = i
                    break
        except Exception:
            pass
    # cistron index for rnap_data.rna_init_event_per_cistron — the index
    # space is the rnap_data_listener's `cistron_ids` list (NOT a structured
    # cistron_data table, which is None in the succinate caches). dnaA's
    # cistron id is "EG10235_RNA" (substring match on the gene id EG10235).
    try:
        rdl = cache["configs"].get("rnap_data_listener", {})
        cistron_ids = rdl.get("cistron_ids")
        if cistron_ids is not None:
            for i, cid in enumerate(cistron_ids):
                if "EG10235" in str(cid):  # dnaA cistron (EG10235_RNA)
                    out["dnaa_cistron_idx"] = i
                    break
        # legacy fallback: structured cistron_data with a gene_id field
        if "dnaa_cistron_idx" not in out:
            cistron_data = cache["configs"]["ecoli-transcript-initiation"].get("cistron_data")
            if cistron_data is not None:
                for i, gid in enumerate(cistron_data["gene_id"]):
                    if str(gid) == "EG10235":
                        out["dnaa_cistron_idx"] = i
                        break
    except Exception:
        pass
    return out


def _series_at_idx(df: pl.DataFrame, col: str, idx: int | None) -> list[float] | None:
    """Pull values at `idx` from a column that is a list-of-arrays."""
    if col not in df.columns or idx is None:
        return None
    vals = []
    for v in df[col].to_list():
        try:
            if v is None or len(v) <= idx:
                vals.append(None)
            else:
                vals.append(float(v[idx]) if v[idx] is not None else None)
        except Exception:
            vals.append(None)
    return vals


def render_dnaa0(run_dir: Path, cache_dir: str, out_path: Path,
                 title: str | None = None) -> int:
    df, div_times, n_gens = _load_history(run_dir)
    idx = _resolve_dnaa_indices(cache_dir)

    times = df["global_time"].to_list()
    oric = df["listeners__replication_data__number_of_oric"].to_list() \
        if "listeners__replication_data__number_of_oric" in df.columns else []
    mass = df["listeners__mass__cell_mass"].to_list() \
        if "listeners__mass__cell_mass" in df.columns else []
    dnaa = _series_at_idx(df, "listeners__monomer_counts",
                          idx.get("dnaa_monomer_idx")) or []

    from v2ecoli.visualizations.dnaa_succinate import DnaaSteadyStateVisualization
    from v2ecoli.core import build_core

    core = build_core()
    viz = DnaaSteadyStateVisualization(
        config={"title": title or "dnaa-0 — succinate steady state",
                "dnaa_band_low": 300.0, "dnaa_band_high": 800.0},
        core=core,
    )
    state = {
        "oric_count":         oric,
        "cell_mass":          mass,
        "dnaa_monomer_total": dnaa,
        "time":               times,
        "division_times":     div_times,
        "_run_labels":        [f"dnaa-0 succinate {n_gens}-gen"],
    }
    result = viz.update(state)
    html = result.get("html", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  ok dnaa-0: {out_path} ({len(html)} chars)")
    print(f"    n_ticks: {len(times)}, oriC unique: {sorted(set(o for o in oric if o is not None))[:10]}, "
          f"DnaA monomer range: [{min(v for v in dnaa if v is not None):.0f}, "
          f"{max(v for v in dnaa if v is not None):.0f}]")
    return 0


def render_dnaa1(run_dir: Path, cache_dir: str, out_path: Path,
                 title: str | None = None) -> int:
    df, div_times, n_gens = _load_history(run_dir)
    idx = _resolve_dnaa_indices(cache_dir)

    times = df["global_time"].to_list()
    mass = df["listeners__mass__cell_mass"].to_list() \
        if "listeners__mass__cell_mass" in df.columns else []
    dnaa = _series_at_idx(df, "listeners__monomer_counts",
                          idx.get("dnaa_monomer_idx")) or []
    mrna = _series_at_idx(df, "listeners__rna_counts__mRNA_counts",
                          idx.get("dnaa_mrna_tu_idx")) or []
    init_ev = _series_at_idx(df, "listeners__rnap_data__rna_init_event_per_cistron",
                             idx.get("dnaa_cistron_idx")) or []

    from v2ecoli.visualizations.dnaa_succinate import DnaaExpressionVisualization
    from v2ecoli.core import build_core

    core = build_core()
    viz = DnaaExpressionVisualization(
        config={"title": title or "dnaa-1 — Mechanism A V=2e-3 (succinate)",
                "dnaa_band_low": 300.0, "dnaa_band_high": 800.0},
        core=core,
    )
    state = {
        "dnaa_monomer_total": dnaa,
        "cell_mass":          mass,
        "dnaa_mrna_count":    mrna,
        "dnaa_init_events":   init_ev,
        "time":               times,
        "division_times":     div_times,
        "_run_labels":        [f"dnaa-1 Mechanism A {n_gens}-gen"],
    }
    result = viz.update(state)
    html = result.get("html", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  ok dnaa-1: {out_path} ({len(html)} chars)")
    print(f"    n_ticks: {len(times)}, indices: {idx}")
    if dnaa:
        nonnull = [v for v in dnaa if v is not None]
        if nonnull:
            print(f"    DnaA monomer range: [{min(nonnull):.0f}, {max(nonnull):.0f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study_slug")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cache-dir", default=None,
                    help="ParCa cache dir (defaults per study)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.study_slug == "dnaa-0-parameter-foundation":
        cache = args.cache_dir or "out/cache-succinate"
        out = args.out or Path(f"studies/{args.study_slug}/viz/dnaa_steady_state.html")
        return render_dnaa0(args.run_dir, cache, out, title=args.title)
    if args.study_slug == "dnaa-1-expression-dynamics":
        cache = args.cache_dir or "out/cache-succinate-mechA-2e-3"
        out = args.out or Path(f"studies/{args.study_slug}/viz/dnaa_expression.html")
        return render_dnaa1(args.run_dir, cache, out, title=args.title)
    print(f"ERROR: unknown study {args.study_slug!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
