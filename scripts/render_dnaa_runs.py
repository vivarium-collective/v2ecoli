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


def _subsample(df: pl.DataFrame, target: int = 2500) -> pl.DataFrame:
    """Take ~``target`` evenly-spaced rows so the embedded viz HTML stays
    small enough for the dashboard to render inline. A full multi-gen run is
    ~20–40k ticks; plotting every one bloats the HTML to multiple MB (which
    the SPA can't embed). Stride-sampling preserves the cycle shape."""
    n = df.height
    if n <= target:
        return df
    stride = max(1, n // target)
    return df.gather_every(stride)


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
    df = _subsample(df)

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

    # Compute the init-rate caption from the FULL (un-subsampled) series so the
    # "N events / mean rate" number stays exact even though the plotted
    # barcode is downsampled for embed size.
    _full_init = _series_at_idx(df, "listeners__rnap_data__rna_init_event_per_cistron",
                                idx.get("dnaa_cistron_idx")) or []
    _total_ev = sum(v for v in _full_init if v is not None)
    _dur_min = (float(df["global_time"].max()) / 60.0) \
        if ("global_time" in df.columns and df.height) else 0.0
    _rate = (_total_ev / _dur_min) if _dur_min > 0 else 0.0
    init_caption = (f"{int(_total_ev)} events over {int(_dur_min)} min; "
                    f"mean {_rate:.2f}/min")

    df = _subsample(df)
    times = df["global_time"].to_list()
    oric = df["listeners__replication_data__number_of_oric"].to_list() \
        if "listeners__replication_data__number_of_oric" in df.columns else []
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
        config={"title": title or "dnaa-1 — V=2e-3 — 7-gen lineage on succinate",
                "dnaa_band_low": 300.0, "dnaa_band_high": 800.0,
                "target_init_rate_per_min": 1.0,
                "init_caption": init_caption,
                "subtitle": "Mechanism A — runtime overwrite of dnaA's "
                            "per-promoter init_prob: "
                            "sim_data.genetic_perturbations[\"TU00259[c]\"] = 2e-3 "
                            "(baseline unchanged: dnaA TE 0.35, mRNA t½ 1.9 min, "
                            "DnaA protein t½ 280 min, C 40 min, D 20 min)"},
        core=core,
    )
    state = {
        "oric_count":         oric,
        "cell_mass":          mass,
        "dnaa_monomer_total": dnaa,
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


def render_chromosome(run_dir: Path, cache_dir: str, out_path: Path,
                      title: str | None = None) -> int:
    """Render the chromosome-state viz (cycle counts, fork map, DnaA-box
    occupancy) from a run's parquet hive."""
    df, div_times, n_gens = _load_history(run_dir)
    df = _subsample(df)
    times = df["global_time"].to_list()

    def _col(name):
        return df[name].to_list() if name in df.columns else []
    oric = _col("listeners__replication_data__number_of_oric")
    chrom = _col("listeners__unique_molecule_counts__full_chromosome")
    repl = _col("listeners__unique_molecule_counts__active_replisome")
    free_boxes = _col("listeners__replication_data__free_DnaA_boxes")
    total_boxes = _col("listeners__replication_data__total_DnaA_boxes")

    # Flatten fork_coordinates (array-per-tick) into parallel (time, position).
    fork_times: list[float] = []
    fork_positions: list[float] = []
    fc = _col("listeners__replication_data__fork_coordinates")
    for t, coords in zip(times, fc):
        if coords is None:
            continue
        for c in coords:
            if c is not None:
                fork_times.append(float(t))
                fork_positions.append(float(c))

    from v2ecoli.visualizations.dnaa_succinate import DnaaChromosomeVisualization
    from v2ecoli.core import build_core
    core = build_core()
    viz = DnaaChromosomeVisualization(
        config={"title": title or "chromosome state (succinate)"}, core=core)
    state = {
        "oric_count":       oric,
        "chromosome_count": chrom,
        "replisome_count":  repl,
        "fork_times":       fork_times,
        "fork_positions":   fork_positions,
        "free_dnaa_boxes":  free_boxes,
        "total_dnaa_boxes": total_boxes,
        "time":             times,
        "division_times":   div_times,
        "_run_labels":      [f"{n_gens}-gen lineage"],
    }
    html = viz.update(state).get("html", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  ok chromosome: {out_path} ({len(html)} chars)")
    print(f"    n_ticks: {len(times)}, forks plotted: {len(fork_times)}, "
          f"chromosomes: {sorted(set(c for c in chrom if c is not None))[:6]}, "
          f"replisomes: {sorted(set(r for r in repl if r is not None))[:6]}")
    return 0


def render_chromosome_map(run_dir: Path, out_path: Path, gen: int = 3,
                          n_snaps: int = 5, title: str | None = None) -> int:
    """Render the circular chromosome-map viz — a row of snapshots of one
    generation's chromosome showing oriC, Ter, replication-fork replisomes,
    and RNAP positions around the 4.6 Mb circle (the v1 'chromosome timeline'
    diagram). Reads ONE generation's partition directly (raw per-gen
    global_time) and samples ``n_snaps`` evenly-spaced timepoints."""
    files = [f for f in glob.glob(str(run_dir / "**" / "*.pq"), recursive=True)
             if f"{os.sep}history{os.sep}" in f and f"generation={gen}/" in f]
    if not files:  # fall back to the first available generation
        allf = [f for f in glob.glob(str(run_dir / "**" / "*.pq"), recursive=True)
                if f"{os.sep}history{os.sep}" in f]
        gens = sorted(set(int(re.search(r"generation=(\d+)", f).group(1)) for f in allf))
        gen = gens[len(gens)//2] if gens else gen
        files = [f for f in allf if f"generation={gen}/" in f]
    if not files:
        raise FileNotFoundError(f"no history pq under {run_dir}")
    d = pl.concat([pl.read_parquet(f) for f in files],
                  how="diagonal_relaxed").sort("global_time")
    n = d.height
    idxs = [min(n - 1, int(round((i + 0.5) / n_snaps * n))) for i in range(n_snaps)]

    def _arr(col, i):
        if col not in d.columns:
            return []
        v = d[col].to_list()[i]
        return [float(x) for x in v if x is not None] if v is not None else []

    gt = d["global_time"].to_list()
    chrom_col = "listeners__unique_molecule_counts__full_chromosome"
    snaps = []
    half = 2.32e6
    for i in idxs:
        forks = _arr("listeners__replication_data__fork_coordinates", i)
        rnaps = _arr("listeners__rnap_data__active_rnap_coordinates", i)
        if len(rnaps) > 150:  # thin RNAP for a light SVG
            stride = len(rnaps) // 150
            rnaps = rnaps[::stride]
        n_chr = d[chrom_col].to_list()[i] if chrom_col in d.columns else 1
        snaps.append({"t_min": gt[i] / 60.0, "n_chr": int(n_chr or 1),
                      "forks": forks, "rnaps": rnaps})
        half = max(half, max((abs(x) for x in forks + rnaps), default=half))

    from v2ecoli.visualizations.dnaa_succinate import DnaaChromosomeMapVisualization
    from v2ecoli.core import build_core
    core = build_core()
    viz = DnaaChromosomeMapVisualization(
        config={"title": title or f"chromosome map — generation {gen}"}, core=core)
    html = viz.update({
        "snapshots": snaps, "half_genome": half,
        "_caption": (f"oriC (green), Ter (red), replisome forks (orange ▲), and "
                     f"RNAP positions (blue ·) along the {2*half/1e6:.1f} Mb "
                     f"circular chromosome — {n_snaps} snapshots through "
                     f"generation {gen}."),
    }).get("html", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  ok chromosome-map: {out_path} ({len(html)} chars)")
    print(f"    gen {gen}, {n_snaps} snaps at min "
          f"{[round(s['t_min'],1) for s in snaps]}, "
          f"forks/snap {[len(s['forks']) for s in snaps]}")
    return 0


def render_dnaa_forms(dills_dir: Path, out_path: Path,
                      title: str | None = None) -> int:
    """Render the dnaa-2 DnaA-form baseline (apo / DnaA-ATP / DnaA-ADP +
    ATP[c]/ADP[c]) per generation, read from a lineage's gen{N}.dill cell
    states. No re-run needed — the nucleotide-binding equilibria already
    populate these bulk species under V-only tuning."""
    import numpy as np
    BULK = {"apo": "PD03831[c]", "atp": "MONOMER0-160[c]", "adp": "MONOMER0-4565[c]",
            "atp_pool": "ATP[c]", "adp_pool": "ADP[c]"}
    files = sorted(glob.glob(str(dills_dir / "gen*.dill")),
                   key=lambda p: int(re.search(r"gen(\d+)", p).group(1)))
    if not files:
        raise FileNotFoundError(f"no gen*.dill under {dills_dir}")
    series = {k: [] for k in BULK}
    gens = []
    for f in files:
        g = int(re.search(r"gen(\d+)", f).group(1))
        st = dill.load(open(f, "rb"))
        bulk = st.get("bulk")
        if bulk is None or not hasattr(bulk, "dtype"):
            continue
        ids = np.array([str(x) for x in bulk["id"]]); cnt = bulk["count"]
        gens.append(g)
        for k, name in BULK.items():
            w = np.where(ids == name)[0]
            series[k].append(int(cnt[w[0]]) if len(w) else 0)

    from v2ecoli.visualizations.dnaa_succinate import DnaaFormsVisualization
    from v2ecoli.core import build_core
    core = build_core()
    viz = DnaaFormsVisualization(
        config={"title": title or "dnaa-2 Step 1 — DnaA-form baseline (V-only)",
                "atp_band_low": 0.2, "atp_band_high": 0.5,
                "dnaa_band_low": 300.0, "dnaa_band_high": 800.0}, core=core)
    html = viz.update({
        "gens": gens, "apo": series["apo"], "atp": series["atp"],
        "adp": series["adp"], "atp_pool": series["atp_pool"],
        "adp_pool": series["adp_pool"],
        "_caption": ("Per-generation birth-state DnaA forms from the V-tuned "
                     "succinate lineage (no mechanism change). The pool is "
                     "~99.8% DnaA-ATP — far above the Boesen [0.2,0.5] band: "
                     "the flagged K_d / [ATP]:[ADP] / expected-fraction "
                     "inconsistency."),
    }).get("html", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    fr = [series["atp"][i] / max(1, series["apo"][i] + series["atp"][i] + series["adp"][i])
          for i in range(len(gens))]
    print(f"  ok dnaa-forms: {out_path} ({len(html)} chars)")
    print(f"    gens {gens}, DnaA-ATP fraction {[round(x,3) for x in fr]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study_slug")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cache-dir", default=None,
                    help="ParCa cache dir (defaults per study)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--viz", default=None,
                    help="Override which viz: 'chromosome' (state panels) or "
                         "'chromosome-map' (circular snapshots).")
    ap.add_argument("--gen", type=int, default=3,
                    help="Generation to snapshot for --viz chromosome-map.")
    args = ap.parse_args()

    default_cache = ("out/cache-succinate-mechA-2e-3"
                     if "expression" in args.study_slug
                     else "out/cache-succinate")
    cache = args.cache_dir or default_cache

    if args.viz == "dnaa-forms":
        # run_dir is reused as the gen_dills dir for this viz.
        out = args.out or Path(f"studies/{args.study_slug}/viz/dnaa_forms_baseline.html")
        return render_dnaa_forms(args.run_dir, out, title=args.title)
    if args.viz == "chromosome-map":
        out = args.out or Path(f"studies/{args.study_slug}/viz/chromosome_map.html")
        return render_chromosome_map(args.run_dir, out, gen=args.gen, title=args.title)
    if args.viz == "chromosome":
        out = args.out or Path(f"studies/{args.study_slug}/viz/chromosome_state.html")
        return render_chromosome(args.run_dir, cache, out, title=args.title)
    if args.study_slug == "dnaa-0-parameter-foundation":
        out = args.out or Path(f"studies/{args.study_slug}/viz/dnaa_steady_state.html")
        return render_dnaa0(args.run_dir, cache, out, title=args.title)
    if args.study_slug == "dnaa-1-expression-dynamics":
        out = args.out or Path(f"studies/{args.study_slug}/viz/dnaa_expression.html")
        return render_dnaa1(args.run_dir, cache, out, title=args.title)
    print(f"ERROR: unknown study {args.study_slug!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
