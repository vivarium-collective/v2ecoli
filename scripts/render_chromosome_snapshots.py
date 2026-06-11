"""Static 3-panel chromosome-state SNAPSHOT figure from a parquet run.

Renders three representative timepoints of one cell's lineage with the existing
v2ecoli chromosome renderer (``_plot_chromosome_map`` -> ``_draw_chromosome`` +
``_draw_replication_bubbles``): the chromosome rim, oriC (green), Ter (red),
replisomes/forks (gold triangles), replication bubbles (green daughter-strand
arcs, multifork-nested), and RNAPs (blue dots).

It reads the per-timestep unique-molecule coordinate columns the emitter
persists when a run is launched with ``V2ECOLI_EMIT_UNIQUE=1``
(``active_RNAP__coordinates`` / ``active_replisome__coordinates`` /
``full_chromosome__unique_index`` etc. -- see scripts/render_chromosome_gif.py).

Three representative timepoints are selected by replication STATE (a no-fork
frame, a single-bubble frame, a multifork frame) and then SORTED BY TIME so the
panels read left->right in ascending time. For the seed-0 showcase lineage the
three frames land at, in chronological order:
  (i)   mid-replication        : t~2 min  -- 1 chromosome, 2 replisomes, a
                                 single replication bubble (green arc from oriC).
  (ii)  post-replication       : t~23 min -- 2 segregated chromosomes, NO forks
                                 (RNAPs on the rim + oriC).
  (iii) multifork re-initiation: t~88 min -- both daughters re-replicating,
                                 4 replisomes + nested bubbles, the most RNAPs.

``chromosome_domain__child_domains`` (the parent->child domain tree) is now
emitted under ``V2ECOLI_EMIT_UNIQUE`` as a flattened ``list<int>`` column
(aligned to ``chromosome_domain__domain_index``), so daughter-strand RNAPs are
placed ON the replication bubbles. Older runs that lack the column fall back to
a ``--dill``-recovered tree, else all-RNAPs-on-rim; bubbles/replisomes are
unaffected either way.

Usage:
  V2ECOLI_EMIT_UNIQUE is NOT needed here (read-only). Point at the run dir:
  .venv/bin/python scripts/render_chromosome_snapshots.py \
      --run .pbg/runs/showcase2-baseline-chromo --seed 0 \
      --out workspace/studies/.../charts/chromosome_snapshots.png
"""
from __future__ import annotations
import argparse, glob, os, sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v2ecoli.visualizations.workflow import _plot_chromosome_map


def _domain_children_from_dill(dill_path):
    if not dill_path or not os.path.exists(dill_path):
        return {}
    import dill
    st = dill.load(open(dill_path, "rb"))
    dom = (st.get("unique") or {}).get("chromosome_domain")
    out = {}
    if dom is not None and hasattr(dom, "dtype") and "_entryState" in dom.dtype.names:
        act = dom[dom["_entryState"].view(np.bool_)]
        if {"domain_index", "child_domains"}.issubset(set(act.dtype.names)):
            for e in act:
                out[int(e["domain_index"])] = [int(k) for k in e["child_domains"] if int(k) >= 0]
    return out


def _domain_children_from_row(row: dict) -> dict:
    """Build the parent->child domain tree from the emitted unique columns.

    ``chromosome_domain__child_domains`` is a flat list<int> of length
    ``2*n_domain`` (row-major (n_domain, 2)), aligned to
    ``chromosome_domain__domain_index``. Negative entries are the no-child
    placeholder. Returns ``{parent: [child, child]}`` for domains that have
    real children. Returns ``{}`` if the columns are absent (old runs).
    """
    di = row.get("chromosome_domain__domain_index")
    cd = row.get("chromosome_domain__child_domains")
    if di is None or cd is None:
        return {}
    di = [int(x) for x in di]
    cd = [int(x) for x in cd]
    if not di or len(cd) != 2 * len(di):
        return {}
    out = {}
    for i, parent in enumerate(di):
        kids = [cd[2 * i], cd[2 * i + 1]]
        kids = [k for k in kids if k >= 0]
        if kids:
            out[parent] = kids
    return out


def _snapshot_from_row(row: dict, domain_children: dict) -> dict:
    def _lst(key):
        v = row.get(key)
        return list(v) if v is not None else []
    fc = _lst("full_chromosome__unique_index")
    rep_c = _lst("active_replisome__coordinates")
    rep_d = _lst("active_replisome__domain_index")
    rnap_c = _lst("active_RNAP__coordinates")
    rnap_d = _lst("active_RNAP__domain_index")
    n_dom = len({int(d) for d in rep_d} | {int(d) for d in rnap_d}) or 1
    # Prefer the per-row emitted domain tree (V2ECOLI_EMIT_UNIQUE now emits
    # chromosome_domain__child_domains); fall back to a dill-derived tree.
    row_tree = _domain_children_from_row(row)
    if row_tree:
        domain_children = row_tree
    return {
        "time": float(row.get("global_time", 0.0)),
        "n_chromosomes": max(1, len(fc)),
        "n_domains": n_dom,
        "fork_coords": [int(c) for c in rep_c],
        "fork_domains": [int(d) for d in rep_d],
        "rnap_coords": [int(c) for c in rnap_c],
        "rnap_domains": [int(d) for d in rnap_d],
        "domain_children": domain_children,
        "n_rnap": len(rnap_c),
    }


def _phase(snap):
    nch = snap["n_chromosomes"]
    nf = len(snap["fork_coords"])
    if nf == 0:
        return "pre-initiation" if nch == 1 else "post-replication (segregated)"
    if nf <= 2:
        return "single replication bubble"
    return "multifork replication"


def _pick_rows(df, domain_children):
    """Choose three representative rows walking one replication round.

    (i)  a no-fork frame (RNAPs + oriC, no bubble),
    (ii) a single-bubble frame (exactly 2 replisomes),
    (iii)a multifork frame (>=4 replisomes, most RNAPs).
    Falls back gracefully if a category is absent.
    """
    rows = [df.row(i, named=True) for i in range(df.height)]
    snaps = [(_snapshot_from_row(r, domain_children), r) for r in rows]

    def by(pred, key=None, reverse=False):
        cand = [(s, r) for (s, r) in snaps if pred(s)]
        if not cand:
            return None
        if key is not None:
            cand.sort(key=lambda sr: key(sr[0]), reverse=reverse)
        return cand[0]

    MAX = 2_320_826  # half-genome bp; |fork|/MAX in [0,1], oriC=0 -> Ter=1

    def fork_frac(s):
        fc = s["fork_coords"]
        return max((abs(c) for c in fc), default=0) / MAX

    # (i) a NO-fork frame -- RNAPs + oriC, no machinery (prefer 1 chromosome,
    #     else the post-replication 2-chromosome segregation frame).
    no_fork = by(lambda s: len(s["fork_coords"]) == 0 and s["n_chromosomes"] == 1) \
        or by(lambda s: len(s["fork_coords"]) == 0)
    # (ii) a SINGLE-bubble frame with forks ~halfway from oriC, so the green
    #      bubble arc reads clearly as emanating from oriC (not collapsed at Ter).
    one_bubble = by(lambda s: len(s["fork_coords"]) == 2,
                    key=lambda s: abs(fork_frac(s) - 0.5))
    # (iii) a MULTIFORK frame -- overlapping rounds (nested bubbles), most forks.
    multifork = by(lambda s: len(s["fork_coords"]) >= 4,
                   key=lambda s: (len(s["fork_coords"]), s["n_rnap"]), reverse=True)

    picks = [p for p in (no_fork, one_bubble, multifork) if p is not None]
    # de-dup by time, keep order
    seen, out = set(), []
    for s, r in picks:
        if s["time"] in seen:
            continue
        seen.add(s["time"]); out.append((s, r))
    # Panels must read left->right in ascending time (chronological order),
    # NOT in state-selection order (no-fork / one-bubble / multifork), which
    # is non-chronological because the post-replication no-fork frame occurs
    # AFTER the mid-replication one-bubble frame.
    out.sort(key=lambda sr: sr[0]["time"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dill", default=None,
                    help="gen dill to recover the domain tree (RNAP-on-bubble placement)")
    a = ap.parse_args()

    files = glob.glob(f"{a.run}/**/history/**/lineage_seed={a.seed}/**/*.pq", recursive=True)
    if not files:
        raise SystemExit(f"no history parquet under {a.run} for seed {a.seed}")
    cols = ["global_time",
            "full_chromosome__unique_index",
            "active_replisome__coordinates", "active_replisome__domain_index",
            "active_RNAP__coordinates", "active_RNAP__domain_index"]
    have = set(pl.scan_parquet(files[0]).collect_schema().names())
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"run missing unique columns {miss} — re-run with V2ECOLI_EMIT_UNIQUE=1")

    # The per-row domain tree (emitted by V2ECOLI_EMIT_UNIQUE) lets the
    # renderer place daughter-strand RNAPs ON the replication bubbles. If the
    # run predates that emit, these columns are absent and we fall back to a
    # dill-derived tree (--dill), else RNAPs go on the rim.
    tree_cols = ["chromosome_domain__domain_index",
                 "chromosome_domain__child_domains"]
    has_tree_cols = all(c in have for c in tree_cols)
    read_cols = cols + [c for c in tree_cols if c in have]

    domain_children = _domain_children_from_dill(a.dill)
    if has_tree_cols:
        print("domain tree: per-row chromosome_domain__child_domains (on-bubble RNAPs)")
    else:
        print(f"domain tree (from dill): {domain_children or 'none — RNAPs on rim'}")

    df = (pl.scan_parquet(files, hive_partitioning=True)
          .filter(pl.col("agent_id").cast(pl.Utf8).str.contains("^0+$"))
          .select(read_cols + ["generation"]).sort("global_time").collect())
    print(f"{df.height} rows for seed {a.seed}")

    picks = _pick_rows(df, domain_children)
    if len(picks) < 3:
        # pad with evenly-spaced frames so we always have 3 panels
        idxs = [0, df.height // 2, df.height - 1]
        extra = [(_snapshot_from_row(df.row(i, named=True), domain_children),
                  df.row(i, named=True)) for i in idxs]
        seen = {s["time"] for s, _ in picks}
        for s, r in extra:
            if len(picks) >= 3:
                break
            if s["time"] not in seen:
                picks.append((s, r)); seen.add(s["time"])
        picks.sort(key=lambda sr: sr[0]["time"])

    picks = picks[:3]
    # picks is already time-ascending; label them as chronological steps.
    labels = ["(i)", "(ii)", "(iii)"]
    print("panels:")
    for lbl, (s, _) in zip(labels, picks):
        print(f"  {lbl} t={s['time']:.0f}s  chrom={s['n_chromosomes']} "
              f"forks={len(s['fork_coords'])} rnap={s['n_rnap']}  [{_phase(s)}]")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0))
    fig.suptitle("Chromosome state at three points across the seed-%d lineage" % a.seed,
                 fontsize=14, y=0.99)
    for ax, lbl, (s, _) in zip(axes, labels, picks):
        _plot_chromosome_map(
            s, ax,
            title=f"{lbl}  t = {s['time']/60:.0f} min  ·  {_phase(s)}\n"
                  f"{s['n_chromosomes']} chr  ·  {len(s['fork_coords'])} replisomes  ·  "
                  f"{s['n_rnap']} RNAPs")
        ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=120, facecolor="white")
    svg = os.path.splitext(a.out)[0] + ".svg"
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}\nwrote {svg}")


if __name__ == "__main__":
    main()
