"""Animated chromosome / replication-machinery GIF from a parquet run.

Reads the per-timestep unique-molecule coordinate columns that the emitter
persists when a run is launched with V2ECOLI_EMIT_UNIQUE=1 (active_replisome /
active_RNAP / chromosome_domain / full_chromosome). Each frame is rendered with
the existing v2ecoli chromosome renderer (_plot_chromosome_map → _draw_chromosome
+ _draw_replication_bubbles): chromosome rim, oriC/Ter, replisomes (gold), the
replication bubble (green daughter-strand arc), and RNAPs on both the rim and the
bubble (blue dots). Frames are assembled into a looping GIF with PIL.

Usage:
  V2ECOLI_EMIT_UNIQUE is NOT needed here (read-only). Just point at the run dir:
  .venv/bin/python scripts/render_chromosome_gif.py \
      --run out/dnaa_gifdata --seed 1 --every 30 \
      --out studies/dnaa-3-box-binding/charts/chromosome_replication.gif
"""
from __future__ import annotations
import argparse, glob, io, os, sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from v2ecoli.visualizations.workflow import _plot_chromosome_map


def _domain_children_from_dill(dill_path):
    """Recover the chromosome-domain tree from a gen dill (the nested
    child_domains column is excluded from parquet — see emitter note)."""
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
    """Build the parent->child domain tree from emitted unique columns.

    ``chromosome_domain__child_domains`` is a flat list<int> of length
    ``2*n_domain`` aligned to ``chromosome_domain__domain_index`` (negatives =
    no-child placeholder). Returns ``{}`` when the columns are absent.
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
        kids = [k for k in (cd[2 * i], cd[2 * i + 1]) if k >= 0]
        if kids:
            out[parent] = kids
    return out


def _snapshot_from_row(row: dict, domain_children: dict) -> dict:
    """Build the snapshot dict _plot_chromosome_map expects from a parquet row."""
    def _lst(key):
        v = row.get(key)
        return list(v) if v is not None else []
    row_tree = _domain_children_from_row(row)
    if row_tree:
        domain_children = row_tree
    fc = _lst("full_chromosome__unique_index")
    rep_c = _lst("active_replisome__coordinates")
    rep_d = _lst("active_replisome__domain_index")
    rnap_c = _lst("active_RNAP__coordinates")
    rnap_d = _lst("active_RNAP__domain_index")
    n_dom = len({int(d) for d in rep_d} | {int(d) for d in rnap_d}) or 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="parquet run dir (with **/history/**)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=30, help="subsample: keep ~1 frame per N seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=int, default=120, help="ms per frame")
    ap.add_argument("--dill", default=None, help="gen dill to recover the domain tree (RNAP-on-bubble placement)")
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
          .select(read_cols).sort("global_time").collect())
    # subsample by time
    t = df["global_time"].to_numpy()
    keep, last = [], -1e9
    for i, tt in enumerate(t):
        if tt - last >= a.every:
            keep.append(i); last = tt
    print(f"{len(df)} rows → {len(keep)} frames (every {a.every}s)")

    frames = []
    for i in keep:
        row = df.row(i, named=True)
        snap = _snapshot_from_row(row, domain_children)
        nch = snap["n_chromosomes"]
        phase = ("pre-initiation" if not snap["fork_coords"] and nch == 1 else
                 "two chromosomes — replication complete" if nch >= 2 else
                 "replication bubble extending")
        # Fixed canvas (no tight bbox) so every frame is identical pixel size →
        # a smooth GIF across the 1-chromosome → 2-chromosome transition.
        fig, ax = plt.subplots(figsize=(6.0, 6.6))
        _plot_chromosome_map(snap, ax,
                             title=f"t = {snap['time']/60:.0f} min   ·   {phase}\n"
                                   f"forks {len(snap['fork_coords'])}   ·   RNAPs {snap['n_rnap']}"
                                   f"   ·   chromosomes {nch}")
        ax.set_aspect("equal"); ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=92, facecolor="white")
        plt.close(fig); buf.seek(0); frames.append(Image.open(buf).convert("RGB"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=a.duration, loop=0)
    print(f"wrote {a.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
