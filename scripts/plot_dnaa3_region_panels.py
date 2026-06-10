"""DnaA-box occupancy by region across the cell cycle — multi-panel schematic.

Per-chromosome view: at each snapshot tick, group active DnaA_box rows by
their `domain_index`, render one chromosome circle per domain, and show
oriC / dnaA_promoter / chromosomal occupancy for that chromosome only.

Requires the per-box arrays from the extended replication_data listener:
  listeners__replication_data__dnaa_box_{domain_index, pool_label,
                                          bound_form, coordinates}

oriC trajectory plotted underneath, spanning the full gen.

Usage:
    python scripts/plot_dnaa3_region_panels.py \\
        --exp-root out/.../experiment_id \\
        --exp-id   experiment_id \\
        --lineage-seed 1 --gen 4 --n-snapshots 5 \\
        --out out/figures/dnaa3_region_panels_gen4.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


POOL_CHROMOSOMAL_HIGH = 0
POOL_ORIC_HIGH = 1
POOL_ORIC_LOW = 2
POOL_PROMOTER_HIGH = 3

FORM_FREE = 0
FORM_ATP = 1
FORM_ADP = 2


DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"
DNAA_APO_ID = "PD03831[c]"


def load_gen(exp_root: str, exp_id: str, lineage_seed: int, gen: int):
    agent = "0" * gen
    pat = (f"{exp_root}/history/experiment_id={exp_id}/variant=0/"
           f"lineage_seed={lineage_seed}/generation={gen}/agent_id={agent}/*.pq")
    files = sorted(glob.glob(pat),
                   key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
    if not files:
        raise SystemExit(f"no data for gen {gen}")
    # Look up DnaA bulk indices from the first file's bulk__id column.
    bulk_ids = pq.read_table(files[0], columns=["bulk__id"]).column(
        "bulk__id")[0].as_py()
    atp_idx = bulk_ids.index(DNAA_ATP_ID)
    adp_idx = bulk_ids.index(DNAA_ADP_ID)
    apo_idx = bulk_ids.index(DNAA_APO_ID) if DNAA_APO_ID in bulk_ids else None
    cols = [
        "global_time",
        "listeners__replication_data__number_of_oric",
        "listeners__replication_data__dnaa_box_domain_index",
        "listeners__replication_data__dnaa_box_pool_label",
        "listeners__replication_data__dnaa_box_bound_form",
        "bulk__count",
    ]
    import pandas as pd
    rows = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    df = pd.concat(rows).sort_values("global_time").reset_index(drop=True)
    if df["listeners__replication_data__dnaa_box_pool_label"].iloc[0] is None \
            or (hasattr(df["listeners__replication_data__dnaa_box_pool_label"].iloc[0], "__len__")
                and len(df["listeners__replication_data__dnaa_box_pool_label"].iloc[0]) == 0):
        raise SystemExit(
            "per-box arrays empty — the gen was emitted without the extended "
            "listener. Re-run with the updated replication_data step.")
    bulk = np.stack(df["bulk__count"].to_numpy())
    df["dnaa_atp_bulk"] = bulk[:, atp_idx]
    df["dnaa_adp_bulk"] = bulk[:, adp_idx]
    df["dnaa_apo_bulk"] = bulk[:, apo_idx] if apo_idx is not None else 0
    df.drop(columns=["bulk__count"], inplace=True)
    return df


def _per_domain_counts(domain_idx: np.ndarray, pool: np.ndarray,
                       form: np.ndarray) -> dict[int, dict]:
    """Group active boxes by domain_index and return per-domain counts.

    Returns: {domain: {pool: {form: count}, "total_per_pool": {pool: int}}}
    """
    out = {}
    for d in np.unique(domain_idx):
        msk = domain_idx == d
        out[int(d)] = {}
        for p in (POOL_CHROMOSOMAL_HIGH, POOL_ORIC_HIGH, POOL_ORIC_LOW,
                  POOL_PROMOTER_HIGH):
            pmsk = msk & (pool == p)
            forms = form[pmsk]
            out[int(d)][p] = {
                FORM_FREE: int(np.count_nonzero(forms == FORM_FREE)),
                FORM_ATP: int(np.count_nonzero(forms == FORM_ATP)),
                FORM_ADP: int(np.count_nonzero(forms == FORM_ADP)),
                "total": int(pmsk.sum()),
            }
    return out


def _cluster_dots(ax, cx: float, cy: float, n: int, bound: int,
                  dot_size: float = 55) -> None:
    if n <= 0:
        return
    cols = max(int(np.ceil(np.sqrt(n * 1.5))), 1)
    rows_ = int(np.ceil(n / cols))
    spacing = 0.10
    drawn = 0
    for r_ in range(rows_):
        for c_ in range(cols):
            if drawn >= n:
                return
            x = cx + (c_ - (cols - 1) / 2) * spacing
            y = cy - (r_ - (rows_ - 1) / 2) * spacing
            if drawn < bound:
                ax.scatter(x, y, s=dot_size, facecolor="#475569",
                           edgecolor="#1e293b", linewidth=0.5, zorder=3)
            else:
                ax.scatter(x, y, s=dot_size, facecolor="white",
                           edgecolor="#dc2626", linewidth=1.3, zorder=3)
            drawn += 1


def _draw_chromosome(ax, counts: dict, label: str | None) -> None:
    """Render one chromosome circle with oriC_low / oriC_high / dnaA_promoter
    / chromosomal annotations."""
    ax.set_aspect("equal")
    ax.axis("off")
    R = 0.85
    # Keep all content within ±2.0 so adjacent columns don't crash into each
    # other; extra vertical room for the top clusters.
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.8, 3.4)

    # Backbone.
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(R * np.cos(th), R * np.sin(th),
            color="#94a3b8", lw=2.0, zorder=1)

    # ter at bottom.
    ax.add_patch(mpatches.Rectangle((-0.07, -R - 0.09), 0.14, 0.14,
                                    facecolor="#dc2626", edgecolor="#7f1d1d",
                                    linewidth=0.8, zorder=3))

    # ----- oriC LOW-AFFINITY (centred, the gate) -----
    ol = counts[POOL_ORIC_LOW]
    ol_bound = ol[FORM_ATP]
    ol_free = ol[FORM_FREE]
    ol_total = ol["total"]
    cx_lo, cy_lo = 0.0, 2.10
    _cluster_dots(ax, cx_lo, cy_lo, ol_total, ol_bound, dot_size=72)
    ax.text(cx_lo, 3.05,
            f"oriC LOW-aff ({ol_bound}/{ol_total})",
            ha="center", va="bottom", fontsize=10, weight="bold",
            color="#0e7490")
    ax.text(cx_lo, 2.85, "K_d = 100 nM · ATP-only",
            ha="center", va="bottom", fontsize=7,
            color="#0e7490", style="italic")
    ax.text(cx_lo, 2.55, f"{ol_bound} bound / {ol_free} free",
            ha="center", va="bottom", fontsize=7.5, color="#0e7490")
    ax.plot([0, cx_lo], [R + 0.02, cy_lo - 0.25],
            color="#94a3b8", lw=0.8, zorder=1)

    # ----- oriC HIGH-affinity (top-right, secondary) -----
    oh = counts[POOL_ORIC_HIGH]
    oh_bound = oh[FORM_ATP] + oh[FORM_ADP]
    cx_hi, cy_hi = 1.55, 1.40
    _cluster_dots(ax, cx_hi, cy_hi, oh["total"], oh_bound, dot_size=55)
    ax.text(cx_hi, cy_hi + 0.40,
            f"oriC hi-aff ({oh_bound}/{oh['total']})",
            ha="center", va="bottom", fontsize=7.5, color="#475569")
    ang = np.pi / 4
    ax.plot([R * np.cos(ang), cx_hi - 0.18],
            [R * np.sin(ang), cy_hi - 0.05],
            color="#94a3b8", lw=0.8, zorder=1)

    # ----- dnaA_promoter (top-left) -----
    pr = counts[POOL_PROMOTER_HIGH]
    pr_bound = pr[FORM_ATP] + pr[FORM_ADP]
    cx_pr, cy_pr = -1.55, 1.40
    _cluster_dots(ax, cx_pr, cy_pr, pr["total"], pr_bound, dot_size=55)
    ax.text(cx_pr, cy_pr + 0.40,
            f"dnaA_prom ({pr_bound}/{pr['total']})",
            ha="center", va="bottom", fontsize=7.5, color="#475569")
    ang = 3 * np.pi / 4
    ax.plot([R * np.cos(ang), cx_pr + 0.18],
            [R * np.sin(ang), cy_pr - 0.05],
            color="#94a3b8", lw=0.8, zorder=1)

    # ----- chromosomal (right-side label only) -----
    ch = counts[POOL_CHROMOSOMAL_HIGH]
    ch_bound = ch[FORM_ATP] + ch[FORM_ADP]
    ax.text(1.30, -0.85,
            f"chromosomal\n{ch_bound} bound / {ch[FORM_FREE]} free\n"
            f"(of {ch['total']})",
            ha="left", va="top", fontsize=7.5, color="#64748b")
    ang = -np.pi / 5
    ax.plot([R * np.cos(ang), 1.28], [R * np.sin(ang), -0.85],
            color="#94a3b8", lw=0.8, zorder=1)

    # ----- row label (left of the figure, once per row) -----
    if label is not None:
        ax.text(-2.02, 0.0, label, ha="left", va="center",
                fontsize=12, color="#0f172a", weight="bold")


def _pick_snapshot_indices(df, n_snapshots: int) -> np.ndarray:
    """Pick snapshot ticks. With n=5: [start, just-before-init, at-init,
    mid-post-init, end-of-gen]. "at-init" is the LAST tick where n_oric is
    still pre-step — the tick on which the initiation decision is taken
    (number_of_oric will step up on the NEXT tick). This preserves the
    pre-fork oriC_low count (still 8 of 8). Falls back to evenly-spaced
    if no initiation event is found.
    """
    n = df["listeners__replication_data__number_of_oric"].to_numpy()
    init_ticks = np.where(np.diff(n) > 0)[0]
    n_rows = len(df)
    if len(init_ticks) == 0 or n_snapshots < 3:
        return np.linspace(0, n_rows - 1, n_snapshots).round().astype(int)
    init = int(init_ticks[0])
    # `init` itself is the LAST tick with the pre-step n_oric value, i.e. the
    # tick on which the initiation gate fires. "just before init" is one tick
    # earlier — useful for showing that occupancy is not yet building up
    # in Phase 2 (mass-only gate).
    pre_init = max(init - 4, 0)
    anchors = [0, pre_init, init]
    remaining = n_snapshots - len(anchors)
    if remaining == 1:
        anchors.append(n_rows - 1)
    elif remaining >= 2:
        post = np.linspace(init + 1, n_rows - 1, remaining + 1)[1:]
        anchors.extend(int(round(x)) for x in post)
    return np.array(sorted(set(anchors)))


def plot(df, gen: int, n_snapshots: int, out_path: Path,
         title_extra: str) -> None:
    t = df["global_time"].to_numpy()
    t_rel_min = (t - t.min()) / 60.0
    idx = _pick_snapshot_indices(df, n_snapshots)

    # Pre-compute per-domain counts for each snapshot, plus list of all unique
    # domain_indices seen across the chosen snapshots — that fixes the row
    # count for the figure.
    snap_counts = []
    domains_seen = []
    for i in idx:
        row = df.iloc[i]
        d = np.asarray(row["listeners__replication_data__dnaa_box_domain_index"], dtype=np.int64)
        p = np.asarray(row["listeners__replication_data__dnaa_box_pool_label"], dtype=np.int64)
        f = np.asarray(row["listeners__replication_data__dnaa_box_bound_form"], dtype=np.int64)
        c = _per_domain_counts(d, p, f)
        snap_counts.append(c)
        domains_seen.append(sorted(c.keys()))

    # All unique domain indices, sorted ascending. Each becomes one row of
    # chromosome panels.
    all_domains = sorted({d for dl in domains_seen for d in dl})
    n_rows = len(all_domains)

    # Layout: thin time-header row + n_rows × n_snapshots chromosome panels +
    # oriC strip + bulk DnaA-ATP/ADP strip.
    fig_h = 4.4 * n_rows + 1.4 + 1.4 + 0.4
    fig = plt.figure(figsize=(4.0 * n_snapshots, fig_h))
    gs = fig.add_gridspec(
        n_rows + 3, n_snapshots,
        height_ratios=[0.4] + [4.0] * n_rows + [1.4, 1.4],
        hspace=0.30, wspace=0.04,
    )

    fig.suptitle(
        f"{title_extra} — DnaA-box occupancy by region across gen {gen}\n"
        "per-chromosome view (rows = domain_index) · "
        "filled = bound · open red ring = free ·  ter (red square)",
        fontsize=11, y=1.00,
    )

    # Determine which snapshot index is the initiation tick (for tagging).
    n_arr = df["listeners__replication_data__number_of_oric"].to_numpy()
    init_ticks = np.where(np.diff(n_arr) > 0)[0]
    init_tick = int(init_ticks[0]) if len(init_ticks) > 0 else -1

    # Track first/last appearance of each domain so empty cells can be
    # labelled "not yet created" vs "replaced by fork-arm".
    domain_first_seen = {d: min(i for i, snap_doms in enumerate(domains_seen)
                                if d in snap_doms) for d in all_domains}

    # Top row: time headers, one per column.
    for col_i, snap_idx in enumerate(idx):
        ax_h = fig.add_subplot(gs[0, col_i])
        ax_h.axis("off")
        tag = ""
        if init_tick >= 0:
            if snap_idx == init_tick:
                tag = "\n(at initiation)"
            elif snap_idx < init_tick and (init_tick - snap_idx) <= 10:
                tag = "\n(just before init)"
        ax_h.text(0.5, 0.0,
                  f"t = {t_rel_min[snap_idx]:.2f} min{tag}",
                  ha="center", va="bottom", fontsize=11, weight="bold",
                  transform=ax_h.transAxes)

    # Build human-readable row labels:
    #   row 0 (earliest domain present)        = parent chromosome
    #   row 1+ (newly-appearing daughter arms) = daughter chromosome 1, 2, ...
    # The "parent" is the domain alive from gen start; daughters appear at
    # initiation. Only the leftmost panel of each row shows the label.
    daughter_n = 0
    row_labels: list[str] = []
    for d in all_domains:
        if domain_first_seen[d] == 0:
            row_labels.append("parent\nchromosome")
        else:
            daughter_n += 1
            row_labels.append(f"daughter\nchromosome {daughter_n}")

    for row_i, dom in enumerate(all_domains):
        for col_i, (snap_idx, counts) in enumerate(zip(idx, snap_counts)):
            ax = fig.add_subplot(gs[row_i + 1, col_i])
            if dom not in counts:
                ax.axis("off")
                if col_i < domain_first_seen[dom]:
                    msg = "(not yet\ncreated)"
                else:
                    msg = "(replaced\nby fork-arm)"
                ax.text(0.5, 0.5, msg,
                        ha="center", va="center", fontsize=9,
                        color="#94a3b8", style="italic",
                        transform=ax.transAxes)
                continue
            # Only render the row label on the leftmost non-empty panel.
            is_first_for_row = (col_i == domain_first_seen[dom])
            label = row_labels[row_i] if is_first_for_row else None
            _draw_chromosome(ax, counts[dom], label)

    # oriC trajectory.
    ax_o = fig.add_subplot(gs[n_rows + 1, :])
    n_oric = df["listeners__replication_data__number_of_oric"].to_numpy()
    ax_o.step(t_rel_min, n_oric, color="#7c3aed", lw=1.7, where="post")
    for i in idx:
        ax_o.axvline(t_rel_min[i], color="#94a3b8", lw=0.8, ls=":", zorder=0)
    ax_o.set_ylim(-0.3, max(int(n_oric.max()) + 1, 4))
    ax_o.set_yticks([0, 1, 2, 3, 4])
    ax_o.set_ylabel("oriC count")
    ax_o.set_xlim(t_rel_min.min(), t_rel_min.max())
    ax_o.tick_params(labelbottom=False)
    for s in ("top", "right"):
        ax_o.spines[s].set_visible(False)
    ax_o.set_title(f"oriC trajectory across gen {gen}",
                   fontsize=10, loc="left")

    # Bulk DnaA-ATP / DnaA-ADP trajectory.
    ax_b = fig.add_subplot(gs[n_rows + 2, :], sharex=ax_o)
    atp = df["dnaa_atp_bulk"].to_numpy()
    adp = df["dnaa_adp_bulk"].to_numpy()
    ax_b.plot(t_rel_min, atp, color="#16a34a", lw=1.4,
              label=f"bulk DnaA-ATP (end {int(atp[-1])})")
    ax_b.plot(t_rel_min, adp, color="#dc2626", lw=1.4,
              label=f"bulk DnaA-ADP (end {int(adp[-1])})")
    for i in idx:
        ax_b.axvline(t_rel_min[i], color="#94a3b8", lw=0.8, ls=":", zorder=0)
    ax_b.set_xlim(t_rel_min.min(), t_rel_min.max())
    ax_b.set_ylabel("bulk count")
    ax_b.set_xlabel(f"time within gen {gen} (min)")
    ax_b.legend(loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.set_title("Free (bulk) DnaA-ATP / DnaA-ADP",
                   fontsize=10, loc="left")

    handles = [
        mpatches.Patch(facecolor="#475569", edgecolor="#1e293b",
                       label="DnaA-bound box"),
        mpatches.Patch(facecolor="white", edgecolor="#dc2626",
                       label="free box (red ring)"),
        mpatches.Patch(facecolor="#dc2626", edgecolor="#7f1d1d",
                       label="ter"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.99, 1.00),
               frameon=False, fontsize=9, ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=1)
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--n-snapshots", type=int, default=5)
    ap.add_argument("--title-extra", default="dnaa-3 Phase 2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load_gen(args.exp_root, args.exp_id, args.lineage_seed, args.gen)
    plot(df, args.gen, args.n_snapshots, Path(args.out), args.title_extra)


if __name__ == "__main__":
    main()
