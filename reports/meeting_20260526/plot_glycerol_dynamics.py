"""Render chromosome + bulk dynamics for the glycerol Stage 1 gen 1 sim."""
import json
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DB_PATH = "out/glycerol_stage1_1gen.db"
OUT_PATH = "reports/meeting_20260526/glycerol_dynamics.png"

# bulk indices used in studies/stage1_sanity_check/plot_diagnostics.py
BULK_DNAA_MONOMER = 11565   # PD03831[c]      — unbound DnaA
BULK_DNAA_ATP     = 10822   # MONOMER0-160[c]
BULK_DNAA_ADP     = 11114   # MONOMER0-4565[c]
BULK_DNAA_MRNA    = 15272   # TU00259[c]      — dnaA-containing TU


def count_unique(unique: dict, name: str) -> int:
    arr = unique.get(name)
    if arr is None:
        return 0
    if isinstance(arr, dict):
        return int(arr.get("_entryState", [0])[0] if arr else 0)
    # snapshot serialized list-of-dicts entries
    if isinstance(arr, list):
        return sum(1 for e in arr if e.get("_entryState", 0))
    return 0


def main():
    conn = sqlite3.connect(DB_PATH)
    sid = conn.execute("SELECT simulation_id FROM simulations LIMIT 1").fetchone()[0]
    rows = conn.execute(
        "SELECT step, state FROM history WHERE simulation_id=? ORDER BY step",
        (sid,),
    ).fetchall()
    print(f"loaded {len(rows)} rows", flush=True)

    sample_every = max(1, len(rows) // 600)

    t = []
    cell_mass = []
    dry_mass = []
    protein_mass = []
    rna_mass = []
    n_oric = []
    n_forks = []
    dnaa_total = []
    dnaa_atp = []
    dnaa_adp = []
    dnaa_unbound = []
    free_boxes = []
    total_boxes = []

    for i, (step, state_str) in enumerate(rows):
        if i % sample_every != 0 and i != len(rows) - 1:
            continue
        s = json.loads(state_str)
        mass = s.get("listeners", {}).get("mass", {}) or {}
        rd = s.get("listeners", {}).get("replication_data", {}) or {}
        bulk = s.get("bulk", []) or []

        t.append(float(s.get("time", step)) / 60.0)
        cell_mass.append(float(mass.get("cell_mass", 0.0) or 0.0))
        dry_mass.append(float(mass.get("dry_mass", 0.0) or 0.0))
        protein_mass.append(float(mass.get("protein_mass", 0.0) or 0.0))
        rna_mass.append(
            float(mass.get("rRna_mass", 0.0) or 0.0)
            + float(mass.get("tRna_mass", 0.0) or 0.0)
            + float(mass.get("mRna_mass", 0.0) or 0.0)
        )
        n_oric.append(int(rd.get("number_of_oric", 0) or 0))
        fc = rd.get("fork_coordinates") or []
        n_forks.append(len(fc) // 2)
        free_boxes.append(int(rd.get("free_DnaA_boxes", 0) or 0))
        total_boxes.append(int(rd.get("total_DnaA_boxes", 0) or 0))

        def b(idx):
            if bulk and idx < len(bulk):
                v = bulk[idx]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    return int(v[1])
                return int(v)
            return 0
        d_un = b(BULK_DNAA_MONOMER)
        d_at = b(BULK_DNAA_ATP)
        d_ad = b(BULK_DNAA_ADP)
        dnaa_unbound.append(d_un)
        dnaa_atp.append(d_at)
        dnaa_adp.append(d_ad)
        dnaa_total.append(d_un + d_at + d_ad)

    t = np.array(t)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"Glycerol Stage 1 — gen 1 dynamics  "
        f"(t_div = {t[-1]:.1f} min, dry_mass {dry_mass[0]:.0f}→{dry_mass[-1]:.0f} fg)",
        fontsize=11, y=0.995,
    )

    # Panel 1: mass components
    ax = axes[0, 0]
    ax.plot(t, cell_mass, color="#0969da", lw=1.4, label="cell mass")
    ax.plot(t, dry_mass, color="#2470d6", lw=1.2, label="dry mass")
    ax.plot(t, protein_mass, color="#1a7f37", lw=1.0, alpha=0.8, label="protein")
    ax.plot(t, rna_mass, color="#bf8700", lw=1.0, alpha=0.8, label="RNA")
    ax.set_ylabel("mass (fg)", fontsize=10)
    ax.set_title("Mass components", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel 2: replication dynamics
    ax = axes[0, 1]
    ax.step(t, n_oric, color="#0969da", lw=1.5, where="post", label="n oriC")
    ax.step(t, n_forks, color="#cf222e", lw=1.5, where="post", label="n active forks")
    ax.set_ylabel("count", fontsize=10)
    ax.set_title("Replication dynamics", fontsize=10)
    ax.set_ylim(-0.3, max(max(n_oric or [1]), max(n_forks or [1]), 2) + 0.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel 3: DnaA bulk counts
    ax = axes[1, 0]
    ax.plot(t, dnaa_total, color="#1f2328", lw=1.5, label="total DnaA")
    ax.plot(t, dnaa_atp, color="#1F77B4", lw=1.2, alpha=0.9, label="DnaA-ATP")
    ax.plot(t, dnaa_adp, color="#D62728", lw=1.2, alpha=0.9, label="DnaA-ADP")
    ax.plot(t, dnaa_unbound, color="#6e40c9", lw=1.0, alpha=0.7, label="apo-DnaA")
    ax.set_xlabel("time (min)", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title("DnaA species (bulk)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel 4: DnaA-box occupancy
    ax = axes[1, 1]
    bound = np.array(total_boxes) - np.array(free_boxes)
    ax.plot(t, total_boxes, color="#1f2328", lw=1.4, label="total DnaA boxes")
    ax.plot(t, free_boxes, color="#6e40c9", lw=1.2, alpha=0.9, label="free")
    ax.plot(t, bound, color="#1a7f37", lw=1.2, alpha=0.9, label="bound")
    ax.set_xlabel("time (min)", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title("DnaA-box occupancy", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.tick_params(labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
