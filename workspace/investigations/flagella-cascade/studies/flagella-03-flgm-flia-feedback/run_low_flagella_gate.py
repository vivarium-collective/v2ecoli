"""Low-flagella initial condition — expose the assembly-gated Class III onset.

The default v2ecoli cache starts already flagellated (~30-35 complete flagella with
FlgM already partly secreted), so the assembly-gated DELAY of Class III is hidden.
This driver applies Maya's flagellum_initial_value.json analog (4 flagella, 500 free
FliA, 800 FlgM, 0 motor) before running the flagella_regulation lineage, so FlgM
secretion is slow at first, FliA stays sequestered, and Class III override rises only
as FlgM is exported — the timed Class II -> Class III cascade.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-03-flgm-flia-feedback/run_low_flagella_gate.py \
        --seconds 900 --sample 30
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.core import load_cache_bundle
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
# Low-flagella start. We deliberately set ONLY the flagellar structures low —
# NOT the equilibrium species FliA (EG11355-MONOMER) / FlgM (G369-MONOMER):
# directly overriding the free-monomer pools makes the FlgM-FliA equilibrium
# solver go negative at steady state. Fewer complete flagella -> slow FlgM export
# -> FliA stays sequestered longer -> the assembly-gated Class III onset is exposed
# (FlgM/FliA keep their calibrated cache values).
INIT = {
    "CPLX0-7452[j]": 4,                 # complete flagella (Maya's flagellum_initial_value: 4)
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
}
# Observable bulk IDs we always read (whether or not we override them).
READ_IDS = ["CPLX0-7452[j]", "EG11355-MONOMER[c]", "G369-MONOMER[c]"]


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def run(seconds, sample, seed, cache_dir):
    cfg = load_cache_bundle(cache_dir)["configs"]["ecoli-flagella-transcription-regulation"]
    rna_ids = list(cfg["rna_ids"])
    tu_II = set(rna_ids.index(r) for r in cfg["flg_classII_rnaids"])
    tu_III = set(rna_ids.index(r) for r in cfg["flg_classIII_rnaids"])

    enable_features("flagella_regulation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    # Apply the low-flagella initial condition in place.
    for name, val in INIT.items():
        try:
            bulk["count"][bulk_name_to_idx(name, bids)] = val
        except Exception as e:
            print("  (skip IC", name, "->", e, ")")
    idx = {k: bulk_name_to_idx(k, bids) for k in READ_IDS}

    rec = {"t": [], "flgM": [], "fliA": [], "flag": [], "II": [], "III": []}

    def snap(t):
        cell = comp.state["agents"]["0"]
        b = _arr(cell["bulk"])
        p = _arr(cell["unique"]["promoter"])
        m = p["_entryState"].view(bool)
        tu, ov = p["TU_index"][m], p["init_prob_override"][m]
        II = ov[np.isin(tu, list(tu_II))]
        III = ov[np.isin(tu, list(tu_III))]
        rec["t"].append(t)
        rec["flgM"].append(int(b["count"][idx["G369-MONOMER[c]"]]))
        rec["fliA"].append(int(b["count"][idx["EG11355-MONOMER[c]"]]))
        rec["flag"].append(int(b["count"][idx["CPLX0-7452[j]"]]))
        rec["II"].append(float(II.mean()) if len(II) else 0.0)
        rec["III"].append(float(III.mean()) if len(III) else 0.0)

    snap(0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        comp.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = rec["t"] / 60.0
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    a.plot(t, rec["flgM"], "-s", ms=3, color="#ff7f0e", label="FlgM G369-MONOMER[c]")
    a.plot(t, rec["fliA"], "-o", ms=3, color="#2ca02c", label="free FliA EG11355-MONOMER[c]")
    ab = a.twinx(); ab.plot(t, rec["flag"], "-^", ms=3, color="#9467bd", alpha=0.7, label="flagella")
    ab.set_ylabel("flagella", color="#9467bd")
    a.set_title("Low-flagella start (4): slow FlgM export → gated FliA release")
    a.set_xlabel("time (min)"); a.set_ylabel("molecule count")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = ab.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    b.plot(t, rec["II"], "-o", ms=3, color="#1f77b4", label="Class II ⟨override⟩")
    b.plot(t, rec["III"], "-s", ms=3, color="#d62728", label="Class III ⟨override⟩")
    b.set_title("Class III override rises only as FliA is freed (timed cascade)")
    b.set_xlabel("time (min)"); b.set_ylabel("mean init_prob_override"); b.legend(fontsize=8)
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/02_lowIC_gated_cascade.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache")
    args = ap.parse_args()
    rec = run(args.seconds, args.sample, args.seed, args.cache_dir)
    figure(rec)
    print(f"FlgM {rec['flgM'][0]}->{rec['flgM'][-1]}  FliA {rec['fliA'][0]}->{rec['fliA'][-1]}  "
          f"ClassIII ⟨ov⟩ {rec['III'][0]:.2e}->{rec['III'][-1]:.2e}")


if __name__ == "__main__":
    main()
