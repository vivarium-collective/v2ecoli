"""Single-followed-lineage, real-division multi-generation test for the
NFsim-wired flagella complexation pipeline.

Added 2026-08-19. Adapts flagella-02-transcription-regulation's
run_lineage_multigen.py -- a proven pattern that already found and fixed a
real dry_mass-drift bug in an earlier manual-splice approach (division state
hand-spliced onto a fresh baseline() rebuild drifted 706.7 -> 262.0 fg over
7 generations and destabilized the metabolism FBA solver). This script
reuses the SAME proven approach instead of re-deriving it: drive real
division through the actual Division step's own daughter-construction
machinery, then prune the resulting 2-agent population back down to one
followed lineage (mother-machine-style tracking, Wang et al. 2010 Curr
Biol) so compute cost stays linear in generation count. The daughter to
keep is chosen by its own decorrelated RNG stream (not a fixed rule) for
the same reason documented in the original script: this model has no
old-pole/new-pole asymmetry mechanism, so an unbiased random choice avoids
any risk of correlating with a fixed positional convention in the divider.

This is also the first REAL (not synthetic) end-to-end test of the
divide_scaffold_species/divide_internal_observables fix added earlier this
session (v2ecoli/library/division.py, v2ecoli/steps/division.py) -- before
that fix, nfsim_scaffold_species and nfsim_internal_observables silently
reset to {} at every division (they weren't part of divide_cell()'s output
at all). Tracked explicitly here so a real, naturally-triggered division
event's before/after state is directly visible, not just the earlier
synthetic divide_cell() call on saved state.

Same standard INIT used throughout this investigation, applied once at
t=0: 4 flagella, 0 motor, free FliA=500, FlgM=800.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/run_nfsim_lineage_multigen.py \
        --generations 2 --sample 120 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _snap(comp, agent_id, idx, t_cum, gen, completed_ever, prev_flag):
    cell = comp.state["agents"][agent_id]
    b = _arr(cell["bulk"])
    nf = _arr(cell["unique"]["nascent_flagellum"])
    nf_mask = nf["_entryState"].view(bool)
    lengths = nf["filament_length"][nf_mask]
    flag = int(b["count"][idx["flag"]])
    flic = int(b["count"][idx["flic"]])
    dry_mass = fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0))

    scaffold = cell.get("nfsim_scaffold_species") or {}
    internal = cell.get("nfsim_internal_observables") or {}

    if prev_flag[0] is not None and flag > prev_flag[0]:
        completed_ever[0] += (flag - prev_flag[0])
    prev_flag[0] = flag

    return {
        "t_cum": t_cum, "gen": gen, "dry_mass": dry_mass,
        "flag": flag, "flic": flic,
        "n_nascent": int(len(lengths)),
        "mean_len": float(lengths.mean()) if len(lengths) else 0.0,
        "max_len": int(lengths.max()) if len(lengths) else 0,
        "completed_ever": completed_ever[0],
        "n_scaffold_entries": len(scaffold),
        "scaffold_total": float(sum(scaffold.values())) if scaffold else 0.0,
        "hook_internal": float(internal.get("flagellar_hook", 0.0)),
        "flagella_internal_cumulative": float(internal.get("flagella", 0.0)),
    }


def run_lineage(n_gens, sample, seconds_cap, seed, cache_dir, nfsim_interval=None):
    enable_features("flagella_nfsim_complexation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    if nfsim_interval is not None:
        for path, subtree in comp.step_paths.items():
            if path and path[-1] == "ecoli-flagella-nfsim-complexation":
                instance = subtree.get("instance") if isinstance(subtree, dict) else None
                if instance is not None:
                    instance.interval = float(nfsim_interval)
                    break

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in INIT.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val
    idx = {
        "flag": bulk_name_to_idx("CPLX0-7452[j]", bids),
        "flic": bulk_name_to_idx("EG10321-MONOMER[e]", bids),
    }

    rows = []
    agent_id = "0"
    gen = 1
    completed_ever = [0]
    prev_flag = [None]

    rows.append(_snap(comp, agent_id, idx, 0.0, gen, completed_ever, prev_flag))

    total = 0.0
    while total < seconds_cap and gen <= n_gens:
        chunk = min(sample, seconds_cap - total)
        comp.run(chunk)
        total += chunk

        agents = comp.state.get("agents", {})
        if len(agents) > 1:
            # Real division happened -- capture scaffold/internal state
            # BEFORE pruning, so the mother -> daughter carry-over is
            # directly visible in the log (the actual point of this test).
            mother_scaffold = rows[-1]["n_scaffold_entries"] if rows else 0
            mother_hook = rows[-1]["hook_internal"] if rows else 0.0
            mother_flagint = rows[-1]["flagella_internal_cumulative"] if rows else 0.0

            candidate_ids = sorted(agents.keys())
            pick_rng = np.random.RandomState(seed=(seed * 1000 + gen) % (2**31 - 1))
            keep_id = candidate_ids[pick_rng.randint(len(candidate_ids))]
            discard_ids = [aid for aid in candidate_ids if aid != keep_id]
            comp.apply({"agents": {"_remove": discard_ids}})
            agent_id = keep_id
            gen += 1
            prev_flag[0] = None
            _birth = _snap(comp, agent_id, idx, total, gen, completed_ever, prev_flag)
            print(f"  gen {gen} begins at t_cum={total:.0f}s, following agent '{agent_id}' "
                  f"-- BIRTH: dry_mass={_birth['dry_mass']:.1f}fg "
                  f"scaffold_entries={_birth['n_scaffold_entries']} (mother had {mother_scaffold}) "
                  f"hook_internal={_birth['hook_internal']:.1f} (mother had {mother_hook:.1f}) "
                  f"flagella_internal_cumulative={_birth['flagella_internal_cumulative']:.1f} "
                  f"(mother had {mother_flagint:.1f})")

        if agent_id not in comp.state.get("agents", {}):
            print(f"  agent '{agent_id}' vanished unexpectedly — stopping")
            break

        row = _snap(comp, agent_id, idx, total, gen, completed_ever, prev_flag)
        rows.append(row)
        if int(total) % 1800 < sample:
            print(f"    t_cum={total:.0f}s ({total/60:.0f}min) gen={gen} agent='{agent_id}' "
                  f"flag={row['flag']} n_nascent={row['n_nascent']} max_len={row['max_len']} "
                  f"free_flic={row['flic']} completed_ever={row['completed_ever']} "
                  f"dry_mass={row['dry_mass']:.1f}fg "
                  f"scaffold_entries={row['n_scaffold_entries']} "
                  f"hook_internal={row['hook_internal']:.1f} "
                  f"flagella_internal_cumulative={row['flagella_internal_cumulative']:.1f}")

    return rows


def _cols(rows, key):
    return np.array([r[key] for r in rows])


def _gen_bounds(rows):
    bounds = []
    for i in range(1, len(rows)):
        if rows[i]["gen"] != rows[i - 1]["gen"]:
            bounds.append(rows[i]["t_cum"] / 60.0)
    return bounds


def _shade(ax, rows):
    gens = sorted({r["gen"] for r in rows})
    colors = ["#eef4ff", "#fff4ee", "#eefff2", "#f7eeff"]
    for gi in gens:
        xs = [r["t_cum"] / 60.0 for r in rows if r["gen"] == gi]
        if xs:
            ax.axvspan(min(xs), max(xs), color=colors[(gi - 1) % len(colors)], alpha=0.5, zorder=0)
    for b in _gen_bounds(rows):
        ax.axvline(b, color="#c0392b", ls="--", lw=1, alpha=0.7)


def figure(rows, n_gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = _cols(rows, "t_cum") / 60.0
    fig, axs = plt.subplots(2, 4, figsize=(22.0, 9.0))
    (a, b, c, d), (e, f, g, h) = axs

    a.plot(t, _cols(rows, "flag"), "-o", ms=2, color="#9467bd", label="complete flagella (this lineage)")
    a.plot(t, _cols(rows, "completed_ever"), "-s", ms=2, color="#d62728", label="cumulative completions")
    _shade(a, rows)
    a.set_title("Complete flagella, followed lineage")
    a.set_xlabel("time (min)"); a.set_ylabel("count"); a.legend(fontsize=8)

    b.plot(t, _cols(rows, "n_nascent"), "-o", ms=2, color="#8c564b")
    _shade(b, rows)
    b.set_title("flagella under construction (n_nascent)")
    b.set_xlabel("time (min)"); b.set_ylabel("count")

    c.plot(t, _cols(rows, "mean_len"), "-o", ms=2, color="#17becf", label="mean filament_length")
    c.plot(t, _cols(rows, "max_len"), "-s", ms=2, color="#17becf", alpha=0.5, label="max filament_length")
    c.axhline(5000, color="gray", ls=":", lw=1, label="target (5,000)")
    _shade(c, rows)
    c.set_title("filament construction progress")
    c.set_xlabel("time (min)"); c.set_ylabel("subunits"); c.legend(fontsize=7)

    d.plot(t, _cols(rows, "flic"), "-o", ms=2, color="#bcbd22")
    _shade(d, rows)
    d.set_title("free FliC monomer (supply pool)")
    d.set_xlabel("time (min)"); d.set_ylabel("count")

    e.plot(t, _cols(rows, "dry_mass"), "-o", ms=2, color="#1f77b4")
    _shade(e, rows)
    e.set_title("dry mass (sanity check: should NOT drift down across generations)")
    e.set_xlabel("time (min)"); e.set_ylabel("fg")

    f.plot(t, _cols(rows, "n_scaffold_entries"), "-o", ms=2, color="#e377c2", label="distinct entries")
    f.plot(t, _cols(rows, "scaffold_total"), "-s", ms=2, color="#e377c2", alpha=0.5, label="total count")
    _shade(f, rows)
    f.set_title("nfsim_scaffold_species (survives division?)")
    f.set_xlabel("time (min)"); f.set_ylabel("count"); f.legend(fontsize=7)

    g.plot(t, _cols(rows, "hook_internal"), "-o", ms=2, color="#ff7f0e")
    _shade(g, rows)
    g.set_title("hook (internal, survives division?)")
    g.set_xlabel("time (min)"); g.set_ylabel("count")

    h.plot(t, _cols(rows, "flagella_internal_cumulative"), "-o", ms=2, color="#9467bd")
    _shade(h, rows)
    h.set_title("hook-basal-body complete, cumulative (survives division?)")
    h.set_xlabel("time (min)"); h.set_ylabel("count")

    fig.suptitle(f"NFsim-driven single-lineage, {n_gens}-generation test (real Division "
                 f"machinery, pruned to 1 followed agent) — does scaffold/internal state "
                 f"survive real division? (shaded=generation, dashed=division)")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/25_nfsim_lineage_multigen_{n_gens}gen.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seconds-cap", type=int, default=36000,
                     help="hard stop on total simulated time, as a safety ceiling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v11")
    ap.add_argument("--nfsim-interval", type=float, default=None)
    args = ap.parse_args()
    rows = run_lineage(args.generations, args.sample, args.seconds_cap, args.seed,
                        args.cache_dir, nfsim_interval=args.nfsim_interval)
    figure(rows, args.generations)
    last = rows[-1]
    print(f"\nFINAL (gen {last['gen']}, t_cum={last['t_cum']:.0f}s / {last['t_cum']/60:.0f}min): "
          f"flag={last['flag']}  completed_ever={last['completed_ever']}  "
          f"n_nascent={last['n_nascent']}  max_len={last['max_len']}  "
          f"free_flic={last['flic']}  dry_mass={last['dry_mass']:.1f}fg  "
          f"scaffold_entries={last['n_scaffold_entries']}  "
          f"hook_internal={last['hook_internal']:.1f}  "
          f"flagella_internal_cumulative={last['flagella_internal_cumulative']:.1f}")


if __name__ == "__main__":
    main()
