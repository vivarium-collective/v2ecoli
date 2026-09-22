"""Single-followed-lineage, real-division multi-generation test for the
NFsim-wired flagella complexation pipeline.

Added 2026-08-19. Adapts flagella-02-transcription-regulation's
run_lineage_multigen.py -- a proven pattern that already found and fixed
a real dry_mass-drift bug in an earlier manual-splice approach (division
state hand-spliced onto a fresh baseline() rebuild drifted 706.7 -> 262.0
fg over 7 generations, destabilizing the metabolism FBA solver). Reuses
that approach: drive real division through the Division step's own
daughter-construction machinery, then prune the resulting 2-agent
population to one followed lineage (mother-machine-style, Wang et al.
2010 Curr Biol) so compute cost stays linear in generation count. The
daughter to keep is chosen by its own decorrelated RNG stream (not a
fixed rule) -- this model has no old-pole/new-pole asymmetry mechanism,
so an unbiased random choice avoids correlating with a fixed positional
convention in the divider.

Also the first REAL (not synthetic) end-to-end test of the
divide_scaffold_species/divide_internal_observables fix
(v2ecoli/library/division.py, v2ecoli/steps/division.py) -- before that
fix, nfsim_scaffold_species and nfsim_internal_observables silently
reset to {} at every division (not part of divide_cell()'s output).
Tracked explicitly here so a real, naturally-triggered division event's
before/after state is directly visible, not just a synthetic
divide_cell() call on saved state.

Same standard INIT used throughout this investigation, applied once at
t=0: 4 flagella, 0 motor, free FliA=500, FlgM=800.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/studies/flagella-04-complexation-nfsim/run_nfsim_lineage_multigen.py \
        --generations 2 --sample 120 --cache-dir out/cache_full_flit_v11
"""
import argparse
import os
import re

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

# Full species list, same as run_nfsim_population_multigen.py's TRACK_IDS --
# added 2026-09-15 so this script tracks everything the population script
# does, for a full panel set on the one followed lineage (previously only
# tracked flag/flic, missing FliA/FlgM/FlgM:FliA/FliS/FLIS-FLIC-CPLX/
# C-ring/export apparatus/motor complex entirely).
TRACK_IDS = {
    "EG10320-MONOMER[c]": "FlhD",
    "MONOMER0-2488[c]": "FlhC",
    "CPLX0-3930[c]": "FlhDC complex",
    "EG11355-MONOMER[c]": "FliA",
    "G369-MONOMER[c]": "FlgM",
    "FLGM-FLIA-CPLX[c]": "FlgM:FliA complex",
    "EG11388-MONOMER[c]": "FliS",
    "FLIS-FLIC-CPLX[e]": "FLIS-FLIC-CPLX",
    "CPLX0-7450[i]": "C-ring",
    "CPLX0-7451[j]": "export apparatus",
    "FLAGELLAR-MOTOR-COMPLEX[j]": "motor complex",
    "EG10321-MONOMER[e]": "free FliC",
    "CPLX0-7452[j]": "complete flagella",
}


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


_RATES_FREEVAR_NAMES = {"_rates_fwd", "_rates_rev", "_rates_fwd_ss", "_rates_rev_ss"}


def _zero_shared_flgm_flia(comp):
    """FlgM:FliA exact-solve Step wired in 2026-09-22 (see ecoli_baseline.py's
    before_steps note) -- zero the shared ecoli-equilibrium Step's own
    FLGM-FLIA-CPLX_RXN copy so it doesn't also move the same three species
    every tick at the relaxed Kd, fighting the dedicated Step's real-Kd
    solve. In-memory only; same closure-walking patch verified in
    run_flgm_flia_ordering_diagnostic.py's own version of this function --
    see that docstring for why a flat instance.rates_fwd isn't available
    for this cache's config shape."""
    for path, subtree in comp.step_paths.items():
        if path and path[-1] == "ecoli-equilibrium":
            instance = subtree.get("instance") if isinstance(subtree, dict) else None
            if instance is None:
                return
            rxn_ids = list(instance.reaction_ids)
            if "FLGM-FLIA-CPLX_RXN" not in rxn_ids:
                return
            idx = rxn_ids.index("FLGM-FLIA-CPLX_RXN")
            f = instance.fluxesAndMoleculesToSS
            for cell, name in zip(f.__closure__, f.__code__.co_freevars):
                nested = cell.cell_contents
                if not (callable(nested) and getattr(nested, "__closure__", None)):
                    continue
                for c2, n2 in zip(nested.__closure__, nested.__code__.co_freevars):
                    if n2 in _RATES_FREEVAR_NAMES:
                        c2.cell_contents[idx] = 0.0
            return


def _snap(comp, agent_id, idx, t_cum, gen, completed_ever, prev_flag):
    cell = comp.state["agents"][agent_id]
    b = _arr(cell["bulk"])
    nf = _arr(cell["unique"]["nascent_flagellum"])
    nf_mask = nf["_entryState"].view(bool)
    lengths = nf["filament_length"][nf_mask]
    flag = int(b["count"][idx["CPLX0-7452[j]"]])
    flic = int(b["count"][idx["EG10321-MONOMER[e]"]])
    dry_mass = fg_magnitude(cell["listeners"]["mass"].get("dry_mass", 0))

    scaffold = cell.get("nfsim_scaffold_species") or {}
    internal = cell.get("nfsim_internal_observables") or {}

    if prev_flag[0] is not None and flag > prev_flag[0]:
        completed_ever[0] += (flag - prev_flag[0])
    prev_flag[0] = flag

    row = {
        "t_cum": t_cum, "gen": gen, "dry_mass": dry_mass,
        "flag": flag, "flic": flic,
        "n_nascent": int(len(lengths)),
        "mean_len": float(lengths.mean()) if len(lengths) else 0.0,
        "max_len": int(lengths.max()) if len(lengths) else 0,
        "completed_ever": completed_ever[0],
        "n_scaffold_entries": len(scaffold),
        "scaffold_total": float(sum(scaffold.values())) if scaffold else 0.0,
        "hook_internal": float(internal.get("flagellar_hook", 0.0)),
        "export_apparatus_subunit_internal": float(
            internal.get("flagellar_export_apparatus_subunit", 0.0)),
        "rod_internal": float(internal.get("flagellar_rod", 0.0)),
        "rod_p_ring_internal": float(internal.get("flagellar_rod_with_p_ring", 0.0)),
        "flagella_internal_cumulative": float(internal.get("flagella", 0.0)),
        "hook_internal_cumulative": float(internal.get("flagellar_hook__cumulative", 0.0)),
        "export_apparatus_subunit_cumulative": float(
            internal.get("flagellar_export_apparatus_subunit__cumulative", 0.0)),
        "rod_cumulative": float(internal.get("flagellar_rod__cumulative", 0.0)),
        "rod_p_ring_cumulative": float(
            internal.get("flagellar_rod_with_p_ring__cumulative", 0.0)),
        "cring_cumulative": float(internal.get("CPLX0-7450[i]__cumulative", 0.0)),
        "export_apparatus_cumulative": float(internal.get("CPLX0-7451[j]__cumulative", 0.0)),
        "motor_complex_cumulative": float(
            internal.get("FLAGELLAR-MOTOR-COMPLEX[j]__cumulative", 0.0)),
    }
    for real_id in TRACK_IDS:
        row[real_id] = int(b["count"][idx[real_id]])
    return row


def run_lineage(n_gens, sample, seconds_cap, seed, cache_dir, nfsim_interval=None):
    enable_features("flagella_nfsim_complexation")
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()
    # _zero_shared_flgm_flia(comp)  # only needed while the dedicated Step
    # is wired in (ecoli_baseline.py) -- reverted 2026-09-22 pending a real
    # FlhDC shutdown mechanism; calling this with the Step unwired would
    # zero the shared solver's FlgM:FliA handling with nothing to replace
    # it. Leave commented until the Step is re-wired again.

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
    idx = {real_id: bulk_name_to_idx(real_id, bids) for real_id in TRACK_IDS}

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


# _shade() drew shaded generation backgrounds + a dashed division line,
# one axis at a time. Replaced 2026-09-04 with a single dotted
# division-line pass, matching run_nfsim_population_multigen.py's
# convention. Kept:
# def _shade(ax, rows):
#     gens = sorted({r["gen"] for r in rows})
#     colors = ["#eef4ff", "#fff4ee", "#eefff2", "#f7eeff"]
#     for gi in gens:
#         xs = [r["t_cum"] / 60.0 for r in rows if r["gen"] == gi]
#         if xs:
#             ax.axvspan(min(xs), max(xs), color=colors[(gi - 1) % len(colors)], alpha=0.5, zorder=0)
#     for b in _gen_bounds(rows):
#         ax.axvline(b, color="#c0392b", ls="--", lw=1, alpha=0.7)


# Merged 2026-09-15: this script's own metric-name-keyed colors, plus
# run_nfsim_population_multigen.py's TRACK_ID-keyed colors (same
# convention, no key overlap between the two sets).
COLORS = {
    "flag": "#9467bd",
    "completed_ever": "#d62728",
    "n_nascent": "black",
    "mean_len": "#17becf",
    "max_len": "#17becf",
    "flic": "#bcbd22",
    "dry_mass": "#1f77b4",
    "n_scaffold_entries": "#e377c2",
    "scaffold_total": "#e377c2",
    "hook_internal": "#8c564b",
    "flagella_internal_cumulative": "#9467bd",
    "EG10320-MONOMER[c]": "#2ca02c",
    "MONOMER0-2488[c]": "#17becf",
    "CPLX0-3930[c]": "#bcbd22",
    "EG11355-MONOMER[c]": "#1f77b4",
    "G369-MONOMER[c]": "#d62728",
    "FLGM-FLIA-CPLX[c]": "#9467bd",
    "EG11388-MONOMER[c]": "#8c564b",
    "FLIS-FLIC-CPLX[e]": "#e377c2",
    "CPLX0-7450[i]": "#1f77b4",
    "export_apparatus_subunit_internal": "#e377c2",
    "CPLX0-7451[j]": "#ff7f0e",
    "FLAGELLAR-MOTOR-COMPLEX[j]": "#2ca02c",
    "rod_internal": "#9467bd",
    "rod_p_ring_internal": "#17becf",
    "EG10321-MONOMER[e]": "#8c564b",
    "CPLX0-7452[j]": "#d62728",
}


# Full species list, matching run_nfsim_population_multigen.py's panel set
# (2026-09-15) -- "__agents__" dropped (always 1 for a followed lineage)
# and "dry_mass" plotted directly instead of a per-agent mean.
# nfsim_scaffold_species/completed_ever overlays are unique to this script
# (not tracked at the population level). Extracted into its own function
# (2026-09-16) so both the combined grid (figure()) and the standalone
# per-panel files (figure_panels()) render from the exact same spec.
def _panel_spec():
    panels = [
        ("Dry mass, this lineage", "dry_mass"),
        ("FlhD, this lineage", "EG10320-MONOMER[c]"),
        ("FlhC, this lineage", "MONOMER0-2488[c]"),
        ("FlhDC complex, this lineage", "CPLX0-3930[c]"),
        ("FliA, this lineage", "EG11355-MONOMER[c]"),
        ("FlgM, this lineage", "G369-MONOMER[c]"),
        ("__overlay_regulatory__", None),
        ("FlgM:FliA complex, this lineage", "FLGM-FLIA-CPLX[c]"),
        ("FliS, this lineage", "EG11388-MONOMER[c]"),
        ("FLIS-FLIC-CPLX (protected FliC), this lineage", "FLIS-FLIC-CPLX[e]"),
        ("__overlay_cascade__", None),
        ("Scaffold species, this lineage", "__scaffold_overlay__"),
        ("C-ring, this lineage", "CPLX0-7450[i]"),
        ("Export apparatus subunit (internal), this lineage", "export_apparatus_subunit_internal"),
        ("Export apparatus, this lineage", "CPLX0-7451[j]"),
        ("Rod (internal), this lineage", "rod_internal"),
        ("Rod+P-ring (internal), this lineage", "rod_p_ring_internal"),
        # "Hook (internal), this lineage" removed 2026-09-16 (Maya's
        # request) -- always reads flat 0 by construction: a completed
        # hook is consumed instantly by the next reaction, so no sampling
        # resolution used in this investigation ever catches it nonzero.
        # Still tracked in _snap()/overlay_cascade, just not its own panel.
        ("Hook-basal-body complete (internal), this lineage", "flagella_internal_cumulative"),
        ("Nascent flagellum, this lineage", "n_nascent"),
        ("Filament length, this lineage", "__filament_progress__"),
        ("Free FliC, this lineage", "EG10321-MONOMER[e]"),
        ("Complete flagella, this lineage", "flag"),
    ]
    # Same cumulative-overlay convention as run_nfsim_population_multigen.py.
    cumulative_overlay = {
        "CPLX0-7450[i]": "cring_cumulative",
        "CPLX0-7451[j]": "export_apparatus_cumulative",
        "rod_internal": "rod_cumulative",
        "rod_p_ring_internal": "rod_p_ring_cumulative",
        "export_apparatus_subunit_internal": "export_apparatus_subunit_cumulative",
        "hook_internal": "hook_internal_cumulative",
    }
    overlay_regulatory = ("EG11355-MONOMER[c]", "G369-MONOMER[c]")
    overlay_cascade = ["CPLX0-7450[i]", "export_apparatus_subunit_internal", "CPLX0-7451[j]",
                        "rod_internal", "rod_p_ring_internal", "hook_internal",
                        "flagella_internal_cumulative", "n_nascent", "CPLX0-7452[j]"]
    return panels, cumulative_overlay, overlay_regulatory, overlay_cascade


def _render_panel(ax, panel_title, key, rows, t, cumulative_overlay, overlay_regulatory, overlay_cascade):
    """Render exactly one panel onto ax. Returns the (possibly overridden,
    for the two overlay panels) display title. Shared by figure() (one
    axis per panel, in a grid) and figure_panels() (one standalone figure
    per panel) -- 2026-09-16, so the two never drift out of sync."""
    if key == "__filament_progress__":
        # Just the raw filament length (2026-09-16, Maya's request) --
        # mean/max were dropped since a single followed lineage never has
        # more than one nascent flagellum at a time, so mean and max were
        # always identical; plotting both was pure redundancy. max_len IS
        # "the filament subunits" here.
        ax.plot(t, _cols(rows, "max_len"), "-o", ms=2, color=COLORS["max_len"])
        ax.axhline(5000, color="gray", ls=":", lw=1, label="target (5,000)")
        ax.set_ylabel("subunits"); ax.legend(fontsize=7)
    elif key == "__scaffold_overlay__":
        ax.plot(t, _cols(rows, "n_scaffold_entries"), "-o", ms=2, color=COLORS["n_scaffold_entries"],
                label="distinct entries")
        ax.plot(t, _cols(rows, "scaffold_total"), "-s", ms=2, color=COLORS["scaffold_total"], alpha=0.5,
                label="total count")
        ax.set_ylabel("count"); ax.legend(fontsize=7)
    elif panel_title == "__overlay_regulatory__":
        for k in overlay_regulatory:
            ax.plot(t, _cols(rows, k), color=COLORS[k], label=TRACK_IDS.get(k, k))
        ax.set_ylabel("count"); ax.legend(fontsize=7)
        panel_title = "FliA / FlgM overlaid, this lineage"
    elif panel_title == "__overlay_cascade__":
        for k in overlay_cascade:
            label = {"export_apparatus_subunit_internal": "export apparatus subunit (internal)",
                     "rod_internal": "rod (internal)",
                     "rod_p_ring_internal": "rod+P-ring (internal)",
                     "hook_internal": "hook (internal)",
                     "flagella_internal_cumulative": "hook-basal-body complete (internal)",
                     "n_nascent": "nascent_flagellum"}.get(k, TRACK_IDS.get(k, k))
            ax.plot(t, _cols(rows, k), "-o", ms=2, color=COLORS[k], label=label)
        ax.set_ylabel("count"); ax.legend(fontsize=6, ncol=2)
        panel_title = "Assembly cascade, overlaid, this lineage"
    elif key in cumulative_overlay:
        ax.plot(t, _cols(rows, key), "-o", ms=2, color=COLORS.get(key, "#333333"), label="live count")
        ax.plot(t, _cols(rows, cumulative_overlay[key]), "-o", ms=2, color="#7f7f7f",
                ls="--", label="cumulative (ever formed)")
        ax.set_ylabel("count"); ax.legend(fontsize=6)
    elif key == "dry_mass":
        ax.plot(t, _cols(rows, key), "-o", ms=2, color=COLORS.get(key, "#333333"))
        ax.set_ylabel("fg")
    else:
        ax.plot(t, _cols(rows, key), "-o", ms=2, color=COLORS.get(key, "#333333"))
        ax.set_ylabel("count")
    ax.set_title(panel_title, fontsize=9)
    return panel_title


def _slug(title):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", title.lower())).strip("_")


def figure(rows, n_gens, chart_number):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    t = _cols(rows, "t_cum") / 60.0
    division_times = _gen_bounds(rows)
    panels, cumulative_overlay, overlay_regulatory, overlay_cascade = _panel_spec()

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).flat
    used_axes = list(axes_flat)[:len(panels)]

    for ax, (panel_title, key) in zip(used_axes, panels):
        _render_panel(ax, panel_title, key, rows, t, cumulative_overlay, overlay_regulatory, overlay_cascade)

    # Division markers: a dotted vertical line on every panel at each real
    # division event, same convention as run_nfsim_population_multigen.py
    # (added there 2026-09-01) -- replaces the old shaded-background-per-axis
    # approach above.
    for ax in used_axes:
        for dt_div in division_times:
            ax.axvline(dt_div, color="#555555", ls=":", lw=1, alpha=0.6, zorder=0)

    for ax in list(np.atleast_1d(axes).flat)[len(panels):]:
        fig.delaxes(ax)
    last_row = (len(panels) - 1) // n_cols
    axes_2d = np.atleast_2d(axes)
    for col in range(n_cols):
        if last_row * n_cols + col < len(panels):
            axes_2d[last_row, col].set_xlabel("time (min)")

    fig.suptitle(f"NFsim-driven single-lineage, {n_gens}-generation test (real Division "
                 f"machinery, pruned to 1 followed agent) — does scaffold/internal state "
                 f"survive real division? (dotted=division)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    charts_dir = f"{STUDY_DIR}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    out = f"{charts_dir}/{chart_number}_nfsim_lineage_multigen_{n_gens}gen.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def figure_panels(rows, n_gens, chart_number):
    """Same panels as figure(), but each saved as its own standalone file
    -- every one gets its own full, visible x-axis (time (min)) and
    legend, since cropping an individual panel out of the shared-x-axis
    grid in figure() loses the axis labels on every row but the bottom
    one. Added 2026-09-16 (Maya's request, for pulling individual panels
    into slides). Saved to charts/{chart_number}_panels/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = _cols(rows, "t_cum") / 60.0
    division_times = _gen_bounds(rows)
    panels, cumulative_overlay, overlay_regulatory, overlay_cascade = _panel_spec()

    panels_dir = f"{STUDY_DIR}/charts/{chart_number}_panels"
    os.makedirs(panels_dir, exist_ok=True)
    written = []
    for panel_title, key in panels:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        display_title = _render_panel(ax, panel_title, key, rows, t,
                                       cumulative_overlay, overlay_regulatory, overlay_cascade)
        for dt_div in division_times:
            ax.axvline(dt_div, color="#555555", ls=":", lw=1, alpha=0.6, zorder=0)
        ax.set_xlabel("time (min)")
        fig.tight_layout()
        out = f"{panels_dir}/{_slug(display_title)}.svg"
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    print(f"wrote {len(written)} individual panels to {panels_dir}")
    return written


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
    charts_dir = f"{STUDY_DIR}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    existing = [int(m.group(1)) for f in os.listdir(charts_dir)
                if (m := re.match(r"^(\d+)_", f))]
    chart_number = max(existing, default=0) + 1
    figure(rows, args.generations, chart_number)
    # Per-metric panel folder generation turned off 2026-09-22 (Maya: just
    # want the combined panel plot, not a folder of individual SVGs per
    # run) -- figure_panels() still defined above if needed again.
    # figure_panels(rows, args.generations, chart_number)
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
