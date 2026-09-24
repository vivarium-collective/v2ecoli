"""FlgM:FliA ordering-hypothesis diagnostic -- single-cell, no division.

Added 2026-09-22, testing the leading untested hypothesis from
MASTER_DOCUMENT.md Section 3.1 (FlgM:FliA equilibrium, Attempt 2 revert):
does running the dedicated exact-solve FlgM:FliA equilibrium Step right
after FlgM secretion (same-tick, "variant A", the Step's original design)
vs. well before secretion (mimicking the shared solver's old position,
"variant B") produce meaningfully different free-FliA/Class-III dynamics
-- or does it just shift the trajectory by about one 2s tick, as the
Step's own docstring predicted analytically but never actually tested?
Per that section's own recommended next step: "a properly-instrumented
single-cell diagnostic with a full per-tick trace of free FliA/FlgM/Y/
Class III override from t=0" -- not another synthetic/reduced model
(the flagella-06 ODE model that would have served this purpose was
deleted, per Maya's direction).

Runs up to THREE single-cell, no-division simulations from the same
seed/start, each just long enough to see the FlgM secretion trajectory
develop after hook-basal-body completion:
  - baseline: current default -- shared ecoli-equilibrium Step only,
    relaxed Kd (2e-7 M), dedicated Step NOT wired in.
  - variant A: dedicated exact-solve Step wired in at its designed
    same-tick-after-secretion position, real Kd (1.8e-10 M). Shared
    solver's own FLGM-FLIA-CPLX_RXN copy is zeroed in-memory (its
    rates_fwd/rates_rev array patched post-build) so it isn't also
    contributing every tick -- see _zero_shared_flgm_flia() below.
  - variant B: same dedicated Step and zeroing, ordering moved to run
    right after complexation instead -- before flis-flic-equilibrium,
    elongation, AND secretion -- mimicking the shared solver's old
    relative position ("well before secretion runs").

Infra note: the dedicated Step's import + step-class registry entry in
ecoli_baseline.py, and its config-getter registration in sim_data.py /
core.py's _CACHE_CONFIG_NAMES, were re-enabled 2026-09-22 (each commented
"reverted 2026-09-01" before that) so this Step is buildable again. The
DEFAULT composite's before_steps list still does not reference the step
name -- the ordering under test here is injected at runtime by this
script (monkeypatching FEATURE_MODULES['flagella_nfsim_complexation']
['before_steps']), not by editing that static list. Revert the
sim_data.py / core.py / ecoli_baseline.py changes (and rebuild the cache)
once this diagnostic is done, per the standing revert-after-test
convention already used once for this exact Step (2026-09-04).

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/studies/flagella-04-complexation-nfsim/run_flgm_flia_ordering_diagnostic.py \
        --seconds-cap 3600 --sample 30 --cache-dir out/cache_full_flit_v12_flisflic_test2
"""
import argparse
import os

import numpy as np

import v2ecoli
from v2ecoli.composites.ecoli_baseline import enable_features, FEATURE_MODULES
from v2ecoli.library.schema import bulk_name_to_idx

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Same standard INIT used throughout this investigation (see
# run_nfsim_lineage_multigen.py): 4 flagella already complete (so
# nascent_flagellum/hbb_count > 0 from t=0, letting FlgM secretion engage
# immediately instead of waiting on fresh assembly), free FliA=500,
# FlgM=800.
INIT = {
    "CPLX0-7452[j]": 4,
    "FLAGELLAR-MOTOR-COMPLEX[j]": 0,
    "EG11355-MONOMER[c]": 500,
    "G369-MONOMER[c]": 800,
}

TRACK_IDS = {
    "EG11355-MONOMER[c]": "free_FliA",
    "G369-MONOMER[c]": "free_FlgM",
    "FLGM-FLIA-CPLX[c]": "FlgM_FliA_complex",
}

# flagella_transcription_regulation.py default -- Y = FliA / (K_fliA +
# FliA), Class III override = Y * basal_prob. Tracking free FliA alone is
# sufficient to reconstruct Y exactly (see that Step's update(), which
# uses nothing else).
K_FLIA = 10.0

BASE_BEFORE_STEPS = [
    'ecoli-flagella-nfsim-complexation',
    'ecoli-flagella-flis-flic-equilibrium',
    'ecoli-flagella-filament-elongation',
    'ecoli-flagella-flgm-secretion',
    'ecoli-flagella-transcription-regulation',
]
FLGM_FLIA_STEP = 'ecoli-flagella-flgm-flia-equilibrium'


def _before_steps(variant):
    """variant: 'baseline' (Step absent), 'A' (same-tick, right after
    secretion -- the Step's own designed position), or 'B' (right after
    complexation -- before flis-flic-equilibrium, elongation, AND
    secretion, mimicking the shared solver's old relative position)."""
    steps = list(BASE_BEFORE_STEPS)
    if variant == 'baseline':
        return steps
    if variant == 'A':
        idx = steps.index('ecoli-flagella-flgm-secretion')
        steps.insert(idx + 1, FLGM_FLIA_STEP)
    elif variant == 'B':
        steps.insert(1, FLGM_FLIA_STEP)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return steps


_RATES_FREEVAR_NAMES = {"_rates_fwd", "_rates_rev", "_rates_fwd_ss", "_rates_rev_ss"}


def _zero_shared_flgm_flia(comp):
    """Patch the shared ecoli-equilibrium Step's own live instance so it no
    longer contributes FLGM-FLIA-CPLX_RXN -- otherwise both it (relaxed
    Kd=2e-7 M) and the dedicated Step (real Kd) would move the same three
    species every tick, confounding the comparison. In-memory only; does
    not touch any cache file on disk.

    This cache's config resolves to the legacy closure-based solver (no
    rates_fn -> instance.fluxesAndMoleculesToSS is a pre-built function
    from equilibrium_ode_solver_factory, not a flat instance.rates_fwd
    array -- confirmed by direct inspection 2026-09-22). The rates arrays
    live as free variables (_rates_fwd/_rates_rev for the kinetic path,
    _rates_fwd_ss/_rates_rev_ss -- a SEPARATE np.where()-derived array --
    for the steady-state path FLGM-FLIA-CPLX_RXN actually uses, since its
    integrate_dt flag is False) inside several nested closures
    (derivatives, derivatives_ss, derivatives_jacobian, their _jit
    variants). Walk all of them and zero every rates array found at this
    reaction's index -- numpy arrays are mutable in place, so this reaches
    every nested function that captured the same array object; redundant
    zeroing of an already-zero array is harmless.
    """
    for path, subtree in comp.step_paths.items():
        if path and path[-1] == "ecoli-equilibrium":
            instance = subtree.get("instance") if isinstance(subtree, dict) else None
            if instance is None:
                print("  WARNING: ecoli-equilibrium subtree has no instance")
                return
            rxn_ids = list(instance.reaction_ids)
            if "FLGM-FLIA-CPLX_RXN" not in rxn_ids:
                print("  WARNING: FLGM-FLIA-CPLX_RXN not in shared solver's reaction_ids")
                return
            idx = rxn_ids.index("FLGM-FLIA-CPLX_RXN")

            f = instance.fluxesAndMoleculesToSS
            zeroed = []
            for cell, name in zip(f.__closure__, f.__code__.co_freevars):
                nested = cell.cell_contents
                if not (callable(nested) and getattr(nested, "__closure__", None)):
                    continue
                for c2, n2 in zip(nested.__closure__, nested.__code__.co_freevars):
                    if n2 in _RATES_FREEVAR_NAMES:
                        arr = c2.cell_contents
                        if arr[idx] != 0.0:
                            arr[idx] = 0.0
                            zeroed.append(f"{name}.{n2}")
            print(f"  zeroed shared-solver FLGM-FLIA-CPLX_RXN (reaction idx {idx}) in: "
                  f"{zeroed if zeroed else '(already zero everywhere)'}")
            return
    print("  WARNING: could not find ecoli-equilibrium step instance")


def _arr(s):
    return s["_data"] if isinstance(s, dict) and "_data" in s else s


def _snap(comp, agent_id, idx, t_cum):
    cell = comp.state["agents"][agent_id]
    b = _arr(cell["bulk"])
    row = {"t": t_cum}
    for real_id, label in TRACK_IDS.items():
        row[label] = int(b["count"][idx[real_id]])
    fliA = row["free_FliA"]
    row["Y"] = fliA / (K_FLIA + fliA)
    return row


def run_variant(variant, seconds_cap, sample, seed, cache_dir):
    print(f"\n=== variant '{variant}' ===")
    feat = FEATURE_MODULES['flagella_nfsim_complexation']
    feat['before_steps'] = _before_steps(variant)
    print(f"  before_steps: {feat['before_steps']}")

    enable_features('flagella_nfsim_complexation')
    comp = v2ecoli.build_composite("ecoli_baseline", cache_dir=cache_dir, seed=seed)
    enable_features()

    if variant != 'baseline':
        _zero_shared_flgm_flia(comp)

    bulk = _arr(comp.state["agents"]["0"]["bulk"])
    bids = bulk["id"]
    for name, val in INIT.items():
        bulk["count"][bulk_name_to_idx(name, bids)] = val
    idx = {real_id: bulk_name_to_idx(real_id, bids) for real_id in TRACK_IDS}

    rows = [_snap(comp, "0", idx, 0.0)]
    total = 0.0
    while total < seconds_cap:
        chunk = min(sample, seconds_cap - total)
        comp.run(chunk)
        total += chunk
        if "0" not in comp.state.get("agents", {}):
            # Real division occurred (mass-based, independent of flagella
            # count -- this script deliberately does not handle it, single-
            # cell/no-division is the whole point of isolating the ordering
            # effect). Stop cleanly with whatever was collected rather than
            # crashing -- confirmed 2026-09-22 this happens around t~2400s
            # in this cache/condition.
            print(f"  agent '0' divided at t={total:.0f}s -- stopping here")
            break
        row = _snap(comp, "0", idx, total)
        rows.append(row)
        if int(total) % 600 < sample:
            print(f"    t={total:.0f}s  FliA={row['free_FliA']}  FlgM={row['free_FlgM']}  "
                  f"complex={row['FlgM_FliA_complex']}  Y={row['Y']:.4f}")
    return rows


def _compare_and_plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    labels = {"baseline": "baseline (relaxed Kd, shared solver)",
              "A": "variant A (same-tick, after secretion)",
              "B": "variant B (before secretion)"}
    colors = {"baseline": "#888888", "A": "#1f77b4", "B": "#d62728"}

    for name, rows in results.items():
        t = np.array([r["t"] for r in rows]) / 60.0
        axes[0].plot(t, [r["free_FliA"] for r in rows], label=labels[name], color=colors[name])
        axes[1].plot(t, [r["free_FlgM"] for r in rows], label=labels[name], color=colors[name])
        axes[2].plot(t, [r["Y"] for r in rows], label=labels[name], color=colors[name])

    axes[0].set_ylabel("free FliA (molecules)")
    axes[1].set_ylabel("free FlgM (molecules)")
    axes[2].set_ylabel("Y = FliA/(K+FliA)\n(Class III override factor)")
    axes[2].set_xlabel("time (min)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("FlgM:FliA ordering-hypothesis diagnostic (single cell, no division)")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nwrote {out_path}")

    if "A" in results and "B" in results:
        a_Y = np.array([r["Y"] for r in results["A"]])
        b_Y = np.array([r["Y"] for r in results["B"]])
        n = min(len(a_Y), len(b_Y))
        max_abs_diff = np.max(np.abs(a_Y[:n] - b_Y[:n]))
        print(f"\nA vs B: max |Y_A - Y_B| over the run = {max_abs_diff:.5f}")
        print("(if this is comparable to A/B's own baseline-relative shift, "
              "ordering plausibly matters; if it's near-zero, ordering is "
              "ruled out as the cause of the 2026-09-01 'not correct' dynamics)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-cap", type=int, default=3600)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/cache_full_flit_v12_flisflic_test2")
    ap.add_argument("--variants", default="baseline,A,B",
                     help="comma-separated subset of baseline,A,B")
    args = ap.parse_args()

    variants = args.variants.split(",")
    results = {}
    for variant in variants:
        results[variant] = run_variant(
            variant, args.seconds_cap, args.sample, args.seed, args.cache_dir)

    charts_dir = f"{STUDY_DIR}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    import re
    existing = [int(m.group(1)) for f in os.listdir(charts_dir)
                if (m := re.match(r"^(\d+)_", f))]
    chart_number = max(existing, default=0) + 1
    out_path = f"{charts_dir}/{chart_number}_flgm_flia_ordering_diagnostic.svg"
    _compare_and_plot(results, out_path)


if __name__ == "__main__":
    main()
