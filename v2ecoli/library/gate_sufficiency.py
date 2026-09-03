"""Measure the replisome-gate-sufficiency sweep (3 arms, 16 runs).

Science core for ``report_cards/gate_sufficiency_card.py``. Same three-way split
as ``replisome_arrest.py``: measurement here, grading in the card, rendering in
the card's HTML.

Reads two artifacts, both of which survive deletion of the raw parquet:

``out/<study>/<arm>/seed<N>/*_summary.json``
    The runner's authoritative per-generation record (``divided`` flag, tau,
    final dry mass). Lives one level above the parquet tree, so it is untouched
    by close-out deletion.

``workspace/studies/<study>/evidence/<arm>__seed<N>.parquet``
    The distilled bundle (see ``analyses/distill_evidence.py``): oriC count, the
    six subunit counts and cell/dry mass per timestep. Its ``--verify`` pass
    checked these margins against the full parquet before the bulk was removed.

What is measured, and why each is the observable it is:

``stall_generation``
    The first generation that RAN and failed to divide. Not "generations
    completed": a lineage that stalls at generation 4 has divided 3 times, and
    conflating the two shifts every stall by one.

``stall_spread``
    max - min stall generation across seeds. This is the seed-consistency
    observable. A single seed's stall generation says nothing about whether the
    stall is a model property.

``subunit_margins``
    ``min(count - demand)`` per pool over the stalling generation, where demand
    is ``6 * n_oriC`` for trimers and ``2 * n_oriC`` for monomers. NEGATIVE means
    that pool was genuinely short. This is the observable the ``==`` operator
    made unmeasurable: under equality every pool failed the comparison whether
    it was short or in surplus.

``ablation_relief``
    Paired per seed: (ablation divided generations) - (mechanistic divided
    generations). Positive means removing DnaG from the gate's requirement list
    extended survival, which is what separates "DnaG is the binding constraint"
    from "DnaG merely looks scarce".

The ablation arm is graded against ITS OWN gate: its config drops
``EG10239-MONOMER[c]`` from the monomer list, so DnaG is not a requirement there
and must not be counted as a shortfall.
"""
from __future__ import annotations

import json
from pathlib import Path

TRIMER_MULT = 6
MONOMER_MULT = 2

ORIC = "listeners__replication_data__number_of_oric"
SUBUNIT_LABELS = {
    "pol_III_core": ("pol III core", TRIMER_MULT),
    "beta_clamp": ("beta clamp", TRIMER_MULT),
    "DnaB_hexamer": ("DnaB hexamer", MONOMER_MULT),
    "DnaG": ("DnaG", MONOMER_MULT),
    "HolB": ("HolB (delta')", MONOMER_MULT),
    "HolA": ("HolA (delta)", MONOMER_MULT),
}
# The ablation arm's gate does not require DnaG.
ABLATED = {"dnag-ablation": {"DnaG"}}


def _summaries(out_root: Path, arm: str) -> dict[int, dict]:
    out = {}
    for d in sorted((out_root / arm).glob("seed*")):
        hits = list(d.glob("*_summary.json"))
        if hits:
            out[int(d.name[4:])] = json.loads(hits[0].read_text())
    return out


def divided_generations(summary: dict) -> int:
    return sum(1 for g in summary.get("gens", []) if g.get("divided"))


def stall_generation(summary: dict) -> "int | None":
    for g in summary.get("gens", []):
        if not g.get("divided"):
            return int(g["gen"])
    return None


def margins(bundle_dir: Path, arm: str, seed: int, generation: int) -> dict:
    """Worst (count - demand) per pool over one generation, from the bundle."""
    import polars as pl
    p = bundle_dir / f"{arm}__seed{seed}.parquet"
    if not p.is_file():
        return {}
    df = pl.read_parquet(p)
    df = df.filter(df["generation"] == generation)
    if df.height == 0:
        return {}
    oric = df[ORIC]
    skip = ABLATED.get(arm, set())
    out = {}
    for col, (label, mult) in SUBUNIT_LABELS.items():
        if col not in df.columns or col in skip:
            continue
        counts = df[col]
        out[label] = {
            "min_count": int(counts.min()),
            "requirement_per_oric": mult,
            "margin": int((counts - oric * mult).min()),
        }
    return out


def measure(out_root, bundle_dir) -> dict:
    """Measure all three arms. Returns the dict the card grades."""
    out_root, bundle_dir = Path(out_root), Path(bundle_dir)
    arms = {}
    for arm in ("mechanistic", "permissive", "dnag-ablation"):
        summ = _summaries(out_root, arm)
        per_seed = {}
        for seed, s in summ.items():
            st = stall_generation(s)
            per_seed[seed] = {
                "divided": divided_generations(s),
                "requested": s.get("generations_requested"),
                "stall_generation": st,
                "final_dry_mass_fg": s["gens"][-1].get("final_dry_mass_fg"),
                "last_tau_min": s["gens"][-1].get("duration_min"),
                "margins": margins(bundle_dir, arm, seed, st) if st else {},
            }
        stalls = [v["stall_generation"] for v in per_seed.values()
                  if v["stall_generation"] is not None]
        divs = [v["divided"] for v in per_seed.values()]
        arms[arm] = {
            "n_seeds": len(per_seed),
            "seeds": per_seed,
            "n_stalled": len(stalls),
            "stall_generations": sorted(stalls),
            "stall_spread": (max(stalls) - min(stalls)) if stalls else None,
            "divided_min": min(divs) if divs else None,
            "divided_max": max(divs) if divs else None,
        }

    # Worst margin across every stalled mechanistic seed, and which pool holds it.
    worst, limiting, per_seed_worst = None, None, {}
    for seed, v in arms["mechanistic"]["seeds"].items():
        if not v["margins"]:
            continue
        lab, info = min(v["margins"].items(), key=lambda kv: kv[1]["margin"])
        per_seed_worst[seed] = {"pool": lab, "margin": info["margin"],
                                "min_count": info["min_count"]}
        if worst is None or info["margin"] < worst:
            worst, limiting = info["margin"], lab

    # Is the same pool worst in EVERY stalled seed? A limiting-resource claim
    # that holds in one seed and not others is not a limiting-resource claim.
    pools = {v["pool"] for v in per_seed_worst.values()}
    limiting_unanimous = len(pools) == 1

    # Paired relief, ablation vs mechanistic at the same seed.
    relief = {}
    for seed, abl in arms["dnag-ablation"]["seeds"].items():
        mech = arms["mechanistic"]["seeds"].get(seed)
        if mech:
            relief[seed] = abl["divided"] - mech["divided"]

    return {
        "arms": arms,
        "worst_margin": worst,
        "limiting_pool": limiting,
        "limiting_pool_unanimous": limiting_unanimous,
        "limiting_pools_seen": sorted(pools),
        "per_seed_worst": per_seed_worst,
        "ablation_relief": relief,
        "ablation_relief_min": min(relief.values()) if relief else None,
        "expected_generations": next(
            (v["requested"] for v in arms["mechanistic"]["seeds"].values()
             if v["requested"]), 12),
    }
