#!/usr/bin/env python
"""Reduce the route-B traces to the numbers the bands grade.

⚠ THE STATISTIC IS THE MEAN, AND THAT IS NOT A STYLE CHOICE. trp-operon repression
in this model is ON/OFF, not graded: TrpR is present at only 42-80 copies, so a
single complex binding and unbinding switches the operon between ~8e-12 and
~2.6e-4. A MEDIAN lands on whichever side of that split it happens to fall — the
same WT/KO pair yields 5.2e7, 1.98 and 4.5e7 across three generations of one run.
The MEAN is stable AND is the biologically right quantity, because transcription
integrates over time: mean = (fraction of time active) x (level when active).

Reads data/route_b_<arm>.json; writes data/route_b_summary.json. No simulation.

    python workspace/studies/genotype-06-trpr-regulon/sims/analyze_route_b.py
"""
from __future__ import annotations

import json
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA = STUDY_DIR / "data"
OUT = DATA / "route_b_summary.json"

# Four-plus orders below the derepressed level; the gap is ~7 orders, so the
# split is not sensitive to where in it this sits.
REPRESSED = 1e-8

PAIRS = {
    "aa_replete": ("wt_withaa", "trpR_withaa",
                   "TrpR is declared ACTIVE in this condition (condition_defs.tsv)"),
    "basal": ("wt", "trpR",
              "TrpR is declared INACTIVE in this condition"),
    "basal_double": ("wt", "trpRtnaA",
                     "trpR + tnaA, basal; same inactive-TrpR caveat"),
}


def window_values(trace: list[dict], gen_steps: int) -> list[float]:
    """Second half of each generation — the window the bands declare."""
    return [r["trp_tu_synth_prob"] for r in trace
            if (r["step"] - 1) % gen_steps > gen_steps / 2]


def arm_stats(arm: str) -> dict | None:
    path = DATA / f"route_b_{arm}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    v = window_values(d["trace"], d["gen_steps"])
    if not v:
        return None
    active = [x for x in v if x >= REPRESSED]
    trpr = [r["trpR_monomer"] for r in d["trace"] if r["trpR_monomer"] is not None]
    return {
        "arm": arm, "n_samples": len(v),
        "time_mean": sum(v) / len(v),
        "fraction_repressed": 1 - len(active) / len(v),
        "level_when_active": (sum(active) / len(active)) if active else None,
        "trpR_monomer_min": min(trpr) if trpr else None,
        "trpR_monomer_max": max(trpr) if trpr else None,
        "generations": d["generations"], "wall_seconds": d["wall_seconds"],
    }


def main() -> int:
    out = {"statistic": "time_mean of trp TU (TU00067) synthesis probability, "
                        "second-half window; ratio KO/WT",
           "repressed_threshold": REPRESSED, "pairs": {}, "arms": {}}
    for key, (wt, ko, note) in PAIRS.items():
        w, k = arm_stats(wt), arm_stats(ko)
        if w is None or k is None:
            out["pairs"][key] = {"status": "missing arm", "wt": wt, "ko": ko}
            continue
        out["arms"][wt], out["arms"][ko] = w, k
        out["pairs"][key] = {
            "note": note, "wt": wt, "ko": ko,
            "ratio_ko_over_wt": k["time_mean"] / w["time_mean"],
        }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}")
    for key, r in out["pairs"].items():
        if "ratio_ko_over_wt" not in r:
            print(f"  {key:<14} {r['status']}"); continue
        # pairs carry arm NAMES; the stats blobs live once in out["arms"].
        w, k = out["arms"][r["wt"]], out["arms"][r["ko"]]
        print(f"  {key:<14} KO/WT = {r['ratio_ko_over_wt']:.2f}x   "
              f"(WT repressed {w['fraction_repressed']:.0%} of the time, "
              f"KO {k['fraction_repressed']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
