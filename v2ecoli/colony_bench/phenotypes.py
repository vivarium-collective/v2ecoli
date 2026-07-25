"""Tier-agnostic phenotype extraction from a sampled colony trajectory.

Reconstructs lineage + division events from cell-id appearance/disappearance
and computes a common phenotype panel. Pure over its input trajectory so it is
unit-testable with synthetic data and reused by every study/tier.
"""
from __future__ import annotations
from typing import Any
import math

Trajectory = list[dict[str, Any]]


def phenotype_extractor(trajectory: Trajectory) -> dict[str, Any]:
    frames = sorted(trajectory, key=lambda f: f["time"])
    birth = {}          # cell_id -> (time, length)
    last_seen = {}      # cell_id -> (time, length, mass)
    length_track = {}   # cell_id -> list[(time, length)] for growth-rate fit
    lineage = {}        # daughter_id -> mother_id
    size_len, size_mass = [], []
    added_length, interdiv = [], []
    exch_sums, exch_n = {}, 0

    prev_ids: set = set()
    for frame in frames:
        t = frame["time"]
        cells = frame["cells"]
        ids = set(cells)
        for cid, cell in cells.items():
            if cid not in birth:
                birth[cid] = (t, float(cell["length"]))
            last_seen[cid] = (t, float(cell["length"]), float(cell["mass"]))
            length_track.setdefault(cid, []).append((t, float(cell["length"])))
            if "exchange" in cell and isinstance(cell["exchange"], dict):
                for mol, val in cell["exchange"].items():
                    exch_sums[mol] = exch_sums.get(mol, 0.0) + float(val)
                exch_n += 1
        gone = prev_ids - ids
        new = ids - prev_ids
        if gone and new:
            for mother in gone:
                _mt, mlen, mmass = last_seen[mother]
                size_len.append(mlen)
                size_mass.append(mmass)
                b_t, b_len = birth[mother]
                added_length.append({"birth_length": b_len,
                                      "delta_length": mlen - b_len})
                # Interdivision time = time of the CURRENT (division) frame
                # minus the mother's birth time — NOT the mother's
                # last-seen time (which can lag the division frame when the
                # mother wasn't sampled on every intermediate frame).
                interdiv.append(t - b_t)
                for daughter in new:
                    lineage[daughter] = mother
        prev_ids = ids

    growth_rate = []
    for cid, series in length_track.items():
        if len(series) >= 3:
            ts = [p[0] for p in series]
            ys = [math.log(p[1]) for p in series if p[1] > 0]
            if len(ys) == len(ts) and len(ts) >= 3:
                growth_rate.append(_slope(ts, ys))

    exchange = ({m: s / exch_n for m, s in exch_sums.items()}
                if exch_n else None)

    return {
        "n_division_events": len(size_len),
        "size_at_division": {"length": size_len, "mass": size_mass},
        "added_length": added_length,
        "interdivision_time": interdiv,
        "growth_rate": growth_rate,
        "exchange": exchange,
        "lineage": lineage,
    }


def _slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0
