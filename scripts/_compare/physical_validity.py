"""Decide whether a lineage cell_mass trajectory is physically valid.

A valid whole-cell basal run grows cell_mass ~2x within a generation and then
halves at division. The pre-fix wrapper bug made cell_mass explode ~18x in one
generation (fail-open partition gate -> evolvers re-applied bulk deltas every
tick). This module turns that distinction into a hard PASS/FAIL gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def segment_generations(cell_mass: np.ndarray, *, drop_frac: float = 0.6) -> list[tuple[int, int]]:
    cm = np.asarray(cell_mass, dtype=float)
    if cm.size == 0:
        return []
    bounds = [0]
    for i in range(cm.size - 1):
        if cm[i] > 0 and cm[i + 1] < drop_frac * cm[i]:
            bounds.append(i + 1)
    bounds.append(cm.size)
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


@dataclass
class Verdict:
    physical: bool
    generations_reached: int
    divisions_detected: int
    per_gen_ratios: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def assess_physical(
    cell_mass: np.ndarray,
    *,
    min_generations: int = 2,
    doubling_band: tuple[float, float] = (1.5, 3.5),
) -> Verdict:
    cm = np.asarray(cell_mass, dtype=float)
    segs = segment_generations(cm)
    divisions = max(len(segs) - 1, 0)
    # complete generations are all but the last (the last may be mid-cycle, no division yet)
    complete = segs[:-1] if len(segs) >= 1 else []
    ratios: list[float] = []
    reasons: list[str] = []
    lo, hi = doubling_band
    for (s, e) in complete:
        if e - s < 2 or cm[s] <= 0:
            continue
        r = float(cm[e - 1] / cm[s])
        ratios.append(r)
        if not (lo <= r <= hi):
            reasons.append(f"generation [{s}:{e}] growth ratio {r:.2f} outside physical band {doubling_band}")
    if divisions < min_generations - 1:
        reasons.append(
            f"only {divisions} division(s) detected; require >= {min_generations - 1} "
            f"for {min_generations} generations"
        )
    physical = len(reasons) == 0 and len(ratios) >= max(min_generations - 1, 1)
    if len(ratios) < max(min_generations - 1, 1) and not reasons:
        reasons.append("no complete generation with a measurable growth ratio")
        physical = False
    return Verdict(
        physical=physical,
        generations_reached=len(segs),
        divisions_detected=divisions,
        per_gen_ratios=ratios,
        reasons=reasons,
    )
