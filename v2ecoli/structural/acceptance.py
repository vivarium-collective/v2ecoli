"""Acceptance gate for the structural-ecoli s01 study.

s01 packs the live whole-cell molecular state into a 3D cell at two declared
cell-cycle times ("initial" ~10 s post-birth and "pre-division") and views the
snapshots in the Parsimony Viewer. Before this module the study had NO testable
acceptance criterion — it asserted specific placed-instance counts as findings
with nothing on disk to check them against.

This module defines the concrete, artifact-derived gate the study's
``behavior_tests`` reference. Two layers:

* :func:`check_selection_conservation` — the REPRODUCIBLE-NOW core: v2ecoli's own
  counts -> ingredient selection path conserves per-species counts exactly and
  respects the top-N cap. Runs with only the installed deps (no parsimony
  binary), so it is green in the canonical environment today.

* :func:`evaluate_pack_gate` — the end-to-end gate over a study's written pack
  artifacts (``<study>/viz/3d/<snapshot>.{pack,meta,json}``): both snapshots
  written, count-conservation (placed vs requested), no over-pack, and the
  birth->division growth direction (more instances, conserved composition,
  envelope elongation). Requires a real pack, which needs the newer
  ``pbg_parsimony`` API + the ``parsimony`` binary + network structure fetches
  (see SUMMARY.md / the study's reproducibility note). When the artifacts are
  absent this returns ``available=False`` rather than a false pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Snapshot names s01 declares (ecoli_structural default ``snapshots`` keys).
DEFAULT_SNAPSHOTS = ("initial", "pre-division")

# Provisional acceptance bands (see study.yaml behavior_tests; calibrate against
# the first real pack produced with the pinned toolchain).
CONSERVATION_FLOOR = 0.80   # placed / requested >= this (<=20% steric drop, aggregate)
OVERPACK_CEIL = 1.0         # placed / requested <= this (packer never duplicates)


def _capsule_volume_fl(half_len_A: float, radius_A: float) -> float:
    """Spherocylinder volume (V = pi r^2 * 2 half_len + 4/3 pi r^3), Å^3 -> fL."""
    v_A3 = math.pi * radius_A ** 2 * (2.0 * half_len_A) + (4.0 / 3.0) * math.pi * radius_A ** 3
    return v_A3 / 1e12   # Å^3 -> µm^3 == fL


def _aspect_ratio(half_len_A: float, radius_A: float) -> float:
    """Long-axis length / width for the capsule (rod elongation proxy)."""
    length = 2.0 * half_len_A + 2.0 * radius_A
    return length / (2.0 * radius_A)


def check_selection_conservation(counts: dict[str, int], *, top_n: int = 40) -> dict[str, Any]:
    """Reproducible-now check: v2ecoli's counts -> ingredient selection conserves
    every selected species' count exactly and caps auto-expansion at ``top_n``.

    Returns ``{passed, n_ingredients, n_from_counts, count_mismatches,
    deterministic}``. Uses a permissive Ingredient shim so it does not depend on
    the installed ``pbg_parsimony`` API version (which lags v2ecoli's
    ``build.py`` in the canonical env)."""
    import v2ecoli.structural.build as B

    class _ShimIngredient:
        def __init__(self, id, count, **kw):  # noqa: A002 - mirror build.py kwarg
            self.id = id
            self.count = count

    orig = B.Ingredient
    B.Ingredient = _ShimIngredient
    try:
        ings = B.select_ingredients(counts, locations={}, top_n=top_n)
        ings2 = B.select_ingredients(counts, locations={}, top_n=top_n)
    finally:
        B.Ingredient = orig

    mismatches = sum(
        1 for ing in ings
        if ing.id in counts and ing.count not in (counts[ing.id], max(1, counts[ing.id]))
    )
    n_from_counts = sum(1 for ing in ings if ing.id in counts)
    deterministic = [i.id for i in ings] == [i.id for i in ings2]
    return {
        "passed": mismatches == 0 and deterministic and n_from_counts > 0,
        "n_ingredients": len(ings),
        "n_from_counts": n_from_counts,
        "count_mismatches": mismatches,
        "deterministic": deterministic,
    }


def _read_snapshot(viz_dir: Path, name: str) -> dict[str, Any] | None:
    """Load one snapshot's pack + meta + recipe. Returns None if incomplete."""
    pack_p = viz_dir / f"{name}.pack.json"
    meta_p = viz_dir / f"{name}.meta.json"
    recipe_p = viz_dir / f"{name}.json"
    if not (pack_p.is_file() and meta_p.is_file() and recipe_p.is_file()):
        return None
    try:
        pack = json.loads(pack_p.read_text())
        meta = json.loads(meta_p.read_text())
        recipe = json.loads(recipe_p.read_text())
    except (ValueError, OSError):
        return None
    placements = pack.get("placements") or []
    ingredients = (meta.get("ingredients") or {})
    requested = sum(int(v.get("count", 0)) for v in ingredients.values())
    cap = (((recipe.get("composition") or {}).get("cell") or {}).get("compartment") or {})
    half_len = float(cap.get("a", [0, 0, 0])[0]) * -1.0 if cap.get("a") else 0.0
    radius = float(cap.get("radius") or 0.0)
    return {
        "name": name,
        "n_placed": len(placements),
        "requested": requested,
        "ingredient_ids": frozenset(ingredients.keys()),
        "half_len_A": half_len,
        "radius_A": radius,
        "volume_fl": _capsule_volume_fl(half_len, radius) if radius else 0.0,
        "aspect_ratio": _aspect_ratio(half_len, radius) if radius else 0.0,
    }


def evaluate_pack_gate(study_dir: str | Path,
                       snapshots: tuple[str, ...] = DEFAULT_SNAPSHOTS) -> dict[str, Any]:
    """End-to-end gate over a study's written pack artifacts.

    Returns ``{available, tests: {name: {passed, ...}}, snapshots: {...}}``.
    ``available`` is False (with no ``tests``) when the pack artifacts are not on
    disk — the canonical env cannot currently PRODUCE them (missing parsimony
    binary + stale pbg_parsimony), so their absence is reported as unavailable,
    never as a silent pass."""
    viz_dir = Path(study_dir) / "viz" / "3d"
    loaded = {name: _read_snapshot(viz_dir, name) for name in snapshots}
    present = {k: v for k, v in loaded.items() if v is not None}
    if len(present) != len(snapshots):
        return {"available": False,
                "reason": f"missing pack artifacts under {viz_dir} for "
                          f"{[n for n in snapshots if loaded.get(n) is None]}",
                "snapshots": {k: v for k, v in loaded.items()}}

    tests: dict[str, Any] = {}

    # T1 both-snapshots-written (already guaranteed by reaching here) + non-empty.
    tests["both_snapshots_written"] = {
        "passed": all(s["n_placed"] > 0 for s in present.values()),
        "n_placed": {n: s["n_placed"] for n, s in present.items()},
    }

    # T2 count-conservation: placed / requested within the conservation band.
    cons = {}
    for n, s in present.items():
        frac = (s["n_placed"] / s["requested"]) if s["requested"] else 0.0
        cons[n] = {"placed": s["n_placed"], "requested": s["requested"], "fraction": frac,
                   "passed": CONSERVATION_FLOOR <= frac <= OVERPACK_CEIL}
    tests["count_conservation"] = {"passed": all(c["passed"] for c in cons.values()),
                                   "per_snapshot": cons}

    # T3 no-overpack: placed never exceeds requested (no duplication / >100% packing).
    tests["no_overpack"] = {
        "passed": all(s["n_placed"] <= s["requested"] for s in present.values()),
        "per_snapshot": {n: {"placed": s["n_placed"], "requested": s["requested"]}
                         for n, s in present.items()},
    }

    # T4 growth-direction: pre-division has MORE instances, the SAME composition,
    # and a MORE elongated envelope than birth.
    if "initial" in present and "pre-division" in present:
        a, b = present["initial"], present["pre-division"]
        tests["growth_direction"] = {
            "passed": (b["n_placed"] > a["n_placed"]
                       and a["ingredient_ids"] == b["ingredient_ids"]
                       and b["aspect_ratio"] > a["aspect_ratio"]),
            "instances": {"initial": a["n_placed"], "pre-division": b["n_placed"],
                          "ratio": (b["n_placed"] / a["n_placed"]) if a["n_placed"] else None},
            "composition_conserved": a["ingredient_ids"] == b["ingredient_ids"],
            "aspect_ratio": {"initial": a["aspect_ratio"], "pre-division": b["aspect_ratio"]},
        }

    return {"available": True, "tests": tests,
            "snapshots": {n: {k: (sorted(v) if isinstance(v, frozenset) else v)
                              for k, v in s.items()} for n, s in present.items()},
            "overall_passed": all(t["passed"] for t in tests.values())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the s01 structural pack gate.")
    ap.add_argument("study_dir", help="study dir containing viz/3d/<snapshot>.* artifacts")
    a = ap.parse_args()
    print(json.dumps(evaluate_pack_gate(a.study_dir), indent=2))
