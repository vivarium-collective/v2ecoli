"""Render the study's figures from the completed run.

Every number here is DERIVED from the run's own artifacts — the per-arm
``*_summary.json`` and the emitted parquet, via
``v2ecoli.library.replisome_arrest``. Nothing is hardcoded, so re-running this
after a fresh run produces figures for that run rather than restating a stale
result. If an arm is missing, this fails loudly instead of drawing a partial
figure.

Two figures, matching the two things the study has to establish:

``replisome_margin_at_arrest``
    Observed vs required copies for all six subunit pools at the arresting
    generation. This is the discriminating one: it shows every pool in surplus,
    which refutes subunit depletion as the cause.

``paired_lineage_arrest``
    Dry mass and generation time per generation for both arms. This shows the
    arrest is real and attributable to the gate — same seed, same cache.

Output goes to ``studies/<name>/viz/*.html``, which the workbench
auto-discovers as ``embed_visualizations`` (``study_spec.discover_viz_html_files``)
— no study.yaml edit required. Files rendered after the latest run are served
un-stale; that freshness check is why this writes at render time rather than
committing a snapshot.

Usage::

    python workspace/studies/mechanistic-replisome-arrest/analyses/render_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

ARM_DIRS = {
    "mechanistic": REPO / "out/mechanistic-replisome-arrest/mechanistic",
    "permissive": REPO / "out/mechanistic-replisome-arrest/permissive",
}
CACHE_DIR = REPO / "out/cache"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;padding:10px;background:#fff;
font:14px system-ui,-apple-system,sans-serif}}</style></head>
<body>{body}</body></html>"""


def _render(cls, core, state, out_path: Path, title: str) -> None:
    html = cls({}, core=core).update(state)["html"]
    out_path.write_text(_PAGE.format(title=title, body=html), encoding="utf-8")
    print(f"  wrote {out_path.relative_to(REPO)}  ({len(html)} bytes)")


def main() -> int:
    from v2ecoli.core import build_core
    from v2ecoli.library import replisome_arrest as ra
    # @as_visualization binds the synthesized Visualization subclass to the
    # decorated FUNCTION's name, so these import as update_* and are classes.
    from v2ecoli.visualizations.paired_lineage_arrest import (
        update_paired_lineage_arrest as PairedLineageArrest,
    )
    from v2ecoli.visualizations.replisome_margin_at_arrest import (
        update_replisome_margin_at_arrest as ReplisomeMarginAtArrest,
    )

    for arm, d in ARM_DIRS.items():
        if not d.is_dir():
            print(f"ERROR: {arm} arm not found at {d}", file=sys.stderr)
            return 1

    m = ra.measure(ARM_DIRS["mechanistic"], ARM_DIRS["permissive"], CACHE_DIR)
    summaries = {a: ra.read_summary(d) for a, d in ARM_DIRS.items()}

    # Both arms must share one cache, or the paired comparison the figure
    # asserts is not the comparison that ran. Check rather than trust.
    fps = {a: (s.get("run_config") or {}).get("cache_fingerprint")
           for a, s in summaries.items()}
    if len(set(fps.values())) != 1:
        print(f"ERROR: arms used different caches: {fps}", file=sys.stderr)
        return 1
    print(f"shared cache fingerprint: {next(iter(fps.values()))}")

    viz_dir = STUDY_DIR / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    core = build_core()

    margins = list(m["subunit_margins"].values())
    _render(
        ReplisomeMarginAtArrest, core,
        {
            "pool_labels": [v["label"] for v in margins],
            # margin = min(count - oriC*mult) over the generation, so the
            # requirement the figure draws is recovered from the pair rather
            # than re-derived from a fixed oriC assumption.
            "min_counts": [float(v["min_count"]) for v in margins],
            "requirements": [float(v["min_count"] - v["margin"])
                             for v in margins],
            "arrest_generation": float(m["arrest_generation"]),
        },
        viz_dir / "replisome_margin_at_arrest.html",
        "Replisome subunit margins at the arrest",
    )

    gens = summaries["permissive"]["gens"]
    mech_gens = summaries["mechanistic"]["gens"]
    cap = (summaries["mechanistic"].get("run_config") or {}).get("max_min")
    _render(
        PairedLineageArrest, core,
        {
            "generations": [float(g["gen"]) for g in gens],
            "mechanistic_mass": [float(g["final_dry_mass_fg"]) for g in mech_gens],
            "permissive_mass": [float(g["final_dry_mass_fg"]) for g in gens],
            "mechanistic_tau": [float(g["duration_min"]) for g in mech_gens],
            "permissive_tau": [float(g["duration_min"]) for g in gens],
            "arrest_generation": float(m["arrest_generation"]),
            "duration_cap_min": float(cap) if cap else 0.0,
        },
        viz_dir / "paired_lineage_arrest.html",
        "Paired lineage — mechanistic vs permissive",
    )

    print(f"\narrest at generation {m['arrest_generation']}; "
          f"worst subunit margin {m['worst_subunit_margin']:+d} "
          f"({m['limiting_pool']}); {m['n_pools_graded']} pools graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
