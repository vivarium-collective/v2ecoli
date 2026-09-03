"""Render the study's figure from the same measurement the card grades."""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;padding:10px;background:#fff;
font:14px system-ui,-apple-system,sans-serif}}</style></head>
<body>{body}</body></html>"""


def main() -> int:
    import yaml
    from v2ecoli.core import build_core
    from v2ecoli.library import dnag_deficit as dd
    from v2ecoli.visualizations.dnag_deficit_cascade import (
        update_dnag_deficit_cascade as Cascade)

    spec = yaml.safe_load((STUDY_DIR / "study.yaml").read_text(encoding="utf-8"))
    cfg = spec["report_card_refs"]["dnag_deficit"]

    def _p(k):
        v = cfg[k]
        return v if str(v).startswith("/") else str(REPO / v)

    m = dd.measure(_p("cache_dir"), _p("bundle_glob"), _p("proteome_script"),
                   fixture=_p("fixture"))
    chain = m["chain"]
    steps = [(c["step"], c["percentile"]) for k, c in chain.items() if k != "operon"]

    core = build_core()
    viz = STUDY_DIR / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    html = Cascade({}, core=core).update({
        "literature_median": float(m["literature"]["median"]),
        "parca_fitted": float((m["parca_expected"] or {})["count"]),
        "simulated_mean": float(m["simulated"]["mean"]),
        "step_labels": [s for s, _ in steps],
        "step_percentiles": [float(p) for _, p in steps],
        "frac_zero": float(m["simulated"]["frac_zero"]),
    })["html"]
    out = viz / "dnag_deficit_cascade.html"
    out.write_text(_PAGE.format(title="DnaG deficit cascade", body=html),
                   encoding="utf-8")
    print(f"  wrote {out.relative_to(REPO)} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
