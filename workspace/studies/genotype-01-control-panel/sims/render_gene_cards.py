#!/usr/bin/env python
"""Render a full build-integrity card per panel gene into viz/report_card/.

The panel table (charts/panel_outcomes.svg) summarizes one verdict per gene;
these are the click-through artifacts behind it — the complete
genotype_build_integrity card (every axis, measured values, details) rendered
per gene, exactly as genotype-00 renders it for its single exemplar.

Fit axes stay ungraded here by design: build outcomes (including the four
step-3 crashes) are the panel run's result, recorded in runs[]/panel_summary;
the cards carry the STRUCTURAL granularity. Resolves the WT knowledge base
once and reuses it across all 8 genes (~2-3 min total).

Run from the study directory:
    python sims/render_gene_cards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR / "sims"))
from run_panel import PANEL, OUT  # noqa: E402


def main() -> int:
    from v2ecoli.library import genotype_build as gb
    from v2ecoli.library.report_card import grade_card, render_html, verdict_json

    out_dir = STUDY_DIR / "viz" / "report_card"
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = gb.resolve_raw_data(None)

    for sym, gid, cls, _wiring in PANEL:
        gdir = OUT / sym
        manifest, genotype_id, spans = gb.make_knockout_bundle([gid], gdir)
        ko = gb.resolve_raw_data(manifest)
        card = gb.measure_structure(wt, ko, [gid], spans)
        card["fit"] = gb.measure_fit(None)
        card["genotype"] = {"gene_ids": [gid], "genotype_id": genotype_id,
                            "manifest": "bundle/reference_bundle.tsv"}
        deleted_bp = sum(r - l + 1 for l, r in spans.values())
        reference = gb._reference([gid], None, deleted_bp)
        reference["title"] = (f"Genotype build integrity — {sym} ({gid}), "
                              f"class {cls}")
        report = grade_card(card, reference)
        vjson = verdict_json(
            report,
            model_ref=reference["stimulus"]["measured_model"],
            reference_model=reference["stimulus"]["reference_model"])
        vjson["title"] = reference["title"]
        vjson["genotype"] = card["genotype"]
        html = render_html(card, reference,
                           model_ref=reference["stimulus"]["measured_model"])
        base = out_dir / f"genotype_build_integrity__{sym}"
        base.with_suffix(".html").write_text(html)
        import json
        base.with_suffix(".verdict.json").write_text(
            json.dumps(vjson, indent=2, sort_keys=True) + "\n")
        print(f"{sym}: {vjson.get('overall')} -> {base.with_suffix('.html').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
