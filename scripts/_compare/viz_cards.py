"""Write each study's report cards as standalone HTML + a verdict sidecar under
studies/<study>/viz/report_card/, the convention the dashboard auto-discovers
(saved_visualizations) and embeds as a `report_card` test module."""
from __future__ import annotations

import json
from pathlib import Path


def write_report_cards(study_dir, cards: list) -> list[Path]:
    """Write <study_dir>/viz/report_card/<name>.{html,verdict.json} per card."""
    out = Path(study_dir) / "viz" / "report_card"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for card in cards:
        name = card["name"]
        hp = out / f"{name}.html"
        hp.write_text(card.get("html") or "", encoding="utf-8")
        written.append(hp)
        verdict = card.get("verdict") or "ungraded"
        vp = out / f"{name}.verdict.json"
        vp.write_text(json.dumps({
            "schema": "report_card_verdict/v1",
            "overall": verdict,
            "groups": {name: {"verdict": verdict, "axes": card.get("axes") or []}},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(vp)
    return written
