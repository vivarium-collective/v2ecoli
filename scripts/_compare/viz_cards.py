"""Write each study's report cards as standalone HTML + a verdict sidecar under
studies/<study>/viz/report_card/, the convention the dashboard auto-discovers
(saved_visualizations) and embeds as a `report_card` test module."""
from __future__ import annotations

import json
import html as _html
from pathlib import Path

_DOC = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>{title}</title><style>body{{font-family:-apple-system,Segoe UI,"
        "Roboto,sans-serif;margin:14px;color:#0f172a}}h3{{margin:14px 0 6px}}"
        "table{{border-collapse:collapse}}td,th{{padding:4px 8px}}</style></head>"
        "<body>{body}</body></html>")


def _card_html(name: str, sections: list) -> str:
    parts = []
    for sec in sections:
        if sec.get("title"):
            parts.append(f"<h3>{_html.escape(str(sec['title']))}</h3>")
        parts.append(sec.get("html", ""))
    return _DOC.format(title=_html.escape(name), body="".join(parts))


def write_report_cards(study_dir, cards: list) -> list:
    """Write <study_dir>/viz/report_card/<name>.{html,verdict.json} per card."""
    out = Path(study_dir) / "viz" / "report_card"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for card in cards:
        name = card["name"]
        hp = out / f"{name}.html"
        hp.write_text(_card_html(name, card.get("sections") or []), encoding="utf-8")
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
