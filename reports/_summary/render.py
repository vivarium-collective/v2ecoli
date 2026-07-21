"""InvestigationSummary dict -> self-contained HTML string. No filesystem reads."""
from __future__ import annotations

import html as _html
from typing import Any

_VERDICTS = ("within_tol", "drift", "mismatch", "ungraded")

_BASE_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;background:var(--bg,#fafafa);color:#1f2937;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:28px 24px 80px}
h1{font-size:1.6em;margin:0 0 4px}
.question{color:var(--gray,#666);font-size:1.05em;margin:0 0 18px}
.rollup{display:flex;gap:10px;margin:0 0 18px;flex-wrap:wrap}
.pill{border-radius:14px;padding:4px 12px;font-weight:600;font-size:0.9em;border:1px solid #0001}
.pill.PASS{background:#d1fae5;color:#065f46}.pill.PARTIAL{background:#fef3c7;color:#92400e}
.pill.FAIL{background:#fee2e2;color:#991b1b}
.dag{font-family:ui-monospace,Menlo,monospace;font-size:0.85em;color:#475569;
 background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;padding:10px 14px;margin:0 0 22px}
table.matrix{border-collapse:collapse;width:100%;margin:0 0 30px;font-size:0.85em}
table.matrix th,table.matrix td{border:1px solid var(--border,#e2e6eb);padding:6px 8px;text-align:center}
table.matrix th.study,table.matrix td.study{text-align:left;font-weight:600;white-space:nowrap}
td.verdict-within_tol{background:#d1fae5}td.verdict-drift{background:#fef3c7}
td.verdict-mismatch{background:#fee2e2}td.verdict-none{background:#f8fafc;color:#cbd5e1}
details.study{background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;margin:0 0 14px;padding:2px 14px}
details.study>summary{cursor:pointer;font-weight:600;padding:10px 0;list-style:none}
.badge{border-radius:10px;padding:1px 8px;font-size:0.78em;margin-left:8px}
.badge.within_tol{background:#d1fae5;color:#065f46}.badge.drift{background:#fef3c7;color:#92400e}
.badge.mismatch{background:#fee2e2;color:#991b1b}.badge.ungraded{background:#eef2f7;color:#475569}
.finding{color:var(--gray,#666);font-weight:400;font-size:0.9em;margin-left:6px}
.card-embed{margin:8px 0 14px;border-top:1px solid var(--border,#e2e6eb);padding-top:10px}
iframe.card-frame{width:100%;border:0}
.missing{color:#b91c1c;font-style:italic;font-size:0.9em}
"""


def _esc(s: Any) -> str:
    return _html.escape(str(s if s is not None else ""))


def _dag(summary: dict) -> str:
    parts = []
    for s in summary["studies"]:
        prereq = s["prerequisites"]
        arrow = f"{', '.join(_esc(p) for p in prereq)} &rarr; " if prereq else ""
        parts.append(f"{arrow}<b>{_esc(s['slug'])}</b>")
    return "<br>".join(parts)


def _matrix_table(summary: dict) -> str:
    cols = summary["matrix"]["columns"]
    if not cols:
        return ""
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for row in summary["matrix"]["rows"]:
        cells = [f'<td class="study">{_esc(row["study"])}</td>']
        for c in cols:
            v = row["cells"].get(c)
            klass = f"verdict-{v}" if v in _VERDICTS else "verdict-none"
            cells.append(f'<td class="{klass}">{_esc(v or "")}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _rollup(summary: dict) -> str:
    r = summary["rollup"]
    order = [("FAIL", r["FAIL"]), ("PARTIAL", r["PARTIAL"]), ("PASS", r["PASS"])]
    pills = [f'<span class="pill {k}">{n} {k}</span>' for k, n in order]
    return f'<div class="rollup">{"".join(pills)}</div>'


_IFRAME_JS = (
    "<script>window.addEventListener('load',function(){"
    "document.querySelectorAll('iframe.card-frame').forEach(function(f){"
    "try{f.style.height=(f.contentDocument.body.scrollHeight+20)+'px';}catch(e){}"
    "});});</script>"
)


def _card_html(card: dict) -> str:
    if card["missing"]:
        return f'<div class="missing">card &ldquo;{_esc(card["name"])}&rdquo; not rendered yet</div>'
    if card["is_full_doc"]:
        srcdoc = _html.escape(card["html"], quote=True)
        return f'<iframe class="card-frame" srcdoc="{srcdoc}"></iframe>'
    return card["html"]  # fragment: inline as-is (inline styles only, no collision)


def _study_section(study: dict) -> str:
    badge_verdict = None
    for c in study["cards"]:
        if c["graded"]:
            badge_verdict = c["overall"]
            break
    badge_cls = badge_verdict if badge_verdict in _VERDICTS else ""
    badge = (
        f'<span class="badge {badge_cls}">{_esc(badge_verdict)}</span>'
        if badge_verdict else ""
    )
    finding = f'<span class="finding">{_esc(study["finding"])}</span>' if study["finding"] else ""
    embeds = []
    for c in study["cards"]:
        open_attr = " open" if c["graded"] else ""
        embeds.append(
            f'<details class="card-embed"{open_attr}>'
            f'<summary>{_esc(c["name"])} card</summary>{_card_html(c)}</details>'
        )
    return (
        f'<details class="study" id="study-{_esc(study["slug"])}" open>'
        f'<summary>{_esc(study["title"])}{badge}{finding}</summary>'
        f'{"".join(embeds)}</details>'
    )


def render(summary: dict, style_css: str = "") -> str:
    head = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_esc(summary['title'])} — report-card summary</title>"
        f"<style>{style_css}\n{_BASE_CSS}</style></head><body><div class=\"wrap\">"
    )
    overview = (
        f"<h1>{_esc(summary['title'])}</h1>"
        f"<p class=\"question\">{_esc(summary['question'])}</p>"
        f"{_rollup(summary)}"
        f"<div class=\"dag\">{_dag(summary)}</div>"
        f"{_matrix_table(summary)}"
    )
    sections = "".join(_study_section(s) for s in summary["studies"])
    return head + overview + sections + _IFRAME_JS + "</div></body></html>"
