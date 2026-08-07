"""InvestigationSummary dict -> self-contained HTML string. No filesystem reads."""
from __future__ import annotations

import html as _html
import json as _json
from typing import Any

from scripts._compare import theme

_VERDICTS = ("within_tol", "drift", "mismatch", "ungraded")

# Verdict -> (glyph, human label). Colors live in theme.STATUS (the shared,
# dataviz-validated palette); "ungraded" isn't a graded outcome so it's pinned
# here alongside it, same convention as scripts/_compare/report.py's
# "not_compared". Every verdict is rendered as glyph+label together, never by
# color alone.
_GLYPH = {k: v["glyph"] for k, v in theme.STATUS.items()}
_GLYPH["ungraded"] = "–"  # –
_LABEL = {k: v["label"] for k, v in theme.STATUS.items()}
_LABEL["ungraded"] = "Ungraded"

# study-level rollup (PASS/PARTIAL/FAIL, from materialize.py's _OUTCOME map)
# reuses the same three graded colors as the per-axis verdicts they're derived
# from: within_tol -> PASS, drift -> PARTIAL, mismatch -> FAIL.
_ROLLUP_STATUS = {"PASS": "within_tol", "PARTIAL": "drift", "FAIL": "mismatch"}

# Theme-sourced CSS custom properties: light tokens as the default `:root`,
# dark tokens gated behind `prefers-color-scheme` so the summary follows the
# reader's OS/browser theme without any JS (same wiring as report.py's
# _THEME_CSS).
_THEME_CSS = (
    theme.css_vars("light") + "\n"
    "@media (prefers-color-scheme: dark) {\n"
    + theme.css_vars("dark") + "\n"
    "}\n"
)

_BASE_CSS = _THEME_CSS + """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;background:var(--bg,#fafafa);color:#1f2937;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:28px 24px 80px}
h1{font-size:1.6em;margin:0 0 4px}
.question{color:var(--gray,#666);font-size:1.05em;margin:0 0 18px}
.rollup{display:flex;gap:10px;margin:0 0 18px;flex-wrap:wrap}
.pill{border-radius:14px;padding:4px 12px;font-weight:600;font-size:0.9em;border:1px solid #0001;color:#fff}
.pill.PASS{background:var(--status-within-tol)}.pill.PARTIAL{background:var(--status-drift)}
.pill.FAIL{background:var(--status-mismatch)}
.dag{font-family:ui-monospace,Menlo,monospace;font-size:0.85em;color:#475569;
 background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;padding:10px 14px;margin:0 0 22px}
table.matrix{border-collapse:collapse;width:100%;margin:0 0 30px;font-size:0.85em}
table.matrix th,table.matrix td{border:1px solid var(--border,#e2e6eb);padding:6px 8px;text-align:center}
table.matrix th.study,table.matrix td.study{text-align:left;font-weight:600;white-space:nowrap}
td.verdict-within_tol{background:var(--status-within-tol);color:#fff}
td.verdict-drift{background:var(--status-drift);color:#fff}
td.verdict-mismatch{background:var(--status-mismatch);color:#fff}
td.verdict-none{background:var(--card,#f8fafc);color:var(--muted,#cbd5e1)}
details.study{background:#fff;border:1px solid var(--border,#e2e6eb);border-radius:8px;margin:0 0 14px;padding:2px 14px}
details.study>summary{cursor:pointer;font-weight:600;padding:10px 0;list-style:none}
.badge{border-radius:10px;padding:1px 8px;font-size:0.78em;margin-left:8px;color:#fff}
.badge.within_tol{background:var(--status-within-tol)}.badge.drift{background:var(--status-drift)}
.badge.mismatch{background:var(--status-mismatch)}.badge.ungraded{background:var(--muted,#475569)}
.finding{color:var(--gray,#666);font-weight:400;font-size:0.9em;margin-left:6px}
.card-embed{margin:8px 0 14px;border-top:1px solid var(--border,#e2e6eb);padding-top:10px}
iframe.card-frame{width:100%;border:0}
.missing{color:#b91c1c;font-style:italic;font-size:0.9em}
nav.topnav{position:sticky;top:0;z-index:10;display:flex;gap:4px;flex-wrap:wrap;
 align-items:center;padding:8px 24px;margin:0 0 0;background:var(--panel,#fff);
 border-bottom:1px solid var(--border,#e2e6eb);box-shadow:0 1px 3px #0000000f;overflow-x:auto}
nav.topnav a{text-decoration:none;font-size:0.82em;color:#475569;padding:3px 9px;
 border-radius:12px;border:1px solid var(--border,#e2e6eb);white-space:nowrap}
nav.topnav a:hover{background:var(--bg,#f1f5f9);color:#111}
nav.topnav a.home{font-weight:700;color:#111;border-color:transparent}
details.config-json{margin:8px 0 6px;border-top:1px solid var(--border,#e2e6eb);padding-top:10px}
details.config-json>summary{cursor:pointer;font-weight:600;font-size:0.9em;color:#374151}
pre.config{background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:10px;
 font-size:12px;line-height:1.4;max-height:420px;overflow:auto;white-space:pre;margin:8px 0 0}
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
            text = f"{_GLYPH[v]} {_esc(_LABEL[v])}" if v in _GLYPH else ""
            cells.append(f'<td class="{klass}">{text}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="matrix"><thead><tr><th class="study">study</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _rollup(summary: dict) -> str:
    r = summary["rollup"]
    order = [("FAIL", r["FAIL"]), ("PARTIAL", r["PARTIAL"]), ("PASS", r["PASS"])]
    pills = [
        f'<span class="pill {k}">{_GLYPH[_ROLLUP_STATUS[k]]} {n} {_esc(k)}</span>'
        for k, n in order
    ]
    return f'<div class="rollup">{"".join(pills)}</div>'


_IFRAME_JS = (
    "<script>window.addEventListener('load',function(){"
    "document.querySelectorAll('iframe.card-frame').forEach(function(f){"
    "try{f.style.height=(f.contentDocument.body.scrollHeight+20)+'px';}catch(e){}"
    "});});</script>"
)


def _topnav(summary: dict) -> str:
    """Sticky top menu: Overview + one link per study, anchoring to each
    study section's id."""
    links = [f'<a class="home" href="#top">{_esc(summary["slug"])}</a>']
    for s in summary["studies"]:
        links.append(f'<a href="#study-{_esc(s["slug"])}">{_esc(s["slug"])}</a>')
    return f'<nav class="topnav">{"".join(links)}</nav>'


def _config_block(study: dict) -> str:
    """Render the study's real baseline config as a JSON block (replaces the
    config report card)."""
    cfg = study.get("config_json") or {}
    if not cfg:
        return ""
    pretty = _json.dumps(cfg, indent=2, sort_keys=True)
    return (
        '<details class="config-json" open><summary>config (JSON)</summary>'
        f'<pre class="config">{_esc(pretty)}</pre></details>'
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
    # glyph+label only for a known/whitelisted verdict; an unrecognized value
    # (e.g. attacker-controlled input) still renders as escaped text, just
    # without a glyph, and never lands in the class attribute unescaped.
    badge_text = (
        f"{_GLYPH[badge_verdict]} {_esc(_LABEL[badge_verdict])}"
        if badge_verdict in _GLYPH else _esc(badge_verdict)
    )
    badge = (
        f'<span class="badge {badge_cls}">{badge_text}</span>'
        if badge_verdict else ""
    )
    finding = f'<span class="finding">{_esc(study["finding"])}</span>' if study["finding"] else ""
    # The config report card is replaced by the actual baseline config JSON.
    embeds = [_config_block(study)]
    for c in study["cards"]:
        if c["name"] == "config":
            continue
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
        f"<style>{style_css}\n{_BASE_CSS}</style></head><body>"
        f"{_topnav(summary)}<div class=\"wrap\" id=\"top\">"
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
