"""Self-contained two-column HTML report renderer.

Renders a list of comparison *sections* (vEcoli left, v2ecoli right) into a
single self-contained HTML page: a summary banner with per-verdict counts, a
legend, a sticky section nav, and one card per section. Rows may carry an
optional ``group`` key to cluster them under sub-headings within a section,
and an optional ``reason``/``max_rel`` for extra context. The output embeds
no external assets (no http/https links), so the file is portable.
"""
from __future__ import annotations

import html as _html
import json as _json
from typing import Any

# verdict -> (human label, glyph, css token). The css token is also used as
# the row class `verdict-<token>` (kept stable for tests + styling).
_VERDICTS = {
    "within_tol": ("within tolerance", "✓", "within_tol"),
    "drift": ("drift", "≈", "drift"),
    "mismatch": ("mismatch", "✗", "mismatch"),
    "not_compared": ("not compared", "–", "not_compared"),
}
_VERDICT_ORDER = ["within_tol", "drift", "mismatch", "not_compared"]

_CSS = """
:root {
  --green:#2e7d32; --amber:#ef6c00; --red:#c62828; --grey:#757575;
  --bg:#f6f7f9; --card:#fff; --ink:#1a1d21; --muted:#6b7280; --line:#e5e7eb;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin:0; background:var(--bg); color:var(--ink); line-height:1.45; }
header.top { background:linear-gradient(135deg,#1f2937,#111827); color:#fff;
  padding:22px 28px; }
header.top h1 { margin:0; font-size:20px; font-weight:650; letter-spacing:.2px; }
header.top .sub { margin-top:4px; color:#cbd5e1; font-size:13px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:14px; }
.legend .item { display:flex; align-items:center; gap:6px; font-size:12px;
  color:#e5e7eb; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot.within_tol{background:var(--green)} .dot.drift{background:var(--amber)}
.dot.mismatch{background:var(--red)} .dot.not_compared{background:var(--grey)}
nav.sticky { position:sticky; top:0; z-index:5; background:rgba(255,255,255,.95);
  backdrop-filter:saturate(1.4) blur(6px); border-bottom:1px solid var(--line);
  padding:10px 28px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
nav.sticky a { text-decoration:none; color:var(--ink); font-size:13px;
  padding:5px 10px; border-radius:999px; border:1px solid var(--line);
  display:inline-flex; align-items:center; gap:7px; }
nav.sticky a:hover { background:var(--bg); }
.minicount { font-size:11px; color:var(--muted); }
main { padding:24px 28px 64px; max-width:1180px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
  margin:0 0 22px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card > .head { padding:16px 18px; border-bottom:1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  flex-wrap:wrap; }
.card > .head h2 { margin:0; font-size:16px; font-weight:620; }
.chips { display:flex; gap:7px; flex-wrap:wrap; }
.chip { font-size:12px; font-weight:600; padding:3px 9px; border-radius:999px;
  color:#fff; display:inline-flex; gap:5px; align-items:center; }
.chip.within_tol{background:var(--green)} .chip.drift{background:var(--amber)}
.chip.mismatch{background:var(--red)} .chip.not_compared{background:var(--grey)}
.chip.zero{ background:#eceff1; color:#90a4ae; }
table { border-collapse:collapse; width:100%; }
thead th { text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.5px; color:var(--muted); padding:9px 18px;
  border-bottom:1px solid var(--line); background:#fafbfc; }
tbody td { padding:9px 18px; border-bottom:1px solid #f1f3f5;
  vertical-align:top; font-size:13px; }
tbody tr:hover { background:#fbfcfe; }
.grouprow td { background:#f3f4f6; font-weight:650; font-size:11px;
  text-transform:uppercase; letter-spacing:.6px; color:var(--muted); }
.metric { font-weight:560; }
.val { font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
  font-size:12px; color:#374151; white-space:pre-wrap; word-break:break-word; }
.col-left, .col-right { width:34%; }
.badge { font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:6px;
  color:#fff; margin-left:8px; white-space:nowrap; }
.verdict-within_tol .badge{background:var(--green)}
.verdict-drift .badge{background:var(--amber)}
.verdict-mismatch .badge{background:var(--red)}
.verdict-not_compared .badge{background:var(--grey)}
.verdict-within_tol td:first-child{box-shadow:inset 3px 0 0 var(--green)}
.verdict-drift td:first-child{box-shadow:inset 3px 0 0 var(--amber)}
.verdict-mismatch td:first-child{box-shadow:inset 3px 0 0 var(--red)}
.verdict-not_compared td:first-child{box-shadow:inset 3px 0 0 var(--grey)}
.reason { color:var(--muted); font-size:11.5px; margin-top:3px; }
.metersmall { font-size:11px; color:var(--muted); margin-top:2px; }
footer { color:var(--muted); font-size:12px; padding:0 28px 40px;
  max-width:1180px; margin:0 auto; }
/* section descriptions */
.sectiondesc { margin:0; padding:12px 18px 0; color:var(--muted); font-size:12.5px; }
.card.content .body { padding:14px 18px 18px; }
/* config as navigable JSON */
.cfgjson { margin:12px 0 0; padding:14px 16px; background:#0f172a; color:#e2e8f0;
  border-radius:10px; font-family:"SF Mono",ui-monospace,Menlo,monospace;
  font-size:12px; line-height:1.5; overflow:auto; max-height:520px;
  white-space:pre; }
details.cfgdrop { margin-top:10px; }
details.cfgdrop > summary { cursor:pointer; font-size:12px; color:var(--muted);
  padding:4px 0; }
/* converted-process cards */
.convgrid { display:flex; flex-direction:column; gap:12px; padding:14px 18px 18px; }
.convcard { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.convcard .ch { display:flex; align-items:center; gap:10px; padding:11px 14px;
  background:#f0fdf4; border-bottom:1px solid #dcfce7; flex-wrap:wrap; }
.convcard .ch .nm { font-weight:650; font-size:14px; }
.convcard .flow { font-family:"SF Mono",ui-monospace,Menlo,monospace;
  font-size:12px; color:#374151; }
.convcard .arrow { color:#16a34a; font-weight:700; }
.convbody { display:grid; grid-template-columns:120px 1fr; gap:2px 14px;
  padding:12px 14px; font-size:12.5px; }
.convbody .k { color:var(--muted); font-weight:600; text-transform:uppercase;
  font-size:10.5px; letter-spacing:.5px; padding-top:3px; }
.convbody .v { font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:12px;
  color:#374151; white-space:pre-wrap; word-break:break-word; }
.port { display:inline-block; background:#eef2ff; color:#3730a3; border-radius:6px;
  padding:1px 7px; margin:2px 4px 2px 0; font-size:11.5px; }
.port b { color:#1e1b4b; } .port .pp { color:#6366f1; }
/* per-config run groups */
.configgroup { margin:0 0 30px; }
.cfghead { display:flex; align-items:center; justify-content:space-between;
  gap:12px; flex-wrap:wrap; margin:0 0 14px; padding:14px 18px;
  background:linear-gradient(135deg,#1e293b,#0f172a); color:#fff;
  border-radius:12px; }
.cfghead h2 { margin:0; font-size:17px; font-weight:650; }
.cfghead .sub { color:#cbd5e1; font-size:12px; margin-top:2px; }
"""


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())


def _e(x: Any) -> str:
    return _html.escape(str(x))


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = {k: 0 for k in _VERDICT_ORDER}
    for r in rows:
        v = r.get("verdict", "not_compared")
        c[v] = c.get(v, 0) + 1
    return c


def _chips(counts: dict[str, int]) -> str:
    out = []
    for v in _VERDICT_ORDER:
        n = counts.get(v, 0)
        label, glyph, tok = _VERDICTS[v]
        cls = tok if n else "zero"
        out.append(f'<span class="chip {cls}">{glyph} {n} {_e(label)}</span>')
    return f'<div class="chips">{"".join(out)}</div>'


def _row_html(row: dict[str, Any]) -> str:
    verdict = row.get("verdict", "not_compared")
    _, glyph, tok = _VERDICTS.get(verdict, _VERDICTS["not_compared"])
    label = _e(row.get("label", ""))
    left = _e(row.get("left", ""))
    right = _e(row.get("right", ""))
    reason = row.get("reason", "")
    meter = ""
    median_rel = row.get("median_rel", None)
    max_rel = row.get("max_rel", None)
    if isinstance(median_rel, (int, float)):
        bits = [f"median rel Δ = {median_rel:.3g}"]
        if isinstance(max_rel, (int, float)):
            bits.append(f"max = {max_rel:.3g}")
        fw = row.get("frac_within", None)
        if isinstance(fw, (int, float)):
            bits.append(f"{fw * 100:.0f}% within tol")
        meter = f'<div class="metersmall">{" · ".join(bits)}</div>'
    elif isinstance(max_rel, (int, float)):
        meter = f'<div class="metersmall">max rel Δ = {max_rel:.3g}</div>'
    reason_html = f'<div class="reason">{_e(reason)}</div>' if reason else ""
    return (
        f'<tr class="verdict-{tok}">'
        f'<td class="metric">{label}'
        f'<span class="badge">{glyph} {tok}</span>{reason_html}{meter}</td>'
        f'<td class="val col-left">{left}</td>'
        f'<td class="val col-right">{right}</td>'
        f'</tr>'
    )


def _rows_block(rows: list[dict[str, Any]]) -> str:
    """Render rows, clustering by optional ``group`` key (insertion order)."""
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    index: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        g = r.get("group", "")
        if g not in index:
            index[g] = []
            groups.append((g, index[g]))
        index[g].append(r)
    has_named = any(g for g, _ in groups)
    out = []
    for gname, grows in groups:
        if gname and has_named:
            out.append(
                f'<tr class="grouprow"><td colspan="3">{_e(gname)}</td></tr>')
        out.extend(_row_html(r) for r in grows)
    return "".join(out)


def _section_html(section: dict[str, Any]) -> str:
    title = section["title"]
    desc = section.get("desc", "")
    desc_html = f'<p class="sectiondesc">{_e(desc)}</p>' if desc else ""

    # Content sections (loaded config, converted processes) carry their own HTML
    # and are NOT comparison tables — no metric/verdict columns or chips.
    if section.get("kind") == "content":
        html = section.get("html", "")
        return (
            f'<section class="card content" id="{_slug(title)}">'
            f'<div class="head"><h2>{_e(title)}</h2></div>'
            f'{desc_html}<div class="body">{html}</div></section>'
        )

    rows = section.get("rows", [])
    extra = section.get("html", "")  # embedded plots (data: URIs)
    body = _rows_block(rows)
    return (
        f'<section class="card" id="{_slug(title)}">'
        f'<div class="head"><h2>{_e(title)}</h2>{_chips(_counts(rows))}</div>'
        f'{desc_html}<table><thead><tr>'
        f'<th>metric</th>'
        f'<th class="col-left">vEcoli</th>'
        f'<th class="col-right">v2ecoli</th>'
        f'</tr></thead><tbody>{body}</tbody></table>{extra}</section>'
    )


def _legend() -> str:
    items = "".join(
        f'<span class="item"><span class="dot {_VERDICTS[k][2]}"></span>'
        f'{_e(_VERDICTS[k][0])}</span>'
        for k in _VERDICT_ORDER
    )
    return f'<div class="legend">{items}</div>'


def _nav(sections: list[dict[str, Any]]) -> str:
    links = []
    for s in sections:
        c = _counts(s.get("rows", []))
        n = sum(c.values())
        off = c.get("mismatch", 0) + c.get("drift", 0)
        mini = f'<span class="minicount">{n - off}/{n} ok</span>' if n else ""
        links.append(
            f'<a href="#{_slug(s["title"])}">{_e(s["title"])} {mini}</a>')
    return f'<nav class="sticky">{"".join(links)}</nav>'


def config_panel_section(resolved_config: dict) -> dict:
    """Render the fully-resolved run configuration as navigable JSON.

    This is reference content, not a comparison — the config is shown verbatim
    (pretty-printed) so a reader can see exactly what drove the run, with the
    vEcoli-only keys that v2ecoli configures internally tucked into a details.
    """
    shown = {k: v for k, v in resolved_config.items() if not k.startswith("_")}
    dropped = resolved_config.get("_dropped_vecoli_keys") or {}
    pretty = _e(_json.dumps(shown, indent=2, sort_keys=True))
    html = f'<pre class="cfgjson">{pretty}</pre>'
    if dropped:
        dp = _e(_json.dumps(dropped, indent=2, sort_keys=True))
        html += (
            '<details class="cfgdrop"><summary>vEcoli-only keys not consumed by '
            'v2ecoli (it configures these internally)</summary>'
            f'<pre class="cfgjson">{dp}</pre></details>')
    return {
        "title": "Loaded config", "kind": "content",
        "desc": "The fully-resolved configuration that drove this run. Processes "
                "added to both models appear under add_processes, with their "
                "wiring in topology and parameters in process_configs.",
        "html": html,
    }


def _port_chips(topology: dict) -> str:
    """Render each port -> store-path wire as a chip."""
    chips = []
    for port, path in (topology or {}).items():
        p = "/".join(str(x) for x in (path if isinstance(path, list) else [path]))
        chips.append(f'<span class="port"><b>{_e(port)}</b> '
                     f'<span class="pp">&rarr; {_e(p)}</span></span>')
    return "".join(chips) or "<span class='pp'>—</span>"


def converted_processes_section(specs: list, ran_in_both: dict) -> dict:
    """Rich card per injected process: source class, kind, config, the converted
    input/output ports (topology), and whether it ran in both engines.

    Placed right after the loaded config: it shows HOW each Vivarium-1.0 process
    declared in the config was translated into a process-bigraph process and
    wired into both models.
    """
    cards = []
    for s in specs:
        ran = ran_in_both.get(s["name"])
        status = ("&#10003; ran in both engines" if ran else
                  "&#10007; did NOT run in both" if ran is False else
                  "&#9679; run-probe unavailable")
        kind_label = {"vivarium_1": "Vivarium&nbsp;1.0", "pbg_native": "PBG-native",
                      }.get(s.get("kind"), _e(s.get("kind", "")))
        cfg = s.get("config")
        cfg_html = _e(_json.dumps(cfg)) if cfg else "default"
        ports = _port_chips(s.get("topology", {}))
        cards.append(
            f'<div class="convcard">'
            f'<div class="ch"><span class="nm">{_e(s["name"])}</span>'
            f'<span class="badge" style="background:var(--green)">{status}</span>'
            f'</div>'
            f'<div class="convbody">'
            f'<div class="k">Source</div>'
            f'<div class="v">{kind_label} &nbsp;<span class="arrow">&rarr;</span>'
            f'&nbsp;process-bigraph<br><span style="color:#6b7280">'
            f'{_e(s["module"])}.{_e(s["qualname"])}</span></div>'
            f'<div class="k">Config</div><div class="v">{cfg_html}</div>'
            f'<div class="k">Ports (in/out)</div><div class="v">{ports}</div>'
            f'<div class="k">Interval</div><div class="v">{_e(s.get("interval"))}'
            f' s</div>'
            f'</div></div>')
    if cards:
        body = f'<div class="convgrid">{"".join(cards)}</div>'
    else:
        body = ('<div class="convgrid"><p style="margin:0;color:#6b7280">No '
                'processes were added by this config.</p></div>')
    n = len(cards)
    return {
        "title": "Converted processes", "kind": "content",
        "desc": f"{n} Vivarium-1.0 process(es) declared in the config, "
                "auto-translated into process-bigraph and injected into the "
                "v2ecoli composite. Each card shows the source class, its "
                "parameters, and the input/output ports (wiring) that were "
                "converted, plus whether it produced updates in both engines.",
        "html": body,
    }


def render_report(sections: list[dict[str, Any]], *, title: str,
                  embedded_html: list[str] | None = None) -> str:
    """Render sections to a single self-contained, organized HTML string.

    Each section is ``{"title": str, "rows": [...], "html"?: str}``; each row
    is ``{"label","left","right","verdict"}`` plus optional
    ``reason``/``max_rel``/``group``. ``group`` clusters rows under
    sub-headings; verdict counts are summarized per section and overall.
    """
    total = _counts([r for s in sections for r in s.get("rows", [])])
    body = "".join(_section_html(s) for s in sections)
    extra = "".join(embedded_html) if embedded_html else ""
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_e(title)}</title><style>{_CSS}</style></head><body>'
        f'<header class="top"><h1>{_e(title)}</h1>'
        f'<div class="sub">vEcoli (left) vs v2ecoli (right) · '
        f'{sum(total.values())} metrics compared</div>'
        f'{_chips(total)}{_legend()}</header>'
        f'{_nav(sections)}'
        f'<main>{body}{extra}</main>'
        f'<footer>Generated by the vEcoli&#8596;v2ecoli comparison harness '
        f'(scripts/compare_harness.py). Self-contained; no external assets.'
        f'</footer></body></html>'
    )


def _group_rows(group: dict) -> list[dict]:
    return [r for s in group.get("sections", []) for r in s.get("rows", [])]


def _verdict_chip(v: str) -> str:
    label, glyph, tok = _VERDICTS.get(v, _VERDICTS["not_compared"])
    return f'<span class="chip {tok}">{glyph} {_e(v.replace("_", " "))}</span>'


def _card_axis_map(group: dict) -> dict:
    """axis-id -> axis dict, pulled from a group's report-card verdict."""
    out: dict[str, dict] = {}
    cv = group.get("card_verdict") or {}
    for g in (cv.get("groups") or {}).values():
        for a in g.get("axes") or []:
            out[a.get("id")] = a
    return out


def _overview_section(groups: list[dict]) -> str:
    """A high-level cross-config table: one row per config/context with its
    report-card verdict per key axis and overall, linking to each run-group."""
    rows = []
    for g in groups:
        gid = g.get("config_id") or _slug(g.get("config") or "run")
        label = g.get("config") or "run"
        ctx = g.get("context") or {}
        ctx_str = " · ".join(str(ctx[k]) for k in ("condition", "media")
                             if ctx.get(k))
        amap = _card_axis_map(g)
        overall = (g.get("card_verdict") or {}).get("overall", "not_compared")

        def cell(axis_id: str) -> str:
            a = amap.get(axis_id)
            if not a:
                return "<td>—</td>"
            return (f'<td>{_verdict_chip(a.get("verdict", "not_compared"))}'
                    f'<div class="metersmall">{_e(a.get("meter", ""))}</div></td>')

        gc = _counts(_group_rows(g))
        n = sum(gc.values())
        ok = n - gc.get("mismatch", 0) - gc.get("drift", 0)
        rows.append(
            f"<tr onclick=\"location.hash='#cfg-{gid}'\" style='cursor:pointer'>"
            f'<td class="metric"><a href="#cfg-{gid}">{_e(label)}</a>'
            f'<div class="reason">{_e(ctx_str)}</div></td>'
            f'{cell("physiology.cell_mass")}{cell("physiology.growth_rate")}'
            f'<td>{ok}/{n} ok</td><td>{_verdict_chip(overall)}</td></tr>')
    return (
        '<section class="card overview" id="overview">'
        '<div class="head"><h2>Overview — all configs</h2></div>'
        '<p class="sectiondesc">Statistical-equivalence verdict per config / '
        'context (vEcoli vs v2ecoli). Click a row to jump to that run.</p>'
        '<table><thead><tr><th>Config / context</th><th>Cell mass</th>'
        '<th>Growth rate</th><th>Sim dynamics</th><th>Overall</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></section>')


def render_grouped_report(groups: list[dict[str, Any]], *, title: str) -> str:
    """Render one or more config run-groups into a single report.

    Each group is ``{"config": label, "config_id"?: slug, "sections": [...],
    "embedded_html"?: [...]}`` — a self-contained run for one loaded config.
    Groups render as separate, individually-navigable blocks so each config is
    visibly its own set of runs.
    """
    total = _counts([r for g in groups for r in _group_rows(g)])

    # Top nav: one pill per config run-group.
    navlinks = []
    for g in groups:
        gid = g.get("config_id") or _slug(g.get("config") or "run")
        c = _counts(_group_rows(g))
        n = sum(c.values())
        off = c.get("mismatch", 0) + c.get("drift", 0)
        mini = f'<span class="minicount">{n - off}/{n} ok</span>' if n else ""
        navlinks.append(
            f'<a href="#cfg-{gid}">{_e(g.get("config") or "run")} {mini}</a>')
    nav_overview = ('<a href="#overview"><b>Overview</b></a>'
                    if len(groups) > 1 else "")
    nav = f'<nav class="sticky">{nav_overview}{"".join(navlinks)}</nav>'

    overview = _overview_section(groups) if len(groups) > 1 else ""

    blocks = []
    for g in groups:
        gid = g.get("config_id") or _slug(g.get("config") or "run")
        label = g.get("config") or "run"
        sec_html = "".join(_section_html(s) for s in g.get("sections", []))
        emb = "".join(g.get("embedded_html") or [])
        gc = _counts(_group_rows(g))
        sub = g.get("subtitle", "")
        sub_html = f'<div class="sub">{_e(sub)}</div>' if sub else ""
        blocks.append(
            f'<section class="configgroup" id="cfg-{gid}">'
            f'<div class="cfghead"><div><h2>Run: {_e(label)}</h2>{sub_html}</div>'
            f'{_chips(gc)}</div>{sec_html}{emb}</section>')
    body = "".join(blocks)
    n_cfg = len(groups)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_e(title)}</title><style>{_CSS}</style></head><body>'
        f'<header class="top"><h1>{_e(title)}</h1>'
        f'<div class="sub">vEcoli (left) vs v2ecoli (right) &middot; '
        f'{n_cfg} config{"s" if n_cfg != 1 else ""} &middot; '
        f'{sum(total.values())} metrics compared</div>'
        f'{_chips(total)}{_legend()}</header>'
        f'{nav}<main>{overview}{body}</main>'
        f'<footer>Generated by the vEcoli&#8596;v2ecoli comparison harness '
        f'(scripts/compare_harness.py). Self-contained; no external assets.'
        f'</footer></body></html>'
    )
