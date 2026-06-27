"""`config` card — the vEcoli config driving this section's run, rendered
deterministically from the config file (always available, unlike the Nextflow
`config_diff` card which needs S3/workflow artifacts). Shows condition, run
shape (seeds/gens), and the config's other top-level settings."""
from __future__ import annotations

import html as _html
import json as _json

from scripts._compare.report_cards import report_card, CardContext, Section

# Keys surfaced first, in this order; everything else follows alphabetically.
_PRIORITY = ["condition", "n_init_sims", "generations", "time_step",
             "max_duration", "variants", "inherit_from"]
_LABEL = {"n_init_sims": "seeds (n_init_sims)", "generations": "generations"}


@report_card("config")
def config_card(ctx: CardContext) -> Section:
    cfg = dict(ctx.config or {})
    cfg.setdefault("condition", ctx.config_name)
    if ctx.seeds:
        cfg.setdefault("n_init_sims", ctx.seeds)
    if ctx.gens:
        cfg.setdefault("generations", ctx.gens)
    keys = [k for k in _PRIORITY if k in cfg] + sorted(
        k for k in cfg if k not in _PRIORITY and not k.startswith("_"))
    rows = []
    for k in keys:
        v = cfg[k]
        vs = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else (
            _html.escape(str(v)) if not isinstance(v, dict) else _html.escape(repr(v)))
        rows.append(
            f'<tr><td style="padding:2px 12px 2px 0;color:#6b7280">'
            f'{_html.escape(_LABEL.get(k, k))}</td><td style="font-family:monospace">'
            f'{vs}</td></tr>')
    table = ('<table style="border-collapse:collapse;font-size:13px">'
             + "".join(rows) + "</table>") if rows else (
        f"<p>config for '{_html.escape(ctx.config_name)}' (no fields resolved)</p>")
    src = ctx.config.get("_source") if isinstance(ctx.config, dict) else None
    note = (f'<p style="color:#6b7280;font-size:12px">source: '
            f'{_html.escape(str(src))}</p>') if src else ""
    # Full JSON config in a browsable (collapsible, scrollable) viewer — the
    # default so every section exposes exactly what was run.
    full = {k: v for k, v in (ctx.config or {}).items() if k != "_source"}
    json_viewer = (
        '<details open style="margin-top:8px"><summary style="cursor:pointer;'
        'color:#374151;font-weight:600">full JSON config</summary>'
        '<pre style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;'
        'padding:10px;font-size:12px;line-height:1.4;max-height:380px;overflow:auto;'
        'white-space:pre">' + _html.escape(_json.dumps(full, indent=2, sort_keys=True))
        + '</pre></details>') if full else ""
    return {"title": f"{ctx.config_name} — config",
            "kind": "content",
            "anchor": f"{ctx.config_name}-config",
            "html": f"<p>vEcoli config driving the <b>{_html.escape(ctx.config_name)}</b> "
                    f"run (config = source of truth for the run shape).</p>"
                    + table + json_viewer + note,
            "verdict": None}
