"""`config` card — the vEcoli config driving this section's run, rendered
deterministically from the config file (always available, unlike the Nextflow
`config_diff` card which needs S3/workflow artifacts). Shows condition, run
shape (seeds/gens), and the config's other top-level settings."""
from __future__ import annotations

import html as _html
import json as _json

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS)

# Keys surfaced first, in this order; everything else follows alphabetically.
_PRIORITY = ["condition", "n_init_sims", "generations", "time_step",
             "max_duration", "variants", "inherit_from"]
_LABEL = {"n_init_sims": "seeds (n_init_sims)", "generations": "generations"}


def _config_html(name: str, seeds: int, gens: int, config: dict) -> str:
    cfg = dict(config or {})
    cfg.setdefault("condition", name)
    if seeds:
        cfg.setdefault("n_init_sims", seeds)
    if gens:
        cfg.setdefault("generations", gens)
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
        f"<p>config for '{_html.escape(name)}' (no fields resolved)</p>")
    src = config.get("_source") if isinstance(config, dict) else None
    note = (f'<p style="color:#6b7280;font-size:12px">source: '
            f'{_html.escape(str(src))}</p>') if src else ""
    # Full JSON config in a browsable (collapsible, scrollable) viewer — the
    # default so every section exposes exactly what was run.
    full = {k: v for k, v in (config or {}).items() if k != "_source"}
    json_viewer = (
        '<details open style="margin-top:8px"><summary style="cursor:pointer;'
        'color:#374151;font-weight:600">full JSON config</summary>'
        '<pre style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;'
        'padding:10px;font-size:12px;line-height:1.4;max-height:380px;overflow:auto;'
        'white-space:pre">' + _html.escape(_json.dumps(full, indent=2, sort_keys=True))
        + '</pre></details>') if full else ""
    return (f"<p>vEcoli config driving the <b>{_html.escape(name)}</b> "
            f"run (config = source of truth for the run shape).</p>"
            + table + json_viewer + note)


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="config_report_card",
         aliases=["config"])
def update_config_report_card(state):
    return {"card_html": _config_html(state["name"], state["seeds"],
                                      state["generations"], state["config"]),
            "verdict": "ungraded", "axes": []}


REPORT_CARD_STEPS["config_report_card"] = update_config_report_card
