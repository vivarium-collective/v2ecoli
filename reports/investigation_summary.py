"""Generate a self-contained report-card summary for an investigation.

Usage:
    python reports/investigation_summary.py --investigation <slug> [--out PATH] [--no-open]
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports._summary.aggregate import aggregate  # noqa: E402
from reports._summary.render import render  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _read_style() -> str:
    css = _REPO / "reports" / "assets" / "style.css"
    try:
        text = css.read_text(encoding='utf-8')
    except OSError:
        return ""
    # keep only the :root{...} token block so the summary matches the report palette
    start = text.find(":root{")
    if start == -1:
        return ""
    end = text.find("}", start)
    return text[start:end + 1] if end != -1 else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Investigation report-card summary generator")
    ap.add_argument("--investigation", required=True, help="investigation slug")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--no-open", action="store_true", help="do not open in a browser")
    args = ap.parse_args(argv)

    ws = _REPO / "workspace"
    inv_yaml = ws / "investigations" / args.investigation / "investigation.yaml"
    if not inv_yaml.exists():
        print(f"error: investigation {args.investigation!r} not found ({inv_yaml})", file=sys.stderr)
        return 2
    summary = aggregate(args.investigation, ws)
    html = render(summary, style_css=_read_style())

    out = Path(args.out) if args.out else (
        _REPO / "reports" / "summaries" / f"{args.investigation}_summary.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    if not args.no_open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except webbrowser.Error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
