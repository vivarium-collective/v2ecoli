#!/usr/bin/env python
"""Headlessly export every investigation's self-contained HTML report.

The investigation report is built CLIENT-SIDE by the vivarium-dashboard SPA
(`_generateInvestigationReport()` → `_buildInvestigationReportHtml`), embedding
each study's figures as `iframe srcdoc` + data URIs. There is no server-side
generator, so this script drives the real "Generate report" button in a headless
Chromium and captures the resulting download — guaranteeing byte-parity with a
manual browser export.

Flow:
  1. discover investigations under workspace/investigations/*/investigation.yaml
  2. serve the dashboard (`vivarium-dashboard serve`) unless --url points at a
     running one
  3. for each investigation: _openInvestigationDetail(slug) → click-equivalent
     _generateInvestigationReport() → capture the download → write
     <out>/investigations/<slug>.html
  4. fail loudly if a report is implausibly small or missing its figure embeds

Usage:
  .venv/bin/python scripts/publish_investigation_reports.py \
      --workspace . --out reports/published
  # reuse an already-running dashboard:
  .venv/bin/python scripts/publish_investigation_reports.py --url http://localhost:52243

Exit code is non-zero if any investigation fails to produce a valid report, so
CI can gate on it.
"""
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright

# A report with figures must embed them; a report that lost its embeds (the
# `_generateReportHtmlForCurrentIset` shortcut bug) has none. We treat "claims
# figures but embeds none" as a hard failure rather than silently publishing a
# stripped report.
MIN_REPORT_BYTES = 20_000


def discover_investigations(ws_root: Path) -> list[str]:
    inv_root = ws_root / "workspace" / "investigations"
    if not inv_root.is_dir():
        inv_root = ws_root / "investigations"  # flat-layout fallback
    return sorted(
        d.name for d in inv_root.iterdir()
        if d.is_dir() and (d / "investigation.yaml").is_file()
    )


def study_figure_count(ws_root: Path, slug: str) -> int:
    """How many figure HTMLs the investigation's studies reference on disk.

    Used only to decide whether a generated report SHOULD contain embeds, so we
    can flag a silently-stripped report. Counts committed figure files under
    reports/figures/<study>/ for each study in the investigation.
    """
    for base in (ws_root / "workspace" / "investigations", ws_root / "investigations"):
        inv_yaml = base / slug / "investigation.yaml"
        if inv_yaml.is_file():
            break
    else:
        return 0
    spec = yaml.safe_load(inv_yaml.read_text(encoding="utf-8")) or {}
    studies = [s.get("name") if isinstance(s, dict) else s
               for s in (spec.get("studies") or [])]
    fig_root = ws_root / "reports" / "figures"
    n = 0
    for st in filter(None, studies):
        d = fig_root / st
        if d.is_dir():
            n += len(list(d.glob("*.html")))
    return n


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthy(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _dashboard_binary(ws_root: Path) -> str:
    """Resolve the vivarium-dashboard CLI: workspace .venv first, then PATH."""
    local = ws_root / ".venv" / "bin" / "vivarium-dashboard"
    if local.exists():
        return str(local)
    found = shutil.which("vivarium-dashboard")
    if found:
        return found
    raise RuntimeError(
        "vivarium-dashboard CLI not found (looked in .venv/bin and PATH); "
        "install it with `uv sync` or `uv pip install vivarium-dashboard`"
    )


def serve_dashboard(ws_root: Path, port: int) -> subprocess.Popen:
    cmd = [_dashboard_binary(ws_root), "serve",
           "--workspace", str(ws_root), "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not _wait_healthy(f"http://127.0.0.1:{port}/", timeout=90):
        out = proc.stdout.read().decode(errors="replace")[-2000:] if proc.stdout else ""
        proc.terminate()
        raise RuntimeError(f"dashboard did not become healthy on :{port}\n{out}")
    return proc


def export_report(page, base_url: str, slug: str, out_path: Path,
                  expect_figures: bool) -> tuple[bool, str]:
    """Drive the live SPA to export one investigation report. Returns (ok, msg).

    The "Generate report" button builds the full report (figure embeds included)
    and hands it to ``window._triggerDownload``. Rather than capture a browser
    download event (which fails *slowly* — a 2-minute timeout — and hides the
    cause when generation rejects), we override ``_triggerDownload`` to resolve a
    promise with the HTML and surface any rejection. Failures return in seconds
    with the real error (e.g. a study 404).
    """
    page.goto(base_url, wait_until="networkidle", timeout=45_000)
    page.wait_for_function(
        "typeof window._generateInvestigationReport === 'function' "
        "&& typeof window._openInvestigationDetail === 'function'",
        timeout=30_000,
    )
    page.evaluate("(s) => window._openInvestigationDetail(s)", slug)

    # The button builds the full report (figure embeds included) and clicks a
    # download link via an IIFE-local _triggerDownload we can't override from the
    # page. So we capture the browser download event for success, and watch the
    # console for the SPA's "report generation failed" rejection so failures
    # return in seconds (with the cause) instead of stalling the full timeout.
    state: dict = {"download": None, "error": None}

    def _on_download(d):
        state["download"] = d

    def _on_console(m):
        if "report generation failed" in m.text.lower():
            state["error"] = m.text

    page.on("download", _on_download)
    page.on("console", _on_console)
    try:
        # Fire-and-forget: discard the returned promise so evaluate() doesn't
        # block until the (async) generation settles.
        page.evaluate("() => { window._generateInvestigationReport(); }")
        deadline = time.time() + 120
        while state["download"] is None and state["error"] is None and time.time() < deadline:
            page.wait_for_timeout(500)
    finally:
        page.remove_listener("download", _on_download)
        page.remove_listener("console", _on_console)

    if state["error"]:
        return False, state["error"].strip()[:200]
    if state["download"] is None:
        return False, "no report produced within 120s"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state["download"].save_as(out_path)
    html = out_path.read_text(encoding="utf-8", errors="replace")
    size = len(html)
    embeds = html.count("<iframe") + html.count("srcdoc") + html.count("data:image")
    if size < MIN_REPORT_BYTES:
        return False, f"report too small ({size} B < {MIN_REPORT_BYTES})"
    if expect_figures and embeds == 0:
        return False, (f"{size} B but ZERO figure embeds while studies reference "
                       f"figures — report was silently stripped")
    return True, f"{size:,} B, {embeds} embed-markers"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=".", help="workspace root (default: .)")
    ap.add_argument("--out", default="reports/published",
                    help="output dir; reports written to <out>/investigations/<slug>.html")
    ap.add_argument("--url", default=None,
                    help="use an already-running dashboard at this URL instead of spawning one")
    ap.add_argument("--port", type=int, default=0, help="port to serve on (default: auto)")
    ap.add_argument("--only", default=None,
                    help="comma-separated investigation slugs to publish (default: all)")
    args = ap.parse_args()

    ws_root = Path(args.workspace).resolve()
    out_dir = Path(args.out).resolve()
    slugs = discover_investigations(ws_root)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        slugs = [s for s in slugs if s in want]
    if not slugs:
        print("no investigations found", file=sys.stderr)
        return 1
    print(f"investigations: {', '.join(slugs)}")

    proc = None
    base_url = args.url
    try:
        if base_url is None:
            port = args.port or _free_port()
            print(f"serving dashboard on :{port} …")
            proc = serve_dashboard(ws_root, port)
            base_url = f"http://127.0.0.1:{port}"
        print(f"using dashboard at {base_url}")

        results: dict[str, tuple[bool, str]] = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(accept_downloads=True)
            for slug in slugs:
                out_path = out_dir / "investigations" / f"{slug}.html"
                expect_figures = study_figure_count(ws_root, slug) > 0
                try:
                    ok, msg = export_report(page, base_url, slug, out_path,
                                            expect_figures)
                except Exception as e:  # noqa: BLE001 — report per-slug, keep going
                    ok, msg = False, f"exception: {e}"
                results[slug] = (ok, msg)
                print(f"  {'✓' if ok else '✗'} {slug}: {msg}")
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    failed = [s for s, (ok, _) in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} reports published to "
          f"{out_dir / 'investigations'}")
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
