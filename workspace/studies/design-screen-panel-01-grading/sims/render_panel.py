"""Render the screen's panel — the ranked arms AND the arms that never arrived.

A screen's scientific output is the ranked panel, so the panel is the figure. But
a ranking alone is a misleading figure here: an arm that terminated before the
graded window contributes no row to the ranking, and a reader seeing N-1 rows has
no way to know N were declared. So the accounting sits in the same frame as the
ranking, which is the study's whole claim rendered as one picture.

⚠ SVG, not HTML, and that is not a style choice. Four independent consumers each
carry their own address-scheme set and ``image:`` is the only one all four accept;
``html:`` does not survive into the published report, and ``file:`` + ``.html``
renders nothing at all while passing both the linter and the audit. An
unrecognised scheme vanishes silently from the charts payload.

Reads only committed artifacts, so the figure re-renders without re-running
anything:
    data/panel_per_cell.tsv          the per-cell seam
    data/panel_accounting.json       declared-vs-observed, generations reached
    viz/report_card/panel_screen.verdict.json   the graded axes

Run:  .venv/bin/python workspace/studies/design-screen-panel-01-grading/sims/render_panel.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
DATA = STUDY / "data"
VERDICT = STUDY / "viz" / "report_card" / "panel_screen.verdict.json"

# Status vocabulary: icon + word + reserved colour. Never colour alone — a reader
# with a colour-vision difference, or a greyscale print, must still get the state.
STATUS = {
    "within_tol": ("✓", "within band", "#1a7f37"),
    "drift":      ("△", "drift",       "#9a6700"),
    "mismatch":   ("✕", "outside band", "#b3261e"),
    "ungraded":   ("○", "ungraded",    "#57606a"),
    "absent":     ("●", "no cells",    "#57606a"),
}

W, ROW, PAD = 1180, 26, 18
COLS = [(24, "arm"), (230, "cells"), (330, "objective"), (450, "growth"),
        (580, "graded state"), (790, "note")]


def esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load():
    rows = []
    for i, line in enumerate((DATA / "panel_per_cell.tsv").read_text().splitlines()):
        parts = line.split("\t")
        if i == 0:
            head = parts
            continue
        rows.append(dict(zip(head, parts)))
    acct = json.loads((DATA / "panel_accounting.json").read_text())
    verdict = json.loads(VERDICT.read_text())
    return rows, acct, verdict


def per_arm(rows):
    """Mean per arm, aggregating within cell then across cells."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["arm"], r["lineage_seed"], r["generation"], r["agent_id"])
        cells[r["arm"]][key].append((float(r["objective_titer"]),
                                     float(r["growth_rate"])))
    out = {}
    for arm, by_cell in cells.items():
        obj = [sum(v[0] for v in vals) / len(vals) for vals in by_cell.values()]
        gro = [sum(v[1] for v in vals) / len(vals) for vals in by_cell.values()]
        out[arm] = {"n": len(by_cell),
                    "objective": sum(obj) / len(obj),
                    "growth": sum(gro) / len(gro)}
    return out


def axis_verdicts(verdict):
    """{stratum: {axis: verdict}} from the card."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for g in (verdict.get("groups") or {}).values():
        for a in g.get("axes") or []:
            aid = a.get("id", "")
            if ".medium=" not in aid:
                continue
            stratum, _, axis = aid.split(".medium=", 1)[1].partition(".")
            out[stratum][axis] = a.get("verdict", "ungraded")
    return out


def render(rows, acct, verdict) -> str:
    stats = per_arm(rows)
    verdicts = axis_verdicts(verdict)
    strata = sorted({a["stratum"] for a in acct["arms"]})
    expected = acct["cells_expected_per_arm"]

    parts, y = [], 0
    parts.append(f'<text x="24" y="34" class="h1">Design panel — ranked arms, '
                 f'and the arms that never arrived</text>')
    parts.append('<text x="24" y="56" class="sub">Synthetic panel. Each arm graded  '
                 'against the named reference within its own stratum.</text>')
    y = 78

    for stratum in strata:
        y += 26
        parts.append(f'<text x="24" y="{y}" class="h2">stratum: {esc(stratum)}</text>')
        y += 8
        parts.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" class="rule"/>')
        y += 20
        for x, label in COLS:
            parts.append(f'<text x="{x}" y="{y}" class="th">{esc(label)}</text>')
        y += 6
        parts.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" class="rule"/>')

        entries = [a for a in acct["arms"] if a["stratum"] == stratum]
        ref = next((a for a in entries if a["design"] == "reference"), None)
        ref_stats = stats.get(ref["arm"]) if ref else None

        # Present arms first, ranked by objective; absent arms last.
        present = [a for a in entries if a["in_panel"]]
        absent = [a for a in entries if not a["in_panel"]]
        present.sort(key=lambda a: -stats[a["arm"]]["objective"])
        best_arm = next((a["arm"] for a in present
                         if a["design"] != "reference"), None)

        for a in present + absent:
            y += ROW
            st = stats.get(a["arm"])
            is_ref = a["design"] == "reference"
            if st and ref_stats:
                obj = f'{st["objective"] / ref_stats["objective"]:.2f}x'
                gro = f'{st["growth"] / ref_stats["growth"]:.2f}x'
                cells = f'{st["n"]} / {expected}'
            else:
                obj = gro = "—"
                cells = f'0 / {expected}'

            if not a["in_panel"]:
                icon, word, colour = STATUS["absent"]
                note = f'terminated at gen {a["generations_reached"]}'
            elif is_ref:
                icon, word, colour = "—", "reference", "#57606a"
                note = "the named comparator"
            elif a["arm"] == best_arm:
                # The card grades THE BEST ARM against the bands, not every arm.
                # An earlier version stamped the stratum's verdict onto every row,
                # reporting a design as failing a band it was never graded
                # against -- moderate at 0.91x growth flagged for a 0.85 floor it
                # clears. Only the graded arm carries a verdict.
                v = verdicts.get(stratum, {})
                worst = "mismatch" if v.get("growth_cost") == "mismatch" else \
                        v.get("objective_vs_reference", "ungraded")
                icon, word, colour = STATUS.get(worst, STATUS["ungraded"])
                note = ("best objective; growth below floor"
                        if v.get("growth_cost") == "mismatch" else "best objective")
            else:
                icon, word, colour = "·", "not graded", "#8c959f"
                note = "ranked, not the graded arm"

            label = a["design"] + ("  (reference)" if is_ref else "")
            parts.append(f'<text x="{COLS[0][0]}" y="{y}" class="td">{esc(label)}</text>')
            parts.append(f'<text x="{COLS[1][0]}" y="{y}" class="tdn">{esc(cells)}</text>')
            parts.append(f'<text x="{COLS[2][0]}" y="{y}" class="tdn">{esc(obj)}</text>')
            parts.append(f'<text x="{COLS[3][0]}" y="{y}" class="tdn">{esc(gro)}</text>')
            parts.append(f'<text x="{COLS[4][0]}" y="{y}" class="td" '
                         f'fill="{colour}">{icon} {esc(word)}</text>')
            parts.append(f'<text x="{COLS[5][0]}" y="{y}" class="tdm">{esc(note)}</text>')

        y += 14
        rv = verdicts.get(stratum, {}).get("ranking_resolvable", "ungraded")
        icon, word, _ = STATUS.get(rv, STATUS["ungraded"])
        parts.append(f'<text x="24" y="{y}" class="foot">ranking resolvable: '
                     f'{icon} {esc(word)} — a ranking is only reported when '
                     f'this passes</text>')
        y += 10

    y += 30
    declared = len(acct["arms"])
    shown = sum(1 for a in acct["arms"] if a["in_panel"])
    parts.append(f'<line x1="24" y1="{y-18}" x2="{W-24}" y2="{y-18}" class="rule"/>')
    parts.append(f'<text x="24" y="{y}" class="foot">{declared} arms declared · '
                 f'{shown} reached the graded window · {declared - shown} recorded as '
                 f'terminated before it. Every declared arm is accounted for.</text>')
    height = y + 24

    style = ("<style>"
             ".h1{font:600 17px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".sub{font:400 12px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".h2{font:600 13px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".th{font:600 11px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".td{font:400 13px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".tdn{font:400 13px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#1f2328}"
             ".tdm{font:400 12px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".foot{font:400 11px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".rule{stroke:#d0d7de;stroke-width:1}"
             "</style>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}">{style}'
            f'<rect width="{W}" height="{height}" fill="#ffffff"/>'
            + "".join(parts) + "</svg>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(STUDY / "charts" / "panel.svg"))
    args = ap.parse_args()
    rows, acct, verdict = load()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, acct, verdict), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
