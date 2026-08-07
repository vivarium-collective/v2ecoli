"""Shared, dataviz-validated visual theme for the comparison harness.

Every report card + the HTML report import from here rather than picking
colors ad hoc. Values are seeded from the vivarium-workbench design tokens
(`vivarium_workbench/static/style.css` `:root` block) and, where the dataviz
method requires a computable check (the ENGINE categorical pair), validated
with the palette validator vendored at `tests/fixtures/validate_palette.js`
— see `tests/compare/test_theme.py` for the pass/fail assertions and the
task-8 report for the exact validator commands + output used to choose the
final hexes.

Color is assigned by the JOB it does, not by taste:

- ``ENGINE``  — categorical identity (which model produced this series).
  Fixed slot order: candidate=slot0, reference=slot1. Deliberately NOT drawn
  from the green/amber/red hue family so an engine color can never be
  mistaken for a status color.
- ``STATUS``  — reserved for report-card verdicts (within_tol/drift/mismatch).
  Never color-alone: each entry carries a glyph and a label alongside the hex.
- ``SURFACE`` — page/card backgrounds and text, light + dark.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# ENGINE: categorical pair identifying which model a series came from.
#
# Seeded from the workbench accent hues (--accent:#3a8 teal, --accent2:#7c3aed
# purple) then snapped to passing steps: the raw teal seed (#33aa88) measured
# only ΔE 8.2 (normal vision, OKLab) from the STATUS "within_tol" green
# (#16a34a) — below the dataviz method's 15-point floor, i.e. an engine color
# that could be mistaken for a status color. Re-stepped the candidate hue
# along the same cool/teal-to-blue family to #0891b2 (cyan-600), which clears
# every status hue by >=17.4 ΔE while keeping the workbench's --accent2 purple
# untouched as "reference". Validated with tests/fixtures/validate_palette.js
# in BOTH light (surface #fafafa) and dark (surface #0b1220) — see
# task-8-report.md for the full validator transcripts (exit 0, all checks
# PASS, no WARN, in both modes).
ENGINE = {
    "candidate": "#0891b2",   # cyan-blue: candidate model (v2ecoli)
    "reference": "#7c3aed",   # workbench --accent2 purple: reference model (vEcoli)
}

# ---------------------------------------------------------------------------
# STATUS: report-card verdict vocabulary (matches scripts/_compare/stats.py
# and verdict.py's "within_tol" / "drift" / "mismatch"). RESERVED — never
# reused for anything else, and never conveyed by color alone: every entry
# pairs its hex with a glyph and a human label.
STATUS = {
    "within_tol": {"color": "#16a34a", "glyph": "✓", "label": "Within tolerance"},  # ✓
    "drift":      {"color": "#d97706", "glyph": "◐", "label": "Drift"},             # ◐
    "mismatch":   {"color": "#dc2626", "glyph": "✗", "label": "Mismatch"},          # ✗
}

# ---------------------------------------------------------------------------
# SURFACE: page/card backgrounds + text, seeded directly from the workbench
# :root (light) and :root[data-theme="dark"] (dark) token blocks.
SURFACE = {
    "light": {
        "bg": "#fafafa",     # --bg
        "ink": "#222222",    # body text color
        "muted": "#666666",  # --gray
        "line": "#e2e6eb",   # --border
        "card": "#ffffff",   # --panel
    },
    "dark": {
        "bg": "#0b1220",     # --bg (dark)
        "ink": "#d3ddeb",    # --text (dark)
        "muted": "#93a3b6",  # --text-muted (dark)
        "line": "#26324c",   # --border (dark)
        "card": "#18233c",   # --surface (dark, cards/panels)
    },
}


def css_vars(mode: str) -> str:
    """Emit a `:root {...}` block of CSS custom properties for `mode`.

    `mode` is "light" or "dark". Exposes engine identity colors, status
    colors, and surface tokens under stable var names so report cards and
    the HTML report share one source of truth.
    """
    if mode not in SURFACE:
        raise ValueError(f"unknown theme mode {mode!r}; expected one of {sorted(SURFACE)}")

    surf = SURFACE[mode]
    lines = [
        ":root {",
        f"  --bg: {surf['bg']};",
        f"  --ink: {surf['ink']};",
        f"  --muted: {surf['muted']};",
        f"  --line: {surf['line']};",
        f"  --card: {surf['card']};",
        f"  --engine-candidate: {ENGINE['candidate']};",
        f"  --engine-reference: {ENGINE['reference']};",
    ]
    for key, entry in STATUS.items():
        lines.append(f"  --status-{key.replace('_', '-')}: {entry['color']};")
    lines.append("}")
    return "\n".join(lines)


# Vendored copy of the dataviz skill's palette validator, so tests are
# hermetic and don't depend on the ephemeral bundled-skills path.
VALIDATOR_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "validate_palette.js"
