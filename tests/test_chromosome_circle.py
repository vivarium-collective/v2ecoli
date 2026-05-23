"""Tests for v2ecoli.visualizations.chromosome_circle.chromosome_circle_svg."""
import re

import pytest

from v2ecoli.visualizations.chromosome_circle import chromosome_circle_svg


GENOME_LEN = 4_641_652  # E. coli K-12 MG1655


def _one_panel(features: dict) -> dict:
    return {"label": "t=0", "chromosomes": [{"features": features}]}


def test_empty_panels_returns_minimal_svg():
    svg = chromosome_circle_svg("empty", panels=[], genome_len=GENOME_LEN)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "empty" in svg


def test_basic_oric_ter_replisome():
    svg = chromosome_circle_svg(
        "basic",
        panels=[_one_panel({
            "oriC": [{"coord": 0, "marker": "circle", "color": "#16a34a", "size": 8, "category": "OriC"}],
            "ter": [{"coord": GENOME_LEN // 2, "marker": "square", "color": "#dc2626", "size": 7, "category": "Ter"}],
            "replisomes": [{"coord": 1_000_000, "marker": "triangle_up", "color": "#f59e0b", "size": 9, "category": "Replisome"}],
        })],
        genome_len=GENOME_LEN,
    )
    # All three legend categories appear
    assert "OriC" in svg
    assert "Ter" in svg
    assert "Replisome" in svg
    # Backbone circle is rendered (stroke="#cbd5e1")
    assert "#cbd5e1" in svg


def test_multiple_panels_stacked_chromosomes():
    panels = [
        {"label": "t=0", "chromosomes": [
            {"features": {"oriC": [{"coord": 0, "marker": "circle", "color": "#16a34a", "size": 8, "category": "OriC"}]}},
        ]},
        {"label": "t=1500s — replicating", "chromosomes": [
            {"features": {"oriC": [{"coord": 0, "marker": "circle", "color": "#16a34a", "size": 8, "category": "OriC"}]}},
            {"features": {"oriC": [{"coord": 0, "marker": "circle", "color": "#16a34a", "size": 8, "category": "OriC"}]}},
        ]},
    ]
    svg = chromosome_circle_svg("multi", panels=panels, genome_len=GENOME_LEN)
    # Both labels appear
    assert "t=0" in svg
    assert "t=1500s" in svg
    # Three backbone circles total (1 + 2)
    assert svg.count('stroke="#cbd5e1"') == 3


def test_legend_dedupes_repeated_category():
    """Repeated DnaA-box markers across many positions should produce ONE legend entry."""
    boxes = [{"coord": i * 50_000, "marker": "circle", "color": "#94a3b8",
              "size": 2, "category": "DnaA-box"} for i in range(20)]
    svg = chromosome_circle_svg(
        "boxes",
        panels=[_one_panel({"boxes": boxes})],
        genome_len=GENOME_LEN,
    )
    # Exactly one "DnaA-box" text in the legend section
    assert svg.count(">DnaA-box<") == 1


def test_html_escape_in_title_and_label():
    svg = chromosome_circle_svg(
        "<title>",
        panels=[{"label": "a & b", "chromosomes": [{"features": {}}]}],
        genome_len=GENOME_LEN,
    )
    # Tag chars escaped
    assert "&lt;title&gt;" in svg
    assert "a &amp; b" in svg


def test_tick_marker_renders_a_line():
    svg = chromosome_circle_svg(
        "tick",
        panels=[_one_panel({
            "ticks": [{"coord": 1_000_000, "marker": "tick", "color": "#000",
                       "size": 5, "category": "Tick"}],
        })],
        genome_len=GENOME_LEN,
    )
    assert "<line " in svg
