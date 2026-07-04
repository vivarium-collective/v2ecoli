"""Tests for the shared SVG chart helpers in scripts._compare.charts."""
from scripts._compare import charts


def test_sparkline_returns_svg():
    snaps = [{"dry_mass": 1.0}, {"dry_mass": 1.2}, {"dry_mass": 1.5}]
    svg = charts.sparkline(snaps, "dry_mass")
    assert svg.startswith("<svg") and "polyline" in svg


def test_multiline_svg_two_series():
    # series is a list of (x, y) point lists (one list per engine/line)
    series = [
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        [(0.0, 3.0), (1.0, 2.0), (2.0, 1.0)],
    ]
    svg, _ = charts.multiline_svg(series)
    assert svg.startswith("<svg")
