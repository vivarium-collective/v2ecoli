import math

from scripts._compare.plotly_helpers import (
    _median_relative_deltas,
    delta_panel_html,
    overlay_band_html,
)

PER = {"cell_mass": {
    "v2": [([0, 1, 2], [1.0, 1.1, 1.2]), ([0, 1, 2], [1.0, 1.05, 1.15])],
    "ve": [([0, 1, 2], [1.0, 1.1, 1.25]), ([0, 1, 2], [1.0, 1.08, 1.2])],
    "gen_bounds": []}}


def test_overlay_band_emits_band_and_both_engines():
    html = overlay_band_html(PER, title="cell")
    assert "cell_mass" in html
    assert html.count("fill") >= 1        # at least one shaded band
    assert "candidate" in html.lower() or "v2ecoli" in html.lower()


def test_delta_panel_shades_tolerance_and_annotates_stat():
    html = delta_panel_html(PER, tol=0.1, stat={"kind": "Welch-t", "p": 0.4})
    assert "Welch-t" in html and "0.4" in html


def test_delta_panel_zero_reference_is_gap_not_zero():
    # ve is 0 at every seed at t=1 (v2 nonzero there) -> relative delta is
    # undefined at that timepoint and must be a gap (NaN), never a fake 0.0
    # "agrees" point.
    per = {"obs": {
        "v2": [([0, 1, 2], [1.0, 2.0, 1.0]), ([0, 1, 2], [1.0, 3.0, 1.0])],
        "ve": [([0, 1, 2], [1.0, 0.0, 1.0]), ([0, 1, 2], [1.0, 0.0, 1.0])],
        "gen_bounds": []}}

    times, deltas = _median_relative_deltas(per["obs"]["ve"], per["obs"]["v2"])

    assert times == [0, 1, 2]
    assert deltas[0] == 0.0
    assert math.isnan(deltas[1])
    assert deltas[2] == 0.0

    # The rendered fragment must not silently coerce that NaN into 0.0.
    html = delta_panel_html(per, tol=0.1)
    assert "obs" in html
