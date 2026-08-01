from scripts._compare.plotly_helpers import overlay_band_html, delta_panel_html

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
