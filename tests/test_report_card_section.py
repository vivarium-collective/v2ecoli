from scripts._compare.report_card_section import build_report_card


def test_build_report_card_equivalent_data():
    # Near-identical distributions (with variance) -> within_tol overall.
    left = {"cell_mass": [1500.0 + i for i in range(30)],
            "growth_rate": [0.0003 + i * 1e-7 for i in range(30)]}
    right = {"cell_mass": [1502.0 + i for i in range(30)],
             "growth_rate": [0.00030 + i * 1e-7 for i in range(30)]}
    verdict, html = build_report_card(left, right)
    assert verdict["schema"] == "report_card_verdict/v1"
    assert verdict["overall"] in ("within_tol", "drift")
    assert "physiology" in verdict["groups"]
    assert html.startswith("<") and "verdict" in html.lower()


def test_build_report_card_divergent_data_flags_mismatch():
    # ~2x shift WITH variance (constant arrays give p=nan -> drift, not mismatch).
    left = {"cell_mass": [1500.0 + i for i in range(30)]}
    right = {"cell_mass": [3000.0 + i for i in range(30)]}
    verdict, _ = build_report_card(left, right)
    masses = verdict["groups"]["physiology"]["axes"]
    assert any(a["verdict"] == "mismatch" for a in masses)


def test_build_report_card_tol_rel_tightens_band():
    # cell_mass left=[1500+i], right=[1502+i] for i in range(30):
    #   left_mean = 1514.5, diff = 2, rel = 2/1514.5 ≈ 0.132%
    # Default tol_rel (0.05 = 5%)  → within_tol  (0.132% < 5%).
    # Tight  tol_rel (0.001 = 0.1%) → NOT within_tol (0.132% > 0.1%).
    left = {"cell_mass": [1500.0 + i for i in range(30)],
            "growth_rate": [0.0003 + i * 1e-7 for i in range(30)]}
    right = {"cell_mass": [1502.0 + i for i in range(30)],
             "growth_rate": [0.00030 + i * 1e-7 for i in range(30)]}
    verdict_default, _ = build_report_card(left, right)
    verdict_tight, _ = build_report_card(left, right, tol_rel=0.001)
    default_axes = verdict_default["groups"]["physiology"]["axes"]
    tight_axes = verdict_tight["groups"]["physiology"]["axes"]
    # Default: cell_mass should be within_tol (0.132% << 5%).
    assert any(a["verdict"] == "within_tol" for a in default_axes)
    # Tight: at least one axis must leave within_tol (0.132% > 0.1%).
    assert any(a["verdict"] != "within_tol" for a in tight_axes)
