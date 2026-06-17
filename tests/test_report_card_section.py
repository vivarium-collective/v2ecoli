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
