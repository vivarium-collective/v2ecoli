from v2ecoli.library.report_card import verdict_json


def _fake_report():
    return {
        "overall": "mismatch",
        "axes": {
            "physiology.doubling_time": {"group": "Physiology", "label": "Doubling time",
                "verdict": "within_tol", "value": 0.84, "meter": "Δ = -2.2%",
                "detail": {"p": 0.014, "cohens_d": -0.26, "delta_rel": -0.022}},
            "fluxes.o2": {"group": "Exchange fluxes", "label": "O2 exchange",
                "verdict": "mismatch", "value": -0.45, "meter": "Δ = -40.4%",
                "detail": {"p": 0.0, "cohens_d": 0.89, "delta_rel": -0.404}},
        },
    }


def test_verdict_json_groups_axes_and_slugs_group_names():
    vj = verdict_json(_fake_report(), model_ref="abc1234",
                      reference_model="vEcoli (v1)", generated="2026-06-13 00:00")
    assert vj["schema"] == "report_card_verdict/v1"
    assert vj["overall"] == "mismatch"
    assert set(vj["groups"]) == {"physiology", "exchange_fluxes"}
    phys = vj["groups"]["physiology"]
    assert phys["axes"][0]["id"] == "physiology.doubling_time"
    assert phys["axes"][0]["verdict"] == "within_tol"
    assert vj["groups"]["exchange_fluxes"]["verdict"] == "mismatch"
    assert vj["groups"]["physiology"]["verdict"] == "within_tol"
