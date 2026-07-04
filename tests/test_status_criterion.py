from v2ecoli.library.card_criteria import grade_axis


def test_status_passes_through_verdict_and_fields():
    g = grade_axis(
        {"verdict": "within_tol", "value": 42.0, "meter": "ok", "detail": {"k": 1}},
        {"type": "status", "criterion_str": "in [35, 55]"},
    )
    assert g["verdict"] == "within_tol"
    assert g["value"] == 42.0
    assert g["criterion_str"] == "in [35, 55]"
    assert g["meter"] == "ok"
    assert g["detail"] == {"k": 1}


def test_status_unknown_verdict_is_ungraded():
    assert grade_axis({"verdict": "bogus"}, {"type": "status"})["verdict"] == "ungraded"


def test_status_missing_node_is_ungraded():
    g = grade_axis(None, {"type": "status"})
    assert g["verdict"] == "ungraded"
    assert g["value"] is None
