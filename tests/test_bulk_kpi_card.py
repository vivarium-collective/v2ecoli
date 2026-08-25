"""``bulk_kpi`` card — generic config-specific bulk-molecule KPI readout, graded
candidate-vs-reference on whatever ``observable_bulk_ids`` a study declared
(emitted on both arms under ``listeners.observable_bulk.<id>``).

General card, not a study config: it grades whatever bulk ids are declared, so
these tests exercise the pure grading helpers + the declare/degrade contract
without needing zarr stores.
"""
from scripts._compare.report_cards import REPORT_CARD_STEPS
from scripts._compare.report_cards import bulk_kpi as bk
from _card_helpers import _run_card, _state


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_grade_rel_bands():
    # WITHIN 5% -> within_tol, <=10% -> drift, above -> mismatch
    assert bk._grade_rel(1.00, 1.00) == "within_tol"
    assert bk._grade_rel(1.04, 1.00) == "within_tol"
    assert bk._grade_rel(1.08, 1.00) == "drift"
    assert bk._grade_rel(1.50, 1.00) == "mismatch"


def test_grade_rel_ungraded_when_missing():
    assert bk._grade_rel(None, 1.0) == "ungraded"
    assert bk._grade_rel(1.0, None) == "ungraded"


def test_grade_rel_zero_reference():
    assert bk._grade_rel(0.0, 0.0) == "within_tol"
    assert bk._grade_rel(1.0, 0.0) == "mismatch"


def test_final_reads_last_value():
    assert bk._final((None, [1.0, 2.0, 3.0])) == 3.0
    assert bk._final((None, [])) is None
    assert bk._final(None) is None


def test_declared_ids_from_state_then_config():
    assert bk._declared_ids({"observable_bulk_ids": ["A[c]"]}) == ["A[c]"]
    assert bk._declared_ids({"config": {"observable_bulk_ids": ["B[c]"]}}) == ["B[c]"]
    assert bk._declared_ids({}) == []


# --------------------------------------------------------------------------- #
# Step contract
# --------------------------------------------------------------------------- #
def test_card_registered():
    assert "bulk_kpi_report_card" in REPORT_CARD_STEPS


def test_card_ungraded_when_nothing_declared():
    out = _run_card("bulk_kpi", _state({}, name="basal"))
    assert out["verdict"] == "ungraded"
    assert "observable_bulk_ids" in out["card_html"]
    assert out["axes"] == []


def test_card_degrades_named_when_declared_but_absent():
    # a declared id neither arm emitted -> ungraded, but the id is named
    state = _state({}, name="basal")
    state["observable_bulk_ids"] = ["VIOLACEIN[c]"]
    out = _run_card("bulk_kpi", state)
    assert out["verdict"] == "ungraded"
    ids = {a["id"] for a in out["axes"]}
    assert "bulk.VIOLACEIN[c]" in ids
    assert "VIOLACEIN[c]" in out["card_html"]
