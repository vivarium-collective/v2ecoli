"""`violacein` bioproduction card — the sms-ecoli-only readout of violacein
secretion flux + yield, graded candidate-vs-reference on study #86's bands.

Lives in sms-ecoli, NOT v2ecoli: v2ecoli's baseline knows nothing about
violacein, so the leaf names, MWs, and grading bands are all sms-ecoli
knowledge. The card reads a configured exchange leaf through the harness's
existing read_pbg_local; when that leaf is absent (emission still off / v2ecoli#547)
it degrades to a clear ungraded status naming the leaf it looked for.
"""
from scripts._compare.report_cards import REPORT_CARD_STEPS
from scripts._compare.report_cards import violacein as vio
from _card_helpers import _run_card, _state


# --------------------------------------------------------------------------- #
# pure helpers (no zarr needed)
# --------------------------------------------------------------------------- #
def test_specific_rate_on_gdcw_is_the_leaf_mean_and_does_NOT_normalize():
    """On the gdcw basis the leaf is ALREADY mmol/gDCW/h, so the specific rate is
    its mean. Dividing by dry mass here would divide by it twice — which grades
    cleanly against a 3% band and is wrong. Dry mass is passed and must be
    ignored: this test fails if the normalisation is reinstated."""
    r = vio._specific_rate((None, [2.0, 2.0]), (None, [0.5, 0.5]), basis="gdcw")
    assert abs(r - 2.0) < 1e-9


def test_specific_rate_is_refused_on_the_counts_basis():
    """A counts leaf is a lineage-cumulative molecule total; its mean is not a
    flux, and mean(count)/mean(dry_mass) is count-per-femtogram, not the
    mmol/gDW/h this axis reports. Unresolved-and-visible beats confidently wrong.
    Also pins the default: an undeclared basis is counts, hence refused."""
    assert vio._specific_rate((None, [2.0, 2.0]), (None, [0.5, 0.5]),
                              basis="counts") is None
    assert vio._specific_rate((None, [2.0, 2.0]), (None, [0.5, 0.5])) is None


def test_specific_rate_none_when_empty():
    assert vio._specific_rate((None, []), (None, [0.5]), basis="gdcw") is None


def test_card_reads_the_basis_from_study_config():
    """The grading layer must see the study's declaration. Catches the basis
    being threaded to the engines but never to the card — which is exactly how
    the double-normalisation reached a graded axis."""
    assert vio._cfg({"config": {"exchange_flux_basis": "gdcw"}},
                    "exchange_flux_basis") == "gdcw"
    assert vio._cfg({}, "exchange_flux_basis") == "counts"


def test_yield_gg_uses_mw_ratio():
    # 1 mmol/s violacein, 5 mmol/s glucose uptake; g/g = (1*MWv)/(5*MWg)
    y = vio._yield_gg((None, [1.0, 1.0]), (None, [5.0, 5.0]),
                      vio_mw=0.34338, glc_mw=0.180156)
    assert abs(y - (1 * 0.34338) / (5 * 0.180156)) < 1e-9


def test_grade_rel_bands_match_86():
    # #86: within_tol < 3%, drift 3-10%, mismatch > 10%
    assert vio._grade_rel(1.00, 1.00, 0.03, 0.10) == "within_tol"
    assert vio._grade_rel(1.02, 1.00, 0.03, 0.10) == "within_tol"
    assert vio._grade_rel(1.05, 1.00, 0.03, 0.10) == "drift"
    assert vio._grade_rel(1.20, 1.00, 0.03, 0.10) == "drift" or \
        vio._grade_rel(1.20, 1.00, 0.03, 0.10) == "mismatch"
    assert vio._grade_rel(1.50, 1.00, 0.03, 0.10) == "mismatch"


def test_grade_rel_ungraded_when_missing():
    assert vio._grade_rel(None, 1.0, 0.03, 0.10) == "ungraded"
    assert vio._grade_rel(1.0, None, 0.03, 0.10) == "ungraded"
    assert vio._grade_rel(1.0, 0.0, 0.03, 0.10) == "ungraded"  # no reference scale


# --------------------------------------------------------------------------- #
# Step contract
# --------------------------------------------------------------------------- #
def test_card_registered():
    assert "violacein_report_card" in REPORT_CARD_STEPS


def test_card_degrades_when_no_leaf_emitted():
    # empty v2_dir/ve_dir -> no zarr -> ungraded status readout, not a crash
    out = _run_card("violacein", _state({}, name="basal"))
    assert out["verdict"] == "ungraded"
    assert "violacein" in out["card_html"].lower()
    # names the two axes it grades, even ungraded
    ids = {a["id"] for a in out["axes"]}
    assert "bioproduction.violacein_rate" in ids
    assert "bioproduction.violacein_yield" in ids
    # tells the reader which leaf it looked for (so the emission gap is visible)
    assert "leaf" in out["card_html"].lower()


def test_axis_grades_candidate_vs_reference():
    # got within 3% of ref -> within_tol
    ax = vio._axis("bioproduction.violacein_rate", "Violacein secretion rate",
                   got=0.051, ref=0.050, within=0.03, drift=0.10, units="mmol/gDW/h")
    assert ax["verdict"] == "within_tol"
    assert ax["id"] == "bioproduction.violacein_rate"
    assert ax["value"] is not None
