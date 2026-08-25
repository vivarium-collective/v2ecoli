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


def _write_sidecar(d, prefix, basis):
    import json
    (d / f"{prefix}_exchange_flux.json").write_text(
        json.dumps({"basis": basis, "leaves": {"product_exchange": "X[c]"}}),
        encoding="utf-8")


def test_basis_is_read_from_the_RUNS_not_the_study_config(tmp_path):
    """Ground truth is what the run computed. Reading it from the study config
    instead is what previously let the engines emit one quantity while the card
    graded another — inside tolerance, because both arms were equally wrong."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "gdcw")
    basis, why = vio._basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis == "gdcw" and why == ""


def test_basis_refused_when_the_two_arms_disagree(tmp_path):
    """The failure this whole surface exists to make impossible: two arms
    carrying different quantities under one leaf name."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "counts")
    basis, why = vio._basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis is None and "different bases" in why


def test_card_END_TO_END_takes_the_basis_from_the_run_over_the_config(tmp_path):
    """⚠ The wiring, not the helper. Every previous version of this test suite
    tested _basis_from_runs directly, so swapping the CALL SITE back to
    state["config"] — the exact bug that shipped — stayed green three times.

    The state below is contradictory on purpose: the runs say gdcw, the config
    says counts. Reading the run yields a computable basis and no refusal; reading
    the config yields a refusal. That difference is the assertion."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "gdcw")
    st = _state({}, name="t", config={"exchange_flux_basis": "counts"})  # IGNORED
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    out = _run_card("violacein", st)
    for ax in out["axes"]:
        assert "unresolved_reason" not in ax["detail"], (
            "the card refused on a basis it should have read from the run — it is "
            "re-deriving from the study config again")


def test_card_END_TO_END_refuses_when_the_runs_say_counts(tmp_path):
    """The other direction, so the test above cannot pass by never refusing."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "counts")
    _write_sidecar(ve, "vecoli", "counts")
    st = _state({}, name="t", config={"exchange_flux_basis": "gdcw"})  # IGNORED
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    out = _run_card("violacein", st)
    assert all("unresolved_reason" in ax["detail"] for ax in out["axes"])
    assert "not graded" in out["card_html"] or "not computed" in out["card_html"]


def test_basis_refused_when_a_sidecar_is_missing(tmp_path):
    """A run that predates the sidecar, or an arm that never emitted one. A
    number whose quantity is unknown is worse than no number."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    basis, why = vio._basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis is None and "vecoli" in why


def test_yield_gg_uses_mw_ratio():
    y = vio._yield_gg((None, [1.0, 1.0]), (None, [5.0, 5.0]),
                      vio_mw=0.34338, glc_mw=0.180156, basis="gdcw")
    assert abs(y - (1 * 0.34338) / (5 * 0.180156)) < 1e-9


def test_yield_gg_is_refused_on_counts():
    """Units cancel in a ratio, but the QUANTITY does not: on counts this is a
    ratio of lineage-cumulative totals carrying the offset inherited across
    division, so it drifts toward the lineage's historical yield while reporting
    the same id, label and band. Silently changing meaning with a setting is the
    failure, not the units."""
    assert vio._yield_gg((None, [1.0]), (None, [5.0]),
                         vio_mw=0.34338, glc_mw=0.180156, basis="counts") is None
    assert vio._yield_gg((None, [1.0]), (None, [5.0]),
                         vio_mw=0.34338, glc_mw=0.180156) is None


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
