"""`violacein` bioproduction card — the sms-ecoli-only readout of violacein
secretion flux + yield, graded candidate-vs-reference on study #86's bands.

Lives in sms-ecoli, NOT v2ecoli: v2ecoli's baseline knows nothing about
violacein, so the leaf names, MWs, and grading bands are all sms-ecoli
knowledge. The card reads a configured exchange leaf through the harness's
existing read_pbg_local; when that leaf is absent (emission still off / v2ecoli#547)
it degrades to a clear ungraded status naming the leaf it looked for.
"""
from scripts._compare.report_cards import REPORT_CARD_STEPS
from scripts._compare.exchange_flux_basis import basis_from_runs
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
    """The PRODUCTION writer, not a hand-rolled json.dump.

    ⚠ This used to re-type the filename and the JSON key by hand, which made the
    on-disk contract between run_comparison_ensemble's writer and this card's
    reader untested from BOTH ends: renaming either in the writer left every test
    below green while every real run lost its basis and the card refused to grade
    an otherwise healthy run. Calling the writer means a rename reds here."""
    from scripts.run_comparison_ensemble import _write_exchange_flux_sidecar
    _write_exchange_flux_sidecar(str(d), prefix,
                                 {"product_exchange": "X[c]"}, basis)


def test_basis_is_read_from_the_RUNS_not_the_study_config(tmp_path):
    """Ground truth is what the run computed. Reading it from the study config
    instead is what previously let the engines emit one quantity while the card
    graded another — inside tolerance, because both arms were equally wrong."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "gdcw")
    basis, why = basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis == "gdcw" and why == ""


def test_basis_refused_when_the_two_arms_disagree(tmp_path):
    """The failure this whole surface exists to make impossible: two arms
    carrying different quantities under one leaf name."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "counts")
    basis, why = basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis is None and "different bases" in why


def test_card_END_TO_END_takes_the_basis_from_the_run_over_the_config(tmp_path):
    """⚠ The wiring, not the helper. Every previous version of this test suite
    tested basis_from_runs directly, so swapping the CALL SITE back to
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
        # ⚠ Asserts the absence of a BASIS refusal specifically, not the absence
        # of any reason. This state carries no traces, so the axes are
        # legitimately ungraded for want of data and now say so — a different
        # refusal from the one under test. Keying on the basis wording keeps the
        # discrimination: had the card re-derived from state["config"] it would
        # refuse with "on the 'counts' basis".
        why = ax["detail"].get("unresolved_reason", "")
        assert "basis" not in why, (
            "the card refused on a basis it should have read from the run — it is "
            f"re-deriving from the study config again: {why!r}")


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
    basis, why = basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
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
    # ⚠ WAS `== "drift" or == "mismatch"`, which is vacuous: a 20% deviation has
    # exactly one correct answer under bands of 3%/10%, and an assertion admitting
    # either cannot fail.
    assert vio._grade_rel(1.20, 1.00, 0.03, 0.10) == "mismatch"
    # Either side of the drift/mismatch edge, deliberately NOT on it: `rel <= drift`
    # is float-fragile at the boundary — `abs(1.10 - 1.00) / 1.00` evaluates to
    # 0.10000000000000009, so an exactly-10% deviation grades `mismatch`, not
    # `drift`. Asserting on that knife edge would pin a float artifact rather than
    # the band, so the band is pinned where it is unambiguous.
    assert vio._grade_rel(1.08, 1.00, 0.03, 0.10) == "drift"
    assert vio._grade_rel(1.12, 1.00, 0.03, 0.10) == "mismatch"
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


# --- the zero-reference hazard, on the SUCCESSFUL gdcw path ------------------
#
# ⛔ MEASURED GAP: the refusal banner and `unresolved_reason` were both gated on
# `basis != "gdcw"`. But `_grade_rel` also returns `ungraded` when `not ref` — so
# a reference arm reading exactly 0.0 on an accepted basis rendered as a table of
# zeros, verdict `ungraded`, meter "—", with NOTHING stating why. `worst()` scores
# `ungraded` at 0, so that roll-up is pass-equivalent.
#
# A 0.0 reference is not hypothetical here: an exchange leaf reads 0.0 both for a
# molecule genuinely not exchanged AND for one whose key cannot be resolved, and
# this lane spent a day reading the second as the first.

def _gdcw_runs(tmp_path):
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(v2, "v2ecoli", "gdcw")
    _write_sidecar(ve, "vecoli", "gdcw")
    return str(v2), str(ve)


def _stub_reader(monkeypatch, v2_vals, ve_vals):
    """Stub the zarr reader so the card's GRADING is exercised without fixtures.

    `_read_seed(dir, prefix, seed, leaves) -> {leaf: (times, values)}` is the only
    door between the card and the stores, so replacing it leaves every line under
    test except the zarr read itself."""
    def _fake(dir_path, prefix, seed, leaves):
        vals = v2_vals if prefix == "v2ecoli" else ve_vals
        t = [0.0, 1.0]
        return {"violacein_exchange": (t, list(vals)),
                "glucose_exchange": (t, [-8.0, -8.0]),
                "dry_mass": (t, [400.0, 400.0])}
    monkeypatch.setattr(vio, "_read_seed", _fake)


def _card_on(monkeypatch, tmp_path, v2_vals, ve_vals):
    v2, ve = _gdcw_runs(tmp_path)
    _stub_reader(monkeypatch, v2_vals, ve_vals)
    st = _state({}, name="t")
    st["v2_dir"], st["ve_dir"] = v2, ve
    return _run_card("violacein", st)


def _rate_axis(out):
    return next(a for a in out["axes"] if a["id"].endswith("violacein_rate"))


def test_a_ZERO_reference_arm_is_not_silently_ungraded(monkeypatch, tmp_path):
    """⚠ The candidate secretes, the reference reads exactly 0.0, the basis is
    accepted. The axis cannot grade — but it must SAY so, because an ungraded
    axis scores 0 in the shared severity model and therefore cannot fail a gate.
    Without a stated reason this renders as a clean table with a blank meter."""
    rate = _rate_axis(_card_on(monkeypatch, tmp_path, [0.15, 0.15], [0.0, 0.0]))
    assert rate["verdict"] == "ungraded"
    why = rate["detail"].get("unresolved_reason", "")
    assert why, "a zero-reference axis was ungraded with no reason given"
    assert "0.0" in why and "unresolvable key" in why, why
    assert "basis" not in why, (
        "reported as a basis refusal, but the basis was accepted: " + why)


def test_the_zero_reference_refusal_reaches_the_reader(monkeypatch, tmp_path):
    """The reason must reach the rendered card too, not only the axis detail —
    the axis detail is not what a human opens."""
    out = _card_on(monkeypatch, tmp_path, [0.15, 0.15], [0.0, 0.0])
    html = out.get("card_html") or out.get("html") or ""
    assert "not computed" in html.lower(), (
        "no banner rendered for a refused axis on an accepted basis")


def test_a_GRADEABLE_gdcw_pair_is_not_refused(monkeypatch, tmp_path):
    """So the two tests above cannot pass by refusing everything."""
    rate = _rate_axis(_card_on(monkeypatch, tmp_path, [0.150, 0.150], [0.149, 0.149]))
    assert rate["verdict"] == "within_tol", rate
    assert not rate["detail"].get("unresolved_reason")


# --- B6: the refusal MESSAGING, end to end -----------------------------------
#
# ⛔ MEASURED GAP: all three of basis_from_runs' documented refusal branches
# survived mutation. Making an unrecorded basis silently become "counts", making
# an unreadable sidecar return counts, and blanking `basis_reason` at the card all
# left the suite green. The grading half of this path is well tested; the
# VISIBILITY half was not — and visibility is the card's whole stated purpose here.
#
# M67 is the consequential one: with `basis_reason` blanked, a run refused for
# "missing sidecar" / "arms disagree" / "runs are stale" instead reports the
# generic "leaves are on the None basis, which is a lineage-cumulative molecule
# total" — wrong for all three, and it sends the reader to diagnose the wrong
# thing. The pre-existing end-to-end refusal test only drove the counts path,
# where basis_reason is "" anyway, so it could not discriminate.

def _card_with_sidecars(tmp_path, v2_basis, ve_basis, **kw):
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    if v2_basis is not None:
        _write_sidecar(v2, "v2ecoli", v2_basis)
    if ve_basis is not None:
        _write_sidecar(ve, "vecoli", ve_basis)
    st = _state({}, name="t", **kw)
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    return _run_card("violacein", st)


def _reason(out):
    return out["axes"][0]["detail"].get("unresolved_reason", "")


def test_two_arms_on_DIFFERENT_bases_say_so_specifically(tmp_path):
    """Not "on the None basis" — the reader must learn the arms disagree, and
    which ran which, because the fix is to re-run one of them."""
    out = _card_with_sidecars(tmp_path, "gdcw", "counts")
    why = _reason(out)
    assert "different bases" in why, why
    assert "gdcw" in why and "counts" in why, why
    # ⚠ Compared through html.escape: the banner escapes the reason, so the raw
    # string is NOT a substring of the rendered card (the quotes around the two
    # basis names become entities). Asserting the escaped form checks the real
    # thing rather than a fragment that happens to survive escaping.
    import html as _h
    assert _h.escape(why) in (out.get("card_html") or ""), (
        "the reason never reached the reader")


def test_a_MISSING_sidecar_says_which_arm_is_missing_it(tmp_path):
    """A run that predates the sidecar, or an arm that never emitted. Naming the
    arm is the difference between a 10-second fix and a hunt."""
    out = _card_with_sidecars(tmp_path, "gdcw", None)
    why = _reason(out)
    assert "sidecar" in why and "vecoli" in why, why
    assert "lineage-cumulative" not in why, (
        "reported as a counts refusal, which is a different diagnosis: " + why)


def test_an_UNRECORDED_basis_is_distinguished_from_counts(tmp_path):
    """A sidecar with no basis key is NOT 'counts' — the quantity is unknown. Two
    such sidecars would otherwise compare equal, slip the agreement check, and be
    described with counts semantics the card has no evidence for."""
    import json
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    for d, prefix in ((v2, "v2ecoli"), (ve, "vecoli")):
        (d / f"{prefix}_exchange_flux.json").write_text(
            json.dumps({"leaves": {"product_exchange": "X[c]"}}))
    st = _state({}, name="t")
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    out = _run_card("violacein", st)
    why = _reason(out)
    assert "no basis" in why or "unknown" in why, why
    assert "lineage-cumulative" not in why, (
        "an unrecorded basis was described with counts semantics: " + why)


def test_an_UNREADABLE_sidecar_is_a_refusal_not_a_default(tmp_path):
    """Corrupt JSON must refuse, not fall back to a quantity nobody recorded."""
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_sidecar(ve, "vecoli", "gdcw")
    (v2 / "v2ecoli_exchange_flux.json").write_text("{not json at all")
    st = _state({}, name="t")
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    out = _run_card("violacein", st)
    why = _reason(out)
    assert "unreadable" in why and "v2ecoli" in why, why
    assert out["axes"][0]["verdict"] == "ungraded"


def test_a_STALE_arm_is_reported_as_stale_and_names_both_shapes(tmp_path):
    """Both arms write into one out_root and nothing cleans it, so a re-run of one
    leaves the other's sidecar in place. The reason must say which is which."""
    from scripts.run_comparison_ensemble import _write_exchange_flux_sidecar
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(); ve.mkdir()
    _write_exchange_flux_sidecar(str(v2), "v2ecoli", {"product_exchange": "X[c]"},
                                 "gdcw", seeds=4, generations=8)
    _write_exchange_flux_sidecar(str(ve), "vecoli", {"product_exchange": "X[c]"},
                                 "gdcw", seeds=1, generations=1)
    st = _state({}, name="t")
    st["v2_dir"], st["ve_dir"] = str(v2), str(ve)
    why = _reason(_run_card("violacein", st))
    assert "stale" in why and "seeds=4" in why and "seeds=1" in why, why
