"""`violacein` card — the bioproduction readout: violacein specific secretion
rate (mmol/gDW/h) and yield on glucose (g/g), graded candidate-vs-reference.

WHY THIS LIVES IN sms-ecoli (not v2ecoli). v2ecoli's baseline carries no
violacein pathway — it does not know the molecule exists. The leaf name the
secretion flux lands on, the molecular weights that turn a molar flux into a
yield, and the acceptance bands are all violacein-specific knowledge, so the
card that reads them belongs here, in the sms-ecoli comparison harness, beside
the other `scripts/_compare/report_cards/` cards.

DATA SOURCE — reads the derived exchange-flux leaf from the matched zarr.
⚠ The claim that this "sidesteps v2ecoli#547 by never touching a listener leaf"
no longer holds on both arms: on the `gdcw` basis the CANDIDATE arm still derives
its leaf from `environment.exchange`, but the REFERENCE arm reads
`listeners.fba_results.external_exchange_fluxes` — the wrapped metabolism's own
per-tick rate, and the leaf genuine vEcoli's own analyses read. #547 concerns
dict leaves dropped on the INJECTION path, which the reference arm does not use,
so that arm is unaffected — but the card is no longer listener-free by design. Emission of that leaf is a
separate (harness/emit-config) concern; until it lands the card degrades to an
ungraded status that names the exact leaf it looked for, making the gap visible
rather than silently green.

⛔ CORRECTED — the paragraph that stood here claimed the relative delta is
"unit-robust: correct whatever native unit the exchange leaf carries, because
both arms are divided through the same conversion". That is the belief this card
was rebuilt to refute. Both arms being equally wrong does NOT make the ratio
meaningful: on the `counts` basis each leaf is a LINEAGE-CUMULATIVE molecule
total whose mean grows with how long the lineage ran, so a ratio of two such
means is not a ratio of rates, and it grades inside a 3% band while measuring
something that is not a flux. A relative delta is robust to a shared unit; it is
not robust to a shared WRONG QUANTITY.

⇒ The axes are therefore computed only from the `gdcw` basis and refused on
`counts`, and the basis is read off the RUNS (`{prefix}_exchange_flux.json`
beside the stores) rather than re-derived from the study config — two readers of
one setting disagreed once and graded a cumulative total as a rate.

⚠ There is no dry-mass normalization any more. On `gdcw` the leaf is already
mmol/gDCW/h, so the specific rate IS its mean; normalising again divided by dry
mass twice. `dry_mass` is still collected and passed to keep the call signature
stable, and is deliberately unused by `_specific_rate`.

Bands are study #86's (within_tol < 3%, drift 3–10%, mismatch > 10%) — tighter
than the harness 5%/10% default because both arms implement the SAME model.
⚠ Those bands live in this module's DEFAULTS; the study YAML's `bands:` are not
read here, so editing them there changes nothing.

⚠ GATE NOTE: a refused axis is `ungraded`, which the shared severity model scores
0 — i.e. no worse than a pass — so refusing RELAXES a gate that could previously
fail. That aggregation semantic is `_SEVERITY`/`worst()` in
`scripts/_compare/verdict.py`, mirroring pbg_v2ecoli's evaluator, and is not this
card's to change. The refusal is surfaced on the axis (`unresolved_reason`) and
in the card body so it is at least visible to a reader.

Reads the zarr stores off state["v2_dir"]/state["ve_dir"] via the harness's
existing ``read_pbg_local`` — same source as the metabolism/trajectory cards.
"""
from __future__ import annotations

import html as _html
import os

from process_bigraph.composite import as_step

from scripts._compare.plotly_helpers import overlay_html
from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.exchange_flux_basis import basis_from_runs
from scripts._compare.verdict import worst

# ---- defaults (overridable per study via config["bioproduction"][...]) ----- #
# Violacein secretion and glucose uptake flux leaves, and the MWs that turn a
# molar flux into a mass yield. Violacein C20H13N3O2 = 343.38 g/mol; glucose
# C6H12O6 = 180.156 g/mol. Expressed g/mmol.
DEFAULTS = {
    "violacein_exchange_leaf": "violacein_exchange",
    "glucose_exchange_leaf": "glucose_exchange",
    "violacein_mw": 0.34338,
    "glucose_mw": 0.180156,
    "within_tol": 0.03,
    "drift": 0.10,
}
_RATE_ID = "bioproduction.violacein_rate"
_YIELD_ID = "bioproduction.violacein_yield"


def _cfg(state: dict, key: str):
    """Resolve a config key from config["bioproduction"][key] or config[key],
    falling back to DEFAULTS."""
    cfg = state.get("config") or {}
    bio = cfg.get("bioproduction") if isinstance(cfg.get("bioproduction"), dict) else {}
    for src in (bio, cfg):
        if isinstance(src, dict) and key in src and src[key] is not None:
            return src[key]
    return DEFAULTS[key]


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _specific_rate(flux_trace, drymass_trace, basis: str = "counts") -> float | None:
    """The mean specific secretion rate, in mmol/gDCW/h — or None if the leaf's
    basis cannot express one.

    ⚠ This axis is computable ONLY from a ``gdcw`` leaf, and the reason is not a
    unit conversion. On that basis the leaf is already a per-tick mmol/gDCW/h
    rate, so the specific rate IS its mean and no normalisation is wanted —
    dividing by dry mass again would divide by it twice, which grades cleanly
    against a 3% band and is wrong.

    On ``counts`` the leaf is a LINEAGE-CUMULATIVE molecule total. Its mean over
    a trace is not a flux (it grows with how long the lineage ran), and
    mean(count)/mean(dry_mass) has units of count per femtogram — not the
    mmol/gDW/h this axis reports. That is refused rather than emitted: a
    mislabelled quantity that looks plausible is the failure this whole path
    exists to prevent, and returning None leaves the axis unresolved and visible
    instead of confidently wrong.
    """
    if str(basis) != "gdcw":
        return None
    if not flux_trace:
        return None
    return _mean(flux_trace[1])


def _yield_gg(vio_trace, glc_trace, *, vio_mw: float, glc_mw: float,
              basis: str = "counts") -> float | None:
    """Yield on glucose (g/g) = (product secreted, g) / (glucose consumed, g).
    Uptake is negative on the exchange convention, so |glucose flux| is used.

    ⚠ Basis-gated for the same reason the rate axis is, even though the units
    cancel in a ratio. On ``gdcw`` this is a ratio of two mean RATES. On
    ``counts`` it is a ratio of two lineage-cumulative totals, each carrying the
    offset inherited across division — so over a multi-generation run it drifts
    toward the whole lineage's historical yield rather than this generation's,
    while reporting the same id, label and band. A number that silently changes
    meaning with a setting is refused rather than graded."""
    if str(basis) != "gdcw":
        return None
    if not vio_trace or not glc_trace:
        return None
    vio = _mean(vio_trace[1])
    glc = _mean(glc_trace[1])
    if vio is None or glc is None or glc == 0:
        return None
    glc_mass = abs(glc) * glc_mw
    if glc_mass == 0:
        return None
    return (vio * vio_mw) / glc_mass


def _grade_rel(got, ref, within: float, drift: float) -> str:
    """within_tol if |got-ref|/|ref| < within; drift if <= drift; else mismatch.
    ungraded when either side is missing or the reference has no scale."""
    if got is None or ref is None or not ref:
        return "ungraded"
    rel = abs(got - ref) / abs(ref)
    if rel < within:
        return "within_tol"
    if rel <= drift:
        return "drift"
    return "mismatch"


def _axis(axis_id: str, label: str, got, ref, *, within: float, drift: float,
          units: str) -> dict:
    verdict = _grade_rel(got, ref, within, drift)
    rel = (abs(got - ref) / abs(ref)) if (got is not None and ref) else None
    meter = f"{rel:.1%} Δ" if rel is not None else "—"
    return {
        "id": axis_id, "label": label, "verdict": verdict,
        "value": got, "meter": meter,
        "detail": {"got": got, "reference": ref, "units": units,
                   "rel_delta": rel, "within_tol": within, "drift": drift},
    }


# --------------------------------------------------------------------------- #
# zarr reading (reuses the harness reader; a leaf absent -> {} -> None trace)
# --------------------------------------------------------------------------- #
def _read_seed(dir_path: str, prefix: str, seed: int, leaves) -> dict:
    from scripts.compare_matched_trajectories import read_pbg_local

    path = os.path.join(dir_path or "", f"{prefix}_seed{seed:02d}.zarr")
    if not dir_path or not os.path.isdir(path):
        return {}
    try:
        return read_pbg_local(path, leaves)
    except Exception:  # noqa: BLE001 — a malformed/partial store must not crash the card
        return {}


def _collect(state: dict, leaves) -> dict:
    """{leaf: {"v2": [trace,...], "ve": [trace,...]}} across seeds, for the
    requested leaves. A trace is (times, values); absent leaves simply don't
    appear in a seed's dict."""
    n_seeds = max(int(state.get("seeds") or 1), 1)
    per = {leaf: {"v2": [], "ve": []} for leaf in leaves}
    for seed in range(n_seeds):
        v2 = _read_seed(state.get("v2_dir"), "v2ecoli", seed, leaves)
        ve = _read_seed(state.get("ve_dir"), "vecoli", seed, leaves)
        for leaf in leaves:
            if leaf in v2:
                per[leaf]["v2"].append(v2[leaf])
            if leaf in ve:
                per[leaf]["ve"].append(ve[leaf])
    return per


def _first(traces):
    return traces[0] if traces else None


# --------------------------------------------------------------------------- #
# card
# --------------------------------------------------------------------------- #
def _no_data_html(vio_leaf: str, glc_leaf: str) -> str:
    return (
        '<p style="color:#6b7280">No violacein secretion flux in the matched '
        'zarr yet — this card looked for the exchange leaf '
        f'<code>{_html.escape(vio_leaf)}</code> (yield also needs '
        f'<code>{_html.escape(glc_leaf)}</code>) and found neither arm carrying '
        'it. Enable exchange-flux emission for this study\'s run (the '
        '<code>environment.exchange</code> store populates despite v2ecoli#547; '
        'the leaf must be emitted into the compact zarr the harness reads). '
        'Once it lands, this card grades candidate vs reference automatically.</p>')


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="violacein_report_card",
         aliases=["violacein"])
def update_violacein_report_card(state):
    vio_leaf = _cfg(state, "violacein_exchange_leaf")
    glc_leaf = _cfg(state, "glucose_exchange_leaf")
    within = float(_cfg(state, "within_tol"))
    drift = float(_cfg(state, "drift"))
    vio_mw = float(_cfg(state, "violacein_mw"))
    glc_mw = float(_cfg(state, "glucose_mw"))

    per = _collect(state, [vio_leaf, glc_leaf, "dry_mass"])

    basis, basis_reason = basis_from_runs(state)

    def rate(arm):
        return _specific_rate(_first(per[vio_leaf][arm]),
                              _first(per["dry_mass"][arm]), basis=basis or "counts")

    def yld(arm):
        return _yield_gg(_first(per[vio_leaf][arm]), _first(per[glc_leaf][arm]),
                         vio_mw=vio_mw, glc_mw=glc_mw, basis=basis or "counts")

    rate_axis = _axis(_RATE_ID, "Violacein secretion rate",
                      rate("v2"), rate("ve"), within=within, drift=drift,
                      units="mmol/gDW/h")
    yield_axis = _axis(_YIELD_ID, "Violacein yield (on glucose)",
                       yld("v2"), yld("ve"), within=within, drift=drift,
                       units="g/g")
    # ⚠ An axis that cannot be computed goes `ungraded`, which verdict.worst()
    # scores as severity 0 — so without this it would be indistinguishable from a
    # PASS and would quietly relax a gate that used to be able to fail. Say why,
    # on the axis and in the card body, so a refusal reads as a refusal.
    if basis != "gdcw":
        why = basis_reason or (
            f"exchange leaves are on the {basis!r} basis, which is a "
            "lineage-cumulative molecule total; neither a specific rate nor a "
            "yield is computable from it")
        for _ax in (rate_axis, yield_axis):
            _ax["meter"] = "not computable"
            _ax["detail"]["unresolved_reason"] = why
    # ⚠ AND ON THE SUCCESSFUL PATH TOO. The block above only fires when the BASIS
    # is refused, but `_grade_rel` also returns `ungraded` when a value is missing
    # or the reference has no scale — and on those an axis previously carried no
    # reason and no banner at all. A reference arm reading exactly 0.0 therefore
    # rendered as a table of zeros, verdict `ungraded`, meter "—", with nothing
    # saying why, and `worst()` scores ungraded 0, so the roll-up was
    # pass-equivalent. That is the zero-ambiguity hazard this card exists to
    # remove, reappearing inside the fix written for it. Every ungraded axis now
    # states its own cause.
    for _ax in (rate_axis, yield_axis):
        if _ax["verdict"] != "ungraded" or _ax["detail"].get("unresolved_reason"):
            continue
        _d = _ax["detail"]
        _got, _ref = _d.get("got"), _d.get("reference")
        if _got is None and _ref is None:
            _why = "neither arm produced a value for this axis"
        elif _ref is None:
            _why = "the reference arm produced no value for this axis"
        elif _got is None:
            _why = "the candidate arm produced no value for this axis"
        else:
            # ⚠ Not "the arms agree at zero". An exchange leaf reads 0.0 both for
            # a molecule genuinely not exchanged AND for one whose key could not
            # be resolved on the chosen basis — the two are indistinguishable
            # here, so this is reported as unmeasured rather than as agreement.
            _why = ("the reference arm's value is exactly 0.0, so a relative "
                    "delta has no scale — and on an exchange leaf 0.0 is also "
                    "what an unresolvable key reads as, so this is not evidence "
                    "the two arms agree")
        _ax["meter"] = "not computable"
        _d["unresolved_reason"] = _why
    axes = [rate_axis, yield_axis]

    have_flux = bool(per[vio_leaf]["v2"] or per[vio_leaf]["ve"])
    title = state.get("name", "")
    if not have_flux:
        html = _no_data_html(vio_leaf, glc_leaf)
    else:
        traj = overlay_html({vio_leaf: per[vio_leaf]}, title=f"{title} — violacein secretion")
        rows = []
        for ax in axes:
            d = ax["detail"]
            rows.append(
                f'<tr><td style="padding:2px 10px">{_html.escape(ax["label"])} '
                f'({_html.escape(d["units"])})</td>'
                f'<td style="padding:2px 10px">{_fmt(d["reference"])}</td>'
                f'<td style="padding:2px 10px">{_fmt(d["got"])}</td>'
                f'<td style="padding:2px 10px">{_html.escape(ax["verdict"])}</td>'
                f'<td style="padding:2px 10px;color:#6b7280">{ax["meter"]}</td></tr>')
        table = ('<table style="border-collapse:collapse;font-size:13px">'
                 '<thead><tr style="text-align:left"><th style="padding:2px 10px">'
                 'axis</th><th>vEcoli</th><th>v2ecoli</th><th>verdict</th>'
                 '<th>Δ</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
        html = traj + table

    # Prepended on BOTH branches: a basis refusal is the reason the axes could not
    # grade, and that stays true when there is also no data — otherwise a reader
    # sees "no flux yet" and never learns the axes would not have graded anyway.
    _refused = [a for a in axes if a["detail"].get("unresolved_reason")]
    if _refused:
        why = _refused[0]["detail"].get("unresolved_reason", "")
        html = (
            '<div style="padding:6px 10px;margin:4px 0;border-left:3px solid '
            '#b45309;background:#fffbeb;font-size:13px">'
            '<b>Bioproduction axes not computed.</b> '
            + _html.escape(why) +
            ' &mdash; any numbers below are traces, not graded results.'
            '</div>' + html)

    return {"card_html": html, "verdict": worst(a["verdict"] for a in axes), "axes": axes}


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) else "—"


REPORT_CARD_STEPS["violacein_report_card"] = update_violacein_report_card
