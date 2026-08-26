"""Which QUANTITY a comparison run's exchange-flux leaves carry.

Every card that grades an exchange-flux leaf needs this and needs it to give the
same answer, so it lives here rather than inside any one card. The leaves are
emitted under ``listeners.exchange_flux.<leaf>`` on both arms, and the same leaf
name carries a DIFFERENT MEASUREMENT depending on the basis the run declared:

``counts``
    a lineage-cumulative molecule total. ``environment.exchange`` accumulates
    (``state + update``) while metabolism writes a per-step delta, and it does
    not reset at division — so the leaf's time-average grows with how long the
    lineage ran and is not a rate.

``gdcw``
    a per-tick rate in mmol/gDCW/h.

⚠ THE CONSEQUENCE FOR A CARD, and it is why this is a refusal and not a default:
a relative delta between two arms is robust to a shared UNIT, but NOT to a shared
WRONG QUANTITY. Two arms both on ``counts`` produce a ratio of two cumulative
totals that lands inside a tight tolerance and grades green while measuring
something that is not a flux. ⇒ A card whose axes are only meaningful on one
basis must ASK, and refuse when the answer is not the basis it needs.

⚠ AND ANY CARD THAT NORMALISES BY MASS MUST BRANCH ON THIS. On ``gdcw`` the leaf
is ALREADY per-gDCW, so dividing by dry mass again divides twice — silently, and
inside tolerance, because both arms are divided twice.

READ OFF THE RUN, NOT THE STUDY CONFIG. See ``basis_from_runs``.
"""
import json
import os

#: Written beside each arm's stores by
#: ``run_comparison_ensemble._write_exchange_flux_sidecar``. The filename, the
#: ``basis`` key and the two arm prefixes are a CONTRACT between that writer and
#: this reader — change one and every card silently refuses every axis, on runs
#: that are otherwise healthy.
SIDECAR_SUFFIX = "_exchange_flux.json"

#: (state key holding the arm's output dir, sidecar filename prefix)
_ARMS = (("v2_dir", "v2ecoli"), ("ve_dir", "vecoli"))


def basis_from_runs(state: dict) -> tuple[str | None, str]:
    """The basis BOTH arms actually ran with, read off the runs themselves.

    ⚠ Deliberately NOT resolved from the study config. The card and the engines
    reading the same YAML by different rules is what previously graded a
    lineage-cumulative total as a rate: both arms were equally wrong, so the
    relative delta looked fine and the axis went green. The run is the ground
    truth for what the run computed.

    Returns ``(basis, reason)``. ``basis`` is None whenever the two arms cannot be
    shown to agree — a missing sidecar (a run that predates this, or an arm that
    never emitted), or two arms that genuinely ran different quantities. All of
    those are refusals, because a number whose quantity is unknown is worse than
    no number. ``reason`` is empty exactly when ``basis`` is not None, and is
    written to be shown to a reader verbatim.
    """
    found, shape = {}, {}
    for key, prefix in _ARMS:
        d = state.get(key)
        path = os.path.join(d or "", f"{prefix}{SIDECAR_SUFFIX}")
        if not d or not os.path.isfile(path):
            return None, f"no exchange-flux sidecar for the {prefix} arm"
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh) or {}
        except Exception:  # noqa: BLE001 — an unreadable sidecar is a refusal
            return None, f"unreadable exchange-flux sidecar for the {prefix} arm"
        basis = doc.get("basis")
        if not basis:
            # ⚠ Distinct from "counts": the quantity is UNRECORDED. Two such
            # sidecars would compare equal and slip through the agreement check
            # below, and the caller would then describe them with counts
            # semantics it has no evidence for.
            return None, (f"the {prefix} arm's sidecar records no basis, so the "
                          "quantity its leaves carry is unknown")
        found[prefix] = str(basis)
        shape[prefix] = (doc.get("seeds"), doc.get("generations"))
    if found["v2ecoli"] != found["vecoli"]:
        return None, (f"the two arms ran different bases "
                      f"(candidate={found['v2ecoli']!r}, reference={found['vecoli']!r})")
    # ⚠ Agreeing on the basis does NOT establish the two sidecars describe the
    # same invocation: both arms write into one out_root and nothing cleans it,
    # so re-running one arm leaves the other's sidecar and stores in place. Two
    # arms of one study share seeds and generations by construction, so a
    # disagreement here means one side is stale.
    if shape["v2ecoli"] != shape["vecoli"] and None not in shape["v2ecoli"] + shape["vecoli"]:
        return None, (f"the two arms' runs do not correspond — candidate ran "
                      f"seeds={shape['v2ecoli'][0]} generations={shape['v2ecoli'][1]}, "
                      f"reference ran seeds={shape['vecoli'][0]} "
                      f"generations={shape['vecoli'][1]}; one of them is stale")
    return found["v2ecoli"], ""
